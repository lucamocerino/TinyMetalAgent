use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
    time::Instant,
};

use half::f16;
use serde::{Deserialize, Serialize};
use tinyagent_core::{Result, TinyAgentError};
use tokenizers::Tokenizer;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
};

const QWEN_IM_END: u32 = 151_645;
const QWEN_END_OF_TEXT: u32 = 151_643;

pub fn format_qwen_chat_prompt(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

#[derive(Debug, Clone)]
pub struct QwenMlxRunConfig {
    pub hf_dir: PathBuf,
    pub prompt: String,
    pub max_tokens: usize,
    pub max_prompt_tokens: usize,
    pub projection_backend: QwenProjectionBackend,
}

#[derive(Debug, Clone)]
pub struct QwenGgufRunConfig {
    pub gguf_path: PathBuf,
    pub tokenizer_dir: PathBuf,
    pub prompt: String,
    pub max_tokens: usize,
    pub max_prompt_tokens: usize,
    pub projection_backend: QwenProjectionBackend,
}

#[derive(Debug, Clone, Serialize)]
pub struct QwenMlxRunResult {
    pub model_dir: PathBuf,
    pub model_format: String,
    pub prompt: String,
    pub prompt_tokens_total: usize,
    pub prompt_tokens_used: usize,
    pub projection_backend: QwenProjectionBackend,
    pub generated_token_ids: Vec<u32>,
    pub generated_text: String,
    pub load_ms: f64,
    pub eval_ms: f64,
    pub total_ms: f64,
    pub token_eval_ms: Vec<f64>,
    pub prompt_eval_ms: f64,
    pub ttft_ms: f64,
    pub decode_eval_ms: f64,
    pub decode_eval_tokens: usize,
    pub decode_tokens_per_second: f64,
    pub end_to_end_generated_tokens_per_second: f64,
    pub avg_token_eval_ms: f64,
    pub min_token_eval_ms: f64,
    pub max_token_eval_ms: f64,
    pub eval_tokens_per_second: f64,
    pub q4_projection_groups_per_eval_token: usize,
    pub q4_projection_groups_total: usize,
    pub note: String,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum QwenProjectionBackend {
    Cpu,
    Metal,
}

#[derive(Debug, Clone, Serialize)]
pub struct Q4AffineMatVecProbeResult {
    pub out_dim: usize,
    pub in_dim: usize,
    pub metal_output: Vec<f32>,
    pub cpu_output: Vec<f32>,
    pub max_abs_error: f32,
}

pub fn run_q4_affine_matvec_probe() -> Result<Q4AffineMatVecProbeResult> {
    let mut tensor = QuantTensor {
        name: "probe.q4".to_string(),
        encoding: QuantEncoding::AffineQ4,
        out_dim: 3,
        in_dim: 8,
        packed_per_row: 1,
        groups_per_row: 1,
        group_size: 64,
        weight: vec![
            pack_q4(&[0, 1, 2, 3, 4, 5, 6, 7]),
            pack_q4(&[8, 7, 6, 5, 4, 3, 2, 1]),
            pack_q4(&[15, 0, 15, 0, 10, 5, 10, 5]),
        ],
        q8_values: Vec::new(),
        scales: vec![0.1, -0.05, 0.025],
        biases: vec![-0.3, 0.2, -0.1],
        metal: None,
    };
    let input = vec![1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 2.0, -0.75];
    let linear_bias = vec![0.5, -0.25, 0.125];
    let cpu_output = tensor.matvec(&input, Some(&linear_bias))?;
    let runtime = Q4MetalRuntime::new()?;
    tensor.upload_metal(&runtime)?;
    let metal_output = tensor.matvec_metal_with_runtime(&input, Some(&linear_bias), &runtime)?;
    let max_abs_error = cpu_output
        .iter()
        .zip(metal_output.iter())
        .map(|(cpu, metal)| (cpu - metal).abs())
        .fold(0.0_f32, f32::max);

    Ok(Q4AffineMatVecProbeResult {
        out_dim: tensor.out_dim,
        in_dim: tensor.in_dim,
        metal_output,
        cpu_output,
        max_abs_error,
    })
}

pub fn run_qwen_mlx_end2end(config: QwenMlxRunConfig) -> Result<QwenMlxRunResult> {
    if config.max_tokens == 0 {
        return Err(config_error("max_tokens must be greater than zero"));
    }
    if config.max_prompt_tokens == 0 {
        return Err(config_error("max_prompt_tokens must be greater than zero"));
    }

    let total_started = Instant::now();
    let load_started = Instant::now();
    let mut model = QwenRuntimeModel::load_mlx(&config.hf_dir, config.projection_backend)?;
    let tokenizer = Tokenizer::from_file(config.hf_dir.join("tokenizer.json"))
        .map_err(|error| config_error(format!("failed to load tokenizer.json: {error}")))?;
    let load_ms = elapsed_ms(load_started);
    run_qwen_loaded(
        &mut model,
        tokenizer,
        config.hf_dir,
        "mlx-affine-q4".to_string(),
        config.prompt,
        config.max_tokens,
        config.max_prompt_tokens,
        load_ms,
        total_started,
    )
}

pub fn run_qwen_gguf_end2end(config: QwenGgufRunConfig) -> Result<QwenMlxRunResult> {
    if config.max_tokens == 0 {
        return Err(config_error("max_tokens must be greater than zero"));
    }
    if config.max_prompt_tokens == 0 {
        return Err(config_error("max_prompt_tokens must be greater than zero"));
    }

    let total_started = Instant::now();
    let load_started = Instant::now();
    let mut model = QwenRuntimeModel::load_gguf(&config.gguf_path, config.projection_backend)?;
    let tokenizer = Tokenizer::from_file(config.tokenizer_dir.join("tokenizer.json"))
        .map_err(|error| config_error(format!("failed to load tokenizer.json: {error}")))?;
    let load_ms = elapsed_ms(load_started);
    run_qwen_loaded(
        &mut model,
        tokenizer,
        config.gguf_path,
        "gguf-q4_0".to_string(),
        config.prompt,
        config.max_tokens,
        config.max_prompt_tokens,
        load_ms,
        total_started,
    )
}

fn run_qwen_loaded(
    model: &mut QwenRuntimeModel,
    tokenizer: Tokenizer,
    model_dir: PathBuf,
    model_format: String,
    prompt: String,
    max_tokens: usize,
    max_prompt_tokens: usize,
    load_ms: f64,
    total_started: Instant,
) -> Result<QwenMlxRunResult> {
    let formatted_prompt = format_qwen_chat_prompt(&prompt);
    let encoding = tokenizer
        .encode(formatted_prompt, true)
        .map_err(|error| config_error(format!("failed to encode prompt: {error}")))?;
    let prompt_tokens_total = encoding.len();
    let mut input_ids = encoding.get_ids().to_vec();
    if input_ids.len() > max_prompt_tokens {
        input_ids = input_ids[input_ids.len() - max_prompt_tokens..].to_vec();
    }
    if input_ids.is_empty() {
        return Err(config_error("prompt encoded to zero tokens"));
    }

    model.allocate_kv_cache(input_ids.len() + max_tokens + 1);

    let eval_started = Instant::now();
    let mut token_eval_ms = Vec::with_capacity(input_ids.len() + max_tokens);
    let prompt_started = Instant::now();
    let hidden = model.prefill_hidden(&input_ids, 0)?;
    let mut next_id = model.project_next_token(&hidden)?;
    let prompt_eval_ms = elapsed_ms(prompt_started);
    let prompt_token_ms = prompt_eval_ms / input_ids.len() as f64;
    token_eval_ms.extend(std::iter::repeat(prompt_token_ms).take(input_ids.len()));
    let mut position = input_ids.len();

    let mut generated_token_ids = Vec::with_capacity(max_tokens);
    for generated_index in 0..max_tokens {
        if next_id == QWEN_IM_END || next_id == QWEN_END_OF_TEXT {
            break;
        }
        generated_token_ids.push(next_id);
        if generated_index + 1 == max_tokens {
            break;
        }

        let started = Instant::now();
        next_id = model.decode_next_token(next_id, position)?;
        token_eval_ms.push(elapsed_ms(started));
        position += 1;
    }
    let eval_ms = elapsed_ms(eval_started);

    let generated_text = tokenizer
        .decode(&generated_token_ids, true)
        .unwrap_or_else(|_| {
            generated_token_ids
                .iter()
                .map(|id| format!("<{id}>"))
                .collect::<Vec<_>>()
                .join("")
        });
    let generated_token_count = generated_token_ids.len();

    let eval_token_count = token_eval_ms.len();
    let prompt_eval_ms = token_eval_ms.iter().take(input_ids.len()).sum::<f64>();
    let decode_eval_ms = token_eval_ms.iter().skip(input_ids.len()).sum::<f64>();
    let decode_eval_tokens = eval_token_count.saturating_sub(input_ids.len());
    let avg_token_eval_ms = if eval_token_count == 0 {
        0.0
    } else {
        token_eval_ms.iter().sum::<f64>() / eval_token_count as f64
    };
    let min_token_eval_ms = token_eval_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let max_token_eval_ms = token_eval_ms.iter().copied().fold(0.0_f64, f64::max);
    let q4_projection_groups_per_eval_token = match model.projection_backend {
        QwenProjectionBackend::Cpu => model.config.num_hidden_layers * 7 + 1,
        QwenProjectionBackend::Metal => model.config.num_hidden_layers * 4 + 1,
    };
    let layer_projection_groups_per_token = match model.projection_backend {
        QwenProjectionBackend::Cpu => model.config.num_hidden_layers * 7,
        QwenProjectionBackend::Metal => model.config.num_hidden_layers * 4,
    };
    let logits_projection_count = 1 + decode_eval_tokens;
    let note = match (model_format.as_str(), model.projection_backend) {
        ("mlx-affine-q4", QwenProjectionBackend::Cpu) => "CPU reference end-to-end over real MLX 4-bit Qwen weights; this validates model graph and q4 affine loading.".to_string(),
        ("mlx-affine-q4", QwenProjectionBackend::Metal) => "Hybrid TinyEngine end-to-end over real MLX 4-bit Qwen weights: q4 affine projections run on reusable Metal kernels with GPU-resident q4 tensors; prompt prefill is batched and decode layers use Metal scalar kernels when supported.".to_string(),
        ("gguf-q4_0", QwenProjectionBackend::Cpu) => "CPU reference end-to-end over GGUF Q4_0 Qwen weights.".to_string(),
        ("gguf-q4_0", QwenProjectionBackend::Metal) => "Hybrid TinyEngine end-to-end over the same GGUF Q4_0 file used by llama.cpp: prompt prefill uses batched Q4_0 Metal matmul; decode keeps hidden/KV/scalar ops/attention on Metal and uses Q8_0 lm_head plus argmax on Metal.".to_string(),
        _ => "TinyEngine Qwen end-to-end run.".to_string(),
    };

    Ok(QwenMlxRunResult {
        model_dir,
        model_format,
        prompt,
        prompt_tokens_total,
        prompt_tokens_used: input_ids.len(),
        projection_backend: model.projection_backend,
        generated_token_ids,
        generated_text,
        load_ms,
        eval_ms,
        total_ms: elapsed_ms(total_started),
        token_eval_ms,
        prompt_eval_ms,
        ttft_ms: prompt_eval_ms,
        decode_eval_ms,
        decode_eval_tokens,
        decode_tokens_per_second: if decode_eval_ms > 0.0 {
            decode_eval_tokens as f64 / (decode_eval_ms / 1000.0)
        } else {
            0.0
        },
        end_to_end_generated_tokens_per_second: if eval_ms > 0.0 {
            generated_token_count as f64 / (eval_ms / 1000.0)
        } else {
            0.0
        },
        avg_token_eval_ms,
        min_token_eval_ms: if min_token_eval_ms.is_finite() {
            min_token_eval_ms
        } else {
            0.0
        },
        max_token_eval_ms,
        eval_tokens_per_second: if eval_ms > 0.0 {
            eval_token_count as f64 / (eval_ms / 1000.0)
        } else {
            0.0
        },
        q4_projection_groups_per_eval_token,
        q4_projection_groups_total: layer_projection_groups_per_token * eval_token_count
            + logits_projection_count,
        note,
    })
}

struct QwenRuntimeModel {
    config: QwenRuntimeConfig,
    projection_backend: QwenProjectionBackend,
    metal_runtime: Option<Q4MetalRuntime>,
    embed_tokens: QuantTensor,
    lm_head: Option<QuantTensor>,
    layers: Vec<QwenLayer>,
    norm_weight: Vec<f32>,
    kv_cache: Vec<KvCache>,
    metal_decode: Option<MetalDecodeState>,
}

impl QwenRuntimeModel {
    fn load_mlx(hf_dir: &Path, projection_backend: QwenProjectionBackend) -> Result<Self> {
        let config = QwenRuntimeConfig::read(hf_dir)?;
        if !config.tie_word_embeddings {
            return Err(config_error(
                "only tied Qwen embeddings are supported by the minimal runtime",
            ));
        }

        let tensors = SafeTensorFile::read(hf_dir.join("model.safetensors"))?;
        let embed_tokens = tensors.load_quant("model.embed_tokens")?;
        ensure_quant_shape(
            "model.embed_tokens",
            &embed_tokens,
            config.vocab_size,
            config.hidden_size,
        )?;
        let norm_weight = tensors.load_bf16("model.norm.weight")?;
        ensure_len("model.norm.weight", &norm_weight, config.hidden_size)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{index}");
            let layer = QwenLayer {
                input_layernorm_weight: tensors
                    .load_bf16(format!("{prefix}.input_layernorm.weight"))?,
                post_attention_layernorm_weight: tensors
                    .load_bf16(format!("{prefix}.post_attention_layernorm.weight"))?,
                q_proj: tensors.load_quant(format!("{prefix}.self_attn.q_proj"))?,
                k_proj: tensors.load_quant(format!("{prefix}.self_attn.k_proj"))?,
                v_proj: tensors.load_quant(format!("{prefix}.self_attn.v_proj"))?,
                o_proj: tensors.load_quant(format!("{prefix}.self_attn.o_proj"))?,
                q_bias: tensors.load_bf16(format!("{prefix}.self_attn.q_proj.bias"))?,
                k_bias: tensors.load_bf16(format!("{prefix}.self_attn.k_proj.bias"))?,
                v_bias: tensors.load_bf16(format!("{prefix}.self_attn.v_proj.bias"))?,
                gate_proj: tensors.load_quant(format!("{prefix}.mlp.gate_proj"))?,
                up_proj: tensors.load_quant(format!("{prefix}.mlp.up_proj"))?,
                down_proj: tensors.load_quant(format!("{prefix}.mlp.down_proj"))?,
            };
            layer.validate_shapes(&config, index)?;
            layers.push(layer);
        }

        let mut model = Self {
            config,
            projection_backend,
            metal_runtime: match projection_backend {
                QwenProjectionBackend::Cpu => None,
                QwenProjectionBackend::Metal => Some(Q4MetalRuntime::new()?),
            },
            embed_tokens,
            lm_head: None,
            layers,
            norm_weight,
            kv_cache: Vec::new(),
            metal_decode: None,
        };
        model.upload_quant_tensors_to_metal()?;
        Ok(model)
    }

    fn load_gguf(gguf_path: &Path, projection_backend: QwenProjectionBackend) -> Result<Self> {
        let tensors = GgufFile::read(gguf_path)?;
        let config = QwenRuntimeConfig::read_gguf(&tensors)?;

        let embed_tokens = tensors.load_q4_0("token_embd.weight")?;
        ensure_quant_shape(
            "token_embd.weight",
            &embed_tokens,
            config.vocab_size,
            config.hidden_size,
        )?;
        let lm_head = if tensors.has_tensor("output.weight") {
            Some(tensors.load_q8_0("output.weight")?)
        } else {
            None
        };
        if let Some(lm_head) = &lm_head {
            ensure_quant_shape(
                "output.weight",
                lm_head,
                config.vocab_size,
                config.hidden_size,
            )?;
        }

        let norm_weight = tensors.load_f32("output_norm.weight")?;
        ensure_len("output_norm.weight", &norm_weight, config.hidden_size)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let prefix = format!("blk.{index}");
            let layer = QwenLayer {
                input_layernorm_weight: tensors.load_f32(format!("{prefix}.attn_norm.weight"))?,
                post_attention_layernorm_weight: tensors
                    .load_f32(format!("{prefix}.ffn_norm.weight"))?,
                q_proj: tensors.load_q4_0(format!("{prefix}.attn_q.weight"))?,
                k_proj: tensors.load_q4_0(format!("{prefix}.attn_k.weight"))?,
                v_proj: tensors.load_q4_0(format!("{prefix}.attn_v.weight"))?,
                o_proj: tensors.load_q4_0(format!("{prefix}.attn_output.weight"))?,
                q_bias: tensors.load_f32(format!("{prefix}.attn_q.bias"))?,
                k_bias: tensors.load_f32(format!("{prefix}.attn_k.bias"))?,
                v_bias: tensors.load_f32(format!("{prefix}.attn_v.bias"))?,
                gate_proj: tensors.load_q4_0(format!("{prefix}.ffn_gate.weight"))?,
                up_proj: tensors.load_q4_0(format!("{prefix}.ffn_up.weight"))?,
                down_proj: tensors.load_q4_0(format!("{prefix}.ffn_down.weight"))?,
            };
            layer.validate_shapes(&config, index)?;
            layers.push(layer);
        }

        let mut model = Self {
            config,
            projection_backend,
            metal_runtime: match projection_backend {
                QwenProjectionBackend::Cpu => None,
                QwenProjectionBackend::Metal => Some(Q4MetalRuntime::new()?),
            },
            embed_tokens,
            lm_head,
            layers,
            norm_weight,
            kv_cache: Vec::new(),
            metal_decode: None,
        };
        model.upload_quant_tensors_to_metal()?;
        Ok(model)
    }

    fn upload_quant_tensors_to_metal(&mut self) -> Result<()> {
        let Some(runtime) = self.metal_runtime.as_ref() else {
            return Ok(());
        };
        self.embed_tokens.upload_metal(runtime)?;
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.upload_metal(runtime)?;
        }
        for layer in &mut self.layers {
            layer.q_proj.upload_metal(runtime)?;
            layer.k_proj.upload_metal(runtime)?;
            layer.v_proj.upload_metal(runtime)?;
            layer.o_proj.upload_metal(runtime)?;
            layer.gate_proj.upload_metal(runtime)?;
            layer.up_proj.upload_metal(runtime)?;
            layer.down_proj.upload_metal(runtime)?;
        }
        Ok(())
    }

    fn allocate_kv_cache(&mut self, max_seq: usize) {
        self.kv_cache = (0..self.layers.len())
            .map(|_| {
                KvCache::new(
                    max_seq,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
            })
            .collect();
        self.metal_decode = match self.metal_runtime.as_ref() {
            Some(runtime) => Some(MetalDecodeState::new(
                &runtime.device,
                max_seq,
                &self.config,
                &self.layers,
                &self.norm_weight,
            )),
            None => None,
        };
    }

    fn forward_hidden(&mut self, token_id: u32, position: usize) -> Result<Vec<f32>> {
        let mut hidden = self.embed_tokens.dequantize_row(token_id as usize)?;
        ensure_len("embedding", &hidden, self.config.hidden_size)?;

        for layer_index in 0..self.layers.len() {
            hidden = self.forward_layer(layer_index, &hidden, position)?;
        }
        Ok(hidden)
    }

    fn forward_hidden_decode(&mut self, token_id: u32, position: usize) -> Result<Vec<f32>> {
        if matches!(self.projection_backend, QwenProjectionBackend::Metal)
            && self.metal_decode.is_some()
        {
            self.forward_hidden_decode_metal(token_id, position)
        } else {
            self.forward_hidden(token_id, position)
        }
    }

    fn forward_hidden_decode_metal(&mut self, token_id: u32, position: usize) -> Result<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        let num_attention_heads = self.config.num_attention_heads;
        let num_key_value_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let rms_norm_eps = self.config.rms_norm_eps;
        let rope_theta = self.config.rope_theta;
        let attention_scale = (head_dim as f32).sqrt().recip();

        let embedding = self.embed_tokens.dequantize_row(token_id as usize)?;
        ensure_len("embedding", &embedding, hidden_size)?;

        let runtime = self
            .metal_runtime
            .as_ref()
            .ok_or_else(|| config_error("Metal decode path missing runtime"))?;
        let decode = self
            .metal_decode
            .as_mut()
            .ok_or_else(|| config_error("Metal decode path missing buffers"))?;
        write_f32_buffer_direct(&decode.hidden, &embedding);

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let dispatch_1d = |len: usize| MTLSize {
            width: len.div_ceil(256) as u64,
            height: 1,
            depth: 1,
        };
        let threads_256 = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let threads_128 = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };

        let set_u32 = |index: u64, value: &u32| {
            encoder.set_bytes(
                index,
                std::mem::size_of::<u32>() as u64,
                (value as *const u32).cast(),
            );
        };
        let set_f32 = |index: u64, value: &f32| {
            encoder.set_bytes(
                index,
                std::mem::size_of::<f32>() as u64,
                (value as *const f32).cast(),
            );
        };
        let set_u32_slice = |index: u64, values: &[u32]| {
            encoder.set_bytes(
                index,
                std::mem::size_of_val(values) as u64,
                values.as_ptr().cast(),
            );
        };

        let encode_q4 = |tensor: &QuantTensor,
                         input: &Buffer,
                         linear_bias: &Buffer,
                         output: &Buffer|
         -> Result<()> {
            let metal = tensor.metal.as_ref().ok_or_else(|| {
                config_error(format!("{} has not been uploaded to Metal", tensor.name))
            })?;
            let q4_weight = metal.q4_weight.as_ref().ok_or_else(|| {
                config_error(format!("{} is missing q4 Metal weight buffer", tensor.name))
            })?;
            let biases = metal.biases.as_ref().ok_or_else(|| {
                config_error(format!("{} is missing q4 Metal bias buffer", tensor.name))
            })?;
            encoder.set_compute_pipeline_state(&runtime.q4_pipeline);
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(q4_weight), 0);
            encoder.set_buffer(2, Some(&metal.scales), 0);
            encoder.set_buffer(3, Some(biases), 0);
            encoder.set_buffer(4, Some(linear_bias), 0);
            encoder.set_buffer(5, Some(output), 0);
            encoder.set_buffer(6, Some(&metal.dims), 0);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: tensor.out_dim.div_ceil(8) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
            Ok(())
        };

        let encode_rmsnorm = |input: &Buffer, weight: &Buffer, output: &Buffer| {
            let len = hidden_size as u32;
            encoder.set_compute_pipeline_state(&runtime.rmsnorm_pipeline);
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(weight), 0);
            encoder.set_buffer(2, Some(output), 0);
            set_u32(3, &len);
            set_f32(4, &rms_norm_eps);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                threads_256,
            );
        };

        let encode_rope = |values: &Buffer, heads: usize| {
            let params = [heads as u32, head_dim as u32, position as u32];
            let len = heads * head_dim / 2;
            encoder.set_compute_pipeline_state(&runtime.rope_pipeline);
            encoder.set_buffer(0, Some(values), 0);
            set_u32_slice(1, &params);
            set_f32(2, &rope_theta);
            encoder.dispatch_thread_groups(dispatch_1d(len), threads_256);
        };

        let encode_kv_write = |layer_state: &MetalLayerState| {
            let params = [position as u32, kv_dim as u32];
            encoder.set_compute_pipeline_state(&runtime.kv_write_pipeline);
            encoder.set_buffer(0, Some(&decode.k), 0);
            encoder.set_buffer(1, Some(&decode.v), 0);
            encoder.set_buffer(2, Some(&layer_state.kv.keys), 0);
            encoder.set_buffer(3, Some(&layer_state.kv.values), 0);
            set_u32_slice(4, &params);
            encoder.dispatch_thread_groups(dispatch_1d(kv_dim), threads_256);
        };

        let encode_attention = |layer_state: &MetalLayerState| {
            let params = [
                position as u32,
                num_attention_heads as u32,
                num_key_value_heads as u32,
                head_dim as u32,
            ];
            encoder.set_compute_pipeline_state(&runtime.attention_pipeline);
            encoder.set_buffer(0, Some(&decode.q), 0);
            encoder.set_buffer(1, Some(&layer_state.kv.keys), 0);
            encoder.set_buffer(2, Some(&layer_state.kv.values), 0);
            encoder.set_buffer(3, Some(&decode.attn), 0);
            set_u32_slice(4, &params);
            set_f32(5, &attention_scale);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: num_attention_heads as u64,
                    height: 1,
                    depth: 1,
                },
                threads_128,
            );
        };

        let encode_add = |lhs: &Buffer, rhs: &Buffer, output: &Buffer, len: usize| {
            let len_u32 = len as u32;
            encoder.set_compute_pipeline_state(&runtime.add_pipeline);
            encoder.set_buffer(0, Some(lhs), 0);
            encoder.set_buffer(1, Some(rhs), 0);
            encoder.set_buffer(2, Some(output), 0);
            set_u32(3, &len_u32);
            encoder.dispatch_thread_groups(dispatch_1d(len), threads_256);
        };

        let encode_swiglu = || {
            let len = intermediate_size as u32;
            encoder.set_compute_pipeline_state(&runtime.swiglu_pipeline);
            encoder.set_buffer(0, Some(&decode.gate), 0);
            encoder.set_buffer(1, Some(&decode.up), 0);
            encoder.set_buffer(2, Some(&decode.swiglu), 0);
            set_u32(3, &len);
            encoder.dispatch_thread_groups(dispatch_1d(intermediate_size), threads_256);
        };

        for (layer, layer_state) in self.layers.iter().zip(&decode.layers) {
            let o_zero_bias = &layer
                .o_proj
                .metal
                .as_ref()
                .ok_or_else(|| config_error("o_proj missing Metal buffers"))?
                .zero_linear_bias;
            let gate_zero_bias = &layer
                .gate_proj
                .metal
                .as_ref()
                .ok_or_else(|| config_error("gate_proj missing Metal buffers"))?
                .zero_linear_bias;
            let up_zero_bias = &layer
                .up_proj
                .metal
                .as_ref()
                .ok_or_else(|| config_error("up_proj missing Metal buffers"))?
                .zero_linear_bias;
            let down_zero_bias = &layer
                .down_proj
                .metal
                .as_ref()
                .ok_or_else(|| config_error("down_proj missing Metal buffers"))?
                .zero_linear_bias;
            encode_rmsnorm(
                &decode.hidden,
                &layer_state.input_norm_weight,
                &decode.attn_input,
            );
            encode_q4(
                &layer.q_proj,
                &decode.attn_input,
                &layer_state.q_bias,
                &decode.q,
            )?;
            encode_q4(
                &layer.k_proj,
                &decode.attn_input,
                &layer_state.k_bias,
                &decode.k,
            )?;
            encode_q4(
                &layer.v_proj,
                &decode.attn_input,
                &layer_state.v_bias,
                &decode.v,
            )?;
            encode_rope(&decode.q, num_attention_heads);
            encode_rope(&decode.k, num_key_value_heads);
            encode_kv_write(layer_state);
            encode_attention(layer_state);
            encode_q4(&layer.o_proj, &decode.attn, o_zero_bias, &decode.attn_out)?;
            encode_add(
                &decode.hidden,
                &decode.attn_out,
                &decode.residual,
                hidden_size,
            );
            encode_rmsnorm(
                &decode.residual,
                &layer_state.post_norm_weight,
                &decode.mlp_input,
            );
            encode_q4(
                &layer.gate_proj,
                &decode.mlp_input,
                gate_zero_bias,
                &decode.gate,
            )?;
            encode_q4(&layer.up_proj, &decode.mlp_input, up_zero_bias, &decode.up)?;
            encode_swiglu();
            encode_q4(
                &layer.down_proj,
                &decode.swiglu,
                down_zero_bias,
                &decode.mlp_out,
            )?;
            encode_add(
                &decode.residual,
                &decode.mlp_out,
                &decode.hidden,
                hidden_size,
            );
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(read_f32_buffer(&decode.hidden, hidden_size))
    }

    fn prefill_hidden(&mut self, token_ids: &[u32], start_position: usize) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(config_error("prefill requires at least one token"));
        }

        let batch = token_ids.len();
        let hidden_size = self.config.hidden_size;
        let mut hidden = Vec::with_capacity(batch * hidden_size);
        for token_id in token_ids {
            let embedding = self.embed_tokens.dequantize_row(*token_id as usize)?;
            ensure_len("embedding", &embedding, hidden_size)?;
            hidden.extend_from_slice(&embedding);
        }

        for layer_index in 0..self.layers.len() {
            hidden = self.forward_layer_batched(layer_index, &hidden, batch, start_position)?;
        }

        let last_offset = (batch - 1) * hidden_size;
        Ok(hidden[last_offset..last_offset + hidden_size].to_vec())
    }

    fn project_logits(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let normalized = rmsnorm(&hidden, &self.norm_weight, self.config.rms_norm_eps);
        let lm_head = self.lm_head.as_ref().unwrap_or(&self.embed_tokens);
        project_quant(
            self.projection_backend,
            self.metal_runtime.as_ref(),
            lm_head,
            &normalized,
            None,
        )
    }

    fn project_next_token(&mut self, hidden: &[f32]) -> Result<u32> {
        if matches!(self.projection_backend, QwenProjectionBackend::Metal)
            && self.metal_decode.is_some()
            && self
                .lm_head
                .as_ref()
                .is_some_and(|tensor| tensor.encoding == QuantEncoding::GgufQ8_0)
        {
            self.project_next_token_metal(hidden)
        } else {
            Ok(argmax(&self.project_logits(hidden)?) as u32)
        }
    }

    fn decode_next_token(&mut self, token_id: u32, position: usize) -> Result<u32> {
        let hidden = self.forward_hidden_decode(token_id, position)?;
        self.project_next_token(&hidden)
    }

    fn project_next_token_metal(&mut self, hidden: &[f32]) -> Result<u32> {
        ensure_len("decode hidden", hidden, self.config.hidden_size)?;
        let runtime = self
            .metal_runtime
            .as_ref()
            .ok_or_else(|| config_error("Metal next-token path missing runtime"))?;
        let decode = self
            .metal_decode
            .as_mut()
            .ok_or_else(|| config_error("Metal next-token path missing buffers"))?;
        let lm_head = self
            .lm_head
            .as_ref()
            .ok_or_else(|| config_error("Metal next-token path requires GGUF output.weight"))?;
        let lm_head_metal = lm_head
            .metal
            .as_ref()
            .ok_or_else(|| config_error("output.weight missing Metal buffers"))?;
        let q8_values = lm_head_metal
            .q8_values
            .as_ref()
            .ok_or_else(|| config_error("output.weight missing q8 Metal values"))?;

        write_f32_buffer_direct(&decode.hidden, hidden);

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let hidden_len = self.config.hidden_size as u32;
        let vocab_len = self.config.vocab_size as u32;

        encoder.set_compute_pipeline_state(&runtime.rmsnorm_pipeline);
        encoder.set_buffer(0, Some(&decode.hidden), 0);
        encoder.set_buffer(1, Some(&decode.output_norm_weight), 0);
        encoder.set_buffer(2, Some(&decode.normed_hidden), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            (&hidden_len as *const u32).cast(),
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<f32>() as u64,
            (&self.config.rms_norm_eps as *const f32).cast(),
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&runtime.q8_pipeline);
        encoder.set_buffer(0, Some(&decode.normed_hidden), 0);
        encoder.set_buffer(1, Some(q8_values), 0);
        encoder.set_buffer(2, Some(&lm_head_metal.scales), 0);
        encoder.set_buffer(3, Some(&lm_head_metal.zero_linear_bias), 0);
        encoder.set_buffer(4, Some(&decode.logits), 0);
        encoder.set_buffer(5, Some(&lm_head_metal.dims), 0);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: lm_head.out_dim as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );

        encoder.set_compute_pipeline_state(&runtime.argmax_pipeline);
        encoder.set_buffer(0, Some(&decode.logits), 0);
        encoder.set_buffer(1, Some(&decode.argmax_index), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            (&vocab_len as *const u32).cast(),
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = decode.argmax_index.contents().cast::<u32>();
        Ok(unsafe { *ptr })
    }

    fn forward_layer(
        &mut self,
        layer_index: usize,
        hidden: &[f32],
        position: usize,
    ) -> Result<Vec<f32>> {
        let layer = &self.layers[layer_index];
        let attn_input = rmsnorm(
            hidden,
            &layer.input_layernorm_weight,
            self.config.rms_norm_eps,
        );

        let runtime = self.metal_runtime.as_ref();
        let backend = self.projection_backend;
        let (mut q, mut k, v) = project_qkv(backend, runtime, layer, &attn_input)?;

        apply_rope(
            &mut q,
            self.config.num_attention_heads,
            self.config.head_dim,
            position,
            self.config.rope_theta,
        );
        apply_rope(
            &mut k,
            self.config.num_key_value_heads,
            self.config.head_dim,
            position,
            self.config.rope_theta,
        );

        self.kv_cache[layer_index].write(position, &k, &v);
        let attn = attention_decode(&q, &self.kv_cache[layer_index], position, &self.config);
        let attn_out = project_quant(backend, runtime, &layer.o_proj, &attn, None)?;

        let mut residual = add(hidden, &attn_out);
        let mlp_input = rmsnorm(
            &residual,
            &layer.post_attention_layernorm_weight,
            self.config.rms_norm_eps,
        );
        let (gate, up) = project_gate_up(backend, runtime, layer, &mlp_input)?;
        let swiglu = swiglu(&gate, &up);
        let mlp_out = project_quant(backend, runtime, &layer.down_proj, &swiglu, None)?;
        add_in_place(&mut residual, &mlp_out);
        Ok(residual)
    }

    fn forward_layer_batched(
        &mut self,
        layer_index: usize,
        hidden: &[f32],
        batch: usize,
        start_position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        if batch == 0 || hidden.len() != batch * hidden_size {
            return Err(config_error(format!(
                "batched layer input length {} does not match batch {batch} * hidden_size {hidden_size}",
                hidden.len()
            )));
        }

        let layer = &self.layers[layer_index];
        let attn_input = rmsnorm_batch(
            hidden,
            &layer.input_layernorm_weight,
            self.config.rms_norm_eps,
            batch,
            hidden_size,
        );

        let runtime = self.metal_runtime.as_ref();
        let backend = self.projection_backend;
        let (mut q, mut k, v) = project_qkv_batched(backend, runtime, layer, &attn_input, batch)?;

        apply_rope_batched(
            &mut q,
            batch,
            self.config.num_attention_heads,
            self.config.head_dim,
            start_position,
            self.config.rope_theta,
        );
        apply_rope_batched(
            &mut k,
            batch,
            self.config.num_key_value_heads,
            self.config.head_dim,
            start_position,
            self.config.rope_theta,
        );

        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        for row in 0..batch {
            let position = start_position + row;
            let kv_offset = row * kv_dim;
            self.kv_cache[layer_index].write(
                position,
                &k[kv_offset..kv_offset + kv_dim],
                &v[kv_offset..kv_offset + kv_dim],
            );
            if let Some(metal_decode) = &self.metal_decode {
                metal_decode.upload_prefill_kv(
                    layer_index,
                    position,
                    &k[kv_offset..kv_offset + kv_dim],
                    &v[kv_offset..kv_offset + kv_dim],
                );
            }
        }

        let mut attn = vec![0.0_f32; batch * hidden_size];
        for row in 0..batch {
            let position = start_position + row;
            let q_offset = row * hidden_size;
            let row_attn = attention_decode(
                &q[q_offset..q_offset + hidden_size],
                &self.kv_cache[layer_index],
                position,
                &self.config,
            );
            attn[q_offset..q_offset + hidden_size].copy_from_slice(&row_attn);
        }

        let attn_out = project_quant_batched(backend, runtime, &layer.o_proj, &attn, batch, None)?;
        let mut residual = add(hidden, &attn_out);
        let mlp_input = rmsnorm_batch(
            &residual,
            &layer.post_attention_layernorm_weight,
            self.config.rms_norm_eps,
            batch,
            hidden_size,
        );
        let (gate, up) = project_gate_up_batched(backend, runtime, layer, &mlp_input, batch)?;
        let swiglu = swiglu(&gate, &up);
        let mlp_out =
            project_quant_batched(backend, runtime, &layer.down_proj, &swiglu, batch, None)?;
        add_in_place(&mut residual, &mlp_out);
        Ok(residual)
    }
}

fn project_quant(
    backend: QwenProjectionBackend,
    runtime: Option<&Q4MetalRuntime>,
    tensor: &QuantTensor,
    input: &[f32],
    linear_bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    match backend {
        QwenProjectionBackend::Cpu => tensor.matvec(input, linear_bias),
        QwenProjectionBackend::Metal => tensor.matvec_metal_with_runtime(
            input,
            linear_bias,
            runtime.ok_or_else(|| config_error("Metal projection backend missing runtime"))?,
        ),
    }
}

fn project_quant_batched(
    backend: QwenProjectionBackend,
    runtime: Option<&Q4MetalRuntime>,
    tensor: &QuantTensor,
    input: &[f32],
    batch: usize,
    linear_bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    if batch == 0 {
        return Err(config_error(
            "batched projection requires a non-empty batch",
        ));
    }
    if input.len() != batch * tensor.in_dim {
        return Err(config_error(format!(
            "{} batched input length {} does not match batch {batch} * {}",
            tensor.name,
            input.len(),
            tensor.in_dim
        )));
    }
    if let Some(bias) = linear_bias {
        if bias.len() != tensor.out_dim {
            return Err(config_error(format!(
                "{} linear bias length {} does not match {}",
                tensor.name,
                bias.len(),
                tensor.out_dim
            )));
        }
    }

    match backend {
        QwenProjectionBackend::Cpu => {
            let mut out = Vec::with_capacity(batch * tensor.out_dim);
            for row in input.chunks_exact(tensor.in_dim) {
                out.extend(tensor.matvec(row, linear_bias)?);
            }
            Ok(out)
        }
        QwenProjectionBackend::Metal => {
            let mut outputs = runtime
                .ok_or_else(|| config_error("Metal projection backend missing runtime"))?
                .matmul_many_batched(
                    input,
                    batch,
                    &[ProjectionRequest {
                        tensor,
                        linear_bias,
                    }],
                )?;
            Ok(outputs.remove(0))
        }
    }
}

fn project_qkv(
    backend: QwenProjectionBackend,
    runtime: Option<&Q4MetalRuntime>,
    layer: &QwenLayer,
    input: &[f32],
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    match backend {
        QwenProjectionBackend::Cpu => Ok((
            layer.q_proj.matvec(input, Some(&layer.q_bias))?,
            layer.k_proj.matvec(input, Some(&layer.k_bias))?,
            layer.v_proj.matvec(input, Some(&layer.v_bias))?,
        )),
        QwenProjectionBackend::Metal => {
            let outputs = runtime
                .ok_or_else(|| config_error("Metal projection backend missing runtime"))?
                .matvec_many(
                    input,
                    &[
                        ProjectionRequest {
                            tensor: &layer.q_proj,
                            linear_bias: Some(&layer.q_bias),
                        },
                        ProjectionRequest {
                            tensor: &layer.k_proj,
                            linear_bias: Some(&layer.k_bias),
                        },
                        ProjectionRequest {
                            tensor: &layer.v_proj,
                            linear_bias: Some(&layer.v_bias),
                        },
                    ],
                )?;
            let [q, k, v]: [Vec<f32>; 3] = outputs.try_into().map_err(|_| {
                config_error("qkv projection produced an unexpected number of outputs")
            })?;
            Ok((q, k, v))
        }
    }
}

fn project_qkv_batched(
    backend: QwenProjectionBackend,
    runtime: Option<&Q4MetalRuntime>,
    layer: &QwenLayer,
    input: &[f32],
    batch: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    match backend {
        QwenProjectionBackend::Cpu => {
            let mut q = Vec::with_capacity(batch * layer.q_proj.out_dim);
            let mut k = Vec::with_capacity(batch * layer.k_proj.out_dim);
            let mut v = Vec::with_capacity(batch * layer.v_proj.out_dim);
            for row in input.chunks_exact(layer.q_proj.in_dim) {
                q.extend(layer.q_proj.matvec(row, Some(&layer.q_bias))?);
                k.extend(layer.k_proj.matvec(row, Some(&layer.k_bias))?);
                v.extend(layer.v_proj.matvec(row, Some(&layer.v_bias))?);
            }
            Ok((q, k, v))
        }
        QwenProjectionBackend::Metal => {
            let outputs = runtime
                .ok_or_else(|| config_error("Metal projection backend missing runtime"))?
                .matmul_many_batched(
                    input,
                    batch,
                    &[
                        ProjectionRequest {
                            tensor: &layer.q_proj,
                            linear_bias: Some(&layer.q_bias),
                        },
                        ProjectionRequest {
                            tensor: &layer.k_proj,
                            linear_bias: Some(&layer.k_bias),
                        },
                        ProjectionRequest {
                            tensor: &layer.v_proj,
                            linear_bias: Some(&layer.v_bias),
                        },
                    ],
                )?;
            let [q, k, v]: [Vec<f32>; 3] = outputs.try_into().map_err(|_| {
                config_error("batched qkv projection produced an unexpected number of outputs")
            })?;
            Ok((q, k, v))
        }
    }
}

fn project_gate_up(
    backend: QwenProjectionBackend,
    runtime: Option<&Q4MetalRuntime>,
    layer: &QwenLayer,
    input: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    match backend {
        QwenProjectionBackend::Cpu => Ok((
            layer.gate_proj.matvec(input, None)?,
            layer.up_proj.matvec(input, None)?,
        )),
        QwenProjectionBackend::Metal => {
            let outputs = runtime
                .ok_or_else(|| config_error("Metal projection backend missing runtime"))?
                .matvec_many(
                    input,
                    &[
                        ProjectionRequest {
                            tensor: &layer.gate_proj,
                            linear_bias: None,
                        },
                        ProjectionRequest {
                            tensor: &layer.up_proj,
                            linear_bias: None,
                        },
                    ],
                )?;
            let [gate, up]: [Vec<f32>; 2] = outputs.try_into().map_err(|_| {
                config_error("gate/up projection produced an unexpected number of outputs")
            })?;
            Ok((gate, up))
        }
    }
}

fn project_gate_up_batched(
    backend: QwenProjectionBackend,
    runtime: Option<&Q4MetalRuntime>,
    layer: &QwenLayer,
    input: &[f32],
    batch: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    match backend {
        QwenProjectionBackend::Cpu => {
            let mut gate = Vec::with_capacity(batch * layer.gate_proj.out_dim);
            let mut up = Vec::with_capacity(batch * layer.up_proj.out_dim);
            for row in input.chunks_exact(layer.gate_proj.in_dim) {
                gate.extend(layer.gate_proj.matvec(row, None)?);
                up.extend(layer.up_proj.matvec(row, None)?);
            }
            Ok((gate, up))
        }
        QwenProjectionBackend::Metal => {
            let outputs = runtime
                .ok_or_else(|| config_error("Metal projection backend missing runtime"))?
                .matmul_many_batched(
                    input,
                    batch,
                    &[
                        ProjectionRequest {
                            tensor: &layer.gate_proj,
                            linear_bias: None,
                        },
                        ProjectionRequest {
                            tensor: &layer.up_proj,
                            linear_bias: None,
                        },
                    ],
                )?;
            let [gate, up]: [Vec<f32>; 2] = outputs.try_into().map_err(|_| {
                config_error("batched gate/up projection produced an unexpected number of outputs")
            })?;
            Ok((gate, up))
        }
    }
}

struct QwenLayer {
    input_layernorm_weight: Vec<f32>,
    post_attention_layernorm_weight: Vec<f32>,
    q_proj: QuantTensor,
    k_proj: QuantTensor,
    v_proj: QuantTensor,
    o_proj: QuantTensor,
    q_bias: Vec<f32>,
    k_bias: Vec<f32>,
    v_bias: Vec<f32>,
    gate_proj: QuantTensor,
    up_proj: QuantTensor,
    down_proj: QuantTensor,
}

impl QwenLayer {
    fn validate_shapes(&self, config: &QwenRuntimeConfig, index: usize) -> Result<()> {
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let prefix = format!("model.layers.{index}");
        ensure_len(
            &format!("{prefix}.input_layernorm.weight"),
            &self.input_layernorm_weight,
            config.hidden_size,
        )?;
        ensure_len(
            &format!("{prefix}.post_attention_layernorm.weight"),
            &self.post_attention_layernorm_weight,
            config.hidden_size,
        )?;
        ensure_quant_shape(
            &format!("{prefix}.self_attn.q_proj"),
            &self.q_proj,
            config.hidden_size,
            config.hidden_size,
        )?;
        ensure_quant_shape(
            &format!("{prefix}.self_attn.k_proj"),
            &self.k_proj,
            kv_dim,
            config.hidden_size,
        )?;
        ensure_quant_shape(
            &format!("{prefix}.self_attn.v_proj"),
            &self.v_proj,
            kv_dim,
            config.hidden_size,
        )?;
        ensure_quant_shape(
            &format!("{prefix}.self_attn.o_proj"),
            &self.o_proj,
            config.hidden_size,
            config.hidden_size,
        )?;
        ensure_len(
            &format!("{prefix}.self_attn.q_proj.bias"),
            &self.q_bias,
            config.hidden_size,
        )?;
        ensure_len(
            &format!("{prefix}.self_attn.k_proj.bias"),
            &self.k_bias,
            kv_dim,
        )?;
        ensure_len(
            &format!("{prefix}.self_attn.v_proj.bias"),
            &self.v_bias,
            kv_dim,
        )?;
        ensure_quant_shape(
            &format!("{prefix}.mlp.gate_proj"),
            &self.gate_proj,
            config.intermediate_size,
            config.hidden_size,
        )?;
        ensure_quant_shape(
            &format!("{prefix}.mlp.up_proj"),
            &self.up_proj,
            config.intermediate_size,
            config.hidden_size,
        )?;
        ensure_quant_shape(
            &format!("{prefix}.mlp.down_proj"),
            &self.down_proj,
            config.hidden_size,
            config.intermediate_size,
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct QwenRuntimeConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
}

impl QwenRuntimeConfig {
    fn read(hf_dir: &Path) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(hf_dir.join("config.json"))
                .map_err(|error| config_error(format!("failed to read config.json: {error}")))?,
        )
        .map_err(|error| config_error(format!("failed to parse config.json: {error}")))?;
        let config = value.get("text_config").unwrap_or(&value);
        let hidden_size = required_usize(config, "hidden_size")?;
        let num_attention_heads = required_usize(config, "num_attention_heads")?;
        if num_attention_heads == 0 {
            return Err(config_error(
                "num_attention_heads must be greater than zero",
            ));
        }
        let head_dim = config
            .get("head_dim")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(hidden_size / num_attention_heads);
        Ok(Self {
            hidden_size,
            intermediate_size: required_usize(config, "intermediate_size")?,
            num_hidden_layers: required_usize(config, "num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads: config
                .get("num_key_value_heads")
                .and_then(serde_json::Value::as_u64)
                .map(|value| value as usize)
                .unwrap_or(num_attention_heads),
            head_dim,
            vocab_size: required_usize(config, "vocab_size")?,
            rope_theta: config
                .get("rope_theta")
                .or_else(|| config.pointer("/rope_parameters/rope_theta"))
                .and_then(|value| value.as_f64())
                .unwrap_or(1_000_000.0) as f32,
            rms_norm_eps: config
                .get("rms_norm_eps")
                .and_then(|value| value.as_f64())
                .unwrap_or(1e-6) as f32,
            tie_word_embeddings: config
                .get("tie_word_embeddings")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true),
        })
    }

    fn read_gguf(gguf: &GgufFile) -> Result<Self> {
        let hidden_size = gguf.required_usize("qwen2.embedding_length")?;
        let num_attention_heads = gguf.required_usize("qwen2.attention.head_count")?;
        if num_attention_heads == 0 {
            return Err(config_error(
                "qwen2.attention.head_count must be greater than zero",
            ));
        }
        Ok(Self {
            hidden_size,
            intermediate_size: gguf.required_usize("qwen2.feed_forward_length")?,
            num_hidden_layers: gguf.required_usize("qwen2.block_count")?,
            num_attention_heads,
            num_key_value_heads: gguf.required_usize("qwen2.attention.head_count_kv")?,
            head_dim: hidden_size / num_attention_heads,
            vocab_size: gguf
                .metadata_array_len("tokenizer.ggml.tokens")
                .ok_or_else(|| config_error("missing tokenizer.ggml.tokens metadata"))?,
            rope_theta: gguf.required_f32("qwen2.rope.freq_base")?,
            rms_norm_eps: gguf.required_f32("qwen2.attention.layer_norm_rms_epsilon")?,
            tie_word_embeddings: !gguf.has_tensor("output.weight"),
        })
    }
}

struct QuantTensor {
    name: String,
    encoding: QuantEncoding,
    out_dim: usize,
    in_dim: usize,
    packed_per_row: usize,
    groups_per_row: usize,
    group_size: usize,
    weight: Vec<u32>,
    q8_values: Vec<i8>,
    scales: Vec<f32>,
    biases: Vec<f32>,
    metal: Option<QuantTensorMetal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantEncoding {
    AffineQ4,
    GgufQ8_0,
}

struct QuantTensorMetal {
    q4_weight: Option<Buffer>,
    q8_values: Option<Buffer>,
    scales: Buffer,
    biases: Option<Buffer>,
    zero_linear_bias: Buffer,
    output: Buffer,
    dims: Buffer,
}

struct Q4MetalRuntime {
    device: Device,
    queue: CommandQueue,
    q4_pipeline: ComputePipelineState,
    q4_batched_pipeline: ComputePipelineState,
    q8_pipeline: ComputePipelineState,
    rmsnorm_pipeline: ComputePipelineState,
    rope_pipeline: ComputePipelineState,
    kv_write_pipeline: ComputePipelineState,
    attention_pipeline: ComputePipelineState,
    add_pipeline: ComputePipelineState,
    swiglu_pipeline: ComputePipelineState,
    argmax_pipeline: ComputePipelineState,
    scratch: Mutex<MetalScratch>,
}

struct MetalScratch {
    input: Option<Buffer>,
    batch_info: Option<Buffer>,
    linear_biases: Vec<Option<Buffer>>,
    outputs: Vec<Option<Buffer>>,
}

struct MetalDecodeState {
    hidden: Buffer,
    attn_input: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    attn_out: Buffer,
    residual: Buffer,
    mlp_input: Buffer,
    normed_hidden: Buffer,
    gate: Buffer,
    up: Buffer,
    swiglu: Buffer,
    mlp_out: Buffer,
    logits: Buffer,
    argmax_index: Buffer,
    output_norm_weight: Buffer,
    layers: Vec<MetalLayerState>,
}

struct MetalLayerState {
    input_norm_weight: Buffer,
    post_norm_weight: Buffer,
    q_bias: Buffer,
    k_bias: Buffer,
    v_bias: Buffer,
    kv: MetalKvCache,
}

struct MetalKvCache {
    keys: Buffer,
    values: Buffer,
}

impl MetalDecodeState {
    fn new(
        device: &Device,
        max_seq: usize,
        config: &QwenRuntimeConfig,
        layers: &[QwenLayer],
        output_norm_weight: &[f32],
    ) -> Self {
        let hidden_size = config.hidden_size;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let intermediate_size = config.intermediate_size;
        let layer_states = layers
            .iter()
            .map(|layer| MetalLayerState {
                input_norm_weight: new_f32_buffer(device, &layer.input_layernorm_weight),
                post_norm_weight: new_f32_buffer(device, &layer.post_attention_layernorm_weight),
                q_bias: new_f32_buffer(device, &layer.q_bias),
                k_bias: new_f32_buffer(device, &layer.k_bias),
                v_bias: new_f32_buffer(device, &layer.v_bias),
                kv: MetalKvCache {
                    keys: new_shared_buffer(device, max_seq * kv_dim * std::mem::size_of::<f32>()),
                    values: new_shared_buffer(
                        device,
                        max_seq * kv_dim * std::mem::size_of::<f32>(),
                    ),
                },
            })
            .collect();

        Self {
            hidden: new_shared_buffer(device, hidden_size * std::mem::size_of::<f32>()),
            attn_input: new_shared_buffer(device, hidden_size * std::mem::size_of::<f32>()),
            q: new_shared_buffer(device, hidden_size * std::mem::size_of::<f32>()),
            k: new_shared_buffer(device, kv_dim * std::mem::size_of::<f32>()),
            v: new_shared_buffer(device, kv_dim * std::mem::size_of::<f32>()),
            attn: new_shared_buffer(device, hidden_size * std::mem::size_of::<f32>()),
            attn_out: new_shared_buffer(device, hidden_size * std::mem::size_of::<f32>()),
            residual: new_shared_buffer(device, hidden_size * std::mem::size_of::<f32>()),
            mlp_input: new_shared_buffer(device, hidden_size * std::mem::size_of::<f32>()),
            normed_hidden: new_shared_buffer(device, hidden_size * std::mem::size_of::<f32>()),
            gate: new_shared_buffer(device, intermediate_size * std::mem::size_of::<f32>()),
            up: new_shared_buffer(device, intermediate_size * std::mem::size_of::<f32>()),
            swiglu: new_shared_buffer(device, intermediate_size * std::mem::size_of::<f32>()),
            mlp_out: new_shared_buffer(device, hidden_size * std::mem::size_of::<f32>()),
            logits: new_shared_buffer(device, config.vocab_size * std::mem::size_of::<f32>()),
            argmax_index: new_shared_buffer(device, std::mem::size_of::<u32>()),
            output_norm_weight: new_f32_buffer(device, output_norm_weight),
            layers: layer_states,
        }
    }

    fn upload_prefill_kv(&self, layer_index: usize, position: usize, key: &[f32], value: &[f32]) {
        let kv_dim = key.len();
        debug_assert_eq!(value.len(), kv_dim);
        let offset = position * kv_dim;
        write_f32_buffer_at(&self.layers[layer_index].kv.keys, offset, key);
        write_f32_buffer_at(&self.layers[layer_index].kv.values, offset, value);
    }
}

struct ProjectionRequest<'a> {
    tensor: &'a QuantTensor,
    linear_bias: Option<&'a [f32]>,
}

fn compile_pipeline(
    device: &Device,
    library: &Library,
    function_name: &str,
) -> Result<ComputePipelineState> {
    let function = library.get_function(function_name, None).map_err(|error| {
        TinyAgentError::Backend(format!(
            "failed to load Metal kernel {function_name}: {error}"
        ))
    })?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| {
            TinyAgentError::Backend(format!(
                "failed to create Metal pipeline {function_name}: {error}"
            ))
        })
}

impl Q4MetalRuntime {
    fn new() -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| {
            TinyAgentError::Configuration(
                "no default Metal device found; TinyEngine requires Apple Silicon or a Metal-capable GPU"
                    .to_string(),
            )
        })?;
        let library = device
            .new_library_with_source(q4_affine_matvec_source(), &CompileOptions::new())
            .map_err(|error| {
                TinyAgentError::Backend(format!("failed to compile q4 matvec kernel: {error}"))
            })?;
        let q4_function = library
            .get_function("q4_affine_matvec_f32", None)
            .map_err(|error| {
                TinyAgentError::Backend(format!("failed to load q4 matvec kernel: {error}"))
            })?;
        let q4_pipeline = device
            .new_compute_pipeline_state_with_function(&q4_function)
            .map_err(|error| {
                TinyAgentError::Backend(format!("failed to create q4 matvec pipeline: {error}"))
            })?;
        let q4_batched_function =
            library
                .get_function("q4_affine_matmat_f32", None)
                .map_err(|error| {
                    TinyAgentError::Backend(format!(
                        "failed to load q4 batched matmul kernel: {error}"
                    ))
                })?;
        let q4_batched_pipeline = device
            .new_compute_pipeline_state_with_function(&q4_batched_function)
            .map_err(|error| {
                TinyAgentError::Backend(format!(
                    "failed to create q4 batched matmul pipeline: {error}"
                ))
            })?;
        let q8_function = library
            .get_function("q8_0_matvec_f32", None)
            .map_err(|error| {
                TinyAgentError::Backend(format!("failed to load q8_0 matvec kernel: {error}"))
            })?;
        let q8_pipeline = device
            .new_compute_pipeline_state_with_function(&q8_function)
            .map_err(|error| {
                TinyAgentError::Backend(format!("failed to create q8_0 matvec pipeline: {error}"))
            })?;
        let rmsnorm_pipeline = compile_pipeline(&device, &library, "rmsnorm_f32")?;
        let rope_pipeline = compile_pipeline(&device, &library, "rope_inplace_f32")?;
        let kv_write_pipeline = compile_pipeline(&device, &library, "write_kv_cache_f32")?;
        let attention_pipeline = compile_pipeline(&device, &library, "attention_decode_f32")?;
        let add_pipeline = compile_pipeline(&device, &library, "add_f32")?;
        let swiglu_pipeline = compile_pipeline(&device, &library, "swiglu_f32")?;
        let argmax_pipeline = compile_pipeline(&device, &library, "argmax_f32")?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            q4_pipeline,
            q4_batched_pipeline,
            q8_pipeline,
            rmsnorm_pipeline,
            rope_pipeline,
            kv_write_pipeline,
            attention_pipeline,
            add_pipeline,
            swiglu_pipeline,
            argmax_pipeline,
            scratch: Mutex::new(MetalScratch {
                input: None,
                batch_info: None,
                linear_biases: Vec::new(),
                outputs: Vec::new(),
            }),
        })
    }

    fn matvec_many(
        &self,
        input: &[f32],
        requests: &[ProjectionRequest<'_>],
    ) -> Result<Vec<Vec<f32>>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        for request in requests {
            if request.tensor.encoding != QuantEncoding::AffineQ4 {
                return Err(config_error(format!(
                    "{} cannot run on the q4 Metal kernel",
                    request.tensor.name
                )));
            }
            request
                .tensor
                .validate_matvec_input(input, request.linear_bias)?;
        }

        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| config_error("Metal scratch buffer lock was poisoned"))?;
        write_f32_buffer(&self.device, &mut scratch.input, input);
        let linear_bias_count = requests
            .iter()
            .filter(|request| request.linear_bias.is_some())
            .count();
        while scratch.linear_biases.len() < linear_bias_count {
            scratch.linear_biases.push(None);
        }
        let mut next_bias_buffer = 0_usize;
        for request in requests {
            if let Some(bias) = request.linear_bias {
                write_f32_buffer(
                    &self.device,
                    &mut scratch.linear_biases[next_bias_buffer],
                    bias,
                );
                next_bias_buffer += 1;
            }
        }
        let buffer_x = scratch
            .input
            .as_ref()
            .ok_or_else(|| config_error("missing Metal input scratch buffer"))?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.q4_pipeline);

        let mut next_bias_buffer = 0_usize;
        for request in requests {
            let metal = request.tensor.metal.as_ref().ok_or_else(|| {
                config_error(format!(
                    "{} has not been uploaded to Metal",
                    request.tensor.name
                ))
            })?;
            let q4_weight = metal.q4_weight.as_ref().ok_or_else(|| {
                config_error(format!(
                    "{} is missing q4 Metal weight buffer",
                    request.tensor.name
                ))
            })?;
            let biases = metal.biases.as_ref().ok_or_else(|| {
                config_error(format!(
                    "{} is missing q4 Metal bias buffer",
                    request.tensor.name
                ))
            })?;
            let linear_bias_buffer = match request.linear_bias {
                Some(_) => {
                    let buffer = scratch.linear_biases[next_bias_buffer]
                        .as_ref()
                        .ok_or_else(|| config_error("missing Metal linear bias scratch buffer"))?;
                    next_bias_buffer += 1;
                    buffer
                }
                None => &metal.zero_linear_bias,
            };

            encoder.set_buffer(0, Some(&buffer_x), 0);
            encoder.set_buffer(1, Some(q4_weight), 0);
            encoder.set_buffer(2, Some(&metal.scales), 0);
            encoder.set_buffer(3, Some(biases), 0);
            encoder.set_buffer(4, Some(linear_bias_buffer), 0);
            encoder.set_buffer(5, Some(&metal.output), 0);
            encoder.set_buffer(6, Some(&metal.dims), 0);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: request.tensor.out_dim.div_ceil(8) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut outputs = Vec::with_capacity(requests.len());
        for request in requests {
            let metal = request
                .tensor
                .metal
                .as_ref()
                .ok_or_else(|| config_error("missing Metal output buffer"))?;
            let ptr = metal.output.contents().cast::<f32>();
            outputs
                .push(unsafe { std::slice::from_raw_parts(ptr, request.tensor.out_dim).to_vec() });
        }
        Ok(outputs)
    }

    fn matmul_many_batched(
        &self,
        input: &[f32],
        batch: usize,
        requests: &[ProjectionRequest<'_>],
    ) -> Result<Vec<Vec<f32>>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        if batch == 0 {
            return Err(config_error(
                "batched Metal projection requires a non-empty batch",
            ));
        }
        for request in requests {
            if request.tensor.encoding != QuantEncoding::AffineQ4 {
                return Err(config_error(format!(
                    "{} cannot run on the batched q4 Metal kernel",
                    request.tensor.name
                )));
            }
            if input.len() != batch * request.tensor.in_dim {
                return Err(config_error(format!(
                    "{} batched input length {} does not match batch {batch} * {}",
                    request.tensor.name,
                    input.len(),
                    request.tensor.in_dim
                )));
            }
            if let Some(bias) = request.linear_bias {
                if bias.len() != request.tensor.out_dim {
                    return Err(config_error(format!(
                        "{} linear bias length {} does not match {}",
                        request.tensor.name,
                        bias.len(),
                        request.tensor.out_dim
                    )));
                }
            }
        }

        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| config_error("Metal scratch buffer lock was poisoned"))?;
        write_f32_buffer(&self.device, &mut scratch.input, input);
        write_u32_buffer(&self.device, &mut scratch.batch_info, &[batch as u32]);

        let linear_bias_count = requests
            .iter()
            .filter(|request| request.linear_bias.is_some())
            .count();
        while scratch.linear_biases.len() < linear_bias_count {
            scratch.linear_biases.push(None);
        }
        let mut next_bias_buffer = 0_usize;
        for request in requests {
            if let Some(bias) = request.linear_bias {
                write_f32_buffer(
                    &self.device,
                    &mut scratch.linear_biases[next_bias_buffer],
                    bias,
                );
                next_bias_buffer += 1;
            }
        }

        while scratch.outputs.len() < requests.len() {
            scratch.outputs.push(None);
        }
        for (index, request) in requests.iter().enumerate() {
            let bytes = (batch * request.tensor.out_dim * std::mem::size_of::<f32>()) as u64;
            ensure_shared_buffer(&self.device, &mut scratch.outputs[index], bytes);
        }

        let buffer_x = scratch
            .input
            .as_ref()
            .ok_or_else(|| config_error("missing Metal input scratch buffer"))?;
        let batch_info = scratch
            .batch_info
            .as_ref()
            .ok_or_else(|| config_error("missing Metal batch info scratch buffer"))?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.q4_batched_pipeline);

        let mut next_bias_buffer = 0_usize;
        for (index, request) in requests.iter().enumerate() {
            let metal = request.tensor.metal.as_ref().ok_or_else(|| {
                config_error(format!(
                    "{} has not been uploaded to Metal",
                    request.tensor.name
                ))
            })?;
            let q4_weight = metal.q4_weight.as_ref().ok_or_else(|| {
                config_error(format!(
                    "{} is missing q4 Metal weight buffer",
                    request.tensor.name
                ))
            })?;
            let biases = metal.biases.as_ref().ok_or_else(|| {
                config_error(format!(
                    "{} is missing q4 Metal bias buffer",
                    request.tensor.name
                ))
            })?;
            let linear_bias_buffer = match request.linear_bias {
                Some(_) => {
                    let buffer = scratch.linear_biases[next_bias_buffer]
                        .as_ref()
                        .ok_or_else(|| config_error("missing Metal linear bias scratch buffer"))?;
                    next_bias_buffer += 1;
                    buffer
                }
                None => &metal.zero_linear_bias,
            };
            let output_buffer = scratch.outputs[index]
                .as_ref()
                .ok_or_else(|| config_error("missing Metal batched output scratch buffer"))?;

            encoder.set_buffer(0, Some(buffer_x), 0);
            encoder.set_buffer(1, Some(q4_weight), 0);
            encoder.set_buffer(2, Some(&metal.scales), 0);
            encoder.set_buffer(3, Some(biases), 0);
            encoder.set_buffer(4, Some(linear_bias_buffer), 0);
            encoder.set_buffer(5, Some(output_buffer), 0);
            encoder.set_buffer(6, Some(&metal.dims), 0);
            encoder.set_buffer(7, Some(batch_info), 0);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: request.tensor.out_dim as u64,
                    height: batch.div_ceil(8) as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut outputs = Vec::with_capacity(requests.len());
        for (index, request) in requests.iter().enumerate() {
            let len = batch * request.tensor.out_dim;
            let output_buffer = scratch.outputs[index]
                .as_ref()
                .ok_or_else(|| config_error("missing Metal batched output scratch buffer"))?;
            let ptr = output_buffer.contents().cast::<f32>();
            outputs.push(unsafe { std::slice::from_raw_parts(ptr, len).to_vec() });
        }
        Ok(outputs)
    }

    fn matvec_q8(
        &self,
        input: &[f32],
        tensor: &QuantTensor,
        linear_bias: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        if tensor.encoding != QuantEncoding::GgufQ8_0 {
            return Err(config_error(format!(
                "{} cannot run on the q8_0 Metal kernel",
                tensor.name
            )));
        }
        tensor.validate_matvec_input(input, linear_bias)?;
        let metal = tensor.metal.as_ref().ok_or_else(|| {
            config_error(format!("{} has not been uploaded to Metal", tensor.name))
        })?;
        let q8_values = metal.q8_values.as_ref().ok_or_else(|| {
            config_error(format!("{} is missing q8 Metal value buffer", tensor.name))
        })?;

        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| config_error("Metal scratch buffer lock was poisoned"))?;
        write_f32_buffer(&self.device, &mut scratch.input, input);
        if let Some(bias) = linear_bias {
            if scratch.linear_biases.is_empty() {
                scratch.linear_biases.push(None);
            }
            write_f32_buffer(&self.device, &mut scratch.linear_biases[0], bias);
        }
        let buffer_x = scratch
            .input
            .as_ref()
            .ok_or_else(|| config_error("missing Metal input scratch buffer"))?;
        let linear_bias_buffer = match linear_bias {
            Some(_) => scratch.linear_biases[0]
                .as_ref()
                .ok_or_else(|| config_error("missing Metal linear bias scratch buffer"))?,
            None => &metal.zero_linear_bias,
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.q8_pipeline);
        encoder.set_buffer(0, Some(&buffer_x), 0);
        encoder.set_buffer(1, Some(q8_values), 0);
        encoder.set_buffer(2, Some(&metal.scales), 0);
        encoder.set_buffer(3, Some(linear_bias_buffer), 0);
        encoder.set_buffer(4, Some(&metal.output), 0);
        encoder.set_buffer(5, Some(&metal.dims), 0);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: tensor.out_dim as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let ptr = metal.output.contents().cast::<f32>();
        Ok(unsafe { std::slice::from_raw_parts(ptr, tensor.out_dim).to_vec() })
    }
}

fn write_f32_buffer(device: &Device, slot: &mut Option<Buffer>, values: &[f32]) {
    let bytes = std::mem::size_of_val(values) as u64;
    ensure_shared_buffer(device, slot, bytes);
    if bytes == 0 {
        return;
    }

    let buffer = slot.as_ref().expect("scratch buffer was allocated above");
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            buffer.contents().cast::<f32>(),
            values.len(),
        );
    }
}

fn write_u32_buffer(device: &Device, slot: &mut Option<Buffer>, values: &[u32]) {
    let bytes = std::mem::size_of_val(values) as u64;
    ensure_shared_buffer(device, slot, bytes);
    if bytes == 0 {
        return;
    }

    let buffer = slot.as_ref().expect("scratch buffer was allocated above");
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            buffer.contents().cast::<u32>(),
            values.len(),
        );
    }
}

fn new_shared_buffer(device: &Device, bytes: usize) -> Buffer {
    device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared)
}

fn new_f32_buffer(device: &Device, values: &[f32]) -> Buffer {
    device.new_buffer_with_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn write_f32_buffer_direct(buffer: &Buffer, values: &[f32]) {
    if values.is_empty() {
        return;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            buffer.contents().cast::<f32>(),
            values.len(),
        );
    }
}

fn write_f32_buffer_at(buffer: &Buffer, offset: usize, values: &[f32]) {
    if values.is_empty() {
        return;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            buffer.contents().cast::<f32>().add(offset),
            values.len(),
        );
    }
}

fn read_f32_buffer(buffer: &Buffer, len: usize) -> Vec<f32> {
    let ptr = buffer.contents().cast::<f32>();
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}

fn ensure_shared_buffer(device: &Device, slot: &mut Option<Buffer>, bytes: u64) {
    if slot.as_ref().is_none_or(|buffer| buffer.length() < bytes) {
        *slot = Some(device.new_buffer(bytes.max(1), MTLResourceOptions::StorageModeShared));
    }
}

impl QuantTensor {
    fn validate_matvec_input(&self, input: &[f32], linear_bias: Option<&[f32]>) -> Result<()> {
        if input.len() != self.in_dim {
            return Err(config_error(format!(
                "{} input length {} does not match {}",
                self.name,
                input.len(),
                self.in_dim
            )));
        }
        if let Some(bias) = linear_bias {
            if bias.len() != self.out_dim {
                return Err(config_error(format!(
                    "{} linear bias length {} does not match {}",
                    self.name,
                    bias.len(),
                    self.out_dim
                )));
            }
        }
        Ok(())
    }

    fn upload_metal(&mut self, runtime: &Q4MetalRuntime) -> Result<()> {
        let dims = [
            self.out_dim as u32,
            self.in_dim as u32,
            self.packed_per_row as u32,
            self.groups_per_row as u32,
            self.group_size as u32,
        ];
        let zero_bias = vec![0.0_f32; self.out_dim];
        let q4_weight = if self.encoding == QuantEncoding::AffineQ4 {
            Some(runtime.device.new_buffer_with_data(
                self.weight.as_ptr().cast(),
                std::mem::size_of_val(self.weight.as_slice()) as u64,
                MTLResourceOptions::StorageModeShared,
            ))
        } else {
            None
        };
        let q8_values = if self.encoding == QuantEncoding::GgufQ8_0 {
            Some(runtime.device.new_buffer_with_data(
                self.q8_values.as_ptr().cast(),
                std::mem::size_of_val(self.q8_values.as_slice()) as u64,
                MTLResourceOptions::StorageModeShared,
            ))
        } else {
            None
        };
        let biases = if self.encoding == QuantEncoding::AffineQ4 {
            Some(runtime.device.new_buffer_with_data(
                self.biases.as_ptr().cast(),
                std::mem::size_of_val(self.biases.as_slice()) as u64,
                MTLResourceOptions::StorageModeShared,
            ))
        } else {
            None
        };
        self.metal = Some(QuantTensorMetal {
            q4_weight,
            q8_values,
            scales: runtime.device.new_buffer_with_data(
                self.scales.as_ptr().cast(),
                std::mem::size_of_val(self.scales.as_slice()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            biases,
            zero_linear_bias: runtime.device.new_buffer_with_data(
                zero_bias.as_ptr().cast(),
                std::mem::size_of_val(zero_bias.as_slice()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            output: runtime.device.new_buffer(
                (self.out_dim * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            dims: runtime.device.new_buffer_with_data(
                dims.as_ptr().cast(),
                std::mem::size_of_val(&dims) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
        });
        Ok(())
    }

    fn dequantize_row(&self, row: usize) -> Result<Vec<f32>> {
        if row >= self.out_dim {
            return Err(config_error(format!(
                "{} row {row} out of range {}",
                self.name, self.out_dim
            )));
        }
        let mut out = vec![0.0_f32; self.in_dim];
        for input in 0..self.in_dim {
            out[input] = self.dequant_value(row, input);
        }
        Ok(out)
    }

    fn matvec(&self, input: &[f32], linear_bias: Option<&[f32]>) -> Result<Vec<f32>> {
        self.validate_matvec_input(input, linear_bias)?;

        let mut out = vec![0.0_f32; self.out_dim];
        for row in 0..self.out_dim {
            let mut acc = 0.0_f32;
            match self.encoding {
                QuantEncoding::AffineQ4 => {
                    for input_index in 0..self.in_dim {
                        acc += input[input_index] * self.dequant_value(row, input_index);
                    }
                }
                QuantEncoding::GgufQ8_0 => {
                    let row_offset = row * self.in_dim;
                    let scale_offset = row * self.groups_per_row;
                    for input_index in 0..self.in_dim {
                        let q = self.q8_values[row_offset + input_index] as f32;
                        let scale = self.scales[scale_offset + input_index / self.group_size];
                        acc += input[input_index] * q * scale;
                    }
                }
            }
            if let Some(bias) = linear_bias {
                acc += bias[row];
            }
            out[row] = acc;
        }
        Ok(out)
    }

    fn matvec_metal_with_runtime(
        &self,
        input: &[f32],
        linear_bias: Option<&[f32]>,
        runtime: &Q4MetalRuntime,
    ) -> Result<Vec<f32>> {
        if self.encoding == QuantEncoding::GgufQ8_0 {
            return runtime.matvec_q8(input, self, linear_bias);
        }
        let mut outputs = runtime.matvec_many(
            input,
            &[ProjectionRequest {
                tensor: self,
                linear_bias,
            }],
        )?;
        Ok(outputs.remove(0))
    }

    #[inline]
    fn dequant_value(&self, row: usize, input: usize) -> f32 {
        let word = self.weight[row * self.packed_per_row + input / 8];
        let q = ((word >> (4 * (input % 8))) & 0xF) as f32;
        let group = input / self.group_size;
        let group_offset = row * self.groups_per_row + group;
        q * self.scales[group_offset] + self.biases[group_offset]
    }
}

fn q4_affine_matvec_source() -> &'static str {
    r#"
        #include <metal_stdlib>
        using namespace metal;

        #define Q4_THREADS 128
        #define Q4_BATCH_TILE 8
        #define Q4_ROW_TILE 8
        #define SCALAR_THREADS 256
        #define MAX_ATTN_SEQ 2048

        kernel void rmsnorm_f32(
            device const float* input [[buffer(0)]],
            device const float* weight [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant uint& len [[buffer(3)]],
            constant float& eps [[buffer(4)]],
            uint tid [[thread_index_in_threadgroup]]
        ) {
            threadgroup float partial[SCALAR_THREADS];
            float sum = 0.0f;
            for (uint index = tid; index < len; index += SCALAR_THREADS) {
                const float value = input[index];
                sum += value * value;
            }
            partial[tid] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = SCALAR_THREADS / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    partial[tid] += partial[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const float inv_rms = rsqrt(partial[0] / float(len) + eps);
            for (uint index = tid; index < len; index += SCALAR_THREADS) {
                output[index] = input[index] * weight[index] * inv_rms;
            }
        }

        kernel void rope_inplace_f32(
            device float* values [[buffer(0)]],
            constant uint* params [[buffer(1)]],
            constant float& theta [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            const uint heads = params[0];
            const uint head_dim = params[1];
            const uint position = params[2];
            const uint half_dim = head_dim / 2;
            const uint total = heads * half_dim;
            if (gid >= total) {
                return;
            }
            const uint head = gid / half_dim;
            const uint index = gid - head * half_dim;
            const uint offset = head * head_dim;
            const float freq = pow(theta, -(2.0f * float(index)) / float(head_dim));
            const float angle = float(position) * freq;
            const float s = sin(angle);
            const float c = cos(angle);
            const float a = values[offset + index];
            const float b = values[offset + index + half_dim];
            values[offset + index] = a * c - b * s;
            values[offset + index + half_dim] = b * c + a * s;
        }

        kernel void write_kv_cache_f32(
            device const float* key [[buffer(0)]],
            device const float* value [[buffer(1)]],
            device float* keys [[buffer(2)]],
            device float* values [[buffer(3)]],
            constant uint* params [[buffer(4)]],
            uint gid [[thread_position_in_grid]]
        ) {
            const uint position = params[0];
            const uint kv_dim = params[1];
            if (gid >= kv_dim) {
                return;
            }
            const uint offset = position * kv_dim + gid;
            keys[offset] = key[gid];
            values[offset] = value[gid];
        }

        kernel void attention_decode_f32(
            device const float* q [[buffer(0)]],
            device const float* keys [[buffer(1)]],
            device const float* values [[buffer(2)]],
            device float* output [[buffer(3)]],
            constant uint* params [[buffer(4)]],
            constant float& scale [[buffer(5)]],
            uint head [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]]
        ) {
            const uint position = params[0];
            const uint num_heads = params[1];
            const uint num_kv_heads = params[2];
            const uint head_dim = params[3];
            if (head >= num_heads || position >= MAX_ATTN_SEQ || tid != 0) {
                return;
            }

            threadgroup float scores[MAX_ATTN_SEQ];
            const uint kv_repeat = num_heads / num_kv_heads;
            const uint kv_head = head / kv_repeat;
            const uint q_offset = head * head_dim;
            float max_score = -3.402823466e+38F;
            for (uint past = 0; past <= position; ++past) {
                float dot = 0.0f;
                const uint kv_offset = (past * num_kv_heads + kv_head) * head_dim;
                for (uint dim = 0; dim < head_dim; ++dim) {
                    dot += q[q_offset + dim] * keys[kv_offset + dim];
                }
                const float score = dot * scale;
                scores[past] = score;
                max_score = max(max_score, score);
            }

            float sum = 0.0f;
            for (uint past = 0; past <= position; ++past) {
                const float score = exp(scores[past] - max_score);
                scores[past] = score;
                sum += score;
            }
            const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (uint dim = 0; dim < head_dim; ++dim) {
                float acc = 0.0f;
                for (uint past = 0; past <= position; ++past) {
                    const uint kv_offset = (past * num_kv_heads + kv_head) * head_dim;
                    acc += scores[past] * inv_sum * values[kv_offset + dim];
                }
                output[q_offset + dim] = acc;
            }
        }

        kernel void add_f32(
            device const float* lhs [[buffer(0)]],
            device const float* rhs [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant uint& len [[buffer(3)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid < len) {
                output[gid] = lhs[gid] + rhs[gid];
            }
        }

        kernel void swiglu_f32(
            device const float* gate [[buffer(0)]],
            device const float* up [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant uint& len [[buffer(3)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid < len) {
                const float g = gate[gid];
                output[gid] = (g / (1.0f + exp(-g))) * up[gid];
            }
        }

        kernel void argmax_f32(
            device const float* values [[buffer(0)]],
            device uint* output_index [[buffer(1)]],
            constant uint& len [[buffer(2)]],
            uint tid [[thread_index_in_threadgroup]]
        ) {
            threadgroup float best_values[SCALAR_THREADS];
            threadgroup uint best_indices[SCALAR_THREADS];

            float best_value = -3.402823466e+38F;
            uint best_index = 0;
            for (uint index = tid; index < len; index += SCALAR_THREADS) {
                const float value = values[index];
                if (value > best_value) {
                    best_value = value;
                    best_index = index;
                }
            }
            best_values[tid] = best_value;
            best_indices[tid] = best_index;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = SCALAR_THREADS / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    const float other_value = best_values[tid + stride];
                    const uint other_index = best_indices[tid + stride];
                    if (other_value > best_values[tid] ||
                        (other_value == best_values[tid] && other_index < best_indices[tid])) {
                        best_values[tid] = other_value;
                        best_indices[tid] = other_index;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                output_index[0] = best_indices[0];
            }
        }

        kernel void q4_affine_matvec_f32(
            device const float* x [[buffer(0)]],
            device const uint* weight [[buffer(1)]],
            device const float* scales [[buffer(2)]],
            device const float* biases [[buffer(3)]],
            device const float* linear_bias [[buffer(4)]],
            device float* out [[buffer(5)]],
            constant uint* dims [[buffer(6)]],
            uint row_tile [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]]
        ) {
            const uint row_base = row_tile * Q4_ROW_TILE;
            const uint out_dim = dims[0];
            const uint in_dim = dims[1];
            const uint packed_per_row = dims[2];
            const uint groups_per_row = dims[3];
            const uint group_size = dims[4];
            if (row_base >= out_dim) {
                return;
            }

            threadgroup float partial[Q4_ROW_TILE * Q4_THREADS];
            float acc[Q4_ROW_TILE];
            for (uint lane = 0; lane < Q4_ROW_TILE; ++lane) {
                acc[lane] = 0.0f;
            }

            for (uint input = tid; input < in_dim; input += Q4_THREADS) {
                const float x_value = x[input];
                for (uint lane = 0; lane < Q4_ROW_TILE; ++lane) {
                    const uint row = row_base + lane;
                    if (row < out_dim) {
                        const uint packed = weight[row * packed_per_row + input / 8];
                        const uint q = (packed >> (4 * (input & 7))) & 0xFu;
                        const uint group = input / group_size;
                        const uint group_offset = row * groups_per_row + group;
                        const float w = float(q) * scales[group_offset] + biases[group_offset];
                        acc[lane] += x_value * w;
                    }
                }
            }
            for (uint lane = 0; lane < Q4_ROW_TILE; ++lane) {
                partial[lane * Q4_THREADS + tid] = acc[lane];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = Q4_THREADS / 2; stride > 0; stride >>= 1) {
                for (uint lane = 0; lane < Q4_ROW_TILE; ++lane) {
                    if (tid < stride) {
                        const uint offset = lane * Q4_THREADS + tid;
                        partial[offset] += partial[offset + stride];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                for (uint lane = 0; lane < Q4_ROW_TILE; ++lane) {
                    const uint row = row_base + lane;
                    if (row < out_dim) {
                        out[row] = partial[lane * Q4_THREADS] + linear_bias[row];
                    }
                }
            }
        }

        kernel void q4_affine_matmat_f32(
            device const float* x [[buffer(0)]],
            device const uint* weight [[buffer(1)]],
            device const float* scales [[buffer(2)]],
            device const float* biases [[buffer(3)]],
            device const float* linear_bias [[buffer(4)]],
            device float* out [[buffer(5)]],
            constant uint* dims [[buffer(6)]],
            constant uint* batch_info [[buffer(7)]],
            uint2 grid [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]]
        ) {
            const uint row = grid.x;
            const uint batch_base = grid.y * Q4_BATCH_TILE;
            const uint batch_count = batch_info[0];
            const uint out_dim = dims[0];
            const uint in_dim = dims[1];
            const uint packed_per_row = dims[2];
            const uint groups_per_row = dims[3];
            const uint group_size = dims[4];
            if (row >= out_dim) {
                return;
            }

            threadgroup float partial[Q4_BATCH_TILE * Q4_THREADS];
            float acc[Q4_BATCH_TILE];
            for (uint lane = 0; lane < Q4_BATCH_TILE; ++lane) {
                acc[lane] = 0.0f;
            }

            for (uint input = tid; input < in_dim; input += Q4_THREADS) {
                const uint packed = weight[row * packed_per_row + input / 8];
                const uint q = (packed >> (4 * (input & 7))) & 0xFu;
                const uint group = input / group_size;
                const uint group_offset = row * groups_per_row + group;
                const float w = float(q) * scales[group_offset] + biases[group_offset];
                for (uint lane = 0; lane < Q4_BATCH_TILE; ++lane) {
                    const uint batch = batch_base + lane;
                    if (batch < batch_count) {
                        acc[lane] += x[batch * in_dim + input] * w;
                    }
                }
            }
            for (uint lane = 0; lane < Q4_BATCH_TILE; ++lane) {
                partial[lane * Q4_THREADS + tid] = acc[lane];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = Q4_THREADS / 2; stride > 0; stride >>= 1) {
                for (uint lane = 0; lane < Q4_BATCH_TILE; ++lane) {
                    if (tid < stride) {
                        const uint offset = lane * Q4_THREADS + tid;
                        partial[offset] += partial[offset + stride];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                for (uint lane = 0; lane < Q4_BATCH_TILE; ++lane) {
                    const uint batch = batch_base + lane;
                    if (batch < batch_count) {
                        out[batch * out_dim + row] =
                            partial[lane * Q4_THREADS] + linear_bias[row];
                    }
                }
            }
        }

        kernel void q8_0_matvec_f32(
            device const float* x [[buffer(0)]],
            device const char* q8_values [[buffer(1)]],
            device const float* scales [[buffer(2)]],
            device const float* linear_bias [[buffer(3)]],
            device float* out [[buffer(4)]],
            constant uint* dims [[buffer(5)]],
            uint row [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]]
        ) {
            const uint out_dim = dims[0];
            const uint in_dim = dims[1];
            const uint groups_per_row = dims[3];
            const uint group_size = dims[4];
            if (row >= out_dim) {
                return;
            }

            threadgroup float partial[Q4_THREADS];
            float acc = 0.0f;
            for (uint input = tid; input < in_dim; input += Q4_THREADS) {
                const uint group = input / group_size;
                const uint scale_offset = row * groups_per_row + group;
                const int q = int(q8_values[row * in_dim + input]);
                acc += x[input] * float(q) * scales[scale_offset];
            }
            partial[tid] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = Q4_THREADS / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    partial[tid] += partial[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                out[row] = partial[0] + linear_bias[row];
            }
        }
    "#
}

fn pack_q4(values: &[u32; 8]) -> u32 {
    values
        .iter()
        .enumerate()
        .fold(0_u32, |packed, (index, value)| {
            packed | ((value & 0xF) << (4 * index))
        })
}

struct KvCache {
    max_seq: usize,
    kv_heads: usize,
    head_dim: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl KvCache {
    fn new(max_seq: usize, kv_heads: usize, head_dim: usize) -> Self {
        let len = max_seq * kv_heads * head_dim;
        Self {
            max_seq,
            kv_heads,
            head_dim,
            keys: vec![0.0; len],
            values: vec![0.0; len],
        }
    }

    fn write(&mut self, position: usize, key: &[f32], value: &[f32]) {
        assert!(position < self.max_seq);
        assert_eq!(key.len(), self.kv_heads * self.head_dim);
        assert_eq!(value.len(), self.kv_heads * self.head_dim);
        let offset = position * self.kv_heads * self.head_dim;
        self.keys[offset..offset + key.len()].copy_from_slice(key);
        self.values[offset..offset + value.len()].copy_from_slice(value);
    }

    #[inline]
    fn key(&self, position: usize, kv_head: usize, dim: usize) -> f32 {
        self.keys[(position * self.kv_heads + kv_head) * self.head_dim + dim]
    }

    #[inline]
    fn value(&self, position: usize, kv_head: usize, dim: usize) -> f32 {
        self.values[(position * self.kv_heads + kv_head) * self.head_dim + dim]
    }
}

struct GgufFile {
    bytes: Vec<u8>,
    data_start: usize,
    metadata: BTreeMap<String, GgufValue>,
    tensors: BTreeMap<String, GgufTensorInfo>,
}

enum GgufValue {
    U32(u32),
    U64(u64),
    F32(f32),
    ArrayLen(usize),
    Other,
}

#[derive(Debug, Clone)]
struct GgufTensorInfo {
    ggml_type: i32,
    dims: Vec<usize>,
    offset: usize,
}

impl GgufFile {
    fn read(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = fs::read(path.as_ref()).map_err(|error| {
            config_error(format!(
                "failed to read GGUF file {}: {error}",
                path.as_ref().display()
            ))
        })?;
        let mut reader = GgufReader::new(&bytes);
        if reader.read_bytes(4)? != b"GGUF" {
            return Err(config_error("invalid GGUF file: missing GGUF magic"));
        }
        let version = reader.read_u32()?;
        if version != 3 {
            return Err(config_error(format!(
                "unsupported GGUF version {version}; expected version 3"
            )));
        }
        let tensor_count = reader.read_u64()? as usize;
        let metadata_count = reader.read_u64()? as usize;

        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = reader.read_string()?;
            let value_type = reader.read_i32()?;
            let value = reader.read_metadata_value(value_type)?;
            metadata.insert(key, value);
        }

        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::U32(value)) => *value as usize,
            Some(GgufValue::U64(value)) => *value as usize,
            _ => 32,
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(config_error(format!(
                "invalid GGUF general.alignment {alignment}"
            )));
        }

        let mut tensors = BTreeMap::new();
        for _ in 0..tensor_count {
            let name = reader.read_string()?;
            let dims_len = reader.read_u32()? as usize;
            let mut dims = Vec::with_capacity(dims_len);
            for _ in 0..dims_len {
                dims.push(reader.read_u64()? as usize);
            }
            let ggml_type = reader.read_i32()?;
            let offset = reader.read_u64()? as usize;
            tensors.insert(
                name,
                GgufTensorInfo {
                    ggml_type,
                    dims,
                    offset,
                },
            );
        }

        let data_start = align_up(reader.offset, alignment)?;
        Ok(Self {
            bytes,
            data_start,
            metadata,
            tensors,
        })
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn required_usize(&self, key: &str) -> Result<usize> {
        match self.metadata.get(key) {
            Some(GgufValue::U32(value)) => Ok(*value as usize),
            Some(GgufValue::U64(value)) => Ok(*value as usize),
            _ => Err(config_error(format!(
                "missing or invalid GGUF metadata `{key}`"
            ))),
        }
    }

    fn required_f32(&self, key: &str) -> Result<f32> {
        match self.metadata.get(key) {
            Some(GgufValue::F32(value)) => Ok(*value),
            _ => Err(config_error(format!(
                "missing or invalid GGUF metadata `{key}`"
            ))),
        }
    }

    fn metadata_array_len(&self, key: &str) -> Option<usize> {
        match self.metadata.get(key) {
            Some(GgufValue::ArrayLen(len)) => Some(*len),
            _ => None,
        }
    }

    fn load_f32(&self, name: impl AsRef<str>) -> Result<Vec<f32>> {
        let name = name.as_ref();
        let info = self.info(name)?;
        if info.ggml_type != GGML_TYPE_F32 {
            return Err(config_error(format!("{name} must have GGUF type F32")));
        }
        let bytes = self.tensor_bytes(name)?;
        if bytes.len() % 4 != 0 {
            return Err(config_error(format!(
                "{name} byte length is not divisible by 4"
            )));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    fn load_q4_0(&self, name: impl AsRef<str>) -> Result<QuantTensor> {
        let name = name.as_ref();
        let info = self.info(name)?;
        if info.ggml_type != GGML_TYPE_Q4_0 || info.dims.len() != 2 {
            return Err(config_error(format!(
                "{name} must be a rank-2 GGUF Q4_0 tensor"
            )));
        }
        let in_dim = info.dims[0];
        let out_dim = info.dims[1];
        if in_dim % 32 != 0 {
            return Err(config_error(format!(
                "{name} input dimension {in_dim} is not divisible by 32"
            )));
        }
        let blocks_per_row = in_dim / 32;
        let packed_per_row = in_dim / 8;
        let bytes = self.tensor_bytes(name)?;
        let expected_bytes = out_dim * blocks_per_row * GGML_Q4_0_BLOCK_BYTES;
        if bytes.len() != expected_bytes {
            return Err(config_error(format!(
                "{name} byte length {} does not match expected {expected_bytes}",
                bytes.len()
            )));
        }

        let mut weight = vec![0_u32; out_dim * packed_per_row];
        let mut scales = Vec::with_capacity(out_dim * blocks_per_row);
        let mut biases = Vec::with_capacity(out_dim * blocks_per_row);
        for row in 0..out_dim {
            for block in 0..blocks_per_row {
                let block_offset = (row * blocks_per_row + block) * GGML_Q4_0_BLOCK_BYTES;
                let d = f16::from_bits(u16::from_le_bytes([
                    bytes[block_offset],
                    bytes[block_offset + 1],
                ]))
                .to_f32();
                scales.push(d);
                biases.push(-8.0 * d);

                for local in 0..32 {
                    let packed = bytes[block_offset + 2 + local % 16];
                    let q = if local < 16 {
                        packed & 0x0F
                    } else {
                        packed >> 4
                    } as u32;
                    let input = block * 32 + local;
                    let word_index = row * packed_per_row + input / 8;
                    weight[word_index] |= q << (4 * (input % 8));
                }
            }
        }

        Ok(QuantTensor {
            name: name.to_string(),
            encoding: QuantEncoding::AffineQ4,
            out_dim,
            in_dim,
            packed_per_row,
            groups_per_row: blocks_per_row,
            group_size: 32,
            weight,
            q8_values: Vec::new(),
            scales,
            biases,
            metal: None,
        })
    }

    fn load_q8_0(&self, name: impl AsRef<str>) -> Result<QuantTensor> {
        let name = name.as_ref();
        let info = self.info(name)?;
        if info.ggml_type != GGML_TYPE_Q8_0 || info.dims.len() != 2 {
            return Err(config_error(format!(
                "{name} must be a rank-2 GGUF Q8_0 tensor"
            )));
        }
        let in_dim = info.dims[0];
        let out_dim = info.dims[1];
        if in_dim % 32 != 0 {
            return Err(config_error(format!(
                "{name} input dimension {in_dim} is not divisible by 32"
            )));
        }
        let blocks_per_row = in_dim / 32;
        let bytes = self.tensor_bytes(name)?;
        let expected_bytes = out_dim * blocks_per_row * GGML_Q8_0_BLOCK_BYTES;
        if bytes.len() != expected_bytes {
            return Err(config_error(format!(
                "{name} byte length {} does not match expected {expected_bytes}",
                bytes.len()
            )));
        }

        let mut q8_values = Vec::with_capacity(out_dim * in_dim);
        let mut scales = Vec::with_capacity(out_dim * blocks_per_row);
        for row in 0..out_dim {
            for block in 0..blocks_per_row {
                let block_offset = (row * blocks_per_row + block) * GGML_Q8_0_BLOCK_BYTES;
                let d = f16::from_bits(u16::from_le_bytes([
                    bytes[block_offset],
                    bytes[block_offset + 1],
                ]))
                .to_f32();
                scales.push(d);
                for local in 0..32 {
                    q8_values.push(i8::from_ne_bytes([bytes[block_offset + 2 + local]]));
                }
            }
        }

        Ok(QuantTensor {
            name: name.to_string(),
            encoding: QuantEncoding::GgufQ8_0,
            out_dim,
            in_dim,
            packed_per_row: 0,
            groups_per_row: blocks_per_row,
            group_size: 32,
            weight: Vec::new(),
            q8_values,
            scales,
            biases: Vec::new(),
            metal: None,
        })
    }

    fn info(&self, name: &str) -> Result<&GgufTensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| config_error(format!("missing GGUF tensor {name}")))
    }

    fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self.info(name)?;
        let start = self.data_start + info.offset;
        let len = gguf_tensor_nbytes(info)?;
        let end = start + len;
        self.bytes
            .get(start..end)
            .ok_or_else(|| config_error(format!("GGUF tensor {name} data is out of bounds")))
    }
}

struct GgufReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> GgufReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let start = self.offset;
        let end = start + len;
        let bytes = self
            .bytes
            .get(start..end)
            .ok_or_else(|| config_error("truncated GGUF file"))?;
        self.offset = end;
        Ok(bytes)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let bytes = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|error| config_error(format!("invalid UTF-8 string in GGUF: {error}")))
    }

    fn read_metadata_value(&mut self, value_type: i32) -> Result<GgufValue> {
        match value_type {
            GGUF_TYPE_UINT32 => Ok(GgufValue::U32(self.read_u32()?)),
            GGUF_TYPE_UINT64 => Ok(GgufValue::U64(self.read_u64()?)),
            GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(self.read_f32()?)),
            GGUF_TYPE_STRING => {
                let _ = self.read_string()?;
                Ok(GgufValue::Other)
            }
            GGUF_TYPE_BOOL => {
                let _ = self.read_bytes(1)?;
                Ok(GgufValue::Other)
            }
            GGUF_TYPE_ARRAY => {
                let element_type = self.read_i32()?;
                let len = self.read_u64()? as usize;
                for _ in 0..len {
                    self.skip_metadata_scalar(element_type)?;
                }
                Ok(GgufValue::ArrayLen(len))
            }
            other => {
                self.skip_metadata_scalar(other)?;
                Ok(GgufValue::Other)
            }
        }
    }

    fn skip_metadata_scalar(&mut self, value_type: i32) -> Result<()> {
        match value_type {
            GGUF_TYPE_UINT8 | GGUF_TYPE_INT8 | GGUF_TYPE_BOOL => {
                let _ = self.read_bytes(1)?;
            }
            GGUF_TYPE_UINT16 | GGUF_TYPE_INT16 => {
                let _ = self.read_bytes(2)?;
            }
            GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => {
                let _ = self.read_bytes(4)?;
            }
            GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => {
                let _ = self.read_bytes(8)?;
            }
            GGUF_TYPE_STRING => {
                let _ = self.read_string()?;
            }
            other => {
                return Err(config_error(format!(
                    "unsupported GGUF metadata value type {other}"
                )));
            }
        }
        Ok(())
    }
}

const GGUF_TYPE_UINT8: i32 = 0;
const GGUF_TYPE_INT8: i32 = 1;
const GGUF_TYPE_UINT16: i32 = 2;
const GGUF_TYPE_INT16: i32 = 3;
const GGUF_TYPE_UINT32: i32 = 4;
const GGUF_TYPE_INT32: i32 = 5;
const GGUF_TYPE_FLOAT32: i32 = 6;
const GGUF_TYPE_BOOL: i32 = 7;
const GGUF_TYPE_STRING: i32 = 8;
const GGUF_TYPE_ARRAY: i32 = 9;
const GGUF_TYPE_UINT64: i32 = 10;
const GGUF_TYPE_INT64: i32 = 11;
const GGUF_TYPE_FLOAT64: i32 = 12;

const GGML_TYPE_F32: i32 = 0;
const GGML_TYPE_Q4_0: i32 = 2;
const GGML_TYPE_Q8_0: i32 = 8;
const GGML_Q4_0_BLOCK_SIZE: usize = 32;
const GGML_Q4_0_BLOCK_BYTES: usize = 18;
const GGML_Q8_0_BLOCK_SIZE: usize = 32;
const GGML_Q8_0_BLOCK_BYTES: usize = 34;

fn gguf_tensor_nbytes(info: &GgufTensorInfo) -> Result<usize> {
    let elements = info
        .dims
        .iter()
        .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| config_error("GGUF tensor element count overflow"))?;
    match info.ggml_type {
        GGML_TYPE_F32 => elements
            .checked_mul(4)
            .ok_or_else(|| config_error("GGUF F32 tensor byte size overflow")),
        GGML_TYPE_Q4_0 => quantized_tensor_nbytes(
            elements,
            GGML_Q4_0_BLOCK_SIZE,
            GGML_Q4_0_BLOCK_BYTES,
            "Q4_0",
        ),
        GGML_TYPE_Q8_0 => quantized_tensor_nbytes(
            elements,
            GGML_Q8_0_BLOCK_SIZE,
            GGML_Q8_0_BLOCK_BYTES,
            "Q8_0",
        ),
        other => Err(config_error(format!(
            "unsupported GGUF tensor type {other}"
        ))),
    }
}

fn quantized_tensor_nbytes(
    elements: usize,
    block_size: usize,
    block_bytes: usize,
    name: &str,
) -> Result<usize> {
    if elements % block_size != 0 {
        return Err(config_error(format!(
            "GGUF {name} tensor element count {elements} is not divisible by {block_size}"
        )));
    }
    (elements / block_size)
        .checked_mul(block_bytes)
        .ok_or_else(|| config_error(format!("GGUF {name} tensor byte size overflow")))
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    let add = alignment - 1;
    value
        .checked_add(add)
        .map(|value| value & !add)
        .ok_or_else(|| config_error("GGUF alignment overflow"))
}

struct SafeTensorFile {
    bytes: Vec<u8>,
    data_start: usize,
    tensors: BTreeMap<String, TensorInfo>,
}

#[derive(Debug, Clone, Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

impl SafeTensorFile {
    fn read(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = fs::read(path.as_ref()).map_err(|error| {
            config_error(format!(
                "failed to read safetensors file {}: {error}",
                path.as_ref().display()
            ))
        })?;
        if bytes.len() < 8 {
            return Err(config_error(
                "invalid safetensors file: shorter than header",
            ));
        }
        let mut len_bytes = [0_u8; 8];
        len_bytes.copy_from_slice(&bytes[..8]);
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        let data_start = 8 + header_len;
        if bytes.len() < data_start {
            return Err(config_error("invalid safetensors file: truncated header"));
        }
        let value: serde_json::Value =
            serde_json::from_slice(&bytes[8..data_start]).map_err(|error| {
                config_error(format!("failed to parse safetensors header: {error}"))
            })?;
        let object = value
            .as_object()
            .ok_or_else(|| config_error("safetensors header is not a JSON object"))?;
        let mut tensors = BTreeMap::new();
        for (name, info) in object {
            if name == "__metadata__" {
                continue;
            }
            let info: TensorInfo = serde_json::from_value(info.clone()).map_err(|error| {
                config_error(format!(
                    "invalid safetensors tensor metadata for {name}: {error}"
                ))
            })?;
            tensors.insert(name.clone(), info);
        }
        Ok(Self {
            bytes,
            data_start,
            tensors,
        })
    }

    fn load_quant(&self, prefix: impl AsRef<str>) -> Result<QuantTensor> {
        let prefix = prefix.as_ref();
        let weight_name = format!("{prefix}.weight");
        let scales_name = format!("{prefix}.scales");
        let biases_name = format!("{prefix}.biases");

        let weight_info = self.info(&weight_name)?;
        if weight_info.dtype != "U32" || weight_info.shape.len() != 2 {
            return Err(config_error(format!(
                "{weight_name} must be a rank-2 U32 tensor"
            )));
        }
        let out_dim = weight_info.shape[0];
        let packed_per_row = weight_info.shape[1];
        let in_dim = packed_per_row * 8;

        let scale_info = self.info(&scales_name)?;
        if scale_info.dtype != "BF16"
            || scale_info.shape.len() != 2
            || scale_info.shape[0] != out_dim
        {
            return Err(config_error(format!(
                "{scales_name} must be BF16 [out_dim, groups]"
            )));
        }
        let groups_per_row = scale_info.shape[1];
        let expected_groups = in_dim.div_ceil(64);
        if groups_per_row != expected_groups {
            return Err(config_error(format!(
                "{scales_name} groups {} does not match expected {}",
                groups_per_row, expected_groups
            )));
        }

        let bias_info = self.info(&biases_name)?;
        if bias_info.dtype != "BF16" || bias_info.shape != scale_info.shape {
            return Err(config_error(format!(
                "{biases_name} must match scale tensor shape"
            )));
        }

        Ok(QuantTensor {
            name: prefix.to_string(),
            encoding: QuantEncoding::AffineQ4,
            out_dim,
            in_dim,
            packed_per_row,
            groups_per_row,
            group_size: 64,
            weight: self.load_u32(weight_name)?,
            q8_values: Vec::new(),
            scales: self.load_bf16(scales_name)?,
            biases: self.load_bf16(biases_name)?,
            metal: None,
        })
    }

    fn load_u32(&self, name: impl AsRef<str>) -> Result<Vec<u32>> {
        let name = name.as_ref();
        let info = self.info(name)?;
        if info.dtype != "U32" {
            return Err(config_error(format!("{name} must have dtype U32")));
        }
        let bytes = self.tensor_bytes(name)?;
        if bytes.len() % 4 != 0 {
            return Err(config_error(format!(
                "{name} byte length is not divisible by 4"
            )));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    fn load_bf16(&self, name: impl AsRef<str>) -> Result<Vec<f32>> {
        let name = name.as_ref();
        let info = self.info(name)?;
        if info.dtype != "BF16" {
            return Err(config_error(format!("{name} must have dtype BF16")));
        }
        let bytes = self.tensor_bytes(name)?;
        if bytes.len() % 2 != 0 {
            return Err(config_error(format!(
                "{name} byte length is not divisible by 2"
            )));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect())
    }

    fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| config_error(format!("missing tensor {name}")))
    }

    fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self.info(name)?;
        let start = self.data_start + info.data_offsets[0];
        let end = self.data_start + info.data_offsets[1];
        self.bytes
            .get(start..end)
            .ok_or_else(|| config_error(format!("tensor {name} data offsets are out of bounds")))
    }
}

fn attention_decode(
    q: &[f32],
    cache: &KvCache,
    position: usize,
    config: &QwenRuntimeConfig,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; config.num_attention_heads * config.head_dim];
    let kv_repeat = config.num_attention_heads / config.num_key_value_heads;
    let scale = (config.head_dim as f32).sqrt().recip();
    let mut scores = vec![0.0_f32; position + 1];

    for head in 0..config.num_attention_heads {
        let kv_head = head / kv_repeat;
        let q_offset = head * config.head_dim;
        for past in 0..=position {
            let mut dot = 0.0_f32;
            for dim in 0..config.head_dim {
                dot += q[q_offset + dim] * cache.key(past, kv_head, dim);
            }
            scores[past] = dot * scale;
        }

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            sum += *score;
        }
        if sum > 0.0 {
            for score in &mut scores {
                *score /= sum;
            }
        }

        for dim in 0..config.head_dim {
            let mut value = 0.0_f32;
            for past in 0..=position {
                value += scores[past] * cache.value(past, kv_head, dim);
            }
            output[q_offset + dim] = value;
        }
    }

    output
}

fn apply_rope(values: &mut [f32], heads: usize, head_dim: usize, position: usize, theta: f32) {
    let half = head_dim / 2;
    for head in 0..heads {
        let offset = head * head_dim;
        for i in 0..half {
            let freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
            let angle = position as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            let a = values[offset + i];
            let b = values[offset + i + half];
            values[offset + i] = a * cos - b * sin;
            values[offset + i + half] = b * cos + a * sin;
        }
    }
}

fn apply_rope_batched(
    values: &mut [f32],
    batch: usize,
    heads: usize,
    head_dim: usize,
    start_position: usize,
    theta: f32,
) {
    let row_len = heads * head_dim;
    debug_assert_eq!(values.len(), batch * row_len);
    for row in 0..batch {
        let position = start_position + row;
        let offset = row * row_len;
        apply_rope(
            &mut values[offset..offset + row_len],
            heads,
            head_dim,
            position,
            theta,
        );
    }
}

fn rmsnorm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let sumsq = input.iter().map(|value| value * value).sum::<f32>();
    let inv_rms = (sumsq / input.len() as f32 + eps).sqrt().recip();
    input
        .iter()
        .zip(weight)
        .map(|(value, weight)| value * weight * inv_rms)
        .collect()
}

fn rmsnorm_batch(input: &[f32], weight: &[f32], eps: f32, rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(input.len(), rows * cols);
    debug_assert_eq!(weight.len(), cols);
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(cols) {
        output.extend(rmsnorm(row, weight, eps));
    }
    output
}

fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(gate, up)| {
            let silu = gate / (1.0 + (-gate).exp());
            silu * up
        })
        .collect()
}

fn add(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    lhs.iter().zip(rhs).map(|(lhs, rhs)| lhs + rhs).collect()
}

fn add_in_place(lhs: &mut [f32], rhs: &[f32]) {
    for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
        *lhs += rhs;
    }
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn required_usize(value: &serde_json::Value, key: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .map(|value| value as usize)
        .ok_or_else(|| config_error(format!("missing or invalid `{key}` in config.json")))
}

fn ensure_len(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(config_error(format!(
            "{name} length {} does not match expected {expected}",
            values.len()
        )))
    }
}

fn ensure_quant_shape(
    name: &str,
    tensor: &QuantTensor,
    expected_out: usize,
    expected_in: usize,
) -> Result<()> {
    if tensor.out_dim == expected_out && tensor.in_dim == expected_in {
        Ok(())
    } else {
        Err(config_error(format!(
            "{name} shape [out={}, in={}] does not match expected [{expected_out}, {expected_in}]",
            tensor.out_dim, tensor.in_dim
        )))
    }
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn config_error(message: impl Into<String>) -> TinyAgentError {
    TinyAgentError::Configuration(message.into())
}
