use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use serde::{Deserialize, Serialize};
use tinyagent_core::{Result, TinyAgentError};
use tokenizers::Tokenizer;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

const QWEN_IM_END: u32 = 151_645;
const QWEN_END_OF_TEXT: u32 = 151_643;

#[derive(Debug, Clone)]
pub struct QwenMlxRunConfig {
    pub hf_dir: PathBuf,
    pub prompt: String,
    pub max_tokens: usize,
    pub max_prompt_tokens: usize,
    pub projection_backend: QwenProjectionBackend,
}

#[derive(Debug, Clone, Serialize)]
pub struct QwenMlxRunResult {
    pub model_dir: PathBuf,
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
        out_dim: 3,
        in_dim: 8,
        packed_per_row: 1,
        groups_per_row: 1,
        weight: vec![
            pack_q4(&[0, 1, 2, 3, 4, 5, 6, 7]),
            pack_q4(&[8, 7, 6, 5, 4, 3, 2, 1]),
            pack_q4(&[15, 0, 15, 0, 10, 5, 10, 5]),
        ],
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
    let mut model = QwenMlxModel::load(&config.hf_dir, config.projection_backend)?;
    let tokenizer = Tokenizer::from_file(config.hf_dir.join("tokenizer.json"))
        .map_err(|error| config_error(format!("failed to load tokenizer.json: {error}")))?;
    let load_ms = elapsed_ms(load_started);

    let formatted_prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        config.prompt
    );
    let encoding = tokenizer
        .encode(formatted_prompt, true)
        .map_err(|error| config_error(format!("failed to encode prompt: {error}")))?;
    let prompt_tokens_total = encoding.len();
    let mut input_ids = encoding.get_ids().to_vec();
    if input_ids.len() > config.max_prompt_tokens {
        input_ids = input_ids[input_ids.len() - config.max_prompt_tokens..].to_vec();
    }
    if input_ids.is_empty() {
        return Err(config_error("prompt encoded to zero tokens"));
    }

    model.allocate_kv_cache(input_ids.len() + config.max_tokens + 1);

    let eval_started = Instant::now();
    let mut token_eval_ms = Vec::with_capacity(input_ids.len() + config.max_tokens);
    let mut logits = Vec::new();
    let mut position = 0_usize;
    for token_id in &input_ids {
        let started = Instant::now();
        logits = model.forward_token(*token_id, position)?;
        token_eval_ms.push(elapsed_ms(started));
        position += 1;
    }

    let mut generated_token_ids = Vec::with_capacity(config.max_tokens);
    for _ in 0..config.max_tokens {
        let next_id = argmax(&logits) as u32;
        if next_id == QWEN_IM_END || next_id == QWEN_END_OF_TEXT {
            break;
        }
        generated_token_ids.push(next_id);

        let started = Instant::now();
        logits = model.forward_token(next_id, position)?;
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

    let eval_token_count = token_eval_ms.len();
    let avg_token_eval_ms = if eval_token_count == 0 {
        0.0
    } else {
        token_eval_ms.iter().sum::<f64>() / eval_token_count as f64
    };
    let min_token_eval_ms = token_eval_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let max_token_eval_ms = token_eval_ms.iter().copied().fold(0.0_f64, f64::max);
    let q4_projection_groups_per_eval_token = match config.projection_backend {
        QwenProjectionBackend::Cpu => model.config.num_hidden_layers * 7 + 1,
        QwenProjectionBackend::Metal => model.config.num_hidden_layers * 4 + 1,
    };

    Ok(QwenMlxRunResult {
        model_dir: config.hf_dir,
        prompt: config.prompt,
        prompt_tokens_total,
        prompt_tokens_used: input_ids.len(),
        projection_backend: config.projection_backend,
        generated_token_ids,
        generated_text,
        load_ms,
        eval_ms,
        total_ms: elapsed_ms(total_started),
        token_eval_ms,
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
        q4_projection_groups_total: q4_projection_groups_per_eval_token * eval_token_count,
        note: match config.projection_backend {
            QwenProjectionBackend::Cpu => "CPU reference end-to-end over real MLX 4-bit Qwen weights; this validates model graph and q4 affine loading.".to_string(),
            QwenProjectionBackend::Metal => "Hybrid TinyEngine end-to-end over real MLX 4-bit Qwen weights: q4 affine projections run on the reusable Metal kernel with GPU-resident q4 tensors; scalar ops and attention are still CPU.".to_string(),
        },
    })
}

struct QwenMlxModel {
    config: QwenRuntimeConfig,
    projection_backend: QwenProjectionBackend,
    metal_runtime: Option<Q4MetalRuntime>,
    embed_tokens: QuantTensor,
    layers: Vec<QwenLayer>,
    norm_weight: Vec<f32>,
    kv_cache: Vec<KvCache>,
}

impl QwenMlxModel {
    fn load(hf_dir: &Path, projection_backend: QwenProjectionBackend) -> Result<Self> {
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
            layers,
            norm_weight,
            kv_cache: Vec::new(),
        };
        model.upload_quant_tensors_to_metal()?;
        Ok(model)
    }

    fn upload_quant_tensors_to_metal(&mut self) -> Result<()> {
        let Some(runtime) = self.metal_runtime.as_ref() else {
            return Ok(());
        };
        self.embed_tokens.upload_metal(runtime)?;
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
    }

    fn forward_token(&mut self, token_id: u32, position: usize) -> Result<Vec<f32>> {
        let mut hidden = self.embed_tokens.dequantize_row(token_id as usize)?;
        ensure_len("embedding", &hidden, self.config.hidden_size)?;

        for layer_index in 0..self.layers.len() {
            hidden = self.forward_layer(layer_index, &hidden, position)?;
        }

        let normalized = rmsnorm(&hidden, &self.norm_weight, self.config.rms_norm_eps);
        project_quant(
            self.projection_backend,
            self.metal_runtime.as_ref(),
            &self.embed_tokens,
            &normalized,
            None,
        )
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
}

struct QuantTensor {
    name: String,
    out_dim: usize,
    in_dim: usize,
    packed_per_row: usize,
    groups_per_row: usize,
    weight: Vec<u32>,
    scales: Vec<f32>,
    biases: Vec<f32>,
    metal: Option<QuantTensorMetal>,
}

struct QuantTensorMetal {
    weight: Buffer,
    scales: Buffer,
    biases: Buffer,
    zero_linear_bias: Buffer,
    dims: Buffer,
}

struct Q4MetalRuntime {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
}

struct ProjectionRequest<'a> {
    tensor: &'a QuantTensor,
    linear_bias: Option<&'a [f32]>,
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
        let function = library
            .get_function("q4_affine_matvec_f32", None)
            .map_err(|error| {
                TinyAgentError::Backend(format!("failed to load q4 matvec kernel: {error}"))
            })?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| {
                TinyAgentError::Backend(format!("failed to create q4 matvec pipeline: {error}"))
            })?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pipeline,
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
            request
                .tensor
                .validate_matvec_input(input, request.linear_bias)?;
        }

        let buffer_x = self.device.new_buffer_with_data(
            input.as_ptr().cast(),
            std::mem::size_of_val(input) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let mut output_buffers = Vec::with_capacity(requests.len());
        let mut linear_bias_buffers = Vec::new();
        for request in requests {
            if let Some(bias) = request.linear_bias {
                linear_bias_buffers.push(self.device.new_buffer_with_data(
                    bias.as_ptr().cast(),
                    std::mem::size_of_val(bias) as u64,
                    MTLResourceOptions::StorageModeShared,
                ));
            }
            output_buffers.push(self.device.new_buffer(
                (request.tensor.out_dim * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        }

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);

        let mut next_bias_buffer = 0_usize;
        for (index, request) in requests.iter().enumerate() {
            let metal = request.tensor.metal.as_ref().ok_or_else(|| {
                config_error(format!(
                    "{} has not been uploaded to Metal",
                    request.tensor.name
                ))
            })?;
            let linear_bias_buffer = match request.linear_bias {
                Some(_) => {
                    let buffer = &linear_bias_buffers[next_bias_buffer];
                    next_bias_buffer += 1;
                    buffer
                }
                None => &metal.zero_linear_bias,
            };

            encoder.set_buffer(0, Some(&buffer_x), 0);
            encoder.set_buffer(1, Some(&metal.weight), 0);
            encoder.set_buffer(2, Some(&metal.scales), 0);
            encoder.set_buffer(3, Some(&metal.biases), 0);
            encoder.set_buffer(4, Some(linear_bias_buffer), 0);
            encoder.set_buffer(5, Some(&output_buffers[index]), 0);
            encoder.set_buffer(6, Some(&metal.dims), 0);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: request.tensor.out_dim as u64,
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

        let mut outputs = Vec::with_capacity(output_buffers.len());
        for (buffer, request) in output_buffers.iter().zip(requests) {
            let ptr = buffer.contents().cast::<f32>();
            outputs
                .push(unsafe { std::slice::from_raw_parts(ptr, request.tensor.out_dim).to_vec() });
        }
        Ok(outputs)
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
        ];
        let zero_bias = vec![0.0_f32; self.out_dim];
        self.metal = Some(QuantTensorMetal {
            weight: runtime.device.new_buffer_with_data(
                self.weight.as_ptr().cast(),
                std::mem::size_of_val(self.weight.as_slice()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            scales: runtime.device.new_buffer_with_data(
                self.scales.as_ptr().cast(),
                std::mem::size_of_val(self.scales.as_slice()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            biases: runtime.device.new_buffer_with_data(
                self.biases.as_ptr().cast(),
                std::mem::size_of_val(self.biases.as_slice()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            zero_linear_bias: runtime.device.new_buffer_with_data(
                zero_bias.as_ptr().cast(),
                std::mem::size_of_val(zero_bias.as_slice()) as u64,
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
            for input_index in 0..self.in_dim {
                acc += input[input_index] * self.dequant_value(row, input_index);
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
        let group = input / 64;
        let group_offset = row * self.groups_per_row + group;
        q * self.scales[group_offset] + self.biases[group_offset]
    }
}

fn q4_affine_matvec_source() -> &'static str {
    r#"
        #include <metal_stdlib>
        using namespace metal;

        #define Q4_THREADS 128

        kernel void q4_affine_matvec_f32(
            device const float* x [[buffer(0)]],
            device const uint* weight [[buffer(1)]],
            device const float* scales [[buffer(2)]],
            device const float* biases [[buffer(3)]],
            device const float* linear_bias [[buffer(4)]],
            device float* out [[buffer(5)]],
            constant uint* dims [[buffer(6)]],
            uint row [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]]
        ) {
            const uint out_dim = dims[0];
            const uint in_dim = dims[1];
            const uint packed_per_row = dims[2];
            const uint groups_per_row = dims[3];
            if (row >= out_dim) {
                return;
            }

            threadgroup float partial[Q4_THREADS];
            float acc = 0.0f;
            for (uint input = tid; input < in_dim; input += Q4_THREADS) {
                const uint packed = weight[row * packed_per_row + input / 8];
                const uint q = (packed >> (4 * (input & 7))) & 0xFu;
                const uint group = input / 64;
                const uint group_offset = row * groups_per_row + group;
                const float w = float(q) * scales[group_offset] + biases[group_offset];
                acc += x[input] * w;
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
            out_dim,
            in_dim,
            packed_per_row,
            groups_per_row,
            weight: self.load_u32(weight_name)?,
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

fn rmsnorm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let sumsq = input.iter().map(|value| value * value).sum::<f32>();
    let inv_rms = (sumsq / input.len() as f32 + eps).sqrt().recip();
    input
        .iter()
        .zip(weight)
        .map(|(value, weight)| value * weight * inv_rms)
        .collect()
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
