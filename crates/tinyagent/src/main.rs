use std::{
    convert::Infallible,
    fs,
    io::{self, Write},
    path::PathBuf,
    process::Command,
    sync::Arc,
    time::Instant,
};

use anyhow::{bail, Context};
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tinyagent_backend_llama::{LlamaServerBackend, LlamaServerConfig};
use tinyagent_backend_metal::{
    format_qwen_chat_prompt, run_add_kernel_probe, run_attention_probe, run_f16_matmul_probe,
    run_f16_matvec_probe, run_greedy_sampler_probe, run_hot_f16_matmul_bench,
    run_hot_f16_matvec_bench, run_kv_cache_append_probe, run_q4_affine_matvec_probe,
    run_qwen_15b_hot_bench, run_qwen_15b_smoke_bench, run_qwen_gguf_end2end, run_qwen_mlx_end2end,
    run_rmsnorm_probe, run_rope_probe, run_softmax_probe, run_swiglu_probe,
    HotKernelBenchmarkResult, MetalBackend, MetalBackendConfig, MetalDeviceInfo, QwenGgufRunConfig,
    QwenMlxRunConfig, QwenMlxRunResult, QwenProjectionBackend,
};
use tinyagent_core::{
    ChatMessage, GenerateRequest, HardwareProfile, InferenceBackend, MessageRole, ModelInfo,
    StubBackend, TinyAgentError,
};
use tma_format::{
    inspect_package, write_metadata, ModelArchitecture, QwenConfig, SourceFormat,
    TmaPackageMetadata,
};
use tokio::net::TcpListener;

#[derive(Debug, Parser)]
#[command(name = "tinyagent")]
#[command(about = "TinyEngine first: custom local Metal inference for 8GB-friendly Qwen models")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Engine(EngineArgs),
    Convert(ConvertArgs),
    Models(SharedArgs),
    Chat(ChatArgs),
    Serve(ServeArgs),
    Bench(SharedArgs),
}

#[derive(Debug, Args)]
struct SharedArgs {
    #[arg(long, default_value = "mac-8gb")]
    profile: String,
    #[arg(long, default_value = "qwen2.5-coder-1.5b")]
    model: String,
    #[arg(long, value_enum, default_value_t = BackendKind::Metal)]
    backend: BackendKind,
    #[arg(long, value_name = "DIR")]
    package: Option<PathBuf>,
    #[arg(long, value_name = "PATH")]
    gguf: Option<PathBuf>,
    #[arg(long, default_value = "llama-server", value_name = "PATH")]
    llama_server_bin: PathBuf,
    #[arg(long, default_value = "127.0.0.1")]
    llama_host: String,
    #[arg(long, default_value_t = 8788)]
    llama_port: u16,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendKind {
    Metal,
    Llama,
    Stub,
}

#[derive(Debug, Args)]
struct EngineArgs {
    #[command(subcommand)]
    command: EngineCommand,
}

#[derive(Debug, Subcommand)]
enum EngineCommand {
    Probe,
    Bench(EngineBenchArgs),
    PhaseBench(EnginePhaseBenchArgs),
    QwenRun(EngineQwenRunArgs),
    Compare(EngineCompareArgs),
    Inspect(EngineInspectArgs),
}

#[derive(Debug, Args)]
struct EngineBenchArgs {
    #[arg(long)]
    hot: bool,
    #[arg(long, default_value_t = 25)]
    iterations: u32,
}

#[derive(Debug, Args)]
struct EngineInspectArgs {
    #[arg(long, value_name = "DIR")]
    package: PathBuf,
}

#[derive(Debug, Args)]
struct EnginePhaseBenchArgs {
    #[arg(long, value_name = "DIR")]
    hf_dir: PathBuf,
    #[arg(long, default_value = "1,32,128,512")]
    prefill_tokens: String,
    #[arg(long, default_value_t = 10)]
    iterations: u32,
    #[arg(long, value_name = "FILE")]
    out: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct EngineQwenRunArgs {
    #[arg(long, value_name = "DIR")]
    hf_dir: PathBuf,
    #[arg(long, default_value = "ciao")]
    prompt: String,
    #[arg(long, default_value_t = 1)]
    max_tokens: usize,
    #[arg(long, default_value_t = 1)]
    max_prompt_tokens: usize,
    #[arg(long, value_enum, default_value_t = QwenProjectionBackendArg::Metal)]
    projection_backend: QwenProjectionBackendArg,
}

#[derive(Debug, Args)]
struct EngineCompareArgs {
    #[arg(long, value_name = "DIR")]
    hf_dir: PathBuf,
    #[arg(long, value_name = "FILE")]
    gguf: PathBuf,
    #[arg(
        long,
        default_value = "/Users/lucamocerino/projects/tools/llama.cpp-b9219/current/llama-completion",
        value_name = "PATH"
    )]
    llama_bin: PathBuf,
    #[arg(long, default_value = "Rispondi in italiano con tre parole: cosa sei?")]
    prompt: String,
    #[arg(long, default_value_t = 4)]
    max_tokens: usize,
    #[arg(long, default_value_t = 512)]
    max_prompt_tokens: usize,
    #[arg(long, value_enum, default_value_t = QwenProjectionBackendArg::Metal)]
    projection_backend: QwenProjectionBackendArg,
    #[arg(long, value_enum, default_value_t = TinyEngineCompareSourceArg::Mlx)]
    tinyengine_source: TinyEngineCompareSourceArg,
    #[arg(long, default_value_t = 512)]
    ctx_size: u32,
    #[arg(long, default_value_t = 999)]
    gpu_layers: i32,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long, default_value_t = 1)]
    runs: usize,
    #[arg(long)]
    llama_warmup: bool,
    #[arg(long, value_name = "FILE")]
    out: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum TinyEngineCompareSourceArg {
    Mlx,
    Gguf,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum QwenProjectionBackendArg {
    Cpu,
    Metal,
}

#[derive(Debug, Args)]
struct ConvertArgs {
    #[arg(value_enum)]
    source: ConvertSource,
    #[arg(value_name = "INPUT")]
    input: PathBuf,
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    #[arg(long, default_value = "qwen2.5-coder-1.5b")]
    model_id: String,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ConvertSource {
    Hf,
    Gguf,
}

#[derive(Debug, Args)]
struct ChatArgs {
    #[command(flatten)]
    shared: SharedArgs,
    #[arg(trailing_var_arg = true)]
    prompt: Vec<String>,
}

#[derive(Debug, Args)]
struct ServeArgs {
    #[command(flatten)]
    shared: SharedArgs,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8787)]
    port: u16,
}

#[derive(Clone)]
struct AppState {
    backend: Arc<dyn InferenceBackend>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Engine(args) => run_engine(args).await,
        Commands::Convert(args) => run_convert(args).await,
        Commands::Models(args) => run_models(args).await,
        Commands::Chat(args) => run_chat(args).await,
        Commands::Serve(args) => run_serve(args).await,
        Commands::Bench(args) => run_bench(args).await,
    }
}

async fn run_engine(args: EngineArgs) -> anyhow::Result<()> {
    match args.command {
        EngineCommand::Probe => {
            let device = MetalDeviceInfo::system_default()?;
            let add_probe = run_add_kernel_probe()?;
            let matmul_probe = run_f16_matmul_probe()?;
            let matvec_probe = run_f16_matvec_probe()?;
            let q4_matvec_probe = run_q4_affine_matvec_probe()?;
            let rmsnorm_probe = run_rmsnorm_probe()?;
            let rope_probe = run_rope_probe()?;
            let swiglu_probe = run_swiglu_probe()?;
            let softmax_probe = run_softmax_probe()?;
            let attention_probe = run_attention_probe()?;
            let greedy_sampler_probe = run_greedy_sampler_probe()?;
            let kv_cache_append_probe = run_kv_cache_append_probe()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "metal": {
                        "available": true,
                        "device": device.name,
                        "custom_kernel_probe": {
                            "name": "vector_add",
                            "output": add_probe,
                            "expected": [11.0, 22.0, 33.0, 44.0]
                        },
                        "f16_matmul_probe": {
                            "kernel": "matmul_f16_f32_tiled_16x16",
                            "shape": {
                                "m": matmul_probe.m,
                                "n": matmul_probe.n,
                                "k": matmul_probe.k
                            },
                            "metal_output": matmul_probe.metal_output,
                            "cpu_output": matmul_probe.cpu_output,
                            "max_abs_error": matmul_probe.max_abs_error
                        },
                        "f16_matvec_probe": {
                            "kernel": "matvec_f16_f32_col_tile_16_reduce_16",
                            "shape": {
                                "n": matvec_probe.n,
                                "k": matvec_probe.k
                            },
                            "metal_output": matvec_probe.metal_output,
                            "cpu_output": matvec_probe.cpu_output,
                            "max_abs_error": matvec_probe.max_abs_error
                        },
                        "q4_affine_matvec_probe": {
                            "kernel": "q4_affine_matvec_f32_reduce_128",
                            "shape": {
                                "out_dim": q4_matvec_probe.out_dim,
                                "in_dim": q4_matvec_probe.in_dim
                            },
                            "metal_output": q4_matvec_probe.metal_output,
                            "cpu_output": q4_matvec_probe.cpu_output,
                            "max_abs_error": q4_matvec_probe.max_abs_error
                        },
                        "rmsnorm_probe": {
                            "kernel": "rmsnorm_f16_f32_threadgroup_reduce_256",
                            "shape": {
                                "rows": rmsnorm_probe.rows,
                                "cols": rmsnorm_probe.cols
                            },
                            "metal_output": rmsnorm_probe.metal_output,
                            "cpu_output": rmsnorm_probe.cpu_output,
                            "max_abs_error": rmsnorm_probe.max_abs_error
                        },
                        "rope_probe": {
                            "kernel": "rope_qwen_f16_f32_half_split",
                            "shape": {
                                "rows": rope_probe.rows,
                                "dims": rope_probe.dims
                            },
                            "metal_output": rope_probe.metal_output,
                            "cpu_output": rope_probe.cpu_output,
                            "max_abs_error": rope_probe.max_abs_error
                        },
                        "swiglu_probe": {
                            "kernel": "swiglu_f32",
                            "shape": {
                                "len": swiglu_probe.len
                            },
                            "metal_output": swiglu_probe.metal_output,
                            "cpu_output": swiglu_probe.cpu_output,
                            "max_abs_error": swiglu_probe.max_abs_error
                        },
                        "softmax_probe": {
                            "kernel": "softmax_f32_threadgroup_max_sum_256",
                            "shape": {
                                "rows": softmax_probe.rows,
                                "cols": softmax_probe.cols
                            },
                            "metal_output": softmax_probe.metal_output,
                            "cpu_output": softmax_probe.cpu_output,
                            "max_abs_error": softmax_probe.max_abs_error
                        },
                        "attention_decode_probe": {
                            "kernel": "attention_decode_f16_f32_single_query",
                            "shape": {
                                "seq": attention_probe.seq,
                                "head_dim": attention_probe.head_dim
                            },
                            "metal_output": attention_probe.metal_output,
                            "cpu_output": attention_probe.cpu_output,
                            "max_abs_error": attention_probe.max_abs_error
                        },
                        "greedy_sampler_probe": {
                            "kernel": "greedy_argmax_f32",
                            "shape": {
                                "len": greedy_sampler_probe.len
                            },
                            "metal_token": greedy_sampler_probe.metal_token,
                            "metal_logit": greedy_sampler_probe.metal_logit,
                            "cpu_token": greedy_sampler_probe.cpu_token,
                            "cpu_logit": greedy_sampler_probe.cpu_logit
                        },
                        "kv_cache_append_probe": {
                            "kernel": "kv_cache_append_f16",
                            "shape": {
                                "seq": kv_cache_append_probe.seq,
                                "head_dim": kv_cache_append_probe.head_dim,
                                "position": kv_cache_append_probe.position
                            },
                            "metal_k_cache": kv_cache_append_probe.metal_k_cache,
                            "metal_v_cache": kv_cache_append_probe.metal_v_cache,
                            "cpu_k_cache": kv_cache_append_probe.cpu_k_cache,
                            "cpu_v_cache": kv_cache_append_probe.cpu_v_cache,
                            "max_abs_error": kv_cache_append_probe.max_abs_error
                        }
                    },
                    "engine": {
                        "status": "scaffolded",
                        "next": "add Qwen-size benchmarks and tokenizer/package loading"
                    }
                }))?
            );
        }
        EngineCommand::Bench(args) => {
            if args.hot {
                let suite = run_qwen_15b_hot_bench(args.iterations)?;
                println!("{}", serde_json::to_string_pretty(&suite)?);
            } else {
                let suite = run_qwen_15b_smoke_bench()?;
                println!("{}", serde_json::to_string_pretty(&suite)?);
            }
        }
        EngineCommand::Inspect(args) => {
            let inspection = inspect_package(args.package)?;
            println!("{}", serde_json::to_string_pretty(&inspection)?);
        }
        EngineCommand::PhaseBench(args) => {
            let report = run_engine_phase_bench(args)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        EngineCommand::QwenRun(args) => {
            let result = run_qwen_mlx_end2end(QwenMlxRunConfig {
                hf_dir: args.hf_dir,
                prompt: args.prompt,
                max_tokens: args.max_tokens,
                max_prompt_tokens: args.max_prompt_tokens,
                projection_backend: match args.projection_backend {
                    QwenProjectionBackendArg::Cpu => QwenProjectionBackend::Cpu,
                    QwenProjectionBackendArg::Metal => QwenProjectionBackend::Metal,
                },
            })?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        EngineCommand::Compare(args) => {
            let report = run_engine_compare(args)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }

    Ok(())
}

fn run_engine_phase_bench(args: EnginePhaseBenchArgs) -> anyhow::Result<PhaseBenchmarkReport> {
    anyhow::ensure!(
        args.iterations > 0,
        "phase benchmark iterations must be greater than zero"
    );

    let prefill_tokens = parse_prefill_tokens(&args.prefill_tokens)?;
    let device = MetalDeviceInfo::system_default()?;
    let model_id = args
        .hf_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("qwen-local")
        .to_string();

    let config_started = Instant::now();
    let qwen_config = read_qwen_config(&args.hf_dir)?
        .ok_or_else(|| anyhow::anyhow!("missing Hugging Face config.json in {:?}", args.hf_dir))?;
    let config_read_ms = elapsed_ms(config_started);

    let tokenizer_started = Instant::now();
    let tokenizer_json_bytes = read_optional_file_len(args.hf_dir.join("tokenizer.json"))?;
    let tokenizer_read_ms = tokenizer_json_bytes.map(|_| elapsed_ms(tokenizer_started));

    let index_started = Instant::now();
    let safetensors_index =
        read_safetensors_index(args.hf_dir.join("model.safetensors.index.json"))?;
    let safetensors_index_read_ms = safetensors_index
        .as_ref()
        .map(|_| elapsed_ms(index_started));
    let safetensors_bytes = read_optional_metadata_len(args.hf_dir.join("model.safetensors"))?;

    let decode_matmul_baseline = run_hot_f16_matmul_bench(
        format!(
            "decode_projection_matmul_f16_1x{}x{}_baseline_hot",
            qwen_config.hidden_size, qwen_config.hidden_size
        ),
        1,
        qwen_config.hidden_size as usize,
        qwen_config.hidden_size as usize,
        args.iterations,
    )?;
    let decode_projection = run_hot_f16_matvec_bench(
        format!(
            "decode_projection_matvec_f16_{}x{}_hot",
            qwen_config.hidden_size, qwen_config.hidden_size
        ),
        qwen_config.hidden_size as usize,
        qwen_config.hidden_size as usize,
        args.iterations,
    )?;

    let mut prefill = Vec::with_capacity(prefill_tokens.len());
    for tokens in prefill_tokens {
        let projection = run_hot_f16_matmul_bench(
            format!(
                "prefill_projection_f16_{}x{}x{}_hot",
                tokens, qwen_config.hidden_size, qwen_config.hidden_size
            ),
            tokens as usize,
            qwen_config.hidden_size as usize,
            qwen_config.hidden_size as usize,
            args.iterations,
        )?;
        prefill.push(build_prefill_phase_report(
            tokens,
            &qwen_config,
            &projection,
            &decode_projection,
            config_read_ms
                + tokenizer_read_ms.unwrap_or(0.0)
                + safetensors_index_read_ms.unwrap_or(0.0),
        ));
    }

    let decode_matvec_speedup_vs_matmul =
        decode_matmul_baseline.avg_elapsed_ms / decode_projection.avg_elapsed_ms;
    let report = PhaseBenchmarkReport {
        benchmark: "qwen-phase-benchmark-v1",
        device: device.name,
        source: PhaseBenchmarkSource {
            model_id,
            hf_dir: args.hf_dir.clone(),
        },
        artifacts: PhaseBenchmarkArtifacts {
            tokenizer_json_bytes,
            safetensors_index_bytes: safetensors_index.as_ref().map(|index| index.index_bytes),
            safetensors_total_size_bytes: safetensors_index
                .as_ref()
                .and_then(|index| index.total_size_bytes)
                .or(safetensors_bytes),
            safetensors_total_parameters: safetensors_index
                .as_ref()
                .and_then(|index| index.total_parameters),
        },
        qwen_config,
        iterations: args.iterations,
        config_read_ms,
        tokenizer_read_ms,
        safetensors_index_read_ms,
        decode_matmul_baseline,
        decode_matvec_speedup_vs_matmul,
        decode_projection,
        prefill,
        note: "Synthetic TinyEngine phase benchmark: prefill uses hot Metal matmul; decode/TTFT now use the dedicated hot Metal matvec path. Full end-to-end inference is not wired yet.".to_string(),
    };

    if let Some(out) = args.out {
        if let Some(parent) = out.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    }

    Ok(report)
}

fn run_engine_compare(args: EngineCompareArgs) -> anyhow::Result<AppleToAppleCompareReport> {
    anyhow::ensure!(args.runs > 0, "compare runs must be greater than zero");
    anyhow::ensure!(args.max_tokens > 0, "max_tokens must be greater than zero");
    anyhow::ensure!(
        args.max_prompt_tokens > 0,
        "max_prompt_tokens must be greater than zero"
    );
    anyhow::ensure!(
        args.hf_dir.is_dir(),
        "TinyEngine HF/MLX model or tokenizer directory does not exist: {:?}",
        args.hf_dir
    );
    anyhow::ensure!(
        args.gguf.is_file(),
        "llama.cpp GGUF model does not exist: {:?}",
        args.gguf
    );
    anyhow::ensure!(
        args.llama_bin.is_file(),
        "llama.cpp completion binary does not exist: {:?}",
        args.llama_bin
    );

    let projection_backend = match args.projection_backend {
        QwenProjectionBackendArg::Cpu => QwenProjectionBackend::Cpu,
        QwenProjectionBackendArg::Metal => QwenProjectionBackend::Metal,
    };
    let device = MetalDeviceInfo::system_default()?;
    let formatted_prompt = format_qwen_chat_prompt(&args.prompt);
    let llama_cpp_version = read_llama_version(&args.llama_bin)?;
    let models = AppleToAppleModels {
        tinyengine_hf_dir: args.hf_dir.clone(),
        tinyengine_safetensors_bytes: match args.tinyengine_source {
            TinyEngineCompareSourceArg::Mlx => {
                optional_file_size(&args.hf_dir.join("model.safetensors"))?
            }
            TinyEngineCompareSourceArg::Gguf => None,
        },
        llama_gguf: args.gguf.clone(),
        llama_gguf_bytes: optional_file_size(&args.gguf)?.unwrap_or(0),
        llama_bin: args.llama_bin.clone(),
    };
    let settings = AppleToAppleSettings {
        prompt: args.prompt.clone(),
        formatted_prompt: formatted_prompt.clone(),
        max_tokens: args.max_tokens,
        max_prompt_tokens: args.max_prompt_tokens,
        projection_backend,
        tinyengine_source: args.tinyengine_source,
        ctx_size: args.ctx_size,
        gpu_layers: args.gpu_layers,
        seed: args.seed,
        llama_warmup: args.llama_warmup,
        runs: args.runs,
    };

    let mut runs = Vec::with_capacity(args.runs);
    for run_index in 0..args.runs {
        let tinyengine = match args.tinyengine_source {
            TinyEngineCompareSourceArg::Mlx => run_qwen_mlx_end2end(QwenMlxRunConfig {
                hf_dir: args.hf_dir.clone(),
                prompt: args.prompt.clone(),
                max_tokens: args.max_tokens,
                max_prompt_tokens: args.max_prompt_tokens,
                projection_backend,
            })?,
            TinyEngineCompareSourceArg::Gguf => run_qwen_gguf_end2end(QwenGgufRunConfig {
                gguf_path: args.gguf.clone(),
                tokenizer_dir: args.hf_dir.clone(),
                prompt: args.prompt.clone(),
                max_tokens: args.max_tokens,
                max_prompt_tokens: args.max_prompt_tokens,
                projection_backend,
            })?,
        };
        let llama_cpp = run_llama_completion(&args, &formatted_prompt)?;
        let comparison = build_apple_to_apple_comparison(&tinyengine, &llama_cpp);
        runs.push(AppleToAppleRun {
            run_index,
            tinyengine,
            llama_cpp,
            comparison,
        });
    }

    let report = AppleToAppleCompareReport {
        benchmark: "qwen2.5-0.5b-apple-to-apple-v1",
        device: device.name,
        llama_cpp_version,
        models,
        settings,
        summary: build_apple_to_apple_summary(&runs),
        runs,
        note: match args.tinyengine_source {
            TinyEngineCompareSourceArg::Mlx => "Same base Qwen2.5 0.5B instruct prompt and deterministic generation settings. Quantization is same class but not bit-identical: TinyEngine uses MLX affine 4-bit weights, llama.cpp uses GGUF weights. llama.cpp decode eval follows its reported convention: max_tokens generated usually means max_tokens - 1 eval runs after prompt eval.".to_string(),
            TinyEngineCompareSourceArg::Gguf => "Same Qwen2.5 0.5B GGUF file, same quantized tensors, same prompt, and deterministic generation settings. TinyEngine maps GGUF Q4_0 blocks and GGUF Q8_0 output.weight onto Metal kernels.".to_string(),
        },
    };

    if let Some(out) = args.out {
        if let Some(parent) = out.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    }

    Ok(report)
}

fn run_llama_completion(
    args: &EngineCompareArgs,
    formatted_prompt: &str,
) -> anyhow::Result<LlamaCompletionRun> {
    let mut argv = vec![
        "-m".to_string(),
        args.gguf.display().to_string(),
        "-p".to_string(),
        formatted_prompt.to_string(),
        "-n".to_string(),
        args.max_tokens.to_string(),
        "--temp".to_string(),
        "0".to_string(),
        "--top-k".to_string(),
        "1".to_string(),
        "--seed".to_string(),
        args.seed.to_string(),
        "--no-display-prompt".to_string(),
        "-no-cnv".to_string(),
        "--simple-io".to_string(),
        "-ngl".to_string(),
        args.gpu_layers.to_string(),
        "-c".to_string(),
        args.ctx_size.to_string(),
    ];
    if !args.llama_warmup {
        argv.push("--no-warmup".to_string());
    }

    let started = Instant::now();
    let output = Command::new(&args.llama_bin)
        .args(&argv)
        .env("LC_ALL", "C")
        .env("LANG", "C")
        .output()
        .with_context(|| format!("failed to run {:?}", args.llama_bin))?;
    let wall_ms = elapsed_ms(started);
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        bail!(
            "llama-completion failed with status {}:\n{}",
            output.status,
            stderr_tail(&stderr, 40)
        );
    }

    let timings = parse_llama_completion_timings(&stderr)?;
    Ok(LlamaCompletionRun {
        argv,
        generated_text: clean_generated_text(&stdout),
        wall_ms,
        load_ms: timings.load_ms,
        prompt_eval_ms: timings.prompt_eval_ms,
        prompt_eval_tokens: timings.prompt_eval_tokens,
        prompt_tokens_per_second: tokens_per_second(
            timings.prompt_eval_tokens,
            timings.prompt_eval_ms,
        ),
        eval_ms: timings.eval_ms,
        eval_runs: timings.eval_runs,
        eval_tokens_per_second: tokens_per_second(timings.eval_runs, timings.eval_ms),
        total_ms: timings.total_ms,
        total_tokens: timings.total_tokens,
        stderr_tail: stderr_tail(&stderr, 30),
    })
}

fn read_llama_version(llama_bin: &PathBuf) -> anyhow::Result<String> {
    let output = Command::new(llama_bin)
        .arg("--version")
        .env("LC_ALL", "C")
        .env("LANG", "C")
        .output()
        .with_context(|| format!("failed to run {:?} --version", llama_bin))?;
    if !output.status.success() {
        bail!("failed to read llama.cpp version from {:?}", llama_bin);
    }
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" | "))
}

fn parse_llama_completion_timings(stderr: &str) -> anyhow::Result<LlamaCompletionTimings> {
    let load_ms = parse_perf_ms(stderr, "load time")?;
    let (prompt_eval_ms, prompt_eval_tokens) = parse_perf_ms_count(stderr, "prompt eval time")?;
    let (eval_ms, eval_runs) = parse_perf_ms_count(stderr, "eval time")?;
    let (total_ms, total_tokens) = parse_perf_ms_count(stderr, "total time")?;
    Ok(LlamaCompletionTimings {
        load_ms,
        prompt_eval_ms,
        prompt_eval_tokens,
        eval_ms,
        eval_runs,
        total_ms,
        total_tokens,
    })
}

fn parse_perf_ms(stderr: &str, label: &str) -> anyhow::Result<f64> {
    for line in stderr.lines() {
        let metric = perf_metric(line);
        if !metric.trim_start().starts_with(label) {
            continue;
        }
        let (_, after_equals) = metric
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("missing `=` in llama.cpp timing line: {line}"))?;
        let raw_ms = after_equals
            .trim_start()
            .split_whitespace()
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing ms value in llama.cpp timing line: {line}"))?;
        return parse_float(raw_ms)
            .with_context(|| format!("invalid ms value `{raw_ms}` in llama.cpp timing line"));
    }
    bail!("missing llama.cpp timing line `{label}`")
}

fn parse_perf_ms_count(stderr: &str, label: &str) -> anyhow::Result<(f64, usize)> {
    for line in stderr.lines() {
        let metric = perf_metric(line);
        if !metric.trim_start().starts_with(label) {
            continue;
        }
        let ms = parse_perf_ms(line, label)?;
        let (_, after_slash) = metric
            .split_once('/')
            .ok_or_else(|| anyhow::anyhow!("missing `/` count in llama.cpp timing line: {line}"))?;
        let raw_count = after_slash
            .trim_start()
            .split_whitespace()
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing count in llama.cpp timing line: {line}"))?;
        let count = raw_count
            .parse::<usize>()
            .with_context(|| format!("invalid count `{raw_count}` in llama.cpp timing line"))?;
        return Ok((ms, count));
    }
    bail!("missing llama.cpp timing line `{label}`")
}

fn perf_metric(line: &str) -> &str {
    line.split_once("common_perf_print:")
        .map(|(_, metric)| metric)
        .unwrap_or(line)
}

fn parse_float(raw: &str) -> anyhow::Result<f64> {
    Ok(raw.replace(',', ".").parse::<f64>()?)
}

fn clean_generated_text(text: &str) -> String {
    text.chars()
        .filter(|ch| *ch == '\n' || *ch == '\t' || !ch.is_control())
        .collect::<String>()
        .trim()
        .to_string()
}

fn stderr_tail(stderr: &str, lines: usize) -> String {
    let all_lines = stderr.lines().collect::<Vec<_>>();
    let start = all_lines.len().saturating_sub(lines);
    all_lines[start..].join("\n")
}

fn optional_file_size(path: &std::path::Path) -> anyhow::Result<Option<u64>> {
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(Some(metadata.len())),
        Ok(_) => Ok(None),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn tokens_per_second(tokens: usize, elapsed_ms: f64) -> f64 {
    if elapsed_ms > 0.0 {
        tokens as f64 / (elapsed_ms / 1000.0)
    } else {
        0.0
    }
}

fn build_apple_to_apple_comparison(
    tinyengine: &QwenMlxRunResult,
    llama_cpp: &LlamaCompletionRun,
) -> AppleToAppleRunComparison {
    AppleToAppleRunComparison {
        generated_text_match: tinyengine.generated_text == llama_cpp.generated_text,
        prompt_token_count_match: tinyengine.prompt_tokens_used == llama_cpp.prompt_eval_tokens,
        decode_eval_count_match: tinyengine.decode_eval_tokens == llama_cpp.eval_runs,
        tiny_prompt_speed_ratio_vs_llama: ratio(
            llama_cpp.prompt_eval_ms,
            tinyengine.prompt_eval_ms,
        ),
        tiny_decode_speed_ratio_vs_llama: ratio(
            tinyengine.decode_tokens_per_second,
            llama_cpp.eval_tokens_per_second,
        ),
        tiny_total_wall_speed_ratio_vs_llama: ratio(llama_cpp.wall_ms, tinyengine.total_ms),
    }
}

fn build_apple_to_apple_summary(runs: &[AppleToAppleRun]) -> AppleToAppleSummary {
    AppleToAppleSummary {
        measured_runs: runs.len(),
        all_generated_text_match: runs.iter().all(|run| run.comparison.generated_text_match),
        all_prompt_token_counts_match: runs
            .iter()
            .all(|run| run.comparison.prompt_token_count_match),
        all_decode_eval_counts_match: runs
            .iter()
            .all(|run| run.comparison.decode_eval_count_match),
        tiny_prompt_eval_ms_median: median(
            runs.iter()
                .map(|run| run.tinyengine.prompt_eval_ms)
                .collect(),
        ),
        llama_prompt_eval_ms_median: median(
            runs.iter()
                .map(|run| run.llama_cpp.prompt_eval_ms)
                .collect(),
        ),
        tiny_decode_tokens_per_second_median: median(
            runs.iter()
                .map(|run| run.tinyengine.decode_tokens_per_second)
                .collect(),
        ),
        llama_decode_tokens_per_second_median: median(
            runs.iter()
                .map(|run| run.llama_cpp.eval_tokens_per_second)
                .collect(),
        ),
        tiny_prompt_speed_ratio_vs_llama_median: median(
            runs.iter()
                .map(|run| run.comparison.tiny_prompt_speed_ratio_vs_llama)
                .collect(),
        ),
        tiny_decode_speed_ratio_vs_llama_median: median(
            runs.iter()
                .map(|run| run.comparison.tiny_decode_speed_ratio_vs_llama)
                .collect(),
        ),
        tiny_total_wall_speed_ratio_vs_llama_median: median(
            runs.iter()
                .map(|run| run.comparison.tiny_total_wall_speed_ratio_vs_llama)
                .collect(),
        ),
    }
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

async fn run_convert(args: ConvertArgs) -> anyhow::Result<()> {
    let source = match args.source {
        ConvertSource::Hf => SourceFormat::HuggingFace,
        ConvertSource::Gguf => SourceFormat::Gguf,
    };
    let mut metadata = TmaPackageMetadata::scaffold(
        args.model_id,
        ModelArchitecture::Qwen25,
        source.clone(),
        args.input.display().to_string(),
    );
    fs::create_dir_all(args.out.join("tensors"))?;
    let tokenizer_copied = if matches!(source, SourceFormat::HuggingFace) {
        let tokenizer_src = args.input.join("tokenizer.json");
        if tokenizer_src.is_file() {
            fs::copy(&tokenizer_src, args.out.join("tokenizer.json"))?;
            metadata.tokenizer_path = Some("tokenizer.json".to_string());
            true
        } else {
            false
        }
    } else {
        false
    };
    if matches!(source, SourceFormat::HuggingFace) {
        metadata.qwen_config = read_qwen_config(&args.input)?;
    }
    write_metadata(&args.out, &metadata)?;
    let inspection = inspect_package(&args.out)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "package": args.out,
            "status": "metadata-only",
            "tokenizer_copied": tokenizer_copied,
            "inspection": inspection,
            "next": "tensor conversion is not implemented yet"
        }))?
    );
    Ok(())
}

fn read_qwen_config(input_dir: &std::path::Path) -> anyhow::Result<Option<QwenConfig>> {
    let config_path = input_dir.join("config.json");
    if !config_path.is_file() {
        return Ok(None);
    }

    let value: serde_json::Value = serde_json::from_str(&fs::read_to_string(&config_path)?)?;
    let config = value.get("text_config").unwrap_or(&value);
    let hidden_size = required_u64(config, "hidden_size")?;
    let num_attention_heads = required_u64(config, "num_attention_heads")?;
    anyhow::ensure!(
        num_attention_heads > 0,
        "`num_attention_heads` must be greater than zero in Hugging Face config.json"
    );
    let head_dim = match config.get("head_dim").and_then(serde_json::Value::as_u64) {
        Some(head_dim) => head_dim,
        None => {
            anyhow::ensure!(
                hidden_size % num_attention_heads == 0,
                "`hidden_size` must be divisible by `num_attention_heads` when `head_dim` is absent"
            );
            hidden_size / num_attention_heads
        }
    };

    Ok(Some(QwenConfig {
        hidden_size,
        intermediate_size: required_u64(config, "intermediate_size")?,
        num_hidden_layers: required_u64(config, "num_hidden_layers")?,
        num_attention_heads,
        num_key_value_heads: config
            .get("num_key_value_heads")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(num_attention_heads),
        head_dim,
        vocab_size: required_u64(config, "vocab_size")?,
        max_position_embeddings: required_u64(config, "max_position_embeddings")?,
        rope_theta: config
            .get("rope_theta")
            .or_else(|| config.pointer("/rope_parameters/rope_theta"))
            .and_then(|value| {
                value
                    .as_u64()
                    .or_else(|| value.as_f64().map(|number| number as u64))
            })
            .unwrap_or(1_000_000),
    }))
}

fn required_u64(value: &serde_json::Value, key: &str) -> anyhow::Result<u64> {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("missing or invalid `{key}` in Hugging Face config.json"))
}

fn parse_prefill_tokens(input: &str) -> anyhow::Result<Vec<u64>> {
    let mut values = Vec::new();
    for raw in input.split(',') {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: u64 = trimmed
            .parse()
            .with_context(|| format!("invalid prefill token count `{trimmed}`"))?;
        anyhow::ensure!(value > 0, "prefill token counts must be greater than zero");
        values.push(value);
    }
    anyhow::ensure!(
        !values.is_empty(),
        "provide at least one prefill token count"
    );
    Ok(values)
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn read_optional_file_len(path: PathBuf) -> anyhow::Result<Option<u64>> {
    if !path.is_file() {
        return Ok(None);
    }
    Ok(Some(fs::read(path)?.len() as u64))
}

fn read_optional_metadata_len(path: PathBuf) -> anyhow::Result<Option<u64>> {
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(Some(metadata.len())),
        Ok(_) => Ok(None),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn read_safetensors_index(path: PathBuf) -> anyhow::Result<Option<SafetensorsIndexSummary>> {
    if !path.is_file() {
        return Ok(None);
    }
    let data = fs::read_to_string(path)?;
    let index_bytes = data.len() as u64;
    let value: serde_json::Value = serde_json::from_str(&data)?;
    let metadata = value.get("metadata");
    Ok(Some(SafetensorsIndexSummary {
        index_bytes,
        total_size_bytes: metadata
            .and_then(|metadata| metadata.get("total_size"))
            .and_then(serde_json::Value::as_u64),
        total_parameters: metadata
            .and_then(|metadata| metadata.get("total_parameters"))
            .and_then(serde_json::Value::as_u64),
    }))
}

fn build_prefill_phase_report(
    tokens: u64,
    config: &QwenConfig,
    projection: &HotKernelBenchmarkResult,
    decode_projection: &HotKernelBenchmarkResult,
    io_metadata_ms: f64,
) -> PrefillPhaseBenchmark {
    let prefill_throughput_gops = projection.throughput_gops.max(0.001);
    let decode_throughput_gops = decode_projection.throughput_gops.max(0.001);
    let prefill_ops = qwen_prefill_ops(tokens, config);
    let decode_ops = qwen_decode_first_token_ops(tokens, config);
    let estimated_prefill_ms = ops_to_ms(prefill_ops, prefill_throughput_gops);
    let estimated_decode_first_token_ms = ops_to_ms(decode_ops, decode_throughput_gops);

    PrefillPhaseBenchmark {
        prompt_tokens: tokens,
        projection_kernel: projection.clone(),
        estimated_full_prefill_ops: prefill_ops,
        estimated_first_decode_ops: decode_ops,
        estimated_full_prefill_ms: estimated_prefill_ms,
        estimated_first_decode_ms: estimated_decode_first_token_ms,
        estimated_ttft_ms: io_metadata_ms + estimated_prefill_ms + estimated_decode_first_token_ms,
    }
}

fn qwen_prefill_ops(tokens: u64, config: &QwenConfig) -> u64 {
    let hidden = config.hidden_size;
    let kv_width = config.num_key_value_heads * config.head_dim;
    let linear_per_layer = 2
        * tokens
        * (2 * hidden * hidden + 2 * hidden * kv_width + 3 * hidden * config.intermediate_size);
    let attention_per_layer = 4 * config.num_attention_heads * tokens * tokens * config.head_dim;
    config.num_hidden_layers * (linear_per_layer + attention_per_layer)
}

fn qwen_decode_first_token_ops(prompt_tokens: u64, config: &QwenConfig) -> u64 {
    let hidden = config.hidden_size;
    let kv_width = config.num_key_value_heads * config.head_dim;
    let linear_per_layer =
        2 * (2 * hidden * hidden + 2 * hidden * kv_width + 3 * hidden * config.intermediate_size);
    let attention_per_layer = 4 * config.num_attention_heads * prompt_tokens * config.head_dim;
    let logits = 2 * hidden * config.vocab_size;
    config.num_hidden_layers * (linear_per_layer + attention_per_layer) + logits
}

fn ops_to_ms(ops: u64, throughput_gops: f64) -> f64 {
    ops as f64 / (throughput_gops * 1_000_000_000.0) * 1000.0
}

#[derive(Debug, Serialize)]
struct PhaseBenchmarkReport {
    benchmark: &'static str,
    device: String,
    source: PhaseBenchmarkSource,
    artifacts: PhaseBenchmarkArtifacts,
    qwen_config: QwenConfig,
    iterations: u32,
    config_read_ms: f64,
    tokenizer_read_ms: Option<f64>,
    safetensors_index_read_ms: Option<f64>,
    decode_matmul_baseline: HotKernelBenchmarkResult,
    decode_matvec_speedup_vs_matmul: f64,
    decode_projection: HotKernelBenchmarkResult,
    prefill: Vec<PrefillPhaseBenchmark>,
    note: String,
}

#[derive(Debug, Serialize)]
struct PhaseBenchmarkSource {
    model_id: String,
    hf_dir: PathBuf,
}

#[derive(Debug, Serialize)]
struct PhaseBenchmarkArtifacts {
    tokenizer_json_bytes: Option<u64>,
    safetensors_index_bytes: Option<u64>,
    safetensors_total_size_bytes: Option<u64>,
    safetensors_total_parameters: Option<u64>,
}

#[derive(Debug)]
struct SafetensorsIndexSummary {
    index_bytes: u64,
    total_size_bytes: Option<u64>,
    total_parameters: Option<u64>,
}

#[derive(Debug, Serialize)]
struct PrefillPhaseBenchmark {
    prompt_tokens: u64,
    projection_kernel: HotKernelBenchmarkResult,
    estimated_full_prefill_ops: u64,
    estimated_first_decode_ops: u64,
    estimated_full_prefill_ms: f64,
    estimated_first_decode_ms: f64,
    estimated_ttft_ms: f64,
}

#[derive(Debug, Serialize)]
struct AppleToAppleCompareReport {
    benchmark: &'static str,
    device: String,
    llama_cpp_version: String,
    models: AppleToAppleModels,
    settings: AppleToAppleSettings,
    summary: AppleToAppleSummary,
    runs: Vec<AppleToAppleRun>,
    note: String,
}

#[derive(Debug, Serialize)]
struct AppleToAppleModels {
    tinyengine_hf_dir: PathBuf,
    tinyengine_safetensors_bytes: Option<u64>,
    llama_gguf: PathBuf,
    llama_gguf_bytes: u64,
    llama_bin: PathBuf,
}

#[derive(Debug, Serialize)]
struct AppleToAppleSettings {
    prompt: String,
    formatted_prompt: String,
    max_tokens: usize,
    max_prompt_tokens: usize,
    projection_backend: QwenProjectionBackend,
    tinyengine_source: TinyEngineCompareSourceArg,
    ctx_size: u32,
    gpu_layers: i32,
    seed: u64,
    llama_warmup: bool,
    runs: usize,
}

#[derive(Debug, Serialize)]
struct AppleToAppleRun {
    run_index: usize,
    tinyengine: QwenMlxRunResult,
    llama_cpp: LlamaCompletionRun,
    comparison: AppleToAppleRunComparison,
}

#[derive(Debug, Serialize)]
struct LlamaCompletionRun {
    argv: Vec<String>,
    generated_text: String,
    wall_ms: f64,
    load_ms: f64,
    prompt_eval_ms: f64,
    prompt_eval_tokens: usize,
    prompt_tokens_per_second: f64,
    eval_ms: f64,
    eval_runs: usize,
    eval_tokens_per_second: f64,
    total_ms: f64,
    total_tokens: usize,
    stderr_tail: String,
}

#[derive(Debug)]
struct LlamaCompletionTimings {
    load_ms: f64,
    prompt_eval_ms: f64,
    prompt_eval_tokens: usize,
    eval_ms: f64,
    eval_runs: usize,
    total_ms: f64,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct AppleToAppleRunComparison {
    generated_text_match: bool,
    prompt_token_count_match: bool,
    decode_eval_count_match: bool,
    tiny_prompt_speed_ratio_vs_llama: f64,
    tiny_decode_speed_ratio_vs_llama: f64,
    tiny_total_wall_speed_ratio_vs_llama: f64,
}

#[derive(Debug, Serialize)]
struct AppleToAppleSummary {
    measured_runs: usize,
    all_generated_text_match: bool,
    all_prompt_token_counts_match: bool,
    all_decode_eval_counts_match: bool,
    tiny_prompt_eval_ms_median: f64,
    llama_prompt_eval_ms_median: f64,
    tiny_decode_tokens_per_second_median: f64,
    llama_decode_tokens_per_second_median: f64,
    tiny_prompt_speed_ratio_vs_llama_median: f64,
    tiny_decode_speed_ratio_vs_llama_median: f64,
    tiny_total_wall_speed_ratio_vs_llama_median: f64,
}

async fn run_models(args: SharedArgs) -> anyhow::Result<()> {
    let profile = load_profile(&args.profile)?;
    let models = vec![configured_model(&args, &profile)];
    println!("{}", serde_json::to_string_pretty(&models)?);
    Ok(())
}

async fn run_chat(args: ChatArgs) -> anyhow::Result<()> {
    let profile = load_profile(&args.shared.profile)?;
    if args.prompt.is_empty() {
        bail!("provide a prompt, for example: tinyagent chat \"ciao\"");
    }

    let backend = build_backend(&args.shared, &profile).await?;
    let request = GenerateRequest {
        model: args.shared.model,
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: args.prompt.join(" "),
            name: None,
            tool_call_id: None,
        }],
        max_tokens: Some(256),
        temperature: Some(0.7),
        stream: true,
        tools: Vec::new(),
    };

    let mut tokens = backend.generate(request).await?;
    while let Some(event) = tokens.next().await {
        let event = event?;
        print!("{}", event.token);
        io::stdout().flush().context("failed to flush stdout")?;
    }
    println!();

    Ok(())
}

async fn run_serve(args: ServeArgs) -> anyhow::Result<()> {
    let profile = load_profile(&args.shared.profile)?;
    let backend = build_backend(&args.shared, &profile).await?;
    let state = AppState { backend };
    let app = Router::new()
        .route("/health", get(health))
        .route("/local/models", get(local_models))
        .route("/local/chat", post(local_chat))
        .with_state(state);

    let address = format!("{}:{}", args.host, args.port);
    let listener = TcpListener::bind(&address)
        .await
        .with_context(|| format!("failed to bind {address}"))?;

    eprintln!(
        "tinyagent serving on http://{address} with profile {}",
        profile.name
    );
    axum::serve(listener, app).await?;
    Ok(())
}

async fn run_bench(args: SharedArgs) -> anyhow::Result<()> {
    let profile = load_profile(&args.profile)?;
    let backend = build_backend(&args, &profile).await?;
    let request = GenerateRequest {
        model: args.model,
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: "Say hello from the benchmark.".to_string(),
            name: None,
            tool_call_id: None,
        }],
        max_tokens: Some(128),
        temperature: Some(0.0),
        stream: true,
        tools: Vec::new(),
    };

    let started = Instant::now();
    let mut stream = backend.generate(request).await?;
    let mut token_events = 0_u64;
    while let Some(event) = stream.next().await {
        let event = event?;
        if !event.token.is_empty() {
            token_events += 1;
        }
    }

    let elapsed = started.elapsed();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "backend": format!("{:?}", args.backend).to_lowercase(),
            "token_events": token_events,
            "elapsed_ms": elapsed.as_millis(),
        }))?
    );

    Ok(())
}

fn load_profile(name: &str) -> anyhow::Result<HardwareProfile> {
    match name {
        "mac-8gb" => Ok(HardwareProfile::mac_8gb()),
        other => bail!("unsupported profile {other}; available profiles: mac-8gb"),
    }
}

async fn build_backend(
    args: &SharedArgs,
    profile: &HardwareProfile,
) -> anyhow::Result<Arc<dyn InferenceBackend>> {
    let mut model = configured_model(args, profile);

    match args.backend {
        BackendKind::Stub => Ok(Arc::new(StubBackend::new(model))),
        BackendKind::Metal => {
            let backend = MetalBackend::new(
                MetalBackendConfig {
                    package_path: args.package.clone(),
                    profile: profile.name.clone(),
                    ctx_size: profile.ctx_size,
                    batch_size: profile.batch_size,
                    ubatch_size: profile.ubatch_size,
                },
                model,
            )?;
            Ok(Arc::new(backend))
        }
        BackendKind::Llama => {
            let gguf = args.gguf.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "llama oracle backend requires --gguf <model.gguf>. Use --backend metal for TinyEngine or --backend stub for scaffold tests."
                )
            })?;
            model.backend = "llama.cpp-oracle".to_string();
            model.status = "loaded".to_string();
            let backend = LlamaServerBackend::spawn(
                LlamaServerConfig {
                    executable: args.llama_server_bin.clone(),
                    host: args.llama_host.clone(),
                    port: args.llama_port,
                    model_path: gguf.clone(),
                    profile: profile.name.clone(),
                    ctx_size: profile.ctx_size,
                    batch_size: profile.batch_size,
                    ubatch_size: profile.ubatch_size,
                    gpu_layers: 999,
                },
                model,
            )
            .await?;
            Ok(Arc::new(backend))
        }
    }
}

fn configured_model(args: &SharedArgs, profile: &HardwareProfile) -> ModelInfo {
    let mut model = ModelInfo::default_qwen_coder_stub();
    model.id = args.model.clone();
    model.recommended_context_8gb = profile.ctx_size;
    model.quantization = "tma-f16-first-q8-q4-planned".to_string();
    match args.backend {
        BackendKind::Metal => {
            model.backend = "custom-metal".to_string();
            model.status = if args.package.is_some() {
                "configured".to_string()
            } else {
                "requires-tma-package".to_string()
            };
        }
        BackendKind::Llama => {
            model.backend = "llama.cpp-oracle".to_string();
            model.status = if args.gguf.is_some() {
                "configured".to_string()
            } else {
                "requires-gguf".to_string()
            };
        }
        BackendKind::Stub => {
            model.backend = "stub".to_string();
            model.status = "stub".to_string();
        }
    }
    model
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "service": "tinyagent"
    }))
}

async fn local_models(
    State(state): State<AppState>,
) -> Result<Json<LocalModelsResponse>, ApiError> {
    let models = state.backend.models().await?;
    Ok(Json(LocalModelsResponse { models }))
}

async fn local_chat(
    State(state): State<AppState>,
    Json(payload): Json<LocalChatRequest>,
) -> Result<Response, ApiError> {
    if !payload.tools.is_empty() {
        return Err(ApiError::not_implemented(
            "tool calls are planned but not implemented in the local API yet",
        ));
    }

    let request = GenerateRequest {
        model: payload.model.clone(),
        messages: payload.messages,
        max_tokens: payload.max_tokens,
        temperature: payload.temperature,
        stream: payload.stream,
        tools: Vec::new(),
    };

    let token_stream = state.backend.generate(request).await?;
    if payload.stream {
        let model = payload.model;
        let chunks = token_stream.map(move |event| {
            let (event_name, data) = match event {
                Ok(event) => {
                    if let Some(reason) = event.finish_reason {
                        (
                            "done",
                            json!({
                                "model": model,
                                "finish_reason": reason
                            }),
                        )
                    } else {
                        (
                            "token",
                            json!({
                                "model": model,
                                "token": event.token
                            }),
                        )
                    }
                }
                Err(error) => (
                    "error",
                    json!({
                        "message": error.to_string(),
                        "type": "backend_error"
                    }),
                ),
            };
            Ok::<Event, Infallible>(Event::default().event(event_name).data(data.to_string()))
        });

        return Ok(Sse::new(chunks).into_response());
    }

    let mut content = String::new();
    let mut finish_reason = "stop".to_string();
    let mut stream = token_stream;
    while let Some(event) = stream.next().await {
        let event = event?;
        content.push_str(&event.token);
        if let Some(reason) = event.finish_reason {
            finish_reason = reason;
        }
    }

    Ok(Json(LocalChatResponse {
        model: payload.model,
        message: LocalAssistantMessage {
            role: "assistant",
            content,
        },
        finish_reason,
    })
    .into_response())
}

#[derive(Debug, Deserialize)]
struct LocalChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    tools: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct LocalChatResponse {
    model: String,
    message: LocalAssistantMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct LocalAssistantMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct LocalModelsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn not_implemented(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            message: message.into(),
        }
    }
}

impl From<TinyAgentError> for ApiError {
    fn from(error: TinyAgentError) -> Self {
        let status = match error {
            TinyAgentError::Unsupported(_) => StatusCode::NOT_IMPLEMENTED,
            TinyAgentError::Configuration(_) => StatusCode::BAD_REQUEST,
            TinyAgentError::Backend(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };

        Self {
            status,
            message: error.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "error": {
                    "message": self.message,
                    "type": "tinyagent_error"
                }
            })),
        )
            .into_response()
    }
}
