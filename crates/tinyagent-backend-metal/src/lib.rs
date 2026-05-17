use std::{path::PathBuf, time::Instant};

use async_trait::async_trait;
use half::f16;
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use serde::Serialize;
use tinyagent_core::{
    GenerateRequest, InferenceBackend, ModelInfo, Result, TinyAgentError, TokenStream,
};

mod qwen_mlx;
pub use qwen_mlx::{
    run_q4_affine_matvec_probe, run_qwen_mlx_end2end, Q4AffineMatVecProbeResult, QwenMlxRunConfig,
    QwenMlxRunResult, QwenProjectionBackend,
};

#[derive(Debug, Clone)]
pub struct MetalBackendConfig {
    pub package_path: Option<PathBuf>,
    pub profile: String,
    pub ctx_size: u32,
    pub batch_size: u32,
    pub ubatch_size: u32,
}

#[derive(Debug, Clone)]
pub struct MetalDeviceInfo {
    pub name: String,
}

impl MetalDeviceInfo {
    pub fn system_default() -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| {
            TinyAgentError::Configuration(
                "no default Metal device found; TinyMetalAgent requires Apple Silicon or a Metal-capable GPU"
                    .to_string(),
            )
        })?;

        Ok(Self {
            name: device.name().to_string(),
        })
    }
}

pub fn run_add_kernel_probe() -> Result<Vec<f32>> {
    let device = Device::system_default().ok_or_else(|| {
        TinyAgentError::Configuration(
            "no default Metal device found; TinyEngine requires Apple Silicon or a Metal-capable GPU"
                .to_string(),
        )
    })?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void vector_add(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* out [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            out[id] = a[id] + b[id];
        }
    "#;

    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to compile Metal probe kernel: {error}"))
        })?;
    let function = library.get_function("vector_add", None).map_err(|error| {
        TinyAgentError::Backend(format!("failed to load Metal probe kernel: {error}"))
    })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to create Metal pipeline: {error}"))
        })?;
    let queue = device.new_command_queue();

    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [10.0_f32, 20.0, 30.0, 40.0];
    let byte_len = std::mem::size_of_val(&a) as u64;
    let buffer_a = device.new_buffer_with_data(
        a.as_ptr().cast(),
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_b = device.new_buffer_with_data(
        b.as_ptr().cast(),
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_out = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&buffer_a), 0);
    encoder.set_buffer(1, Some(&buffer_b), 0);
    encoder.set_buffer(2, Some(&buffer_out), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: a.len() as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: a.len() as u64,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let ptr = buffer_out.contents().cast::<f32>();
    let values = unsafe { std::slice::from_raw_parts(ptr, a.len()) }.to_vec();
    Ok(values)
}

#[derive(Debug, Clone)]
pub struct MatmulProbeResult {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub metal_output: Vec<f32>,
    pub cpu_output: Vec<f32>,
    pub max_abs_error: f32,
}

#[derive(Debug, Clone)]
pub struct MatVecProbeResult {
    pub n: usize,
    pub k: usize,
    pub metal_output: Vec<f32>,
    pub cpu_output: Vec<f32>,
    pub max_abs_error: f32,
}

#[derive(Debug, Clone)]
pub struct RmsNormProbeResult {
    pub rows: usize,
    pub cols: usize,
    pub metal_output: Vec<f32>,
    pub cpu_output: Vec<f32>,
    pub max_abs_error: f32,
}

#[derive(Debug, Clone)]
pub struct RopeProbeResult {
    pub rows: usize,
    pub dims: usize,
    pub metal_output: Vec<f32>,
    pub cpu_output: Vec<f32>,
    pub max_abs_error: f32,
}

#[derive(Debug, Clone)]
pub struct SwiGluProbeResult {
    pub len: usize,
    pub metal_output: Vec<f32>,
    pub cpu_output: Vec<f32>,
    pub max_abs_error: f32,
}

#[derive(Debug, Clone)]
pub struct SoftmaxProbeResult {
    pub rows: usize,
    pub cols: usize,
    pub metal_output: Vec<f32>,
    pub cpu_output: Vec<f32>,
    pub max_abs_error: f32,
}

#[derive(Debug, Clone)]
pub struct AttentionProbeResult {
    pub seq: usize,
    pub head_dim: usize,
    pub metal_output: Vec<f32>,
    pub cpu_output: Vec<f32>,
    pub max_abs_error: f32,
}

#[derive(Debug, Clone)]
pub struct GreedySamplerProbeResult {
    pub len: usize,
    pub metal_token: u32,
    pub metal_logit: f32,
    pub cpu_token: u32,
    pub cpu_logit: f32,
}

#[derive(Debug, Clone)]
pub struct KvCacheAppendProbeResult {
    pub seq: usize,
    pub head_dim: usize,
    pub position: usize,
    pub metal_k_cache: Vec<f32>,
    pub metal_v_cache: Vec<f32>,
    pub cpu_k_cache: Vec<f32>,
    pub cpu_v_cache: Vec<f32>,
    pub max_abs_error: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct KernelBenchmarkSuite {
    pub device: String,
    pub preset: String,
    pub note: String,
    pub results: Vec<KernelBenchmarkResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct KernelBenchmarkResult {
    pub name: String,
    pub shape: String,
    pub logical_ops: u64,
    pub elapsed_ms_cold: f64,
    pub throughput_gops_cold: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HotKernelBenchmarkSuite {
    pub device: String,
    pub preset: String,
    pub iterations: u32,
    pub results: Vec<HotKernelBenchmarkResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HotKernelBenchmarkResult {
    pub name: String,
    pub shape: String,
    pub logical_ops_per_iter: u64,
    pub total_elapsed_ms: f64,
    pub avg_elapsed_ms: f64,
    pub throughput_gops: f64,
}

pub fn run_f16_matmul_probe() -> Result<MatmulProbeResult> {
    let m = 2_usize;
    let n = 4_usize;
    let k = 3_usize;
    let a = f16_vec(&[1.0, -2.0, 3.5, 4.0, 0.5, -1.5]);
    let b = f16_vec(&[
        0.25, -1.0, 2.0, 0.5, 1.5, 0.75, -0.5, 2.0, -1.25, 1.0, 0.25, -0.75,
    ]);

    let metal_output = run_f16_matmul(&a, &b, m, n, k)?;
    let cpu_output = cpu_matmul_f16_to_f32(&a, &b, m, n, k);
    let max_abs_error = metal_output
        .iter()
        .zip(cpu_output.iter())
        .map(|(metal, cpu)| (metal - cpu).abs())
        .fold(0.0_f32, f32::max);

    Ok(MatmulProbeResult {
        m,
        n,
        k,
        metal_output,
        cpu_output,
        max_abs_error,
    })
}

pub fn run_f16_matvec_probe() -> Result<MatVecProbeResult> {
    let n = 4_usize;
    let k = 3_usize;
    let x = f16_vec(&[1.0, -2.0, 3.5]);
    let w = f16_vec(&[
        0.25, -1.0, 2.0, 0.5, 1.5, 0.75, -0.5, 2.0, -1.25, 1.0, 0.25, -0.75,
    ]);

    let metal_output = run_f16_matvec(&x, &w, n, k)?;
    let cpu_output = cpu_matvec_f16_to_f32(&x, &w, n, k);
    let max_abs_error = metal_output
        .iter()
        .zip(cpu_output.iter())
        .map(|(metal, cpu)| (metal - cpu).abs())
        .fold(0.0_f32, f32::max);

    Ok(MatVecProbeResult {
        n,
        k,
        metal_output,
        cpu_output,
        max_abs_error,
    })
}

pub fn run_rmsnorm_probe() -> Result<RmsNormProbeResult> {
    let rows = 2_usize;
    let cols = 8_usize;
    let x = f16_vec(&[
        1.0, -2.0, 3.0, -4.0, 0.5, -0.25, 2.5, -1.5, -1.0, 0.75, -0.5, 1.25, 2.0, -3.0, 4.0, -2.5,
    ]);
    let weight = f16_vec(&[1.0, 0.75, 1.25, 0.5, 1.5, 0.25, 1.0, 2.0]);
    let eps = 1e-6_f32;

    let metal_output = run_rmsnorm_f16_to_f32(&x, &weight, rows, cols, eps)?;
    let cpu_output = cpu_rmsnorm_f16_to_f32(&x, &weight, rows, cols, eps);
    let max_abs_error = metal_output
        .iter()
        .zip(cpu_output.iter())
        .map(|(metal, cpu)| (metal - cpu).abs())
        .fold(0.0_f32, f32::max);

    Ok(RmsNormProbeResult {
        rows,
        cols,
        metal_output,
        cpu_output,
        max_abs_error,
    })
}

pub fn run_rope_probe() -> Result<RopeProbeResult> {
    let rows = 2_usize;
    let dims = 8_usize;
    let x = f16_vec(&[
        1.0, -2.0, 3.0, -4.0, 0.5, -0.25, 2.5, -1.5, -1.0, 0.75, -0.5, 1.25, 2.0, -3.0, 4.0, -2.5,
    ]);
    let cos = [
        0.9950042, 0.9800666, 0.9553365, 0.9210610, 0.9950042, 0.9800666, 0.9553365, 0.9210610,
    ];
    let sin = [
        0.0998334, 0.1986693, 0.2955202, 0.3894183, 0.0998334, 0.1986693, 0.2955202, 0.3894183,
    ];

    let metal_output = run_rope_f16_to_f32(&x, &cos, &sin, rows, dims)?;
    let cpu_output = cpu_rope_f16_to_f32(&x, &cos, &sin, rows, dims);
    let max_abs_error = metal_output
        .iter()
        .zip(cpu_output.iter())
        .map(|(metal, cpu)| (metal - cpu).abs())
        .fold(0.0_f32, f32::max);

    Ok(RopeProbeResult {
        rows,
        dims,
        metal_output,
        cpu_output,
        max_abs_error,
    })
}

pub fn run_swiglu_probe() -> Result<SwiGluProbeResult> {
    let gate = [-3.0_f32, -1.0, -0.25, 0.0, 0.5, 1.0, 2.0, 4.0];
    let up = [0.5_f32, -2.0, 3.0, 1.5, -1.25, 0.75, 2.5, -0.5];

    let metal_output = run_swiglu_f32(&gate, &up)?;
    let cpu_output = cpu_swiglu_f32(&gate, &up);
    let max_abs_error = metal_output
        .iter()
        .zip(cpu_output.iter())
        .map(|(metal, cpu)| (metal - cpu).abs())
        .fold(0.0_f32, f32::max);

    Ok(SwiGluProbeResult {
        len: gate.len(),
        metal_output,
        cpu_output,
        max_abs_error,
    })
}

pub fn run_softmax_probe() -> Result<SoftmaxProbeResult> {
    let rows = 2_usize;
    let cols = 6_usize;
    let x = [
        1.0_f32, -2.0, 3.5, 0.25, -0.75, 2.0, -1.5, 0.0, 0.5, 4.0, -3.0, 1.25,
    ];

    let metal_output = run_softmax_f32(&x, rows, cols)?;
    let cpu_output = cpu_softmax_f32(&x, rows, cols);
    let max_abs_error = metal_output
        .iter()
        .zip(cpu_output.iter())
        .map(|(metal, cpu)| (metal - cpu).abs())
        .fold(0.0_f32, f32::max);

    Ok(SoftmaxProbeResult {
        rows,
        cols,
        metal_output,
        cpu_output,
        max_abs_error,
    })
}

pub fn run_attention_probe() -> Result<AttentionProbeResult> {
    let seq = 4_usize;
    let head_dim = 4_usize;
    let q = f16_vec(&[0.5, -1.0, 1.5, 0.25]);
    let k = f16_vec(&[
        1.0, 0.0, -0.5, 0.75, -1.0, 0.25, 0.5, -0.25, 0.5, 1.5, -1.0, 0.0, 0.25, -0.75, 1.25, 0.5,
    ]);
    let v = f16_vec(&[
        0.25, 1.0, -1.0, 0.5, -0.5, 0.75, 1.5, -1.25, 1.0, -0.25, 0.5, 0.75, -1.5, 0.0, 0.25, 1.25,
    ]);
    let scale = (head_dim as f32).sqrt().recip();

    let metal_output = run_attention_decode_f16_to_f32(&q, &k, &v, seq, head_dim, scale)?;
    let cpu_output = cpu_attention_decode_f16_to_f32(&q, &k, &v, seq, head_dim, scale);
    let max_abs_error = metal_output
        .iter()
        .zip(cpu_output.iter())
        .map(|(metal, cpu)| (metal - cpu).abs())
        .fold(0.0_f32, f32::max);

    Ok(AttentionProbeResult {
        seq,
        head_dim,
        metal_output,
        cpu_output,
        max_abs_error,
    })
}

pub fn run_greedy_sampler_probe() -> Result<GreedySamplerProbeResult> {
    let logits = [
        -1.0_f32, 0.25, 3.5, 2.75, 3.5, -0.5, 1.25, 0.0, 4.25, 4.25, -2.0, 0.75,
    ];
    let (metal_token, metal_logit) = run_greedy_argmax_f32(&logits)?;
    let (cpu_token, cpu_logit) = cpu_greedy_argmax_f32(&logits);

    Ok(GreedySamplerProbeResult {
        len: logits.len(),
        metal_token,
        metal_logit,
        cpu_token,
        cpu_logit,
    })
}

pub fn run_kv_cache_append_probe() -> Result<KvCacheAppendProbeResult> {
    let seq = 4_usize;
    let head_dim = 4_usize;
    let position = 2_usize;
    let k_new = f16_vec(&[0.5, -1.0, 1.5, 0.25]);
    let v_new = f16_vec(&[-0.75, 0.5, 2.0, -1.25]);
    let initial = f16_vec(&[-9.0; 16]);

    let (metal_k_cache_f16, metal_v_cache_f16) =
        run_kv_cache_append_f16(&initial, &initial, &k_new, &v_new, seq, head_dim, position)?;
    let (cpu_k_cache_f16, cpu_v_cache_f16) =
        cpu_kv_cache_append_f16(&initial, &initial, &k_new, &v_new, seq, head_dim, position);

    let metal_k_cache = f16_to_f32_vec(&metal_k_cache_f16);
    let metal_v_cache = f16_to_f32_vec(&metal_v_cache_f16);
    let cpu_k_cache = f16_to_f32_vec(&cpu_k_cache_f16);
    let cpu_v_cache = f16_to_f32_vec(&cpu_v_cache_f16);
    let max_abs_error = metal_k_cache
        .iter()
        .chain(metal_v_cache.iter())
        .zip(cpu_k_cache.iter().chain(cpu_v_cache.iter()))
        .map(|(metal, cpu)| (metal - cpu).abs())
        .fold(0.0_f32, f32::max);

    Ok(KvCacheAppendProbeResult {
        seq,
        head_dim,
        position,
        metal_k_cache,
        metal_v_cache,
        cpu_k_cache,
        cpu_v_cache,
        max_abs_error,
    })
}

pub fn run_qwen_15b_smoke_bench() -> Result<KernelBenchmarkSuite> {
    let device = MetalDeviceInfo::system_default()?;
    let mut results = Vec::new();

    results.push(time_kernel(
        "matmul_f16_1x1536x1536",
        "m=1,n=1536,k=1536",
        2 * 1536 * 1536,
        || {
            let a = deterministic_f16_vec(1536, 0xA11CE);
            let b = deterministic_f16_vec(1536 * 1536, 0xB0B);
            let output = run_f16_matmul(&a, &b, 1, 1536, 1536)?;
            assert_eq!(output.len(), 1536);
            Ok(())
        },
    )?);

    results.push(time_kernel(
        "matvec_f16_1536x1536_decode",
        "n=1536,k=1536",
        2 * 1536 * 1536,
        || {
            let x = deterministic_f16_vec(1536, 0xDEC0DE);
            let w = deterministic_f16_vec(1536 * 1536, 0xDEC0DF);
            let output = run_f16_matvec(&x, &w, 1536, 1536)?;
            assert_eq!(output.len(), 1536);
            Ok(())
        },
    )?);

    results.push(time_kernel(
        "rmsnorm_f16_1x1536",
        "rows=1,cols=1536",
        1536 * 4,
        || {
            let x = deterministic_f16_vec(1536, 0xC0FFEE);
            let weight = deterministic_f16_vec(1536, 0xD00D);
            let output = run_rmsnorm_f16_to_f32(&x, &weight, 1, 1536, 1e-6)?;
            assert_eq!(output.len(), 1536);
            Ok(())
        },
    )?);

    results.push(time_kernel(
        "rope_f16_1x128",
        "rows=1,dims=128",
        128 * 6,
        || {
            let x = deterministic_f16_vec(128, 0xFA57);
            let cos = deterministic_unit_vec(128, 0xC05);
            let sin = deterministic_unit_vec(128, 0x51A);
            let output = run_rope_f16_to_f32(&x, &cos, &sin, 1, 128)?;
            assert_eq!(output.len(), 128);
            Ok(())
        },
    )?);

    results.push(time_kernel(
        "softmax_f32_1x1024",
        "rows=1,cols=1024",
        1024 * 5,
        || {
            let x = deterministic_f32_vec(1024, 0x50F7);
            let output = run_softmax_f32(&x, 1, 1024)?;
            assert_eq!(output.len(), 1024);
            Ok(())
        },
    )?);

    results.push(time_kernel(
        "attention_decode_f16_seq128_dim128",
        "seq=128,head_dim=128",
        2 * 128 * 128 + 128 * 128,
        || {
            let q = deterministic_f16_vec(128, 0x1234);
            let k = deterministic_f16_vec(128 * 128, 0x5678);
            let v = deterministic_f16_vec(128 * 128, 0x9ABC);
            let output =
                run_attention_decode_f16_to_f32(&q, &k, &v, 128, 128, 128_f32.sqrt().recip())?;
            assert_eq!(output.len(), 128);
            Ok(())
        },
    )?);

    results.push(time_kernel(
        "greedy_argmax_f32_vocab151936",
        "vocab=151936",
        151_936,
        || {
            let logits = deterministic_f32_vec(151_936, 0x515151);
            let _ = run_greedy_argmax_f32(&logits)?;
            Ok(())
        },
    )?);

    results.push(time_kernel(
        "kv_cache_append_f16_seq4096_dim128",
        "seq=4096,head_dim=128,position=2048",
        128 * 2,
        || {
            let k_cache = deterministic_f16_vec(4096 * 128, 0xAA01);
            let v_cache = deterministic_f16_vec(4096 * 128, 0xAA02);
            let k_new = deterministic_f16_vec(128, 0xAA03);
            let v_new = deterministic_f16_vec(128, 0xAA04);
            let (k_out, v_out) =
                run_kv_cache_append_f16(&k_cache, &v_cache, &k_new, &v_new, 4096, 128, 2048)?;
            assert_eq!(k_out.len(), 4096 * 128);
            assert_eq!(v_out.len(), 4096 * 128);
            Ok(())
        },
    )?);

    Ok(KernelBenchmarkSuite {
        device: device.name,
        preset: "qwen2.5-1.5b-smoke".to_string(),
        note: "Cold timings include current per-call Metal shader compilation; next step is persistent pipeline caching for real steady-state numbers.".to_string(),
        results,
    })
}

pub fn run_qwen_15b_hot_bench(iterations: u32) -> Result<HotKernelBenchmarkSuite> {
    if iterations == 0 {
        return Err(TinyAgentError::Configuration(
            "hot benchmark iterations must be greater than zero".to_string(),
        ));
    }

    let device = MetalDeviceInfo::system_default()?;
    let matmul = run_hot_f16_matmul_bench("matmul_f16_1x1536x1536_hot", 1, 1536, 1536, iterations)?;
    let matvec =
        run_hot_f16_matvec_bench("matvec_f16_1536x1536_decode_hot", 1536, 1536, iterations)?;

    Ok(HotKernelBenchmarkSuite {
        device: device.name,
        preset: "qwen2.5-1.5b-hot".to_string(),
        iterations,
        results: vec![matmul, matvec],
    })
}

pub fn run_hot_f16_matmul_bench(
    name: impl Into<String>,
    m: usize,
    n: usize,
    k: usize,
    iterations: u32,
) -> Result<HotKernelBenchmarkResult> {
    let name = name.into();
    let device = Device::system_default().ok_or_else(|| {
        TinyAgentError::Configuration(
            "no default Metal device found; TinyEngine requires Apple Silicon or a Metal-capable GPU"
                .to_string(),
        )
    })?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;

        #define TILE 16

        kernel void matmul_f16_f32_tiled(
            device const half* a [[buffer(0)]],
            device const half* b [[buffer(1)]],
            device float* out [[buffer(2)]],
            constant uint* dims [[buffer(3)]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgid [[threadgroup_position_in_grid]]
        ) {
            const uint m = dims[0];
            const uint n = dims[1];
            const uint kdim = dims[2];
            const uint row = tgid.y * TILE + tid.y;
            const uint col = tgid.x * TILE + tid.x;

            threadgroup half tile_a[TILE][TILE];
            threadgroup half tile_b[TILE][TILE];

            float acc = 0.0f;
            for (uint tile = 0; tile < (kdim + TILE - 1) / TILE; tile++) {
                const uint a_col = tile * TILE + tid.x;
                const uint b_row = tile * TILE + tid.y;

                tile_a[tid.y][tid.x] = (row < m && a_col < kdim)
                    ? a[row * kdim + a_col]
                    : half(0.0h);
                tile_b[tid.y][tid.x] = (b_row < kdim && col < n)
                    ? b[b_row * n + col]
                    : half(0.0h);

                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint i = 0; i < TILE; i++) {
                    acc += float(tile_a[tid.y][i]) * float(tile_b[i][tid.x]);
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (row < m && col < n) {
                out[row * n + col] = acc;
            }
        }
    "#;

    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to compile hot matmul kernel: {error}"))
        })?;
    let function = library
        .get_function("matmul_f16_f32_tiled", None)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to load hot matmul kernel: {error}"))
        })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to create hot matmul pipeline: {error}"))
        })?;
    let queue = device.new_command_queue();

    let a = deterministic_f16_vec(m * k, 0xA11CE);
    let b = deterministic_f16_vec(k * n, 0xB0B);
    let buffer_a = device.new_buffer_with_data(
        a.as_ptr().cast(),
        std::mem::size_of_val(a.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_b = device.new_buffer_with_data(
        b.as_ptr().cast(),
        std::mem::size_of_val(b.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_byte_len = (m * n * std::mem::size_of::<f32>()) as u64;
    let buffer_out = device.new_buffer(out_byte_len, MTLResourceOptions::StorageModeShared);
    let dims = [m as u32, n as u32, k as u32];
    let buffer_dims = device.new_buffer_with_data(
        dims.as_ptr().cast(),
        std::mem::size_of_val(&dims) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    dispatch_f16_matmul_once(
        &queue,
        &pipeline,
        &buffer_a,
        &buffer_b,
        &buffer_out,
        &buffer_dims,
        m,
        n,
    );

    let started = Instant::now();
    for _ in 0..iterations {
        dispatch_f16_matmul_once(
            &queue,
            &pipeline,
            &buffer_a,
            &buffer_b,
            &buffer_out,
            &buffer_dims,
            m,
            n,
        );
    }
    let elapsed = started.elapsed();
    let total_elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let avg_elapsed_ms = total_elapsed_ms / iterations as f64;
    let logical_ops_per_iter = (2 * m * n * k) as u64;
    let throughput_gops =
        (logical_ops_per_iter * iterations as u64) as f64 / elapsed.as_secs_f64() / 1_000_000_000.0;

    let first_value = unsafe { *buffer_out.contents().cast::<f32>() };
    if !first_value.is_finite() {
        return Err(TinyAgentError::Backend(
            "hot matmul produced a non-finite output".to_string(),
        ));
    }

    Ok(HotKernelBenchmarkResult {
        name,
        shape: format!("m={m},n={n},k={k}"),
        logical_ops_per_iter,
        total_elapsed_ms,
        avg_elapsed_ms,
        throughput_gops,
    })
}

pub fn run_hot_f16_matvec_bench(
    name: impl Into<String>,
    n: usize,
    k: usize,
    iterations: u32,
) -> Result<HotKernelBenchmarkResult> {
    if iterations == 0 {
        return Err(TinyAgentError::Configuration(
            "hot matvec benchmark iterations must be greater than zero".to_string(),
        ));
    }

    let name = name.into();
    let device = Device::system_default().ok_or_else(|| {
        TinyAgentError::Configuration(
            "no default Metal device found; TinyEngine requires Apple Silicon or a Metal-capable GPU"
                .to_string(),
        )
    })?;

    let library = device
        .new_library_with_source(f16_matvec_source(), &CompileOptions::new())
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to compile hot matvec kernel: {error}"))
        })?;
    let function = library
        .get_function("matvec_f16_f32", None)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to load hot matvec kernel: {error}"))
        })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to create hot matvec pipeline: {error}"))
        })?;
    let queue = device.new_command_queue();

    let x = deterministic_f16_vec(k, 0xDEC0DE);
    let w = deterministic_f16_vec(k * n, 0xDEC0DF);
    let buffer_x = device.new_buffer_with_data(
        x.as_ptr().cast(),
        std::mem::size_of_val(x.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_w = device.new_buffer_with_data(
        w.as_ptr().cast(),
        std::mem::size_of_val(w.as_slice()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_byte_len = (n * std::mem::size_of::<f32>()) as u64;
    let buffer_out = device.new_buffer(out_byte_len, MTLResourceOptions::StorageModeShared);
    let dims = [n as u32, k as u32];
    let buffer_dims = device.new_buffer_with_data(
        dims.as_ptr().cast(),
        std::mem::size_of_val(&dims) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    dispatch_f16_matvec_once(
        &queue,
        &pipeline,
        &buffer_x,
        &buffer_w,
        &buffer_out,
        &buffer_dims,
        n,
    );

    let started = Instant::now();
    for _ in 0..iterations {
        dispatch_f16_matvec_once(
            &queue,
            &pipeline,
            &buffer_x,
            &buffer_w,
            &buffer_out,
            &buffer_dims,
            n,
        );
    }
    let elapsed = started.elapsed();
    let total_elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let avg_elapsed_ms = total_elapsed_ms / iterations as f64;
    let logical_ops_per_iter = (2 * n * k) as u64;
    let throughput_gops =
        (logical_ops_per_iter * iterations as u64) as f64 / elapsed.as_secs_f64() / 1_000_000_000.0;

    let first_value = unsafe { *buffer_out.contents().cast::<f32>() };
    if !first_value.is_finite() {
        return Err(TinyAgentError::Backend(
            "hot matvec produced a non-finite output".to_string(),
        ));
    }

    Ok(HotKernelBenchmarkResult {
        name,
        shape: format!("n={n},k={k}"),
        logical_ops_per_iter,
        total_elapsed_ms,
        avg_elapsed_ms,
        throughput_gops,
    })
}

fn f16_matvec_source() -> &'static str {
    r#"
        #include <metal_stdlib>
        using namespace metal;

        #define COL_TILE 16
        #define REDUCE_THREADS 16

        kernel void matvec_f16_f32(
            device const half* x [[buffer(0)]],
            device const half* w [[buffer(1)]],
            device float* out [[buffer(2)]],
            constant uint* dims [[buffer(3)]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgid [[threadgroup_position_in_grid]]
        ) {
            const uint n = dims[0];
            const uint kdim = dims[1];
            const uint col = tgid.x * COL_TILE + tid.x;
            const uint lane = tid.y;

            threadgroup float partial[REDUCE_THREADS][COL_TILE];
            float acc = 0.0f;
            if (col < n) {
                for (uint i = lane; i < kdim; i += REDUCE_THREADS) {
                    acc += float(x[i]) * float(w[i * n + col]);
                }
            }
            partial[lane][tid.x] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = REDUCE_THREADS / 2; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    partial[lane][tid.x] += partial[lane + stride][tid.x];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (lane == 0 && col < n) {
                out[col] = partial[0][tid.x];
            }
        }
    "#
}

fn dispatch_f16_matmul_once(
    queue: &metal::CommandQueueRef,
    pipeline: &metal::ComputePipelineStateRef,
    buffer_a: &metal::BufferRef,
    buffer_b: &metal::BufferRef,
    buffer_out: &metal::BufferRef,
    buffer_dims: &metal::BufferRef,
    m: usize,
    n: usize,
) {
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buffer_a), 0);
    encoder.set_buffer(1, Some(buffer_b), 0);
    encoder.set_buffer(2, Some(buffer_out), 0);
    encoder.set_buffer(3, Some(buffer_dims), 0);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: n.div_ceil(16) as u64,
            height: m.div_ceil(16) as u64,
            depth: 1,
        },
        MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

fn dispatch_f16_matvec_once(
    queue: &metal::CommandQueueRef,
    pipeline: &metal::ComputePipelineStateRef,
    buffer_x: &metal::BufferRef,
    buffer_w: &metal::BufferRef,
    buffer_out: &metal::BufferRef,
    buffer_dims: &metal::BufferRef,
    n: usize,
) {
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buffer_x), 0);
    encoder.set_buffer(1, Some(buffer_w), 0);
    encoder.set_buffer(2, Some(buffer_out), 0);
    encoder.set_buffer(3, Some(buffer_dims), 0);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: n.div_ceil(16) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

pub fn run_f16_matmul(a: &[f16], b: &[f16], m: usize, n: usize, k: usize) -> Result<Vec<f32>> {
    if a.len() != m * k {
        return Err(TinyAgentError::Configuration(format!(
            "lhs length {} does not match m*k {}",
            a.len(),
            m * k
        )));
    }
    if b.len() != k * n {
        return Err(TinyAgentError::Configuration(format!(
            "rhs length {} does not match k*n {}",
            b.len(),
            k * n
        )));
    }

    let device = Device::system_default().ok_or_else(|| {
        TinyAgentError::Configuration(
            "no default Metal device found; TinyEngine requires Apple Silicon or a Metal-capable GPU"
                .to_string(),
        )
    })?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;

        #define TILE 16

        kernel void matmul_f16_f32_tiled(
            device const half* a [[buffer(0)]],
            device const half* b [[buffer(1)]],
            device float* out [[buffer(2)]],
            constant uint* dims [[buffer(3)]],
            uint2 tid [[thread_position_in_threadgroup]],
            uint2 tgid [[threadgroup_position_in_grid]]
        ) {
            const uint m = dims[0];
            const uint n = dims[1];
            const uint kdim = dims[2];
            const uint row = tgid.y * TILE + tid.y;
            const uint col = tgid.x * TILE + tid.x;

            threadgroup half tile_a[TILE][TILE];
            threadgroup half tile_b[TILE][TILE];

            float acc = 0.0f;
            for (uint tile = 0; tile < (kdim + TILE - 1) / TILE; tile++) {
                const uint a_col = tile * TILE + tid.x;
                const uint b_row = tile * TILE + tid.y;

                tile_a[tid.y][tid.x] = (row < m && a_col < kdim)
                    ? a[row * kdim + a_col]
                    : half(0.0h);
                tile_b[tid.y][tid.x] = (b_row < kdim && col < n)
                    ? b[b_row * n + col]
                    : half(0.0h);

                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint i = 0; i < TILE; i++) {
                    acc += float(tile_a[tid.y][i]) * float(tile_b[i][tid.x]);
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (row < m && col < n) {
                out[row * n + col] = acc;
            }
        }
    "#;

    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to compile f16 matmul kernel: {error}"))
        })?;
    let function = library
        .get_function("matmul_f16_f32_tiled", None)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to load f16 matmul kernel: {error}"))
        })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to create f16 matmul pipeline: {error}"))
        })?;
    let queue = device.new_command_queue();

    let buffer_a = device.new_buffer_with_data(
        a.as_ptr().cast(),
        std::mem::size_of_val(a) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_b = device.new_buffer_with_data(
        b.as_ptr().cast(),
        std::mem::size_of_val(b) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_byte_len = (m * n * std::mem::size_of::<f32>()) as u64;
    let buffer_out = device.new_buffer(out_byte_len, MTLResourceOptions::StorageModeShared);
    let dims = [m as u32, n as u32, k as u32];
    let buffer_dims = device.new_buffer_with_data(
        dims.as_ptr().cast(),
        std::mem::size_of_val(&dims) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&buffer_a), 0);
    encoder.set_buffer(1, Some(&buffer_b), 0);
    encoder.set_buffer(2, Some(&buffer_out), 0);
    encoder.set_buffer(3, Some(&buffer_dims), 0);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: n.div_ceil(16) as u64,
            height: m.div_ceil(16) as u64,
            depth: 1,
        },
        MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let ptr = buffer_out.contents().cast::<f32>();
    let output = unsafe { std::slice::from_raw_parts(ptr, m * n) }.to_vec();
    Ok(output)
}

pub fn run_f16_matvec(x: &[f16], w: &[f16], n: usize, k: usize) -> Result<Vec<f32>> {
    if x.len() != k {
        return Err(TinyAgentError::Configuration(format!(
            "input length {} does not match k {k}",
            x.len()
        )));
    }
    if w.len() != k * n {
        return Err(TinyAgentError::Configuration(format!(
            "weight length {} does not match k*n {}",
            w.len(),
            k * n
        )));
    }

    let device = Device::system_default().ok_or_else(|| {
        TinyAgentError::Configuration(
            "no default Metal device found; TinyEngine requires Apple Silicon or a Metal-capable GPU"
                .to_string(),
        )
    })?;

    let library = device
        .new_library_with_source(f16_matvec_source(), &CompileOptions::new())
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to compile f16 matvec kernel: {error}"))
        })?;
    let function = library
        .get_function("matvec_f16_f32", None)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to load f16 matvec kernel: {error}"))
        })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to create f16 matvec pipeline: {error}"))
        })?;
    let queue = device.new_command_queue();

    let buffer_x = device.new_buffer_with_data(
        x.as_ptr().cast(),
        std::mem::size_of_val(x) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_w = device.new_buffer_with_data(
        w.as_ptr().cast(),
        std::mem::size_of_val(w) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_byte_len = (n * std::mem::size_of::<f32>()) as u64;
    let buffer_out = device.new_buffer(out_byte_len, MTLResourceOptions::StorageModeShared);
    let dims = [n as u32, k as u32];
    let buffer_dims = device.new_buffer_with_data(
        dims.as_ptr().cast(),
        std::mem::size_of_val(&dims) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    dispatch_f16_matvec_once(
        &queue,
        &pipeline,
        &buffer_x,
        &buffer_w,
        &buffer_out,
        &buffer_dims,
        n,
    );

    let ptr = buffer_out.contents().cast::<f32>();
    let output = unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec();
    Ok(output)
}

pub fn run_rmsnorm_f16_to_f32(
    x: &[f16],
    weight: &[f16],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if x.len() != rows * cols {
        return Err(TinyAgentError::Configuration(format!(
            "input length {} does not match rows*cols {}",
            x.len(),
            rows * cols
        )));
    }
    if weight.len() != cols {
        return Err(TinyAgentError::Configuration(format!(
            "weight length {} does not match cols {cols}",
            weight.len()
        )));
    }

    let device = Device::system_default().ok_or_else(|| {
        TinyAgentError::Configuration(
            "no default Metal device found; TinyEngine requires Apple Silicon or a Metal-capable GPU"
                .to_string(),
        )
    })?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;

        #define RMS_THREADS 256

        kernel void rmsnorm_f16_f32(
            device const half* x [[buffer(0)]],
            device const half* weight [[buffer(1)]],
            device float* out [[buffer(2)]],
            constant uint* dims [[buffer(3)]],
            constant float& eps [[buffer(4)]],
            uint tid [[thread_index_in_threadgroup]],
            uint3 tgid [[threadgroup_position_in_grid]]
        ) {
            const uint rows = dims[0];
            const uint cols = dims[1];
            const uint row = tgid.x;
            if (row >= rows) {
                return;
            }

            threadgroup float partial[RMS_THREADS];

            float sumsq = 0.0f;
            for (uint col = tid; col < cols; col += RMS_THREADS) {
                const float value = float(x[row * cols + col]);
                sumsq += value * value;
            }
            partial[tid] = sumsq;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = RMS_THREADS / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    partial[tid] += partial[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const float inv_rms = rsqrt(partial[0] / float(cols) + eps);
            for (uint col = tid; col < cols; col += RMS_THREADS) {
                out[row * cols + col] = float(x[row * cols + col]) * float(weight[col]) * inv_rms;
            }
        }
    "#;

    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to compile RMSNorm kernel: {error}"))
        })?;
    let function = library
        .get_function("rmsnorm_f16_f32", None)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to load RMSNorm kernel: {error}"))
        })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to create RMSNorm pipeline: {error}"))
        })?;
    let queue = device.new_command_queue();

    let buffer_x = device.new_buffer_with_data(
        x.as_ptr().cast(),
        std::mem::size_of_val(x) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_weight = device.new_buffer_with_data(
        weight.as_ptr().cast(),
        std::mem::size_of_val(weight) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_byte_len = (rows * cols * std::mem::size_of::<f32>()) as u64;
    let buffer_out = device.new_buffer(out_byte_len, MTLResourceOptions::StorageModeShared);
    let dims = [rows as u32, cols as u32];
    let buffer_dims = device.new_buffer_with_data(
        dims.as_ptr().cast(),
        std::mem::size_of_val(&dims) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_eps = device.new_buffer_with_data(
        (&eps as *const f32).cast(),
        std::mem::size_of::<f32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&buffer_x), 0);
    encoder.set_buffer(1, Some(&buffer_weight), 0);
    encoder.set_buffer(2, Some(&buffer_out), 0);
    encoder.set_buffer(3, Some(&buffer_dims), 0);
    encoder.set_buffer(4, Some(&buffer_eps), 0);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: rows as u64,
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

    let ptr = buffer_out.contents().cast::<f32>();
    let output = unsafe { std::slice::from_raw_parts(ptr, rows * cols) }.to_vec();
    Ok(output)
}

pub fn run_rope_f16_to_f32(
    x: &[f16],
    cos: &[f32],
    sin: &[f32],
    rows: usize,
    dims: usize,
) -> Result<Vec<f32>> {
    if dims % 2 != 0 {
        return Err(TinyAgentError::Configuration(format!(
            "RoPE dims must be even, got {dims}"
        )));
    }
    if x.len() != rows * dims {
        return Err(TinyAgentError::Configuration(format!(
            "input length {} does not match rows*dims {}",
            x.len(),
            rows * dims
        )));
    }
    if cos.len() != dims || sin.len() != dims {
        return Err(TinyAgentError::Configuration(format!(
            "cos/sin lengths must match dims {dims}, got {}/{}",
            cos.len(),
            sin.len()
        )));
    }

    let device = Device::system_default().ok_or_else(|| {
        TinyAgentError::Configuration(
            "no default Metal device found; TinyEngine requires Apple Silicon or a Metal-capable GPU"
                .to_string(),
        )
    })?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void rope_qwen_f16_f32(
            device const half* x [[buffer(0)]],
            device const float* cos [[buffer(1)]],
            device const float* sin [[buffer(2)]],
            device float* out [[buffer(3)]],
            constant uint* dims [[buffer(4)]],
            uint id [[thread_position_in_grid]]
        ) {
            const uint rows = dims[0];
            const uint dim = dims[1];
            const uint total = rows * dim;
            if (id >= total) {
                return;
            }

            const uint col = id % dim;
            const uint row = id / dim;
            const uint half_dim = dim / 2;
            const uint pair_col = (col < half_dim) ? col + half_dim : col - half_dim;
            const float rotated = (col < half_dim)
                ? -float(x[row * dim + pair_col])
                : float(x[row * dim + pair_col]);

            out[id] = float(x[id]) * cos[col] + rotated * sin[col];
        }
    "#;

    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to compile RoPE kernel: {error}"))
        })?;
    let function = library
        .get_function("rope_qwen_f16_f32", None)
        .map_err(|error| TinyAgentError::Backend(format!("failed to load RoPE kernel: {error}")))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to create RoPE pipeline: {error}"))
        })?;
    let queue = device.new_command_queue();

    let buffer_x = device.new_buffer_with_data(
        x.as_ptr().cast(),
        std::mem::size_of_val(x) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_cos = device.new_buffer_with_data(
        cos.as_ptr().cast(),
        std::mem::size_of_val(cos) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_sin = device.new_buffer_with_data(
        sin.as_ptr().cast(),
        std::mem::size_of_val(sin) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_byte_len = (rows * dims * std::mem::size_of::<f32>()) as u64;
    let buffer_out = device.new_buffer(out_byte_len, MTLResourceOptions::StorageModeShared);
    let kernel_dims = [rows as u32, dims as u32];
    let buffer_dims = device.new_buffer_with_data(
        kernel_dims.as_ptr().cast(),
        std::mem::size_of_val(&kernel_dims) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&buffer_x), 0);
    encoder.set_buffer(1, Some(&buffer_cos), 0);
    encoder.set_buffer(2, Some(&buffer_sin), 0);
    encoder.set_buffer(3, Some(&buffer_out), 0);
    encoder.set_buffer(4, Some(&buffer_dims), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: (rows * dims) as u64,
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

    let ptr = buffer_out.contents().cast::<f32>();
    let output = unsafe { std::slice::from_raw_parts(ptr, rows * dims) }.to_vec();
    Ok(output)
}

pub fn run_swiglu_f32(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    if gate.len() != up.len() {
        return Err(TinyAgentError::Configuration(format!(
            "gate/up lengths must match, got {}/{}",
            gate.len(),
            up.len()
        )));
    }

    let device = Device::system_default().ok_or_else(|| {
        TinyAgentError::Configuration(
            "no default Metal device found; TinyEngine requires Apple Silicon or a Metal-capable GPU"
                .to_string(),
        )
    })?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void swiglu_f32(
            device const float* gate [[buffer(0)]],
            device const float* up [[buffer(1)]],
            device float* out [[buffer(2)]],
            constant uint& len [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id >= len) {
                return;
            }

            const float g = gate[id];
            const float silu = g / (1.0f + exp(-g));
            out[id] = silu * up[id];
        }
    "#;

    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to compile SwiGLU kernel: {error}"))
        })?;
    let function = library.get_function("swiglu_f32", None).map_err(|error| {
        TinyAgentError::Backend(format!("failed to load SwiGLU kernel: {error}"))
    })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to create SwiGLU pipeline: {error}"))
        })?;
    let queue = device.new_command_queue();

    let buffer_gate = device.new_buffer_with_data(
        gate.as_ptr().cast(),
        std::mem::size_of_val(gate) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_up = device.new_buffer_with_data(
        up.as_ptr().cast(),
        std::mem::size_of_val(up) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_byte_len = std::mem::size_of_val(gate) as u64;
    let buffer_out = device.new_buffer(out_byte_len, MTLResourceOptions::StorageModeShared);
    let len = gate.len() as u32;
    let buffer_len = device.new_buffer_with_data(
        (&len as *const u32).cast(),
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&buffer_gate), 0);
    encoder.set_buffer(1, Some(&buffer_up), 0);
    encoder.set_buffer(2, Some(&buffer_out), 0);
    encoder.set_buffer(3, Some(&buffer_len), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: gate.len() as u64,
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

    let ptr = buffer_out.contents().cast::<f32>();
    let output = unsafe { std::slice::from_raw_parts(ptr, gate.len()) }.to_vec();
    Ok(output)
}

pub fn run_softmax_f32(x: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
    if x.len() != rows * cols {
        return Err(TinyAgentError::Configuration(format!(
            "input length {} does not match rows*cols {}",
            x.len(),
            rows * cols
        )));
    }

    let device = Device::system_default().ok_or_else(|| {
        TinyAgentError::Configuration(
            "no default Metal device found; TinyEngine requires Apple Silicon or a Metal-capable GPU"
                .to_string(),
        )
    })?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;

        #define SOFTMAX_THREADS 256

        kernel void softmax_f32(
            device const float* x [[buffer(0)]],
            device float* out [[buffer(1)]],
            constant uint* dims [[buffer(2)]],
            uint tid [[thread_index_in_threadgroup]],
            uint3 tgid [[threadgroup_position_in_grid]]
        ) {
            const uint rows = dims[0];
            const uint cols = dims[1];
            const uint row = tgid.x;
            if (row >= rows) {
                return;
            }

            threadgroup float partial[SOFTMAX_THREADS];

            float local_max = -3.4028234663852886e38f;
            for (uint col = tid; col < cols; col += SOFTMAX_THREADS) {
                local_max = max(local_max, x[row * cols + col]);
            }
            partial[tid] = local_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = SOFTMAX_THREADS / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    partial[tid] = max(partial[tid], partial[tid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const float row_max = partial[0];

            float local_sum = 0.0f;
            for (uint col = tid; col < cols; col += SOFTMAX_THREADS) {
                local_sum += exp(x[row * cols + col] - row_max);
            }
            partial[tid] = local_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = SOFTMAX_THREADS / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    partial[tid] += partial[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const float inv_sum = 1.0f / partial[0];

            for (uint col = tid; col < cols; col += SOFTMAX_THREADS) {
                out[row * cols + col] = exp(x[row * cols + col] - row_max) * inv_sum;
            }
        }
    "#;

    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to compile softmax kernel: {error}"))
        })?;
    let function = library.get_function("softmax_f32", None).map_err(|error| {
        TinyAgentError::Backend(format!("failed to load softmax kernel: {error}"))
    })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to create softmax pipeline: {error}"))
        })?;
    let queue = device.new_command_queue();

    let buffer_x = device.new_buffer_with_data(
        x.as_ptr().cast(),
        std::mem::size_of_val(x) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_byte_len = std::mem::size_of_val(x) as u64;
    let buffer_out = device.new_buffer(out_byte_len, MTLResourceOptions::StorageModeShared);
    let dims = [rows as u32, cols as u32];
    let buffer_dims = device.new_buffer_with_data(
        dims.as_ptr().cast(),
        std::mem::size_of_val(&dims) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&buffer_x), 0);
    encoder.set_buffer(1, Some(&buffer_out), 0);
    encoder.set_buffer(2, Some(&buffer_dims), 0);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: rows as u64,
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

    let ptr = buffer_out.contents().cast::<f32>();
    let output = unsafe { std::slice::from_raw_parts(ptr, rows * cols) }.to_vec();
    Ok(output)
}

pub fn run_attention_decode_f16_to_f32(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    seq: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    if seq > 256 {
        return Err(TinyAgentError::Configuration(format!(
            "attention decode scaffold supports seq <= 256, got {seq}"
        )));
    }
    if q.len() != head_dim {
        return Err(TinyAgentError::Configuration(format!(
            "query length {} does not match head_dim {head_dim}",
            q.len()
        )));
    }
    if k.len() != seq * head_dim || v.len() != seq * head_dim {
        return Err(TinyAgentError::Configuration(format!(
            "key/value lengths must match seq*head_dim {}, got {}/{}",
            seq * head_dim,
            k.len(),
            v.len()
        )));
    }

    let device = Device::system_default().ok_or_else(|| {
        TinyAgentError::Configuration(
            "no default Metal device found; TinyEngine requires Apple Silicon or a Metal-capable GPU"
                .to_string(),
        )
    })?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;

        #define ATTN_THREADS 256

        kernel void attention_decode_f16_f32(
            device const half* q [[buffer(0)]],
            device const half* k [[buffer(1)]],
            device const half* v [[buffer(2)]],
            device float* out [[buffer(3)]],
            constant uint* dims [[buffer(4)]],
            constant float& scale [[buffer(5)]],
            uint tid [[thread_index_in_threadgroup]]
        ) {
            const uint seq = dims[0];
            const uint head_dim = dims[1];

            threadgroup float scores[ATTN_THREADS];

            float score = -3.4028234663852886e38f;
            if (tid < seq) {
                float dot = 0.0f;
                for (uint col = 0; col < head_dim; col++) {
                    dot += float(q[col]) * float(k[tid * head_dim + col]);
                }
                score = dot * scale;
            }
            scores[tid] = score;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = ATTN_THREADS / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    scores[tid] = max(scores[tid], scores[tid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const float max_score = scores[0];

            float weight = 0.0f;
            if (tid < seq) {
                weight = exp(score - max_score);
            }
            scores[tid] = weight;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = ATTN_THREADS / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    scores[tid] += scores[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const float inv_sum = 1.0f / scores[0];

            if (tid < seq) {
                scores[tid] = weight * inv_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint col = tid; col < head_dim; col += ATTN_THREADS) {
                float acc = 0.0f;
                for (uint row = 0; row < seq; row++) {
                    acc += scores[row] * float(v[row * head_dim + col]);
                }
                out[col] = acc;
            }
        }
    "#;

    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .map_err(|error| {
            TinyAgentError::Backend(format!(
                "failed to compile attention decode kernel: {error}"
            ))
        })?;
    let function = library
        .get_function("attention_decode_f16_f32", None)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to load attention decode kernel: {error}"))
        })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| {
            TinyAgentError::Backend(format!(
                "failed to create attention decode pipeline: {error}"
            ))
        })?;
    let queue = device.new_command_queue();

    let buffer_q = device.new_buffer_with_data(
        q.as_ptr().cast(),
        std::mem::size_of_val(q) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_k = device.new_buffer_with_data(
        k.as_ptr().cast(),
        std::mem::size_of_val(k) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_v = device.new_buffer_with_data(
        v.as_ptr().cast(),
        std::mem::size_of_val(v) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_byte_len = (head_dim * std::mem::size_of::<f32>()) as u64;
    let buffer_out = device.new_buffer(out_byte_len, MTLResourceOptions::StorageModeShared);
    let dims = [seq as u32, head_dim as u32];
    let buffer_dims = device.new_buffer_with_data(
        dims.as_ptr().cast(),
        std::mem::size_of_val(&dims) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_scale = device.new_buffer_with_data(
        (&scale as *const f32).cast(),
        std::mem::size_of::<f32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&buffer_q), 0);
    encoder.set_buffer(1, Some(&buffer_k), 0);
    encoder.set_buffer(2, Some(&buffer_v), 0);
    encoder.set_buffer(3, Some(&buffer_out), 0);
    encoder.set_buffer(4, Some(&buffer_dims), 0);
    encoder.set_buffer(5, Some(&buffer_scale), 0);
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

    let ptr = buffer_out.contents().cast::<f32>();
    let output = unsafe { std::slice::from_raw_parts(ptr, head_dim) }.to_vec();
    Ok(output)
}

pub fn run_greedy_argmax_f32(logits: &[f32]) -> Result<(u32, f32)> {
    if logits.is_empty() {
        return Err(TinyAgentError::Configuration(
            "cannot sample from empty logits".to_string(),
        ));
    }

    let device = Device::system_default().ok_or_else(|| {
        TinyAgentError::Configuration(
            "no default Metal device found; TinyEngine requires Apple Silicon or a Metal-capable GPU"
                .to_string(),
        )
    })?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;

        #define ARGMAX_THREADS 256

        struct ArgMaxPair {
            float value;
            uint index;
        };

        static inline ArgMaxPair better_pair(ArgMaxPair a, ArgMaxPair b) {
            if (b.value > a.value || (b.value == a.value && b.index < a.index)) {
                return b;
            }
            return a;
        }

        kernel void greedy_argmax_f32(
            device const float* logits [[buffer(0)]],
            device uint* out_index [[buffer(1)]],
            device float* out_value [[buffer(2)]],
            constant uint& len [[buffer(3)]],
            uint tid [[thread_index_in_threadgroup]]
        ) {
            threadgroup ArgMaxPair partial[ARGMAX_THREADS];

            ArgMaxPair best;
            best.value = -3.4028234663852886e38f;
            best.index = 0;

            for (uint i = tid; i < len; i += ARGMAX_THREADS) {
                ArgMaxPair candidate;
                candidate.value = logits[i];
                candidate.index = i;
                best = better_pair(best, candidate);
            }
            partial[tid] = best;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = ARGMAX_THREADS / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    partial[tid] = better_pair(partial[tid], partial[tid + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                out_index[0] = partial[0].index;
                out_value[0] = partial[0].value;
            }
        }
    "#;

    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to compile greedy sampler kernel: {error}"))
        })?;
    let function = library
        .get_function("greedy_argmax_f32", None)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to load greedy sampler kernel: {error}"))
        })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to create greedy sampler pipeline: {error}"))
        })?;
    let queue = device.new_command_queue();

    let buffer_logits = device.new_buffer_with_data(
        logits.as_ptr().cast(),
        std::mem::size_of_val(logits) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_index = device.new_buffer(
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_value = device.new_buffer(
        std::mem::size_of::<f32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let len = logits.len() as u32;
    let buffer_len = device.new_buffer_with_data(
        (&len as *const u32).cast(),
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&buffer_logits), 0);
    encoder.set_buffer(1, Some(&buffer_index), 0);
    encoder.set_buffer(2, Some(&buffer_value), 0);
    encoder.set_buffer(3, Some(&buffer_len), 0);
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

    let token = unsafe { *buffer_index.contents().cast::<u32>() };
    let logit = unsafe { *buffer_value.contents().cast::<f32>() };
    Ok((token, logit))
}

pub fn run_kv_cache_append_f16(
    k_cache: &[f16],
    v_cache: &[f16],
    k_new: &[f16],
    v_new: &[f16],
    seq: usize,
    head_dim: usize,
    position: usize,
) -> Result<(Vec<f16>, Vec<f16>)> {
    if position >= seq {
        return Err(TinyAgentError::Configuration(format!(
            "KV position {position} is outside seq {seq}"
        )));
    }
    if k_cache.len() != seq * head_dim || v_cache.len() != seq * head_dim {
        return Err(TinyAgentError::Configuration(format!(
            "KV cache lengths must match seq*head_dim {}, got {}/{}",
            seq * head_dim,
            k_cache.len(),
            v_cache.len()
        )));
    }
    if k_new.len() != head_dim || v_new.len() != head_dim {
        return Err(TinyAgentError::Configuration(format!(
            "new KV lengths must match head_dim {head_dim}, got {}/{}",
            k_new.len(),
            v_new.len()
        )));
    }

    let device = Device::system_default().ok_or_else(|| {
        TinyAgentError::Configuration(
            "no default Metal device found; TinyEngine requires Apple Silicon or a Metal-capable GPU"
                .to_string(),
        )
    })?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void kv_cache_append_f16(
            device half* k_cache [[buffer(0)]],
            device half* v_cache [[buffer(1)]],
            device const half* k_new [[buffer(2)]],
            device const half* v_new [[buffer(3)]],
            constant uint* dims [[buffer(4)]],
            uint id [[thread_position_in_grid]]
        ) {
            const uint head_dim = dims[0];
            const uint position = dims[1];
            if (id >= head_dim) {
                return;
            }

            const uint offset = position * head_dim + id;
            k_cache[offset] = k_new[id];
            v_cache[offset] = v_new[id];
        }
    "#;

    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to compile KV append kernel: {error}"))
        })?;
    let function = library
        .get_function("kv_cache_append_f16", None)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to load KV append kernel: {error}"))
        })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| {
            TinyAgentError::Backend(format!("failed to create KV append pipeline: {error}"))
        })?;
    let queue = device.new_command_queue();

    let buffer_k_cache = device.new_buffer_with_data(
        k_cache.as_ptr().cast(),
        std::mem::size_of_val(k_cache) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_v_cache = device.new_buffer_with_data(
        v_cache.as_ptr().cast(),
        std::mem::size_of_val(v_cache) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_k_new = device.new_buffer_with_data(
        k_new.as_ptr().cast(),
        std::mem::size_of_val(k_new) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buffer_v_new = device.new_buffer_with_data(
        v_new.as_ptr().cast(),
        std::mem::size_of_val(v_new) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let dims = [head_dim as u32, position as u32];
    let buffer_dims = device.new_buffer_with_data(
        dims.as_ptr().cast(),
        std::mem::size_of_val(&dims) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&buffer_k_cache), 0);
    encoder.set_buffer(1, Some(&buffer_v_cache), 0);
    encoder.set_buffer(2, Some(&buffer_k_new), 0);
    encoder.set_buffer(3, Some(&buffer_v_new), 0);
    encoder.set_buffer(4, Some(&buffer_dims), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: head_dim as u64,
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

    let k_ptr = buffer_k_cache.contents().cast::<f16>();
    let v_ptr = buffer_v_cache.contents().cast::<f16>();
    let k_out = unsafe { std::slice::from_raw_parts(k_ptr, seq * head_dim) }.to_vec();
    let v_out = unsafe { std::slice::from_raw_parts(v_ptr, seq * head_dim) }.to_vec();
    Ok((k_out, v_out))
}

fn f16_vec(values: &[f32]) -> Vec<f16> {
    values.iter().copied().map(f16::from_f32).collect()
}

fn deterministic_f32_vec(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = ((state >> 8) as f32) / ((1_u32 << 24) as f32);
            (unit - 0.5) * 2.0
        })
        .collect()
}

fn deterministic_f16_vec(len: usize, seed: u32) -> Vec<f16> {
    deterministic_f32_vec(len, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn deterministic_unit_vec(len: usize, seed: u32) -> Vec<f32> {
    deterministic_f32_vec(len, seed)
        .into_iter()
        .map(|value| value.clamp(-1.0, 1.0))
        .collect()
}

fn f16_to_f32_vec(values: &[f16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

fn time_kernel(
    name: &str,
    shape: &str,
    logical_ops: u64,
    run: impl FnOnce() -> Result<()>,
) -> Result<KernelBenchmarkResult> {
    let started = Instant::now();
    run()?;
    let elapsed = started.elapsed();
    let elapsed_ms_cold = elapsed.as_secs_f64() * 1000.0;
    let throughput_gops_cold = if elapsed.as_secs_f64() > 0.0 {
        logical_ops as f64 / elapsed.as_secs_f64() / 1_000_000_000.0
    } else {
        0.0
    };

    Ok(KernelBenchmarkResult {
        name: name.to_string(),
        shape: shape.to_string(),
        logical_ops,
        elapsed_ms_cold,
        throughput_gops_cold,
    })
}

fn cpu_matmul_f16_to_f32(a: &[f16], b: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0_f32;
            for i in 0..k {
                acc += a[row * k + i].to_f32() * b[i * n + col].to_f32();
            }
            out[row * n + col] = acc;
        }
    }
    out
}

fn cpu_matvec_f16_to_f32(x: &[f16], w: &[f16], n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    for col in 0..n {
        let mut acc = 0.0_f32;
        for i in 0..k {
            acc += x[i].to_f32() * w[i * n + col].to_f32();
        }
        out[col] = acc;
    }
    out
}

fn cpu_rmsnorm_f16_to_f32(
    x: &[f16],
    weight: &[f16],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        let mut sumsq = 0.0_f32;
        for col in 0..cols {
            let value = x[row * cols + col].to_f32();
            sumsq += value * value;
        }
        let inv_rms = (sumsq / cols as f32 + eps).sqrt().recip();
        for col in 0..cols {
            out[row * cols + col] = x[row * cols + col].to_f32() * weight[col].to_f32() * inv_rms;
        }
    }
    out
}

fn cpu_rope_f16_to_f32(x: &[f16], cos: &[f32], sin: &[f32], rows: usize, dims: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * dims];
    let half_dim = dims / 2;
    for row in 0..rows {
        for col in 0..dims {
            let pair_col = if col < half_dim {
                col + half_dim
            } else {
                col - half_dim
            };
            let rotated = if col < half_dim {
                -x[row * dims + pair_col].to_f32()
            } else {
                x[row * dims + pair_col].to_f32()
            };
            let index = row * dims + col;
            out[index] = x[index].to_f32() * cos[col] + rotated * sin[col];
        }
    }
    out
}

fn cpu_swiglu_f32(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(gate, up)| {
            let silu = gate / (1.0 + (-gate).exp());
            silu * up
        })
        .collect()
}

fn cpu_softmax_f32(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        let row_slice = &x[row * cols..(row + 1) * cols];
        let row_max = row_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for col in 0..cols {
            let value = (x[row * cols + col] - row_max).exp();
            out[row * cols + col] = value;
            sum += value;
        }
        for col in 0..cols {
            out[row * cols + col] /= sum;
        }
    }
    out
}

fn cpu_attention_decode_f16_to_f32(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    seq: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0_f32; seq];
    for row in 0..seq {
        let mut dot = 0.0_f32;
        for col in 0..head_dim {
            dot += q[col].to_f32() * k[row * head_dim + col].to_f32();
        }
        scores[row] = dot * scale;
    }

    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for score in &mut scores {
        *score = (*score - max_score).exp();
        sum += *score;
    }
    for score in &mut scores {
        *score /= sum;
    }

    let mut out = vec![0.0_f32; head_dim];
    for col in 0..head_dim {
        let mut acc = 0.0_f32;
        for row in 0..seq {
            acc += scores[row] * v[row * head_dim + col].to_f32();
        }
        out[col] = acc;
    }
    out
}

fn cpu_greedy_argmax_f32(logits: &[f32]) -> (u32, f32) {
    let mut best_index = 0_usize;
    let mut best_value = logits[0];
    for (index, value) in logits.iter().copied().enumerate().skip(1) {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    (best_index as u32, best_value)
}

fn cpu_kv_cache_append_f16(
    k_cache: &[f16],
    v_cache: &[f16],
    k_new: &[f16],
    v_new: &[f16],
    seq: usize,
    head_dim: usize,
    position: usize,
) -> (Vec<f16>, Vec<f16>) {
    let mut k_out = k_cache.to_vec();
    let mut v_out = v_cache.to_vec();
    let offset = position * head_dim;
    k_out[offset..offset + head_dim].copy_from_slice(k_new);
    v_out[offset..offset + head_dim].copy_from_slice(v_new);
    debug_assert_eq!(k_out.len(), seq * head_dim);
    debug_assert_eq!(v_out.len(), seq * head_dim);
    (k_out, v_out)
}

#[derive(Debug, Clone)]
pub struct MetalBackend {
    config: MetalBackendConfig,
    model: ModelInfo,
    device: MetalDeviceInfo,
}

impl MetalBackend {
    pub fn new(config: MetalBackendConfig, model: ModelInfo) -> Result<Self> {
        let device = MetalDeviceInfo::system_default()?;
        Ok(Self {
            config,
            model,
            device,
        })
    }

    pub fn device(&self) -> &MetalDeviceInfo {
        &self.device
    }

    pub fn config(&self) -> &MetalBackendConfig {
        &self.config
    }
}

#[async_trait]
impl InferenceBackend for MetalBackend {
    async fn models(&self) -> Result<Vec<ModelInfo>> {
        let mut model = self.model.clone();
        model.backend = "custom-metal".to_string();
        model.status = if self.config.package_path.is_some() {
            format!("device:{};kernels-pending", self.device.name)
        } else {
            format!("device:{};requires-tma-package", self.device.name)
        };
        Ok(vec![model])
    }

    async fn generate(&self, _request: GenerateRequest) -> Result<TokenStream> {
        Err(TinyAgentError::Unsupported(
            "custom Metal inference is scaffolded but kernels are not implemented yet: next steps are TMA loading, tokenizer wiring, f16 matmul parity, RMSNorm, RoPE, attention, KV cache, and sampling"
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        run_add_kernel_probe, run_attention_probe, run_f16_matmul_probe, run_f16_matvec_probe,
        run_greedy_sampler_probe, run_kv_cache_append_probe, run_rmsnorm_probe, run_rope_probe,
        run_softmax_probe, run_swiglu_probe, MetalDeviceInfo,
    };

    #[test]
    fn metal_probe_does_not_panic() {
        let _ = MetalDeviceInfo::system_default();
    }

    #[test]
    fn add_kernel_probe_matches_cpu() {
        if let Ok(values) = run_add_kernel_probe() {
            assert_eq!(values, vec![11.0, 22.0, 33.0, 44.0]);
        }
    }

    #[test]
    fn f16_matmul_probe_matches_cpu() {
        if let Ok(probe) = run_f16_matmul_probe() {
            assert!(
                probe.max_abs_error <= 0.001,
                "max_abs_error={} metal={:?} cpu={:?}",
                probe.max_abs_error,
                probe.metal_output,
                probe.cpu_output
            );
        }
    }

    #[test]
    fn f16_matvec_probe_matches_cpu() {
        if let Ok(probe) = run_f16_matvec_probe() {
            assert!(
                probe.max_abs_error <= 0.001,
                "max_abs_error={} metal={:?} cpu={:?}",
                probe.max_abs_error,
                probe.metal_output,
                probe.cpu_output
            );
        }
    }

    #[test]
    fn rmsnorm_probe_matches_cpu() {
        if let Ok(probe) = run_rmsnorm_probe() {
            assert!(
                probe.max_abs_error <= 0.0005,
                "max_abs_error={} metal={:?} cpu={:?}",
                probe.max_abs_error,
                probe.metal_output,
                probe.cpu_output
            );
        }
    }

    #[test]
    fn rope_probe_matches_cpu() {
        if let Ok(probe) = run_rope_probe() {
            assert!(
                probe.max_abs_error <= 0.0005,
                "max_abs_error={} metal={:?} cpu={:?}",
                probe.max_abs_error,
                probe.metal_output,
                probe.cpu_output
            );
        }
    }

    #[test]
    fn swiglu_probe_matches_cpu() {
        if let Ok(probe) = run_swiglu_probe() {
            assert!(
                probe.max_abs_error <= 0.000001,
                "max_abs_error={} metal={:?} cpu={:?}",
                probe.max_abs_error,
                probe.metal_output,
                probe.cpu_output
            );
        }
    }

    #[test]
    fn softmax_probe_matches_cpu() {
        if let Ok(probe) = run_softmax_probe() {
            assert!(
                probe.max_abs_error <= 0.000001,
                "max_abs_error={} metal={:?} cpu={:?}",
                probe.max_abs_error,
                probe.metal_output,
                probe.cpu_output
            );
        }
    }

    #[test]
    fn attention_probe_matches_cpu() {
        if let Ok(probe) = run_attention_probe() {
            assert!(
                probe.max_abs_error <= 0.000001,
                "max_abs_error={} metal={:?} cpu={:?}",
                probe.max_abs_error,
                probe.metal_output,
                probe.cpu_output
            );
        }
    }

    #[test]
    fn greedy_sampler_probe_matches_cpu() {
        if let Ok(probe) = run_greedy_sampler_probe() {
            assert_eq!(probe.metal_token, probe.cpu_token);
            assert_eq!(probe.metal_logit, probe.cpu_logit);
        }
    }

    #[test]
    fn kv_cache_append_probe_matches_cpu() {
        if let Ok(probe) = run_kv_cache_append_probe() {
            assert_eq!(probe.max_abs_error, 0.0);
            assert_eq!(probe.metal_k_cache, probe.cpu_k_cache);
            assert_eq!(probe.metal_v_cache, probe.cpu_v_cache);
        }
    }
}
