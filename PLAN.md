# TinyEngine first, TinyAgent later

The project is now engine-first.

**TinyEngine** is a from-scratch C/Metal inference engine for Qwen-compatible open-source models on Apple Silicon Macs with 8 GB of unified memory. **TinyAgent** comes later as the local agent runtime on top of the engine.

`llama.cpp` stays in the repository only as an optional oracle/reference backend for validating kernels, logits, tokenization, and sampling. It is not the product runtime.

## Product vision

Democratize local AI by making small open-source models easy to run on normal consumer Macs:

- local-only
- fast and lean
- Apple Silicon GPU via Metal
- Qwen-focused first
- 8 GB RAM as the design constraint
- minimal dependencies in the product runtime
- no cloud model providers
- Python only as a thin binding/tooling layer over the C ABI
- no heavyweight agent framework

## What "bare metal" means here

On macOS, "bare metal" means hand-written native Metal compute kernels and a minimal native runtime. It does not mean running without macOS. The engine still uses the Metal API, Apple GPU drivers, and normal macOS process isolation.

## Phase split

| Phase | Name | Goal |
| --- | --- | --- |
| 1 | TinyEngine | Load Qwen-compatible GGUF models and run inference with custom C/Metal kernels |
| 2 | TinyAgent | Add local tools, sessions, memory, and agent loop on top of TinyEngine |

Agent features wait until engine correctness and memory behavior are proven.

## First model targets

Only Qwen-compatible dense models are in scope at first.

| Order | Model family | Size | Why |
| --- | --- | ---: | --- |
| 1 | Qwen2.5-Coder Instruct | 1.5B | small, useful for coding, feasible for first kernels |
| 2 | Qwen2.5 Instruct | 1.5B | general assistant baseline |
| 3 | Qwen2.5 Instruct | 3B | upper 8 GB target after quantization |

Out of scope for now:

- 7B models
- MoE models
- multiple model families
- generic GGUF execution
- Qwen3 until Qwen2.5 works end-to-end

## Architecture

```text
┌────────────────────────────────────────────┐
│             C tinyengine tools             │
│       smoke / generate / bench / test      │
└─────────────────────┬──────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────┐
│              GGUF Qwen model               │
│  mmap loader / tokenizer / tensor views    │
└─────────────────────┬──────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────┐
│            C ABI runtime core              │
│  Qwen decode / sampler / KV cache / ops    │
└─────────────────────┬──────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────┐
│             Metal compute kernels          │
│  matmul / RMSNorm / RoPE / attention       │
└─────────────────────┬──────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────┐
│            Apple Silicon GPU               │
│          M1/M2/M3/M4 unified memory        │
└────────────────────────────────────────────┘

Optional side path:

┌────────────────────────────────────────────┐
│ llama.cpp oracle backend                   │
│ used only for parity checks and debugging  │
└────────────────────────────────────────────┘
```

## Repository layout

```text
tiny-metal-agent
├── c/                           # public C ABI, C tools, CPU reference ops, Metal backend
├── python/                      # thin ctypes binding over the C ABI
├── models/manifests/            # curated Qwen 8GB targets
└── profiles/                    # hardware profiles
```

## Model loading strategy

The active runtime path loads Qwen-compatible GGUF directly from C. The loader memory-maps GGUF
v2/v3 files, parses model metadata, tensor descriptors, quantized tensor families, and tokenizer
token/merge arrays. A custom package format can be revisited later only if direct GGUF becomes a
maintenance or performance blocker.

## Dependency policy

The product runtime should stay buildable with only the platform toolchain and OS frameworks:

1. C runtime and CLI tools: C/C++ compiler, `libm`, and Foundation/Metal on Darwin.
2. Python: standard-library-only `ctypes` binding for inspection, tests, and benchmarks.
3. `llama.cpp`: external optional oracle, never linked into the product runtime.
4. No Rust/Cargo, pip packages, NumPy/PyTorch, cloud SDKs, Electron, or agent frameworks.

`make -C c all` must remain the dependency-minimal product build. Targets such as `test`,
`oracle`, `benchmark`, and `autotune` may require optional development tools.

### Platform support: Apple-only

TinyEngine targets Apple Silicon (Metal) exclusively. There is no portable/CPU fallback build and
no non-Apple stub path. This is enforced at three layers so unsupported builds fail fast and clearly:

1. `c/include/tinyengine.h` raises `#error` when `__APPLE__` is not defined. Because it is the root
   public header included by every translation unit, this guards the whole library.
2. `c/Makefile` raises `$(error ...)` for any non-`Darwin` `uname` before compilation starts.
3. `c/src/metal_backend.mm` is unconditionally Apple/Metal source; the former `#else` non-Apple stub
   branch was removed.

A portable backend could be revisited later, but until then Apple-only keeps the runtime lean and
avoids carrying an untested cross-platform surface.

## Workload-specific kernel policy

Prompt prefill, decode, and short prompts should not share one hidden default. Runtime policy uses
`TINYENGINE_WORKLOAD=auto|short|long|decode` so benchmark and autotune runs can select the intended
kernel regime explicitly. In `auto`, long-prefill heuristics currently trigger from batch size; the
benchmark tools set `short` or `long` explicitly when comparing workload-specific profiles.

## llama.cpp oracle strategy

Keep `llama.cpp` for:

- reference next-token output
- logits comparison
- tokenizer/prompt sanity checks
- performance baseline
- regression checks while replacing kernels

Do not use it as the normal TinyEngine inference path.

## Kernel correctness order

Every kernel needs CPU/reference parity tests before it is used in generation.

1. Metal device probe
2. trivial vector-add proof-of-life kernel
3. f16 matmul with CPU parity, then tiled 16x16 threadgroup-memory path
4. RMSNorm with threadgroup reduction and CPU parity
5. RoPE with Qwen/HF half-split rotate_half semantics and CPU parity
6. SwiGLU/MLP activation with CPU parity
7. stable softmax with threadgroup max/sum reduction and CPU parity
8. single-query attention decode with CPU parity
9. KV cache append/read layout with CPU parity
10. greedy sampler with Metal argmax reduction and CPU parity
11. single-token Qwen forward pass
12. multi-token generation
13. Q8 quantized matmul
14. Q4 quantized matmul

## Benchmark strategy

Benchmark in two stages:

1. Cold-path benchmarks: include current one-shot Metal shader compilation and dispatch. These expose startup overhead and validate memory pressure.
2. Steady-state benchmarks: reuse compiled pipeline states and buffers. These are the numbers that matter for decode throughput.
3. Apple-to-apple optimization-loop benchmarks: compare TinyEngine C against `llama-completion` on the same Qwen2.5-Coder 3B prompt and deterministic generation settings, saving JSON under `benchmarks/`. `make -C c benchmark` writes `benchmarks/c-qwen2.5-coder-3b-q4_0-te-vs-llama.json` with TinyEngine C timings, llama.cpp timings, text parity, and speed ratios.

## C ABI runtime track

The C ABI is now the only implementation track. `llama.cpp` is the oracle/reference; the C runtime is the product path and should grow real GGUF inference incrementally.

Current C scope:

1. Public `tinyengine.h` ABI with runtime options, architecture detection, kernel planning, quantization/op capability masks, and generation callbacks.
2. Architecture-aware Apple Silicon kernel plan defaults for M1/M2/M3/M4.
3. `ctypes` Python binding for the ABI.
4. GGUF v2/v3 loader that memory-maps the model, validates Qwen2 metadata, parses tensor descriptors, counts quantized tensor families, and exposes `te_model_info` plus tensor descriptor lookup.
5. GGUF tokenizer token/merge parsing plus C/Python APIs for Qwen chat formatting, tokenization, and detokenization.
6. CPU reference ops for F32 tensor reads, Q4_0/Q8_0 row dequantization, rank-2 matvec, RMSNorm, RoPE, attention decode, SwiGLU, residual add, and argmax.
7. `make -C c test` kernel fixture covering tokenizer BPE merge behavior, Q4_0 nibble layout, Q8_0 signed-byte dequantization, matvec orientation, RMSNorm, RoPE, attention decode, SwiGLU, residual add, and argmax. Every C kernel iteration should extend this target before optimization.
8. `te_generate` C executable plus `make -C c oracle`, which compares TinyEngine C output with `llama-completion` on the real Qwen2.5 GGUF and deterministic prompt, and verifies the C prompt token count against llama.cpp.
9. `make -C c benchmark`, which repeats the C-vs-llama oracle prompt, saves JSON benchmark evidence, and reports speed ratios for each optimization iteration.
10. Real C `te_generate` path with Qwen chat tokenization, prompt prefill, decode KV cache, lm_head argmax, and text parity against llama.cpp. It uses pthread row-parallel matvec for CPU fallback and an experimental Darwin Metal backend for large Q4/Q8 matvecs. Correctness is comparable, but performance is still far below llama.cpp; the next target is batching/fusing the hot Q4/Q8/lm_head/decode state path on Metal.

Initial Qwen 1.5B smoke shapes:

| Kernel | Shape |
| --- | --- |
| matmul | `m=1,n=1536,k=1536` |
| RMSNorm | `rows=1,cols=1536` |
| RoPE | `rows=1,dims=128` |
| softmax | `rows=1,cols=1024` |
| attention decode | `seq=128,head_dim=128` |
| greedy sampler | `vocab=151936` |
| KV append | `seq=4096,head_dim=128` |

## 8 GB memory policy

Default target:

```text
model: Qwen 1.5B first, Qwen 3B after quantization
context: 4096
loaded models: 1
KV cache: explicitly budgeted
scratch buffers: explicitly budgeted
```

The engine should refuse configurations that would cause heavy swapping.

## Current scaffold checklist

1. Keep the C ABI as the sole runtime surface.
2. Keep Python as a thin binding/tooling layer over the C ABI.
3. Keep llama.cpp as oracle only.
4. Extend `make -C c test` before each kernel/runtime optimization.
5. Move the hot Q4/Q8/lm_head/decode state path to batched/fused Metal kernels.
6. Validate with C tests, oracle checks, and benchmark JSON evidence.
