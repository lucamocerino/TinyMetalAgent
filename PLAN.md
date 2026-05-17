# TinyEngine first, TinyAgent later

The project is now engine-first.

**TinyEngine** is a from-scratch Metal inference engine for Qwen-compatible open-source models on Apple Silicon Macs with 8 GB of unified memory. **TinyAgent** comes later as the local agent runtime on top of the engine.

`llama.cpp` stays in the repository only as an optional oracle/reference backend for validating kernels, logits, tokenization, and sampling. It is not the product runtime.

## Product vision

Democratize local AI by making small open-source models easy to run on normal consumer Macs:

- local-only
- fast and lean
- Apple Silicon GPU via Metal
- Qwen-focused first
- 8 GB RAM as the design constraint
- no cloud model providers
- no Python runtime in the product
- no heavyweight agent framework

## What "bare metal" means here

On macOS, "bare metal" means hand-written native Metal compute kernels and a minimal native runtime. It does not mean running without macOS. The engine still uses the Metal API, Apple GPU drivers, and normal macOS process isolation.

## Phase split

| Phase | Name | Goal |
| --- | --- | --- |
| 1 | TinyEngine | Load converted Qwen model packages and run inference with custom Metal kernels |
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
│                tinyengine CLI              │
│  probe / convert / bench / generate        │
└─────────────────────┬──────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────┐
│              TMA model package             │
│  metadata.json / tokenizer.json / tensors  │
└─────────────────────┬──────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────┐
│           custom Metal runtime             │
│  loader / tokenizer / sampler / KV cache   │
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
├── crates/
│   ├── tinyagent                # temporary CLI binary; will expose TinyEngine commands first
│   ├── tinyagent-core           # shared request/model/profile types
│   ├── tinyagent-backend-metal  # custom Metal backend scaffold
│   ├── tinyagent-backend-llama  # oracle/reference backend only
│   └── tma-format               # directory-based model package metadata
├── models/manifests/            # curated Qwen 8GB targets
└── profiles/                    # hardware profiles
```

## Model package format

Start with a directory package, not a binary archive:

```text
model.tma/
├── metadata.json
├── tokenizer.json
└── tensors/
    ├── model.embed_tokens.weight.bin
    ├── model.layers.0.self_attn.q_proj.weight.bin
    └── ...
```

The first `.tma` format is intentionally debuggable and easy to change. A single-file archive can come later after the tensor layout stabilizes.

## Converter strategy

Priority:

1. Hugging Face safetensors + tokenizer.json -> `.tma` fp16 package
2. TinyEngine f16 forward pass
3. custom Q8 package and kernels
4. custom Q4 package and kernels
5. GGUF importer translating GGUF into TMA tensors

GGUF is useful, but not first, because GGUF quant layouts add complexity before the engine has correct f16 kernels.

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

1. Save engine-first plan.
2. Add custom Metal backend crate, proof-of-life vector-add kernel, and f16 matmul parity kernel.
3. Keep llama backend as oracle only.
4. Add `.tma` package metadata crate.
5. Add metadata-only converter commands.
6. Add Qwen 8GB target manifests.
7. Validate workspace with formatting, tests, and CLI smoke commands.
