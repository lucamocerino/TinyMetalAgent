# TinyEngine / TinyAgent

Phase 1 is **TinyEngine**: a from-scratch local Metal inference engine for Qwen-class open-source models that can run on consumer Apple Silicon Macs with 8 GB of unified memory.

Phase 2 is **TinyAgent**: the lightweight agent layer on top of the engine, with local tools, sessions, and memory.

The goal is to democratize local AI by making open-source models simple, fast, and lean. No cloud providers, no Python runtime, no Electron app, no heavyweight agent framework.

## Status

Early scaffold. Custom Metal is the product path. `llama.cpp` is kept only as an optional **oracle backend** to verify kernels and logits while TinyEngine is built.

## Current commands

Probe the local Metal device:
This also runs a tiny custom Metal `vector_add` kernel and an `f16` matmul kernel with CPU parity.

```bash
cargo run -p tinyagent -- engine probe
```

Run the current Qwen-size cold benchmark:

```bash
cargo run -p tinyagent -- engine bench
cargo run -p tinyagent -- engine bench --hot --iterations 25
cargo run -p tinyagent -- engine phase-bench \
  --hf-dir ../mlx-qwen-lab/models/Qwen2.5-0.5B-Instruct-4bit \
  --out benchmarks/qwen-phase-benchmark.json
cargo run --release -p tinyagent -- engine qwen-run \
  --hf-dir ../mlx-qwen-lab/models/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "Rispondi in italiano con tre parole: cosa sei?" \
  --max-prompt-tokens 10 \
  --max-tokens 4 \
  --projection-backend metal
```

`engine phase-bench` reads a real Hugging Face/MLX Qwen directory and writes per-phase prefill
and synthetic TTFT estimates until full end-to-end TinyEngine inference is wired. Decode estimates
use the dedicated Metal `matvec_f16_f32` path instead of the generic `m=1` tiled matmul baseline.
`engine qwen-run` executes the full Qwen2.5 graph over real MLX 4-bit weights. With
`--projection-backend metal`, q4 affine projections use reusable Metal pipelines and GPU-resident
weight buffers. Q/K/V and gate/up projections are batched into shared command buffers to cut launch
overhead; scalar ops, RoPE, attention, and residual math are still CPU in this milestone.

Create a metadata-only `.tma` package scaffold from a local Hugging Face or GGUF source:
For Hugging Face directories, `tokenizer.json` is copied and Qwen dimensions are read from `config.json` when present.

```bash
cargo run -p tinyagent -- convert hf ./models/qwen-hf --out ./models/qwen.tma
cargo run -p tinyagent -- convert gguf ./models/qwen.gguf --out ./models/qwen.tma
cargo run -p tinyagent -- engine inspect --package ./models/qwen.tma
```

Inspect the default custom Metal model configuration:

```bash
cargo run -p tinyagent -- models
```

Use the llama oracle only for reference checks:

```bash
cargo run -p tinyagent -- chat \
  --backend llama \
  --gguf ./models/qwen.gguf \
  --llama-server-bin /path/to/llama-server \
  "ciao"
```

Development stub:

```bash
cargo run -p tinyagent -- chat --backend stub "ciao"
```

See `PLAN.md` for the implementation roadmap.
