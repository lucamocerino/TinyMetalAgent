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
cargo run --release -p tinyagent -- engine compare \
  --tinyengine-source gguf \
  --hf-dir ../mlx-qwen-lab/models/Qwen2.5-0.5B-Instruct-4bit \
  --gguf ../models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --llama-bin ../tools/llama.cpp-b9219/current/llama-completion \
  --prompt "Rispondi in italiano con tre parole: cosa sei?" \
  --max-prompt-tokens 512 \
  --max-tokens 4 \
  --runs 3 \
  --out benchmarks/qwen-gguf-q4_0-batched-prefill-gpu-decode-compare.json
```

`engine phase-bench` reads a real Hugging Face/MLX Qwen directory and writes per-phase prefill
and synthetic TTFT estimates until full end-to-end TinyEngine inference is wired. Decode estimates
use the dedicated Metal `matvec_f16_f32` path instead of the generic `m=1` tiled matmul baseline.
`engine qwen-run` executes the full Qwen2.5 graph over real MLX 4-bit weights. With
`--projection-backend metal`, q4 affine projections use reusable Metal pipelines and GPU-resident
weight buffers. Prompt prefill is batched across tokens; decode keeps hidden state, KV cache,
RMSNorm, RoPE, attention, residuals, SwiGLU, Q8 `lm_head`, and argmax on Metal where the GGUF path
supports it.
`engine compare` runs a repeatable optimization-loop benchmark against `llama-completion` using the
same Qwen2.5 0.5B prompt, deterministic generation settings, and prompt token count. It reports
load, prompt/TTFT, decode, wall-clock, output parity, and speed ratios as JSON. With
`--tinyengine-source gguf`, TinyEngine and llama.cpp use the same GGUF file and quantized tensors.
The first same-quant baseline is Qwen2.5 0.5B GGUF Q4_0; TinyEngine maps Q4_0 blocks and GGUF Q8_0
`output.weight` onto Metal kernels. Prompt logits are skipped for intermediate prompt tokens, so
only the last prompt token and decode steps project through `lm_head`. The current Metal path also
reuses input, linear-bias, and output buffers across projections, uses a batch-tiled Q4 prefill
kernel, and uses a row-tiled Q4 decode kernel. The latest 3-run same-quant benchmark is
`benchmarks/qwen-gguf-q4_0-batched-prefill-gpu-decode-compare.json`: TinyEngine median prompt/TTFT
is 303.5 ms and decode is 32.72 tok/s, while llama.cpp is 48.25 ms and 121.26 tok/s.

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
