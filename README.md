<p align="center">
  <img src="docs/assets/tiny-metal-agent-logo.svg" alt="TinyMetalAgent logo" width="520">
</p>

# TinyEngine / TinyAgent

Phase 1 is **TinyEngine**: a from-scratch local Metal inference engine for Qwen-class open-source models that can run on consumer Apple Silicon Macs with 8 GB of unified memory.

Phase 2 is **TinyAgent**: the lightweight agent layer on top of the engine, with local tools, sessions, and memory.

The goal is to democratize local AI by making open-source models simple, fast, and lean. No cloud providers, no Electron app, no heavyweight agent framework; Python is only a thin binding/tooling layer over the C ABI.

## Mission

TinyMetalAgent exists to help students practice with local agentic programming tools on educational projects.
The project is built around a simple idea: learners should be able to experiment with an AI coding assistant
locally, on normal consumer hardware, while studying programming languages, algorithms, tests, debugging,
and software engineering workflows.

The long-term mission is educational first. TinyAgent should help students build small school-style projects,
ask programming questions, inspect files, create exercises, write tests, and learn by seeing what an agent is
doing step by step. TinyEngine keeps the inference stack local and inspectable so learners can also understand
the lower-level pieces: model loading, tokenization, quantized kernels, Apple Silicon GPU execution, terminal
agent UX, safety boundaries, and repeatable benchmarking.

This repository is intentionally small and dependency-light so students and contributors can read the code,
modify it, and experiment with the full stack without needing cloud model APIs or heavyweight desktop apps.

## Status

TinyEngine is now C-first, with a thin Python `ctypes` binding for inspection, tests, and tooling. Custom C/Metal is the product path. `llama.cpp` is kept only as an optional **oracle backend** to verify tokenization, generated text, and performance while TinyEngine is built.

This is an alpha-stage project. It is useful for experimentation, learning, and early local-agent workflows,
but it is not yet a polished replacement for mature inference engines or commercial coding agents.

## Dependencies

The product runtime keeps dependencies minimal:

| Surface | Required dependencies |
| --- | --- |
| C runtime and CLI tools | C compiler, C++/Objective-C++ compiler on Darwin, system `libm`, Apple Foundation/Metal frameworks on Darwin |
| Python binding | Python 3 standard library only (`ctypes`) |
| C tests | Python 3 standard library for generated guard fixtures |
| Oracle/benchmark/autotune | Optional external `llama-completion` binary and local GGUF model |

No Rust, Cargo, pip packages, NumPy, PyTorch, Electron, or cloud SDKs are required.

## Current commands

Build the C ABI runtime and Python binding:

```bash
make -C c clean all
make -C c test
c/build/te_smoke ../models/qwen2.5-coder-3b-instruct-q4_0-te.gguf
make -C c oracle
make -C c benchmark
PYTHONPATH=python TINYENGINE_LIBRARY=$PWD/c/build/libtinyengine.dylib python3 - <<'PY'
from tinyengine import Model, capabilities, detect_arch, make_kernel_plan
path = "../models/qwen2.5-coder-3b-instruct-q4_0-te.gguf"
print(detect_arch())
print(make_kernel_plan())
print(capabilities())
with Model(path) as model:
    print(model.info())
    print(model.tensor_info("token_embd.weight"))
    print(model.tensor_info("output.weight"))
    print(len(model.dequantize_row("token_embd.weight", 0)))
PY
```

Install the Python package and CLI in editable mode:

```bash
python3 -m pip install -e .
TINYENGINE_LIBRARY=$PWD/c/build/libtinyengine.dylib python3 -m tinyagent --help
```

`make -C c all` builds only the product runtime library and C CLI tools. Python and `llama.cpp`
are optional development surfaces used by `test`, `oracle`, `benchmark`, and binding examples.

The C ABI memory-maps GGUF v2/v3 files, parses metadata, the tensor directory, and GGUF tokenizer
token/merge arrays, exposes Qwen model/tensor/tokenizer descriptors, and runs a real Qwen decode
path with KV cache. CPU reference ops cover F32 reads, Q4_0/Q8_0 row dequantization, rank-2 matvec,
RMSNorm, RoPE, attention decode, SwiGLU, residual add, and argmax.

Every C kernel iteration should add or update `make -C c test`, which checks tokenizer BPE merge
behavior, Q4_0 nibble layout, Q8_0 signed bytes, matvec orientation, RMSNorm, RoPE, attention
decode, SwiGLU, residual add, and argmax on a tiny synthetic GGUF fixture. Large Q4_0/Q8_0 matvecs
use the experimental Metal backend by default on Darwin; set `TINYENGINE_METAL_MATVEC=0` to force
the CPU reference path.

`make -C c oracle` runs the C generation executable against `llama-completion` on the real Qwen2.5
GGUF and deterministic prompt; it also verifies that the C Qwen chat prompt token count matches
llama.cpp. Override `GGUF=...` and `LLAMA_BIN=...` when those live outside the default local paths.
The C path is correctness-comparable but still far slower than llama.cpp until more of the hot path
is batched/fused on Metal; `llama.cpp` remains the external oracle for correctness and performance
checks.

`make -C c benchmark` repeats the same deterministic prompt, writes
`benchmarks/c-qwen2.5-coder-3b-q4_0-te-vs-llama.json`, and records TinyEngine C timings,
llama.cpp prompt and decode timings, text parity, and speed ratios for the optimization loop.
The release benchmark target is `qwen2.5-coder-3b-instruct-q4_0-te.gguf`; the next optimization
target is reducing cold-load latency and improving the Q4/Q8/lm_head/decode Metal hot path.

Set `TINYENGINE_WORKLOAD=short|long|decode|auto` to make workload-specific kernel policy explicit.
`make -C c benchmark-long` sets `TINYENGINE_WORKLOAD=long`; `autotune` sets `short` or `long` per
workload before testing candidate kernel profiles.

See `PLAN.md` for the implementation roadmap.

## tinyagent (local coding agent)

`tinyagent` is a terminal coding agent that runs the LLM
100% locally through TinyEngine. The `bin/tinyagent` launcher sets `PYTHONPATH` and
auto-discovers a local Qwen GGUF, so the simplest possible invocation just drops you
into the interactive chat:

```bash
bin/tinyagent                       # interactive REPL in the current directory
```

tinyagent is dual-mode: **ask it a question** ("explain a linked list", "what is a
hash map?") and it answers conversationally without touching your files; **describe a
task** ("create fizzbuzz.py and run it") and it acts using tools. Use `/ask <question>`
to force a one-off chat-only answer, or bare `/ask` to toggle a persistent chat-only
mode (the prompt becomes `tinyagent(ask)>`).

It auto-discovers `qwen2.5-coder-3b-instruct-q4_0-te.gguf`, searching both next to the repo
(`../models/`) and inside it (`models/`). Override the model with `--model <path.gguf>` or the
`TINYAGENT_MODEL` env var.

One-shot (non-interactive) mode runs a single task and exits:

```bash
bin/tinyagent --root /path/to/project --yes "Create fizzbuzz.py and run python3 -m pytest -q"
bin/tinyagent --ask "Explain how a linked list works"   # chat-only, no tools
```

Useful flags: `--root` (workspace, default cwd), `--yes` (auto-approve tools),
`--ask` (chat-only, never use tools), `--dry-run` (never execute write/exec),
`--no-network`, `--max-steps`, `--ctx`, `--daemon`, `--verbose`, `--resume`/`--save`
(persist a session). `--ctx` defaults to the detected hardware recommendation, and
`--daemon` reuses a persistent local model process across CLI invocations. In the REPL,
type `/help` for commands (`/ask`, `/tools`, `/context`, `/diff`, `/clear`,
`/compact`, `/approve auto|ask`, `/save`, `/exit`), `?` for quick help, `!cmd` to run
a shell command in the workspace, or mention files with `@path`.
Slash commands autocomplete with Tab in terminals that support readline; typing `/`
and pressing Enter lists the available commands. File mentions autocomplete with Tab
after `@`.
Add `bin/` to your `PATH` (or symlink `bin/tinyagent`) to type just `tinyagent`.

The engine decodes with low-temperature top-k sampling (default `temperature=0.3`,
`top_k=40`) to avoid greedy repetition loops while keeping tool-call JSON intact;
override via `TINYENGINE_TEMPERATURE`, `TINYENGINE_TOP_K`, `TINYENGINE_SEED`.

Chat/ask answers stream to the terminal token-by-token as the model decodes them,
so you get live output instead of waiting for the whole reply.

By default the terminal UI keeps output concise. Use `--verbose` to show model
load/daemon reuse, context rendering, agent steps, full tool arguments/results, and
model/tool timings when debugging.

## Model setup

Model weights are not included in this repository. Place a trusted Qwen-compatible GGUF at one of
the auto-discovered paths, pass `--model <path.gguf>`, or set `TINYAGENT_MODEL`.

The easiest setup path downloads the official pre-quantized Qwen GGUF that `tinyagent` prefers:

```bash
scripts/prepare_qwen_model.sh
```

To quantize locally from FP16 with llama.cpp instead of downloading a pre-quantized GGUF:

```bash
scripts/prepare_qwen_model.sh --mode quantize --target coder --quant Q4_0 --llama-cpp ../tools/llama.cpp
```

See `docs/MODELS.md` for filenames, checksum guidance, memory expectations, and oracle/benchmark
setup.

## Safety

TinyEngine inference is local. TinyAgent tools may still read files, write files, run shell commands,
or use network tools depending on approval mode and flags. Use `--dry-run`, `--ask`, and
`--no-network` for safer evaluation, and avoid `--yes` in untrusted repositories.

See `docs/SAFETY.md` and `SECURITY.md`.

## Limitations

TinyMetalAgent has important limitations that come from both the current model target and the architecture:

- **Model quality is bounded by small local models.** The default Qwen2.5-Coder 3B Q4_0 target is useful for
  lightweight coding tasks, explanations, and controlled tool-use loops, but it can still produce incomplete
  plans, markdown instead of tool calls, or requests for more context when a larger model would infer intent.
- **Context is intentionally small on 8 GB Macs.** The runtime defaults to a conservative context window on
  low-memory Apple Silicon machines. TinyAgent uses dynamic tool windows, file mentions, and compaction to
  stay within that budget, but long conversations and large codebases still need careful context management.
- **No cross-turn KV reuse yet.** Each turn can still require prefill over the active conversation. The daemon
  avoids repeated model loading, but full prompt-prefix or KV-cache reuse is future work.
- **Performance is alpha-grade.** Decode is usable for short local interactions, but cold start, long prompts,
  and multi-step tool loops are still slower than mature engines. `llama.cpp` remains the reference oracle for
  correctness and performance comparisons.
- **Apple Silicon only.** The product runtime targets macOS with Metal. There is no supported Linux, Windows,
  CUDA, CPU-only, or browser backend.
- **Qwen-first, not generic GGUF.** The loader parses GGUF metadata and tensors, but the execution path is
  focused on Qwen2-compatible dense models and Q4_0/Q8_0 quantization first.
- **Tool execution is powerful.** TinyAgent can read files, write files, run shell commands, and optionally use
  network tools. Keep approvals enabled in untrusted workspaces and prefer `--dry-run`, `--ask`, and
  `--no-network` when evaluating behavior.

## Open-source project files

- `CONTRIBUTING.md` explains development setup, testing expectations, and PR guidance.
- `CODE_OF_CONDUCT.md` sets community behavior expectations.
- `SECURITY.md` explains vulnerability reporting and local-agent security boundaries.
- `SUPPORT.md` explains where to ask for help.
- `CHANGELOG.md` and `RELEASE.md` track release notes and release process.
- `benchmarks/README.md` explains benchmark artifact policy.
