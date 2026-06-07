<p align="center">
  <img src="docs/assets/tiny-metal-agent-logo.png" alt="TinyMetalAgent logo" width="420">
</p>

# TinyMetalAgent

TinyMetalAgent is a local, student-focused AI coding agent for educational programming projects. It runs a
Qwen-compatible open-source model locally through a custom C/Metal inference engine, with a thin Python terminal
agent on top.

The project has two layers:

| Layer | Role |
| --- | --- |
| **TinyAgent** | The local terminal coding agent for student projects, tools, sessions, and REPL UX. |
| **TinyEngine** | The C/Metal inference engine that loads Qwen-compatible GGUF models and runs local decode. |

## TinyMetalAgent local coding agent

`tinyagent` is the terminal coding agent included in this repository. It runs 100% locally through TinyEngine.
The `bin/tinyagent` launcher sets `PYTHONPATH` and auto-discovers a local Qwen GGUF, so the simplest possible
invocation just drops you into the interactive chat:

<p align="center">
  <img src="docs/assets/tinyagent-demo.svg" alt="TinyAgent terminal demo creating a Matrix class and tests" width="760">
</p>

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

Add `bin/` to your `PATH` (or symlink `bin/tinyagent`) to type just `tinyagent`.

### CLI options

| Option | Purpose |
| --- | --- |
| `--root PATH` | Workspace root. Defaults to the current directory. |
| `--yes` | Auto-approve tool calls. Useful for trusted local experiments. |
| `--ask` | Chat-only mode. The agent answers without using tools. |
| `--dry-run` | Simulate write/exec tools without changing files or running commands. |
| `--no-network` | Disable network tools. |
| `--max-steps N` | Limit the number of agent/tool loop steps. |
| `--ctx N` | Model context tokens. Defaults to the detected hardware recommendation. |
| `--daemon` | Reuse a persistent local model process across CLI invocations. |
| `--verbose` | Show model load, prompt rendering, steps, tool args/results, and timings. |
| `--resume FILE` / `--save FILE` | Resume or persist a local session JSON file. |

### REPL commands and shortcuts

| Input | Action |
| --- | --- |
| `/help` | Show full interactive help. |
| `/ask [question]` | Ask a one-off chat-only question, or toggle persistent chat-only mode with no argument. |
| `/tools` | List available tools. |
| `/context` | Show context/token usage. |
| `/diff` | Show a Git diff summary for the workspace. |
| `/clear` | Reset the conversation. |
| `/compact` | Summarize history to free context. |
| `/approve auto` / `/approve ask` | Change approval mode. |
| `/save FILE` | Save the session. |
| `/exit` | Exit the REPL. |
| `?` | Show quick help. |
| `!cmd` | Run a shell command in the workspace. |
| `@path` | Attach a file or directory listing to the next prompt. |

Slash commands autocomplete with Tab in terminals that support readline. Typing `/`
and pressing Enter lists the available commands. File mentions autocomplete with Tab
after `@`.

### Runtime behavior

- Chat/ask answers stream token-by-token, so you see output while the model decodes.
- The terminal UI is concise by default; use `--verbose` for debugging details.
- Sampling defaults to `temperature=0.3` and `top_k=40` to avoid greedy repetition while
  keeping tool-call JSON stable.
- Override sampling with `TINYENGINE_TEMPERATURE`, `TINYENGINE_TOP_K`, and
  `TINYENGINE_SEED`.

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

## Installation

One-command local setup for most users:

```bash
./install.sh              # build, test, install package, verify CLI
./install.sh --with-model # also download the default local Qwen GGUF
```

Model download is opt-in because model files are large and governed by their own upstream terms.

## Developer build and Python binding

For contributors who want lower-level engine and Python binding checks:

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

Install only the Python package and CLI in editable mode:

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

## Kernel and runtime optimization techniques

TinyEngine is built as a learning-oriented inference engine, but it still uses several practical optimization
techniques to make local inference feasible on resource-constrained Apple Silicon machines:

| Area | Techniques used |
| --- | --- |
| Quantized math | GGUF `Q4_0` and `Q8_0` tensor support, block-wise dequantization, optimized Q4/Q8 matvec and matmul paths, and special lm_head projection kernels. |
| Metal acceleration | Hand-written Metal compute kernels for quantized matvec/matmul, RMSNorm, RoPE, attention, SwiGLU/MLP, residual add, and argmax-style projection. |
| Kernel fusion | Fused QKV projection paths, fused gate/up SwiGLU MLP paths, fused FFN down-projection plus residual add, and fused decode/prefill layer dispatch paths where supported. |
| Prefill vs decode policy | `TINYENGINE_WORKLOAD=short|long|decode|auto` selects different kernel regimes for short prompts, long prefill, and token-by-token decode instead of relying on one hidden default. |
| Batched prefill | Prompt tokens can be processed in batches so long prompts do not always fall back to purely token-by-token execution. |
| Memory movement | The runtime memory-maps GGUF weights, uses Metal buffers over mapped model data where possible, warms model pages for first-use latency, and keeps selected KV/cache buffers resident for longer workloads. |
| Precision tradeoffs | Some intermediate FFN/KV paths can use half-precision storage to reduce memory bandwidth and scratch-buffer pressure. |
| Command-buffer strategy | Several paths reduce per-layer dispatch overhead by grouping work into fused layer or all-layer command-buffer flows. |
| Profiling and autotune | `TINYENGINE_METAL_PROFILE`, benchmark JSON output, and `make -C c autotune`/`autotune-quick` compare kernel profiles such as matmul on/off, FlashAttention on/off, fused FFN options, argmax CPU/GPU, and QKV fusion. |
| CPU reference parity | Every optimized path is backed by CPU/reference checks in `make -C c test`, including tokenizer behavior, quant layout, matvec orientation, RMSNorm, RoPE, attention decode, SwiGLU, residual add, and argmax. |

These optimizations are intentionally transparent: most switches are exposed through environment variables so
students and contributors can observe tradeoffs, reproduce benchmark runs, and learn how kernel choices affect
latency, throughput, memory pressure, and correctness.

### `TINYENGINE_WORKLOAD` policy

Transformer inference has two very different phases:

- **Prefill** processes the prompt tokens and fills the KV cache. It benefits from batched work, larger matrix
  operations, fewer per-token dispatches, and kernels that can amortize setup cost across many tokens.
- **Decode** generates one token at a time after prefill. It is latency-sensitive and benefits from small,
  fused dispatches, fast lm_head projection/argmax, and avoiding heavyweight long-prefill kernels.

TinyEngine exposes this choice explicitly through `TINYENGINE_WORKLOAD`:

| Value | Intended use | Runtime behavior |
| --- | --- | --- |
| `short` | Short prompts and interactive agent turns. | Avoids long-prefill-only paths that can add overhead on small batches. |
| `long` | Long prompt prefill and benchmark-long style runs. | Enables long-prefill heuristics such as K/V matmul choices, paired FFN/SwiGLU paths, and larger attention/prefill strategies when batch size justifies them. |
| `decode` | Token-by-token generation experiments. | Biases decisions away from prefill-specific kernels so decode latency is easier to isolate. |
| `auto` | Default general-purpose mode. | Chooses based on batch size; current heuristics treat larger batches as long-prefill workloads. |

The benchmark and autotune tools set this deliberately instead of hiding it:

```bash
TINYENGINE_WORKLOAD=short make -C c benchmark
TINYENGINE_WORKLOAD=long make -C c benchmark-long
make -C c autotune AUTOTUNE_WORKLOADS=short,long
```

This is useful when experimenting on 8 GB machines because the fastest strategy for a short REPL request is not
necessarily the fastest strategy for a long prompt. Making the policy visible helps contributors compare
latency, throughput, and memory-pressure tradeoffs without changing source code.

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

## AI assistance disclosure

This repository was developed with substantial assistance from GPT-5.5. The generated code,
documentation, tests, and release scaffolding were reviewed, edited, and validated by the project
maintainer before publication. The project should still be treated as alpha software: inspect the
code, run the tests, and evaluate safety boundaries before using it in real coursework or projects.
