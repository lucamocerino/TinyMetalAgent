# Contributing to TinyMetalAgent

Thanks for helping improve TinyEngine and TinyAgent. This project is intentionally small, local-first, Apple Silicon focused, and dependency-light.

## Development setup

TinyEngine currently supports macOS on Apple Silicon only. The product build uses the platform toolchain and Apple Metal frameworks.

```bash
git clone https://github.com/lucamocerino/tiny-metal-agent.git
cd tiny-metal-agent
make -C c clean all
make -C c test
```

For the Python bindings and CLI, use the source tree directly or install in editable mode:

```bash
python3 -m pip install -e .
PYTHONPATH=python TINYENGINE_LIBRARY=$PWD/c/build/libtinyengine.dylib python3 -m tinyagent --help
```

Model-backed commands require a local Qwen-compatible GGUF file. See `docs/MODELS.md`.

## Project direction

Keep changes aligned with the current constraints:

- C ABI runtime is the product path.
- Python remains a thin standard-library `ctypes` binding and CLI/tooling layer.
- `llama.cpp` is an optional oracle for parity and benchmarks, not a runtime dependency.
- Apple Silicon and Metal are the supported target.
- Avoid new dependencies unless they are clearly justified and optional.

## What to run before opening a PR

At minimum:

```bash
make -C c clean all
make -C c test
python3 -m compileall -q python
PYTHONPATH=python TINYENGINE_LIBRARY=$PWD/c/build/libtinyengine.dylib python3 -m tinyagent --help
scripts/prepare_qwen_model.sh --dry-run --target coder
```

When changing generation, tokenization, model loading, or Metal kernels, also run the relevant oracle and benchmark targets with a local model:

```bash
make -C c oracle GGUF=/path/to/model.gguf LLAMA_BIN=/path/to/llama-completion
make -C c benchmark GGUF=/path/to/model.gguf LLAMA_BIN=/path/to/llama-completion
```

## Pull request guidelines

- Keep PRs focused on one behavior or subsystem.
- Include tests for runtime, tokenizer, model-loading, and kernel changes.
- Include benchmark evidence when changing performance-sensitive code.
- Document user-facing behavior changes in `README.md`, `docs/`, or `CHANGELOG.md`.
- Do not commit GGUF model files, generated build outputs, local logs, or secrets.

## Coding conventions

- C code is compiled with `-Wall -Wextra -Werror -pedantic`.
- Prefer explicit error returns over silent fallbacks.
- Keep the public C ABI stable; bump `TE_ABI_VERSION` only for ABI changes.
- Keep Python standard-library-only unless the dependency is optional development tooling.
- Avoid broad exception swallowing in Python. Surface actionable errors.

## Benchmark artifacts

Commit curated benchmark evidence only. Put repeatable summaries and notable results under `benchmarks/`; keep local scratch runs out of git unless they support a PR. See `benchmarks/README.md`.
