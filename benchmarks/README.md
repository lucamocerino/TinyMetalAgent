# Benchmark artifacts

This directory stores curated benchmark evidence for TinyEngine optimization work.

## What belongs here

Commit artifacts when they are useful for review or historical comparison:

- Final benchmark JSON for a PR or milestone.
- Small matrix comparisons that explain a chosen kernel/profile.
- Summaries referenced from release notes or issues.

Avoid committing local scratch runs, repeated failed experiments, large raw logs, or files containing local secrets or private paths.

Generated benchmark JSON and stderr files are ignored by default. If a result is curated and should be part of review history, add it intentionally:

```bash
git add -f benchmarks/example-result.json
```

## Reproducing the standard benchmark

```bash
make -C c benchmark GGUF=/path/to/model.gguf LLAMA_BIN=/path/to/llama-completion
```

The default output path is:

```text
benchmarks/c-qwen2.5-coder-3b-q4_0-te-vs-llama.json
```

For long-prompt testing:

```bash
make -C c benchmark-long GGUF=/path/to/model.gguf LLAMA_BIN=/path/to/llama-completion
```

## Naming convention

Use descriptive names that include:

- Model family and quantization.
- Workload or kernel profile.
- Whether it is compared against llama.cpp.
- Run count or milestone when relevant.

Example:

```text
c-qwen2.5-coder-3b-q4_0-te-final-verified-vs-llama.json
```
