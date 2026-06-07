# Model setup

TinyEngine expects a local Qwen-compatible GGUF model file. Model weights are intentionally not stored in this repository.

## Recommended local layout

The CLI auto-discovers models in these locations, in order:

```text
../models/qwen2.5-coder-3b-instruct-q4_0-te.gguf
models/qwen2.5-coder-3b-instruct-q4_0-te.gguf
```

You can also pass a path explicitly:

```bash
bin/tinyagent --model /path/to/model.gguf --ask "Explain what TinyEngine is"
TINYAGENT_MODEL=/path/to/model.gguf bin/tinyagent --ask "Explain what TinyEngine is"
```

## Downloading models

Use Qwen-compatible GGUF files from a trusted source. Check the model card and repository license before downloading. Recommended targets are tracked in `models/manifests/`.

The helper script supports the release model target used by this repository:

```bash
# TinyAgent's preferred coding model. Writes models/qwen2.5-coder-3b-instruct-q4_0-te.gguf.
scripts/prepare_qwen_model.sh
```

By default the script downloads the official pre-quantized Qwen2.5-Coder 3B GGUF from Qwen's Hugging Face GGUF repository. This is the fastest setup path for new contributors.

To quantize locally from FP16 with llama.cpp:

```bash
scripts/prepare_qwen_model.sh \
  --mode quantize \
  --target coder \
  --quant Q4_0 \
  --llama-cpp ../tools/llama.cpp
```

The script looks for `llama-quantize` under the provided llama.cpp checkout, or on `PATH`. Use `--fp16 /path/to/model-fp16.gguf` to quantize an existing FP16 GGUF without downloading it again.

For reproducible local notes, record:

- Model source URL.
- Upstream license.
- Quantization.
- Filename.
- File size.
- Checksum.

Example checksum command:

```bash
shasum -a 256 /path/to/model.gguf
```

## Runtime requirements

- Apple Silicon Mac.
- macOS with Metal support.
- Enough disk space for model weights.
- Enough unified memory for the model, KV cache, and scratch buffers.

The default 8 GB target uses a single loaded model and a 4096-token context budget. Larger models or longer contexts may cause memory pressure.

## llama.cpp oracle

`make -C c oracle` and `make -C c benchmark` can compare TinyEngine against an external `llama-completion` binary. This is optional and not part of the product runtime.

```bash
make -C c oracle GGUF=/path/to/model.gguf LLAMA_BIN=/path/to/llama-completion
make -C c benchmark GGUF=/path/to/model.gguf LLAMA_BIN=/path/to/llama-completion
```

## Troubleshooting

- **`no local Qwen GGUF found`**: pass `--model`, set `TINYAGENT_MODEL`, or place `qwen2.5-coder-3b-instruct-q4_0-te.gguf` in `models/` or `../models/`.
- **Download fails with 404**: check the upstream Hugging Face repository and pass `--quant` for a quantization file that exists, such as `Q4_0`, `Q4_K_M`, or `Q8_0`.
- **Local quantization cannot find llama.cpp**: build llama.cpp and pass `--llama-cpp /path/to/llama.cpp`, or put `llama-quantize` on `PATH`.
- **`TINYENGINE_LIBRARY` errors**: build the C runtime with `make -C c all` and set `TINYENGINE_LIBRARY=$PWD/c/build/libtinyengine.dylib` when running Python directly.
- **Out of memory or swapping**: use a smaller model, lower context, or a more aggressive quantization.
- **Unsupported model architecture**: TinyEngine is Qwen-focused first and does not aim to execute arbitrary GGUF model families yet.
