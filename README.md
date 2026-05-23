# TinyEngine / TinyAgent

Phase 1 is **TinyEngine**: a from-scratch local Metal inference engine for Qwen-class open-source models that can run on consumer Apple Silicon Macs with 8 GB of unified memory.

Phase 2 is **TinyAgent**: the lightweight agent layer on top of the engine, with local tools, sessions, and memory.

The goal is to democratize local AI by making open-source models simple, fast, and lean. No cloud providers, no Electron app, no heavyweight agent framework; Python is only a thin binding/tooling layer over the C ABI.

## Status

TinyEngine is now C-first, with a thin Python `ctypes` binding for inspection, tests, and tooling. Custom C/Metal is the product path. `llama.cpp` is kept only as an optional **oracle backend** to verify tokenization, generated text, and performance while TinyEngine is built.

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
c/build/te_smoke ../models/qwen2.5-0.5b-instruct-q4_0.gguf
make -C c oracle
make -C c benchmark
PYTHONPATH=python TINYENGINE_LIBRARY=$PWD/c/build/libtinyengine.dylib python3 - <<'PY'
from tinyengine import Model, capabilities, detect_arch, make_kernel_plan
path = "../models/qwen2.5-0.5b-instruct-q4_0.gguf"
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
`benchmarks/c-qwen2.5-0.5b-q4_0-vs-llama.json`, and records TinyEngine C timings, llama.cpp prompt
and decode timings, text parity, and speed ratios for the optimization loop. The current real C
reference benchmark is correctness-comparable but not performance-comparable yet: TinyEngine C
median total is 2601.02 ms and 1.55 tok/s, while llama.cpp total is 71.32 ms with 123.76 tok/s
decode on the same prompt. The next optimization target is moving Q4/Q8 matvec, lm_head, and decode
state to the Metal backend.

Set `TINYENGINE_WORKLOAD=short|long|decode|auto` to make workload-specific kernel policy explicit.
`make -C c benchmark-long` sets `TINYENGINE_WORKLOAD=long`; `autotune` sets `short` or `long` per
workload before testing candidate kernel profiles.

See `PLAN.md` for the implementation roadmap.
