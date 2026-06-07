from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


TE_ABI_VERSION = 3
TE_QUANT_COUNT = 22


class TinyEngineError(RuntimeError):
    pass


class _ArchInfo(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("name", ctypes.c_char * 64),
        ("cpu_cores", ctypes.c_uint32),
        ("gpu_cores", ctypes.c_uint32),
        ("unified_memory_bytes", ctypes.c_uint64),
        ("recommended_max_context", ctypes.c_uint32),
    ]


class _RuntimeOptions(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("target_arch", ctypes.c_int),
        ("context_tokens", ctypes.c_uint32),
        ("batch_tokens", ctypes.c_uint32),
        ("memory_limit_bytes", ctypes.c_uint64),
        ("kernel_cache_dir", ctypes.c_char_p),
        ("flags", ctypes.c_uint32),
    ]


class _KernelPlan(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("target_arch", ctypes.c_int),
        ("quant_mask", ctypes.c_uint64 * 2),
        ("optimized_quant_mask", ctypes.c_uint64 * 2),
        ("vector_op_mask", ctypes.c_uint64),
        ("optimization_flags", ctypes.c_uint32),
        ("q4_prefill_batch_tile", ctypes.c_uint32),
        ("q4_decode_row_tile", ctypes.c_uint32),
        ("q8_lm_head_row_tile", ctypes.c_uint32),
        ("dot_threads", ctypes.c_uint32),
        ("preferred_alignment_bytes", ctypes.c_uint32),
        ("max_context_tokens", ctypes.c_uint32),
        ("memory_budget_bytes", ctypes.c_uint64),
        ("metal_function_suffix", ctypes.c_char * 32),
    ]


class _Capabilities(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("known_quant_mask", ctypes.c_uint64 * 2),
        ("optimized_quant_mask", ctypes.c_uint64 * 2),
        ("vector_op_mask", ctypes.c_uint64),
        ("optimization_flags", ctypes.c_uint32),
        ("preferred_alignment_bytes", ctypes.c_uint32),
        ("backend_name", ctypes.c_char * 32),
        ("notes", ctypes.c_char * 160),
    ]


class _ModelInfo(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("gguf_version", ctypes.c_uint32),
        ("metadata_kv_count", ctypes.c_uint64),
        ("tensor_count", ctypes.c_uint64),
        ("tensor_data_offset", ctypes.c_uint64),
        ("tensor_data_bytes", ctypes.c_uint64),
        ("parameter_count", ctypes.c_uint64),
        ("file_size_bytes", ctypes.c_uint64),
        ("name", ctypes.c_char * 128),
        ("architecture", ctypes.c_char * 32),
        ("context_length", ctypes.c_uint32),
        ("embedding_length", ctypes.c_uint32),
        ("block_count", ctypes.c_uint32),
        ("feed_forward_length", ctypes.c_uint32),
        ("attention_head_count", ctypes.c_uint32),
        ("attention_head_count_kv", ctypes.c_uint32),
        ("head_dim", ctypes.c_uint32),
        ("vocab_size", ctypes.c_uint32),
        ("rms_norm_epsilon", ctypes.c_float),
        ("rope_freq_base", ctypes.c_float),
        ("quant_tensor_counts", ctypes.c_uint64 * TE_QUANT_COUNT),
        ("reserved", ctypes.c_uint32 * 16),
    ]


class _TensorInfo(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("quant", ctypes.c_int),
        ("ggml_type", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint64 * 4),
        ("elements", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("relative_offset", ctypes.c_uint64),
        ("absolute_offset", ctypes.c_uint64),
        ("name", ctypes.c_char * 128),
        ("reserved", ctypes.c_uint32 * 16),
    ]


class _TokenizerInfo(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("model", ctypes.c_char * 32),
        ("pre", ctypes.c_char * 32),
        ("token_count", ctypes.c_uint64),
        ("token_type_count", ctypes.c_uint64),
        ("merge_count", ctypes.c_uint64),
        ("bos_token_id", ctypes.c_uint32),
        ("eos_token_id", ctypes.c_uint32),
        ("padding_token_id", ctypes.c_uint32),
        ("add_bos_token", ctypes.c_int),
        ("reserved", ctypes.c_uint32 * 16),
    ]


@dataclass(frozen=True)
class ArchInfo:
    kind: int
    name: str
    cpu_cores: int
    gpu_cores: int
    unified_memory_bytes: int
    recommended_max_context: int


@dataclass(frozen=True)
class RuntimeOptions:
    target_arch: int = 0
    context_tokens: int = 512
    batch_tokens: int = 128
    memory_limit_bytes: int = 0
    kernel_cache_dir: Optional[str] = None
    flags: int = 0

    def _to_c(self) -> _RuntimeOptions:
        cache_dir = self.kernel_cache_dir.encode() if self.kernel_cache_dir else None
        return _RuntimeOptions(
            TE_ABI_VERSION,
            self.target_arch,
            self.context_tokens,
            self.batch_tokens,
            self.memory_limit_bytes,
            cache_dir,
            self.flags,
        )


@dataclass(frozen=True)
class KernelPlan:
    target_arch: int
    quant_mask: tuple[int, int]
    optimized_quant_mask: tuple[int, int]
    vector_op_mask: int
    optimization_flags: int
    q4_prefill_batch_tile: int
    q4_decode_row_tile: int
    q8_lm_head_row_tile: int
    dot_threads: int
    preferred_alignment_bytes: int
    max_context_tokens: int
    memory_budget_bytes: int
    metal_function_suffix: str


@dataclass(frozen=True)
class Capabilities:
    known_quant_mask: tuple[int, int]
    optimized_quant_mask: tuple[int, int]
    vector_op_mask: int
    optimization_flags: int
    preferred_alignment_bytes: int
    backend_name: str
    notes: str


@dataclass(frozen=True)
class ModelInfo:
    gguf_version: int
    metadata_kv_count: int
    tensor_count: int
    tensor_data_offset: int
    tensor_data_bytes: int
    parameter_count: int
    file_size_bytes: int
    name: str
    architecture: str
    context_length: int
    embedding_length: int
    block_count: int
    feed_forward_length: int
    attention_head_count: int
    attention_head_count_kv: int
    head_dim: int
    vocab_size: int
    rms_norm_epsilon: float
    rope_freq_base: float
    quant_tensor_counts: tuple[int, ...]


@dataclass(frozen=True)
class TensorInfo:
    name: str
    quant: int
    ggml_type: int
    dims: tuple[int, ...]
    elements: int
    bytes: int
    relative_offset: int
    absolute_offset: int


@dataclass(frozen=True)
class TokenizerInfo:
    model: str
    pre: str
    token_count: int
    token_type_count: int
    merge_count: int
    bos_token_id: int
    eos_token_id: int
    padding_token_id: int
    add_bos_token: bool


_TOKEN_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p)


def _candidate_libraries() -> list[Path]:
    env = os.environ.get("TINYENGINE_LIBRARY")
    suffix = "dylib" if os.uname().sysname == "Darwin" else "so"
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path.cwd() / "c" / "build" / f"libtinyengine.{suffix}",
            here.parents[2] / "c" / "build" / f"libtinyengine.{suffix}",
            here.parent / f"libtinyengine.{suffix}",
        ]
    )
    return candidates


def _load_library() -> ctypes.CDLL:
    for candidate in _candidate_libraries():
        if candidate.is_file():
            lib = ctypes.CDLL(str(candidate))
            _configure_library(lib)
            return lib
    searched = "\n".join(str(path) for path in _candidate_libraries())
    raise TinyEngineError(f"Could not find libtinyengine. Searched:\n{searched}")


def _configure_library(lib: ctypes.CDLL) -> None:
    lib.te_strerror.argtypes = [ctypes.c_int]
    lib.te_strerror.restype = ctypes.c_char_p
    lib.te_detect_arch.argtypes = [ctypes.POINTER(_ArchInfo)]
    lib.te_detect_arch.restype = ctypes.c_int
    lib.te_get_capabilities.argtypes = [ctypes.POINTER(_Capabilities)]
    lib.te_get_capabilities.restype = ctypes.c_int
    lib.te_make_kernel_plan.argtypes = [ctypes.POINTER(_RuntimeOptions), ctypes.POINTER(_KernelPlan)]
    lib.te_make_kernel_plan.restype = ctypes.c_int
    lib.te_kernel_plan_supports_quant.argtypes = [ctypes.POINTER(_KernelPlan), ctypes.c_int]
    lib.te_kernel_plan_supports_quant.restype = ctypes.c_int
    lib.te_kernel_plan_optimizes_quant.argtypes = [ctypes.POINTER(_KernelPlan), ctypes.c_int]
    lib.te_kernel_plan_optimizes_quant.restype = ctypes.c_int
    lib.te_kernel_plan_supports_op.argtypes = [ctypes.POINTER(_KernelPlan), ctypes.c_int]
    lib.te_kernel_plan_supports_op.restype = ctypes.c_int
    lib.te_model_load_gguf.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(_RuntimeOptions),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.te_model_load_gguf.restype = ctypes.c_int
    lib.te_model_get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ModelInfo)]
    lib.te_model_get_info.restype = ctypes.c_int
    lib.te_model_get_tokenizer_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_TokenizerInfo)]
    lib.te_model_get_tokenizer_info.restype = ctypes.c_int
    lib.te_model_get_tensor_info.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(_TensorInfo),
    ]
    lib.te_model_get_tensor_info.restype = ctypes.c_int
    lib.te_format_qwen_chat_prompt.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.te_format_qwen_chat_prompt.restype = ctypes.c_int
    lib.te_model_tokenize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.te_model_tokenize.restype = ctypes.c_int
    lib.te_model_detokenize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.te_model_detokenize.restype = ctypes.c_int
    lib.te_model_read_f32_tensor.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.te_model_read_f32_tensor.restype = ctypes.c_int
    lib.te_model_dequantize_row_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.te_model_dequantize_row_f32.restype = ctypes.c_int
    lib.te_model_matvec_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.te_model_matvec_f32.restype = ctypes.c_int
    lib.te_model_free.argtypes = [ctypes.c_void_p]
    lib.te_context_create.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_RuntimeOptions),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.te_context_create.restype = ctypes.c_int
    lib.te_context_free.argtypes = [ctypes.c_void_p]
    lib.te_generate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
        _TOKEN_CALLBACK,
        ctypes.c_void_p,
    ]
    lib.te_generate.restype = ctypes.c_int
    lib.te_generate_raw.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
        _TOKEN_CALLBACK,
        ctypes.c_void_p,
    ]
    lib.te_generate_raw.restype = ctypes.c_int


def _check(lib: ctypes.CDLL, status: int) -> None:
    if status != 0:
        message = lib.te_strerror(status).decode()
        raise TinyEngineError(message)


def detect_arch() -> ArchInfo:
    lib = _load_library()
    raw = _ArchInfo()
    _check(lib, lib.te_detect_arch(ctypes.byref(raw)))
    return ArchInfo(
        kind=raw.kind,
        name=raw.name.split(b"\0", 1)[0].decode(),
        cpu_cores=raw.cpu_cores,
        gpu_cores=raw.gpu_cores,
        unified_memory_bytes=raw.unified_memory_bytes,
        recommended_max_context=raw.recommended_max_context,
    )


def make_kernel_plan(options: RuntimeOptions = RuntimeOptions()) -> KernelPlan:
    lib = _load_library()
    raw_options = options._to_c()
    raw_plan = _KernelPlan()
    _check(lib, lib.te_make_kernel_plan(ctypes.byref(raw_options), ctypes.byref(raw_plan)))
    return KernelPlan(
        target_arch=raw_plan.target_arch,
        quant_mask=(raw_plan.quant_mask[0], raw_plan.quant_mask[1]),
        optimized_quant_mask=(raw_plan.optimized_quant_mask[0], raw_plan.optimized_quant_mask[1]),
        vector_op_mask=raw_plan.vector_op_mask,
        optimization_flags=raw_plan.optimization_flags,
        q4_prefill_batch_tile=raw_plan.q4_prefill_batch_tile,
        q4_decode_row_tile=raw_plan.q4_decode_row_tile,
        q8_lm_head_row_tile=raw_plan.q8_lm_head_row_tile,
        dot_threads=raw_plan.dot_threads,
        preferred_alignment_bytes=raw_plan.preferred_alignment_bytes,
        max_context_tokens=raw_plan.max_context_tokens,
        memory_budget_bytes=raw_plan.memory_budget_bytes,
        metal_function_suffix=raw_plan.metal_function_suffix.split(b"\0", 1)[0].decode(),
    )


def capabilities() -> Capabilities:
    lib = _load_library()
    raw = _Capabilities()
    _check(lib, lib.te_get_capabilities(ctypes.byref(raw)))
    return Capabilities(
        known_quant_mask=(raw.known_quant_mask[0], raw.known_quant_mask[1]),
        optimized_quant_mask=(raw.optimized_quant_mask[0], raw.optimized_quant_mask[1]),
        vector_op_mask=raw.vector_op_mask,
        optimization_flags=raw.optimization_flags,
        preferred_alignment_bytes=raw.preferred_alignment_bytes,
        backend_name=raw.backend_name.split(b"\0", 1)[0].decode(),
        notes=raw.notes.split(b"\0", 1)[0].decode(),
    )


def format_qwen_chat_prompt(prompt: str) -> str:
    lib = _load_library()
    capacity = len(prompt.encode()) + 64
    while True:
        out = ctypes.create_string_buffer(capacity)
        written = ctypes.c_size_t()
        status = lib.te_format_qwen_chat_prompt(prompt.encode(), out, capacity, ctypes.byref(written))
        if status == 0:
            return out.value.decode()
        if written.value + 1 > capacity:
            capacity = written.value + 1
            continue
        _check(lib, status)


class Model:
    def __init__(self, gguf_path: str | os.PathLike[str], options: RuntimeOptions = RuntimeOptions()):
        self._lib = _load_library()
        self._options = options
        raw_options = options._to_c()
        self._model = ctypes.c_void_p()
        _check(
            self._lib,
            self._lib.te_model_load_gguf(
                os.fsencode(gguf_path),
                ctypes.byref(raw_options),
                ctypes.byref(self._model),
            ),
        )

    def info(self) -> ModelInfo:
        raw = _ModelInfo()
        _check(self._lib, self._lib.te_model_get_info(self._model, ctypes.byref(raw)))
        return ModelInfo(
            gguf_version=raw.gguf_version,
            metadata_kv_count=raw.metadata_kv_count,
            tensor_count=raw.tensor_count,
            tensor_data_offset=raw.tensor_data_offset,
            tensor_data_bytes=raw.tensor_data_bytes,
            parameter_count=raw.parameter_count,
            file_size_bytes=raw.file_size_bytes,
            name=raw.name.split(b"\0", 1)[0].decode(),
            architecture=raw.architecture.split(b"\0", 1)[0].decode(),
            context_length=raw.context_length,
            embedding_length=raw.embedding_length,
            block_count=raw.block_count,
            feed_forward_length=raw.feed_forward_length,
            attention_head_count=raw.attention_head_count,
            attention_head_count_kv=raw.attention_head_count_kv,
            head_dim=raw.head_dim,
            vocab_size=raw.vocab_size,
            rms_norm_epsilon=float(raw.rms_norm_epsilon),
            rope_freq_base=float(raw.rope_freq_base),
            quant_tensor_counts=tuple(raw.quant_tensor_counts),
        )

    def tensor_info(self, name: str) -> TensorInfo:
        raw = _TensorInfo()
        _check(self._lib, self._lib.te_model_get_tensor_info(self._model, name.encode(), ctypes.byref(raw)))
        return TensorInfo(
            name=raw.name.split(b"\0", 1)[0].decode(),
            quant=raw.quant,
            ggml_type=raw.ggml_type,
            dims=tuple(raw.dims[: raw.n_dims]),
            elements=raw.elements,
            bytes=raw.bytes,
            relative_offset=raw.relative_offset,
            absolute_offset=raw.absolute_offset,
        )

    def tokenizer_info(self) -> TokenizerInfo:
        raw = _TokenizerInfo()
        _check(self._lib, self._lib.te_model_get_tokenizer_info(self._model, ctypes.byref(raw)))
        return TokenizerInfo(
            model=raw.model.split(b"\0", 1)[0].decode(),
            pre=raw.pre.split(b"\0", 1)[0].decode(),
            token_count=raw.token_count,
            token_type_count=raw.token_type_count,
            merge_count=raw.merge_count,
            bos_token_id=raw.bos_token_id,
            eos_token_id=raw.eos_token_id,
            padding_token_id=raw.padding_token_id,
            add_bos_token=bool(raw.add_bos_token),
        )

    def tokenize(self, text: str, parse_special: bool = True) -> list[int]:
        capacity = max(len(text.encode()) + 16, 16)
        while True:
            out = (ctypes.c_uint32 * capacity)()
            written = ctypes.c_size_t()
            status = self._lib.te_model_tokenize(
                self._model,
                text.encode(),
                int(parse_special),
                out,
                capacity,
                ctypes.byref(written),
            )
            if status == 0:
                return list(out[: written.value])
            if written.value > capacity:
                capacity = written.value
                continue
            _check(self._lib, status)

    def detokenize(self, tokens: list[int] | tuple[int, ...], skip_special: bool = True) -> str:
        token_array = (ctypes.c_uint32 * len(tokens))(*tokens)
        capacity = max(len(tokens) * 16 + 16, 16)
        while True:
            out = ctypes.create_string_buffer(capacity)
            written = ctypes.c_size_t()
            status = self._lib.te_model_detokenize(
                self._model,
                token_array,
                len(tokens),
                int(skip_special),
                out,
                capacity,
                ctypes.byref(written),
            )
            if status == 0:
                return out.value.decode()
            if written.value + 1 > capacity:
                capacity = written.value + 1
                continue
            _check(self._lib, status)

    def read_f32_tensor(self, name: str) -> list[float]:
        info = self.tensor_info(name)
        values = (ctypes.c_float * info.elements)()
        written = ctypes.c_size_t()
        _check(
            self._lib,
            self._lib.te_model_read_f32_tensor(
                self._model,
                name.encode(),
                values,
                info.elements,
                ctypes.byref(written),
            ),
        )
        return list(values[: written.value])

    def dequantize_row(self, name: str, row_index: int) -> list[float]:
        info = self.tensor_info(name)
        if len(info.dims) != 2:
            raise TinyEngineError(f"{name} is not rank-2")
        cols = info.dims[0]
        values = (ctypes.c_float * cols)()
        written = ctypes.c_size_t()
        _check(
            self._lib,
            self._lib.te_model_dequantize_row_f32(
                self._model,
                name.encode(),
                row_index,
                values,
                cols,
                ctypes.byref(written),
            ),
        )
        return list(values[: written.value])

    def matvec(self, name: str, input_values: list[float] | tuple[float, ...]) -> list[float]:
        info = self.tensor_info(name)
        if len(info.dims) != 2:
            raise TinyEngineError(f"{name} is not rank-2")
        input_array = (ctypes.c_float * len(input_values))(*input_values)
        rows = info.dims[1]
        out = (ctypes.c_float * rows)()
        written = ctypes.c_size_t()
        _check(
            self._lib,
            self._lib.te_model_matvec_f32(
                self._model,
                name.encode(),
                input_array,
                len(input_values),
                out,
                rows,
                ctypes.byref(written),
            ),
        )
        return list(out[: written.value])

    def generate(
        self,
        prompt: str,
        max_tokens: int = 16,
        on_token: Optional[Callable[[str, int], None]] = None,
    ) -> str:
        return self._run_generate(self._lib.te_generate, prompt, max_tokens, on_token)

    def generate_raw(
        self,
        prompt: str,
        max_tokens: int = 256,
        on_token: Optional[Callable[[str, int], None]] = None,
    ) -> str:
        """Generate from a pre-formatted prompt verbatim (no chat-template wrap).

        Special tokens such as ``<|im_start|>`` / ``<|im_end|>`` in ``prompt`` are
        parsed; generation stops at EOS (``<|im_end|>``) or ``max_tokens``.
        """
        return self._run_generate(self._lib.te_generate_raw, prompt, max_tokens, on_token)

    def _run_generate(
        self,
        fn: Callable[..., int],
        prompt: str,
        max_tokens: int,
        on_token: Optional[Callable[[str, int], None]],
    ) -> str:
        raw_options = self._options._to_c()
        context = ctypes.c_void_p()
        _check(
            self._lib,
            self._lib.te_context_create(self._model, ctypes.byref(raw_options), ctypes.byref(context)),
        )
        chunks: list[str] = []

        def callback(text: bytes, token_id: int, _userdata: ctypes.c_void_p) -> None:
            chunk = text.decode(errors="replace")
            chunks.append(chunk)
            if on_token is not None:
                on_token(chunk, token_id)

        c_callback = _TOKEN_CALLBACK(callback)
        try:
            _check(
                self._lib,
                fn(
                    context,
                    prompt.encode(),
                    max_tokens,
                    c_callback,
                    None,
                ),
            )
        finally:
            self._lib.te_context_free(context)
        return "".join(chunks)

    def close(self) -> None:
        if getattr(self, "_model", None):
            self._lib.te_model_free(self._model)
            self._model = None

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
