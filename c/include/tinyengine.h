#ifndef TINYENGINE_H
#define TINYENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(TINYENGINE_BUILD)
#    define TE_API __declspec(dllexport)
#  else
#    define TE_API __declspec(dllimport)
#  endif
#else
#  define TE_API __attribute__((visibility("default")))
#endif

#define TE_ABI_VERSION 3u
#define TE_QUANT_MASK_WORDS 2u

typedef enum te_status {
    TE_STATUS_OK = 0,
    TE_STATUS_INVALID_ARGUMENT = 1,
    TE_STATUS_IO_ERROR = 2,
    TE_STATUS_UNSUPPORTED = 3,
    TE_STATUS_OUT_OF_MEMORY = 4,
    TE_STATUS_RUNTIME_ERROR = 5
} te_status;

typedef enum te_arch_kind {
    TE_ARCH_AUTO = 0,
    TE_ARCH_APPLE_M1 = 1,
    TE_ARCH_APPLE_M2 = 2,
    TE_ARCH_APPLE_M3 = 3,
    TE_ARCH_APPLE_M4 = 4,
    TE_ARCH_APPLE_GENERIC = 100
} te_arch_kind;

typedef enum te_quant_kind {
    TE_QUANT_UNKNOWN = 0,
    TE_QUANT_F32 = 1,
    TE_QUANT_F16 = 2,
    TE_QUANT_BF16 = 3,
    TE_QUANT_GGUF_Q4_0 = 4,
    TE_QUANT_GGUF_Q4_1 = 5,
    TE_QUANT_GGUF_Q5_0 = 6,
    TE_QUANT_GGUF_Q5_1 = 7,
    TE_QUANT_GGUF_Q8_0 = 8,
    TE_QUANT_GGUF_Q2_K = 9,
    TE_QUANT_GGUF_Q3_K = 10,
    TE_QUANT_GGUF_Q4_K = 11,
    TE_QUANT_GGUF_Q5_K = 12,
    TE_QUANT_GGUF_Q6_K = 13,
    TE_QUANT_GGUF_Q8_K = 14,
    TE_QUANT_GGUF_IQ2_XXS = 15,
    TE_QUANT_GGUF_IQ2_XS = 16,
    TE_QUANT_GGUF_IQ3_XXS = 17,
    TE_QUANT_GGUF_IQ3_S = 18,
    TE_QUANT_GGUF_IQ4_NL = 19,
    TE_QUANT_GGUF_IQ4_XS = 20,
    TE_QUANT_MXFP4 = 21,
    TE_QUANT_COUNT = 22
} te_quant_kind;

typedef enum te_vector_op {
    TE_OP_MATVEC = 0,
    TE_OP_MATMUL = 1,
    TE_OP_BATCHED_PREFILL = 2,
    TE_OP_DECODE_TOKEN = 3,
    TE_OP_RMSNORM = 4,
    TE_OP_ROPE = 5,
    TE_OP_CAUSAL_ATTENTION = 6,
    TE_OP_SWIGLU = 7,
    TE_OP_RESIDUAL_ADD = 8,
    TE_OP_ARGMAX = 9,
    TE_OP_SAMPLING = 10,
    TE_OP_KV_CACHE = 11,
    TE_OP_TOKENIZE = 12,
    TE_OP_COUNT = 13
} te_vector_op;

typedef enum te_optimization_flag {
    TE_OPT_METAL = 1u << 0u,
    TE_OPT_RUNTIME_COMPILE = 1u << 1u,
    TE_OPT_ARCH_SPECIALIZED = 1u << 2u,
    TE_OPT_KERNEL_CACHE = 1u << 3u,
    TE_OPT_FUSED_QKV = 1u << 4u,
    TE_OPT_FUSED_Q8_ARGMAX = 1u << 5u,
    TE_OPT_GPU_RESIDENT_KV = 1u << 6u,
    TE_OPT_BATCHED_PREFILL = 1u << 7u
} te_optimization_flag;

typedef struct te_arch_info {
    te_arch_kind kind;
    char name[64];
    uint32_t cpu_cores;
    uint32_t gpu_cores;
    uint64_t unified_memory_bytes;
    uint32_t recommended_max_context;
} te_arch_info;

typedef struct te_runtime_options {
    uint32_t abi_version;
    te_arch_kind target_arch;
    uint32_t context_tokens;
    uint32_t batch_tokens;
    uint64_t memory_limit_bytes;
    const char *kernel_cache_dir;
    uint32_t flags;
} te_runtime_options;

typedef struct te_capabilities {
    uint32_t abi_version;
    uint64_t known_quant_mask[TE_QUANT_MASK_WORDS];
    uint64_t optimized_quant_mask[TE_QUANT_MASK_WORDS];
    uint64_t vector_op_mask;
    uint32_t optimization_flags;
    uint32_t preferred_alignment_bytes;
    char backend_name[32];
    char notes[160];
} te_capabilities;

typedef struct te_kernel_plan {
    uint32_t abi_version;
    te_arch_kind target_arch;
    uint64_t quant_mask[TE_QUANT_MASK_WORDS];
    uint64_t optimized_quant_mask[TE_QUANT_MASK_WORDS];
    uint64_t vector_op_mask;
    uint32_t optimization_flags;
    uint32_t q4_prefill_batch_tile;
    uint32_t q4_decode_row_tile;
    uint32_t q8_lm_head_row_tile;
    uint32_t dot_threads;
    uint32_t preferred_alignment_bytes;
    uint32_t max_context_tokens;
    uint64_t memory_budget_bytes;
    char metal_function_suffix[32];
} te_kernel_plan;

typedef struct te_model_info {
    uint32_t abi_version;
    uint32_t gguf_version;
    uint64_t metadata_kv_count;
    uint64_t tensor_count;
    uint64_t tensor_data_offset;
    uint64_t tensor_data_bytes;
    uint64_t parameter_count;
    uint64_t file_size_bytes;
    char name[128];
    char architecture[32];
    uint32_t context_length;
    uint32_t embedding_length;
    uint32_t block_count;
    uint32_t feed_forward_length;
    uint32_t attention_head_count;
    uint32_t attention_head_count_kv;
    uint32_t head_dim;
    uint32_t vocab_size;
    float rms_norm_epsilon;
    float rope_freq_base;
    uint64_t quant_tensor_counts[TE_QUANT_COUNT];
    uint32_t reserved[16];
} te_model_info;

typedef struct te_tensor_info {
    uint32_t abi_version;
    te_quant_kind quant;
    uint32_t ggml_type;
    uint32_t n_dims;
    uint64_t dims[4];
    uint64_t elements;
    uint64_t bytes;
    uint64_t relative_offset;
    uint64_t absolute_offset;
    char name[128];
    uint32_t reserved[16];
} te_tensor_info;

typedef struct te_tokenizer_info {
    uint32_t abi_version;
    char model[32];
    char pre[32];
    uint64_t token_count;
    uint64_t token_type_count;
    uint64_t merge_count;
    uint32_t bos_token_id;
    uint32_t eos_token_id;
    uint32_t padding_token_id;
    int add_bos_token;
    uint32_t reserved[16];
} te_tokenizer_info;

typedef struct te_model te_model;
typedef struct te_context te_context;

typedef void (*te_token_callback)(const char *text, uint32_t token_id, void *userdata);

TE_API const char *te_version(void);
TE_API const char *te_strerror(te_status status);
TE_API const char *te_quant_name(te_quant_kind quant);
TE_API const char *te_vector_op_name(te_vector_op op);

TE_API te_runtime_options te_default_options(void);
TE_API te_status te_detect_arch(te_arch_info *out_info);
TE_API te_status te_get_capabilities(te_capabilities *out_capabilities);
TE_API te_status te_make_kernel_plan(const te_runtime_options *options, te_kernel_plan *out_plan);
TE_API te_status te_compile_kernels(const te_runtime_options *options, const te_kernel_plan *plan);
TE_API int te_kernel_plan_supports_quant(const te_kernel_plan *plan, te_quant_kind quant);
TE_API int te_kernel_plan_optimizes_quant(const te_kernel_plan *plan, te_quant_kind quant);
TE_API int te_kernel_plan_supports_op(const te_kernel_plan *plan, te_vector_op op);

TE_API te_status te_model_load_gguf(
    const char *path,
    const te_runtime_options *options,
    te_model **out_model
);
TE_API te_status te_model_get_info(const te_model *model, te_model_info *out_info);
TE_API te_status te_model_get_tokenizer_info(
    const te_model *model,
    te_tokenizer_info *out_info
);
TE_API te_status te_model_get_tensor_info(
    const te_model *model,
    const char *name,
    te_tensor_info *out_info
);
TE_API te_status te_format_qwen_chat_prompt(
    const char *prompt,
    char *out,
    size_t out_capacity,
    size_t *out_written
);
TE_API te_status te_model_tokenize(
    const te_model *model,
    const char *text,
    int parse_special,
    uint32_t *out_tokens,
    size_t out_capacity,
    size_t *out_written
);
TE_API te_status te_model_detokenize(
    const te_model *model,
    const uint32_t *tokens,
    size_t token_count,
    int skip_special,
    char *out,
    size_t out_capacity,
    size_t *out_written
);
TE_API te_status te_model_read_f32_tensor(
    const te_model *model,
    const char *name,
    float *out,
    size_t out_capacity,
    size_t *out_written
);
TE_API te_status te_model_dequantize_row_f32(
    const te_model *model,
    const char *name,
    uint64_t row_index,
    float *out,
    size_t out_capacity,
    size_t *out_written
);
TE_API te_status te_model_matvec_f32(
    const te_model *model,
    const char *name,
    const float *input,
    size_t input_len,
    float *out,
    size_t out_capacity,
    size_t *out_written
);
TE_API void te_model_free(te_model *model);

TE_API te_status te_rmsnorm_f32(
    const float *input,
    const float *weight,
    size_t len,
    float epsilon,
    float *out
);
TE_API te_status te_rope_f32(
    float *values,
    size_t heads,
    size_t head_dim,
    size_t position,
    float theta
);
TE_API te_status te_attention_decode_f32(
    const float *query,
    const float *key_cache,
    const float *value_cache,
    size_t position,
    size_t heads,
    size_t kv_heads,
    size_t head_dim,
    float *out
);
TE_API te_status te_swiglu_f32(const float *gate, const float *up, size_t len, float *out);
TE_API te_status te_add_f32(const float *lhs, const float *rhs, size_t len, float *out);
TE_API te_status te_argmax_f32(const float *values, size_t len, uint32_t *out_index);

TE_API te_status te_context_create(
    te_model *model,
    const te_runtime_options *options,
    te_context **out_context
);
TE_API void te_context_free(te_context *context);

TE_API te_status te_generate(
    te_context *context,
    const char *prompt,
    uint32_t max_tokens,
    te_token_callback callback,
    void *userdata
);

#ifdef __cplusplus
}
#endif

#endif
