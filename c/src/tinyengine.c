#include "tinyengine_internal.h"

#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

static const char *const TE_VERSION = "0.1.0-c-abi";

const char *te_version(void) {
    return TE_VERSION;
}

const char *te_strerror(te_status status) {
    switch (status) {
        case TE_STATUS_OK:
            return "ok";
        case TE_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case TE_STATUS_IO_ERROR:
            return "I/O error";
        case TE_STATUS_UNSUPPORTED:
            return "unsupported";
        case TE_STATUS_OUT_OF_MEMORY:
            return "out of memory";
        case TE_STATUS_RUNTIME_ERROR:
            return "runtime error";
        default:
            return "unknown error";
    }
}

const char *te_quant_name(te_quant_kind quant) {
    switch (quant) {
        case TE_QUANT_F32:
            return "F32";
        case TE_QUANT_F16:
            return "F16";
        case TE_QUANT_BF16:
            return "BF16";
        case TE_QUANT_GGUF_Q4_0:
            return "GGUF_Q4_0";
        case TE_QUANT_GGUF_Q4_1:
            return "GGUF_Q4_1";
        case TE_QUANT_GGUF_Q5_0:
            return "GGUF_Q5_0";
        case TE_QUANT_GGUF_Q5_1:
            return "GGUF_Q5_1";
        case TE_QUANT_GGUF_Q8_0:
            return "GGUF_Q8_0";
        case TE_QUANT_GGUF_Q2_K:
            return "GGUF_Q2_K";
        case TE_QUANT_GGUF_Q3_K:
            return "GGUF_Q3_K";
        case TE_QUANT_GGUF_Q4_K:
            return "GGUF_Q4_K";
        case TE_QUANT_GGUF_Q5_K:
            return "GGUF_Q5_K";
        case TE_QUANT_GGUF_Q6_K:
            return "GGUF_Q6_K";
        case TE_QUANT_GGUF_Q8_K:
            return "GGUF_Q8_K";
        case TE_QUANT_GGUF_IQ2_XXS:
            return "GGUF_IQ2_XXS";
        case TE_QUANT_GGUF_IQ2_XS:
            return "GGUF_IQ2_XS";
        case TE_QUANT_GGUF_IQ3_XXS:
            return "GGUF_IQ3_XXS";
        case TE_QUANT_GGUF_IQ3_S:
            return "GGUF_IQ3_S";
        case TE_QUANT_GGUF_IQ4_NL:
            return "GGUF_IQ4_NL";
        case TE_QUANT_GGUF_IQ4_XS:
            return "GGUF_IQ4_XS";
        case TE_QUANT_MXFP4:
            return "MXFP4";
        case TE_QUANT_UNKNOWN:
        case TE_QUANT_COUNT:
        default:
            return "UNKNOWN";
    }
}

const char *te_vector_op_name(te_vector_op op) {
    switch (op) {
        case TE_OP_MATVEC:
            return "matvec";
        case TE_OP_MATMUL:
            return "matmul";
        case TE_OP_BATCHED_PREFILL:
            return "batched-prefill";
        case TE_OP_DECODE_TOKEN:
            return "decode-token";
        case TE_OP_RMSNORM:
            return "rmsnorm";
        case TE_OP_ROPE:
            return "rope";
        case TE_OP_CAUSAL_ATTENTION:
            return "causal-attention";
        case TE_OP_SWIGLU:
            return "swiglu";
        case TE_OP_RESIDUAL_ADD:
            return "residual-add";
        case TE_OP_ARGMAX:
            return "argmax";
        case TE_OP_SAMPLING:
            return "sampling";
        case TE_OP_KV_CACHE:
            return "kv-cache";
        case TE_OP_TOKENIZE:
            return "tokenize";
        case TE_OP_COUNT:
        default:
            return "unknown";
    }
}

te_runtime_options te_default_options(void) {
    te_runtime_options options;
    memset(&options, 0, sizeof(options));
    options.abi_version = TE_ABI_VERSION;
    options.target_arch = TE_ARCH_AUTO;
    options.context_tokens = 512;
    options.batch_tokens = 128;
    options.memory_limit_bytes = 0;
    options.kernel_cache_dir = NULL;
    options.flags = 0;
    return options;
}

te_status te_detect_arch(te_arch_info *out_info) {
    if (out_info == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }

    memset(out_info, 0, sizeof(*out_info));
    out_info->kind = TE_ARCH_APPLE_GENERIC;
    strncpy(out_info->name, "Apple Silicon", sizeof(out_info->name) - 1u);
    out_info->recommended_max_context = 512;

#if defined(__APPLE__)
    char brand[128];
    size_t brand_len = sizeof(brand);
    if (sysctlbyname("machdep.cpu.brand_string", brand, &brand_len, NULL, 0) == 0 && brand_len > 0) {
        brand[sizeof(brand) - 1u] = '\0';
        out_info->kind = te_arch_from_brand(brand);
        strncpy(out_info->name, brand, sizeof(out_info->name) - 1u);
        out_info->name[sizeof(out_info->name) - 1u] = '\0';
    }

    uint64_t memsize = 0;
    size_t memsize_len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &memsize_len, NULL, 0) == 0) {
        out_info->unified_memory_bytes = memsize;
        if (memsize <= 8ull * 1024ull * 1024ull * 1024ull) {
            out_info->recommended_max_context = 512;
        } else if (memsize <= 16ull * 1024ull * 1024ull * 1024ull) {
            out_info->recommended_max_context = 2048;
        } else {
            out_info->recommended_max_context = 4096;
        }
    }

    int cpu_cores = 0;
    size_t cpu_cores_len = sizeof(cpu_cores);
    if (sysctlbyname("hw.physicalcpu", &cpu_cores, &cpu_cores_len, NULL, 0) == 0 && cpu_cores > 0) {
        out_info->cpu_cores = (uint32_t)cpu_cores;
    }
#endif

    return TE_STATUS_OK;
}

te_status te_get_capabilities(te_capabilities *out_capabilities) {
    if (out_capabilities == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }

    memset(out_capabilities, 0, sizeof(*out_capabilities));
    out_capabilities->abi_version = TE_ABI_VERSION;
    for (te_quant_kind quant = TE_QUANT_F32; quant < TE_QUANT_COUNT; ++quant) {
        te_quant_mask_set(out_capabilities->known_quant_mask, quant);
    }
    te_quant_mask_set(out_capabilities->optimized_quant_mask, TE_QUANT_GGUF_Q4_0);
    te_quant_mask_set(out_capabilities->optimized_quant_mask, TE_QUANT_GGUF_Q8_0);
    out_capabilities->vector_op_mask = te_all_vector_ops_mask();
    out_capabilities->optimization_flags = te_default_optimization_flags();
    out_capabilities->preferred_alignment_bytes = 128;
    strncpy(out_capabilities->backend_name, "metal-c-abi", sizeof(out_capabilities->backend_name) - 1u);
    strncpy(
        out_capabilities->notes,
        "Qwen-first Apple Silicon runtime plan: all GGUF quant families are API-visible; Q4_0/Q8_0 are optimized first.",
        sizeof(out_capabilities->notes) - 1u
    );
    return TE_STATUS_OK;
}

te_status te_make_kernel_plan(const te_runtime_options *options, te_kernel_plan *out_plan) {
    te_runtime_options effective_options = options != NULL ? *options : te_default_options();
    if (out_plan == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (effective_options.abi_version != 0 && effective_options.abi_version != TE_ABI_VERSION) {
        return TE_STATUS_UNSUPPORTED;
    }

    te_arch_kind arch = effective_options.target_arch;
    if (arch == TE_ARCH_AUTO) {
        te_arch_info info;
        te_status status = te_detect_arch(&info);
        if (status != TE_STATUS_OK) {
            return status;
        }
        arch = info.kind;
    }

    memset(out_plan, 0, sizeof(*out_plan));
    out_plan->abi_version = TE_ABI_VERSION;
    out_plan->target_arch = arch;
    out_plan->max_context_tokens =
        effective_options.context_tokens != 0 ? effective_options.context_tokens : 512;
    out_plan->memory_budget_bytes = effective_options.memory_limit_bytes;
    for (te_quant_kind quant = TE_QUANT_F32; quant < TE_QUANT_COUNT; ++quant) {
        te_quant_mask_set(out_plan->quant_mask, quant);
    }
    te_quant_mask_set(out_plan->optimized_quant_mask, TE_QUANT_GGUF_Q4_0);
    te_quant_mask_set(out_plan->optimized_quant_mask, TE_QUANT_GGUF_Q8_0);
    out_plan->vector_op_mask = te_all_vector_ops_mask();
    out_plan->optimization_flags = te_default_optimization_flags();
    out_plan->preferred_alignment_bytes = 128;
    te_fill_plan_for_arch(arch, out_plan);
    return TE_STATUS_OK;
}

te_status te_compile_kernels(const te_runtime_options *options, const te_kernel_plan *plan) {
    if (plan == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (plan->abi_version != TE_ABI_VERSION) {
        return TE_STATUS_UNSUPPORTED;
    }
    (void)options;
    return TE_STATUS_OK;
}

int te_kernel_plan_supports_quant(const te_kernel_plan *plan, te_quant_kind quant) {
    if (plan == NULL) {
        return 0;
    }
    return te_quant_mask_has(plan->quant_mask, quant);
}

int te_kernel_plan_optimizes_quant(const te_kernel_plan *plan, te_quant_kind quant) {
    if (plan == NULL) {
        return 0;
    }
    return te_quant_mask_has(plan->optimized_quant_mask, quant);
}

int te_kernel_plan_supports_op(const te_kernel_plan *plan, te_vector_op op) {
    if (plan == NULL || op < 0 || op >= TE_OP_COUNT) {
        return 0;
    }
    return (plan->vector_op_mask & (1ull << (uint32_t)op)) != 0u;
}

te_status te_model_load_gguf(
    const char *path,
    const te_runtime_options *options,
    te_model **out_model
) {
    if (path == NULL || out_model == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (!te_file_exists(path)) {
        return TE_STATUS_IO_ERROR;
    }

    te_model *model = (te_model *)calloc(1, sizeof(*model));
    if (model == NULL) {
        return TE_STATUS_OUT_OF_MEMORY;
    }

    model->path = te_strdup(path);
    if (model->path == NULL) {
        free(model);
        return TE_STATUS_OUT_OF_MEMORY;
    }
    model->options = options != NULL ? *options : te_default_options();
    te_status status = te_make_kernel_plan(&model->options, &model->plan);
    if (status != TE_STATUS_OK) {
        te_model_free(model);
        return status;
    }
    status = te_model_parse_gguf(model, path);
    if (status != TE_STATUS_OK) {
        te_model_free(model);
        return status;
    }

    *out_model = model;
    return TE_STATUS_OK;
}

te_status te_model_get_info(const te_model *model, te_model_info *out_info) {
    if (model == NULL || out_info == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    *out_info = model->info;
    return TE_STATUS_OK;
}

te_status te_model_get_tokenizer_info(
    const te_model *model,
    te_tokenizer_info *out_info
) {
    if (model == NULL || out_info == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    *out_info = model->tokenizer;
    return TE_STATUS_OK;
}

te_status te_model_get_tensor_info(
    const te_model *model,
    const char *name,
    te_tensor_info *out_info
) {
    if (model == NULL || name == NULL || out_info == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (model->tensors == NULL) {
        return TE_STATUS_RUNTIME_ERROR;
    }

    const te_gguf_tensor *tensor = te_model_find_tensor(model, name);
    if (tensor == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }

    memset(out_info, 0, sizeof(*out_info));
    out_info->abi_version = TE_ABI_VERSION;
    out_info->quant = tensor->quant;
    out_info->ggml_type = tensor->ggml_type;
    out_info->n_dims = tensor->n_dims;
    for (uint32_t dim = 0; dim < tensor->n_dims && dim < 4u; ++dim) {
        out_info->dims[dim] = tensor->dims[dim];
    }
    out_info->elements = tensor->elements;
    out_info->bytes = tensor->bytes;
    out_info->relative_offset = tensor->relative_offset;
    out_info->absolute_offset = tensor->absolute_offset;
    strncpy(out_info->name, tensor->name, sizeof(out_info->name) - 1u);
    out_info->name[sizeof(out_info->name) - 1u] = '\0';
    return TE_STATUS_OK;
}

void te_model_free(te_model *model) {
    if (model == NULL) {
        return;
    }
    te_model_release_gguf(model);
    free(model->path);
    free(model);
}

te_status te_context_create(
    te_model *model,
    const te_runtime_options *options,
    te_context **out_context
) {
    if (model == NULL || out_context == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }

    te_context *context = (te_context *)calloc(1, sizeof(*context));
    if (context == NULL) {
        return TE_STATUS_OUT_OF_MEMORY;
    }
    context->model = model;
    context->options = options != NULL ? *options : model->options;
    *out_context = context;
    return TE_STATUS_OK;
}

void te_context_free(te_context *context) {
    free(context);
}

te_status te_generate(
    te_context *context,
    const char *prompt,
    uint32_t max_tokens,
    te_token_callback callback,
    void *userdata
) {
    if (context == NULL || prompt == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (max_tokens == 0) {
        return TE_STATUS_OK;
    }
    static const char *const pieces[] = {"Mi", " ch", "iamo", " Alex"};
    static const uint32_t ids[] = {41887u, 521u, 34214u, 8515u};
    const char *generation_mode = getenv("TINYENGINE_GENERATION");
    if (generation_mode == NULL || strcmp(generation_mode, "stub") != 0) {
        return te_qwen_generate_reference(context, prompt, max_tokens, callback, userdata);
    }

    const uint32_t count = max_tokens < 4u ? max_tokens : 4u;
    for (uint32_t index = 0; index < count; ++index) {
        if (callback != NULL) {
            callback(pieces[index], ids[index], userdata);
        }
    }
    return TE_STATUS_OK;
}

char *te_strdup(const char *value) {
    if (value == NULL) {
        return NULL;
    }
    const size_t len = strlen(value);
    char *copy = (char *)malloc(len + 1u);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, value, len + 1u);
    return copy;
}

te_arch_kind te_arch_from_brand(const char *brand) {
    if (brand == NULL) {
        return TE_ARCH_APPLE_GENERIC;
    }
    if (strstr(brand, "M4") != NULL) {
        return TE_ARCH_APPLE_M4;
    }
    if (strstr(brand, "M3") != NULL) {
        return TE_ARCH_APPLE_M3;
    }
    if (strstr(brand, "M2") != NULL) {
        return TE_ARCH_APPLE_M2;
    }
    if (strstr(brand, "M1") != NULL) {
        return TE_ARCH_APPLE_M1;
    }
    return TE_ARCH_APPLE_GENERIC;
}

void te_fill_plan_for_arch(te_arch_kind arch, te_kernel_plan *plan) {
    plan->dot_threads = 128;
    plan->q4_prefill_batch_tile = 8;
    plan->q4_decode_row_tile = 8;
    plan->q8_lm_head_row_tile = 4;
    strncpy(plan->metal_function_suffix, "apple_generic", sizeof(plan->metal_function_suffix) - 1u);

    switch (arch) {
        case TE_ARCH_APPLE_M1:
            plan->q4_prefill_batch_tile = 8;
            plan->q4_decode_row_tile = 8;
            plan->q8_lm_head_row_tile = 4;
            strncpy(plan->metal_function_suffix, "m1", sizeof(plan->metal_function_suffix) - 1u);
            break;
        case TE_ARCH_APPLE_M2:
            plan->q4_prefill_batch_tile = 8;
            plan->q4_decode_row_tile = 8;
            plan->q8_lm_head_row_tile = 4;
            strncpy(plan->metal_function_suffix, "m2", sizeof(plan->metal_function_suffix) - 1u);
            break;
        case TE_ARCH_APPLE_M3:
        case TE_ARCH_APPLE_M4:
            plan->q4_prefill_batch_tile = 16;
            plan->q4_decode_row_tile = 8;
            plan->q8_lm_head_row_tile = 4;
            strncpy(plan->metal_function_suffix, "m3_m4", sizeof(plan->metal_function_suffix) - 1u);
            break;
        case TE_ARCH_AUTO:
        case TE_ARCH_APPLE_GENERIC:
        default:
            break;
    }

    plan->metal_function_suffix[sizeof(plan->metal_function_suffix) - 1u] = '\0';
}

int te_file_exists(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    fclose(file);
    return 1;
}

void te_quant_mask_set(uint64_t mask[TE_QUANT_MASK_WORDS], te_quant_kind quant) {
    if (quant <= TE_QUANT_UNKNOWN || quant >= TE_QUANT_COUNT) {
        return;
    }
    const uint32_t bit = (uint32_t)quant;
    const uint32_t word = bit / 64u;
    const uint32_t offset = bit % 64u;
    if (word >= TE_QUANT_MASK_WORDS) {
        return;
    }
    mask[word] |= 1ull << offset;
}

int te_quant_mask_has(const uint64_t mask[TE_QUANT_MASK_WORDS], te_quant_kind quant) {
    if (quant <= TE_QUANT_UNKNOWN || quant >= TE_QUANT_COUNT) {
        return 0;
    }
    const uint32_t bit = (uint32_t)quant;
    const uint32_t word = bit / 64u;
    const uint32_t offset = bit % 64u;
    if (word >= TE_QUANT_MASK_WORDS) {
        return 0;
    }
    return (mask[word] & (1ull << offset)) != 0u;
}

uint64_t te_all_vector_ops_mask(void) {
    uint64_t mask = 0;
    for (uint32_t op = 0; op < (uint32_t)TE_OP_COUNT; ++op) {
        mask |= 1ull << op;
    }
    return mask;
}

uint32_t te_default_optimization_flags(void) {
    return TE_OPT_METAL |
           TE_OPT_RUNTIME_COMPILE |
           TE_OPT_ARCH_SPECIALIZED |
           TE_OPT_KERNEL_CACHE |
           TE_OPT_FUSED_QKV |
           TE_OPT_FUSED_Q8_ARGMAX |
           TE_OPT_GPU_RESIDENT_KV |
           TE_OPT_BATCHED_PREFILL;
}
