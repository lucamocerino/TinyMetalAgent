#include "tinyengine_internal.h"
#include "metal_backend.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define TE_GGUF_MAGIC 0x46554747u
#define TE_GGUF_MAX_DIMS 4u
#define TE_GGUF_MAX_COUNT (1u << 20u)
#define TE_GGUF_MAX_NAME_LEN 4096u

typedef enum te_gguf_value_type {
    TE_GGUF_VALUE_UINT8 = 0,
    TE_GGUF_VALUE_INT8 = 1,
    TE_GGUF_VALUE_UINT16 = 2,
    TE_GGUF_VALUE_INT16 = 3,
    TE_GGUF_VALUE_UINT32 = 4,
    TE_GGUF_VALUE_INT32 = 5,
    TE_GGUF_VALUE_FLOAT32 = 6,
    TE_GGUF_VALUE_BOOL = 7,
    TE_GGUF_VALUE_STRING = 8,
    TE_GGUF_VALUE_ARRAY = 9,
    TE_GGUF_VALUE_UINT64 = 10,
    TE_GGUF_VALUE_INT64 = 11,
    TE_GGUF_VALUE_FLOAT64 = 12
} te_gguf_value_type;

typedef struct te_gguf_string_ref {
    const char *ptr;
    uint64_t len;
} te_gguf_string_ref;

typedef struct te_gguf_kv {
    te_gguf_string_ref key;
    uint32_t type;
    uint64_t value_pos;
} te_gguf_kv;

typedef struct te_gguf_cursor {
    const uint8_t *data;
    uint64_t size;
    uint64_t pos;
} te_gguf_cursor;

typedef struct te_gguf_tensor_type_info {
    uint32_t type;
    te_quant_kind quant;
    uint32_t block_elems;
    uint32_t block_bytes;
} te_gguf_tensor_type_info;

static const te_gguf_tensor_type_info TE_GGUF_TENSOR_TYPES[] = {
    {0u, TE_QUANT_F32, 1u, 4u},
    {1u, TE_QUANT_F16, 1u, 2u},
    {2u, TE_QUANT_GGUF_Q4_0, 32u, 18u},
    {3u, TE_QUANT_GGUF_Q4_1, 32u, 20u},
    {6u, TE_QUANT_GGUF_Q5_0, 32u, 22u},
    {7u, TE_QUANT_GGUF_Q5_1, 32u, 24u},
    {8u, TE_QUANT_GGUF_Q8_0, 32u, 34u},
    {10u, TE_QUANT_GGUF_Q2_K, 256u, 84u},
    {11u, TE_QUANT_GGUF_Q3_K, 256u, 110u},
    {12u, TE_QUANT_GGUF_Q4_K, 256u, 144u},
    {13u, TE_QUANT_GGUF_Q5_K, 256u, 176u},
    {14u, TE_QUANT_GGUF_Q6_K, 256u, 210u},
    {15u, TE_QUANT_GGUF_Q8_K, 256u, 292u},
    {16u, TE_QUANT_GGUF_IQ2_XXS, 256u, 66u},
    {17u, TE_QUANT_GGUF_IQ2_XS, 256u, 74u},
    {18u, TE_QUANT_GGUF_IQ3_XXS, 256u, 98u},
    {20u, TE_QUANT_GGUF_IQ4_NL, 256u, 50u},
    {21u, TE_QUANT_GGUF_IQ3_S, 256u, 110u},
    {23u, TE_QUANT_GGUF_IQ4_XS, 256u, 136u},
    {30u, TE_QUANT_BF16, 1u, 2u}
};

static te_status te_map_file(const char *path, void **out_mapping, size_t *out_len);
static void te_unmap_file(void *mapping, size_t mapping_len);
static te_status te_parse_gguf(te_model *model);
static te_status te_parse_metadata(te_model *model, te_gguf_cursor *cursor, te_gguf_kv *kvs);
static te_status te_parse_tensors(te_model *model, te_gguf_cursor *cursor);
static te_status te_extract_model_info(te_model *model, const te_gguf_kv *kvs);
static te_status te_parse_tokenizer_arrays(
    te_model *model,
    const te_gguf_cursor *root,
    const te_gguf_kv *kvs
);
static int te_cursor_read(te_gguf_cursor *cursor, void *out, uint64_t len);
static int te_cursor_skip(te_gguf_cursor *cursor, uint64_t len);
static int te_cursor_u32(te_gguf_cursor *cursor, uint32_t *out);
static int te_cursor_i32(te_gguf_cursor *cursor, int32_t *out);
static int te_cursor_u64(te_gguf_cursor *cursor, uint64_t *out);
static int te_cursor_i64(te_gguf_cursor *cursor, int64_t *out);
static int te_cursor_f32(te_gguf_cursor *cursor, float *out);
static int te_cursor_f64(te_gguf_cursor *cursor, double *out);
static int te_cursor_string_ref(te_gguf_cursor *cursor, te_gguf_string_ref *out);
static te_status te_skip_gguf_value(te_gguf_cursor *cursor, uint32_t type, uint32_t depth);
static uint64_t te_scalar_size(uint32_t type);
static int te_checked_mul_u64(uint64_t lhs, uint64_t rhs, uint64_t *out);
static int te_checked_add_u64(uint64_t lhs, uint64_t rhs, uint64_t *out);
static int te_align_up(uint64_t value, uint64_t alignment, uint64_t *out);
static const te_gguf_kv *te_find_kv(const te_gguf_kv *kvs, uint64_t count, const char *key);
static int te_get_u64(const te_gguf_cursor *root, const te_gguf_kv *kv, uint64_t *out);
static int te_get_f32(const te_gguf_cursor *root, const te_gguf_kv *kv, float *out);
static int te_get_bool(const te_gguf_cursor *root, const te_gguf_kv *kv, int *out);
static int te_get_string(const te_gguf_cursor *root, const te_gguf_kv *kv, char *out, size_t out_len);
static int te_get_array_len(const te_gguf_cursor *root, const te_gguf_kv *kv, uint32_t *type, uint64_t *len);
static te_status te_copy_string_array(
    const te_gguf_cursor *root,
    const te_gguf_kv *kv,
    char ***out_items,
    uint64_t *out_count
);
static te_status te_copy_i32_array(
    const te_gguf_cursor *root,
    const te_gguf_kv *kv,
    int32_t **out_items,
    uint64_t *out_count
);
static int te_get_prefixed_u32(
    const te_gguf_cursor *root,
    const te_gguf_kv *kvs,
    uint64_t count,
    const char *prefix,
    const char *suffix,
    uint32_t *out
);
static int te_get_prefixed_f32(
    const te_gguf_cursor *root,
    const te_gguf_kv *kvs,
    uint64_t count,
    const char *prefix,
    const char *suffix,
    float *out
);
static const te_gguf_tensor_type_info *te_tensor_type_info(uint32_t type);
static te_quant_kind te_quant_from_ggml_type(uint32_t type);
static int te_tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes);
static int te_string_ref_equals(te_gguf_string_ref ref, const char *value);
static char *te_string_ref_dup(te_gguf_string_ref ref);
static void te_string_ref_copy(char *out, size_t out_len, te_gguf_string_ref ref);

te_status te_model_parse_gguf(te_model *model, const char *path) {
    if (model == NULL || path == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }

    te_status status = te_map_file(path, &model->mapping, &model->mapping_len);
    if (status != TE_STATUS_OK) {
        return status;
    }

    status = te_parse_gguf(model);
    if (status != TE_STATUS_OK) {
        te_model_release_gguf(model);
    }
    return status;
}

void te_model_release_gguf(te_model *model) {
    if (model == NULL) {
        return;
    }
    if (model->tensors != NULL) {
        for (uint64_t i = 0; i < model->tensor_count; ++i) {
            free(model->tensors[i].name);
        }
        free(model->tensors);
        model->tensors = NULL;
    }
    te_tokenizer_release(model);
    model->tensor_count = 0;
    // Drop the GPU buffer/residency view of these weights before unmapping so
    // the Metal runtime never references freed memory.
    (void)te_metal_release_model(model->mapping, model->mapping_len);
    te_unmap_file(model->mapping, model->mapping_len);
    model->mapping = NULL;
    model->mapping_len = 0;
}

static te_status te_map_file(const char *path, void **out_mapping, size_t *out_len) {
    *out_mapping = NULL;
    *out_len = 0;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return TE_STATUS_IO_ERROR;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return TE_STATUS_IO_ERROR;
    }
    if ((uint64_t)st.st_size > (uint64_t)SIZE_MAX) {
        close(fd);
        return TE_STATUS_UNSUPPORTED;
    }

    void *mapping = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) {
        return errno == ENOMEM ? TE_STATUS_OUT_OF_MEMORY : TE_STATUS_IO_ERROR;
    }

    *out_mapping = mapping;
    *out_len = (size_t)st.st_size;
    return TE_STATUS_OK;
}

static void te_unmap_file(void *mapping, size_t mapping_len) {
    if (mapping != NULL && mapping_len != 0) {
        munmap(mapping, mapping_len);
    }
}

static te_status te_parse_gguf(te_model *model) {
    te_gguf_cursor cursor = {
        .data = (const uint8_t *)model->mapping,
        .size = (uint64_t)model->mapping_len,
        .pos = 0
    };

    uint32_t magic = 0;
    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;
    if (!te_cursor_u32(&cursor, &magic) ||
        !te_cursor_u32(&cursor, &model->info.gguf_version) ||
        !te_cursor_u64(&cursor, &n_tensors) ||
        !te_cursor_u64(&cursor, &n_kv)) {
        return TE_STATUS_RUNTIME_ERROR;
    }
    if (magic != TE_GGUF_MAGIC) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (model->info.gguf_version < 2u || model->info.gguf_version > 3u) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (n_kv > TE_GGUF_MAX_COUNT || n_tensors > TE_GGUF_MAX_COUNT) {
        return TE_STATUS_UNSUPPORTED;
    }

    model->info.abi_version = TE_ABI_VERSION;
    model->info.metadata_kv_count = n_kv;
    model->info.tensor_count = n_tensors;
    model->info.file_size_bytes = (uint64_t)model->mapping_len;
    model->tensor_count = n_tensors;
    model->gguf_alignment = 32;

    te_gguf_kv *kvs = (te_gguf_kv *)calloc((size_t)n_kv, sizeof(*kvs));
    if (kvs == NULL && n_kv != 0) {
        return TE_STATUS_OUT_OF_MEMORY;
    }

    te_status status = te_parse_metadata(model, &cursor, kvs);
    if (status == TE_STATUS_OK) {
        status = te_extract_model_info(model, kvs);
    }
    if (status == TE_STATUS_OK) {
        status = te_parse_tensors(model, &cursor);
    }
    if (status == TE_STATUS_OK) {
        status = te_extract_model_info(model, kvs);
    }

    free(kvs);
    return status;
}

static te_status te_parse_metadata(te_model *model, te_gguf_cursor *cursor, te_gguf_kv *kvs) {
    for (uint64_t i = 0; i < model->info.metadata_kv_count; ++i) {
        if (!te_cursor_string_ref(cursor, &kvs[i].key) || kvs[i].key.len > TE_GGUF_MAX_NAME_LEN) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        if (!te_cursor_u32(cursor, &kvs[i].type)) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        kvs[i].value_pos = cursor->pos;
        te_status status = te_skip_gguf_value(cursor, kvs[i].type, 0);
        if (status != TE_STATUS_OK) {
            return status;
        }
    }
    return TE_STATUS_OK;
}

static te_status te_parse_tensors(te_model *model, te_gguf_cursor *cursor) {
    uint64_t alignment = model->gguf_alignment != 0u ? model->gguf_alignment : 32u;
    const te_gguf_cursor root = {
        .data = (const uint8_t *)model->mapping,
        .size = (uint64_t)model->mapping_len,
        .pos = 0
    };

    model->tensors = (te_gguf_tensor *)calloc((size_t)model->tensor_count, sizeof(model->tensors[0]));
    if (model->tensors == NULL && model->tensor_count != 0) {
        return TE_STATUS_OUT_OF_MEMORY;
    }

    for (uint64_t i = 0; i < model->tensor_count; ++i) {
        te_gguf_string_ref name = {0};
        te_gguf_tensor *tensor = &model->tensors[i];
        if (!te_cursor_string_ref(cursor, &name) || name.len > TE_GGUF_MAX_NAME_LEN) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        tensor->name = te_string_ref_dup(name);
        if (tensor->name == NULL) {
            return TE_STATUS_OUT_OF_MEMORY;
        }

        if (!te_cursor_u32(cursor, &tensor->n_dims) ||
            tensor->n_dims == 0u ||
            tensor->n_dims > TE_GGUF_MAX_DIMS) {
            return TE_STATUS_UNSUPPORTED;
        }

        tensor->elements = 1;
        for (uint32_t dim = 0; dim < tensor->n_dims; ++dim) {
            if (!te_cursor_u64(cursor, &tensor->dims[dim])) {
                return TE_STATUS_RUNTIME_ERROR;
            }
            if (tensor->dims[dim] == 0u ||
                !te_checked_mul_u64(tensor->elements, tensor->dims[dim], &tensor->elements)) {
                return TE_STATUS_RUNTIME_ERROR;
            }
        }
        if (!te_cursor_u32(cursor, &tensor->ggml_type) ||
            !te_cursor_u64(cursor, &tensor->relative_offset)) {
            return TE_STATUS_RUNTIME_ERROR;
        }

        tensor->quant = te_quant_from_ggml_type(tensor->ggml_type);
        if (tensor->quant > TE_QUANT_UNKNOWN && tensor->quant < TE_QUANT_COUNT) {
            model->info.quant_tensor_counts[tensor->quant] += 1u;
        }
        if (!te_checked_add_u64(model->info.parameter_count, tensor->elements, &model->info.parameter_count)) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        if (te_tensor_nbytes(tensor->ggml_type, tensor->elements, &tensor->bytes)) {
            if (!te_checked_add_u64(model->info.tensor_data_bytes, tensor->bytes, &model->info.tensor_data_bytes)) {
                return TE_STATUS_RUNTIME_ERROR;
            }
        }
    }

    if (!te_align_up(cursor->pos, alignment, &model->info.tensor_data_offset)) {
        return TE_STATUS_RUNTIME_ERROR;
    }
    for (uint64_t i = 0; i < model->tensor_count; ++i) {
        te_gguf_tensor *tensor = &model->tensors[i];
        if (!te_checked_add_u64(
                model->info.tensor_data_offset,
                tensor->relative_offset,
                &tensor->absolute_offset)) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        if (tensor->bytes != 0u &&
            (tensor->absolute_offset > root.size ||
             tensor->bytes > root.size - tensor->absolute_offset)) {
            return TE_STATUS_RUNTIME_ERROR;
        }
    }

    return TE_STATUS_OK;
}

static te_status te_extract_model_info(te_model *model, const te_gguf_kv *kvs) {
    const te_gguf_cursor root = {
        .data = (const uint8_t *)model->mapping,
        .size = (uint64_t)model->mapping_len,
        .pos = 0
    };
    const uint64_t count = model->info.metadata_kv_count;
    model->tokenizer.abi_version = TE_ABI_VERSION;

    const te_gguf_kv *name = te_find_kv(kvs, count, "general.name");
    if (name != NULL) {
        te_get_string(&root, name, model->info.name, sizeof(model->info.name));
    }

    const te_gguf_kv *arch = te_find_kv(kvs, count, "general.architecture");
    if (arch == NULL || !te_get_string(&root, arch, model->info.architecture, sizeof(model->info.architecture))) {
        return TE_STATUS_UNSUPPORTED;
    }

    const te_gguf_kv *alignment = te_find_kv(kvs, count, "general.alignment");
    uint64_t alignment_value = 32;
    if (alignment != NULL && te_get_u64(&root, alignment, &alignment_value)) {
        if (alignment_value == 0u ||
            alignment_value > (1u << 20u) ||
            (alignment_value & (alignment_value - 1u)) != 0u) {
            return TE_STATUS_UNSUPPORTED;
        }
    }
    model->gguf_alignment = alignment_value;

    char prefix[64];
    if (snprintf(prefix, sizeof(prefix), "%s.", model->info.architecture) >= (int)sizeof(prefix)) {
        return TE_STATUS_UNSUPPORTED;
    }

    te_get_prefixed_u32(&root, kvs, count, prefix, "context_length", &model->info.context_length);
    te_get_prefixed_u32(&root, kvs, count, prefix, "embedding_length", &model->info.embedding_length);
    te_get_prefixed_u32(&root, kvs, count, prefix, "block_count", &model->info.block_count);
    te_get_prefixed_u32(&root, kvs, count, prefix, "feed_forward_length", &model->info.feed_forward_length);
    te_get_prefixed_u32(&root, kvs, count, prefix, "attention.head_count", &model->info.attention_head_count);
    if (!te_get_prefixed_u32(
            &root,
            kvs,
            count,
            prefix,
            "attention.head_count_kv",
            &model->info.attention_head_count_kv)) {
        model->info.attention_head_count_kv = model->info.attention_head_count;
    }
    te_get_prefixed_f32(
        &root,
        kvs,
        count,
        prefix,
        "attention.layer_norm_rms_epsilon",
        &model->info.rms_norm_epsilon);
    if (!te_get_prefixed_f32(&root, kvs, count, prefix, "rope.freq_base", &model->info.rope_freq_base)) {
        model->info.rope_freq_base =
            strcmp(model->info.architecture, "qwen2") == 0 ? 1000000.0f : 10000.0f;
    }

    if (model->info.embedding_length != 0u && model->info.attention_head_count != 0u) {
        if (model->tensors != NULL &&
            model->info.embedding_length % model->info.attention_head_count != 0u) {
            return TE_STATUS_UNSUPPORTED;
        }
        model->info.head_dim = model->info.embedding_length / model->info.attention_head_count;
    }

    const te_gguf_kv *tokens = te_find_kv(kvs, count, "tokenizer.ggml.tokens");
    uint32_t array_type = 0;
    uint64_t token_count = 0;
    if (tokens != NULL &&
        te_get_array_len(&root, tokens, &array_type, &token_count) &&
        array_type == TE_GGUF_VALUE_STRING &&
        token_count <= UINT32_MAX) {
        model->info.vocab_size = (uint32_t)token_count;
    }
    model->tokenizer.token_count = token_count;

    if (model->tokens == NULL && token_count != 0u) {
        te_status status = te_parse_tokenizer_arrays(model, &root, kvs);
        if (status != TE_STATUS_OK) {
            return status;
        }
    }

    const te_gguf_kv *token_types = te_find_kv(kvs, count, "tokenizer.ggml.token_type");
    uint64_t token_type_count = 0;
    if (token_types != NULL &&
        te_get_array_len(&root, token_types, &array_type, &token_type_count) &&
        (array_type == TE_GGUF_VALUE_INT32 || array_type == TE_GGUF_VALUE_UINT32)) {
        model->tokenizer.token_type_count = token_type_count;
    }

    const te_gguf_kv *merges = te_find_kv(kvs, count, "tokenizer.ggml.merges");
    uint64_t merge_count = 0;
    if (merges != NULL &&
        te_get_array_len(&root, merges, &array_type, &merge_count) &&
        array_type == TE_GGUF_VALUE_STRING) {
        model->tokenizer.merge_count = merge_count;
    }

    const te_gguf_kv *tokenizer_model = te_find_kv(kvs, count, "tokenizer.ggml.model");
    if (tokenizer_model != NULL) {
        te_get_string(&root, tokenizer_model, model->tokenizer.model, sizeof(model->tokenizer.model));
    }
    const te_gguf_kv *tokenizer_pre = te_find_kv(kvs, count, "tokenizer.ggml.pre");
    if (tokenizer_pre != NULL) {
        te_get_string(&root, tokenizer_pre, model->tokenizer.pre, sizeof(model->tokenizer.pre));
    }
    const te_gguf_kv *bos = te_find_kv(kvs, count, "tokenizer.ggml.bos_token_id");
    uint64_t token_id = 0;
    if (bos != NULL && te_get_u64(&root, bos, &token_id) && token_id <= UINT32_MAX) {
        model->tokenizer.bos_token_id = (uint32_t)token_id;
    }
    const te_gguf_kv *eos = te_find_kv(kvs, count, "tokenizer.ggml.eos_token_id");
    if (eos != NULL && te_get_u64(&root, eos, &token_id) && token_id <= UINT32_MAX) {
        model->tokenizer.eos_token_id = (uint32_t)token_id;
    }
    const te_gguf_kv *padding = te_find_kv(kvs, count, "tokenizer.ggml.padding_token_id");
    if (padding != NULL && te_get_u64(&root, padding, &token_id) && token_id <= UINT32_MAX) {
        model->tokenizer.padding_token_id = (uint32_t)token_id;
    }
    const te_gguf_kv *add_bos = te_find_kv(kvs, count, "tokenizer.ggml.add_bos_token");
    if (add_bos != NULL) {
        te_get_bool(&root, add_bos, &model->tokenizer.add_bos_token);
    }

    const te_gguf_tensor *embed = te_model_find_tensor(model, "token_embd.weight");
    if (embed != NULL && embed->n_dims >= 2u && embed->dims[1] <= UINT32_MAX) {
        if (model->info.vocab_size == 0u) {
            model->info.vocab_size = (uint32_t)embed->dims[1];
        }
        if (model->info.embedding_length == 0u && embed->dims[0] <= UINT32_MAX) {
            model->info.embedding_length = (uint32_t)embed->dims[0];
        }
    }

    if (model->tensors != NULL &&
        (model->info.embedding_length == 0u ||
         model->info.block_count == 0u ||
         model->info.feed_forward_length == 0u ||
         model->info.attention_head_count == 0u ||
         model->info.attention_head_count_kv == 0u ||
         model->info.vocab_size == 0u ||
         model->info.rms_norm_epsilon == 0.0f)) {
        return TE_STATUS_UNSUPPORTED;
    }

    return TE_STATUS_OK;
}

static te_status te_parse_tokenizer_arrays(
    te_model *model,
    const te_gguf_cursor *root,
    const te_gguf_kv *kvs
) {
    const uint64_t count = model->info.metadata_kv_count;
    const te_gguf_kv *tokens_kv = te_find_kv(kvs, count, "tokenizer.ggml.tokens");
    const te_gguf_kv *types_kv = te_find_kv(kvs, count, "tokenizer.ggml.token_type");
    const te_gguf_kv *merges_kv = te_find_kv(kvs, count, "tokenizer.ggml.merges");
    char **token_texts = NULL;
    char **merge_texts = NULL;
    int32_t *token_types = NULL;
    uint64_t token_count = 0;
    uint64_t type_count = 0;
    uint64_t merge_count = 0;

    if (tokens_kv == NULL) {
        return TE_STATUS_OK;
    }

    te_status status = te_copy_string_array(root, tokens_kv, &token_texts, &token_count);
    if (status != TE_STATUS_OK) {
        return status;
    }
    if (token_count > UINT32_MAX || token_count > SIZE_MAX) {
        status = TE_STATUS_UNSUPPORTED;
        goto fail;
    }

    if (types_kv != NULL) {
        status = te_copy_i32_array(root, types_kv, &token_types, &type_count);
        if (status != TE_STATUS_OK) {
            goto fail;
        }
        if (type_count != token_count) {
            status = TE_STATUS_UNSUPPORTED;
            goto fail;
        }
    }

    if (merges_kv != NULL) {
        status = te_copy_string_array(root, merges_kv, &merge_texts, &merge_count);
        if (status != TE_STATUS_OK) {
            goto fail;
        }
        if (merge_count > UINT32_MAX || merge_count > SIZE_MAX) {
            status = TE_STATUS_UNSUPPORTED;
            goto fail;
        }
    }

    model->tokens = (te_token_entry *)calloc((size_t)token_count, sizeof(model->tokens[0]));
    if (model->tokens == NULL && token_count != 0u) {
        status = TE_STATUS_OUT_OF_MEMORY;
        goto fail;
    }
    for (uint64_t index = 0; index < token_count; ++index) {
        model->tokens[index].text = token_texts[index];
        model->tokens[index].id = (uint32_t)index;
        model->tokens[index].type = token_types != NULL ? token_types[index] : 0;
        model->tokens[index].is_special =
            (uint8_t)(model->tokens[index].type != 1 ||
                      (strncmp(model->tokens[index].text, "<|", 2u) == 0));
        token_texts[index] = NULL;
    }
    model->token_count = token_count;
    model->tokenizer.token_count = token_count;

    model->merges = (te_merge_entry *)calloc((size_t)merge_count, sizeof(model->merges[0]));
    if (model->merges == NULL && merge_count != 0u) {
        status = TE_STATUS_OUT_OF_MEMORY;
        goto fail;
    }
    for (uint64_t index = 0; index < merge_count; ++index) {
        model->merges[index].text = merge_texts[index];
        model->merges[index].rank = (uint32_t)index;
        merge_texts[index] = NULL;
    }
    model->merge_count = merge_count;
    model->tokenizer.merge_count = merge_count;

    status = te_tokenizer_build_maps(model);

fail:
    if (token_texts != NULL) {
        for (uint64_t index = 0; index < token_count; ++index) {
            free(token_texts[index]);
        }
        free(token_texts);
    }
    if (merge_texts != NULL) {
        for (uint64_t index = 0; index < merge_count; ++index) {
            free(merge_texts[index]);
        }
        free(merge_texts);
    }
    free(token_types);
    if (status != TE_STATUS_OK) {
        te_tokenizer_release(model);
    }
    return status;
}

static int te_cursor_read(te_gguf_cursor *cursor, void *out, uint64_t len) {
    if (cursor == NULL || len > cursor->size || cursor->pos > cursor->size - len) {
        return 0;
    }
    if (out != NULL && len != 0u) {
        memcpy(out, cursor->data + cursor->pos, (size_t)len);
    }
    cursor->pos += len;
    return 1;
}

static int te_cursor_skip(te_gguf_cursor *cursor, uint64_t len) {
    return te_cursor_read(cursor, NULL, len);
}

static int te_cursor_u32(te_gguf_cursor *cursor, uint32_t *out) {
    uint8_t bytes[4];
    if (!te_cursor_read(cursor, bytes, sizeof(bytes))) {
        return 0;
    }
    *out = ((uint32_t)bytes[0]) |
           ((uint32_t)bytes[1] << 8u) |
           ((uint32_t)bytes[2] << 16u) |
           ((uint32_t)bytes[3] << 24u);
    return 1;
}

static int te_cursor_i32(te_gguf_cursor *cursor, int32_t *out) {
    uint32_t raw = 0;
    if (!te_cursor_u32(cursor, &raw)) {
        return 0;
    }
    memcpy(out, &raw, sizeof(raw));
    return 1;
}

static int te_cursor_u64(te_gguf_cursor *cursor, uint64_t *out) {
    uint8_t bytes[8];
    if (!te_cursor_read(cursor, bytes, sizeof(bytes))) {
        return 0;
    }
    *out = ((uint64_t)bytes[0]) |
           ((uint64_t)bytes[1] << 8u) |
           ((uint64_t)bytes[2] << 16u) |
           ((uint64_t)bytes[3] << 24u) |
           ((uint64_t)bytes[4] << 32u) |
           ((uint64_t)bytes[5] << 40u) |
           ((uint64_t)bytes[6] << 48u) |
           ((uint64_t)bytes[7] << 56u);
    return 1;
}

static int te_cursor_i64(te_gguf_cursor *cursor, int64_t *out) {
    uint64_t raw = 0;
    if (!te_cursor_u64(cursor, &raw)) {
        return 0;
    }
    memcpy(out, &raw, sizeof(raw));
    return 1;
}

static int te_cursor_f32(te_gguf_cursor *cursor, float *out) {
    uint32_t raw = 0;
    if (!te_cursor_u32(cursor, &raw)) {
        return 0;
    }
    memcpy(out, &raw, sizeof(raw));
    return 1;
}

static int te_cursor_f64(te_gguf_cursor *cursor, double *out) {
    uint64_t raw = 0;
    if (!te_cursor_u64(cursor, &raw)) {
        return 0;
    }
    memcpy(out, &raw, sizeof(raw));
    return 1;
}

static int te_cursor_string_ref(te_gguf_cursor *cursor, te_gguf_string_ref *out) {
    uint64_t len = 0;
    if (!te_cursor_u64(cursor, &len) ||
        len > cursor->size ||
        cursor->pos > cursor->size - len) {
        return 0;
    }
    out->ptr = (const char *)(cursor->data + cursor->pos);
    out->len = len;
    cursor->pos += len;
    return 1;
}

static te_status te_skip_gguf_value(te_gguf_cursor *cursor, uint32_t type, uint32_t depth) {
    if (depth > 8u) {
        return TE_STATUS_UNSUPPORTED;
    }

    const uint64_t scalar = te_scalar_size(type);
    if (scalar != 0u) {
        return te_cursor_skip(cursor, scalar) ? TE_STATUS_OK : TE_STATUS_RUNTIME_ERROR;
    }
    if (type == TE_GGUF_VALUE_STRING) {
        te_gguf_string_ref ignored = {0};
        return te_cursor_string_ref(cursor, &ignored) ? TE_STATUS_OK : TE_STATUS_RUNTIME_ERROR;
    }
    if (type == TE_GGUF_VALUE_ARRAY) {
        uint32_t element_type = 0;
        uint64_t len = 0;
        if (!te_cursor_u32(cursor, &element_type) || !te_cursor_u64(cursor, &len)) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        const uint64_t element_size = te_scalar_size(element_type);
        if (element_size != 0u) {
            uint64_t bytes = 0;
            if (!te_checked_mul_u64(len, element_size, &bytes)) {
                return TE_STATUS_RUNTIME_ERROR;
            }
            return te_cursor_skip(cursor, bytes) ? TE_STATUS_OK : TE_STATUS_RUNTIME_ERROR;
        }
        for (uint64_t i = 0; i < len; ++i) {
            te_status status = te_skip_gguf_value(cursor, element_type, depth + 1u);
            if (status != TE_STATUS_OK) {
                return status;
            }
        }
        return TE_STATUS_OK;
    }
    return TE_STATUS_UNSUPPORTED;
}

static uint64_t te_scalar_size(uint32_t type) {
    switch (type) {
        case TE_GGUF_VALUE_UINT8:
        case TE_GGUF_VALUE_INT8:
        case TE_GGUF_VALUE_BOOL:
            return 1;
        case TE_GGUF_VALUE_UINT16:
        case TE_GGUF_VALUE_INT16:
            return 2;
        case TE_GGUF_VALUE_UINT32:
        case TE_GGUF_VALUE_INT32:
        case TE_GGUF_VALUE_FLOAT32:
            return 4;
        case TE_GGUF_VALUE_UINT64:
        case TE_GGUF_VALUE_INT64:
        case TE_GGUF_VALUE_FLOAT64:
            return 8;
        default:
            return 0;
    }
}

static int te_checked_mul_u64(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (lhs != 0u && rhs > UINT64_MAX / lhs) {
        return 0;
    }
    *out = lhs * rhs;
    return 1;
}

static int te_checked_add_u64(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (rhs > UINT64_MAX - lhs) {
        return 0;
    }
    *out = lhs + rhs;
    return 1;
}

static int te_align_up(uint64_t value, uint64_t alignment, uint64_t *out) {
    uint64_t add = 0;
    uint64_t sum = 0;
    if (alignment == 0u) {
        return 0;
    }
    add = alignment - 1u;
    if (!te_checked_add_u64(value, add, &sum)) {
        return 0;
    }
    *out = sum / alignment * alignment;
    return 1;
}

static const te_gguf_kv *te_find_kv(const te_gguf_kv *kvs, uint64_t count, const char *key) {
    if (kvs == NULL || key == NULL) {
        return NULL;
    }
    for (uint64_t i = 0; i < count; ++i) {
        if (te_string_ref_equals(kvs[i].key, key)) {
            return &kvs[i];
        }
    }
    return NULL;
}

static int te_get_u64(const te_gguf_cursor *root, const te_gguf_kv *kv, uint64_t *out) {
    te_gguf_cursor cursor = *root;
    cursor.pos = kv->value_pos;
    switch (kv->type) {
        case TE_GGUF_VALUE_UINT32: {
            uint32_t value = 0;
            if (!te_cursor_u32(&cursor, &value)) {
                return 0;
            }
            *out = value;
            return 1;
        }
        case TE_GGUF_VALUE_UINT64:
            return te_cursor_u64(&cursor, out);
        case TE_GGUF_VALUE_INT32: {
            int32_t value = 0;
            if (!te_cursor_i32(&cursor, &value) || value < 0) {
                return 0;
            }
            *out = (uint64_t)value;
            return 1;
        }
        case TE_GGUF_VALUE_INT64: {
            int64_t value = 0;
            if (!te_cursor_i64(&cursor, &value) || value < 0) {
                return 0;
            }
            *out = (uint64_t)value;
            return 1;
        }
        default:
            return 0;
    }
}

static int te_get_f32(const te_gguf_cursor *root, const te_gguf_kv *kv, float *out) {
    te_gguf_cursor cursor = *root;
    cursor.pos = kv->value_pos;
    if (kv->type == TE_GGUF_VALUE_FLOAT32) {
        return te_cursor_f32(&cursor, out);
    }
    if (kv->type == TE_GGUF_VALUE_FLOAT64) {
        double value = 0.0;
        if (!te_cursor_f64(&cursor, &value)) {
            return 0;
        }
        *out = (float)value;
        return 1;
    }
    return 0;
}

static int te_get_bool(const te_gguf_cursor *root, const te_gguf_kv *kv, int *out) {
    te_gguf_cursor cursor = *root;
    uint8_t value = 0;
    cursor.pos = kv->value_pos;
    if (kv->type != TE_GGUF_VALUE_BOOL || !te_cursor_read(&cursor, &value, sizeof(value))) {
        return 0;
    }
    *out = value != 0u;
    return 1;
}

static int te_get_string(const te_gguf_cursor *root, const te_gguf_kv *kv, char *out, size_t out_len) {
    te_gguf_cursor cursor = *root;
    te_gguf_string_ref value = {0};
    cursor.pos = kv->value_pos;
    if (kv->type != TE_GGUF_VALUE_STRING || !te_cursor_string_ref(&cursor, &value)) {
        return 0;
    }
    te_string_ref_copy(out, out_len, value);
    return 1;
}

static int te_get_array_len(const te_gguf_cursor *root, const te_gguf_kv *kv, uint32_t *type, uint64_t *len) {
    te_gguf_cursor cursor = *root;
    cursor.pos = kv->value_pos;
    if (kv->type != TE_GGUF_VALUE_ARRAY ||
        !te_cursor_u32(&cursor, type) ||
        !te_cursor_u64(&cursor, len)) {
        return 0;
    }
    return 1;
}

static te_status te_copy_string_array(
    const te_gguf_cursor *root,
    const te_gguf_kv *kv,
    char ***out_items,
    uint64_t *out_count
) {
    te_gguf_cursor cursor = *root;
    uint32_t element_type = 0;
    uint64_t count = 0;
    char **items = NULL;

    if (out_items == NULL || out_count == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    *out_items = NULL;
    *out_count = 0;

    cursor.pos = kv->value_pos;
    if (kv->type != TE_GGUF_VALUE_ARRAY ||
        !te_cursor_u32(&cursor, &element_type) ||
        !te_cursor_u64(&cursor, &count) ||
        element_type != TE_GGUF_VALUE_STRING ||
        count > SIZE_MAX) {
        return TE_STATUS_UNSUPPORTED;
    }

    items = (char **)calloc((size_t)count, sizeof(items[0]));
    if (items == NULL && count != 0u) {
        return TE_STATUS_OUT_OF_MEMORY;
    }
    for (uint64_t index = 0; index < count; ++index) {
        te_gguf_string_ref value = {0};
        if (!te_cursor_string_ref(&cursor, &value)) {
            for (uint64_t cleanup = 0; cleanup < index; ++cleanup) {
                free(items[cleanup]);
            }
            free(items);
            return TE_STATUS_RUNTIME_ERROR;
        }
        items[index] = te_string_ref_dup(value);
        if (items[index] == NULL) {
            for (uint64_t cleanup = 0; cleanup < index; ++cleanup) {
                free(items[cleanup]);
            }
            free(items);
            return TE_STATUS_OUT_OF_MEMORY;
        }
    }

    *out_items = items;
    *out_count = count;
    return TE_STATUS_OK;
}

static te_status te_copy_i32_array(
    const te_gguf_cursor *root,
    const te_gguf_kv *kv,
    int32_t **out_items,
    uint64_t *out_count
) {
    te_gguf_cursor cursor = *root;
    uint32_t element_type = 0;
    uint64_t count = 0;
    int32_t *items = NULL;

    if (out_items == NULL || out_count == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    *out_items = NULL;
    *out_count = 0;

    cursor.pos = kv->value_pos;
    if (kv->type != TE_GGUF_VALUE_ARRAY ||
        !te_cursor_u32(&cursor, &element_type) ||
        !te_cursor_u64(&cursor, &count) ||
        element_type != TE_GGUF_VALUE_INT32 ||
        count > SIZE_MAX / sizeof(items[0])) {
        return TE_STATUS_UNSUPPORTED;
    }

    items = (int32_t *)calloc((size_t)count, sizeof(items[0]));
    if (items == NULL && count != 0u) {
        return TE_STATUS_OUT_OF_MEMORY;
    }
    for (uint64_t index = 0; index < count; ++index) {
        if (!te_cursor_i32(&cursor, &items[index])) {
            free(items);
            return TE_STATUS_RUNTIME_ERROR;
        }
    }

    *out_items = items;
    *out_count = count;
    return TE_STATUS_OK;
}

static int te_get_prefixed_u32(
    const te_gguf_cursor *root,
    const te_gguf_kv *kvs,
    uint64_t count,
    const char *prefix,
    const char *suffix,
    uint32_t *out
) {
    char key[128];
    uint64_t value = 0;
    if (snprintf(key, sizeof(key), "%s%s", prefix, suffix) >= (int)sizeof(key)) {
        return 0;
    }
    const te_gguf_kv *kv = te_find_kv(kvs, count, key);
    if (kv == NULL || !te_get_u64(root, kv, &value) || value > UINT32_MAX) {
        return 0;
    }
    *out = (uint32_t)value;
    return 1;
}

static int te_get_prefixed_f32(
    const te_gguf_cursor *root,
    const te_gguf_kv *kvs,
    uint64_t count,
    const char *prefix,
    const char *suffix,
    float *out
) {
    char key[128];
    if (snprintf(key, sizeof(key), "%s%s", prefix, suffix) >= (int)sizeof(key)) {
        return 0;
    }
    const te_gguf_kv *kv = te_find_kv(kvs, count, key);
    return kv != NULL && te_get_f32(root, kv, out);
}

static const te_gguf_tensor_type_info *te_tensor_type_info(uint32_t type) {
    const size_t count = sizeof(TE_GGUF_TENSOR_TYPES) / sizeof(TE_GGUF_TENSOR_TYPES[0]);
    for (size_t i = 0; i < count; ++i) {
        if (TE_GGUF_TENSOR_TYPES[i].type == type) {
            return &TE_GGUF_TENSOR_TYPES[i];
        }
    }
    return NULL;
}

static te_quant_kind te_quant_from_ggml_type(uint32_t type) {
    const te_gguf_tensor_type_info *info = te_tensor_type_info(type);
    return info != NULL ? info->quant : TE_QUANT_UNKNOWN;
}

static int te_tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    const te_gguf_tensor_type_info *info = te_tensor_type_info(type);
    uint64_t blocks = 0;
    if (info == NULL || info->block_elems == 0u) {
        return 0;
    }
    blocks = (elements + (uint64_t)info->block_elems - 1u) / (uint64_t)info->block_elems;
    return te_checked_mul_u64(blocks, info->block_bytes, bytes);
}

const te_gguf_tensor *te_model_find_tensor(const te_model *model, const char *name) {
    if (model == NULL || name == NULL || model->tensors == NULL) {
        return NULL;
    }
    for (uint64_t i = 0; i < model->tensor_count; ++i) {
        if (model->tensors[i].name != NULL && strcmp(model->tensors[i].name, name) == 0) {
            return &model->tensors[i];
        }
    }
    return NULL;
}

te_status te_model_tensor_data(
    const te_model *model,
    const te_gguf_tensor *tensor,
    const uint8_t **out_data,
    size_t *out_len
) {
    if (model == NULL || tensor == NULL || out_data == NULL || out_len == NULL || model->mapping == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (tensor->bytes > (uint64_t)SIZE_MAX ||
        tensor->absolute_offset > (uint64_t)model->mapping_len ||
        tensor->bytes > (uint64_t)model->mapping_len - tensor->absolute_offset) {
        return TE_STATUS_RUNTIME_ERROR;
    }
    *out_data = (const uint8_t *)model->mapping + tensor->absolute_offset;
    *out_len = (size_t)tensor->bytes;
    return TE_STATUS_OK;
}

static int te_string_ref_equals(te_gguf_string_ref ref, const char *value) {
    const size_t len = strlen(value);
    return ref.len == (uint64_t)len && memcmp(ref.ptr, value, len) == 0;
}

static char *te_string_ref_dup(te_gguf_string_ref ref) {
    if (ref.len > SIZE_MAX - 1u) {
        return NULL;
    }
    char *copy = (char *)malloc((size_t)ref.len + 1u);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, ref.ptr, (size_t)ref.len);
    copy[ref.len] = '\0';
    return copy;
}

static void te_string_ref_copy(char *out, size_t out_len, te_gguf_string_ref ref) {
    size_t len = 0;
    if (out == NULL || out_len == 0u) {
        return;
    }
    len = ref.len < (uint64_t)(out_len - 1u) ? (size_t)ref.len : out_len - 1u;
    memcpy(out, ref.ptr, len);
    out[len] = '\0';
}
