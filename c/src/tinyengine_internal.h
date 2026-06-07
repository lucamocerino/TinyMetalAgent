#ifndef TINYENGINE_INTERNAL_H
#define TINYENGINE_INTERNAL_H

#include "tinyengine.h"

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

typedef struct te_gguf_tensor {
    char *name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t ggml_type;
    te_quant_kind quant;
    uint64_t relative_offset;
    uint64_t absolute_offset;
    uint64_t elements;
    uint64_t bytes;
} te_gguf_tensor;

typedef struct te_token_entry {
    char *text;
    uint32_t id;
    int32_t type;
    uint8_t is_special;
} te_token_entry;

typedef struct te_merge_entry {
    char *text;
    uint32_t rank;
} te_merge_entry;

typedef struct te_string_map_entry {
    const char *key;
    uint32_t value;
    uint8_t occupied;
} te_string_map_entry;

typedef struct te_string_map {
    te_string_map_entry *entries;
    size_t capacity;
    size_t count;
} te_string_map;

struct te_model {
    char *path;
    te_runtime_options options;
    te_kernel_plan plan;
    void *mapping;
    size_t mapping_len;
    te_model_info info;
    te_tokenizer_info tokenizer;
    te_token_entry *tokens;
    uint64_t token_count;
    te_merge_entry *merges;
    uint64_t merge_count;
    te_string_map token_to_id;
    te_string_map merge_to_rank;
    te_gguf_tensor *tensors;
    uint64_t tensor_count;
    uint64_t gguf_alignment;
};

struct te_context {
    te_model *model;
    te_runtime_options options;
};

char *te_strdup(const char *value);
te_arch_kind te_arch_from_brand(const char *brand);
void te_fill_plan_for_arch(te_arch_kind arch, te_kernel_plan *plan);
int te_file_exists(const char *path);
te_status te_model_parse_gguf(te_model *model, const char *path);
void te_model_release_gguf(te_model *model);
void te_tokenizer_release(te_model *model);
te_status te_tokenizer_build_maps(te_model *model);
te_status te_qwen_generate_reference(
    te_context *context,
    const char *prompt,
    uint32_t max_tokens,
    te_token_callback callback,
    void *userdata
);
te_status te_qwen_generate_raw(
    te_context *context,
    const char *prompt,
    uint32_t max_tokens,
    te_token_callback callback,
    void *userdata
);
const te_gguf_tensor *te_model_find_tensor(const te_model *model, const char *name);
te_status te_model_tensor_data(
    const te_model *model,
    const te_gguf_tensor *tensor,
    const uint8_t **out_data,
    size_t *out_len
);
void te_quant_mask_set(uint64_t mask[TE_QUANT_MASK_WORDS], te_quant_kind quant);
int te_quant_mask_has(const uint64_t mask[TE_QUANT_MASK_WORDS], te_quant_kind quant);
uint64_t te_all_vector_ops_mask(void);
uint32_t te_default_optimization_flags(void);

#endif
