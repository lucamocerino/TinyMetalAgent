#ifndef TINYENGINE_METAL_BACKEND_H
#define TINYENGINE_METAL_BACKEND_H

#include "tinyengine.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

te_status te_metal_matvec_f32(
    const void *mapping,
    size_t mapping_len,
    uint64_t tensor_offset,
    uint32_t ggml_type,
    const float *input,
    size_t cols,
    size_t rows,
    float *out
);

te_status te_metal_matvec_argmax_f32(
    const void *mapping,
    size_t mapping_len,
    uint64_t tensor_offset,
    uint32_t ggml_type,
    const float *input,
    size_t cols,
    size_t rows,
    uint32_t *out_index
);

te_status te_metal_project_argmax_f32(
    const void *mapping,
    size_t mapping_len,
    uint64_t tensor_offset,
    uint32_t ggml_type,
    const float *hidden_in,
    const float *norm_weight,
    size_t cols,
    size_t rows,
    float epsilon,
    uint32_t *out_index
);

te_status te_metal_matvec2_f32(
    const void *mapping,
    size_t mapping_len,
    uint64_t tensor_a_offset,
    uint64_t tensor_b_offset,
    uint32_t ggml_type,
    const float *input,
    size_t cols,
    size_t rows,
    float *out_a,
    float *out_b
);

te_status te_metal_matvec_batch_f32(
    const void *mapping,
    size_t mapping_len,
    uint64_t tensor_offset,
    uint32_t ggml_type,
    const float *input,
    size_t batch,
    size_t cols,
    size_t rows,
    float *out
);

te_status te_metal_qkv_f32(
    const void *mapping,
    size_t mapping_len,
    uint64_t q_offset,
    uint64_t k_offset,
    uint64_t v_offset,
    uint32_t ggml_type,
    const float *input,
    size_t hidden,
    size_t kv,
    float *q_out,
    float *k_out,
    float *v_out
);

te_status te_metal_qkv_batch_f32(
    const void *mapping,
    size_t mapping_len,
    uint64_t q_offset,
    uint64_t k_offset,
    uint64_t v_offset,
    uint32_t ggml_type,
    const float *input,
    size_t batch,
    size_t hidden,
    size_t kv,
    float *q_out,
    float *k_out,
    float *v_out
);

te_status te_metal_mlp_f32(
    const void *mapping,
    size_t mapping_len,
    uint64_t gate_offset,
    uint64_t up_offset,
    uint64_t down_offset,
    uint32_t ggml_type,
    const float *input,
    size_t hidden,
    size_t ffn,
    float *out
);

te_status te_metal_post_attn_mlp_f32(
    const void *mapping,
    size_t mapping_len,
    uint64_t output_offset,
    uint64_t gate_offset,
    uint64_t up_offset,
    uint64_t down_offset,
    uint32_t ggml_type,
    const float *hidden_in,
    const float *attn,
    const float *ffn_norm_weight,
    size_t hidden,
    size_t ffn,
    float epsilon,
    float *out
);

te_status te_metal_mlp_batch_f32(
    const void *mapping,
    size_t mapping_len,
    uint64_t gate_offset,
    uint64_t up_offset,
    uint64_t down_offset,
    uint32_t ggml_type,
    const float *input,
    size_t batch,
    size_t hidden,
    size_t ffn,
    float *out
);

te_status te_metal_post_attn_mlp_batch_f32(
    const void *mapping,
    size_t mapping_len,
    uint64_t output_offset,
    uint64_t gate_offset,
    uint64_t up_offset,
    uint64_t down_offset,
    uint32_t ggml_type,
    const float *hidden_in,
    const float *attn,
    const float *ffn_norm_weight,
    size_t batch,
    size_t hidden,
    size_t ffn,
    float epsilon,
    float *out
);

te_status te_metal_decode_layer_f32(
    const void *mapping,
    size_t mapping_len,
    uint64_t q_offset,
    uint64_t k_offset,
    uint64_t v_offset,
    uint64_t output_offset,
    uint64_t gate_offset,
    uint64_t up_offset,
    uint64_t down_offset,
    uint32_t ggml_type,
    const float *hidden_in,
    const float *attn_norm_weight,
    const float *ffn_norm_weight,
    const float *q_bias,
    const float *k_bias,
    const float *v_bias,
    const float *rope_cos,
    const float *rope_sin,
    float *key_cache,
    float *value_cache,
    size_t position,
    size_t context_tokens,
    size_t hidden,
    size_t kv,
    size_t heads,
    size_t kv_heads,
    size_t head_dim,
    size_t ffn,
    float epsilon,
    float *out
);

te_status te_metal_decode_all_layers_f32(
    const void *mapping,
    size_t mapping_len,
    const uint64_t *q_offsets,
    const uint64_t *k_offsets,
    const uint64_t *v_offsets,
    const uint64_t *output_offsets,
    const uint64_t *gate_offsets,
    const uint64_t *up_offsets,
    const uint64_t *down_offsets,
    size_t layers,
    uint32_t ggml_type,
    const float *hidden_in,
    const float *attn_norm_weights,
    const float *ffn_norm_weights,
    const float *q_biases,
    const float *k_biases,
    const float *v_biases,
    const float *rope_cos,
    const float *rope_sin,
    float *key_cache,
    float *value_cache,
    size_t position,
    size_t context_tokens,
    size_t hidden,
    size_t kv,
    size_t heads,
    size_t kv_heads,
    size_t head_dim,
    size_t ffn,
    float epsilon,
    float *out,
    uint64_t head_offset,
    uint32_t head_ggml_type,
    const float *output_norm_weight,
    size_t vocab,
    uint32_t *out_token_id
);

te_status te_metal_prefill_layer_f32(
    const void *mapping,
    size_t mapping_len,
    uint64_t q_offset,
    uint64_t k_offset,
    uint64_t v_offset,
    uint64_t output_offset,
    uint64_t gate_offset,
    uint64_t up_offset,
    uint64_t down_offset,
    uint32_t ggml_type,
    const float *hidden_in,
    const float *attn_norm_weight,
    const float *ffn_norm_weight,
    const float *q_bias,
    const float *k_bias,
    const float *v_bias,
    const float *rope_cos,
    const float *rope_sin,
    float *key_cache,
    float *value_cache,
    size_t batch,
    size_t context_tokens,
    size_t hidden,
    size_t kv,
    size_t heads,
    size_t kv_heads,
    size_t head_dim,
    size_t ffn,
    float epsilon,
    float *out
);

te_status te_metal_prefill_all_layers_f32(
    const void *mapping,
    size_t mapping_len,
    const uint64_t *q_offsets,
    const uint64_t *k_offsets,
    const uint64_t *v_offsets,
    const uint64_t *output_offsets,
    const uint64_t *gate_offsets,
    const uint64_t *up_offsets,
    const uint64_t *down_offsets,
    size_t layers,
    uint32_t ggml_type,
    const float *hidden_in,
    const float *attn_norm_weights,
    const float *ffn_norm_weights,
    const float *q_biases,
    const float *k_biases,
    const float *v_biases,
    const float *rope_cos,
    const float *rope_sin,
    float *key_cache,
    float *value_cache,
    size_t batch,
    size_t context_tokens,
    size_t hidden,
    size_t kv,
    size_t heads,
    size_t kv_heads,
    size_t head_dim,
    size_t ffn,
    float epsilon,
    float *out
);

#ifdef __cplusplus
}
#endif

#endif
