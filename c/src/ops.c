#include "tinyengine_internal.h"
#include "metal_backend.h"

#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#define TE_GGML_TYPE_F32 0u
#define TE_GGML_TYPE_Q4_0 2u
#define TE_GGML_TYPE_Q8_0 8u
#define TE_QK4_0 32u
#define TE_QK8_0 32u
#define TE_Q4_0_BLOCK_BYTES 18u
#define TE_Q8_0_BLOCK_BYTES 34u
#define TE_ATTENTION_STACK_SCORES 1024u

typedef enum te_matvec_kind {
    TE_MATVEC_Q4_0 = 0,
    TE_MATVEC_Q8_0 = 1,
    TE_MATVEC_F32 = 2
} te_matvec_kind;

typedef struct te_matvec_task {
    te_matvec_kind kind;
    const uint8_t *data;
    const float *input;
    float *out;
    size_t cols;
    size_t row_bytes;
    size_t row_begin;
    size_t row_end;
} te_matvec_task;

static float te_f16_to_f32(uint16_t bits);
static float te_read_f32_le(const uint8_t *bytes);
static uint16_t te_read_u16_le(const uint8_t *bytes);
static te_status te_require_output(size_t required, float *out, size_t out_capacity, size_t *out_written);
static te_status te_tensor_rank2_shape(
    const te_gguf_tensor *tensor,
    size_t *cols,
    size_t *rows
);
static te_status te_dequantize_q4_0_row(
    const uint8_t *data,
    const te_gguf_tensor *tensor,
    uint64_t row_index,
    float *out
);
static te_status te_dequantize_q8_0_row(
    const uint8_t *data,
    const te_gguf_tensor *tensor,
    uint64_t row_index,
    float *out
);
static te_status te_dequantize_f32_row(
    const uint8_t *data,
    const te_gguf_tensor *tensor,
    uint64_t row_index,
    float *out
);
static float te_q4_0_dot_row(const uint8_t *row, const float *input, size_t cols);
static float te_q8_0_dot_row(const uint8_t *row, const float *input, size_t cols);
static float te_f32_dot_row(const uint8_t *row, const float *input, size_t cols);
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
static float te_q4_0_dot_row_neon(const uint8_t *row, const float *input, size_t cols);
static float te_q8_0_dot_row_neon(const uint8_t *row, const float *input, size_t cols);
static float te_sum_f32x4(float32x4_t value);
#endif
static te_status te_matvec_rows(
    te_matvec_kind kind,
    const uint8_t *data,
    const float *input,
    size_t cols,
    size_t rows,
    size_t row_bytes,
    float *out
);
static void *te_matvec_worker(void *userdata);
static size_t te_matvec_worker_count(size_t cols, size_t rows);

te_status te_model_read_f32_tensor(
    const te_model *model,
    const char *name,
    float *out,
    size_t out_capacity,
    size_t *out_written
) {
    if (model == NULL || name == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    const te_gguf_tensor *tensor = te_model_find_tensor(model, name);
    if (tensor == NULL || tensor->ggml_type != TE_GGML_TYPE_F32) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (tensor->elements > SIZE_MAX) {
        return TE_STATUS_UNSUPPORTED;
    }

    const size_t required = (size_t)tensor->elements;
    te_status status = te_require_output(required, out, out_capacity, out_written);
    if (status != TE_STATUS_OK) {
        return status;
    }

    const uint8_t *data = NULL;
    size_t data_len = 0;
    status = te_model_tensor_data(model, tensor, &data, &data_len);
    if (status != TE_STATUS_OK) {
        return status;
    }
    if (data_len != required * sizeof(float)) {
        return TE_STATUS_RUNTIME_ERROR;
    }
    for (size_t i = 0; i < required; ++i) {
        out[i] = te_read_f32_le(data + i * sizeof(float));
    }
    return TE_STATUS_OK;
}

te_status te_model_dequantize_row_f32(
    const te_model *model,
    const char *name,
    uint64_t row_index,
    float *out,
    size_t out_capacity,
    size_t *out_written
) {
    if (model == NULL || name == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    const te_gguf_tensor *tensor = te_model_find_tensor(model, name);
    if (tensor == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }

    size_t cols = 0;
    size_t rows = 0;
    te_status status = te_tensor_rank2_shape(tensor, &cols, &rows);
    if (status != TE_STATUS_OK) {
        return status;
    }
    if (row_index >= (uint64_t)rows) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    status = te_require_output(cols, out, out_capacity, out_written);
    if (status != TE_STATUS_OK) {
        return status;
    }

    const uint8_t *data = NULL;
    size_t data_len = 0;
    status = te_model_tensor_data(model, tensor, &data, &data_len);
    if (status != TE_STATUS_OK) {
        return status;
    }
    (void)data_len;

    switch (tensor->ggml_type) {
        case TE_GGML_TYPE_Q4_0:
            return te_dequantize_q4_0_row(data, tensor, row_index, out);
        case TE_GGML_TYPE_Q8_0:
            return te_dequantize_q8_0_row(data, tensor, row_index, out);
        case TE_GGML_TYPE_F32:
            return te_dequantize_f32_row(data, tensor, row_index, out);
        default:
            return TE_STATUS_UNSUPPORTED;
    }
}

te_status te_model_matvec_f32(
    const te_model *model,
    const char *name,
    const float *input,
    size_t input_len,
    float *out,
    size_t out_capacity,
    size_t *out_written
) {
    if (model == NULL || name == NULL || input == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    const te_gguf_tensor *tensor = te_model_find_tensor(model, name);
    if (tensor == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }

    size_t cols = 0;
    size_t rows = 0;
    te_status status = te_tensor_rank2_shape(tensor, &cols, &rows);
    if (status != TE_STATUS_OK) {
        return status;
    }
    if (input_len != cols) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    status = te_require_output(rows, out, out_capacity, out_written);
    if (status != TE_STATUS_OK) {
        return status;
    }

    const uint8_t *data = NULL;
    size_t data_len = 0;
    status = te_model_tensor_data(model, tensor, &data, &data_len);
    if (status != TE_STATUS_OK) {
        return status;
    }
    (void)data_len;

    switch (tensor->ggml_type) {
        case TE_GGML_TYPE_Q4_0: {
            if (cols % TE_QK4_0 != 0u) {
                return TE_STATUS_UNSUPPORTED;
            }
            const size_t blocks_per_row = cols / TE_QK4_0;
            const size_t row_bytes = blocks_per_row * TE_Q4_0_BLOCK_BYTES;
            status = te_metal_matvec_f32(
                model->mapping,
                model->mapping_len,
                tensor->absolute_offset,
                tensor->ggml_type,
                input,
                cols,
                rows,
                out);
            if (status == TE_STATUS_OK) {
                return TE_STATUS_OK;
            }
            if (status != TE_STATUS_UNSUPPORTED) {
                return status;
            }
            return te_matvec_rows(TE_MATVEC_Q4_0, data, input, cols, rows, row_bytes, out);
        }
        case TE_GGML_TYPE_Q8_0: {
            if (cols % TE_QK8_0 != 0u) {
                return TE_STATUS_UNSUPPORTED;
            }
            const size_t blocks_per_row = cols / TE_QK8_0;
            const size_t row_bytes = blocks_per_row * TE_Q8_0_BLOCK_BYTES;
            status = te_metal_matvec_f32(
                model->mapping,
                model->mapping_len,
                tensor->absolute_offset,
                tensor->ggml_type,
                input,
                cols,
                rows,
                out);
            if (status == TE_STATUS_OK) {
                return TE_STATUS_OK;
            }
            if (status != TE_STATUS_UNSUPPORTED) {
                return status;
            }
            return te_matvec_rows(TE_MATVEC_Q8_0, data, input, cols, rows, row_bytes, out);
        }
        case TE_GGML_TYPE_F32: {
            const size_t row_bytes = cols * sizeof(float);
            return te_matvec_rows(TE_MATVEC_F32, data, input, cols, rows, row_bytes, out);
        }
        default:
            return TE_STATUS_UNSUPPORTED;
    }
}

te_status te_rmsnorm_f32(
    const float *input,
    const float *weight,
    size_t len,
    float epsilon,
    float *out
) {
    if (input == NULL || weight == NULL || out == NULL || len == 0u || epsilon <= 0.0f) {
        return TE_STATUS_INVALID_ARGUMENT;
    }

    double sum_squares = 0.0;
    for (size_t i = 0; i < len; ++i) {
        const double value = (double)input[i];
        sum_squares += value * value;
    }
    const float scale = 1.0f / sqrtf((float)(sum_squares / (double)len) + epsilon);
    for (size_t i = 0; i < len; ++i) {
        out[i] = input[i] * scale * weight[i];
    }
    return TE_STATUS_OK;
}

te_status te_rope_f32(
    float *values,
    size_t heads,
    size_t head_dim,
    size_t position,
    float theta
) {
    if (values == NULL || heads == 0u || head_dim == 0u || (head_dim % 2u) != 0u || theta <= 0.0f) {
        return TE_STATUS_INVALID_ARGUMENT;
    }

    const size_t half = head_dim / 2u;
    for (size_t head = 0; head < heads; ++head) {
        const size_t offset = head * head_dim;
        for (size_t index = 0; index < half; ++index) {
            const float freq = powf(theta, -(2.0f * (float)index) / (float)head_dim);
            const float angle = (float)position * freq;
            const float sin_value = sinf(angle);
            const float cos_value = cosf(angle);
            const float a = values[offset + index];
            const float b = values[offset + index + half];
            values[offset + index] = a * cos_value - b * sin_value;
            values[offset + index + half] = b * cos_value + a * sin_value;
        }
    }
    return TE_STATUS_OK;
}

te_status te_attention_decode_f32(
    const float *query,
    const float *key_cache,
    const float *value_cache,
    size_t position,
    size_t heads,
    size_t kv_heads,
    size_t head_dim,
    float *out
) {
    if (query == NULL || key_cache == NULL || value_cache == NULL || out == NULL ||
        heads == 0u || kv_heads == 0u || head_dim == 0u || (heads % kv_heads) != 0u) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (position == SIZE_MAX) {
        return TE_STATUS_UNSUPPORTED;
    }

    const size_t kv_repeat = heads / kv_heads;
    const float scale = 1.0f / sqrtf((float)head_dim);
    float scores_stack[TE_ATTENTION_STACK_SCORES];
    const size_t score_count = position + 1u;
    float *scores = score_count <= TE_ATTENTION_STACK_SCORES
        ? scores_stack
        : (float *)malloc(score_count * sizeof(scores[0]));
    if (scores == NULL) {
        return TE_STATUS_OUT_OF_MEMORY;
    }

    for (size_t head = 0; head < heads; ++head) {
        const size_t kv_head = head / kv_repeat;
        const size_t q_offset = head * head_dim;
        for (size_t past = 0; past <= position; ++past) {
            const size_t kv_offset = (past * kv_heads + kv_head) * head_dim;
            float dot = 0.0f;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                dot += query[q_offset + dim] * key_cache[kv_offset + dim];
            }
            scores[past] = dot * scale;
        }

        float max_score = -FLT_MAX;
        for (size_t past = 0; past <= position; ++past) {
            if (scores[past] > max_score) {
                max_score = scores[past];
            }
        }
        float sum = 0.0f;
        for (size_t past = 0; past <= position; ++past) {
            scores[past] = expf(scores[past] - max_score);
            sum += scores[past];
        }
        if (sum > 0.0f) {
            for (size_t past = 0; past <= position; ++past) {
                scores[past] /= sum;
            }
        }

        for (size_t dim = 0; dim < head_dim; ++dim) {
            float value = 0.0f;
            for (size_t past = 0; past <= position; ++past) {
                const size_t kv_offset = (past * kv_heads + kv_head) * head_dim;
                value += scores[past] * value_cache[kv_offset + dim];
            }
            out[q_offset + dim] = value;
        }
    }

    if (scores != scores_stack) {
        free(scores);
    }
    return TE_STATUS_OK;
}

te_status te_swiglu_f32(const float *gate, const float *up, size_t len, float *out) {
    if (gate == NULL || up == NULL || out == NULL || len == 0u) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < len; ++index) {
        const float silu = gate[index] / (1.0f + expf(-gate[index]));
        out[index] = silu * up[index];
    }
    return TE_STATUS_OK;
}

te_status te_add_f32(const float *lhs, const float *rhs, size_t len, float *out) {
    if (lhs == NULL || rhs == NULL || out == NULL || len == 0u) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < len; ++index) {
        out[index] = lhs[index] + rhs[index];
    }
    return TE_STATUS_OK;
}

te_status te_argmax_f32(const float *values, size_t len, uint32_t *out_index) {
    if (values == NULL || out_index == NULL || len == 0u || len > UINT32_MAX) {
        return TE_STATUS_INVALID_ARGUMENT;
    }

    float best = -FLT_MAX;
    uint32_t best_index = 0;
    for (size_t i = 0; i < len; ++i) {
        if (values[i] > best) {
            best = values[i];
            best_index = (uint32_t)i;
        }
    }
    *out_index = best_index;
    return TE_STATUS_OK;
}

static float te_f16_to_f32(uint16_t bits) {
    const uint32_t sign = ((uint32_t)bits & 0x8000u) << 16u;
    uint32_t exponent = ((uint32_t)bits >> 10u) & 0x1fu;
    uint32_t mantissa = (uint32_t)bits & 0x03ffu;
    uint32_t out = 0;

    if (exponent == 0u) {
        if (mantissa == 0u) {
            out = sign;
        } else {
            int32_t unbiased_exponent = -14;
            while ((mantissa & 0x0400u) == 0u) {
                mantissa <<= 1u;
                --unbiased_exponent;
            }
            mantissa &= 0x03ffu;
            out = sign | ((uint32_t)(unbiased_exponent + 127) << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 31u) {
        out = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        out = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }

    float value = 0.0f;
    memcpy(&value, &out, sizeof(value));
    return value;
}

static float te_read_f32_le(const uint8_t *bytes) {
    const uint32_t raw = ((uint32_t)bytes[0]) |
                         ((uint32_t)bytes[1] << 8u) |
                         ((uint32_t)bytes[2] << 16u) |
                         ((uint32_t)bytes[3] << 24u);
    float value = 0.0f;
    memcpy(&value, &raw, sizeof(value));
    return value;
}

static uint16_t te_read_u16_le(const uint8_t *bytes) {
    return (uint16_t)(((uint16_t)bytes[0]) | ((uint16_t)bytes[1] << 8u));
}

static te_status te_require_output(size_t required, float *out, size_t out_capacity, size_t *out_written) {
    if (out_written == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    *out_written = required;
    if (out == NULL || out_capacity < required) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    return TE_STATUS_OK;
}

static te_status te_tensor_rank2_shape(
    const te_gguf_tensor *tensor,
    size_t *cols,
    size_t *rows
) {
    if (tensor == NULL || cols == NULL || rows == NULL || tensor->n_dims != 2u) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (tensor->dims[0] > SIZE_MAX || tensor->dims[1] > SIZE_MAX) {
        return TE_STATUS_UNSUPPORTED;
    }
    *cols = (size_t)tensor->dims[0];
    *rows = (size_t)tensor->dims[1];
    return TE_STATUS_OK;
}

static te_status te_dequantize_q4_0_row(
    const uint8_t *data,
    const te_gguf_tensor *tensor,
    uint64_t row_index,
    float *out
) {
    size_t cols = 0;
    size_t rows = 0;
    te_status status = te_tensor_rank2_shape(tensor, &cols, &rows);
    if (status != TE_STATUS_OK) {
        return status;
    }
    if (row_index >= (uint64_t)rows || cols % TE_QK4_0 != 0u) {
        return TE_STATUS_UNSUPPORTED;
    }
    const size_t blocks_per_row = cols / TE_QK4_0;
    const uint8_t *row = data + (size_t)row_index * blocks_per_row * TE_Q4_0_BLOCK_BYTES;
    for (size_t block = 0; block < blocks_per_row; ++block) {
        const uint8_t *src = row + block * TE_Q4_0_BLOCK_BYTES;
        const float scale = te_f16_to_f32(te_read_u16_le(src));
        const uint8_t *qs = src + 2u;
        for (size_t j = 0; j < 16u; ++j) {
            out[block * TE_QK4_0 + j] = ((float)(qs[j] & 0x0fu) - 8.0f) * scale;
            out[block * TE_QK4_0 + j + 16u] = ((float)(qs[j] >> 4u) - 8.0f) * scale;
        }
    }
    return TE_STATUS_OK;
}

static te_status te_dequantize_q8_0_row(
    const uint8_t *data,
    const te_gguf_tensor *tensor,
    uint64_t row_index,
    float *out
) {
    size_t cols = 0;
    size_t rows = 0;
    te_status status = te_tensor_rank2_shape(tensor, &cols, &rows);
    if (status != TE_STATUS_OK) {
        return status;
    }
    if (row_index >= (uint64_t)rows || cols % TE_QK8_0 != 0u) {
        return TE_STATUS_UNSUPPORTED;
    }
    const size_t blocks_per_row = cols / TE_QK8_0;
    const uint8_t *row = data + (size_t)row_index * blocks_per_row * TE_Q8_0_BLOCK_BYTES;
    for (size_t block = 0; block < blocks_per_row; ++block) {
        const uint8_t *src = row + block * TE_Q8_0_BLOCK_BYTES;
        const float scale = te_f16_to_f32(te_read_u16_le(src));
        const int8_t *qs = (const int8_t *)(src + 2u);
        for (size_t j = 0; j < TE_QK8_0; ++j) {
            out[block * TE_QK8_0 + j] = (float)qs[j] * scale;
        }
    }
    return TE_STATUS_OK;
}

static te_status te_dequantize_f32_row(
    const uint8_t *data,
    const te_gguf_tensor *tensor,
    uint64_t row_index,
    float *out
) {
    size_t cols = 0;
    size_t rows = 0;
    te_status status = te_tensor_rank2_shape(tensor, &cols, &rows);
    if (status != TE_STATUS_OK) {
        return status;
    }
    if (row_index >= (uint64_t)rows) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    const uint8_t *row = data + (size_t)row_index * cols * sizeof(float);
    for (size_t col = 0; col < cols; ++col) {
        out[col] = te_read_f32_le(row + col * sizeof(float));
    }
    return TE_STATUS_OK;
}

static float te_q4_0_dot_row(const uint8_t *row, const float *input, size_t cols) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return te_q4_0_dot_row_neon(row, input, cols);
#else
    float sum = 0.0f;
    const size_t blocks_per_row = cols / TE_QK4_0;
    for (size_t block = 0; block < blocks_per_row; ++block) {
        const uint8_t *src = row + block * TE_Q4_0_BLOCK_BYTES;
        const float scale = te_f16_to_f32(te_read_u16_le(src));
        const uint8_t *qs = src + 2u;
        const float *x = input + block * TE_QK4_0;
        for (size_t j = 0; j < 16u; ++j) {
            sum += (((float)(qs[j] & 0x0fu) - 8.0f) * scale) * x[j];
            sum += (((float)(qs[j] >> 4u) - 8.0f) * scale) * x[j + 16u];
        }
    }
    return sum;
#endif
}

static float te_q8_0_dot_row(const uint8_t *row, const float *input, size_t cols) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return te_q8_0_dot_row_neon(row, input, cols);
#else
    float sum = 0.0f;
    const size_t blocks_per_row = cols / TE_QK8_0;
    for (size_t block = 0; block < blocks_per_row; ++block) {
        const uint8_t *src = row + block * TE_Q8_0_BLOCK_BYTES;
        const float scale = te_f16_to_f32(te_read_u16_le(src));
        const int8_t *qs = (const int8_t *)(src + 2u);
        const float *x = input + block * TE_QK8_0;
        for (size_t j = 0; j < TE_QK8_0; ++j) {
            sum += ((float)qs[j] * scale) * x[j];
        }
    }
    return sum;
#endif
}

static float te_f32_dot_row(const uint8_t *row, const float *input, size_t cols) {
    float sum = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
        sum += te_read_f32_le(row + col * sizeof(float)) * input[col];
    }
    return sum;
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
static float te_q4_0_dot_row_neon(const uint8_t *row, const float *input, size_t cols) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    const size_t blocks_per_row = cols / TE_QK4_0;
    const int16x8_t offset = vdupq_n_s16(8);
    for (size_t block = 0; block < blocks_per_row; ++block) {
        const uint8_t *src = row + block * TE_Q4_0_BLOCK_BYTES;
        const float32x4_t scale = vdupq_n_f32(te_f16_to_f32(te_read_u16_le(src)));
        const uint8x16_t packed = vld1q_u8(src + 2u);
        const uint8x16_t low_nibbles = vandq_u8(packed, vdupq_n_u8(0x0fu));
        const uint8x16_t high_nibbles = vshrq_n_u8(packed, 4);

        int16x8_t low0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(low_nibbles)));
        int16x8_t low1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(low_nibbles)));
        int16x8_t high0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(high_nibbles)));
        int16x8_t high1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(high_nibbles)));
        low0 = vsubq_s16(low0, offset);
        low1 = vsubq_s16(low1, offset);
        high0 = vsubq_s16(high0, offset);
        high1 = vsubq_s16(high1, offset);

        const float *x = input + block * TE_QK4_0;
        acc0 = vmlaq_f32(
            acc0,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(low0))), scale),
            vld1q_f32(x));
        acc1 = vmlaq_f32(
            acc1,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(low0))), scale),
            vld1q_f32(x + 4));
        acc2 = vmlaq_f32(
            acc2,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(low1))), scale),
            vld1q_f32(x + 8));
        acc3 = vmlaq_f32(
            acc3,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(low1))), scale),
            vld1q_f32(x + 12));
        acc0 = vmlaq_f32(
            acc0,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(high0))), scale),
            vld1q_f32(x + 16));
        acc1 = vmlaq_f32(
            acc1,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(high0))), scale),
            vld1q_f32(x + 20));
        acc2 = vmlaq_f32(
            acc2,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(high1))), scale),
            vld1q_f32(x + 24));
        acc3 = vmlaq_f32(
            acc3,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(high1))), scale),
            vld1q_f32(x + 28));
    }
    return te_sum_f32x4(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
}

static float te_q8_0_dot_row_neon(const uint8_t *row, const float *input, size_t cols) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    const size_t blocks_per_row = cols / TE_QK8_0;
    for (size_t block = 0; block < blocks_per_row; ++block) {
        const uint8_t *src = row + block * TE_Q8_0_BLOCK_BYTES;
        const float32x4_t scale = vdupq_n_f32(te_f16_to_f32(te_read_u16_le(src)));
        const int8x16_t q0 = vld1q_s8((const int8_t *)(src + 2u));
        const int8x16_t q1 = vld1q_s8((const int8_t *)(src + 18u));
        const float *x = input + block * TE_QK8_0;
        acc0 = vmlaq_f32(
            acc0,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(q0))))), scale),
            vld1q_f32(x));
        acc1 = vmlaq_f32(
            acc1,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(q0))))), scale),
            vld1q_f32(x + 4));
        acc2 = vmlaq_f32(
            acc2,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(q0))))), scale),
            vld1q_f32(x + 8));
        acc3 = vmlaq_f32(
            acc3,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(q0))))), scale),
            vld1q_f32(x + 12));
        acc0 = vmlaq_f32(
            acc0,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(q1))))), scale),
            vld1q_f32(x + 16));
        acc1 = vmlaq_f32(
            acc1,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(q1))))), scale),
            vld1q_f32(x + 20));
        acc2 = vmlaq_f32(
            acc2,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(q1))))), scale),
            vld1q_f32(x + 24));
        acc3 = vmlaq_f32(
            acc3,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(q1))))), scale),
            vld1q_f32(x + 28));
    }
    return te_sum_f32x4(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
}

static float te_sum_f32x4(float32x4_t value) {
#if defined(__aarch64__)
    return vaddvq_f32(value);
#else
    const float32x2_t pair = vadd_f32(vget_low_f32(value), vget_high_f32(value));
    const float32x2_t sum = vpadd_f32(pair, pair);
    return vget_lane_f32(sum, 0);
#endif
}
#endif

static te_status te_matvec_rows(
    te_matvec_kind kind,
    const uint8_t *data,
    const float *input,
    size_t cols,
    size_t rows,
    size_t row_bytes,
    float *out
) {
    const size_t worker_count = te_matvec_worker_count(cols, rows);
    if (worker_count <= 1u) {
        te_matvec_task task = {
            .kind = kind,
            .data = data,
            .input = input,
            .out = out,
            .cols = cols,
            .row_bytes = row_bytes,
            .row_begin = 0u,
            .row_end = rows
        };
        te_matvec_worker(&task);
        return TE_STATUS_OK;
    }

    pthread_t *threads = (pthread_t *)calloc(worker_count, sizeof(threads[0]));
    te_matvec_task *tasks = (te_matvec_task *)calloc(worker_count, sizeof(tasks[0]));
    if (threads == NULL || tasks == NULL) {
        free(threads);
        free(tasks);
        return TE_STATUS_OUT_OF_MEMORY;
    }

    size_t started = 0;
    size_t sequential_begin = rows;
    for (size_t index = 0; index < worker_count; ++index) {
        const size_t begin = rows * index / worker_count;
        const size_t end = rows * (index + 1u) / worker_count;
        tasks[index] = (te_matvec_task){
            .kind = kind,
            .data = data,
            .input = input,
            .out = out,
            .cols = cols,
            .row_bytes = row_bytes,
            .row_begin = begin,
            .row_end = end
        };
        const int rc = pthread_create(&threads[index], NULL, te_matvec_worker, &tasks[index]);
        if (rc != 0) {
            sequential_begin = begin;
            break;
        }
        ++started;
    }
    for (size_t index = 0; index < started; ++index) {
        pthread_join(threads[index], NULL);
    }
    if (sequential_begin < rows) {
        te_matvec_task task = {
            .kind = kind,
            .data = data,
            .input = input,
            .out = out,
            .cols = cols,
            .row_bytes = row_bytes,
            .row_begin = sequential_begin,
            .row_end = rows
        };
        te_matvec_worker(&task);
    }

    free(threads);
    free(tasks);
    return TE_STATUS_OK;
}

static void *te_matvec_worker(void *userdata) {
    const te_matvec_task *task = (const te_matvec_task *)userdata;
    for (size_t row = task->row_begin; row < task->row_end; ++row) {
        const uint8_t *row_data = task->data + row * task->row_bytes;
        switch (task->kind) {
            case TE_MATVEC_Q4_0:
                task->out[row] = te_q4_0_dot_row(row_data, task->input, task->cols);
                break;
            case TE_MATVEC_Q8_0:
                task->out[row] = te_q8_0_dot_row(row_data, task->input, task->cols);
                break;
            case TE_MATVEC_F32:
                task->out[row] = te_f32_dot_row(row_data, task->input, task->cols);
                break;
        }
    }
    return NULL;
}

static size_t te_matvec_worker_count(size_t cols, size_t rows) {
    if (rows < 512u || cols < 512u) {
        return 1u;
    }
    long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu_count < 2) {
        return 1u;
    }
    size_t workers = (size_t)cpu_count;
    if (workers > 8u) {
        workers = 8u;
    }
    if (workers > rows) {
        workers = rows;
    }
    return workers;
}
