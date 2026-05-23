#include "tinyengine.h"
#include "../src/metal_backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void write_u32(FILE *file, uint32_t value) {
    unsigned char bytes[4] = {
        (unsigned char)(value & 0xffu),
        (unsigned char)((value >> 8u) & 0xffu),
        (unsigned char)((value >> 16u) & 0xffu),
        (unsigned char)((value >> 24u) & 0xffu),
    };
    if (fwrite(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
        abort();
    }
}

static void write_u64(FILE *file, uint64_t value) {
    unsigned char bytes[8];
    for (size_t index = 0; index < sizeof(bytes); ++index) {
        bytes[index] = (unsigned char)((value >> (index * 8u)) & 0xffu);
    }
    if (fwrite(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
        abort();
    }
}

static void write_f32(FILE *file, float value) {
    uint32_t raw = 0;
    memcpy(&raw, &value, sizeof(raw));
    write_u32(file, raw);
}

static void write_string(FILE *file, const char *value) {
    const size_t len = strlen(value);
    write_u64(file, (uint64_t)len);
    if (fwrite(value, 1, len, file) != len) {
        abort();
    }
}

static void write_kv_string(FILE *file, const char *key, const char *value) {
    write_string(file, key);
    write_u32(file, 8u);
    write_string(file, value);
}

static void write_kv_u32(FILE *file, const char *key, uint32_t value) {
    write_string(file, key);
    write_u32(file, 4u);
    write_u32(file, value);
}

static void write_kv_f32(FILE *file, const char *key, float value) {
    write_string(file, key);
    write_u32(file, 6u);
    write_f32(file, value);
}

static void write_kv_bool(FILE *file, const char *key, int value) {
    write_string(file, key);
    write_u32(file, 7u);
    fputc(value != 0 ? 1 : 0, file);
}

static void write_kv_array_string(FILE *file, const char *key, const char *const *values, uint64_t count) {
    write_string(file, key);
    write_u32(file, 9u);
    write_u32(file, 8u);
    write_u64(file, count);
    for (uint64_t index = 0; index < count; ++index) {
        write_string(file, values[index]);
    }
}

static void write_kv_array_i32(FILE *file, const char *key, const int32_t *values, uint64_t count) {
    write_string(file, key);
    write_u32(file, 9u);
    write_u32(file, 5u);
    write_u64(file, count);
    for (uint64_t index = 0; index < count; ++index) {
        write_u32(file, (uint32_t)values[index]);
    }
}

static void write_tensor_info(
    FILE *file,
    const char *name,
    uint32_t n_dims,
    const uint64_t *dims,
    uint32_t ggml_type,
    uint64_t offset
) {
    write_string(file, name);
    write_u32(file, n_dims);
    for (uint32_t index = 0; index < n_dims; ++index) {
        write_u64(file, dims[index]);
    }
    write_u32(file, ggml_type);
    write_u64(file, offset);
}

static void pad_to_alignment(FILE *file, long alignment) {
    const long pos = ftell(file);
    if (pos < 0) {
        abort();
    }
    const long rem = pos % alignment;
    if (rem == 0) {
        return;
    }
    for (long index = 0; index < alignment - rem; ++index) {
        fputc(0, file);
    }
}

static void write_q4_row(FILE *file, uint16_t scale_bits, int row_kind) {
    fputc(scale_bits & 0xffu, file);
    fputc((scale_bits >> 8u) & 0xffu, file);
    for (int index = 0; index < 16; ++index) {
        unsigned char low = 0;
        unsigned char high = 0;
        if (row_kind == 0) {
            low = (unsigned char)index;
            high = (unsigned char)(15 - index);
        } else {
            low = 8u;
            high = 8u;
        }
        fputc((int)(low | (high << 4u)), file);
    }
}

static void fill_q4_row(unsigned char *row, uint16_t scale_bits, int row_kind) {
    row[0] = (unsigned char)(scale_bits & 0xffu);
    row[1] = (unsigned char)((scale_bits >> 8u) & 0xffu);
    for (int index = 0; index < 16; ++index) {
        unsigned char low = 0;
        unsigned char high = 0;
        if (row_kind == 0) {
            low = (unsigned char)index;
            high = (unsigned char)(15 - index);
        } else {
            low = 8u;
            high = 8u;
        }
        row[2 + index] = (unsigned char)(low | (high << 4u));
    }
}

static void write_q8_row(FILE *file, uint16_t scale_bits, int row_kind) {
    fputc(scale_bits & 0xffu, file);
    fputc((scale_bits >> 8u) & 0xffu, file);
    for (int index = 0; index < 32; ++index) {
        const signed char value = row_kind == 0 ? (signed char)(index - 16) : (signed char)1;
        fputc((unsigned char)value, file);
    }
}

static void write_fixture(const char *path) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        abort();
    }

    write_u32(file, 0x46554747u);
    write_u32(file, 3u);
    write_u64(file, 3u);
    write_u64(file, 17u);

    write_kv_string(file, "general.name", "tiny-c-kernel-test");
    write_kv_string(file, "general.architecture", "qwen2");
    write_kv_u32(file, "qwen2.embedding_length", 32u);
    write_kv_u32(file, "qwen2.block_count", 1u);
    write_kv_u32(file, "qwen2.feed_forward_length", 64u);
    write_kv_u32(file, "qwen2.attention.head_count", 1u);
    write_kv_u32(file, "qwen2.attention.head_count_kv", 1u);
    write_kv_f32(file, "qwen2.attention.layer_norm_rms_epsilon", 1e-6f);
    write_kv_string(file, "tokenizer.ggml.model", "gpt2");
    write_kv_string(file, "tokenizer.ggml.pre", "qwen2");
    const char *const tokens[3] = {"a", "b", "ab"};
    write_kv_array_string(file, "tokenizer.ggml.tokens", tokens, 3u);
    const int32_t token_types[3] = {1, 1, 1};
    write_kv_array_i32(file, "tokenizer.ggml.token_type", token_types, 3u);
    const char *const merges[1] = {"a b"};
    write_kv_array_string(file, "tokenizer.ggml.merges", merges, 1u);
    write_kv_u32(file, "tokenizer.ggml.bos_token_id", 0u);
    write_kv_u32(file, "tokenizer.ggml.eos_token_id", 1u);
    write_kv_u32(file, "tokenizer.ggml.padding_token_id", 0u);
    write_kv_bool(file, "tokenizer.ggml.add_bos_token", 0);

    const uint64_t rank2[2] = {32u, 3u};
    const uint64_t rank1[1] = {32u};
    write_tensor_info(file, "token_embd.weight", 2u, rank2, 2u, 0u);
    write_tensor_info(file, "output.weight", 2u, rank2, 8u, 54u);
    write_tensor_info(file, "output_norm.weight", 1u, rank1, 0u, 156u);
    pad_to_alignment(file, 32);

    write_q4_row(file, 0x3c00u, 0);
    write_q4_row(file, 0x3800u, 1);
    write_q4_row(file, 0x3800u, 1);
    write_q8_row(file, 0x3400u, 0);
    write_q8_row(file, 0x3c00u, 1);
    write_q8_row(file, 0x3c00u, 1);
    for (uint32_t index = 0; index < 32u; ++index) {
        write_f32(file, 1.0f + (float)index / 32.0f);
    }

    if (fclose(file) != 0) {
        abort();
    }
}

static void expect_status(te_status status, const char *label) {
    if (status != TE_STATUS_OK) {
        fprintf(stderr, "%s failed: %s\n", label, te_strerror(status));
        exit(1);
    }
}

static void expect_status_one_of(te_status status, te_status first, te_status second, const char *label) {
    if (status != first && status != second) {
        fprintf(stderr, "%s failed: %s\n", label, te_strerror(status));
        exit(1);
    }
}

static void expect_close(float actual, float expected, float tolerance, const char *label) {
    if (fabsf(actual - expected) > tolerance) {
        fprintf(stderr, "%s mismatch: got %.8f expected %.8f\n", label, actual, expected);
        exit(1);
    }
}

static void expect_u32(uint32_t actual, uint32_t expected, const char *label) {
    if (actual != expected) {
        fprintf(stderr, "%s mismatch: got %u expected %u\n", label, actual, expected);
        exit(1);
    }
}

static void test_model_ops(const char *path) {
    te_model *model = NULL;
    te_runtime_options options = te_default_options();
    expect_status(te_model_load_gguf(path, &options, &model), "te_model_load_gguf");

    te_model_info model_info;
    expect_status(te_model_get_info(model, &model_info), "te_model_get_info");
    expect_u32(model_info.embedding_length, 32u, "embedding_length");
    expect_u32(model_info.vocab_size, 3u, "vocab_size");

    te_tokenizer_info tokenizer_info;
    expect_status(te_model_get_tokenizer_info(model, &tokenizer_info), "te_model_get_tokenizer_info");
    if (strcmp(tokenizer_info.model, "gpt2") != 0 || strcmp(tokenizer_info.pre, "qwen2") != 0) {
        fprintf(stderr, "tokenizer strings mismatch: model=%s pre=%s\n", tokenizer_info.model, tokenizer_info.pre);
        exit(1);
    }
    expect_u32((uint32_t)tokenizer_info.token_count, 3u, "token_count");
    expect_u32((uint32_t)tokenizer_info.token_type_count, 3u, "token_type_count");
    expect_u32((uint32_t)tokenizer_info.merge_count, 1u, "merge_count");
    expect_u32(tokenizer_info.eos_token_id, 1u, "eos_token_id");
    expect_u32(tokenizer_info.padding_token_id, 0u, "padding_token_id");
    expect_u32((uint32_t)tokenizer_info.add_bos_token, 0u, "add_bos_token");

    te_tensor_info tensor_info;
    expect_status(te_model_get_tensor_info(model, "token_embd.weight", &tensor_info), "token tensor info");
    expect_u32((uint32_t)tensor_info.dims[0], 32u, "token dims[0]");
    expect_u32((uint32_t)tensor_info.dims[1], 3u, "token dims[1]");

    uint32_t token_ids[4];
    size_t token_count = 0;
    expect_status(te_model_tokenize(model, "ab", 1, token_ids, 4u, &token_count), "tokenize ab");
    expect_u32((uint32_t)token_count, 1u, "tokenize ab count");
    expect_u32(token_ids[0], 2u, "tokenize ab id");
    char decoded[8];
    size_t decoded_len = 0;
    expect_status(te_model_detokenize(model, token_ids, token_count, 1, decoded, sizeof(decoded), &decoded_len), "detokenize ab");
    if (strcmp(decoded, "ab") != 0 || decoded_len != 2u) {
        fprintf(stderr, "detokenize mismatch: %s len=%zu\n", decoded, decoded_len);
        exit(1);
    }

    float row[32];
    size_t written = 0;
    expect_status(
        te_model_dequantize_row_f32(model, "token_embd.weight", 0u, row, 32u, &written),
        "q4 dequant row");
    if (written != 32u) {
        fprintf(stderr, "q4 dequant wrote %zu elements\n", written);
        exit(1);
    }
    for (uint32_t index = 0; index < 16u; ++index) {
        expect_close(row[index], (float)((int)index - 8), 0.0f, "q4 low nibble");
        expect_close(row[index + 16u], (float)(7 - (int)index), 0.0f, "q4 high nibble");
    }

    expect_status(
        te_model_dequantize_row_f32(model, "output.weight", 0u, row, 32u, &written),
        "q8 dequant row");
    for (uint32_t index = 0; index < 32u; ++index) {
        expect_close(row[index], (float)((int)index - 16) * 0.25f, 0.0f, "q8 signed byte");
    }

    float ones[32];
    for (uint32_t index = 0; index < 32u; ++index) {
        ones[index] = 1.0f;
    }
    float out[3];
    expect_status(te_model_matvec_f32(model, "token_embd.weight", ones, 32u, out, 3u, &written), "q4 matvec");
    expect_close(out[0], -16.0f, 0.0f, "q4 matvec row0");
    expect_close(out[1], 0.0f, 0.0f, "q4 matvec row1");

    expect_status(te_model_matvec_f32(model, "output.weight", ones, 32u, out, 3u, &written), "q8 matvec");
    expect_close(out[0], -4.0f, 0.0f, "q8 matvec row0");
    expect_close(out[1], 32.0f, 0.0f, "q8 matvec row1");

    float norm_weight[32];
    expect_status(te_model_read_f32_tensor(model, "output_norm.weight", norm_weight, 32u, &written), "read f32");
    expect_close(norm_weight[0], 1.0f, 0.0f, "f32 tensor first");
    expect_close(norm_weight[31], 1.96875f, 0.0f, "f32 tensor last");

    float normed[32];
    expect_status(te_rmsnorm_f32(ones, norm_weight, 32u, 1e-6f, normed), "rmsnorm");
    expect_close(normed[0], 0.9999995f, 1e-5f, "rmsnorm first");
    expect_close(normed[31], 1.9687490f, 1e-5f, "rmsnorm last");

    float rope_values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    expect_status(te_rope_f32(rope_values, 1u, 4u, 1u, 10000.0f), "rope");
    const float sin0 = sinf(1.0f);
    const float cos0 = cosf(1.0f);
    const float sin1 = sinf(0.01f);
    const float cos1 = cosf(0.01f);
    expect_close(rope_values[0], 1.0f * cos0 - 3.0f * sin0, 1e-6f, "rope first half");
    expect_close(rope_values[2], 3.0f * cos0 + 1.0f * sin0, 1e-6f, "rope second half");
    expect_close(rope_values[1], 2.0f * cos1 - 4.0f * sin1, 1e-6f, "rope low freq first half");
    expect_close(rope_values[3], 4.0f * cos1 + 2.0f * sin1, 1e-6f, "rope low freq second half");

    const float query[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float key_cache[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float value_cache[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float attn_out[4];
    expect_status(
        te_attention_decode_f32(query, key_cache, value_cache, 1u, 2u, 1u, 2u, attn_out),
        "attention decode");
    const float score = 1.0f / sqrtf(2.0f);
    const float exp_hi = expf(score);
    const float exp_lo = 1.0f;
    const float weight_hi = exp_hi / (exp_hi + exp_lo);
    const float weight_lo = exp_lo / (exp_hi + exp_lo);
    expect_close(attn_out[0], weight_hi * 10.0f + weight_lo * 30.0f, 1e-5f, "attention head0 dim0");
    expect_close(attn_out[1], weight_hi * 20.0f + weight_lo * 40.0f, 1e-5f, "attention head0 dim1");
    expect_close(attn_out[2], weight_lo * 10.0f + weight_hi * 30.0f, 1e-5f, "attention head1 dim0");
    expect_close(attn_out[3], weight_lo * 20.0f + weight_hi * 40.0f, 1e-5f, "attention head1 dim1");

    const float gate[2] = {0.0f, 1.0f};
    const float up_values[2] = {2.0f, 3.0f};
    float swiglu[2];
    expect_status(te_swiglu_f32(gate, up_values, 2u, swiglu), "swiglu");
    expect_close(swiglu[0], 0.0f, 0.0f, "swiglu zero");
    expect_close(swiglu[1], 3.0f / (1.0f + expf(-1.0f)), 1e-6f, "swiglu one");

    const float add_lhs[3] = {1.0f, -2.0f, 3.0f};
    const float add_rhs[3] = {4.0f, 5.0f, -6.0f};
    float add_out[3];
    expect_status(te_add_f32(add_lhs, add_rhs, 3u, add_out), "add");
    expect_close(add_out[0], 5.0f, 0.0f, "add first");
    expect_close(add_out[1], 3.0f, 0.0f, "add second");
    expect_close(add_out[2], -3.0f, 0.0f, "add third");

    const float logits[5] = {-1.0f, 2.0f, 5.0f, 5.0f, 3.0f};
    uint32_t argmax = 0;
    expect_status(te_argmax_f32(logits, 5u, &argmax), "argmax");
    expect_u32(argmax, 2u, "argmax first tie");

    te_model_free(model);
}

static void test_metal_decode_fusion_guards(void) {
    float value = 0.0f;
    const te_status status = te_metal_post_attn_mlp_f32(
        NULL,
        0u,
        0u,
        0u,
        0u,
        0u,
        2u,
        &value,
        &value,
        &value,
        1u,
        1u,
        1e-6f,
        &value);
    expect_status_one_of(status, TE_STATUS_INVALID_ARGUMENT, TE_STATUS_UNSUPPORTED, "decode fusion guard");
}

#include "generated_metal_guard_tests.inc"

static void test_metal_qkv_pair_q4(void) {
    const size_t hidden = 32u;
    const size_t kv = 8u;
    const size_t row_bytes = 18u;
    const size_t q_offset = 0u;
    const size_t k_offset = q_offset + hidden * row_bytes;
    const size_t v_offset = k_offset + kv * row_bytes;
    const size_t mapping_len = v_offset + kv * row_bytes;
    unsigned char *mapping = (unsigned char *)calloc(mapping_len, 1u);
    if (mapping == NULL) {
        abort();
    }

    for (size_t row = 0; row < hidden; ++row) {
        fill_q4_row(mapping + q_offset + row * row_bytes, 0x3c00u, 1);
    }
    for (size_t row = 0; row < kv; ++row) {
        fill_q4_row(mapping + k_offset + row * row_bytes, 0x3c00u, 0);
        fill_q4_row(mapping + v_offset + row * row_bytes, 0x3800u, 0);
    }

    float input[32];
    for (size_t index = 0; index < hidden; ++index) {
        input[index] = 1.0f;
    }
    float q_out[32];
    float k_out[8];
    float v_out[8];
    const te_status status = te_metal_qkv_f32(
        mapping,
        mapping_len,
        q_offset,
        k_offset,
        v_offset,
        2u,
        input,
        hidden,
        kv,
        q_out,
        k_out,
        v_out);
    if (status == TE_STATUS_UNSUPPORTED) {
        free(mapping);
        return;
    }
    expect_status(status, "metal qkv pair q4");
    for (size_t row = 0; row < hidden; ++row) {
        expect_close(q_out[row], 0.0f, 1e-5f, "metal qkv q row");
    }
    for (size_t row = 0; row < kv; ++row) {
        expect_close(k_out[row], -16.0f, 1e-5f, "metal qkv k row");
        expect_close(v_out[row], -8.0f, 1e-5f, "metal qkv v row");
    }
    free(mapping);
}

static void test_metal_qkv_batch_q4(void) {
    const size_t batch = 3u;
    const size_t hidden = 32u;
    const size_t kv = 8u;
    const size_t row_bytes = 18u;
    const size_t q_offset = 0u;
    const size_t k_offset = q_offset + hidden * row_bytes;
    const size_t v_offset = k_offset + kv * row_bytes;
    const size_t mapping_len = v_offset + kv * row_bytes;
    unsigned char *mapping = (unsigned char *)calloc(mapping_len, 1u);
    if (mapping == NULL) {
        abort();
    }

    for (size_t row = 0; row < hidden; ++row) {
        fill_q4_row(mapping + q_offset + row * row_bytes, 0x3c00u, 1);
    }
    for (size_t row = 0; row < kv; ++row) {
        fill_q4_row(mapping + k_offset + row * row_bytes, 0x3c00u, 0);
        fill_q4_row(mapping + v_offset + row * row_bytes, 0x3800u, 0);
    }

    float input[3 * 32];
    for (size_t index = 0; index < hidden; ++index) {
        input[index] = 1.0f;
        input[hidden + index] = 0.0f;
        input[2u * hidden + index] = 2.0f;
    }
    float q_out[3 * 32];
    float k_out[3 * 8];
    float v_out[3 * 8];
    const te_status status = te_metal_qkv_batch_f32(
        mapping,
        mapping_len,
        q_offset,
        k_offset,
        v_offset,
        2u,
        input,
        batch,
        hidden,
        kv,
        q_out,
        k_out,
        v_out);
    if (status == TE_STATUS_UNSUPPORTED) {
        free(mapping);
        return;
    }
    expect_status(status, "metal qkv batch q4");
    for (size_t batch_index = 0; batch_index < batch; ++batch_index) {
        const float factor = batch_index == 0u ? 1.0f : (batch_index == 1u ? 0.0f : 2.0f);
        for (size_t row = 0; row < hidden; ++row) {
            expect_close(q_out[batch_index * hidden + row], 0.0f, 1e-5f, "metal qkv batch q row");
        }
        for (size_t row = 0; row < kv; ++row) {
            expect_close(k_out[batch_index * kv + row], -16.0f * factor, 1e-5f, "metal qkv batch k row");
            expect_close(v_out[batch_index * kv + row], -8.0f * factor, 1e-5f, "metal qkv batch v row");
        }
    }
    free(mapping);
}

int main(void) {
    const char *path = "build/kernel-test.gguf";
    write_fixture(path);
    test_model_ops(path);
    test_metal_decode_fusion_guards();
    test_generated_metal_guard_tests();
    test_metal_qkv_pair_q4();
    test_metal_qkv_batch_q4();
    remove(path);
    puts("kernel-ops-ok");
    return 0;
}
