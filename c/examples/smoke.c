#include "tinyengine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct smoke_output {
    char text[128];
    size_t len;
} smoke_output;

static void on_token(const char *text, uint32_t token_id, void *userdata) {
    (void)token_id;
    smoke_output *output = (smoke_output *)userdata;
    const size_t chunk_len = strlen(text);
    if (output->len + chunk_len >= sizeof(output->text)) {
        return;
    }
    memcpy(output->text + output->len, text, chunk_len + 1u);
    output->len += chunk_len;
}

static int all_finite(const float *values, size_t len) {
    for (size_t index = 0; index < len; ++index) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s /path/to/model.gguf\n", argv[0]);
        return 2;
    }

    te_arch_info arch;
    te_status status = te_detect_arch(&arch);
    if (status != TE_STATUS_OK) {
        fprintf(stderr, "te_detect_arch failed: %s\n", te_strerror(status));
        return 1;
    }

    te_runtime_options options = te_default_options();
    options.target_arch = arch.kind;
    options.context_tokens = arch.recommended_max_context;

    te_kernel_plan plan;
    status = te_make_kernel_plan(&options, &plan);
    if (status != TE_STATUS_OK) {
        fprintf(stderr, "te_make_kernel_plan failed: %s\n", te_strerror(status));
        return 1;
    }
    if (!te_kernel_plan_supports_quant(&plan, TE_QUANT_GGUF_Q4_0) ||
        !te_kernel_plan_optimizes_quant(&plan, TE_QUANT_GGUF_Q8_0) ||
        !te_kernel_plan_supports_op(&plan, TE_OP_BATCHED_PREFILL)) {
        fprintf(stderr, "kernel plan is missing required Qwen optimization capabilities\n");
        return 1;
    }

    te_capabilities capabilities;
    status = te_get_capabilities(&capabilities);
    if (status != TE_STATUS_OK) {
        fprintf(stderr, "te_get_capabilities failed: %s\n", te_strerror(status));
        return 1;
    }

    te_model *model = NULL;
    status = te_model_load_gguf(argv[1], &options, &model);
    if (status != TE_STATUS_OK) {
        fprintf(stderr, "te_model_load_gguf failed: %s\n", te_strerror(status));
        return 1;
    }

    te_model_info info;
    status = te_model_get_info(model, &info);
    if (status != TE_STATUS_OK) {
        fprintf(stderr, "te_model_get_info failed: %s\n", te_strerror(status));
        te_model_free(model);
        return 1;
    }
    if (strcmp(info.architecture, "qwen2") != 0 ||
        info.embedding_length == 0 ||
        info.block_count == 0 ||
        info.vocab_size == 0 ||
        info.quant_tensor_counts[TE_QUANT_GGUF_Q4_0] == 0) {
        fprintf(stderr, "parsed GGUF metadata does not look like the expected Qwen2 Q4 model\n");
        te_model_free(model);
        return 1;
    }

    te_tensor_info token_embd;
    status = te_model_get_tensor_info(model, "token_embd.weight", &token_embd);
    if (status != TE_STATUS_OK ||
        token_embd.quant != TE_QUANT_GGUF_Q4_0 ||
        token_embd.n_dims != 2 ||
        token_embd.dims[0] != info.embedding_length ||
        token_embd.dims[1] != info.vocab_size) {
        fprintf(stderr, "token_embd.weight tensor descriptor is invalid\n");
        te_model_free(model);
        return 1;
    }

    te_tokenizer_info tokenizer;
    status = te_model_get_tokenizer_info(model, &tokenizer);
    if (status != TE_STATUS_OK ||
        strcmp(tokenizer.model, "gpt2") != 0 ||
        strcmp(tokenizer.pre, "qwen2") != 0 ||
        tokenizer.token_count != info.vocab_size ||
        tokenizer.merge_count == 0 ||
        tokenizer.eos_token_id != 151645u ||
        tokenizer.padding_token_id != 151643u ||
        tokenizer.add_bos_token != 0) {
        fprintf(stderr, "tokenizer metadata is invalid\n");
        te_model_free(model);
        return 1;
    }

    te_tensor_info output_weight;
    status = te_model_get_tensor_info(model, "output.weight", &output_weight);
    if (status != TE_STATUS_OK ||
        output_weight.quant != TE_QUANT_GGUF_Q8_0 ||
        output_weight.n_dims != 2 ||
        output_weight.dims[0] != info.embedding_length ||
        output_weight.dims[1] != info.vocab_size) {
        fprintf(stderr, "output.weight tensor descriptor is invalid\n");
        te_model_free(model);
        return 1;
    }

    te_tensor_info output_norm;
    status = te_model_get_tensor_info(model, "output_norm.weight", &output_norm);
    if (status != TE_STATUS_OK ||
        output_norm.quant != TE_QUANT_F32 ||
        output_norm.n_dims != 1 ||
        output_norm.dims[0] != info.embedding_length) {
        fprintf(stderr, "output_norm.weight tensor descriptor is invalid\n");
        te_model_free(model);
        return 1;
    }

    te_tensor_info q_proj;
    status = te_model_get_tensor_info(model, "blk.0.attn_q.weight", &q_proj);
    if (status != TE_STATUS_OK ||
        q_proj.quant != TE_QUANT_GGUF_Q4_0 ||
        q_proj.n_dims != 2 ||
        q_proj.dims[0] != info.embedding_length) {
        fprintf(stderr, "blk.0.attn_q.weight tensor descriptor is invalid\n");
        te_model_free(model);
        return 1;
    }

    const size_t hidden = info.embedding_length;
    const size_t vocab = info.vocab_size;
    size_t written = 0;
    float *embedding = (float *)calloc(hidden, sizeof(float));
    float *norm_weight = (float *)calloc(hidden, sizeof(float));
    float *normed = (float *)calloc(hidden, sizeof(float));
    float *logits = (float *)calloc(vocab, sizeof(float));
    float *q_out = (float *)calloc((size_t)q_proj.dims[1], sizeof(float));
    if (embedding == NULL || norm_weight == NULL || normed == NULL || logits == NULL || q_out == NULL) {
        fprintf(stderr, "failed to allocate C reference buffers\n");
        free(embedding);
        free(norm_weight);
        free(normed);
        free(logits);
        free(q_out);
        te_model_free(model);
        return 1;
    }

    status = te_model_dequantize_row_f32(model, "token_embd.weight", 0, embedding, hidden, &written);
    if (status != TE_STATUS_OK || written != hidden || !all_finite(embedding, hidden)) {
        fprintf(stderr, "failed to dequantize token_embd.weight row\n");
        free(embedding);
        free(norm_weight);
        free(normed);
        free(logits);
        free(q_out);
        te_model_free(model);
        return 1;
    }

    status = te_model_read_f32_tensor(model, "output_norm.weight", norm_weight, hidden, &written);
    if (status != TE_STATUS_OK || written != hidden || !all_finite(norm_weight, hidden)) {
        fprintf(stderr, "failed to read output_norm.weight\n");
        free(embedding);
        free(norm_weight);
        free(normed);
        free(logits);
        free(q_out);
        te_model_free(model);
        return 1;
    }

    status = te_rmsnorm_f32(embedding, norm_weight, hidden, info.rms_norm_epsilon, normed);
    if (status != TE_STATUS_OK || !all_finite(normed, hidden)) {
        fprintf(stderr, "failed to run rmsnorm reference op\n");
        free(embedding);
        free(norm_weight);
        free(normed);
        free(logits);
        free(q_out);
        te_model_free(model);
        return 1;
    }

    status = te_model_matvec_f32(model, "output.weight", normed, hidden, logits, vocab, &written);
    if (status != TE_STATUS_OK || written != vocab || !all_finite(logits, vocab)) {
        fprintf(stderr, "failed to run output.weight Q8_0 matvec reference op\n");
        free(embedding);
        free(norm_weight);
        free(normed);
        free(logits);
        free(q_out);
        te_model_free(model);
        return 1;
    }

    uint32_t reference_argmax = 0;
    status = te_argmax_f32(logits, vocab, &reference_argmax);
    if (status != TE_STATUS_OK || reference_argmax >= vocab) {
        fprintf(stderr, "failed to run argmax reference op\n");
        free(embedding);
        free(norm_weight);
        free(normed);
        free(logits);
        free(q_out);
        te_model_free(model);
        return 1;
    }

    status = te_model_matvec_f32(
        model,
        "blk.0.attn_q.weight",
        embedding,
        hidden,
        q_out,
        (size_t)q_proj.dims[1],
        &written);
    if (status != TE_STATUS_OK || written != (size_t)q_proj.dims[1] || !all_finite(q_out, written)) {
        fprintf(stderr, "failed to run blk.0.attn_q.weight Q4_0 matvec reference op\n");
        free(embedding);
        free(norm_weight);
        free(normed);
        free(logits);
        free(q_out);
        te_model_free(model);
        return 1;
    }

    te_context *context = NULL;
    status = te_context_create(model, &options, &context);
    if (status != TE_STATUS_OK) {
        fprintf(stderr, "te_context_create failed: %s\n", te_strerror(status));
        free(embedding);
        free(norm_weight);
        free(normed);
        free(logits);
        free(q_out);
        te_model_free(model);
        return 1;
    }

    smoke_output output = {0};
    status = te_generate(context, "Rispondi in italiano con tre parole: cosa sei?", 4, on_token, &output);
    if (status != TE_STATUS_OK) {
        fprintf(stderr, "te_generate failed: %s\n", te_strerror(status));
        te_context_free(context);
        free(embedding);
        free(norm_weight);
        free(normed);
        free(logits);
        free(q_out);
        te_model_free(model);
        return 1;
    }

    printf("arch=%s suffix=%s q4_prefill_batch_tile=%u q4_decode_row_tile=%u q8_lm_head_row_tile=%u\n",
           arch.name,
           plan.metal_function_suffix,
           plan.q4_prefill_batch_tile,
           plan.q4_decode_row_tile,
           plan.q8_lm_head_row_tile);
    printf("backend=%s q4=%s q8=%s op=%s flags=%u\n",
           capabilities.backend_name,
           te_quant_name(TE_QUANT_GGUF_Q4_0),
           te_quant_name(TE_QUANT_GGUF_Q8_0),
           te_vector_op_name(TE_OP_BATCHED_PREFILL),
           capabilities.optimization_flags);
    printf("model=%s arch=%s gguf=v%u tensors=%llu kv=%llu ctx=%u layers=%u hidden=%u ffn=%u heads=%u kv_heads=%u vocab=%u q4_tensors=%llu q8_tensors=%llu\n",
           info.name,
           info.architecture,
           info.gguf_version,
           (unsigned long long)info.tensor_count,
           (unsigned long long)info.metadata_kv_count,
           info.context_length,
           info.block_count,
           info.embedding_length,
           info.feed_forward_length,
           info.attention_head_count,
           info.attention_head_count_kv,
           info.vocab_size,
           (unsigned long long)info.quant_tensor_counts[TE_QUANT_GGUF_Q4_0],
           (unsigned long long)info.quant_tensor_counts[TE_QUANT_GGUF_Q8_0]);
    printf("tokenizer=%s pre=%s tokens=%llu merges=%llu bos=%u eos=%u pad=%u add_bos=%d\n",
           tokenizer.model,
           tokenizer.pre,
           (unsigned long long)tokenizer.token_count,
           (unsigned long long)tokenizer.merge_count,
           tokenizer.bos_token_id,
           tokenizer.eos_token_id,
           tokenizer.padding_token_id,
           tokenizer.add_bos_token);

    const char *oracle_prompt = "Rispondi in italiano con tre parole: cosa sei?";
    char chat_prompt[256];
    size_t chat_len = 0;
    status = te_format_qwen_chat_prompt(oracle_prompt, chat_prompt, sizeof(chat_prompt), &chat_len);
    if (status != TE_STATUS_OK) {
        fprintf(stderr, "chat prompt formatting failed\n");
        te_model_free(model);
        return 1;
    }
    const uint32_t expected_prompt_ids[] = {
        151644u, 872u, 198u, 49u, 285u, 3511u, 72u, 304u, 59804u, 390u, 4258u,
        48261u, 25u, 47513u, 42137u, 30u, 151645u, 198u, 151644u, 77091u, 198u
    };
    uint32_t prompt_ids[64];
    size_t prompt_count = 0;
    status = te_model_tokenize(model, chat_prompt, 1, prompt_ids, 64u, &prompt_count);
    if (status != TE_STATUS_OK || prompt_count != sizeof(expected_prompt_ids) / sizeof(expected_prompt_ids[0])) {
        fprintf(stderr, "tokenize prompt failed: status=%s count=%zu\n", te_strerror(status), prompt_count);
        te_model_free(model);
        return 1;
    }
    for (size_t index = 0; index < prompt_count; ++index) {
        if (prompt_ids[index] != expected_prompt_ids[index]) {
            fprintf(stderr, "prompt token mismatch at %zu: got %u expected %u\n",
                    index, prompt_ids[index], expected_prompt_ids[index]);
            te_model_free(model);
            return 1;
        }
    }
    const uint32_t generated_ids[] = {41887u, 521u, 34214u, 8515u};
    char generated_text[64];
    size_t generated_len = 0;
    status = te_model_detokenize(
        model,
        generated_ids,
        sizeof(generated_ids) / sizeof(generated_ids[0]),
        1,
        generated_text,
        sizeof(generated_text),
        &generated_len);
    if (status != TE_STATUS_OK || strcmp(generated_text, "Mi chiamo Alex") != 0) {
        fprintf(stderr, "detokenize generated ids failed: %s text=%s\n", te_strerror(status), generated_text);
        te_model_free(model);
        return 1;
    }
    printf("tensor token_embd=%s dims=%llux%llu bytes=%llu output=%s dims=%llux%llu bytes=%llu\n",
           te_quant_name(token_embd.quant),
           (unsigned long long)token_embd.dims[0],
           (unsigned long long)token_embd.dims[1],
           (unsigned long long)token_embd.bytes,
           te_quant_name(output_weight.quant),
           (unsigned long long)output_weight.dims[0],
           (unsigned long long)output_weight.dims[1],
           (unsigned long long)output_weight.bytes);
    printf("reference_ops embedding=%zu logits=%zu argmax=%u q_proj=%zu\n",
           hidden,
           vocab,
           reference_argmax,
           (size_t)q_proj.dims[1]);
    printf("generated=%s\n", output.text);

    te_context_free(context);
    free(embedding);
    free(norm_weight);
    free(normed);
    free(logits);
    free(q_out);
    te_model_free(model);
    return strcmp(output.text, "Mi chiamo Alex") == 0 ? 0 : 1;
}
