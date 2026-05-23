#include "tinyengine_internal.h"
#include "metal_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef struct te_qwen_state {
    size_t hidden;
    size_t ffn;
    size_t layers;
    size_t heads;
    size_t kv_heads;
    size_t head_dim;
    size_t kv_dim;
    size_t vocab;
    size_t context_tokens;
    float *hidden_buf;
    float *norm;
    float *q;
    float *k;
    float *v;
    float *attn;
    float *proj;
    float *gate;
    float *up;
    float *swiglu;
    float *logits;
    float *attn_norm_weights;
    float *ffn_norm_weights;
    float *q_biases;
    float *k_biases;
    float *v_biases;
    float *output_norm_weight;
    float *rope_cos;
    float *rope_sin;
    float *key_cache;
    float *value_cache;
} te_qwen_state;

typedef struct te_qwen_profile {
    int enabled;
    double total_ms;
    double tokenize_ms;
    double init_ms;
    double embed_ms;
    double q_ms;
    double k_ms;
    double v_ms;
    double o_ms;
    double gate_ms;
    double up_ms;
    double down_ms;
    double lm_head_ms;
    double other_matvec_ms;
    uint64_t q_calls;
    uint64_t k_calls;
    uint64_t v_calls;
    uint64_t o_calls;
    uint64_t gate_calls;
    uint64_t up_calls;
    uint64_t down_calls;
    uint64_t lm_head_calls;
    uint64_t other_matvec_calls;
} te_qwen_profile;

static te_qwen_profile TE_QWEN_PROFILE;

static te_status te_qwen_state_init(te_qwen_state *state, const te_context *context, size_t required_context);
static void te_qwen_state_release(te_qwen_state *state);
static te_status te_qwen_prefill_prompt_batch(
    te_context *context,
    te_qwen_state *state,
    const uint32_t *prompt_tokens,
    size_t prompt_count
);
static te_status te_qwen_forward_token(te_context *context, te_qwen_state *state, uint32_t token_id, size_t position);
static te_status te_qwen_forward_project_token(
    te_context *context,
    te_qwen_state *state,
    uint32_t token_id,
    size_t position,
    uint32_t *out_token_id
);
static te_status te_qwen_project_next(te_context *context, te_qwen_state *state, uint32_t *out_token_id);
static te_status te_qwen_read_f32_exact(te_model *model, const char *name, float *out, size_t len);
static te_status te_qwen_rope_cached(
    float *values,
    size_t heads,
    size_t head_dim,
    const float *cos_table,
    const float *sin_table,
    size_t position
);
static te_status te_qwen_matvec_exact(
    te_model *model,
    const char *name,
    const float *input,
    size_t input_len,
    float *out,
    size_t out_len
);
static te_status te_qwen_matvec2_exact(
    te_model *model,
    const char *name_a,
    const char *name_b,
    const float *input,
    size_t input_len,
    float *out_a,
    float *out_b,
    size_t out_len
);
static te_status te_qwen_matvec_batch_exact(
    te_model *model,
    const char *name,
    const float *input,
    size_t batch,
    size_t input_len,
    float *out,
    size_t out_len
);
static te_status te_qwen_qkv_exact(
    te_model *model,
    const char *q_name,
    const char *k_name,
    const char *v_name,
    const float *input,
    size_t hidden,
    size_t kv,
    float *q,
    float *k,
    float *v
);
static te_status te_qwen_qkv_batch_exact(
    te_model *model,
    const char *q_name,
    const char *k_name,
    const char *v_name,
    const float *input,
    size_t batch,
    size_t hidden,
    size_t kv,
    float *q,
    float *k,
    float *v
);
static te_status te_qwen_mlp_exact(
    te_model *model,
    const char *gate_name,
    const char *up_name,
    const char *down_name,
    const float *input,
    size_t hidden,
    size_t ffn,
    float *gate,
    float *up,
    float *swiglu,
    float *out
);
static te_status te_qwen_post_attn_mlp_exact(
    te_model *model,
    const char *output_name,
    const char *gate_name,
    const char *up_name,
    const char *down_name,
    const float *hidden_in,
    const float *attn,
    const float *ffn_norm_weight,
    size_t hidden,
    size_t ffn,
    float epsilon,
    float *out
);
static te_status te_qwen_mlp_batch_exact(
    te_model *model,
    const char *gate_name,
    const char *up_name,
    const char *down_name,
    const float *input,
    size_t batch,
    size_t hidden,
    size_t ffn,
    float *out
);
static te_status te_qwen_post_attn_mlp_batch_exact(
    te_model *model,
    const char *output_name,
    const char *gate_name,
    const char *up_name,
    const char *down_name,
    const float *hidden_in,
    const float *attn,
    const float *ffn_norm_weight,
    size_t batch,
    size_t hidden,
    size_t ffn,
    float epsilon,
    float *out
);
static te_status te_qwen_decode_layer_exact(
    te_model *model,
    const char *q_name,
    const char *k_name,
    const char *v_name,
    const char *output_name,
    const char *gate_name,
    const char *up_name,
    const char *down_name,
    te_qwen_state *state,
    size_t layer,
    size_t position
);
static te_status te_qwen_decode_all_layers_exact(
    te_model *model,
    te_qwen_state *state,
    size_t position,
    const te_gguf_tensor *head_tensor,
    uint32_t *out_token_id
);
static te_status te_qwen_prefill_layer_exact(
    te_model *model,
    const char *q_name,
    const char *k_name,
    const char *v_name,
    const char *output_name,
    const char *gate_name,
    const char *up_name,
    const char *down_name,
    te_qwen_state *state,
    size_t layer,
    size_t batch,
    float *hidden
);
static te_status te_qwen_prefill_all_layers_exact(
    te_model *model,
    te_qwen_state *state,
    size_t batch,
    float *hidden
);
static int te_qwen_name(char *out, size_t out_len, const char *format, size_t layer);
static int te_qwen_checked_mul(size_t lhs, size_t rhs, size_t *out);
static int te_qwen_stop_token(const te_model *model, uint32_t token_id);
static te_status te_qwen_emit_token(
    te_model *model,
    uint32_t token_id,
    te_token_callback callback,
    void *userdata
);
static int te_qwen_profile_enabled(void);
static double te_qwen_now_ms(void);
static void te_qwen_profile_reset(void);
static void te_qwen_profile_add_matvec(const char *name, double elapsed_ms);
static void te_qwen_profile_print(void);

te_status te_qwen_generate_reference(
    te_context *context,
    const char *prompt,
    uint32_t max_tokens,
    te_token_callback callback,
    void *userdata
) {
    if (context == NULL || context->model == NULL || prompt == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    const double total_start = te_qwen_now_ms();
    te_qwen_profile_reset();

    te_model *model = context->model;
    size_t chat_len = 0;
    const double tokenize_start = te_qwen_now_ms();
    te_status status = te_format_qwen_chat_prompt(prompt, NULL, 0u, &chat_len);
    if (status != TE_STATUS_INVALID_ARGUMENT || chat_len == 0u) {
        return status == TE_STATUS_OK ? TE_STATUS_RUNTIME_ERROR : status;
    }
    char *chat = (char *)malloc(chat_len + 1u);
    if (chat == NULL) {
        return TE_STATUS_OUT_OF_MEMORY;
    }
    status = te_format_qwen_chat_prompt(prompt, chat, chat_len + 1u, &chat_len);
    if (status != TE_STATUS_OK) {
        free(chat);
        return status;
    }

    const size_t token_capacity = chat_len == 0u ? 1u : chat_len;
    if (token_capacity > SIZE_MAX / sizeof(uint32_t)) {
        free(chat);
        return TE_STATUS_UNSUPPORTED;
    }
    uint32_t *prompt_tokens = (uint32_t *)calloc(token_capacity, sizeof(prompt_tokens[0]));
    if (prompt_tokens == NULL) {
        free(chat);
        return TE_STATUS_OUT_OF_MEMORY;
    }
    size_t prompt_count = 0;
    status = te_model_tokenize(model, chat, 1, prompt_tokens, token_capacity, &prompt_count);
    free(chat);
    if (status != TE_STATUS_OK || prompt_count == 0u) {
        free(prompt_tokens);
        return status == TE_STATUS_OK ? TE_STATUS_RUNTIME_ERROR : status;
    }
    if (TE_QWEN_PROFILE.enabled) {
        TE_QWEN_PROFILE.tokenize_ms += te_qwen_now_ms() - tokenize_start;
    }

    size_t required_context = 0;
    if (!te_qwen_checked_mul((size_t)max_tokens, 1u, &required_context) ||
        prompt_count > SIZE_MAX - (size_t)max_tokens) {
        free(prompt_tokens);
        return TE_STATUS_UNSUPPORTED;
    }
    required_context = prompt_count + (size_t)max_tokens;

    te_qwen_state state;
    const double init_start = te_qwen_now_ms();
    status = te_qwen_state_init(&state, context, required_context);
    if (status != TE_STATUS_OK) {
        free(prompt_tokens);
        return status;
    }
    if (TE_QWEN_PROFILE.enabled) {
        TE_QWEN_PROFILE.init_ms += te_qwen_now_ms() - init_start;
    }

    uint32_t next_id = 0;
    const char *batch_prefill = getenv("TINYENGINE_BATCH_PREFILL");
    if (prompt_count > 1u && (batch_prefill == NULL || strcmp(batch_prefill, "0") != 0)) {
        status = te_qwen_prefill_prompt_batch(context, &state, prompt_tokens, prompt_count);
        if (status == TE_STATUS_UNSUPPORTED) {
            for (size_t index = 0; index < prompt_count; ++index) {
                status = te_qwen_forward_token(context, &state, prompt_tokens[index], index);
                if (status != TE_STATUS_OK) {
                    goto done;
                }
            }
        } else if (status != TE_STATUS_OK) {
            goto done;
        }
    } else {
        for (size_t index = 0; index < prompt_count; ++index) {
            status = te_qwen_forward_token(context, &state, prompt_tokens[index], index);
            if (status != TE_STATUS_OK) {
                goto done;
            }
        }
    }
    status = te_qwen_project_next(context, &state, &next_id);
    if (status != TE_STATUS_OK) {
        goto done;
    }

    size_t position = prompt_count;
    for (uint32_t generated = 0; generated < max_tokens; ++generated) {
        if (te_qwen_stop_token(model, next_id)) {
            break;
        }
        status = te_qwen_emit_token(model, next_id, callback, userdata);
        if (status != TE_STATUS_OK) {
            goto done;
        }
        if (generated + 1u == max_tokens) {
            break;
        }
        status = te_qwen_forward_project_token(context, &state, next_id, position, &next_id);
        if (status != TE_STATUS_OK) {
            goto done;
        }
        ++position;
    }

done:
    if (TE_QWEN_PROFILE.enabled) {
        TE_QWEN_PROFILE.total_ms = te_qwen_now_ms() - total_start;
        te_qwen_profile_print();
    }
    te_qwen_state_release(&state);
    free(prompt_tokens);
    return status;
}

static te_status te_qwen_state_init(te_qwen_state *state, const te_context *context, size_t required_context) {
    if (state == NULL || context == NULL || context->model == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    memset(state, 0, sizeof(*state));
    const te_model_info *info = &context->model->info;
    if (info->architecture[0] == '\0' || strcmp(info->architecture, "qwen2") != 0 ||
        info->embedding_length == 0u || info->feed_forward_length == 0u ||
        info->block_count == 0u || info->attention_head_count == 0u ||
        info->attention_head_count_kv == 0u || info->head_dim == 0u ||
        info->vocab_size == 0u) {
        return TE_STATUS_UNSUPPORTED;
    }

    state->hidden = info->embedding_length;
    state->ffn = info->feed_forward_length;
    state->layers = info->block_count;
    state->heads = info->attention_head_count;
    state->kv_heads = info->attention_head_count_kv;
    state->head_dim = info->head_dim;
    state->kv_dim = state->kv_heads * state->head_dim;
    state->vocab = info->vocab_size;
    state->context_tokens = context->options.context_tokens != 0u ? context->options.context_tokens : 512u;
    if (required_context > state->context_tokens ||
        state->heads * state->head_dim != state->hidden ||
        state->kv_dim == 0u) {
        return TE_STATUS_UNSUPPORTED;
    }

    size_t cache_tokens = 0;
    size_t cache_values = 0;
    size_t norm_values = 0;
    size_t kv_bias_values = 0;
    size_t rope_values = 0;
    if (!te_qwen_checked_mul(state->layers, state->context_tokens, &cache_tokens) ||
        !te_qwen_checked_mul(cache_tokens, state->kv_dim, &cache_values) ||
        !te_qwen_checked_mul(state->layers, state->hidden, &norm_values) ||
        !te_qwen_checked_mul(state->layers, state->kv_dim, &kv_bias_values) ||
        !te_qwen_checked_mul(state->context_tokens, state->head_dim / 2u, &rope_values)) {
        return TE_STATUS_UNSUPPORTED;
    }

    state->hidden_buf = (float *)calloc(state->hidden, sizeof(float));
    state->norm = (float *)calloc(state->hidden, sizeof(float));
    state->q = (float *)calloc(state->hidden, sizeof(float));
    state->k = (float *)calloc(state->kv_dim, sizeof(float));
    state->v = (float *)calloc(state->kv_dim, sizeof(float));
    state->attn = (float *)calloc(state->hidden, sizeof(float));
    state->proj = (float *)calloc(state->hidden, sizeof(float));
    state->gate = (float *)calloc(state->ffn, sizeof(float));
    state->up = (float *)calloc(state->ffn, sizeof(float));
    state->swiglu = (float *)calloc(state->ffn, sizeof(float));
    state->logits = (float *)calloc(state->vocab, sizeof(float));
    state->attn_norm_weights = (float *)calloc(norm_values, sizeof(float));
    state->ffn_norm_weights = (float *)calloc(norm_values, sizeof(float));
    state->q_biases = (float *)calloc(norm_values, sizeof(float));
    state->k_biases = (float *)calloc(kv_bias_values, sizeof(float));
    state->v_biases = (float *)calloc(kv_bias_values, sizeof(float));
    state->output_norm_weight = (float *)calloc(state->hidden, sizeof(float));
    state->rope_cos = (float *)calloc(rope_values, sizeof(float));
    state->rope_sin = (float *)calloc(rope_values, sizeof(float));
    state->key_cache = (float *)calloc(cache_values, sizeof(float));
    state->value_cache = (float *)calloc(cache_values, sizeof(float));

    if (state->hidden_buf == NULL || state->norm == NULL || state->q == NULL ||
        state->k == NULL || state->v == NULL || state->attn == NULL ||
        state->proj == NULL || state->gate == NULL || state->up == NULL ||
        state->swiglu == NULL || state->logits == NULL ||
        state->attn_norm_weights == NULL || state->ffn_norm_weights == NULL ||
        state->q_biases == NULL || state->k_biases == NULL || state->v_biases == NULL ||
        state->output_norm_weight == NULL || state->rope_cos == NULL || state->rope_sin == NULL ||
        state->key_cache == NULL || state->value_cache == NULL) {
        te_qwen_state_release(state);
        return TE_STATUS_OUT_OF_MEMORY;
    }
    const size_t rope_half = state->head_dim / 2u;
    for (size_t position = 0; position < state->context_tokens; ++position) {
        for (size_t index = 0; index < rope_half; ++index) {
            const float freq = powf(context->model->info.rope_freq_base, -(2.0f * (float)index) / (float)state->head_dim);
            const float angle = (float)position * freq;
            state->rope_cos[position * rope_half + index] = cosf(angle);
            state->rope_sin[position * rope_half + index] = sinf(angle);
        }
    }
    te_model *model = context->model;
    for (size_t layer = 0; layer < state->layers; ++layer) {
        char name[64];
        float *attn_norm = state->attn_norm_weights + layer * state->hidden;
        float *ffn_norm = state->ffn_norm_weights + layer * state->hidden;
        float *q_bias = state->q_biases + layer * state->hidden;
        float *k_bias = state->k_biases + layer * state->kv_dim;
        float *v_bias = state->v_biases + layer * state->kv_dim;
        if (!te_qwen_name(name, sizeof(name), "blk.%zu.attn_norm.weight", layer)) {
            te_qwen_state_release(state);
            return TE_STATUS_UNSUPPORTED;
        }
        te_status status = te_qwen_read_f32_exact(model, name, attn_norm, state->hidden);
        if (status != TE_STATUS_OK) {
            te_qwen_state_release(state);
            return status;
        }
        if (!te_qwen_name(name, sizeof(name), "blk.%zu.ffn_norm.weight", layer)) {
            te_qwen_state_release(state);
            return TE_STATUS_UNSUPPORTED;
        }
        status = te_qwen_read_f32_exact(model, name, ffn_norm, state->hidden);
        if (status != TE_STATUS_OK) {
            te_qwen_state_release(state);
            return status;
        }
        if (!te_qwen_name(name, sizeof(name), "blk.%zu.attn_q.bias", layer)) {
            te_qwen_state_release(state);
            return TE_STATUS_UNSUPPORTED;
        }
        status = te_qwen_read_f32_exact(model, name, q_bias, state->hidden);
        if (status != TE_STATUS_OK) {
            te_qwen_state_release(state);
            return status;
        }
        if (!te_qwen_name(name, sizeof(name), "blk.%zu.attn_k.bias", layer)) {
            te_qwen_state_release(state);
            return TE_STATUS_UNSUPPORTED;
        }
        status = te_qwen_read_f32_exact(model, name, k_bias, state->kv_dim);
        if (status != TE_STATUS_OK) {
            te_qwen_state_release(state);
            return status;
        }
        if (!te_qwen_name(name, sizeof(name), "blk.%zu.attn_v.bias", layer)) {
            te_qwen_state_release(state);
            return TE_STATUS_UNSUPPORTED;
        }
        status = te_qwen_read_f32_exact(model, name, v_bias, state->kv_dim);
        if (status != TE_STATUS_OK) {
            te_qwen_state_release(state);
            return status;
        }
    }
    te_status status = te_qwen_read_f32_exact(model, "output_norm.weight", state->output_norm_weight, state->hidden);
    if (status != TE_STATUS_OK) {
        te_qwen_state_release(state);
        return status;
    }
    return TE_STATUS_OK;
}

static void te_qwen_state_release(te_qwen_state *state) {
    if (state == NULL) {
        return;
    }
    free(state->hidden_buf);
    free(state->norm);
    free(state->q);
    free(state->k);
    free(state->v);
    free(state->attn);
    free(state->proj);
    free(state->gate);
    free(state->up);
    free(state->swiglu);
    free(state->logits);
    free(state->attn_norm_weights);
    free(state->ffn_norm_weights);
    free(state->q_biases);
    free(state->k_biases);
    free(state->v_biases);
    free(state->output_norm_weight);
    free(state->rope_cos);
    free(state->rope_sin);
    free(state->key_cache);
    free(state->value_cache);
    memset(state, 0, sizeof(*state));
}

static te_status te_qwen_prefill_prompt_batch(
    te_context *context,
    te_qwen_state *state,
    const uint32_t *prompt_tokens,
    size_t prompt_count
) {
    if (context == NULL || context->model == NULL || state == NULL || prompt_tokens == NULL || prompt_count == 0u) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (prompt_count > state->context_tokens) {
        return TE_STATUS_UNSUPPORTED;
    }

    te_model *model = context->model;
    size_t hidden_values = 0;
    size_t kv_values = 0;
    size_t ffn_values = 0;
    if (!te_qwen_checked_mul(prompt_count, state->hidden, &hidden_values) ||
        !te_qwen_checked_mul(prompt_count, state->kv_dim, &kv_values) ||
        !te_qwen_checked_mul(prompt_count, state->ffn, &ffn_values)) {
        return TE_STATUS_UNSUPPORTED;
    }

    float *hidden = (float *)calloc(hidden_values, sizeof(float));
    float *norm = (float *)calloc(hidden_values, sizeof(float));
    float *q = (float *)calloc(hidden_values, sizeof(float));
    float *k = (float *)calloc(kv_values, sizeof(float));
    float *v = (float *)calloc(kv_values, sizeof(float));
    float *attn = (float *)calloc(hidden_values, sizeof(float));
    float *proj = (float *)calloc(hidden_values, sizeof(float));
    float *mlp = (float *)calloc(hidden_values, sizeof(float));
    if (hidden == NULL || norm == NULL || q == NULL || k == NULL || v == NULL ||
        attn == NULL || proj == NULL || mlp == NULL) {
        free(hidden);
        free(norm);
        free(q);
        free(k);
        free(v);
        free(attn);
        free(proj);
        free(mlp);
        return TE_STATUS_OUT_OF_MEMORY;
    }

    te_status status = TE_STATUS_OK;
    const double embed_start = te_qwen_now_ms();
    for (size_t token_index = 0; token_index < prompt_count; ++token_index) {
        size_t written = 0;
        status = te_model_dequantize_row_f32(
            model,
            "token_embd.weight",
            prompt_tokens[token_index],
            hidden + token_index * state->hidden,
            state->hidden,
            &written);
        if (status != TE_STATUS_OK || written != state->hidden) {
            status = status == TE_STATUS_OK ? TE_STATUS_RUNTIME_ERROR : status;
            goto cleanup;
        }
    }
    if (TE_QWEN_PROFILE.enabled) {
        TE_QWEN_PROFILE.embed_ms += te_qwen_now_ms() - embed_start;
    }

    status = te_qwen_prefill_all_layers_exact(model, state, prompt_count, hidden);
    if (status == TE_STATUS_OK) {
        goto copy_last_hidden;
    }
    if (status != TE_STATUS_UNSUPPORTED) {
        goto cleanup;
    }

    for (size_t layer = 0; layer < state->layers; ++layer) {
        char q_name[64];
        char k_name[64];
        char v_name[64];
        char out_name[64];
        char gate_name[64];
        char up_name[64];
        char down_name[64];
        if (!te_qwen_name(q_name, sizeof(q_name), "blk.%zu.attn_q.weight", layer) ||
            !te_qwen_name(k_name, sizeof(k_name), "blk.%zu.attn_k.weight", layer) ||
            !te_qwen_name(v_name, sizeof(v_name), "blk.%zu.attn_v.weight", layer) ||
            !te_qwen_name(out_name, sizeof(out_name), "blk.%zu.attn_output.weight", layer) ||
            !te_qwen_name(gate_name, sizeof(gate_name), "blk.%zu.ffn_gate.weight", layer) ||
            !te_qwen_name(up_name, sizeof(up_name), "blk.%zu.ffn_up.weight", layer) ||
            !te_qwen_name(down_name, sizeof(down_name), "blk.%zu.ffn_down.weight", layer)) {
            status = TE_STATUS_UNSUPPORTED;
            goto cleanup;
        }

        status = te_qwen_prefill_layer_exact(
            model,
            q_name,
            k_name,
            v_name,
            out_name,
            gate_name,
            up_name,
            down_name,
            state,
            layer,
            prompt_count,
            hidden);
        if (status == TE_STATUS_OK) {
            continue;
        }
        if (status != TE_STATUS_UNSUPPORTED) {
            goto cleanup;
        }

        const float *attn_norm_weight = state->attn_norm_weights + layer * state->hidden;
        const float *ffn_norm_weight = state->ffn_norm_weights + layer * state->hidden;
        for (size_t token_index = 0; token_index < prompt_count; ++token_index) {
            status = te_rmsnorm_f32(
                hidden + token_index * state->hidden,
                attn_norm_weight,
                state->hidden,
                model->info.rms_norm_epsilon,
                norm + token_index * state->hidden);
            if (status != TE_STATUS_OK) {
                goto cleanup;
            }
        }

        status = te_qwen_qkv_batch_exact(
            model,
            q_name,
            k_name,
            v_name,
            norm,
            prompt_count,
            state->hidden,
            state->kv_dim,
            q,
            k,
            v);
        if (status != TE_STATUS_OK) {
            goto cleanup;
        }

        const float *q_bias = state->q_biases + layer * state->hidden;
        const float *k_bias = state->k_biases + layer * state->kv_dim;
        const float *v_bias = state->v_biases + layer * state->kv_dim;
        for (size_t token_index = 0; token_index < prompt_count; ++token_index) {
            float *token_q = q + token_index * state->hidden;
            float *token_k = k + token_index * state->kv_dim;
            float *token_v = v + token_index * state->kv_dim;
            for (size_t index = 0; index < state->hidden; ++index) {
                token_q[index] += q_bias[index];
            }
            for (size_t index = 0; index < state->kv_dim; ++index) {
                token_k[index] += k_bias[index];
                token_v[index] += v_bias[index];
            }
            status = te_qwen_rope_cached(
                token_q,
                state->heads,
                state->head_dim,
                state->rope_cos,
                state->rope_sin,
                token_index);
            if (status != TE_STATUS_OK) {
                goto cleanup;
            }
            status = te_qwen_rope_cached(
                token_k,
                state->kv_heads,
                state->head_dim,
                state->rope_cos,
                state->rope_sin,
                token_index);
            if (status != TE_STATUS_OK) {
                goto cleanup;
            }
            const size_t cache_offset = (layer * state->context_tokens + token_index) * state->kv_dim;
            memcpy(state->key_cache + cache_offset, token_k, state->kv_dim * sizeof(float));
            memcpy(state->value_cache + cache_offset, token_v, state->kv_dim * sizeof(float));
        }

        const size_t layer_cache_offset = layer * state->context_tokens * state->kv_dim;
        for (size_t token_index = 0; token_index < prompt_count; ++token_index) {
            status = te_attention_decode_f32(
                q + token_index * state->hidden,
                state->key_cache + layer_cache_offset,
                state->value_cache + layer_cache_offset,
                token_index,
                state->heads,
                state->kv_heads,
                state->head_dim,
                attn + token_index * state->hidden);
            if (status != TE_STATUS_OK) {
                goto cleanup;
            }
        }

        status = te_qwen_post_attn_mlp_batch_exact(
            model,
            out_name,
            gate_name,
            up_name,
            down_name,
            hidden,
            attn,
            ffn_norm_weight,
            prompt_count,
            state->hidden,
            state->ffn,
            model->info.rms_norm_epsilon,
            hidden);
        if (status == TE_STATUS_UNSUPPORTED) {
            status = te_qwen_matvec_batch_exact(model, out_name, attn, prompt_count, state->hidden, proj, state->hidden);
            if (status != TE_STATUS_OK) {
                goto cleanup;
            }
            for (size_t token_index = 0; token_index < prompt_count; ++token_index) {
                status = te_add_f32(
                    hidden + token_index * state->hidden,
                    proj + token_index * state->hidden,
                    state->hidden,
                    hidden + token_index * state->hidden);
                if (status != TE_STATUS_OK) {
                    goto cleanup;
                }
                status = te_rmsnorm_f32(
                    hidden + token_index * state->hidden,
                    ffn_norm_weight,
                    state->hidden,
                    model->info.rms_norm_epsilon,
                    norm + token_index * state->hidden);
                if (status != TE_STATUS_OK) {
                    goto cleanup;
                }
            }
            status = te_qwen_mlp_batch_exact(
                model,
                gate_name,
                up_name,
                down_name,
                norm,
                prompt_count,
                state->hidden,
                state->ffn,
                mlp);
            if (status != TE_STATUS_OK) {
                goto cleanup;
            }
            for (size_t token_index = 0; token_index < prompt_count; ++token_index) {
                status = te_add_f32(
                    hidden + token_index * state->hidden,
                    mlp + token_index * state->hidden,
                    state->hidden,
                    hidden + token_index * state->hidden);
                if (status != TE_STATUS_OK) {
                    goto cleanup;
                }
            }
        } else if (status != TE_STATUS_OK) {
            goto cleanup;
        }
    }

copy_last_hidden:
    memcpy(state->hidden_buf, hidden + (prompt_count - 1u) * state->hidden, state->hidden * sizeof(float));

cleanup:
    free(hidden);
    free(norm);
    free(q);
    free(k);
    free(v);
    free(attn);
    free(proj);
    free(mlp);
    return status;
}

static te_status te_qwen_forward_token(te_context *context, te_qwen_state *state, uint32_t token_id, size_t position) {
    te_model *model = context->model;
    if (position >= state->context_tokens) {
        return TE_STATUS_UNSUPPORTED;
    }
    size_t written = 0;
    const double embed_start = te_qwen_now_ms();
    te_status status = te_model_dequantize_row_f32(
        model,
        "token_embd.weight",
        token_id,
        state->hidden_buf,
        state->hidden,
        &written);
    if (status != TE_STATUS_OK || written != state->hidden) {
        return status == TE_STATUS_OK ? TE_STATUS_RUNTIME_ERROR : status;
    }
    if (TE_QWEN_PROFILE.enabled) {
        TE_QWEN_PROFILE.embed_ms += te_qwen_now_ms() - embed_start;
    }

    status = te_qwen_decode_all_layers_exact(model, state, position, NULL, NULL);
    if (status == TE_STATUS_OK) {
        return TE_STATUS_OK;
    }
    if (status != TE_STATUS_UNSUPPORTED) {
        return status;
    }

    for (size_t layer = 0; layer < state->layers; ++layer) {
        char q_name[64];
        char k_name[64];
        char v_name[64];
        char out_name[64];
        char gate_name[64];
        char up_name[64];
        char down_name[64];
        if (!te_qwen_name(q_name, sizeof(q_name), "blk.%zu.attn_q.weight", layer) ||
            !te_qwen_name(k_name, sizeof(k_name), "blk.%zu.attn_k.weight", layer) ||
            !te_qwen_name(v_name, sizeof(v_name), "blk.%zu.attn_v.weight", layer) ||
            !te_qwen_name(out_name, sizeof(out_name), "blk.%zu.attn_output.weight", layer) ||
            !te_qwen_name(gate_name, sizeof(gate_name), "blk.%zu.ffn_gate.weight", layer) ||
            !te_qwen_name(up_name, sizeof(up_name), "blk.%zu.ffn_up.weight", layer) ||
            !te_qwen_name(down_name, sizeof(down_name), "blk.%zu.ffn_down.weight", layer)) {
            return TE_STATUS_UNSUPPORTED;
        }

        status = te_qwen_decode_layer_exact(
            model,
            q_name,
            k_name,
            v_name,
            out_name,
            gate_name,
            up_name,
            down_name,
            state,
            layer,
            position);
        if (status == TE_STATUS_OK) {
            continue;
        }
        if (status != TE_STATUS_UNSUPPORTED) {
            return status;
        }

        const float *attn_norm_weight = state->attn_norm_weights + layer * state->hidden;
        status = te_rmsnorm_f32(
            state->hidden_buf,
            attn_norm_weight,
            state->hidden,
            model->info.rms_norm_epsilon,
            state->norm);
        if (status != TE_STATUS_OK) {
            return status;
        }

        status = te_qwen_qkv_exact(
            model,
            q_name,
            k_name,
            v_name,
            state->norm,
            state->hidden,
            state->kv_dim,
            state->q,
            state->k,
            state->v);
        if (status != TE_STATUS_OK) {
            return status;
        }
        status = te_add_f32(state->q, state->q_biases + layer * state->hidden, state->hidden, state->q);
        if (status != TE_STATUS_OK) {
            return status;
        }

        status = te_add_f32(state->k, state->k_biases + layer * state->kv_dim, state->kv_dim, state->k);
        if (status != TE_STATUS_OK) {
            return status;
        }

        status = te_add_f32(state->v, state->v_biases + layer * state->kv_dim, state->kv_dim, state->v);
        if (status != TE_STATUS_OK) {
            return status;
        }

        status = te_qwen_rope_cached(
            state->q,
            state->heads,
            state->head_dim,
            state->rope_cos,
            state->rope_sin,
            position);
        if (status != TE_STATUS_OK) {
            return status;
        }
        status = te_qwen_rope_cached(
            state->k,
            state->kv_heads,
            state->head_dim,
            state->rope_cos,
            state->rope_sin,
            position);
        if (status != TE_STATUS_OK) {
            return status;
        }

        const size_t cache_offset = (layer * state->context_tokens + position) * state->kv_dim;
        memcpy(state->key_cache + cache_offset, state->k, state->kv_dim * sizeof(float));
        memcpy(state->value_cache + cache_offset, state->v, state->kv_dim * sizeof(float));
        const size_t layer_cache_offset = layer * state->context_tokens * state->kv_dim;
        status = te_attention_decode_f32(
            state->q,
            state->key_cache + layer_cache_offset,
            state->value_cache + layer_cache_offset,
            position,
            state->heads,
            state->kv_heads,
            state->head_dim,
            state->attn);
        if (status != TE_STATUS_OK) {
            return status;
        }

        const float *ffn_norm_weight = state->ffn_norm_weights + layer * state->hidden;
        status = te_qwen_post_attn_mlp_exact(
            model,
            out_name,
            gate_name,
            up_name,
            down_name,
            state->hidden_buf,
            state->attn,
            ffn_norm_weight,
            state->hidden,
            state->ffn,
            model->info.rms_norm_epsilon,
            state->hidden_buf);
        if (status == TE_STATUS_OK) {
            continue;
        }
        if (status != TE_STATUS_UNSUPPORTED) {
            return status;
        }

        status = te_qwen_matvec_exact(model, out_name, state->attn, state->hidden, state->proj, state->hidden);
        if (status != TE_STATUS_OK) {
            return status;
        }
        status = te_add_f32(state->hidden_buf, state->proj, state->hidden, state->hidden_buf);
        if (status != TE_STATUS_OK) {
            return status;
        }

        status = te_rmsnorm_f32(
            state->hidden_buf,
            ffn_norm_weight,
            state->hidden,
            model->info.rms_norm_epsilon,
            state->norm);
        if (status != TE_STATUS_OK) {
            return status;
        }

        status = te_qwen_mlp_exact(
            model,
            gate_name,
            up_name,
            down_name,
            state->norm,
            state->hidden,
            state->ffn,
            state->gate,
            state->up,
            state->swiglu,
            state->proj);
        if (status != TE_STATUS_OK) {
            return status;
        }
        status = te_add_f32(state->hidden_buf, state->proj, state->hidden, state->hidden_buf);
        if (status != TE_STATUS_OK) {
            return status;
        }
    }
    return TE_STATUS_OK;
}

static te_status te_qwen_forward_project_token(
    te_context *context,
    te_qwen_state *state,
    uint32_t token_id,
    size_t position,
    uint32_t *out_token_id
) {
    if (context == NULL || context->model == NULL || state == NULL || out_token_id == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (position >= state->context_tokens) {
        return TE_STATUS_UNSUPPORTED;
    }

    te_model *model = context->model;
    size_t written = 0;
    const double embed_start = te_qwen_now_ms();
    te_status status = te_model_dequantize_row_f32(
        model,
        "token_embd.weight",
        token_id,
        state->hidden_buf,
        state->hidden,
        &written);
    if (status != TE_STATUS_OK || written != state->hidden) {
        return status == TE_STATUS_OK ? TE_STATUS_RUNTIME_ERROR : status;
    }
    if (TE_QWEN_PROFILE.enabled) {
        TE_QWEN_PROFILE.embed_ms += te_qwen_now_ms() - embed_start;
    }

    const char *disabled = getenv("TINYENGINE_DECODE_PROJECT");
    const char *head = te_model_find_tensor(model, "output.weight") != NULL ? "output.weight" : "token_embd.weight";
    const te_gguf_tensor *head_tensor = te_model_find_tensor(model, head);
    if (disabled != NULL && strcmp(disabled, "0") != 0) {
        status = te_qwen_decode_all_layers_exact(model, state, position, head_tensor, out_token_id);
        if (status == TE_STATUS_OK) {
            return TE_STATUS_OK;
        }
        if (status != TE_STATUS_UNSUPPORTED) {
            return status;
        }
    }

    status = te_qwen_decode_all_layers_exact(model, state, position, NULL, NULL);
    if (status != TE_STATUS_OK) {
        return status;
    }
    return te_qwen_project_next(context, state, out_token_id);
}

static te_status te_qwen_project_next(te_context *context, te_qwen_state *state, uint32_t *out_token_id) {
    te_model *model = context->model;
    const char *head = te_model_find_tensor(model, "output.weight") != NULL ? "output.weight" : "token_embd.weight";
    const char *metal_argmax = getenv("TINYENGINE_METAL_ARGMAX");
    const te_gguf_tensor *head_tensor = te_model_find_tensor(model, head);
    if ((metal_argmax == NULL || strcmp(metal_argmax, "0") != 0) &&
        head_tensor != NULL &&
        head_tensor->n_dims == 2u &&
        head_tensor->dims[0] == state->hidden &&
        head_tensor->dims[1] == state->vocab) {
        const double start = te_qwen_now_ms();
        te_status status = te_metal_project_argmax_f32(
            model->mapping,
            model->mapping_len,
            head_tensor->absolute_offset,
            head_tensor->ggml_type,
            state->hidden_buf,
            state->output_norm_weight,
            state->hidden,
            state->vocab,
            model->info.rms_norm_epsilon,
            out_token_id);
        if (status == TE_STATUS_OK) {
            if (TE_QWEN_PROFILE.enabled) {
                te_qwen_profile_add_matvec(head, te_qwen_now_ms() - start);
            }
            return TE_STATUS_OK;
        }
        if (status != TE_STATUS_UNSUPPORTED) {
            return status;
        }
    }
    te_status status = te_rmsnorm_f32(
        state->hidden_buf,
        state->output_norm_weight,
        state->hidden,
        model->info.rms_norm_epsilon,
        state->norm);
    if (status != TE_STATUS_OK) {
        return status;
    }
    if ((metal_argmax == NULL || strcmp(metal_argmax, "0") != 0) &&
        head_tensor != NULL &&
        head_tensor->n_dims == 2u &&
        head_tensor->dims[0] == state->hidden &&
        head_tensor->dims[1] == state->vocab) {
        const double start = te_qwen_now_ms();
        status = te_metal_matvec_argmax_f32(
            model->mapping,
            model->mapping_len,
            head_tensor->absolute_offset,
            head_tensor->ggml_type,
            state->norm,
            state->hidden,
            state->vocab,
            out_token_id);
        if (status == TE_STATUS_OK) {
            if (TE_QWEN_PROFILE.enabled) {
                te_qwen_profile_add_matvec(head, te_qwen_now_ms() - start);
            }
            return TE_STATUS_OK;
        }
        if (status != TE_STATUS_UNSUPPORTED) {
            return status;
        }
    }
    status = te_qwen_matvec_exact(model, head, state->norm, state->hidden, state->logits, state->vocab);
    if (status != TE_STATUS_OK) {
        return status;
    }
    return te_argmax_f32(state->logits, state->vocab, out_token_id);
}

static te_status te_qwen_decode_layer_exact(
    te_model *model,
    const char *q_name,
    const char *k_name,
    const char *v_name,
    const char *output_name,
    const char *gate_name,
    const char *up_name,
    const char *down_name,
    te_qwen_state *state,
    size_t layer,
    size_t position
) {
    const char *disabled = getenv("TINYENGINE_FUSED_DECODE_LAYER");
    if (disabled != NULL && strcmp(disabled, "0") == 0) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (model == NULL || state == NULL || position >= state->context_tokens) {
        return TE_STATUS_INVALID_ARGUMENT;
    }

    const te_gguf_tensor *q_tensor = te_model_find_tensor(model, q_name);
    const te_gguf_tensor *k_tensor = te_model_find_tensor(model, k_name);
    const te_gguf_tensor *v_tensor = te_model_find_tensor(model, v_name);
    const te_gguf_tensor *output_tensor = te_model_find_tensor(model, output_name);
    const te_gguf_tensor *gate_tensor = te_model_find_tensor(model, gate_name);
    const te_gguf_tensor *up_tensor = te_model_find_tensor(model, up_name);
    const te_gguf_tensor *down_tensor = te_model_find_tensor(model, down_name);
    if (q_tensor == NULL || k_tensor == NULL || v_tensor == NULL ||
        output_tensor == NULL || gate_tensor == NULL || up_tensor == NULL || down_tensor == NULL ||
        q_tensor->ggml_type != k_tensor->ggml_type ||
        q_tensor->ggml_type != v_tensor->ggml_type ||
        q_tensor->ggml_type != output_tensor->ggml_type ||
        q_tensor->ggml_type != gate_tensor->ggml_type ||
        q_tensor->ggml_type != up_tensor->ggml_type ||
        q_tensor->ggml_type != down_tensor->ggml_type ||
        q_tensor->n_dims != 2u || k_tensor->n_dims != 2u || v_tensor->n_dims != 2u ||
        output_tensor->n_dims != 2u || gate_tensor->n_dims != 2u ||
        up_tensor->n_dims != 2u || down_tensor->n_dims != 2u ||
        q_tensor->dims[0] != state->hidden || q_tensor->dims[1] != state->hidden ||
        k_tensor->dims[0] != state->hidden || k_tensor->dims[1] != state->kv_dim ||
        v_tensor->dims[0] != state->hidden || v_tensor->dims[1] != state->kv_dim ||
        output_tensor->dims[0] != state->hidden || output_tensor->dims[1] != state->hidden ||
        gate_tensor->dims[0] != state->hidden || gate_tensor->dims[1] != state->ffn ||
        up_tensor->dims[0] != state->hidden || up_tensor->dims[1] != state->ffn ||
        down_tensor->dims[0] != state->ffn || down_tensor->dims[1] != state->hidden) {
        return TE_STATUS_UNSUPPORTED;
    }

    const size_t layer_cache_offset = layer * state->context_tokens * state->kv_dim;
    const size_t rope_half = state->head_dim / 2u;
    const double start = te_qwen_now_ms();
    te_status status = te_metal_decode_layer_f32(
        model->mapping,
        model->mapping_len,
        q_tensor->absolute_offset,
        k_tensor->absolute_offset,
        v_tensor->absolute_offset,
        output_tensor->absolute_offset,
        gate_tensor->absolute_offset,
        up_tensor->absolute_offset,
        down_tensor->absolute_offset,
        q_tensor->ggml_type,
        state->hidden_buf,
        state->attn_norm_weights + layer * state->hidden,
        state->ffn_norm_weights + layer * state->hidden,
        state->q_biases + layer * state->hidden,
        state->k_biases + layer * state->kv_dim,
        state->v_biases + layer * state->kv_dim,
        state->rope_cos + position * rope_half,
        state->rope_sin + position * rope_half,
        state->key_cache + layer_cache_offset,
        state->value_cache + layer_cache_offset,
        position,
        state->context_tokens,
        state->hidden,
        state->kv_dim,
        state->heads,
        state->kv_heads,
        state->head_dim,
        state->ffn,
        model->info.rms_norm_epsilon,
        state->hidden_buf);
    if (status == TE_STATUS_OK && TE_QWEN_PROFILE.enabled) {
        const double elapsed = te_qwen_now_ms() - start;
        const double denom = (double)(state->hidden + state->kv_dim + state->kv_dim +
            state->hidden + state->ffn + state->ffn + state->hidden);
        te_qwen_profile_add_matvec(q_name, elapsed * (double)state->hidden / denom);
        te_qwen_profile_add_matvec(k_name, elapsed * (double)state->kv_dim / denom);
        te_qwen_profile_add_matvec(v_name, elapsed * (double)state->kv_dim / denom);
        te_qwen_profile_add_matvec(output_name, elapsed * (double)state->hidden / denom);
        te_qwen_profile_add_matvec(gate_name, elapsed * (double)state->ffn / denom);
        te_qwen_profile_add_matvec(up_name, elapsed * (double)state->ffn / denom);
        te_qwen_profile_add_matvec(down_name, elapsed * (double)state->hidden / denom);
    }
    return status;
}

static te_status te_qwen_decode_all_layers_exact(
    te_model *model,
    te_qwen_state *state,
    size_t position,
    const te_gguf_tensor *head_tensor,
    uint32_t *out_token_id
) {
    const char *disabled = getenv("TINYENGINE_DECODE_ALL_LAYERS");
    if (disabled != NULL && strcmp(disabled, "0") == 0) {
        return TE_STATUS_UNSUPPORTED;
    }
    const char *legacy_disabled = getenv("TINYENGINE_FUSED_DECODE_LAYER");
    if (legacy_disabled != NULL && strcmp(legacy_disabled, "0") == 0) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (model == NULL || state == NULL || position >= state->context_tokens) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (out_token_id != NULL &&
        (head_tensor == NULL ||
         head_tensor->n_dims != 2u ||
         head_tensor->dims[0] != state->hidden ||
         head_tensor->dims[1] != state->vocab)) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (state->layers > SIZE_MAX / (7u * sizeof(uint64_t))) {
        return TE_STATUS_UNSUPPORTED;
    }

    uint64_t *offsets = (uint64_t *)calloc(state->layers * 7u, sizeof(offsets[0]));
    if (offsets == NULL) {
        return TE_STATUS_OUT_OF_MEMORY;
    }
    uint64_t *q_offsets = offsets;
    uint64_t *k_offsets = q_offsets + state->layers;
    uint64_t *v_offsets = k_offsets + state->layers;
    uint64_t *output_offsets = v_offsets + state->layers;
    uint64_t *gate_offsets = output_offsets + state->layers;
    uint64_t *up_offsets = gate_offsets + state->layers;
    uint64_t *down_offsets = up_offsets + state->layers;

    te_status status = TE_STATUS_OK;
    uint32_t ggml_type = 0u;
    for (size_t layer = 0; layer < state->layers; ++layer) {
        char q_name[64];
        char k_name[64];
        char v_name[64];
        char output_name[64];
        char gate_name[64];
        char up_name[64];
        char down_name[64];
        if (!te_qwen_name(q_name, sizeof(q_name), "blk.%zu.attn_q.weight", layer) ||
            !te_qwen_name(k_name, sizeof(k_name), "blk.%zu.attn_k.weight", layer) ||
            !te_qwen_name(v_name, sizeof(v_name), "blk.%zu.attn_v.weight", layer) ||
            !te_qwen_name(output_name, sizeof(output_name), "blk.%zu.attn_output.weight", layer) ||
            !te_qwen_name(gate_name, sizeof(gate_name), "blk.%zu.ffn_gate.weight", layer) ||
            !te_qwen_name(up_name, sizeof(up_name), "blk.%zu.ffn_up.weight", layer) ||
            !te_qwen_name(down_name, sizeof(down_name), "blk.%zu.ffn_down.weight", layer)) {
            status = TE_STATUS_UNSUPPORTED;
            goto done;
        }
        const te_gguf_tensor *q_tensor = te_model_find_tensor(model, q_name);
        const te_gguf_tensor *k_tensor = te_model_find_tensor(model, k_name);
        const te_gguf_tensor *v_tensor = te_model_find_tensor(model, v_name);
        const te_gguf_tensor *output_tensor = te_model_find_tensor(model, output_name);
        const te_gguf_tensor *gate_tensor = te_model_find_tensor(model, gate_name);
        const te_gguf_tensor *up_tensor = te_model_find_tensor(model, up_name);
        const te_gguf_tensor *down_tensor = te_model_find_tensor(model, down_name);
        if (q_tensor == NULL || k_tensor == NULL || v_tensor == NULL ||
            output_tensor == NULL || gate_tensor == NULL || up_tensor == NULL || down_tensor == NULL ||
            q_tensor->ggml_type != k_tensor->ggml_type ||
            q_tensor->ggml_type != v_tensor->ggml_type ||
            q_tensor->ggml_type != output_tensor->ggml_type ||
            q_tensor->ggml_type != gate_tensor->ggml_type ||
            q_tensor->ggml_type != up_tensor->ggml_type ||
            q_tensor->ggml_type != down_tensor->ggml_type ||
            q_tensor->n_dims != 2u || k_tensor->n_dims != 2u || v_tensor->n_dims != 2u ||
            output_tensor->n_dims != 2u || gate_tensor->n_dims != 2u ||
            up_tensor->n_dims != 2u || down_tensor->n_dims != 2u ||
            q_tensor->dims[0] != state->hidden || q_tensor->dims[1] != state->hidden ||
            k_tensor->dims[0] != state->hidden || k_tensor->dims[1] != state->kv_dim ||
            v_tensor->dims[0] != state->hidden || v_tensor->dims[1] != state->kv_dim ||
            output_tensor->dims[0] != state->hidden || output_tensor->dims[1] != state->hidden ||
            gate_tensor->dims[0] != state->hidden || gate_tensor->dims[1] != state->ffn ||
            up_tensor->dims[0] != state->hidden || up_tensor->dims[1] != state->ffn ||
            down_tensor->dims[0] != state->ffn || down_tensor->dims[1] != state->hidden ||
            (layer != 0u && q_tensor->ggml_type != ggml_type)) {
            status = TE_STATUS_UNSUPPORTED;
            goto done;
        }
        if (layer == 0u) {
            ggml_type = q_tensor->ggml_type;
        }
        q_offsets[layer] = q_tensor->absolute_offset;
        k_offsets[layer] = k_tensor->absolute_offset;
        v_offsets[layer] = v_tensor->absolute_offset;
        output_offsets[layer] = output_tensor->absolute_offset;
        gate_offsets[layer] = gate_tensor->absolute_offset;
        up_offsets[layer] = up_tensor->absolute_offset;
        down_offsets[layer] = down_tensor->absolute_offset;
    }

    const size_t rope_half = state->head_dim / 2u;
    const double start = te_qwen_now_ms();
    status = te_metal_decode_all_layers_f32(
        model->mapping,
        model->mapping_len,
        q_offsets,
        k_offsets,
        v_offsets,
        output_offsets,
        gate_offsets,
        up_offsets,
        down_offsets,
        state->layers,
        ggml_type,
        state->hidden_buf,
        state->attn_norm_weights,
        state->ffn_norm_weights,
        state->q_biases,
        state->k_biases,
        state->v_biases,
        state->rope_cos + position * rope_half,
        state->rope_sin + position * rope_half,
        state->key_cache,
        state->value_cache,
        position,
        state->context_tokens,
        state->hidden,
        state->kv_dim,
        state->heads,
        state->kv_heads,
        state->head_dim,
        state->ffn,
        model->info.rms_norm_epsilon,
        state->hidden_buf,
        head_tensor != NULL ? head_tensor->absolute_offset : 0u,
        head_tensor != NULL ? head_tensor->ggml_type : 0u,
        out_token_id != NULL ? state->output_norm_weight : NULL,
        out_token_id != NULL ? state->vocab : 0u,
        out_token_id);
    if (status == TE_STATUS_OK && TE_QWEN_PROFILE.enabled) {
        const double elapsed = te_qwen_now_ms() - start;
        const double denom = (double)(state->hidden + state->kv_dim + state->kv_dim +
            state->hidden + state->ffn + state->ffn + state->hidden);
        const double per_layer = elapsed / (double)state->layers;
        for (size_t layer = 0; layer < state->layers; ++layer) {
            (void)layer;
            te_qwen_profile_add_matvec("attn_q.weight", per_layer * (double)state->hidden / denom);
            te_qwen_profile_add_matvec("attn_k.weight", per_layer * (double)state->kv_dim / denom);
            te_qwen_profile_add_matvec("attn_v.weight", per_layer * (double)state->kv_dim / denom);
            te_qwen_profile_add_matvec("attn_output.weight", per_layer * (double)state->hidden / denom);
            te_qwen_profile_add_matvec("ffn_gate.weight", per_layer * (double)state->ffn / denom);
            te_qwen_profile_add_matvec("ffn_up.weight", per_layer * (double)state->ffn / denom);
            te_qwen_profile_add_matvec("ffn_down.weight", per_layer * (double)state->hidden / denom);
        }
    }

done:
    free(offsets);
    return status;
}

static te_status te_qwen_prefill_all_layers_exact(
    te_model *model,
    te_qwen_state *state,
    size_t batch,
    float *hidden
) {
    const char *disabled = getenv("TINYENGINE_PREFILL_ALL_LAYERS");
    if (disabled != NULL && strcmp(disabled, "0") == 0) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (model == NULL || state == NULL || hidden == NULL || batch == 0u || batch > state->context_tokens) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (state->layers > SIZE_MAX / (7u * sizeof(uint64_t))) {
        return TE_STATUS_UNSUPPORTED;
    }

    uint64_t *offsets = (uint64_t *)calloc(state->layers * 7u, sizeof(offsets[0]));
    if (offsets == NULL) {
        return TE_STATUS_OUT_OF_MEMORY;
    }
    uint64_t *q_offsets = offsets;
    uint64_t *k_offsets = q_offsets + state->layers;
    uint64_t *v_offsets = k_offsets + state->layers;
    uint64_t *output_offsets = v_offsets + state->layers;
    uint64_t *gate_offsets = output_offsets + state->layers;
    uint64_t *up_offsets = gate_offsets + state->layers;
    uint64_t *down_offsets = up_offsets + state->layers;

    te_status status = TE_STATUS_OK;
    uint32_t ggml_type = 0u;
    for (size_t layer = 0; layer < state->layers; ++layer) {
        char q_name[64];
        char k_name[64];
        char v_name[64];
        char output_name[64];
        char gate_name[64];
        char up_name[64];
        char down_name[64];
        if (!te_qwen_name(q_name, sizeof(q_name), "blk.%zu.attn_q.weight", layer) ||
            !te_qwen_name(k_name, sizeof(k_name), "blk.%zu.attn_k.weight", layer) ||
            !te_qwen_name(v_name, sizeof(v_name), "blk.%zu.attn_v.weight", layer) ||
            !te_qwen_name(output_name, sizeof(output_name), "blk.%zu.attn_output.weight", layer) ||
            !te_qwen_name(gate_name, sizeof(gate_name), "blk.%zu.ffn_gate.weight", layer) ||
            !te_qwen_name(up_name, sizeof(up_name), "blk.%zu.ffn_up.weight", layer) ||
            !te_qwen_name(down_name, sizeof(down_name), "blk.%zu.ffn_down.weight", layer)) {
            status = TE_STATUS_UNSUPPORTED;
            goto done;
        }
        const te_gguf_tensor *q_tensor = te_model_find_tensor(model, q_name);
        const te_gguf_tensor *k_tensor = te_model_find_tensor(model, k_name);
        const te_gguf_tensor *v_tensor = te_model_find_tensor(model, v_name);
        const te_gguf_tensor *output_tensor = te_model_find_tensor(model, output_name);
        const te_gguf_tensor *gate_tensor = te_model_find_tensor(model, gate_name);
        const te_gguf_tensor *up_tensor = te_model_find_tensor(model, up_name);
        const te_gguf_tensor *down_tensor = te_model_find_tensor(model, down_name);
        if (q_tensor == NULL || k_tensor == NULL || v_tensor == NULL ||
            output_tensor == NULL || gate_tensor == NULL || up_tensor == NULL || down_tensor == NULL ||
            q_tensor->ggml_type != k_tensor->ggml_type ||
            q_tensor->ggml_type != v_tensor->ggml_type ||
            q_tensor->ggml_type != output_tensor->ggml_type ||
            q_tensor->ggml_type != gate_tensor->ggml_type ||
            q_tensor->ggml_type != up_tensor->ggml_type ||
            q_tensor->ggml_type != down_tensor->ggml_type ||
            q_tensor->n_dims != 2u || k_tensor->n_dims != 2u || v_tensor->n_dims != 2u ||
            output_tensor->n_dims != 2u || gate_tensor->n_dims != 2u ||
            up_tensor->n_dims != 2u || down_tensor->n_dims != 2u ||
            q_tensor->dims[0] != state->hidden || q_tensor->dims[1] != state->hidden ||
            k_tensor->dims[0] != state->hidden || k_tensor->dims[1] != state->kv_dim ||
            v_tensor->dims[0] != state->hidden || v_tensor->dims[1] != state->kv_dim ||
            output_tensor->dims[0] != state->hidden || output_tensor->dims[1] != state->hidden ||
            gate_tensor->dims[0] != state->hidden || gate_tensor->dims[1] != state->ffn ||
            up_tensor->dims[0] != state->hidden || up_tensor->dims[1] != state->ffn ||
            down_tensor->dims[0] != state->ffn || down_tensor->dims[1] != state->hidden ||
            (layer != 0u && q_tensor->ggml_type != ggml_type)) {
            status = TE_STATUS_UNSUPPORTED;
            goto done;
        }
        if (layer == 0u) {
            ggml_type = q_tensor->ggml_type;
        }
        q_offsets[layer] = q_tensor->absolute_offset;
        k_offsets[layer] = k_tensor->absolute_offset;
        v_offsets[layer] = v_tensor->absolute_offset;
        output_offsets[layer] = output_tensor->absolute_offset;
        gate_offsets[layer] = gate_tensor->absolute_offset;
        up_offsets[layer] = up_tensor->absolute_offset;
        down_offsets[layer] = down_tensor->absolute_offset;
    }

    const double start = te_qwen_now_ms();
    status = te_metal_prefill_all_layers_f32(
        model->mapping,
        model->mapping_len,
        q_offsets,
        k_offsets,
        v_offsets,
        output_offsets,
        gate_offsets,
        up_offsets,
        down_offsets,
        state->layers,
        ggml_type,
        hidden,
        state->attn_norm_weights,
        state->ffn_norm_weights,
        state->q_biases,
        state->k_biases,
        state->v_biases,
        state->rope_cos,
        state->rope_sin,
        state->key_cache,
        state->value_cache,
        batch,
        state->context_tokens,
        state->hidden,
        state->kv_dim,
        state->heads,
        state->kv_heads,
        state->head_dim,
        state->ffn,
        model->info.rms_norm_epsilon,
        hidden);
    if (status == TE_STATUS_OK && TE_QWEN_PROFILE.enabled) {
        const double elapsed = te_qwen_now_ms() - start;
        const double denom = (double)(state->hidden + state->kv_dim + state->kv_dim +
            state->hidden + state->ffn + state->ffn + state->hidden);
        const double per_layer = elapsed / (double)state->layers;
        for (size_t layer = 0; layer < state->layers; ++layer) {
            (void)layer;
            te_qwen_profile_add_matvec("attn_q.weight", per_layer * (double)state->hidden / denom);
            te_qwen_profile_add_matvec("attn_k.weight", per_layer * (double)state->kv_dim / denom);
            te_qwen_profile_add_matvec("attn_v.weight", per_layer * (double)state->kv_dim / denom);
            te_qwen_profile_add_matvec("attn_output.weight", per_layer * (double)state->hidden / denom);
            te_qwen_profile_add_matvec("ffn_gate.weight", per_layer * (double)state->ffn / denom);
            te_qwen_profile_add_matvec("ffn_up.weight", per_layer * (double)state->ffn / denom);
            te_qwen_profile_add_matvec("ffn_down.weight", per_layer * (double)state->hidden / denom);
        }
    }

done:
    free(offsets);
    return status;
}

static te_status te_qwen_prefill_layer_exact(
    te_model *model,
    const char *q_name,
    const char *k_name,
    const char *v_name,
    const char *output_name,
    const char *gate_name,
    const char *up_name,
    const char *down_name,
    te_qwen_state *state,
    size_t layer,
    size_t batch,
    float *hidden
) {
    const char *disabled = getenv("TINYENGINE_FUSED_PREFILL_LAYER");
    if (disabled != NULL && strcmp(disabled, "0") == 0) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (model == NULL || state == NULL || hidden == NULL || batch == 0u || batch > state->context_tokens) {
        return TE_STATUS_INVALID_ARGUMENT;
    }

    const te_gguf_tensor *q_tensor = te_model_find_tensor(model, q_name);
    const te_gguf_tensor *k_tensor = te_model_find_tensor(model, k_name);
    const te_gguf_tensor *v_tensor = te_model_find_tensor(model, v_name);
    const te_gguf_tensor *output_tensor = te_model_find_tensor(model, output_name);
    const te_gguf_tensor *gate_tensor = te_model_find_tensor(model, gate_name);
    const te_gguf_tensor *up_tensor = te_model_find_tensor(model, up_name);
    const te_gguf_tensor *down_tensor = te_model_find_tensor(model, down_name);
    if (q_tensor == NULL || k_tensor == NULL || v_tensor == NULL ||
        output_tensor == NULL || gate_tensor == NULL || up_tensor == NULL || down_tensor == NULL ||
        q_tensor->ggml_type != k_tensor->ggml_type ||
        q_tensor->ggml_type != v_tensor->ggml_type ||
        q_tensor->ggml_type != output_tensor->ggml_type ||
        q_tensor->ggml_type != gate_tensor->ggml_type ||
        q_tensor->ggml_type != up_tensor->ggml_type ||
        q_tensor->ggml_type != down_tensor->ggml_type ||
        q_tensor->n_dims != 2u || k_tensor->n_dims != 2u || v_tensor->n_dims != 2u ||
        output_tensor->n_dims != 2u || gate_tensor->n_dims != 2u ||
        up_tensor->n_dims != 2u || down_tensor->n_dims != 2u ||
        q_tensor->dims[0] != state->hidden || q_tensor->dims[1] != state->hidden ||
        k_tensor->dims[0] != state->hidden || k_tensor->dims[1] != state->kv_dim ||
        v_tensor->dims[0] != state->hidden || v_tensor->dims[1] != state->kv_dim ||
        output_tensor->dims[0] != state->hidden || output_tensor->dims[1] != state->hidden ||
        gate_tensor->dims[0] != state->hidden || gate_tensor->dims[1] != state->ffn ||
        up_tensor->dims[0] != state->hidden || up_tensor->dims[1] != state->ffn ||
        down_tensor->dims[0] != state->ffn || down_tensor->dims[1] != state->hidden) {
        return TE_STATUS_UNSUPPORTED;
    }

    const size_t layer_cache_offset = layer * state->context_tokens * state->kv_dim;
    const double start = te_qwen_now_ms();
    te_status status = te_metal_prefill_layer_f32(
        model->mapping,
        model->mapping_len,
        q_tensor->absolute_offset,
        k_tensor->absolute_offset,
        v_tensor->absolute_offset,
        output_tensor->absolute_offset,
        gate_tensor->absolute_offset,
        up_tensor->absolute_offset,
        down_tensor->absolute_offset,
        q_tensor->ggml_type,
        hidden,
        state->attn_norm_weights + layer * state->hidden,
        state->ffn_norm_weights + layer * state->hidden,
        state->q_biases + layer * state->hidden,
        state->k_biases + layer * state->kv_dim,
        state->v_biases + layer * state->kv_dim,
        state->rope_cos,
        state->rope_sin,
        state->key_cache + layer_cache_offset,
        state->value_cache + layer_cache_offset,
        batch,
        state->context_tokens,
        state->hidden,
        state->kv_dim,
        state->heads,
        state->kv_heads,
        state->head_dim,
        state->ffn,
        model->info.rms_norm_epsilon,
        hidden);
    if (status == TE_STATUS_OK && TE_QWEN_PROFILE.enabled) {
        const double elapsed = te_qwen_now_ms() - start;
        const double denom = (double)(state->hidden + state->kv_dim + state->kv_dim +
            state->hidden + state->ffn + state->ffn + state->hidden);
        te_qwen_profile_add_matvec(q_name, elapsed * (double)state->hidden / denom);
        te_qwen_profile_add_matvec(k_name, elapsed * (double)state->kv_dim / denom);
        te_qwen_profile_add_matvec(v_name, elapsed * (double)state->kv_dim / denom);
        te_qwen_profile_add_matvec(output_name, elapsed * (double)state->hidden / denom);
        te_qwen_profile_add_matvec(gate_name, elapsed * (double)state->ffn / denom);
        te_qwen_profile_add_matvec(up_name, elapsed * (double)state->ffn / denom);
        te_qwen_profile_add_matvec(down_name, elapsed * (double)state->hidden / denom);
    }
    return status;
}

static te_status te_qwen_mlp_exact(
    te_model *model,
    const char *gate_name,
    const char *up_name,
    const char *down_name,
    const float *input,
    size_t hidden,
    size_t ffn,
    float *gate,
    float *up,
    float *swiglu,
    float *out
) {
    const te_gguf_tensor *gate_tensor = te_model_find_tensor(model, gate_name);
    const te_gguf_tensor *up_tensor = te_model_find_tensor(model, up_name);
    const te_gguf_tensor *down_tensor = te_model_find_tensor(model, down_name);
    if (gate_tensor != NULL && up_tensor != NULL && down_tensor != NULL &&
        gate_tensor->ggml_type == up_tensor->ggml_type &&
        gate_tensor->ggml_type == down_tensor->ggml_type &&
        gate_tensor->n_dims == 2u && up_tensor->n_dims == 2u && down_tensor->n_dims == 2u &&
        gate_tensor->dims[0] == hidden && up_tensor->dims[0] == hidden &&
        gate_tensor->dims[1] == ffn && up_tensor->dims[1] == ffn &&
        down_tensor->dims[0] == ffn && down_tensor->dims[1] == hidden) {
        const double start = te_qwen_now_ms();
        te_status status = te_metal_mlp_f32(
            model->mapping,
            model->mapping_len,
            gate_tensor->absolute_offset,
            up_tensor->absolute_offset,
            down_tensor->absolute_offset,
            gate_tensor->ggml_type,
            input,
            hidden,
            ffn,
            out);
        if (status == TE_STATUS_OK) {
            if (TE_QWEN_PROFILE.enabled) {
                const double elapsed = te_qwen_now_ms() - start;
                te_qwen_profile_add_matvec(gate_name, elapsed / 3.0);
                te_qwen_profile_add_matvec(up_name, elapsed / 3.0);
                te_qwen_profile_add_matvec(down_name, elapsed / 3.0);
            }
            return TE_STATUS_OK;
        }
        if (status != TE_STATUS_UNSUPPORTED) {
            return status;
        }
    }

    te_status status = te_qwen_matvec2_exact(model, gate_name, up_name, input, hidden, gate, up, ffn);
    if (status != TE_STATUS_OK) {
        return status;
    }
    status = te_swiglu_f32(gate, up, ffn, swiglu);
    if (status != TE_STATUS_OK) {
        return status;
    }
    return te_qwen_matvec_exact(model, down_name, swiglu, ffn, out, hidden);
}

static te_status te_qwen_post_attn_mlp_exact(
    te_model *model,
    const char *output_name,
    const char *gate_name,
    const char *up_name,
    const char *down_name,
    const float *hidden_in,
    const float *attn,
    const float *ffn_norm_weight,
    size_t hidden,
    size_t ffn,
    float epsilon,
    float *out
) {
    const char *disabled = getenv("TINYENGINE_FUSED_DECODE_POST_ATTN");
    if (disabled != NULL && strcmp(disabled, "0") == 0) {
        return TE_STATUS_UNSUPPORTED;
    }
    const te_gguf_tensor *output_tensor = te_model_find_tensor(model, output_name);
    const te_gguf_tensor *gate_tensor = te_model_find_tensor(model, gate_name);
    const te_gguf_tensor *up_tensor = te_model_find_tensor(model, up_name);
    const te_gguf_tensor *down_tensor = te_model_find_tensor(model, down_name);
    if (output_tensor == NULL || gate_tensor == NULL || up_tensor == NULL || down_tensor == NULL ||
        output_tensor->ggml_type != gate_tensor->ggml_type ||
        output_tensor->ggml_type != up_tensor->ggml_type ||
        output_tensor->ggml_type != down_tensor->ggml_type ||
        output_tensor->n_dims != 2u || gate_tensor->n_dims != 2u ||
        up_tensor->n_dims != 2u || down_tensor->n_dims != 2u ||
        output_tensor->dims[0] != hidden || output_tensor->dims[1] != hidden ||
        gate_tensor->dims[0] != hidden || up_tensor->dims[0] != hidden ||
        gate_tensor->dims[1] != ffn || up_tensor->dims[1] != ffn ||
        down_tensor->dims[0] != ffn || down_tensor->dims[1] != hidden) {
        return TE_STATUS_UNSUPPORTED;
    }

    const double start = te_qwen_now_ms();
    te_status status = te_metal_post_attn_mlp_f32(
        model->mapping,
        model->mapping_len,
        output_tensor->absolute_offset,
        gate_tensor->absolute_offset,
        up_tensor->absolute_offset,
        down_tensor->absolute_offset,
        output_tensor->ggml_type,
        hidden_in,
        attn,
        ffn_norm_weight,
        hidden,
        ffn,
        epsilon,
        out);
    if (status == TE_STATUS_OK) {
        if (TE_QWEN_PROFILE.enabled) {
            const double elapsed = te_qwen_now_ms() - start;
            const double denom = (double)(hidden + ffn + ffn + hidden);
            te_qwen_profile_add_matvec(output_name, elapsed * (double)hidden / denom);
            te_qwen_profile_add_matvec(gate_name, elapsed * (double)ffn / denom);
            te_qwen_profile_add_matvec(up_name, elapsed * (double)ffn / denom);
            te_qwen_profile_add_matvec(down_name, elapsed * (double)hidden / denom);
        }
        return TE_STATUS_OK;
    }
    return status;
}

static te_status te_qwen_mlp_batch_exact(
    te_model *model,
    const char *gate_name,
    const char *up_name,
    const char *down_name,
    const float *input,
    size_t batch,
    size_t hidden,
    size_t ffn,
    float *out
) {
    const te_gguf_tensor *gate_tensor = te_model_find_tensor(model, gate_name);
    const te_gguf_tensor *up_tensor = te_model_find_tensor(model, up_name);
    const te_gguf_tensor *down_tensor = te_model_find_tensor(model, down_name);
    if (gate_tensor == NULL || up_tensor == NULL || down_tensor == NULL ||
        gate_tensor->ggml_type != up_tensor->ggml_type ||
        gate_tensor->ggml_type != down_tensor->ggml_type ||
        gate_tensor->n_dims != 2u || up_tensor->n_dims != 2u || down_tensor->n_dims != 2u ||
        gate_tensor->dims[0] != hidden || up_tensor->dims[0] != hidden ||
        gate_tensor->dims[1] != ffn || up_tensor->dims[1] != ffn ||
        down_tensor->dims[0] != ffn || down_tensor->dims[1] != hidden) {
        return TE_STATUS_UNSUPPORTED;
    }

    const double start = te_qwen_now_ms();
    te_status status = te_metal_mlp_batch_f32(
        model->mapping,
        model->mapping_len,
        gate_tensor->absolute_offset,
        up_tensor->absolute_offset,
        down_tensor->absolute_offset,
        gate_tensor->ggml_type,
        input,
        batch,
        hidden,
        ffn,
        out);
    if (status == TE_STATUS_OK) {
        if (TE_QWEN_PROFILE.enabled) {
            const double elapsed = te_qwen_now_ms() - start;
            te_qwen_profile_add_matvec(gate_name, elapsed / 3.0);
            te_qwen_profile_add_matvec(up_name, elapsed / 3.0);
            te_qwen_profile_add_matvec(down_name, elapsed / 3.0);
        }
        return TE_STATUS_OK;
    }
    return status;
}

static te_status te_qwen_post_attn_mlp_batch_exact(
    te_model *model,
    const char *output_name,
    const char *gate_name,
    const char *up_name,
    const char *down_name,
    const float *hidden_in,
    const float *attn,
    const float *ffn_norm_weight,
    size_t batch,
    size_t hidden,
    size_t ffn,
    float epsilon,
    float *out
) {
    const char *disabled = getenv("TINYENGINE_FUSED_POST_ATTN");
    if (disabled != NULL && strcmp(disabled, "0") == 0) {
        return TE_STATUS_UNSUPPORTED;
    }
    const te_gguf_tensor *output_tensor = te_model_find_tensor(model, output_name);
    const te_gguf_tensor *gate_tensor = te_model_find_tensor(model, gate_name);
    const te_gguf_tensor *up_tensor = te_model_find_tensor(model, up_name);
    const te_gguf_tensor *down_tensor = te_model_find_tensor(model, down_name);
    if (output_tensor == NULL || gate_tensor == NULL || up_tensor == NULL || down_tensor == NULL ||
        output_tensor->ggml_type != gate_tensor->ggml_type ||
        output_tensor->ggml_type != up_tensor->ggml_type ||
        output_tensor->ggml_type != down_tensor->ggml_type ||
        output_tensor->n_dims != 2u || gate_tensor->n_dims != 2u ||
        up_tensor->n_dims != 2u || down_tensor->n_dims != 2u ||
        output_tensor->dims[0] != hidden || output_tensor->dims[1] != hidden ||
        gate_tensor->dims[0] != hidden || up_tensor->dims[0] != hidden ||
        gate_tensor->dims[1] != ffn || up_tensor->dims[1] != ffn ||
        down_tensor->dims[0] != ffn || down_tensor->dims[1] != hidden) {
        return TE_STATUS_UNSUPPORTED;
    }

    const double start = te_qwen_now_ms();
    te_status status = te_metal_post_attn_mlp_batch_f32(
        model->mapping,
        model->mapping_len,
        output_tensor->absolute_offset,
        gate_tensor->absolute_offset,
        up_tensor->absolute_offset,
        down_tensor->absolute_offset,
        output_tensor->ggml_type,
        hidden_in,
        attn,
        ffn_norm_weight,
        batch,
        hidden,
        ffn,
        epsilon,
        out);
    if (status == TE_STATUS_OK) {
        if (TE_QWEN_PROFILE.enabled) {
            const double elapsed = te_qwen_now_ms() - start;
            const double denom = (double)(hidden + ffn + ffn + hidden);
            te_qwen_profile_add_matvec(output_name, elapsed * (double)hidden / denom);
            te_qwen_profile_add_matvec(gate_name, elapsed * (double)ffn / denom);
            te_qwen_profile_add_matvec(up_name, elapsed * (double)ffn / denom);
            te_qwen_profile_add_matvec(down_name, elapsed * (double)hidden / denom);
        }
        return TE_STATUS_OK;
    }
    return status;
}

static te_status te_qwen_read_f32_exact(te_model *model, const char *name, float *out, size_t len) {
    size_t written = 0;
    te_status status = te_model_read_f32_tensor(model, name, out, len, &written);
    if (status != TE_STATUS_OK) {
        return status;
    }
    return written == len ? TE_STATUS_OK : TE_STATUS_RUNTIME_ERROR;
}

static te_status te_qwen_rope_cached(
    float *values,
    size_t heads,
    size_t head_dim,
    const float *cos_table,
    const float *sin_table,
    size_t position
) {
    if (values == NULL || cos_table == NULL || sin_table == NULL ||
        heads == 0u || head_dim == 0u || (head_dim % 2u) != 0u) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    const size_t half = head_dim / 2u;
    const float *cos_row = cos_table + position * half;
    const float *sin_row = sin_table + position * half;
    for (size_t head = 0; head < heads; ++head) {
        const size_t offset = head * head_dim;
        for (size_t index = 0; index < half; ++index) {
            const float a = values[offset + index];
            const float b = values[offset + index + half];
            const float cos_value = cos_row[index];
            const float sin_value = sin_row[index];
            values[offset + index] = a * cos_value - b * sin_value;
            values[offset + index + half] = b * cos_value + a * sin_value;
        }
    }
    return TE_STATUS_OK;
}

static te_status te_qwen_matvec_exact(
    te_model *model,
    const char *name,
    const float *input,
    size_t input_len,
    float *out,
    size_t out_len
) {
    size_t written = 0;
    const double start = te_qwen_now_ms();
    te_status status = te_model_matvec_f32(model, name, input, input_len, out, out_len, &written);
    if (TE_QWEN_PROFILE.enabled) {
        te_qwen_profile_add_matvec(name, te_qwen_now_ms() - start);
    }
    if (status != TE_STATUS_OK) {
        return status;
    }
    return written == out_len ? TE_STATUS_OK : TE_STATUS_RUNTIME_ERROR;
}

static te_status te_qwen_matvec_batch_exact(
    te_model *model,
    const char *name,
    const float *input,
    size_t batch,
    size_t input_len,
    float *out,
    size_t out_len
) {
    const te_gguf_tensor *tensor = te_model_find_tensor(model, name);
    if (tensor == NULL ||
        tensor->n_dims != 2u ||
        tensor->dims[0] != input_len ||
        tensor->dims[1] != out_len) {
        return TE_STATUS_UNSUPPORTED;
    }

    const double start = te_qwen_now_ms();
    te_status status = te_metal_matvec_batch_f32(
        model->mapping,
        model->mapping_len,
        tensor->absolute_offset,
        tensor->ggml_type,
        input,
        batch,
        input_len,
        out_len,
        out);
    if (TE_QWEN_PROFILE.enabled && status == TE_STATUS_OK) {
        te_qwen_profile_add_matvec(name, te_qwen_now_ms() - start);
    }
    return status;
}

static int te_qwen_profile_enabled(void) {
    const char *value = getenv("TINYENGINE_PROFILE");
    return value != NULL && strcmp(value, "0") != 0;
}

static double te_qwen_now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void te_qwen_profile_reset(void) {
    memset(&TE_QWEN_PROFILE, 0, sizeof(TE_QWEN_PROFILE));
    TE_QWEN_PROFILE.enabled = te_qwen_profile_enabled();
}

static void te_qwen_profile_add_matvec(const char *name, double elapsed_ms) {
    if (strstr(name, "attn_q.weight") != NULL) {
        TE_QWEN_PROFILE.q_ms += elapsed_ms;
        TE_QWEN_PROFILE.q_calls += 1u;
    } else if (strstr(name, "attn_k.weight") != NULL) {
        TE_QWEN_PROFILE.k_ms += elapsed_ms;
        TE_QWEN_PROFILE.k_calls += 1u;
    } else if (strstr(name, "attn_v.weight") != NULL) {
        TE_QWEN_PROFILE.v_ms += elapsed_ms;
        TE_QWEN_PROFILE.v_calls += 1u;
    } else if (strstr(name, "attn_output.weight") != NULL) {
        TE_QWEN_PROFILE.o_ms += elapsed_ms;
        TE_QWEN_PROFILE.o_calls += 1u;
    } else if (strstr(name, "ffn_gate.weight") != NULL) {
        TE_QWEN_PROFILE.gate_ms += elapsed_ms;
        TE_QWEN_PROFILE.gate_calls += 1u;
    } else if (strstr(name, "ffn_up.weight") != NULL) {
        TE_QWEN_PROFILE.up_ms += elapsed_ms;
        TE_QWEN_PROFILE.up_calls += 1u;
    } else if (strstr(name, "ffn_down.weight") != NULL) {
        TE_QWEN_PROFILE.down_ms += elapsed_ms;
        TE_QWEN_PROFILE.down_calls += 1u;
    } else if (strcmp(name, "output.weight") == 0 || strcmp(name, "token_embd.weight") == 0) {
        TE_QWEN_PROFILE.lm_head_ms += elapsed_ms;
        TE_QWEN_PROFILE.lm_head_calls += 1u;
    } else {
        TE_QWEN_PROFILE.other_matvec_ms += elapsed_ms;
        TE_QWEN_PROFILE.other_matvec_calls += 1u;
    }
}

static void te_qwen_profile_print(void) {
    const double matvec_ms =
        TE_QWEN_PROFILE.q_ms +
        TE_QWEN_PROFILE.k_ms +
        TE_QWEN_PROFILE.v_ms +
        TE_QWEN_PROFILE.o_ms +
        TE_QWEN_PROFILE.gate_ms +
        TE_QWEN_PROFILE.up_ms +
        TE_QWEN_PROFILE.down_ms +
        TE_QWEN_PROFILE.lm_head_ms +
        TE_QWEN_PROFILE.other_matvec_ms;
    fprintf(
        stderr,
        "tinyengine_profile: total_ms=%.2f tokenize_ms=%.2f init_ms=%.2f embed_ms=%.2f matvec_ms=%.2f "
        "q=%.2f/%llu k=%.2f/%llu v=%.2f/%llu o=%.2f/%llu gate=%.2f/%llu up=%.2f/%llu "
        "down=%.2f/%llu lm_head=%.2f/%llu other=%.2f/%llu\n",
        TE_QWEN_PROFILE.total_ms,
        TE_QWEN_PROFILE.tokenize_ms,
        TE_QWEN_PROFILE.init_ms,
        TE_QWEN_PROFILE.embed_ms,
        matvec_ms,
        TE_QWEN_PROFILE.q_ms,
        (unsigned long long)TE_QWEN_PROFILE.q_calls,
        TE_QWEN_PROFILE.k_ms,
        (unsigned long long)TE_QWEN_PROFILE.k_calls,
        TE_QWEN_PROFILE.v_ms,
        (unsigned long long)TE_QWEN_PROFILE.v_calls,
        TE_QWEN_PROFILE.o_ms,
        (unsigned long long)TE_QWEN_PROFILE.o_calls,
        TE_QWEN_PROFILE.gate_ms,
        (unsigned long long)TE_QWEN_PROFILE.gate_calls,
        TE_QWEN_PROFILE.up_ms,
        (unsigned long long)TE_QWEN_PROFILE.up_calls,
        TE_QWEN_PROFILE.down_ms,
        (unsigned long long)TE_QWEN_PROFILE.down_calls,
        TE_QWEN_PROFILE.lm_head_ms,
        (unsigned long long)TE_QWEN_PROFILE.lm_head_calls,
        TE_QWEN_PROFILE.other_matvec_ms,
        (unsigned long long)TE_QWEN_PROFILE.other_matvec_calls);
}

static te_status te_qwen_matvec2_exact(
    te_model *model,
    const char *name_a,
    const char *name_b,
    const float *input,
    size_t input_len,
    float *out_a,
    float *out_b,
    size_t out_len
) {
    const te_gguf_tensor *tensor_a = te_model_find_tensor(model, name_a);
    const te_gguf_tensor *tensor_b = te_model_find_tensor(model, name_b);
    if (tensor_a != NULL && tensor_b != NULL &&
        tensor_a->ggml_type == tensor_b->ggml_type &&
        tensor_a->n_dims == 2u && tensor_b->n_dims == 2u &&
        tensor_a->dims[0] == input_len && tensor_b->dims[0] == input_len &&
        tensor_a->dims[1] == out_len && tensor_b->dims[1] == out_len) {
        const double start = te_qwen_now_ms();
        te_status status = te_metal_matvec2_f32(
            model->mapping,
            model->mapping_len,
            tensor_a->absolute_offset,
            tensor_b->absolute_offset,
            tensor_a->ggml_type,
            input,
            input_len,
            out_len,
            out_a,
            out_b);
        if (status == TE_STATUS_OK) {
            if (TE_QWEN_PROFILE.enabled) {
                const double elapsed = te_qwen_now_ms() - start;
                te_qwen_profile_add_matvec(name_a, elapsed * 0.5);
                te_qwen_profile_add_matvec(name_b, elapsed * 0.5);
            }
            return TE_STATUS_OK;
        }
        if (status != TE_STATUS_UNSUPPORTED) {
            return status;
        }
    }

    te_status status = te_qwen_matvec_exact(model, name_a, input, input_len, out_a, out_len);
    if (status != TE_STATUS_OK) {
        return status;
    }
    return te_qwen_matvec_exact(model, name_b, input, input_len, out_b, out_len);
}

static te_status te_qwen_qkv_exact(
    te_model *model,
    const char *q_name,
    const char *k_name,
    const char *v_name,
    const float *input,
    size_t hidden,
    size_t kv,
    float *q,
    float *k,
    float *v
) {
    const te_gguf_tensor *q_tensor = te_model_find_tensor(model, q_name);
    const te_gguf_tensor *k_tensor = te_model_find_tensor(model, k_name);
    const te_gguf_tensor *v_tensor = te_model_find_tensor(model, v_name);
    const char *qkv_env = getenv("TINYENGINE_METAL_QKV");
    if ((qkv_env == NULL || strcmp(qkv_env, "0") != 0) &&
        q_tensor != NULL && k_tensor != NULL && v_tensor != NULL &&
        q_tensor->ggml_type == k_tensor->ggml_type &&
        q_tensor->ggml_type == v_tensor->ggml_type &&
        q_tensor->n_dims == 2u && k_tensor->n_dims == 2u && v_tensor->n_dims == 2u &&
        q_tensor->dims[0] == hidden && q_tensor->dims[1] == hidden &&
        k_tensor->dims[0] == hidden && k_tensor->dims[1] == kv &&
        v_tensor->dims[0] == hidden && v_tensor->dims[1] == kv) {
        const double start = te_qwen_now_ms();
        te_status status = te_metal_qkv_f32(
            model->mapping,
            model->mapping_len,
            q_tensor->absolute_offset,
            k_tensor->absolute_offset,
            v_tensor->absolute_offset,
            q_tensor->ggml_type,
            input,
            hidden,
            kv,
            q,
            k,
            v);
        if (status == TE_STATUS_OK) {
            if (TE_QWEN_PROFILE.enabled) {
                const double elapsed = te_qwen_now_ms() - start;
                const double denom = (double)(hidden + kv + kv);
                te_qwen_profile_add_matvec(q_name, elapsed * (double)hidden / denom);
                te_qwen_profile_add_matvec(k_name, elapsed * (double)kv / denom);
                te_qwen_profile_add_matvec(v_name, elapsed * (double)kv / denom);
            }
            return TE_STATUS_OK;
        }
        if (status != TE_STATUS_UNSUPPORTED) {
            return status;
        }
    }

    te_status status = te_qwen_matvec_exact(model, q_name, input, hidden, q, hidden);
    if (status != TE_STATUS_OK) {
        return status;
    }
    status = te_qwen_matvec_exact(model, k_name, input, hidden, k, kv);
    if (status != TE_STATUS_OK) {
        return status;
    }
    return te_qwen_matvec_exact(model, v_name, input, hidden, v, kv);
}

static te_status te_qwen_qkv_batch_exact(
    te_model *model,
    const char *q_name,
    const char *k_name,
    const char *v_name,
    const float *input,
    size_t batch,
    size_t hidden,
    size_t kv,
    float *q,
    float *k,
    float *v
) {
    const te_gguf_tensor *q_tensor = te_model_find_tensor(model, q_name);
    const te_gguf_tensor *k_tensor = te_model_find_tensor(model, k_name);
    const te_gguf_tensor *v_tensor = te_model_find_tensor(model, v_name);
    if (q_tensor == NULL || k_tensor == NULL || v_tensor == NULL ||
        q_tensor->ggml_type != k_tensor->ggml_type ||
        q_tensor->ggml_type != v_tensor->ggml_type ||
        q_tensor->n_dims != 2u || k_tensor->n_dims != 2u || v_tensor->n_dims != 2u ||
        q_tensor->dims[0] != hidden || q_tensor->dims[1] != hidden ||
        k_tensor->dims[0] != hidden || k_tensor->dims[1] != kv ||
        v_tensor->dims[0] != hidden || v_tensor->dims[1] != kv) {
        return TE_STATUS_UNSUPPORTED;
    }

    const double start = te_qwen_now_ms();
    te_status status = te_metal_qkv_batch_f32(
        model->mapping,
        model->mapping_len,
        q_tensor->absolute_offset,
        k_tensor->absolute_offset,
        v_tensor->absolute_offset,
        q_tensor->ggml_type,
        input,
        batch,
        hidden,
        kv,
        q,
        k,
        v);
    if (status == TE_STATUS_OK) {
        if (TE_QWEN_PROFILE.enabled) {
            const double elapsed = te_qwen_now_ms() - start;
            const double denom = (double)(hidden + kv + kv);
            te_qwen_profile_add_matvec(q_name, elapsed * (double)hidden / denom);
            te_qwen_profile_add_matvec(k_name, elapsed * (double)kv / denom);
            te_qwen_profile_add_matvec(v_name, elapsed * (double)kv / denom);
        }
        return TE_STATUS_OK;
    }
    return status;
}

static int te_qwen_name(char *out, size_t out_len, const char *format, size_t layer) {
    return out != NULL &&
           out_len != 0u &&
           snprintf(out, out_len, format, layer) > 0 &&
           strlen(out) < out_len;
}

static int te_qwen_checked_mul(size_t lhs, size_t rhs, size_t *out) {
    if (out == NULL || (lhs != 0u && rhs > SIZE_MAX / lhs)) {
        return 0;
    }
    *out = lhs * rhs;
    return 1;
}

static int te_qwen_stop_token(const te_model *model, uint32_t token_id) {
    return token_id == model->tokenizer.eos_token_id || token_id == model->tokenizer.padding_token_id;
}

static te_status te_qwen_emit_token(
    te_model *model,
    uint32_t token_id,
    te_token_callback callback,
    void *userdata
) {
    if (callback == NULL) {
        return TE_STATUS_OK;
    }
    char stack_text[256];
    size_t written = 0;
    te_status status = te_model_detokenize(model, &token_id, 1u, 1, stack_text, sizeof(stack_text), &written);
    if (status == TE_STATUS_OK) {
        callback(stack_text, token_id, userdata);
        return TE_STATUS_OK;
    }
    if (written == 0u || written > SIZE_MAX - 1u) {
        return status;
    }
    char *heap_text = (char *)malloc(written + 1u);
    if (heap_text == NULL) {
        return TE_STATUS_OUT_OF_MEMORY;
    }
    status = te_model_detokenize(model, &token_id, 1u, 1, heap_text, written + 1u, &written);
    if (status == TE_STATUS_OK) {
        callback(heap_text, token_id, userdata);
    }
    free(heap_text);
    return status;
}
