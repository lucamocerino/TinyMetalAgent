#include "tinyengine_internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct te_token_vec {
    uint32_t *items;
    size_t len;
    size_t cap;
} te_token_vec;

static void te_string_map_destroy(te_string_map *map);
static te_status te_string_map_init(te_string_map *map, size_t count);
static te_status te_string_map_insert(te_string_map *map, const char *key, uint32_t value);
static int te_string_map_get(const te_string_map *map, const char *key, uint32_t *out_value);
static te_status te_string_map_get_pair(
    const te_string_map *map,
    const char *left,
    const char *right,
    uint32_t *out_value,
    int *out_found
);
static uint64_t te_hash_string(const char *value);
static size_t te_next_power_of_two(size_t value);
static int te_is_gpt2_identity_byte(uint32_t byte);
static uint32_t te_byte_to_unicode(uint8_t byte);
static int te_unicode_to_byte(uint32_t codepoint, uint8_t *out);
static size_t te_utf8_encoded_len(uint32_t codepoint);
static int te_utf8_encode(uint32_t codepoint, char out[4], size_t *out_len);
static int te_utf8_decode(const char *text, size_t len, size_t *pos, uint32_t *out_codepoint);
static char *te_symbol_from_byte(uint8_t byte);
static char *te_concat2(const char *left, const char *right);
static void te_free_symbols(char **symbols, size_t count);
static te_status te_token_vec_push(te_token_vec *vec, uint32_t token);
static void te_token_vec_free(te_token_vec *vec);
static int te_find_special_token(
    const te_model *model,
    const char *text,
    size_t pos,
    uint32_t *out_id,
    size_t *out_len
);
static te_status te_bpe_segment(
    const te_model *model,
    const uint8_t *bytes,
    size_t len,
    te_token_vec *out
);

te_status te_format_qwen_chat_prompt(
    const char *prompt,
    char *out,
    size_t out_capacity,
    size_t *out_written
) {
    static const char *const prefix = "<|im_start|>user\n";
    static const char *const suffix = "<|im_end|>\n<|im_start|>assistant\n";
    if (prompt == NULL || out_written == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }

    const size_t prefix_len = strlen(prefix);
    const size_t prompt_len = strlen(prompt);
    const size_t suffix_len = strlen(suffix);
    if (prompt_len > SIZE_MAX - prefix_len ||
        suffix_len > SIZE_MAX - prefix_len - prompt_len) {
        return TE_STATUS_UNSUPPORTED;
    }
    const size_t required = prefix_len + prompt_len + suffix_len;
    *out_written = required;
    if (out == NULL || out_capacity <= required) {
        return TE_STATUS_INVALID_ARGUMENT;
    }

    memcpy(out, prefix, prefix_len);
    memcpy(out + prefix_len, prompt, prompt_len);
    memcpy(out + prefix_len + prompt_len, suffix, suffix_len);
    out[required] = '\0';
    return TE_STATUS_OK;
}

te_status te_model_tokenize(
    const te_model *model,
    const char *text,
    int parse_special,
    uint32_t *out_tokens,
    size_t out_capacity,
    size_t *out_written
) {
    if (model == NULL || text == NULL || out_written == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (model->tokens == NULL || model->token_count == 0u || model->token_to_id.entries == NULL) {
        return TE_STATUS_UNSUPPORTED;
    }

    te_token_vec vec = {0};
    const size_t text_len = strlen(text);
    size_t pos = 0;
    te_status status = TE_STATUS_OK;
    while (pos < text_len) {
        uint32_t special_id = 0;
        size_t special_len = 0;
        if (parse_special &&
            te_find_special_token(model, text, pos, &special_id, &special_len) &&
            special_len != 0u) {
            status = te_token_vec_push(&vec, special_id);
            if (status != TE_STATUS_OK) {
                goto done;
            }
            pos += special_len;
            continue;
        }

        size_t end = text_len;
        if (parse_special) {
            for (size_t scan = pos + 1u; scan < text_len; ++scan) {
                if (te_find_special_token(model, text, scan, &special_id, &special_len)) {
                    end = scan;
                    break;
                }
            }
        }
        status = te_bpe_segment(model, (const uint8_t *)text + pos, end - pos, &vec);
        if (status != TE_STATUS_OK) {
            goto done;
        }
        pos = end;
    }

    *out_written = vec.len;
    if (out_tokens == NULL || out_capacity < vec.len) {
        status = TE_STATUS_INVALID_ARGUMENT;
        goto done;
    }
    if (vec.len != 0u) {
        memcpy(out_tokens, vec.items, vec.len * sizeof(out_tokens[0]));
    }

done:
    te_token_vec_free(&vec);
    return status;
}

te_status te_model_detokenize(
    const te_model *model,
    const uint32_t *tokens,
    size_t token_count,
    int skip_special,
    char *out,
    size_t out_capacity,
    size_t *out_written
) {
    if (model == NULL || tokens == NULL || out_written == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (model->tokens == NULL || model->token_count == 0u) {
        return TE_STATUS_UNSUPPORTED;
    }

    size_t required = 0;
    int overflow = 0;
    if (out != NULL && out_capacity != 0u) {
        out[0] = '\0';
    }

    for (size_t token_index = 0; token_index < token_count; ++token_index) {
        const uint32_t token_id = tokens[token_index];
        if ((uint64_t)token_id >= model->token_count || model->tokens[token_id].text == NULL) {
            return TE_STATUS_INVALID_ARGUMENT;
        }
        const te_token_entry *entry = &model->tokens[token_id];
        if (skip_special && entry->is_special) {
            continue;
        }

        const char *piece = entry->text;
        const size_t piece_len = strlen(piece);
        size_t pos = 0;
        while (pos < piece_len) {
            const size_t before = pos;
            uint32_t codepoint = 0;
            uint8_t byte = 0;
            char encoded[4];
            size_t encoded_len = 0;
            if (!te_utf8_decode(piece, piece_len, &pos, &codepoint)) {
                return TE_STATUS_RUNTIME_ERROR;
            }
            if (te_unicode_to_byte(codepoint, &byte)) {
                if (out != NULL && required + 1u < out_capacity) {
                    out[required] = (char)byte;
                } else {
                    overflow = 1;
                }
                required += 1u;
            } else {
                if (!te_utf8_encode(codepoint, encoded, &encoded_len)) {
                    return TE_STATUS_RUNTIME_ERROR;
                }
                if (out != NULL && encoded_len < out_capacity && required <= out_capacity - encoded_len - 1u) {
                    memcpy(out + required, encoded, encoded_len);
                } else {
                    overflow = 1;
                }
                required += encoded_len;
            }
            if (pos == before) {
                return TE_STATUS_RUNTIME_ERROR;
            }
        }
    }

    *out_written = required;
    if (out == NULL || out_capacity <= required || overflow) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    out[required] = '\0';
    return TE_STATUS_OK;
}

void te_tokenizer_release(te_model *model) {
    if (model == NULL) {
        return;
    }
    te_string_map_destroy(&model->token_to_id);
    te_string_map_destroy(&model->merge_to_rank);
    if (model->tokens != NULL) {
        for (uint64_t index = 0; index < model->token_count; ++index) {
            free(model->tokens[index].text);
        }
        free(model->tokens);
        model->tokens = NULL;
    }
    if (model->merges != NULL) {
        for (uint64_t index = 0; index < model->merge_count; ++index) {
            free(model->merges[index].text);
        }
        free(model->merges);
        model->merges = NULL;
    }
    model->token_count = 0;
    model->merge_count = 0;
}

te_status te_tokenizer_build_maps(te_model *model) {
    if (model == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (model->tokens == NULL || model->token_count == 0u) {
        return TE_STATUS_OK;
    }
    if (model->token_to_id.entries != NULL) {
        return TE_STATUS_OK;
    }

    te_status status = te_string_map_init(&model->token_to_id, (size_t)model->token_count);
    if (status != TE_STATUS_OK) {
        return status;
    }
    for (uint64_t index = 0; index < model->token_count; ++index) {
        status = te_string_map_insert(&model->token_to_id, model->tokens[index].text, (uint32_t)index);
        if (status != TE_STATUS_OK) {
            te_string_map_destroy(&model->token_to_id);
            return status;
        }
    }

    status = te_string_map_init(&model->merge_to_rank, (size_t)model->merge_count);
    if (status != TE_STATUS_OK) {
        te_string_map_destroy(&model->token_to_id);
        return status;
    }
    for (uint64_t index = 0; index < model->merge_count; ++index) {
        status = te_string_map_insert(&model->merge_to_rank, model->merges[index].text, (uint32_t)index);
        if (status != TE_STATUS_OK) {
            te_string_map_destroy(&model->token_to_id);
            te_string_map_destroy(&model->merge_to_rank);
            return status;
        }
    }
    return TE_STATUS_OK;
}

static void te_string_map_destroy(te_string_map *map) {
    if (map == NULL) {
        return;
    }
    free(map->entries);
    map->entries = NULL;
    map->capacity = 0;
    map->count = 0;
}

static te_status te_string_map_init(te_string_map *map, size_t count) {
    if (map == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (count > SIZE_MAX / 4u) {
        return TE_STATUS_UNSUPPORTED;
    }
    const size_t capacity = te_next_power_of_two(count * 2u + 16u);
    map->entries = (te_string_map_entry *)calloc(capacity, sizeof(map->entries[0]));
    if (map->entries == NULL && capacity != 0u) {
        return TE_STATUS_OUT_OF_MEMORY;
    }
    map->capacity = capacity;
    map->count = 0;
    return TE_STATUS_OK;
}

static te_status te_string_map_insert(te_string_map *map, const char *key, uint32_t value) {
    if (map == NULL || key == NULL || map->entries == NULL || map->capacity == 0u) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    uint64_t hash = te_hash_string(key);
    for (size_t probe = 0; probe < map->capacity; ++probe) {
        const size_t index = (size_t)((hash + probe) & (uint64_t)(map->capacity - 1u));
        if (!map->entries[index].occupied) {
            map->entries[index].key = key;
            map->entries[index].value = value;
            map->entries[index].occupied = 1u;
            map->count += 1u;
            return TE_STATUS_OK;
        }
        if (strcmp(map->entries[index].key, key) == 0) {
            map->entries[index].value = value;
            return TE_STATUS_OK;
        }
    }
    return TE_STATUS_RUNTIME_ERROR;
}

static int te_string_map_get(const te_string_map *map, const char *key, uint32_t *out_value) {
    if (map == NULL || key == NULL || out_value == NULL || map->entries == NULL || map->capacity == 0u) {
        return 0;
    }
    uint64_t hash = te_hash_string(key);
    for (size_t probe = 0; probe < map->capacity; ++probe) {
        const size_t index = (size_t)((hash + probe) & (uint64_t)(map->capacity - 1u));
        if (!map->entries[index].occupied) {
            return 0;
        }
        if (strcmp(map->entries[index].key, key) == 0) {
            *out_value = map->entries[index].value;
            return 1;
        }
    }
    return 0;
}

static te_status te_string_map_get_pair(
    const te_string_map *map,
    const char *left,
    const char *right,
    uint32_t *out_value,
    int *out_found
) {
    if (left == NULL || right == NULL || out_value == NULL || out_found == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    *out_found = 0;

    const size_t left_len = strlen(left);
    const size_t right_len = strlen(right);
    if (left_len > SIZE_MAX - right_len - 2u) {
        return TE_STATUS_UNSUPPORTED;
    }
    const size_t key_len = left_len + right_len + 1u;
    char stack_key[1024];
    char *key = stack_key;
    if (key_len + 1u > sizeof(stack_key)) {
        key = (char *)malloc(key_len + 1u);
        if (key == NULL) {
            return TE_STATUS_OUT_OF_MEMORY;
        }
    }

    memcpy(key, left, left_len);
    key[left_len] = ' ';
    memcpy(key + left_len + 1u, right, right_len);
    key[key_len] = '\0';
    *out_found = te_string_map_get(map, key, out_value);
    if (key != stack_key) {
        free(key);
    }
    return TE_STATUS_OK;
}

static uint64_t te_hash_string(const char *value) {
    uint64_t hash = 1469598103934665603ull;
    while (*value != '\0') {
        hash ^= (uint8_t)*value;
        hash *= 1099511628211ull;
        ++value;
    }
    return hash;
}

static size_t te_next_power_of_two(size_t value) {
    size_t out = 1u;
    while (out < value && out <= SIZE_MAX / 2u) {
        out <<= 1u;
    }
    return out;
}

static int te_is_gpt2_identity_byte(uint32_t byte) {
    return (byte >= 33u && byte <= 126u) ||
           (byte >= 161u && byte <= 172u) ||
           (byte >= 174u && byte <= 255u);
}

static uint32_t te_byte_to_unicode(uint8_t byte) {
    if (te_is_gpt2_identity_byte(byte)) {
        return byte;
    }
    uint32_t extra = 0;
    for (uint32_t candidate = 0; candidate < 256u; ++candidate) {
        if (te_is_gpt2_identity_byte(candidate)) {
            continue;
        }
        if (candidate == byte) {
            return 256u + extra;
        }
        ++extra;
    }
    return byte;
}

static int te_unicode_to_byte(uint32_t codepoint, uint8_t *out) {
    if (out == NULL) {
        return 0;
    }
    for (uint32_t byte = 0; byte < 256u; ++byte) {
        if (te_byte_to_unicode((uint8_t)byte) == codepoint) {
            *out = (uint8_t)byte;
            return 1;
        }
    }
    return 0;
}

static size_t te_utf8_encoded_len(uint32_t codepoint) {
    if (codepoint <= 0x7fu) {
        return 1u;
    }
    if (codepoint <= 0x7ffu) {
        return 2u;
    }
    if (codepoint <= 0xffffu) {
        return 3u;
    }
    if (codepoint <= 0x10ffffu) {
        return 4u;
    }
    return 0u;
}

static int te_utf8_encode(uint32_t codepoint, char out[4], size_t *out_len) {
    const size_t len = te_utf8_encoded_len(codepoint);
    if (out == NULL || out_len == NULL || len == 0u) {
        return 0;
    }
    if (len == 1u) {
        out[0] = (char)codepoint;
    } else if (len == 2u) {
        out[0] = (char)(0xc0u | (codepoint >> 6u));
        out[1] = (char)(0x80u | (codepoint & 0x3fu));
    } else if (len == 3u) {
        out[0] = (char)(0xe0u | (codepoint >> 12u));
        out[1] = (char)(0x80u | ((codepoint >> 6u) & 0x3fu));
        out[2] = (char)(0x80u | (codepoint & 0x3fu));
    } else {
        out[0] = (char)(0xf0u | (codepoint >> 18u));
        out[1] = (char)(0x80u | ((codepoint >> 12u) & 0x3fu));
        out[2] = (char)(0x80u | ((codepoint >> 6u) & 0x3fu));
        out[3] = (char)(0x80u | (codepoint & 0x3fu));
    }
    *out_len = len;
    return 1;
}

static int te_utf8_decode(const char *text, size_t len, size_t *pos, uint32_t *out_codepoint) {
    if (text == NULL || pos == NULL || out_codepoint == NULL || *pos >= len) {
        return 0;
    }
    const uint8_t first = (uint8_t)text[*pos];
    if (first < 0x80u) {
        *out_codepoint = first;
        *pos += 1u;
        return 1;
    }

    size_t need = 0;
    uint32_t value = 0;
    if ((first & 0xe0u) == 0xc0u) {
        need = 2u;
        value = first & 0x1fu;
    } else if ((first & 0xf0u) == 0xe0u) {
        need = 3u;
        value = first & 0x0fu;
    } else if ((first & 0xf8u) == 0xf0u) {
        need = 4u;
        value = first & 0x07u;
    } else {
        return 0;
    }
    if (need == 0u || *pos > len - need) {
        return 0;
    }
    for (size_t index = 1u; index < need; ++index) {
        const uint8_t byte = (uint8_t)text[*pos + index];
        if ((byte & 0xc0u) != 0x80u) {
            return 0;
        }
        value = (value << 6u) | (uint32_t)(byte & 0x3fu);
    }
    *pos += need;
    *out_codepoint = value;
    return 1;
}

static char *te_symbol_from_byte(uint8_t byte) {
    char encoded[4];
    size_t len = 0;
    if (!te_utf8_encode(te_byte_to_unicode(byte), encoded, &len)) {
        return NULL;
    }
    char *symbol = (char *)malloc(len + 1u);
    if (symbol == NULL) {
        return NULL;
    }
    memcpy(symbol, encoded, len);
    symbol[len] = '\0';
    return symbol;
}

static char *te_concat2(const char *left, const char *right) {
    const size_t left_len = strlen(left);
    const size_t right_len = strlen(right);
    if (left_len > SIZE_MAX - right_len - 1u) {
        return NULL;
    }
    char *out = (char *)malloc(left_len + right_len + 1u);
    if (out == NULL) {
        return NULL;
    }
    memcpy(out, left, left_len);
    memcpy(out + left_len, right, right_len);
    out[left_len + right_len] = '\0';
    return out;
}

static void te_free_symbols(char **symbols, size_t count) {
    if (symbols == NULL) {
        return;
    }
    for (size_t index = 0; index < count; ++index) {
        free(symbols[index]);
    }
    free(symbols);
}

static te_status te_token_vec_push(te_token_vec *vec, uint32_t token) {
    if (vec == NULL) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (vec->len == vec->cap) {
        const size_t new_cap = vec->cap == 0u ? 16u : vec->cap * 2u;
        if (new_cap < vec->cap || new_cap > SIZE_MAX / sizeof(vec->items[0])) {
            return TE_STATUS_UNSUPPORTED;
        }
        uint32_t *items = (uint32_t *)realloc(vec->items, new_cap * sizeof(vec->items[0]));
        if (items == NULL) {
            return TE_STATUS_OUT_OF_MEMORY;
        }
        vec->items = items;
        vec->cap = new_cap;
    }
    vec->items[vec->len++] = token;
    return TE_STATUS_OK;
}

static void te_token_vec_free(te_token_vec *vec) {
    if (vec == NULL) {
        return;
    }
    free(vec->items);
    vec->items = NULL;
    vec->len = 0;
    vec->cap = 0;
}

static int te_find_special_token(
    const te_model *model,
    const char *text,
    size_t pos,
    uint32_t *out_id,
    size_t *out_len
) {
    if (text[pos] != '<') {
        return 0;
    }
    size_t best_len = 0;
    uint32_t best_id = 0;
    for (uint64_t index = 0; index < model->token_count; ++index) {
        const te_token_entry *entry = &model->tokens[index];
        if (!entry->is_special || entry->text == NULL) {
            continue;
        }
        const size_t len = strlen(entry->text);
        if (len <= best_len) {
            continue;
        }
        if (strncmp(text + pos, entry->text, len) == 0) {
            best_len = len;
            best_id = entry->id;
        }
    }
    if (best_len == 0u) {
        return 0;
    }
    *out_id = best_id;
    *out_len = best_len;
    return 1;
}

static te_status te_bpe_segment(
    const te_model *model,
    const uint8_t *bytes,
    size_t len,
    te_token_vec *out
) {
    if (len == 0u) {
        return TE_STATUS_OK;
    }
    char **symbols = (char **)calloc(len, sizeof(symbols[0]));
    if (symbols == NULL) {
        return TE_STATUS_OUT_OF_MEMORY;
    }
    for (size_t index = 0; index < len; ++index) {
        symbols[index] = te_symbol_from_byte(bytes[index]);
        if (symbols[index] == NULL) {
            te_free_symbols(symbols, len);
            return TE_STATUS_OUT_OF_MEMORY;
        }
    }

    size_t count = len;
    while (count > 1u) {
        int found = 0;
        uint32_t best_rank = UINT32_MAX;
        size_t best_index = 0;
        for (size_t index = 0; index + 1u < count; ++index) {
            uint32_t rank = 0;
            int has_rank = 0;
            te_status status = te_string_map_get_pair(
                &model->merge_to_rank,
                symbols[index],
                symbols[index + 1u],
                &rank,
                &has_rank);
            if (status != TE_STATUS_OK) {
                te_free_symbols(symbols, count);
                return status;
            }
            if (has_rank && (!found || rank < best_rank)) {
                found = 1;
                best_rank = rank;
                best_index = index;
            }
        }
        if (!found) {
            break;
        }
        char *merged = te_concat2(symbols[best_index], symbols[best_index + 1u]);
        if (merged == NULL) {
            te_free_symbols(symbols, count);
            return TE_STATUS_OUT_OF_MEMORY;
        }
        free(symbols[best_index]);
        free(symbols[best_index + 1u]);
        symbols[best_index] = merged;
        for (size_t index = best_index + 1u; index + 1u < count; ++index) {
            symbols[index] = symbols[index + 1u];
        }
        symbols[count - 1u] = NULL;
        --count;
    }

    for (size_t index = 0; index < count; ++index) {
        uint32_t token_id = 0;
        if (!te_string_map_get(&model->token_to_id, symbols[index], &token_id)) {
            te_free_symbols(symbols, count);
            return TE_STATUS_RUNTIME_ERROR;
        }
        te_status status = te_token_vec_push(out, token_id);
        if (status != TE_STATUS_OK) {
            te_free_symbols(symbols, count);
            return status;
        }
    }

    te_free_symbols(symbols, count);
    return TE_STATUS_OK;
}
