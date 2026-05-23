#include "tinyengine.h"

#include <stdio.h>
#include <stdlib.h>

static void on_token(const char *text, uint32_t token_id, void *userdata) {
    (void)token_id;
    (void)userdata;
    fputs(text, stdout);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s /path/to/model.gguf prompt [max_tokens]\n", argv[0]);
        return 2;
    }

    uint32_t max_tokens = 16;
    if (argc == 4) {
        char *end = NULL;
        const unsigned long parsed = strtoul(argv[3], &end, 10);
        if (end == argv[3] || *end != '\0' || parsed > UINT32_MAX) {
            fprintf(stderr, "invalid max_tokens: %s\n", argv[3]);
            return 2;
        }
        max_tokens = (uint32_t)parsed;
    }

    te_runtime_options options = te_default_options();
    te_model *model = NULL;
    te_status status = te_model_load_gguf(argv[1], &options, &model);
    if (status != TE_STATUS_OK) {
        fprintf(stderr, "te_model_load_gguf failed: %s\n", te_strerror(status));
        return 1;
    }

    te_context *context = NULL;
    status = te_context_create(model, &options, &context);
    if (status != TE_STATUS_OK) {
        fprintf(stderr, "te_context_create failed: %s\n", te_strerror(status));
        te_model_free(model);
        return 1;
    }

    status = te_generate(context, argv[2], max_tokens, on_token, NULL);
    if (status != TE_STATUS_OK) {
        fprintf(stderr, "te_generate failed: %s\n", te_strerror(status));
        te_context_free(context);
        te_model_free(model);
        return 1;
    }
    fputc('\n', stdout);

    te_context_free(context);
    te_model_free(model);
    return 0;
}
