CC ?= cc
BUILD_DIR ?= build
SRC_DIR := src
INCLUDE_DIR := include

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
SHARED_EXT := dylib
SHARED_FLAGS := -dynamiclib -install_name @rpath/libtinyengine.dylib
RPATH_FLAGS := -Wl,-rpath,@loader_path
METAL_OBJECTS := $(BUILD_DIR)/metal_backend.o
METAL_LDFLAGS := -framework Foundation -framework Metal
else
$(error TinyEngine supports Apple platforms only (Darwin); detected '$(UNAME_S)')
endif

CFLAGS ?= -std=c11 -O3 -Wall -Wextra -Werror -pedantic -fPIC
CXXFLAGS ?= -std=c++11 -O3 -Wall -Wextra -Werror -fPIC -fobjc-arc
CPPFLAGS ?= -I$(INCLUDE_DIR)

LIB := $(BUILD_DIR)/libtinyengine.$(SHARED_EXT)
SMOKE := $(BUILD_DIR)/te_smoke
GENERATE := $(BUILD_DIR)/te_generate
KERNEL_TEST := $(BUILD_DIR)/te_kernel_tests
GENERATED_METAL_GUARD_TESTS := $(BUILD_DIR)/generated_metal_guard_tests.inc
OBJECTS := $(BUILD_DIR)/tinyengine.o $(BUILD_DIR)/gguf.o $(BUILD_DIR)/ops.o $(BUILD_DIR)/tokenizer.o $(BUILD_DIR)/qwen.o $(METAL_OBJECTS)
LDLIBS ?= -lm
GGUF ?= ../../models/qwen2.5-coder-3b-instruct-q4_0-te.gguf
LLAMA_BIN ?= ../../tools/llama.cpp-b9219/current/llama-completion
ORACLE_PROMPT ?= Rispondi in italiano con tre parole: cosa sei?
ORACLE_C_TOKENS ?= 4
ORACLE_LLAMA_PREDICT ?= 4
ORACLE_SEED ?= 42
ORACLE_GPU_LAYERS ?= 999
ORACLE_CTX_SIZE ?= 512
BENCHMARK_RUNS ?= 3
BENCHMARK_OUT ?= ../benchmarks/c-qwen2.5-coder-3b-q4_0-te-vs-llama.json
BENCHMARK_ALLOW_MISMATCH ?= 0
TINYENGINE_WORKLOAD ?= auto
LONG_BENCHMARK_PROMPT ?= Contesto: questo benchmark contiene una richiesta volutamente lunga per misurare la fase di prefill su input piu estesi. Ripeti mentalmente che devi ignorare il contesto descrittivo e rispondere solo alla domanda finale. Dettagli: agente, runtime, kernel Metal, memoria KV, prompt lungo, ottimizzazione, confronto con llama.cpp, misurazione stabile, batch prefill, decode su GPU, sincronizzazioni ridotte, cache persistente, throughput, latenza, accuratezza, oracle, output breve. Ancora contesto non rilevante: il sistema deve mantenere lo stesso comportamento deterministico anche quando il prompt cresce e contiene molte parole prima della domanda finale. Domanda finale: Rispondi in italiano con tre parole: cosa sei?
LONG_BENCHMARK_CTX_SIZE ?= 1024
LONG_BENCHMARK_C_TOKENS ?= 4
LONG_BENCHMARK_LLAMA_PREDICT ?= 4
LONG_BENCHMARK_ALLOW_MISMATCH ?= 0
LONG_BENCHMARK_OUT ?= ../benchmarks/c-qwen2.5-coder-3b-q4_0-te-long-prompt-1024ctx-vs-llama.json
AUTOTUNE_RUNS ?= 3
AUTOTUNE_PROFILES ?= default,matmul-off,gateup-matmul-off,kv-matmul-on,kv-matmul-off,q8-llama-off,flash-attn-off,flash-longtile-off,ffn-half-off,ffn-pair-swiglu-off,ffn-down-add-off,ffn-down-add-on,ffn-gate-half-on,argmax-cpu,prefill-layer,decode-layer,qkv-off
AUTOTUNE_WORKLOADS ?= short,long
AUTOTUNE_OUT_DIR ?= ../benchmarks/autotune
AUTOTUNE_MIN_IMPROVEMENT ?= 0.02
AUTOTUNE_COOLDOWN_SEC ?= 0

.PHONY: all runtime tools clean smoke test check oracle benchmark benchmark-long autotune autotune-long autotune-quick

all: runtime tools

runtime: $(LIB)

tools: $(SMOKE) $(GENERATE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(SRC_DIR)/tinyengine_internal.h $(INCLUDE_DIR)/tinyengine.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.mm $(SRC_DIR)/metal_backend.h $(INCLUDE_DIR)/tinyengine.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/metal_backend.o: $(SRC_DIR)/metal/metal_kernels.metal.inc $(SRC_DIR)/metal/metal_backend_runtime.mm.inc $(SRC_DIR)/metal/metal_backend_layers.mm.inc

$(GENERATED_METAL_GUARD_TESTS): tests/generate_metal_guard_tests.py $(SRC_DIR)/metal_backend.h | $(BUILD_DIR)
	python3 $< > $@

$(LIB): $(OBJECTS)
	$(CXX) $(SHARED_FLAGS) $^ $(LDLIBS) $(METAL_LDFLAGS) -o $@

$(SMOKE): examples/smoke.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -L$(BUILD_DIR) -ltinyengine $(RPATH_FLAGS) $(LDLIBS) -o $@

$(GENERATE): examples/generate.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -L$(BUILD_DIR) -ltinyengine $(RPATH_FLAGS) $(LDLIBS) -o $@

$(KERNEL_TEST): tests/kernel_ops.c $(LIB) $(GENERATED_METAL_GUARD_TESTS)
	$(CC) $(CPPFLAGS) -I$(BUILD_DIR) $(CFLAGS) $< -L$(BUILD_DIR) -ltinyengine $(RPATH_FLAGS) $(LDLIBS) -o $@

smoke: $(SMOKE)
	$(SMOKE)

test check: $(KERNEL_TEST)
	$(KERNEL_TEST)

oracle: $(GENERATE)
	PYTHONPATH=../python TINYENGINE_LIBRARY=$(CURDIR)/$(LIB) python3 tests/oracle_compare.py \
		--tiny-bin $(CURDIR)/$(GENERATE) \
		--llama-bin $(abspath $(LLAMA_BIN)) \
		--gguf $(abspath $(GGUF)) \
		--prompt "$(ORACLE_PROMPT)" \
		--c-max-tokens $(ORACLE_C_TOKENS) \
		--llama-predict $(ORACLE_LLAMA_PREDICT) \
		--seed $(ORACLE_SEED) \
		--gpu-layers $(ORACLE_GPU_LAYERS) \
		--ctx-size $(ORACLE_CTX_SIZE)

benchmark: $(LIB)
	mkdir -p $(dir $(abspath $(BENCHMARK_OUT)))
	PYTHONPATH=../python TINYENGINE_LIBRARY=$(CURDIR)/$(LIB) TINYENGINE_WORKLOAD=$(TINYENGINE_WORKLOAD) python3 tests/benchmark_compare.py \
		--llama-bin $(abspath $(LLAMA_BIN)) \
		--gguf $(abspath $(GGUF)) \
		--prompt "$(ORACLE_PROMPT)" \
		--c-max-tokens $(ORACLE_C_TOKENS) \
		--llama-predict $(ORACLE_LLAMA_PREDICT) \
		--seed $(ORACLE_SEED) \
		--gpu-layers $(ORACLE_GPU_LAYERS) \
		--ctx-size $(ORACLE_CTX_SIZE) \
		--runs $(BENCHMARK_RUNS) \
		$(if $(filter 1 true yes,$(BENCHMARK_ALLOW_MISMATCH)),--allow-mismatch,) \
		--out $(abspath $(BENCHMARK_OUT))

benchmark-long:
	$(MAKE) benchmark \
		ORACLE_PROMPT="$(LONG_BENCHMARK_PROMPT)" \
		ORACLE_C_TOKENS=$(LONG_BENCHMARK_C_TOKENS) \
		ORACLE_LLAMA_PREDICT=$(LONG_BENCHMARK_LLAMA_PREDICT) \
		ORACLE_CTX_SIZE=$(LONG_BENCHMARK_CTX_SIZE) \
		TINYENGINE_WORKLOAD=long \
		BENCHMARK_ALLOW_MISMATCH=$(LONG_BENCHMARK_ALLOW_MISMATCH) \
		BENCHMARK_OUT=$(LONG_BENCHMARK_OUT)

autotune: $(LIB)
	mkdir -p $(abspath $(AUTOTUNE_OUT_DIR))
	PYTHONPATH=../python TINYENGINE_LIBRARY=$(CURDIR)/$(LIB) python3 tests/autotune_microkernels.py \
		--llama-bin $(abspath $(LLAMA_BIN)) \
		--gguf $(abspath $(GGUF)) \
		--runs $(AUTOTUNE_RUNS) \
		--profiles "$(AUTOTUNE_PROFILES)" \
		--workloads "$(AUTOTUNE_WORKLOADS)" \
		--out-dir $(abspath $(AUTOTUNE_OUT_DIR)) \
		--min-improvement $(AUTOTUNE_MIN_IMPROVEMENT) \
		--cooldown-sec $(AUTOTUNE_COOLDOWN_SEC) \
		--seed $(ORACLE_SEED) \
		--gpu-layers $(ORACLE_GPU_LAYERS) \
		--short-prompt "$(ORACLE_PROMPT)" \
		--short-ctx-size $(ORACLE_CTX_SIZE) \
		--short-c-max-tokens $(ORACLE_C_TOKENS) \
		--short-llama-predict $(ORACLE_LLAMA_PREDICT) \
		--long-prompt "$(LONG_BENCHMARK_PROMPT)" \
		--long-ctx-size $(LONG_BENCHMARK_CTX_SIZE) \
		--long-c-max-tokens $(LONG_BENCHMARK_C_TOKENS) \
		--long-llama-predict $(LONG_BENCHMARK_LLAMA_PREDICT)

autotune-long:
	$(MAKE) autotune AUTOTUNE_WORKLOADS=long

autotune-quick:
	$(MAKE) autotune AUTOTUNE_RUNS=1 AUTOTUNE_WORKLOADS=short AUTOTUNE_PROFILES=default,kv-matmul-on,flash-attn-off,ffn-half-off,ffn-down-add-on,ffn-gate-half-on,matmul-off AUTOTUNE_MIN_IMPROVEMENT=0

clean:
	rm -rf $(BUILD_DIR)
