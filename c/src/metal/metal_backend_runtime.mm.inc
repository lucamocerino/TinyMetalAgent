// Included by metal_backend.mm. Contains Metal runtime state, profiling, env gates, and helpers.
static constexpr NSUInteger TE_METAL_Q4_ROW_TILE = 4;
static constexpr NSUInteger TE_METAL_Q4_LLAMA_ROW_TILE = 2;
static constexpr NSUInteger TE_METAL_Q4_LLAMA_THREADS = 64;
static constexpr NSUInteger TE_METAL_Q4_PAIR_ROW_TILE = 4;
static constexpr NSUInteger TE_METAL_Q4_BATCH_ROW_TILE = 1;
static constexpr NSUInteger TE_METAL_Q4_BATCH_TILE = 7;
static constexpr NSUInteger TE_METAL_Q4_BATCH_LLAMA_ROW_TILE = 8;
static constexpr NSUInteger TE_METAL_Q4_BATCH_LLAMA_TILE = 1;
static constexpr NSUInteger TE_METAL_Q4_MATMUL_ROW_TILE = 64;
static constexpr NSUInteger TE_METAL_Q4_MATMUL_BATCH_TILE = 32;
static constexpr NSUInteger TE_METAL_Q4_MATMUL_THREADS = 128;
static constexpr NSUInteger TE_METAL_Q4_MATMUL_SHMEM = 8192;
static constexpr NSUInteger TE_METAL_Q4_MATMUL_PAIR_SWIGLU_SHMEM = 16384;
static constexpr NSUInteger TE_METAL_Q4_BATCH_THREADS = 128;
static constexpr NSUInteger TE_METAL_Q8_LLAMA_ROW_TILE = 2;
static constexpr NSUInteger TE_METAL_Q8_LLAMA_THREADS = 128;
static constexpr NSUInteger TE_METAL_Q8_LLAMA_SHMEM = TE_METAL_Q8_LLAMA_ROW_TILE * 32 * sizeof(float);
static constexpr NSUInteger TE_METAL_ARGMAX_THREADS = 256;
static constexpr NSUInteger TE_METAL_VECTOR_THREADS = 256;
static constexpr NSUInteger TE_METAL_ATTENTION_THREADS = 128;
static constexpr NSUInteger TE_METAL_FLASH_ATTENTION_THREADS = 128;
static constexpr NSUInteger TE_METAL_FLASH_ATTENTION_GQA_MAX_REPEAT = 8;
static constexpr NSUInteger TE_METAL_RMSNORM_THREADS = 128;

@interface TEMetalRuntime : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> q4Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q4LlamaPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q4PairPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q4PairLlamaPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q4BatchPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q4BatchLlamaPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q4MatmulPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q4MatmulPairSwigluHalfPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q4MatmulSwigluPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q4MatmulHalfInputPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q4MatmulHalfInputAddPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q4MatmulSwigluHalfPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q4MatmulStoreHalfPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q4MatmulSwigluGateHalfPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q4BatchPairPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q4BatchPairLlamaPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q8Pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> q8LlamaPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionQkMmaPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionTileMmaPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionStreamMmaPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> argmaxBlocksPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> argmaxFinishPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> swigluPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> addPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> rmsnormPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> vectorCopyPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> touchPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> qkvBiasRopeCachePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> qkvBiasRopeCacheBatchPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> qkvBiasRopeCacheBatchHalfPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> qkvBiasRopeCacheBatchTransposePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionDecodePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionFlashPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionFlashLongPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionFlashLongTransposePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionFlashMmaTransposePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionFlashGqaPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> attentionFlashHalfPipeline;
@property(nonatomic, strong) id<MTLBuffer> mappingBuffer;
@property(nonatomic, strong) id<MTLBuffer> inputBuffer;
@property(nonatomic, strong) id<MTLBuffer> outputBuffer;
@property(nonatomic, strong) id<MTLBuffer> output2Buffer;
@property(nonatomic, strong) id<MTLBuffer> output3Buffer;
@property(nonatomic, strong) id<MTLBuffer> scratchBuffer;
@property(nonatomic, strong) id<MTLBuffer> normBuffer;
@property(nonatomic, strong) id<MTLBuffer> dimsBuffer;
@property(nonatomic, strong) id<MTLBuffer> dims2Buffer;
@property(nonatomic, strong) id<MTLBuffer> scalarBuffer;
@property(nonatomic, strong) id<MTLBuffer> weightBuffer;
@property(nonatomic, strong) id<MTLBuffer> weight2Buffer;
@property(nonatomic, strong) id<MTLBuffer> biasBuffer;
@property(nonatomic, strong) id<MTLBuffer> bias2Buffer;
@property(nonatomic, strong) id<MTLBuffer> bias3Buffer;
@property(nonatomic, strong) id<MTLBuffer> ropeCosBuffer;
@property(nonatomic, strong) id<MTLBuffer> ropeSinBuffer;
@property(nonatomic, strong) id<MTLBuffer> keyCacheBuffer;
@property(nonatomic, strong) id<MTLBuffer> valueCacheBuffer;
@property(nonatomic, strong) id<MTLBuffer> keyCacheHalfBuffer;
@property(nonatomic, strong) id<MTLBuffer> valueCacheHalfBuffer;
@property(nonatomic, strong) id<MTLBuffer> keyCacheTransposeBuffer;
@property(nonatomic, strong) id<MTLBuffer> valueCacheTransposeBuffer;
@property(nonatomic, assign) const void *mappingPtr;
@property(nonatomic, assign) size_t mappingLen;
@property(nonatomic, assign) const void *warmedMappingPtr;
@property(nonatomic, assign) size_t warmedMappingLen;
@property(nonatomic, assign) size_t inputCapacity;
@property(nonatomic, assign) size_t outputCapacity;
@property(nonatomic, assign) size_t output2Capacity;
@property(nonatomic, assign) size_t output3Capacity;
@property(nonatomic, assign) size_t scratchCapacity;
@property(nonatomic, assign) size_t normCapacity;
@property(nonatomic, assign) size_t weightCapacity;
@property(nonatomic, assign) size_t weight2Capacity;
@property(nonatomic, assign) size_t biasCapacity;
@property(nonatomic, assign) size_t bias2Capacity;
@property(nonatomic, assign) size_t bias3Capacity;
@property(nonatomic, assign) size_t ropeCosCapacity;
@property(nonatomic, assign) size_t ropeSinCapacity;
@property(nonatomic, assign) size_t keyCacheCapacity;
@property(nonatomic, assign) size_t valueCacheCapacity;
@property(nonatomic, assign) size_t keyCacheHalfCapacity;
@property(nonatomic, assign) size_t valueCacheHalfCapacity;
@property(nonatomic, assign) size_t keyCacheTransposeCapacity;
@property(nonatomic, assign) size_t valueCacheTransposeCapacity;
@property(nonatomic, assign) BOOL residentKvCacheValid;
@property(nonatomic, assign) const void *residentKvMappingPtr;
@property(nonatomic, assign) size_t residentKvLayers;
@property(nonatomic, assign) size_t residentKvContextTokens;
@property(nonatomic, assign) size_t residentKvKv;
@property(nonatomic, assign) size_t residentKvFilledTokens;
@property(nonatomic, assign) BOOL flashAttentionEnabled;
@property(nonatomic, assign) BOOL q4FfnHalfEnabled;
@property(nonatomic, assign) BOOL q4FfnDownAddEnabled;
@property(nonatomic, assign) BOOL q4FfnGateHalfEnabled;
@end

@implementation TEMetalRuntime
@end

static std::mutex TE_METAL_MUTEX;
static TEMetalRuntime *TE_METAL_RUNTIME = nil;

enum TEMetalProfileBucketId {
    TE_METAL_PROFILE_MATVEC_BATCH,
    TE_METAL_PROFILE_MATVEC_ARGMAX,
    TE_METAL_PROFILE_QKV,
    TE_METAL_PROFILE_QKV_BATCH,
    TE_METAL_PROFILE_MLP,
    TE_METAL_PROFILE_MLP_BATCH,
    TE_METAL_PROFILE_POST_ATTN_MLP,
    TE_METAL_PROFILE_POST_ATTN_MLP_BATCH,
    TE_METAL_PROFILE_MATVEC2,
    TE_METAL_PROFILE_MATVEC,
    TE_METAL_PROFILE_DECODE_LAYER,
    TE_METAL_PROFILE_DECODE_ALL_LAYERS,
    TE_METAL_PROFILE_DECODE_PROJECT,
    TE_METAL_PROFILE_PREFILL_LAYER,
    TE_METAL_PROFILE_PREFILL_ALL_LAYERS,
    TE_METAL_PROFILE_COUNT
};

struct TEMetalProfileBucket {
    const char *name;
    unsigned long long calls;
    double cpu_ms;
    double gpu_ms;
};

static TEMetalProfileBucket TE_METAL_PROFILE[TE_METAL_PROFILE_COUNT] = {
    {"matvec_batch", 0, 0.0, 0.0},
    {"matvec_argmax", 0, 0.0, 0.0},
    {"qkv", 0, 0.0, 0.0},
    {"qkv_batch", 0, 0.0, 0.0},
    {"mlp", 0, 0.0, 0.0},
    {"mlp_batch", 0, 0.0, 0.0},
    {"post_attn_mlp", 0, 0.0, 0.0},
    {"post_attn_mlp_batch", 0, 0.0, 0.0},
    {"matvec2", 0, 0.0, 0.0},
    {"matvec", 0, 0.0, 0.0},
    {"decode_layer", 0, 0.0, 0.0},
    {"decode_all_layers", 0, 0.0, 0.0},
    {"decode_project", 0, 0.0, 0.0},
    {"prefill_layer", 0, 0.0, 0.0},
    {"prefill_all_layers", 0, 0.0, 0.0},
};

static bool TE_METAL_PROFILE_REGISTERED = false;

static bool te_metal_profile_enabled(void) {
    const char *value = getenv("TINYENGINE_METAL_PROFILE");
    return value != nullptr && strcmp(value, "0") != 0;
}

static bool te_metal_prefill_stage_profile_enabled(void) {
    const char *value = getenv("TINYENGINE_METAL_PREFILL_STAGE_PROFILE");
    return value != nullptr && strcmp(value, "0") != 0;
}

static bool te_metal_flash_attn_requested(void) {
    const char *value = getenv("TINYENGINE_FLASH_ATTN");
    return value == nullptr || strcmp(value, "0") != 0;
}

static double te_metal_now_ms(void) {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static void te_metal_profile_print(void) {
    if (!te_metal_profile_enabled()) {
        return;
    }
    for (size_t index = 0; index < TE_METAL_PROFILE_COUNT; ++index) {
        const TEMetalProfileBucket &bucket = TE_METAL_PROFILE[index];
        if (bucket.calls == 0) {
            continue;
        }
        fprintf(
            stderr,
            "tinyengine_metal_profile: name=%s calls=%llu cpu_ms=%.3f gpu_ms=%.3f avg_cpu_ms=%.3f avg_gpu_ms=%.3f\n",
            bucket.name,
            bucket.calls,
            bucket.cpu_ms,
            bucket.gpu_ms,
            bucket.cpu_ms / (double)bucket.calls,
            bucket.gpu_ms / (double)bucket.calls);
    }
}

static void te_metal_profile_register(void) {
    if (!TE_METAL_PROFILE_REGISTERED && te_metal_profile_enabled()) {
        atexit(te_metal_profile_print);
        TE_METAL_PROFILE_REGISTERED = true;
    }
}

static void te_metal_profile_record(
    TEMetalProfileBucketId bucket_id,
    id<MTLCommandBuffer> commandBuffer,
    double cpu_ms
) {
    if (!te_metal_profile_enabled() || bucket_id < 0 || bucket_id >= TE_METAL_PROFILE_COUNT) {
        return;
    }
    double gpu_ms = 0.0;
    const CFTimeInterval gpu_start = commandBuffer.GPUStartTime;
    const CFTimeInterval gpu_end = commandBuffer.GPUEndTime;
    if (gpu_end > gpu_start) {
        gpu_ms = (gpu_end - gpu_start) * 1000.0;
    }
    TEMetalProfileBucket &bucket = TE_METAL_PROFILE[bucket_id];
    bucket.calls += 1;
    bucket.cpu_ms += cpu_ms;
    bucket.gpu_ms += gpu_ms;
}

static bool te_metal_enabled(void) {
    const char *value = getenv("TINYENGINE_METAL_MATVEC");
    return value == nullptr || strcmp(value, "0") != 0;
}

static bool te_metal_q4_llama_requested(void) {
    const char *value = getenv("TINYENGINE_Q4_LLAMA");
    return value == nullptr || strcmp(value, "0") != 0;
}

static bool te_metal_use_llama_q4(size_t cols) {
    return te_metal_q4_llama_requested() && (cols % 32u) == 0u;
}

static bool te_metal_q4_pair_llama_requested(void) {
    const char *value = getenv("TINYENGINE_Q4_PAIR_LLAMA");
    return value == nullptr || strcmp(value, "0") != 0;
}

static bool te_metal_use_llama_q4_pair(size_t cols) {
    return te_metal_q4_pair_llama_requested() && (cols % 32u) == 0u;
}

static bool te_metal_q4_batch_llama_requested(void) {
    const char *value = getenv("TINYENGINE_Q4_BATCH_LLAMA");
    return value == nullptr || strcmp(value, "0") != 0;
}

static bool te_metal_use_llama_q4_batch(size_t cols) {
    return te_metal_q4_batch_llama_requested() && (cols % 32u) == 0u;
}

static bool te_metal_q4_batch_pair_llama_requested(void) {
    const char *value = getenv("TINYENGINE_Q4_BATCH_PAIR_LLAMA");
    return value == nullptr || strcmp(value, "0") != 0;
}

static bool te_metal_use_llama_q4_batch_pair(size_t cols) {
    return te_metal_q4_batch_pair_llama_requested() && (cols % 32u) == 0u;
}

static bool te_metal_q4_matmul_requested(void) {
    const char *value = getenv("TINYENGINE_Q4_MATMUL");
    return value == nullptr || strcmp(value, "0") != 0;
}

static bool te_metal_q4_matmul_pair_requested(void) {
    const char *value = getenv("TINYENGINE_Q4_MATMUL_PAIR");
    return value == nullptr || strcmp(value, "0") != 0;
}

static bool te_metal_q4_matmul_gateup_requested(void) {
    const char *value = getenv("TINYENGINE_Q4_MATMUL_GATEUP");
    return value == nullptr || strcmp(value, "0") != 0;
}

enum TEMetalWorkloadKind {
    TE_METAL_WORKLOAD_AUTO,
    TE_METAL_WORKLOAD_SHORT,
    TE_METAL_WORKLOAD_LONG,
    TE_METAL_WORKLOAD_DECODE
};

static TEMetalWorkloadKind te_metal_workload_kind(void) {
    const char *value = getenv("TINYENGINE_WORKLOAD");
    if (value == nullptr || strcmp(value, "auto") == 0 || value[0] == '\0') {
        return TE_METAL_WORKLOAD_AUTO;
    }
    if (strcmp(value, "short") == 0) {
        return TE_METAL_WORKLOAD_SHORT;
    }
    if (strcmp(value, "long") == 0) {
        return TE_METAL_WORKLOAD_LONG;
    }
    if (strcmp(value, "decode") == 0) {
        return TE_METAL_WORKLOAD_DECODE;
    }
    return TE_METAL_WORKLOAD_AUTO;
}

static bool te_metal_long_prefill_workload(size_t batch) {
    switch (te_metal_workload_kind()) {
        case TE_METAL_WORKLOAD_LONG:
            return true;
        case TE_METAL_WORKLOAD_SHORT:
        case TE_METAL_WORKLOAD_DECODE:
            return false;
        case TE_METAL_WORKLOAD_AUTO:
        default:
            return batch >= 64u;
    }
}

static bool te_metal_q4_matmul_kv_requested(size_t batch) {
    const char *value = getenv("TINYENGINE_Q4_MATMUL_KV");
    if (value != nullptr) {
        return strcmp(value, "0") != 0;
    }
    return te_metal_long_prefill_workload(batch);
}

static bool te_metal_q4_ffn_pair_swiglu_requested(size_t batch) {
    const char *value = getenv("TINYENGINE_Q4_FFN_PAIR_SWIGLU");
    if (value != nullptr) {
        return strcmp(value, "0") != 0;
    }
    return te_metal_long_prefill_workload(batch);
}

static bool te_metal_q4_ffn_half_requested(void) {
    const char *value = getenv("TINYENGINE_Q4_FFN_HALF");
    return value == nullptr || strcmp(value, "0") != 0;
}

static bool te_metal_q4_ffn_down_add_requested(void) {
    const char *value = getenv("TINYENGINE_Q4_FFN_DOWN_ADD");
    return value == nullptr || strcmp(value, "0") != 0;
}

static bool te_metal_q4_ffn_gate_half_requested(void) {
    const char *value = getenv("TINYENGINE_Q4_FFN_GATE_HALF");
    return value != nullptr && strcmp(value, "0") != 0;
}

static bool te_metal_use_q4_matmul(size_t cols, size_t batch) {
    return te_metal_q4_matmul_requested() && batch > 1u && (cols % 32u) == 0u;
}

static bool te_metal_use_q4_matmul_pair(size_t cols, size_t batch) {
    return te_metal_q4_matmul_pair_requested() && te_metal_use_q4_matmul(cols, batch);
}

static bool te_metal_q8_llama_requested(void) {
    const char *value = getenv("TINYENGINE_Q8_LLAMA");
    return value == nullptr || strcmp(value, "0") != 0;
}

static bool te_metal_use_llama_q8(size_t cols) {
    return te_metal_q8_llama_requested() && (cols % 32u) == 0u;
}

static bool te_metal_use_flash_attention(TEMetalRuntime *runtime, size_t heads, size_t kv_heads, size_t head_dim) {
    return runtime.flashAttentionEnabled && runtime.attentionFlashPipeline != nil &&
        head_dim == 64u && kv_heads != 0u && heads >= kv_heads && (heads % kv_heads) == 0u;
}

static bool te_metal_flash_attention_half_kv_requested(void) {
    const char *value = getenv("TINYENGINE_FLASH_ATTN_HALF_KV");
    return value != nullptr && strcmp(value, "0") != 0;
}

static bool te_metal_flash_attention_gqa_requested(void) {
    const char *value = getenv("TINYENGINE_FLASH_ATTN_GQA");
    return value != nullptr && strcmp(value, "0") != 0;
}

static bool te_metal_resident_kv_requested(size_t tokens) {
    const char *value = getenv("TINYENGINE_RESIDENT_KV");
    if (value != nullptr) {
        return strcmp(value, "0") != 0;
    }
    return tokens >= 512u;
}

static bool te_metal_transposed_prefill_kv_requested(size_t batch) {
    const char *value = getenv("TINYENGINE_TRANSPOSED_KV");
    if (value != nullptr) {
        return strcmp(value, "0") != 0;
    }
    return batch >= 512u;
}

static bool te_metal_mma_prefill_attention_requested(size_t batch) {
    const char *value = getenv("TINYENGINE_MMA_PREFILL_ATTN");
    if (value != nullptr) {
        return strcmp(value, "0") != 0;
    }
    return batch >= 512u;
}

static bool te_metal_flash_attention_long_tile_requested(size_t batch) {
    const char *value = getenv("TINYENGINE_FLASH_ATTN_LONG_TILE");
    if (value != nullptr) {
        return strcmp(value, "0") != 0;
    }
    return te_metal_workload_kind() == TE_METAL_WORKLOAD_LONG && batch >= 512u;
}

static bool te_metal_use_flash_attention_long_tile(
    TEMetalRuntime *runtime,
    size_t batch,
    size_t heads,
    size_t kv_heads,
    size_t head_dim
) {
    return te_metal_flash_attention_long_tile_requested(batch) &&
        runtime.attentionFlashLongPipeline != nil && head_dim == 64u &&
        kv_heads != 0u && heads >= kv_heads && (heads % kv_heads) == 0u;
}

static bool te_metal_use_flash_attention_gqa(
    TEMetalRuntime *runtime,
    size_t batch,
    size_t heads,
    size_t kv_heads,
    size_t head_dim
) {
    if (!te_metal_flash_attention_gqa_requested() || runtime.attentionFlashGqaPipeline == nil ||
        head_dim != 64u || kv_heads == 0u || heads < kv_heads || (heads % kv_heads) != 0u ||
        batch < 64u) {
        return false;
    }
    const size_t kv_repeat = heads / kv_heads;
    return kv_repeat > 1u && kv_repeat <= TE_METAL_FLASH_ATTENTION_GQA_MAX_REPEAT;
}

static bool te_metal_use_q4_ffn_half(TEMetalRuntime *runtime, bool useGateMatmul, bool useDownMatmul) {
    return useGateMatmul && useDownMatmul && runtime.q4FfnHalfEnabled &&
        runtime.q4MatmulHalfInputPipeline != nil && runtime.q4MatmulSwigluHalfPipeline != nil;
}

static bool te_metal_use_q4_ffn_pair_swiglu(TEMetalRuntime *runtime, bool useGateMatmul, bool useFfnHalf, size_t batch) {
    return useGateMatmul && useFfnHalf && te_metal_q4_ffn_pair_swiglu_requested(batch) &&
        runtime.q4MatmulPairSwigluHalfPipeline != nil;
}

static bool te_metal_use_q4_ffn_down_add(TEMetalRuntime *runtime, bool useFfnHalf, size_t batch) {
    const char *value = getenv("TINYENGINE_Q4_FFN_DOWN_ADD");
    const bool selected = value != nullptr ? strcmp(value, "0") != 0 : te_metal_long_prefill_workload(batch);
    return useFfnHalf && selected && runtime.q4FfnDownAddEnabled && runtime.q4MatmulHalfInputAddPipeline != nil;
}

static bool te_metal_use_q4_ffn_gate_half(TEMetalRuntime *runtime, bool useFfnHalf, size_t batch) {
    (void)batch;
    const char *value = getenv("TINYENGINE_Q4_FFN_GATE_HALF");
    const bool selected = value != nullptr && strcmp(value, "0") != 0;
    return useFfnHalf && selected && runtime.q4FfnGateHalfEnabled &&
        runtime.q4MatmulStoreHalfPipeline != nil && runtime.q4MatmulSwigluGateHalfPipeline != nil;
}

static te_status te_metal_init_locked(void) {
    te_metal_profile_register();
    if (TE_METAL_RUNTIME != nil) {
        return TE_STATUS_OK;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return TE_STATUS_UNSUPPORTED;
        }

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:te_metal_source() options:nil error:&error];
        if (library == nil) {
            if (error != nil) {
                fprintf(stderr, "tinyengine_metal_error: %s\n", [[error localizedDescription] UTF8String]);
            }
            return TE_STATUS_RUNTIME_ERROR;
        }
        const bool q4LlamaRequested = te_metal_q4_llama_requested();
        const bool q4PairLlamaRequested = te_metal_q4_pair_llama_requested();
        const bool q4BatchLlamaRequested = te_metal_q4_batch_llama_requested();
        const bool q4BatchPairLlamaRequested = te_metal_q4_batch_pair_llama_requested();
        const bool q4MatmulRequested = te_metal_q4_matmul_requested();
        const bool q4FfnHalfRequested = q4MatmulRequested && te_metal_q4_ffn_half_requested();
        const bool q4FfnDownAddRequested = q4FfnHalfRequested && te_metal_q4_ffn_down_add_requested();
        const bool q4FfnGateHalfRequested = q4FfnHalfRequested && te_metal_q4_ffn_gate_half_requested();
        const bool q8LlamaRequested = te_metal_q8_llama_requested();
        const bool flashAttnRequested = te_metal_flash_attn_requested();
        id<MTLFunction> q4 = [library newFunctionWithName:@"te_q4_0_matvec_f32"];
        id<MTLFunction> q4Llama = q4LlamaRequested
            ? [library newFunctionWithName:@"te_q4_0_matvec_llama_f32"]
            : nil;
        id<MTLFunction> q4Pair = [library newFunctionWithName:@"te_q4_0_matvec_pair_f32"];
        id<MTLFunction> q4PairLlama = q4PairLlamaRequested
            ? [library newFunctionWithName:@"te_q4_0_matvec_pair_llama_f32"]
            : nil;
        id<MTLFunction> q4Batch = [library newFunctionWithName:@"te_q4_0_matvec_batch_f32"];
        id<MTLFunction> q4BatchLlama = q4BatchLlamaRequested
            ? [library newFunctionWithName:@"te_q4_0_matvec_batch_llama_f32"]
            : nil;
        id<MTLFunction> q4Matmul = q4MatmulRequested
            ? [library newFunctionWithName:@"te_q4_0_matmat_llama_f32"]
            : nil;
        id<MTLFunction> q4MatmulPairSwigluHalf = q4MatmulRequested
            ? [library newFunctionWithName:@"te_q4_0_matmat_pair_swiglu_llama_h_f32"]
            : nil;
        id<MTLFunction> q4MatmulSwiglu = q4MatmulRequested
            ? [library newFunctionWithName:@"te_q4_0_matmat_swiglu_llama_f32"]
            : nil;
        id<MTLFunction> q4MatmulHalfInput = q4FfnHalfRequested
            ? [library newFunctionWithName:@"te_q4_0_matmat_h_llama_f32"]
            : nil;
        id<MTLFunction> q4MatmulHalfInputAdd = q4FfnDownAddRequested
            ? [library newFunctionWithName:@"te_q4_0_matmat_h_add_llama_f32"]
            : nil;
        id<MTLFunction> q4MatmulSwigluHalf = q4FfnHalfRequested
            ? [library newFunctionWithName:@"te_q4_0_matmat_swiglu_llama_h_f32"]
            : nil;
        id<MTLFunction> q4MatmulStoreHalf = q4FfnGateHalfRequested
            ? [library newFunctionWithName:@"te_q4_0_matmat_store_h_llama_f32"]
            : nil;
        id<MTLFunction> q4MatmulSwigluGateHalf = q4FfnGateHalfRequested
            ? [library newFunctionWithName:@"te_q4_0_matmat_swiglu_gate_h_llama_h_f32"]
            : nil;
        id<MTLFunction> q4BatchPair = [library newFunctionWithName:@"te_q4_0_matvec_batch_pair_f32"];
        id<MTLFunction> q4BatchPairLlama = q4BatchPairLlamaRequested
            ? [library newFunctionWithName:@"te_q4_0_matvec_batch_pair_llama_f32"]
            : nil;
        id<MTLFunction> q8 = [library newFunctionWithName:@"te_q8_0_matvec_f32"];
        id<MTLFunction> q8Llama = q8LlamaRequested
            ? [library newFunctionWithName:@"te_q8_0_matvec_llama_f32"]
            : nil;
        id<MTLFunction> attentionQkMma = [library newFunctionWithName:@"te_attention_qk_mma_f32_kernel"];
        id<MTLFunction> attentionTileMma = [library newFunctionWithName:@"te_attention_tile_mma_f32_kernel"];
        id<MTLFunction> attentionStreamMma = [library newFunctionWithName:@"te_attention_stream_mma_f32_kernel"];
        id<MTLFunction> argmaxBlocks = [library newFunctionWithName:@"te_argmax_blocks_f32"];
        id<MTLFunction> argmaxFinish = [library newFunctionWithName:@"te_argmax_finish_f32"];
        id<MTLFunction> swiglu = [library newFunctionWithName:@"te_swiglu_f32"];
        id<MTLFunction> add = [library newFunctionWithName:@"te_add_f32_kernel"];
        id<MTLFunction> rmsnorm = [library newFunctionWithName:@"te_rmsnorm_batch_f32"];
        id<MTLFunction> copy = [library newFunctionWithName:@"te_copy_f32_kernel"];
        id<MTLFunction> touch = [library newFunctionWithName:@"te_touch_u8_stride"];
        id<MTLFunction> qkvBiasRopeCache = [library newFunctionWithName:@"te_qkv_bias_rope_cache_f32"];
        id<MTLFunction> qkvBiasRopeCacheBatch = [library newFunctionWithName:@"te_qkv_bias_rope_cache_batch_f32"];
        id<MTLFunction> qkvBiasRopeCacheBatchTranspose =
            [library newFunctionWithName:@"te_qkv_bias_rope_cache_batch_t_f32"];
        const bool flashAttnHalfKvRequested = flashAttnRequested && te_metal_flash_attention_half_kv_requested();
        id<MTLFunction> qkvBiasRopeCacheBatchHalf = flashAttnHalfKvRequested
            ? [library newFunctionWithName:@"te_qkv_bias_rope_cache_batch_h_f32"]
            : nil;
        id<MTLFunction> attentionDecode = [library newFunctionWithName:@"te_attention_decode_f32_kernel"];
        id<MTLFunction> attentionFlash = flashAttnRequested
            ? [library newFunctionWithName:@"te_attention_flash_f32_kernel"]
            : nil;
        id<MTLFunction> attentionFlashLong = flashAttnRequested
            ? [library newFunctionWithName:@"te_attention_flash_t128_f32_kernel"]
            : nil;
        id<MTLFunction> attentionFlashLongTranspose = flashAttnRequested
            ? [library newFunctionWithName:@"te_attention_flash_t128_t_f32_kernel"]
            : nil;
        id<MTLFunction> attentionFlashMmaTranspose = flashAttnRequested
            ? [library newFunctionWithName:@"te_attention_flash_mma_t_f32_kernel"]
            : nil;
        const bool flashAttnGqaRequested = flashAttnRequested && te_metal_flash_attention_gqa_requested();
        id<MTLFunction> attentionFlashGqa = flashAttnGqaRequested
            ? [library newFunctionWithName:@"te_attention_flash_gqa_f32_kernel"]
            : nil;
        id<MTLFunction> attentionFlashHalf = flashAttnHalfKvRequested
            ? [library newFunctionWithName:@"te_attention_flash_h_f32_kernel"]
            : nil;
        if (q4 == nil || (q4LlamaRequested && q4Llama == nil) ||
            (q4PairLlamaRequested && q4PairLlama == nil) ||
            (q4BatchLlamaRequested && q4BatchLlama == nil) ||
            (q4BatchPairLlamaRequested && q4BatchPairLlama == nil) ||
            (q4MatmulRequested && (q4Matmul == nil || q4MatmulPairSwigluHalf == nil || q4MatmulSwiglu == nil)) ||
            (q4FfnHalfRequested && (q4MatmulHalfInput == nil || q4MatmulSwigluHalf == nil)) ||
            (q4FfnDownAddRequested && q4MatmulHalfInputAdd == nil) ||
            (q4FfnGateHalfRequested && (q4MatmulStoreHalf == nil || q4MatmulSwigluGateHalf == nil)) ||
            q4Pair == nil || q4Batch == nil || q4BatchPair == nil || q8 == nil ||
            attentionQkMma == nil || attentionTileMma == nil || attentionStreamMma == nil ||
            (q8LlamaRequested && q8Llama == nil) ||
            argmaxBlocks == nil || argmaxFinish == nil ||
            swiglu == nil || add == nil || rmsnorm == nil ||
            copy == nil || qkvBiasRopeCache == nil || qkvBiasRopeCacheBatch == nil ||
            qkvBiasRopeCacheBatchTranspose == nil ||
            (flashAttnHalfKvRequested && qkvBiasRopeCacheBatchHalf == nil) ||
            attentionDecode == nil || (flashAttnRequested && attentionFlash == nil) ||
            (flashAttnRequested && attentionFlashLong == nil) ||
            (flashAttnRequested && attentionFlashLongTranspose == nil) ||
            (flashAttnRequested && attentionFlashMmaTranspose == nil) ||
            (flashAttnGqaRequested && attentionFlashGqa == nil) ||
            (flashAttnHalfKvRequested && attentionFlashHalf == nil)) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        id<MTLComputePipelineState> q4Pipeline = [device newComputePipelineStateWithFunction:q4 error:&error];
        id<MTLComputePipelineState> q4LlamaPipeline = q4LlamaRequested
            ? [device newComputePipelineStateWithFunction:q4Llama error:&error]
            : nil;
        id<MTLComputePipelineState> q4PairPipeline = [device newComputePipelineStateWithFunction:q4Pair error:&error];
        id<MTLComputePipelineState> q4PairLlamaPipeline = q4PairLlamaRequested
            ? [device newComputePipelineStateWithFunction:q4PairLlama error:&error]
            : nil;
        id<MTLComputePipelineState> q4BatchPipeline = [device newComputePipelineStateWithFunction:q4Batch error:&error];
        id<MTLComputePipelineState> q4BatchLlamaPipeline = q4BatchLlamaRequested
            ? [device newComputePipelineStateWithFunction:q4BatchLlama error:&error]
            : nil;
        id<MTLComputePipelineState> q4MatmulPipeline = q4MatmulRequested
            ? [device newComputePipelineStateWithFunction:q4Matmul error:&error]
            : nil;
        id<MTLComputePipelineState> q4MatmulPairSwigluHalfPipeline = q4MatmulRequested
            ? [device newComputePipelineStateWithFunction:q4MatmulPairSwigluHalf error:&error]
            : nil;
        id<MTLComputePipelineState> q4MatmulSwigluPipeline = q4MatmulRequested
            ? [device newComputePipelineStateWithFunction:q4MatmulSwiglu error:&error]
            : nil;
        id<MTLComputePipelineState> q4MatmulHalfInputPipeline = q4FfnHalfRequested
            ? [device newComputePipelineStateWithFunction:q4MatmulHalfInput error:&error]
            : nil;
        id<MTLComputePipelineState> q4MatmulHalfInputAddPipeline = q4FfnDownAddRequested
            ? [device newComputePipelineStateWithFunction:q4MatmulHalfInputAdd error:&error]
            : nil;
        id<MTLComputePipelineState> q4MatmulSwigluHalfPipeline = q4FfnHalfRequested
            ? [device newComputePipelineStateWithFunction:q4MatmulSwigluHalf error:&error]
            : nil;
        id<MTLComputePipelineState> q4MatmulStoreHalfPipeline = q4FfnGateHalfRequested
            ? [device newComputePipelineStateWithFunction:q4MatmulStoreHalf error:&error]
            : nil;
        id<MTLComputePipelineState> q4MatmulSwigluGateHalfPipeline = q4FfnGateHalfRequested
            ? [device newComputePipelineStateWithFunction:q4MatmulSwigluGateHalf error:&error]
            : nil;
        id<MTLComputePipelineState> q4BatchPairPipeline =
            [device newComputePipelineStateWithFunction:q4BatchPair error:&error];
        id<MTLComputePipelineState> q4BatchPairLlamaPipeline = q4BatchPairLlamaRequested
            ? [device newComputePipelineStateWithFunction:q4BatchPairLlama error:&error]
            : nil;
        id<MTLComputePipelineState> q8Pipeline = [device newComputePipelineStateWithFunction:q8 error:&error];
        id<MTLComputePipelineState> q8LlamaPipeline = q8LlamaRequested
            ? [device newComputePipelineStateWithFunction:q8Llama error:&error]
            : nil;
        id<MTLComputePipelineState> attentionQkMmaPipeline =
            [device newComputePipelineStateWithFunction:attentionQkMma error:&error];
        id<MTLComputePipelineState> attentionTileMmaPipeline =
            [device newComputePipelineStateWithFunction:attentionTileMma error:&error];
        id<MTLComputePipelineState> attentionStreamMmaPipeline =
            [device newComputePipelineStateWithFunction:attentionStreamMma error:&error];
        id<MTLComputePipelineState> argmaxBlocksPipeline =
            [device newComputePipelineStateWithFunction:argmaxBlocks error:&error];
        id<MTLComputePipelineState> argmaxFinishPipeline =
            [device newComputePipelineStateWithFunction:argmaxFinish error:&error];
        id<MTLComputePipelineState> swigluPipeline = [device newComputePipelineStateWithFunction:swiglu error:&error];
        id<MTLComputePipelineState> addPipeline = [device newComputePipelineStateWithFunction:add error:&error];
        id<MTLComputePipelineState> rmsnormPipeline = [device newComputePipelineStateWithFunction:rmsnorm error:&error];
        id<MTLComputePipelineState> vectorCopyPipeline = [device newComputePipelineStateWithFunction:copy error:&error];
        id<MTLComputePipelineState> touchPipeline = touch != nil
            ? [device newComputePipelineStateWithFunction:touch error:&error]
            : nil;
        id<MTLComputePipelineState> qkvBiasRopeCachePipeline =
            [device newComputePipelineStateWithFunction:qkvBiasRopeCache error:&error];
        id<MTLComputePipelineState> qkvBiasRopeCacheBatchPipeline =
            [device newComputePipelineStateWithFunction:qkvBiasRopeCacheBatch error:&error];
        id<MTLComputePipelineState> qkvBiasRopeCacheBatchTransposePipeline =
            [device newComputePipelineStateWithFunction:qkvBiasRopeCacheBatchTranspose error:&error];
        id<MTLComputePipelineState> qkvBiasRopeCacheBatchHalfPipeline = flashAttnHalfKvRequested
            ? [device newComputePipelineStateWithFunction:qkvBiasRopeCacheBatchHalf error:&error]
            : nil;
        id<MTLComputePipelineState> attentionDecodePipeline =
            [device newComputePipelineStateWithFunction:attentionDecode error:&error];
        id<MTLComputePipelineState> attentionFlashPipeline = flashAttnRequested
            ? [device newComputePipelineStateWithFunction:attentionFlash error:&error]
            : nil;
        id<MTLComputePipelineState> attentionFlashLongPipeline = flashAttnRequested
            ? [device newComputePipelineStateWithFunction:attentionFlashLong error:&error]
            : nil;
        id<MTLComputePipelineState> attentionFlashLongTransposePipeline = flashAttnRequested
            ? [device newComputePipelineStateWithFunction:attentionFlashLongTranspose error:&error]
            : nil;
        id<MTLComputePipelineState> attentionFlashMmaTransposePipeline = flashAttnRequested
            ? [device newComputePipelineStateWithFunction:attentionFlashMmaTranspose error:&error]
            : nil;
        id<MTLComputePipelineState> attentionFlashGqaPipeline = flashAttnGqaRequested
            ? [device newComputePipelineStateWithFunction:attentionFlashGqa error:&error]
            : nil;
        id<MTLComputePipelineState> attentionFlashHalfPipeline = flashAttnHalfKvRequested
            ? [device newComputePipelineStateWithFunction:attentionFlashHalf error:&error]
            : nil;
        if (q4Pipeline == nil || (q4LlamaRequested && q4LlamaPipeline == nil) ||
            (q4PairLlamaRequested && q4PairLlamaPipeline == nil) ||
            (q4BatchLlamaRequested && q4BatchLlamaPipeline == nil) ||
            (q4BatchPairLlamaRequested && q4BatchPairLlamaPipeline == nil) ||
            (q4MatmulRequested && (q4MatmulPipeline == nil || q4MatmulPairSwigluHalfPipeline == nil || q4MatmulSwigluPipeline == nil)) ||
            (q4FfnHalfRequested && (q4MatmulHalfInputPipeline == nil || q4MatmulSwigluHalfPipeline == nil)) ||
            (q4FfnDownAddRequested && q4MatmulHalfInputAddPipeline == nil) ||
            (q4FfnGateHalfRequested && (q4MatmulStoreHalfPipeline == nil || q4MatmulSwigluGateHalfPipeline == nil)) ||
            q4PairPipeline == nil || q4BatchPipeline == nil ||
            q4BatchPairPipeline == nil || q8Pipeline == nil ||
            attentionQkMmaPipeline == nil || attentionTileMmaPipeline == nil || attentionStreamMmaPipeline == nil ||
            (q8LlamaRequested && q8LlamaPipeline == nil) ||
            argmaxBlocksPipeline == nil || argmaxFinishPipeline == nil ||
            swigluPipeline == nil || addPipeline == nil || rmsnormPipeline == nil ||
            vectorCopyPipeline == nil || qkvBiasRopeCachePipeline == nil ||
            qkvBiasRopeCacheBatchPipeline == nil || qkvBiasRopeCacheBatchTransposePipeline == nil ||
            attentionDecodePipeline == nil ||
            (flashAttnHalfKvRequested && qkvBiasRopeCacheBatchHalfPipeline == nil) ||
            (flashAttnRequested && attentionFlashPipeline == nil) ||
            (flashAttnRequested && attentionFlashLongPipeline == nil) ||
            (flashAttnRequested && attentionFlashLongTransposePipeline == nil) ||
            (flashAttnRequested && attentionFlashMmaTransposePipeline == nil) ||
            (flashAttnGqaRequested && attentionFlashGqaPipeline == nil) ||
            (flashAttnHalfKvRequested && attentionFlashHalfPipeline == nil)) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }

        TEMetalRuntime *runtime = [TEMetalRuntime new];
        runtime.device = device;
        runtime.queue = queue;
        runtime.q4Pipeline = q4Pipeline;
        runtime.q4LlamaPipeline = q4LlamaPipeline;
        runtime.q4PairPipeline = q4PairPipeline;
        runtime.q4PairLlamaPipeline = q4PairLlamaPipeline;
        runtime.q4BatchPipeline = q4BatchPipeline;
        runtime.q4BatchLlamaPipeline = q4BatchLlamaPipeline;
        runtime.q4MatmulPipeline = q4MatmulPipeline;
        runtime.q4MatmulPairSwigluHalfPipeline = q4MatmulPairSwigluHalfPipeline;
        runtime.q4MatmulSwigluPipeline = q4MatmulSwigluPipeline;
        runtime.q4MatmulHalfInputPipeline = q4MatmulHalfInputPipeline;
        runtime.q4MatmulHalfInputAddPipeline = q4MatmulHalfInputAddPipeline;
        runtime.q4MatmulSwigluHalfPipeline = q4MatmulSwigluHalfPipeline;
        runtime.q4MatmulStoreHalfPipeline = q4MatmulStoreHalfPipeline;
        runtime.q4MatmulSwigluGateHalfPipeline = q4MatmulSwigluGateHalfPipeline;
        runtime.q4BatchPairPipeline = q4BatchPairPipeline;
        runtime.q4BatchPairLlamaPipeline = q4BatchPairLlamaPipeline;
        runtime.q8Pipeline = q8Pipeline;
        runtime.q8LlamaPipeline = q8LlamaPipeline;
        runtime.attentionQkMmaPipeline = attentionQkMmaPipeline;
        runtime.attentionTileMmaPipeline = attentionTileMmaPipeline;
        runtime.attentionStreamMmaPipeline = attentionStreamMmaPipeline;
        runtime.argmaxBlocksPipeline = argmaxBlocksPipeline;
        runtime.argmaxFinishPipeline = argmaxFinishPipeline;
        runtime.swigluPipeline = swigluPipeline;
        runtime.addPipeline = addPipeline;
        runtime.rmsnormPipeline = rmsnormPipeline;
        runtime.vectorCopyPipeline = vectorCopyPipeline;
        runtime.touchPipeline = touchPipeline;
        runtime.qkvBiasRopeCachePipeline = qkvBiasRopeCachePipeline;
        runtime.qkvBiasRopeCacheBatchPipeline = qkvBiasRopeCacheBatchPipeline;
        runtime.qkvBiasRopeCacheBatchHalfPipeline = qkvBiasRopeCacheBatchHalfPipeline;
        runtime.qkvBiasRopeCacheBatchTransposePipeline = qkvBiasRopeCacheBatchTransposePipeline;
        runtime.attentionDecodePipeline = attentionDecodePipeline;
        runtime.attentionFlashPipeline = attentionFlashPipeline;
        runtime.attentionFlashLongPipeline = attentionFlashLongPipeline;
        runtime.attentionFlashLongTransposePipeline = attentionFlashLongTransposePipeline;
        runtime.attentionFlashMmaTransposePipeline = attentionFlashMmaTransposePipeline;
        runtime.attentionFlashGqaPipeline = attentionFlashGqaPipeline;
        runtime.attentionFlashHalfPipeline = attentionFlashHalfPipeline;
        runtime.flashAttentionEnabled = flashAttnRequested;
        runtime.q4FfnHalfEnabled = q4FfnHalfRequested;
        runtime.q4FfnDownAddEnabled = q4FfnDownAddRequested;
        runtime.q4FfnGateHalfEnabled = q4FfnGateHalfRequested;
        runtime.mappingBuffer = nil;
        runtime.inputBuffer = nil;
        runtime.outputBuffer = nil;
        runtime.output2Buffer = nil;
        runtime.output3Buffer = nil;
        runtime.scratchBuffer = nil;
        runtime.normBuffer = nil;
        runtime.dimsBuffer = nil;
        runtime.dims2Buffer = nil;
        runtime.scalarBuffer = nil;
        runtime.weightBuffer = nil;
        runtime.weight2Buffer = nil;
        runtime.biasBuffer = nil;
        runtime.bias2Buffer = nil;
        runtime.bias3Buffer = nil;
        runtime.ropeCosBuffer = nil;
        runtime.ropeSinBuffer = nil;
        runtime.keyCacheBuffer = nil;
        runtime.valueCacheBuffer = nil;
        runtime.keyCacheHalfBuffer = nil;
        runtime.valueCacheHalfBuffer = nil;
        runtime.keyCacheTransposeBuffer = nil;
        runtime.valueCacheTransposeBuffer = nil;
        runtime.mappingPtr = nullptr;
        runtime.mappingLen = 0;
        runtime.inputCapacity = 0;
        runtime.outputCapacity = 0;
        runtime.output2Capacity = 0;
        runtime.output3Capacity = 0;
        runtime.scratchCapacity = 0;
        runtime.normCapacity = 0;
        runtime.weightCapacity = 0;
        runtime.weight2Capacity = 0;
        runtime.biasCapacity = 0;
        runtime.bias2Capacity = 0;
        runtime.bias3Capacity = 0;
        runtime.ropeCosCapacity = 0;
        runtime.ropeSinCapacity = 0;
        runtime.keyCacheCapacity = 0;
        runtime.valueCacheCapacity = 0;
        runtime.keyCacheHalfCapacity = 0;
        runtime.valueCacheHalfCapacity = 0;
        runtime.keyCacheTransposeCapacity = 0;
        runtime.valueCacheTransposeCapacity = 0;
        runtime.residentKvCacheValid = NO;
        runtime.residentKvMappingPtr = nullptr;
        runtime.residentKvLayers = 0;
        runtime.residentKvContextTokens = 0;
        runtime.residentKvKv = 0;
        runtime.residentKvFilledTokens = 0;
        TE_METAL_RUNTIME = runtime;
    }
    return TE_STATUS_OK;
}
