// Included by metal_backend.mm. High-level Qwen layer orchestration.
//
// These entry points compose the primitive Metal kernels into fused
// post-attention MLP, decode-layer, and prefill-layer flows while staying
// in the same translation unit as the static runtime helpers.

// Single-token fused tail: attention residual, FFN norm, gate/up/down MLP, and
// final residual write.
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
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (mapping == nullptr || hidden_in == nullptr || attn == nullptr || ffn_norm_weight == nullptr ||
        out == nullptr || hidden == 0 || ffn == 0 || epsilon <= 0.0f) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (hidden > UINT32_MAX || ffn > UINT32_MAX ||
        output_offset > mapping_len || gate_offset > mapping_len || up_offset > mapping_len || down_offset > mapping_len ||
        ggml_type != TE_METAL_GGML_TYPE_Q4_0) {
        return TE_STATUS_UNSUPPORTED;
    }

    @autoreleasepool {
        std::lock_guard<std::mutex> lock(TE_METAL_MUTEX);
        te_status status = te_metal_init_locked();
        if (status != TE_STATUS_OK) {
            return status;
        }
        TEMetalRuntime *runtime = TE_METAL_RUNTIME;
        if (runtime.mappingBuffer == nil || runtime.mappingPtr != mapping || runtime.mappingLen != mapping_len) {
            runtime.mappingBuffer = [runtime.device newBufferWithBytesNoCopy:(void *)mapping
                                                                      length:mapping_len
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
            runtime.mappingPtr = mapping;
            runtime.mappingLen = mapping_len;
            if (runtime.mappingBuffer == nil) {
                return TE_STATUS_UNSUPPORTED;
            }
        }

        const size_t hidden_blocks = (hidden + 31u) / 32u;
        const size_t ffn_blocks = (ffn + 31u) / 32u;
        const size_t hidden_row_bytes = hidden_blocks * 18u;
        const size_t ffn_row_bytes = ffn_blocks * 18u;
        const size_t hidden_bytes = hidden * sizeof(float);
        const size_t ffn_bytes = ffn * sizeof(float);
        const size_t max_output_bytes = hidden_bytes > ffn_bytes ? hidden_bytes : ffn_bytes;
        if (hidden_row_bytes > UINT32_MAX || ffn_row_bytes > UINT32_MAX ||
            output_offset > UINT64_MAX - hidden_row_bytes * hidden ||
            gate_offset > UINT64_MAX - hidden_row_bytes * ffn ||
            up_offset > UINT64_MAX - hidden_row_bytes * ffn ||
            down_offset > UINT64_MAX - ffn_row_bytes * hidden) {
            return TE_STATUS_UNSUPPORTED;
        }

        if (runtime.inputBuffer == nil || runtime.inputCapacity < hidden_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:hidden_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = hidden_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < max_output_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:max_output_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = max_output_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < ffn_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:ffn_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = ffn_bytes;
        }
        if (runtime.output3Buffer == nil || runtime.output3Capacity < hidden_bytes) {
            runtime.output3Buffer = [runtime.device newBufferWithLength:hidden_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output3Capacity = hidden_bytes;
        }
        if (runtime.scratchBuffer == nil || runtime.scratchCapacity < ffn_bytes) {
            runtime.scratchBuffer = [runtime.device newBufferWithLength:ffn_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.scratchCapacity = ffn_bytes;
        }
        if (runtime.weightBuffer == nil || runtime.weightCapacity < hidden_bytes) {
            runtime.weightBuffer = [runtime.device newBufferWithLength:hidden_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.weightCapacity = hidden_bytes;
        }
        if (runtime.inputBuffer == nil || runtime.outputBuffer == nil || runtime.output2Buffer == nil ||
            runtime.output3Buffer == nil || runtime.scratchBuffer == nil || runtime.weightBuffer == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }

        memcpy([runtime.inputBuffer contents], attn, hidden_bytes);
        memcpy([runtime.output3Buffer contents], hidden_in, hidden_bytes);
        memcpy([runtime.weightBuffer contents], ffn_norm_weight, hidden_bytes);

        const uint32_t output_dims[3] = {(uint32_t)hidden, (uint32_t)hidden, (uint32_t)hidden_row_bytes};
        const uint32_t gate_dims[3] = {(uint32_t)ffn, (uint32_t)hidden, (uint32_t)hidden_row_bytes};
        const uint32_t down_dims[3] = {(uint32_t)hidden, (uint32_t)ffn, (uint32_t)ffn_row_bytes};
        const uint32_t norm_dims[2] = {(uint32_t)hidden, 1u};
        const uint32_t hidden_len = (uint32_t)hidden;
        const uint32_t ffn_len = (uint32_t)ffn;

        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }

        const bool useOutputLlama = te_metal_use_llama_q4(hidden);
        const bool useDownLlama = te_metal_use_llama_q4(ffn);
        const bool useQ4PairLlama = te_metal_use_llama_q4_pair(hidden);
        const NSUInteger outputTile = useOutputLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE;
        const NSUInteger downTile = useDownLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE;
        const NSUInteger ffnTile = useQ4PairLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_PAIR_ROW_TILE;
        const MTLSize outputThreads = MTLSizeMake(useOutputLlama ? TE_METAL_Q4_LLAMA_THREADS : 128, 1, 1);
        const MTLSize downThreads = MTLSizeMake(useDownLlama ? TE_METAL_Q4_LLAMA_THREADS : 128, 1, 1);
        const MTLSize q4Threads = MTLSizeMake(128, 1, 1);
        const MTLSize pairThreads = MTLSizeMake(useQ4PairLlama ? TE_METAL_Q4_LLAMA_THREADS : 128, 1, 1);
        const MTLSize vectorThreads = MTLSizeMake(256, 1, 1);
        const MTLSize outputRows = MTLSizeMake((hidden + outputTile - 1u) / outputTile, 1, 1);
        const MTLSize downRows = MTLSizeMake((hidden + downTile - 1u) / downTile, 1, 1);
        const MTLSize ffnRows = MTLSizeMake((ffn + ffnTile - 1u) / ffnTile, 1, 1);
        const MTLSize hiddenVectorGroups = MTLSizeMake((hidden + 255u) / 256u, 1, 1);
        const MTLSize ffnVectorGroups = MTLSizeMake((ffn + 255u) / 256u, 1, 1);

        [encoder setComputePipelineState:useOutputLlama ? runtime.q4LlamaPipeline : runtime.q4Pipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)output_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBytes:output_dims length:sizeof(output_dims) atIndex:3];
        [encoder dispatchThreadgroups:outputRows threadsPerThreadgroup:outputThreads];

        [encoder setComputePipelineState:runtime.addPipeline];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:2];
        [encoder setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
        [encoder dispatchThreadgroups:hiddenVectorGroups threadsPerThreadgroup:vectorThreads];

        [encoder setComputePipelineState:runtime.rmsnormPipeline];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.weightBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:2];
        [encoder setBytes:norm_dims length:sizeof(norm_dims) atIndex:3];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:q4Threads];

        [encoder setComputePipelineState:useQ4PairLlama ? runtime.q4PairLlamaPipeline : runtime.q4PairPipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)gate_offset atIndex:1];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)up_offset atIndex:2];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:3];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:4];
        [encoder setBytes:gate_dims length:sizeof(gate_dims) atIndex:5];
        [encoder dispatchThreadgroups:ffnRows threadsPerThreadgroup:pairThreads];

        [encoder setComputePipelineState:runtime.swigluPipeline];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
        [encoder setBytes:&ffn_len length:sizeof(ffn_len) atIndex:3];
        [encoder dispatchThreadgroups:ffnVectorGroups threadsPerThreadgroup:vectorThreads];

        [encoder setComputePipelineState:useDownLlama ? runtime.q4LlamaPipeline : runtime.q4Pipeline];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)down_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBytes:down_dims length:sizeof(down_dims) atIndex:3];
        [encoder dispatchThreadgroups:downRows threadsPerThreadgroup:downThreads];

        [encoder setComputePipelineState:runtime.addPipeline];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:2];
        [encoder setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
        [encoder dispatchThreadgroups:hiddenVectorGroups threadsPerThreadgroup:vectorThreads];
        [encoder endEncoding];

        const double profileStart = te_metal_now_ms();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        te_metal_profile_record(
            TE_METAL_PROFILE_POST_ATTN_MLP,
            commandBuffer,
            te_metal_now_ms() - profileStart);
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        memcpy(out, [runtime.output3Buffer contents], hidden_bytes);
        return TE_STATUS_OK;
    }
}

// Batched fused tail used by prompt prefill. This mirrors the single-token
// tail while using batch-aware Q4 mat-mat kernels when the workload is large
// enough to benefit.
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
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (mapping == nullptr || hidden_in == nullptr || attn == nullptr || ffn_norm_weight == nullptr ||
        out == nullptr || batch == 0 || hidden == 0 || ffn == 0 || epsilon <= 0.0f) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (batch > UINT32_MAX || hidden > UINT32_MAX || ffn > UINT32_MAX ||
        output_offset > mapping_len || gate_offset > mapping_len || up_offset > mapping_len || down_offset > mapping_len ||
        ggml_type != TE_METAL_GGML_TYPE_Q4_0 ||
        batch > SIZE_MAX / hidden || batch * hidden > SIZE_MAX / sizeof(float) ||
        batch > SIZE_MAX / ffn || batch * ffn > SIZE_MAX / sizeof(float)) {
        return TE_STATUS_UNSUPPORTED;
    }

    @autoreleasepool {
        std::lock_guard<std::mutex> lock(TE_METAL_MUTEX);
        te_status status = te_metal_init_locked();
        if (status != TE_STATUS_OK) {
            return status;
        }
        TEMetalRuntime *runtime = TE_METAL_RUNTIME;
        if (runtime.mappingBuffer == nil || runtime.mappingPtr != mapping || runtime.mappingLen != mapping_len) {
            runtime.mappingBuffer = [runtime.device newBufferWithBytesNoCopy:(void *)mapping
                                                                      length:mapping_len
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
            runtime.mappingPtr = mapping;
            runtime.mappingLen = mapping_len;
            if (runtime.mappingBuffer == nil) {
                return TE_STATUS_UNSUPPORTED;
            }
        }

        const size_t hidden_blocks = (hidden + 31u) / 32u;
        const size_t ffn_blocks = (ffn + 31u) / 32u;
        const size_t hidden_row_bytes = hidden_blocks * 18u;
        const size_t ffn_row_bytes = ffn_blocks * 18u;
        const size_t hidden_values = batch * hidden;
        const size_t ffn_values = batch * ffn;
        const size_t hidden_bytes = hidden_values * sizeof(float);
        const size_t ffn_bytes = ffn_values * sizeof(float);
        const size_t weight_bytes = hidden * sizeof(float);
        const size_t max_output_bytes = hidden_bytes > ffn_bytes ? hidden_bytes : ffn_bytes;
        if (hidden_row_bytes > UINT32_MAX || ffn_row_bytes > UINT32_MAX ||
            hidden_values > UINT32_MAX || ffn_values > UINT32_MAX ||
            output_offset > UINT64_MAX - hidden_row_bytes * hidden ||
            gate_offset > UINT64_MAX - hidden_row_bytes * ffn ||
            up_offset > UINT64_MAX - hidden_row_bytes * ffn ||
            down_offset > UINT64_MAX - ffn_row_bytes * hidden) {
            return TE_STATUS_UNSUPPORTED;
        }

        if (runtime.inputBuffer == nil || runtime.inputCapacity < hidden_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:hidden_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = hidden_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < max_output_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:max_output_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = max_output_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < ffn_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:ffn_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = ffn_bytes;
        }
        if (runtime.output3Buffer == nil || runtime.output3Capacity < hidden_bytes) {
            runtime.output3Buffer = [runtime.device newBufferWithLength:hidden_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output3Capacity = hidden_bytes;
        }
        if (runtime.scratchBuffer == nil || runtime.scratchCapacity < ffn_bytes) {
            runtime.scratchBuffer = [runtime.device newBufferWithLength:ffn_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.scratchCapacity = ffn_bytes;
        }
        if (runtime.weightBuffer == nil || runtime.weightCapacity < weight_bytes) {
            runtime.weightBuffer = [runtime.device newBufferWithLength:weight_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.weightCapacity = weight_bytes;
        }
        if (runtime.inputBuffer == nil || runtime.outputBuffer == nil || runtime.output2Buffer == nil ||
            runtime.output3Buffer == nil || runtime.scratchBuffer == nil || runtime.weightBuffer == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }

        memcpy([runtime.inputBuffer contents], attn, hidden_bytes);
        memcpy([runtime.output3Buffer contents], hidden_in, hidden_bytes);
        memcpy([runtime.weightBuffer contents], ffn_norm_weight, weight_bytes);

        const uint32_t output_dims[4] = {(uint32_t)hidden, (uint32_t)hidden, (uint32_t)hidden_row_bytes, (uint32_t)batch};
        const uint32_t gate_dims[4] = {(uint32_t)ffn, (uint32_t)hidden, (uint32_t)hidden_row_bytes, (uint32_t)batch};
        const uint32_t down_dims[4] = {(uint32_t)hidden, (uint32_t)ffn, (uint32_t)ffn_row_bytes, (uint32_t)batch};
        const uint32_t norm_dims[2] = {(uint32_t)hidden, (uint32_t)batch};
        const uint32_t hidden_len = (uint32_t)hidden_values;
        const uint32_t ffn_len = (uint32_t)ffn_values;

        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }

        const bool useOutputMatmul = te_metal_use_q4_matmul(hidden, batch);
        const bool useOutputLlama = !useOutputMatmul && te_metal_use_llama_q4_batch(hidden);
        const bool useGateMatmul = te_metal_q4_matmul_gateup_requested() &&
            te_metal_use_q4_matmul_pair(hidden, batch);
        const bool useGateLlama = !useGateMatmul && te_metal_use_llama_q4_batch_pair(hidden);
        const bool useDownMatmul = te_metal_use_q4_matmul(ffn, batch);
        const bool useDownLlama = !useDownMatmul && te_metal_use_llama_q4_batch(ffn);
        const bool useFfnHalf = te_metal_use_q4_ffn_half(runtime, useGateMatmul, useDownMatmul);
        const bool useFfnGateHalf = te_metal_use_q4_ffn_gate_half(runtime, useFfnHalf, batch);
        const NSUInteger outputRowTile = useOutputMatmul
            ? TE_METAL_Q4_MATMUL_ROW_TILE
            : (useOutputLlama ? TE_METAL_Q4_BATCH_LLAMA_ROW_TILE : TE_METAL_Q4_BATCH_ROW_TILE);
        const NSUInteger outputBatchTile = useOutputMatmul
            ? TE_METAL_Q4_MATMUL_BATCH_TILE
            : (useOutputLlama ? TE_METAL_Q4_BATCH_LLAMA_TILE : TE_METAL_Q4_BATCH_TILE);
        const NSUInteger gateRowTile = useGateMatmul
            ? TE_METAL_Q4_MATMUL_ROW_TILE
            : (useGateLlama ? TE_METAL_Q4_BATCH_LLAMA_ROW_TILE : TE_METAL_Q4_BATCH_ROW_TILE);
        const NSUInteger gateBatchTile = useGateMatmul
            ? TE_METAL_Q4_MATMUL_BATCH_TILE
            : (useGateLlama ? TE_METAL_Q4_BATCH_LLAMA_TILE : TE_METAL_Q4_BATCH_TILE);
        const NSUInteger downRowTile = useDownMatmul
            ? TE_METAL_Q4_MATMUL_ROW_TILE
            : (useDownLlama ? TE_METAL_Q4_BATCH_LLAMA_ROW_TILE : TE_METAL_Q4_BATCH_ROW_TILE);
        const NSUInteger downBatchTile = useDownMatmul
            ? TE_METAL_Q4_MATMUL_BATCH_TILE
            : (useDownLlama ? TE_METAL_Q4_BATCH_LLAMA_TILE : TE_METAL_Q4_BATCH_TILE);
        const MTLSize outputThreads = MTLSizeMake(
            useOutputMatmul ? TE_METAL_Q4_MATMUL_THREADS : (useOutputLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS),
            1,
            1);
        const MTLSize gateThreads = MTLSizeMake(
            useGateMatmul ? TE_METAL_Q4_MATMUL_THREADS : (useGateLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS),
            1,
            1);
        const MTLSize downThreads = MTLSizeMake(
            useDownMatmul ? TE_METAL_Q4_MATMUL_THREADS : (useDownLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS),
            1,
            1);
        const MTLSize normThreads = MTLSizeMake(128, 1, 1);
        const MTLSize outputRows = useOutputMatmul
            ? MTLSizeMake((batch + outputBatchTile - 1u) / outputBatchTile, (hidden + outputRowTile - 1u) / outputRowTile, 1)
            : MTLSizeMake((hidden + outputRowTile - 1u) / outputRowTile, (batch + outputBatchTile - 1u) / outputBatchTile, 1);
        const MTLSize gateRows = useGateMatmul
            ? MTLSizeMake((batch + gateBatchTile - 1u) / gateBatchTile, (ffn + gateRowTile - 1u) / gateRowTile, 1)
            : MTLSizeMake((ffn + gateRowTile - 1u) / gateRowTile, (batch + gateBatchTile - 1u) / gateBatchTile, 1);
        const MTLSize downRows = useDownMatmul
            ? MTLSizeMake((batch + downBatchTile - 1u) / downBatchTile, (hidden + downRowTile - 1u) / downRowTile, 1)
            : MTLSizeMake((hidden + downRowTile - 1u) / downRowTile, (batch + downBatchTile - 1u) / downBatchTile, 1);
        const MTLSize vectorThreads = MTLSizeMake(256, 1, 1);
        const MTLSize hiddenVectorGroups = MTLSizeMake((hidden_values + 255u) / 256u, 1, 1);
        const MTLSize ffnVectorGroups = MTLSizeMake((ffn_values + 255u) / 256u, 1, 1);
        const MTLSize normGroups = MTLSizeMake(batch, 1, 1);

        [encoder setComputePipelineState:useOutputMatmul
            ? runtime.q4MatmulPipeline
            : (useOutputLlama ? runtime.q4BatchLlamaPipeline : runtime.q4BatchPipeline)];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)output_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBytes:output_dims length:sizeof(output_dims) atIndex:3];
        if (useOutputMatmul) {
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
        }
        [encoder dispatchThreadgroups:outputRows threadsPerThreadgroup:outputThreads];

        [encoder setComputePipelineState:runtime.addPipeline];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:2];
        [encoder setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
        [encoder dispatchThreadgroups:hiddenVectorGroups threadsPerThreadgroup:vectorThreads];

        [encoder setComputePipelineState:runtime.rmsnormPipeline];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.weightBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:2];
        [encoder setBytes:norm_dims length:sizeof(norm_dims) atIndex:3];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
        [encoder dispatchThreadgroups:normGroups threadsPerThreadgroup:normThreads];

        if (useGateMatmul) {
            [encoder setComputePipelineState:useFfnGateHalf ? runtime.q4MatmulStoreHalfPipeline : runtime.q4MatmulPipeline];
            [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)gate_offset atIndex:1];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
            [encoder setBytes:gate_dims length:sizeof(gate_dims) atIndex:3];
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
            [encoder dispatchThreadgroups:gateRows threadsPerThreadgroup:gateThreads];

            [encoder setComputePipelineState:useFfnGateHalf
                ? runtime.q4MatmulSwigluGateHalfPipeline
                : (useFfnHalf ? runtime.q4MatmulSwigluHalfPipeline : runtime.q4MatmulSwigluPipeline)];
            [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)up_offset atIndex:1];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
            [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:3];
            [encoder setBytes:gate_dims length:sizeof(gate_dims) atIndex:4];
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
            [encoder dispatchThreadgroups:gateRows threadsPerThreadgroup:gateThreads];
        } else {
            [encoder setComputePipelineState:useGateLlama ? runtime.q4BatchPairLlamaPipeline : runtime.q4BatchPairPipeline];
            [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)gate_offset atIndex:1];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)up_offset atIndex:2];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:3];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:4];
            [encoder setBytes:gate_dims length:sizeof(gate_dims) atIndex:5];
            [encoder dispatchThreadgroups:gateRows threadsPerThreadgroup:gateThreads];
        }

        if (!useGateMatmul) {
            [encoder setComputePipelineState:runtime.swigluPipeline];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
            [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
            [encoder setBytes:&ffn_len length:sizeof(ffn_len) atIndex:3];
            [encoder dispatchThreadgroups:ffnVectorGroups threadsPerThreadgroup:vectorThreads];
        }

        [encoder setComputePipelineState:useDownMatmul
            ? (useFfnHalf ? runtime.q4MatmulHalfInputPipeline : runtime.q4MatmulPipeline)
            : (useDownLlama ? runtime.q4BatchLlamaPipeline : runtime.q4BatchPipeline)];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)down_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBytes:down_dims length:sizeof(down_dims) atIndex:3];
        if (useDownMatmul) {
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
        }
        [encoder dispatchThreadgroups:downRows threadsPerThreadgroup:downThreads];

        [encoder setComputePipelineState:runtime.addPipeline];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:2];
        [encoder setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
        [encoder dispatchThreadgroups:hiddenVectorGroups threadsPerThreadgroup:vectorThreads];
        [encoder endEncoding];

        const double profileStart = te_metal_now_ms();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        te_metal_profile_record(
            TE_METAL_PROFILE_POST_ATTN_MLP_BATCH,
            commandBuffer,
            te_metal_now_ms() - profileStart);
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        memcpy(out, [runtime.output3Buffer contents], hidden_bytes);
        return TE_STATUS_OK;
    }
}

// One decode layer for a single generated token. Kept separate from the
// all-layer path for fallback, debugging, and targeted profiling.
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
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (mapping == nullptr || hidden_in == nullptr || attn_norm_weight == nullptr ||
        ffn_norm_weight == nullptr || q_bias == nullptr || k_bias == nullptr || v_bias == nullptr ||
        rope_cos == nullptr || rope_sin == nullptr || key_cache == nullptr || value_cache == nullptr ||
        out == nullptr || hidden == 0 || kv == 0 || heads == 0 || kv_heads == 0 ||
        head_dim == 0 || ffn == 0 || epsilon <= 0.0f) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (position >= context_tokens || (head_dim % 2u) != 0u || heads % kv_heads != 0u ||
        heads > UINT32_MAX || kv_heads > UINT32_MAX || head_dim > UINT32_MAX ||
        hidden > UINT32_MAX || kv > UINT32_MAX || ffn > UINT32_MAX ||
        position > UINT32_MAX || context_tokens > UINT32_MAX ||
        heads * head_dim != hidden || kv_heads * head_dim != kv ||
        q_offset > mapping_len || k_offset > mapping_len || v_offset > mapping_len ||
        output_offset > mapping_len || gate_offset > mapping_len ||
        up_offset > mapping_len || down_offset > mapping_len ||
        ggml_type != TE_METAL_GGML_TYPE_Q4_0) {
        return TE_STATUS_UNSUPPORTED;
    }
    const size_t cache_tokens = position + 1u;
    if (cache_tokens > 8192u ||
        hidden > SIZE_MAX / sizeof(float) || kv > SIZE_MAX / sizeof(float) ||
        ffn > SIZE_MAX / sizeof(float) || cache_tokens > SIZE_MAX / kv ||
        cache_tokens * kv > SIZE_MAX / sizeof(float)) {
        return TE_STATUS_UNSUPPORTED;
    }

    @autoreleasepool {
        std::lock_guard<std::mutex> lock(TE_METAL_MUTEX);
        te_status status = te_metal_init_locked();
        if (status != TE_STATUS_OK) {
            return status;
        }
        TEMetalRuntime *runtime = TE_METAL_RUNTIME;
        if (runtime.mappingBuffer == nil || runtime.mappingPtr != mapping || runtime.mappingLen != mapping_len) {
            runtime.mappingBuffer = [runtime.device newBufferWithBytesNoCopy:(void *)mapping
                                                                      length:mapping_len
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
            runtime.mappingPtr = mapping;
            runtime.mappingLen = mapping_len;
            if (runtime.mappingBuffer == nil) {
                return TE_STATUS_UNSUPPORTED;
            }
        }

        const size_t hidden_blocks = (hidden + 31u) / 32u;
        const size_t ffn_blocks = (ffn + 31u) / 32u;
        const size_t hidden_row_bytes = hidden_blocks * 18u;
        const size_t ffn_row_bytes = ffn_blocks * 18u;
        const size_t hidden_bytes = hidden * sizeof(float);
        const size_t kv_bytes = kv * sizeof(float);
        const size_t ffn_bytes = ffn * sizeof(float);
        const size_t rope_bytes = (head_dim / 2u) * sizeof(float);
        const size_t cache_bytes = cache_tokens * kv_bytes;
        const size_t max_output_bytes = hidden_bytes > ffn_bytes ? hidden_bytes : ffn_bytes;
        const size_t max_scratch_bytes = ffn_bytes > kv_bytes ? ffn_bytes : kv_bytes;
        const size_t max_output2_bytes = max_output_bytes > kv_bytes ? max_output_bytes : kv_bytes;
        if (hidden_row_bytes > UINT32_MAX || ffn_row_bytes > UINT32_MAX ||
            q_offset > UINT64_MAX - hidden_row_bytes * hidden ||
            k_offset > UINT64_MAX - hidden_row_bytes * kv ||
            v_offset > UINT64_MAX - hidden_row_bytes * kv ||
            output_offset > UINT64_MAX - hidden_row_bytes * hidden ||
            gate_offset > UINT64_MAX - hidden_row_bytes * ffn ||
            up_offset > UINT64_MAX - hidden_row_bytes * ffn ||
            down_offset > UINT64_MAX - ffn_row_bytes * hidden) {
            return TE_STATUS_UNSUPPORTED;
        }

        if (runtime.inputBuffer == nil || runtime.inputCapacity < hidden_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:hidden_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = hidden_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < max_output_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:max_output_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = max_output_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < max_output2_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:max_output2_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = max_output2_bytes;
        }
        if (runtime.output3Buffer == nil || runtime.output3Capacity < hidden_bytes) {
            runtime.output3Buffer = [runtime.device newBufferWithLength:hidden_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output3Capacity = hidden_bytes;
        }
        if (runtime.scratchBuffer == nil || runtime.scratchCapacity < max_scratch_bytes) {
            runtime.scratchBuffer = [runtime.device newBufferWithLength:max_scratch_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.scratchCapacity = max_scratch_bytes;
        }
        if (runtime.weightBuffer == nil || runtime.weightCapacity < hidden_bytes) {
            runtime.weightBuffer = [runtime.device newBufferWithLength:hidden_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.weightCapacity = hidden_bytes;
        }
        if (runtime.weight2Buffer == nil || runtime.weight2Capacity < hidden_bytes) {
            runtime.weight2Buffer = [runtime.device newBufferWithLength:hidden_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.weight2Capacity = hidden_bytes;
        }
        if (runtime.biasBuffer == nil || runtime.biasCapacity < hidden_bytes) {
            runtime.biasBuffer = [runtime.device newBufferWithLength:hidden_bytes
                                                             options:MTLResourceStorageModeShared];
            runtime.biasCapacity = hidden_bytes;
        }
        if (runtime.bias2Buffer == nil || runtime.bias2Capacity < kv_bytes) {
            runtime.bias2Buffer = [runtime.device newBufferWithLength:kv_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.bias2Capacity = kv_bytes;
        }
        if (runtime.bias3Buffer == nil || runtime.bias3Capacity < kv_bytes) {
            runtime.bias3Buffer = [runtime.device newBufferWithLength:kv_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.bias3Capacity = kv_bytes;
        }
        if (runtime.ropeCosBuffer == nil || runtime.ropeCosCapacity < rope_bytes) {
            runtime.ropeCosBuffer = [runtime.device newBufferWithLength:rope_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.ropeCosCapacity = rope_bytes;
        }
        if (runtime.ropeSinBuffer == nil || runtime.ropeSinCapacity < rope_bytes) {
            runtime.ropeSinBuffer = [runtime.device newBufferWithLength:rope_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.ropeSinCapacity = rope_bytes;
        }
        if (runtime.keyCacheBuffer == nil || runtime.keyCacheCapacity < cache_bytes) {
            runtime.keyCacheBuffer = [runtime.device newBufferWithLength:cache_bytes
                                                                 options:MTLResourceStorageModeShared];
            runtime.keyCacheCapacity = cache_bytes;
        }
        if (runtime.valueCacheBuffer == nil || runtime.valueCacheCapacity < cache_bytes) {
            runtime.valueCacheBuffer = [runtime.device newBufferWithLength:cache_bytes
                                                                   options:MTLResourceStorageModeShared];
            runtime.valueCacheCapacity = cache_bytes;
        }
        if (runtime.inputBuffer == nil || runtime.outputBuffer == nil || runtime.output2Buffer == nil ||
            runtime.output3Buffer == nil || runtime.scratchBuffer == nil ||
            runtime.weightBuffer == nil || runtime.weight2Buffer == nil ||
            runtime.biasBuffer == nil || runtime.bias2Buffer == nil || runtime.bias3Buffer == nil ||
            runtime.ropeCosBuffer == nil || runtime.ropeSinBuffer == nil ||
            runtime.keyCacheBuffer == nil || runtime.valueCacheBuffer == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }

        memcpy([runtime.inputBuffer contents], hidden_in, hidden_bytes);
        memcpy([runtime.weightBuffer contents], attn_norm_weight, hidden_bytes);
        memcpy([runtime.weight2Buffer contents], ffn_norm_weight, hidden_bytes);
        memcpy([runtime.biasBuffer contents], q_bias, hidden_bytes);
        memcpy([runtime.bias2Buffer contents], k_bias, kv_bytes);
        memcpy([runtime.bias3Buffer contents], v_bias, kv_bytes);
        memcpy([runtime.ropeCosBuffer contents], rope_cos, rope_bytes);
        memcpy([runtime.ropeSinBuffer contents], rope_sin, rope_bytes);
        memcpy([runtime.keyCacheBuffer contents], key_cache, cache_bytes);
        memcpy([runtime.valueCacheBuffer contents], value_cache, cache_bytes);

        const uint32_t hidden_len = (uint32_t)hidden;
        const uint32_t ffn_len = (uint32_t)ffn;
        const uint32_t q_dims[3] = {(uint32_t)hidden, (uint32_t)hidden, (uint32_t)hidden_row_bytes};
        const uint32_t kv_dims[3] = {(uint32_t)kv, (uint32_t)hidden, (uint32_t)hidden_row_bytes};
        const uint32_t output_dims[3] = {(uint32_t)hidden, (uint32_t)hidden, (uint32_t)hidden_row_bytes};
        const uint32_t gate_dims[3] = {(uint32_t)ffn, (uint32_t)hidden, (uint32_t)hidden_row_bytes};
        const uint32_t down_dims[3] = {(uint32_t)hidden, (uint32_t)ffn, (uint32_t)ffn_row_bytes};
        const uint32_t norm_dims[2] = {(uint32_t)hidden, 1u};
        const uint32_t bias_dims[6] = {
            (uint32_t)hidden,
            (uint32_t)kv,
            (uint32_t)heads,
            (uint32_t)kv_heads,
            (uint32_t)head_dim,
            (uint32_t)position};
        const uint32_t attention_dims[6] = {
            1u,
            (uint32_t)position,
            (uint32_t)heads,
            (uint32_t)kv_heads,
            (uint32_t)head_dim,
            (uint32_t)cache_tokens};

        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }

        const bool useQLlama = te_metal_use_llama_q4(hidden);
        const bool usePairLlama = te_metal_use_llama_q4_pair(hidden);
        const bool useOutputLlama = te_metal_use_llama_q4(hidden);
        const bool useDownLlama = te_metal_use_llama_q4(ffn);
        const bool useGateLlama = te_metal_use_llama_q4_pair(hidden);
        const NSUInteger qTile = useQLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE;
        const NSUInteger kvTile = usePairLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_PAIR_ROW_TILE;
        const NSUInteger outputTile = useOutputLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE;
        const NSUInteger downTile = useDownLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE;
        const NSUInteger gateTile = useGateLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_PAIR_ROW_TILE;
        const bool useFlashAttention = te_metal_use_flash_attention(runtime, heads, kv_heads, head_dim);
        const MTLSize vectorThreads = MTLSizeMake(TE_METAL_VECTOR_THREADS, 1, 1);
        const MTLSize normThreads = MTLSizeMake(TE_METAL_RMSNORM_THREADS, 1, 1);
        const MTLSize attentionThreads = MTLSizeMake(useFlashAttention ? TE_METAL_FLASH_ATTENTION_THREADS : TE_METAL_ATTENTION_THREADS, 1, 1);
        const MTLSize copyGroups = MTLSizeMake((hidden + TE_METAL_VECTOR_THREADS - 1u) / TE_METAL_VECTOR_THREADS, 1, 1);
        const MTLSize qThreads = MTLSizeMake(useQLlama ? TE_METAL_Q4_LLAMA_THREADS : 128, 1, 1);
        const MTLSize pairThreads = MTLSizeMake(usePairLlama ? TE_METAL_Q4_LLAMA_THREADS : 128, 1, 1);
        const MTLSize outputThreads = MTLSizeMake(useOutputLlama ? TE_METAL_Q4_LLAMA_THREADS : 128, 1, 1);
        const MTLSize downThreads = MTLSizeMake(useDownLlama ? TE_METAL_Q4_LLAMA_THREADS : 128, 1, 1);
        const MTLSize gateThreads = MTLSizeMake(useGateLlama ? TE_METAL_Q4_LLAMA_THREADS : 128, 1, 1);
        const MTLSize qGroups = MTLSizeMake((hidden + qTile - 1u) / qTile, 1, 1);
        const MTLSize kvGroups = MTLSizeMake((kv + kvTile - 1u) / kvTile, 1, 1);
        const MTLSize outputGroups = MTLSizeMake((hidden + outputTile - 1u) / outputTile, 1, 1);
        const MTLSize downGroups = MTLSizeMake((hidden + downTile - 1u) / downTile, 1, 1);
        const MTLSize gateGroups = MTLSizeMake((ffn + gateTile - 1u) / gateTile, 1, 1);
        const MTLSize ffnVectorGroups = MTLSizeMake((ffn + TE_METAL_VECTOR_THREADS - 1u) / TE_METAL_VECTOR_THREADS, 1, 1);
        const NSUInteger bias_items = std::max(std::max(heads * (head_dim / 2u), kv_heads * (head_dim / 2u)), kv);
        const MTLSize biasGroups = MTLSizeMake((bias_items + TE_METAL_VECTOR_THREADS - 1u) / TE_METAL_VECTOR_THREADS, 1, 1);

        [encoder setComputePipelineState:runtime.vectorCopyPipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:1];
        [encoder setBytes:&hidden_len length:sizeof(hidden_len) atIndex:2];
        [encoder dispatchThreadgroups:copyGroups threadsPerThreadgroup:vectorThreads];

        [encoder setComputePipelineState:runtime.rmsnormPipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.weightBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:2];
        [encoder setBytes:norm_dims length:sizeof(norm_dims) atIndex:3];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:normThreads];

        [encoder setComputePipelineState:useQLlama ? runtime.q4LlamaPipeline : runtime.q4Pipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)q_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBytes:q_dims length:sizeof(q_dims) atIndex:3];
        [encoder dispatchThreadgroups:qGroups threadsPerThreadgroup:qThreads];

        [encoder setComputePipelineState:usePairLlama ? runtime.q4PairLlamaPipeline : runtime.q4PairPipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)k_offset atIndex:1];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)v_offset atIndex:2];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:3];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:4];
        [encoder setBytes:kv_dims length:sizeof(kv_dims) atIndex:5];
        [encoder dispatchThreadgroups:kvGroups threadsPerThreadgroup:pairThreads];

        [encoder setComputePipelineState:runtime.qkvBiasRopeCachePipeline];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.biasBuffer offset:0 atIndex:3];
        [encoder setBuffer:runtime.bias2Buffer offset:0 atIndex:4];
        [encoder setBuffer:runtime.bias3Buffer offset:0 atIndex:5];
        [encoder setBuffer:runtime.ropeCosBuffer offset:0 atIndex:6];
        [encoder setBuffer:runtime.ropeSinBuffer offset:0 atIndex:7];
        [encoder setBuffer:runtime.keyCacheBuffer offset:0 atIndex:8];
        [encoder setBuffer:runtime.valueCacheBuffer offset:0 atIndex:9];
        [encoder setBytes:bias_dims length:sizeof(bias_dims) atIndex:10];
        [encoder dispatchThreadgroups:biasGroups threadsPerThreadgroup:vectorThreads];

        [encoder setComputePipelineState:useFlashAttention ? runtime.attentionFlashPipeline : runtime.attentionDecodePipeline];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.keyCacheBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.valueCacheBuffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:3];
        [encoder setBytes:attention_dims length:sizeof(attention_dims) atIndex:4];
        if (!useFlashAttention) {
            [encoder setThreadgroupMemoryLength:cache_tokens * sizeof(float) atIndex:0];
        }
        [encoder dispatchThreadgroups:MTLSizeMake(1, heads, 1) threadsPerThreadgroup:attentionThreads];

        [encoder setComputePipelineState:useOutputLlama ? runtime.q4LlamaPipeline : runtime.q4Pipeline];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)output_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBytes:output_dims length:sizeof(output_dims) atIndex:3];
        [encoder dispatchThreadgroups:outputGroups threadsPerThreadgroup:outputThreads];

        [encoder setComputePipelineState:runtime.addPipeline];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:2];
        [encoder setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
        [encoder dispatchThreadgroups:copyGroups threadsPerThreadgroup:vectorThreads];

        [encoder setComputePipelineState:runtime.rmsnormPipeline];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.weight2Buffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:2];
        [encoder setBytes:norm_dims length:sizeof(norm_dims) atIndex:3];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:normThreads];

        [encoder setComputePipelineState:useGateLlama ? runtime.q4PairLlamaPipeline : runtime.q4PairPipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)gate_offset atIndex:1];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)up_offset atIndex:2];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:3];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:4];
        [encoder setBytes:gate_dims length:sizeof(gate_dims) atIndex:5];
        [encoder dispatchThreadgroups:gateGroups threadsPerThreadgroup:gateThreads];

        [encoder setComputePipelineState:runtime.swigluPipeline];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
        [encoder setBytes:&ffn_len length:sizeof(ffn_len) atIndex:3];
        [encoder dispatchThreadgroups:ffnVectorGroups threadsPerThreadgroup:vectorThreads];

        [encoder setComputePipelineState:useDownLlama ? runtime.q4LlamaPipeline : runtime.q4Pipeline];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)down_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBytes:down_dims length:sizeof(down_dims) atIndex:3];
        [encoder dispatchThreadgroups:downGroups threadsPerThreadgroup:downThreads];

        [encoder setComputePipelineState:runtime.addPipeline];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:2];
        [encoder setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
        [encoder dispatchThreadgroups:copyGroups threadsPerThreadgroup:vectorThreads];
        [encoder endEncoding];

        const double profileStart = te_metal_now_ms();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        te_metal_profile_record(
            TE_METAL_PROFILE_DECODE_LAYER,
            commandBuffer,
            te_metal_now_ms() - profileStart);
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        memcpy(out, [runtime.output3Buffer contents], hidden_bytes);
        memcpy(
            key_cache + position * kv,
            (const float *)[runtime.keyCacheBuffer contents] + position * kv,
            kv_bytes);
        memcpy(
            value_cache + position * kv,
            (const float *)[runtime.valueCacheBuffer contents] + position * kv,
            kv_bytes);
        return TE_STATUS_OK;
    }
}

// Full decode stack for one generated token. This is the hot path that keeps
// layer-to-layer tensors resident on the GPU across the transformer stack.
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
) {
    const char *disabled = getenv("TINYENGINE_DECODE_ALL_LAYERS");
    if (disabled != nullptr && strcmp(disabled, "0") == 0) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (mapping == nullptr || q_offsets == nullptr || k_offsets == nullptr || v_offsets == nullptr ||
        output_offsets == nullptr || gate_offsets == nullptr || up_offsets == nullptr || down_offsets == nullptr ||
        hidden_in == nullptr || attn_norm_weights == nullptr || ffn_norm_weights == nullptr ||
        q_biases == nullptr || k_biases == nullptr || v_biases == nullptr ||
        rope_cos == nullptr || rope_sin == nullptr || key_cache == nullptr || value_cache == nullptr ||
        out == nullptr || layers == 0 || hidden == 0 || kv == 0 || heads == 0 ||
        kv_heads == 0 || head_dim == 0 || ffn == 0 || epsilon <= 0.0f) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    const bool project_argmax = out_token_id != nullptr;
    if (project_argmax && (output_norm_weight == nullptr || vocab == 0u)) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (project_argmax && head_ggml_type != TE_METAL_GGML_TYPE_Q4_0 && head_ggml_type != TE_METAL_GGML_TYPE_Q8_0) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (position >= context_tokens || (head_dim % 2u) != 0u || heads % kv_heads != 0u ||
        heads * head_dim != hidden || kv_heads * head_dim != kv ||
        layers > UINT32_MAX || hidden > UINT32_MAX || kv > UINT32_MAX ||
        heads > UINT32_MAX || kv_heads > UINT32_MAX || head_dim > UINT32_MAX ||
        ffn > UINT32_MAX || position > UINT32_MAX || context_tokens > UINT32_MAX ||
        ggml_type != TE_METAL_GGML_TYPE_Q4_0 ||
        (project_argmax && (vocab > UINT32_MAX || head_offset > mapping_len))) {
        return TE_STATUS_UNSUPPORTED;
    }
    const size_t cache_tokens = position + 1u;
    if (cache_tokens > 8192u ||
        hidden > SIZE_MAX / sizeof(float) || kv > SIZE_MAX / sizeof(float) ||
        ffn > SIZE_MAX / sizeof(float) ||
        layers > SIZE_MAX / hidden || layers * hidden > SIZE_MAX / sizeof(float) ||
        layers > SIZE_MAX / kv || layers * kv > SIZE_MAX / sizeof(float) ||
        cache_tokens > SIZE_MAX / kv || cache_tokens * kv > SIZE_MAX / sizeof(float) ||
        layers > SIZE_MAX / cache_tokens || layers * cache_tokens > SIZE_MAX / kv ||
        layers * cache_tokens * kv > SIZE_MAX / sizeof(float)) {
        return TE_STATUS_UNSUPPORTED;
    }
    for (size_t layer = 0; layer < layers; ++layer) {
        if (q_offsets[layer] > mapping_len || k_offsets[layer] > mapping_len ||
            v_offsets[layer] > mapping_len || output_offsets[layer] > mapping_len ||
            gate_offsets[layer] > mapping_len || up_offsets[layer] > mapping_len ||
            down_offsets[layer] > mapping_len) {
            return TE_STATUS_UNSUPPORTED;
        }
    }

    @autoreleasepool {
        std::lock_guard<std::mutex> lock(TE_METAL_MUTEX);
        te_status status = te_metal_init_locked();
        if (status != TE_STATUS_OK) {
            return status;
        }
        TEMetalRuntime *runtime = TE_METAL_RUNTIME;
        if (runtime.mappingBuffer == nil || runtime.mappingPtr != mapping || runtime.mappingLen != mapping_len) {
            runtime.mappingBuffer = [runtime.device newBufferWithBytesNoCopy:(void *)mapping
                                                                      length:mapping_len
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
            runtime.mappingPtr = mapping;
            runtime.mappingLen = mapping_len;
            if (runtime.mappingBuffer == nil) {
                return TE_STATUS_UNSUPPORTED;
            }
        }

        const size_t hidden_blocks = (hidden + 31u) / 32u;
        const size_t ffn_blocks = (ffn + 31u) / 32u;
        const size_t head_blocks = (hidden + 31u) / 32u;
        const size_t hidden_row_bytes = hidden_blocks * 18u;
        const size_t ffn_row_bytes = ffn_blocks * 18u;
        const size_t head_row_bytes = head_ggml_type == TE_METAL_GGML_TYPE_Q8_0
            ? head_blocks * 34u
            : head_blocks * 18u;
        const size_t hidden_bytes = hidden * sizeof(float);
        const size_t kv_bytes = kv * sizeof(float);
        const size_t ffn_bytes = ffn * sizeof(float);
        const size_t logits_bytes = project_argmax ? vocab * sizeof(float) : 0u;
        const size_t argmax_blocks = project_argmax
            ? (vocab + TE_METAL_ARGMAX_THREADS - 1u) / TE_METAL_ARGMAX_THREADS
            : 0u;
        const size_t block_value_bytes = argmax_blocks * sizeof(float);
        const size_t block_index_bytes = argmax_blocks * sizeof(uint32_t);
        const size_t rope_bytes = (head_dim / 2u) * sizeof(float);
        const size_t cache_bytes = cache_tokens * kv_bytes;
        const size_t all_weight_bytes = layers * hidden_bytes;
        const size_t all_weight_plus_output_bytes = project_argmax ? all_weight_bytes + hidden_bytes : all_weight_bytes;
        const size_t all_kv_bias_bytes = layers * kv_bytes;
        const size_t all_cache_bytes = layers * cache_bytes;
        size_t max_output_bytes = hidden_bytes > ffn_bytes ? hidden_bytes : ffn_bytes;
        size_t max_scratch_bytes = ffn_bytes > kv_bytes ? ffn_bytes : kv_bytes;
        size_t max_output2_bytes = max_output_bytes > kv_bytes ? max_output_bytes : kv_bytes;
        if (project_argmax) {
            max_output_bytes = max_output_bytes > logits_bytes ? max_output_bytes : logits_bytes;
            max_scratch_bytes = max_scratch_bytes > block_value_bytes ? max_scratch_bytes : block_value_bytes;
            max_output2_bytes = max_output2_bytes > block_index_bytes ? max_output2_bytes : block_index_bytes;
        }
        if (hidden_row_bytes > UINT32_MAX || ffn_row_bytes > UINT32_MAX ||
            (project_argmax && (head_row_bytes > UINT32_MAX || argmax_blocks == 0u || argmax_blocks > UINT32_MAX ||
             vocab > SIZE_MAX / sizeof(float) ||
             head_offset > UINT64_MAX - head_row_bytes * vocab ||
             all_weight_bytes > SIZE_MAX - hidden_bytes))) {
            return TE_STATUS_UNSUPPORTED;
        }
        for (size_t layer = 0; layer < layers; ++layer) {
            if (q_offsets[layer] > UINT64_MAX - hidden_row_bytes * hidden ||
                k_offsets[layer] > UINT64_MAX - hidden_row_bytes * kv ||
                v_offsets[layer] > UINT64_MAX - hidden_row_bytes * kv ||
                output_offsets[layer] > UINT64_MAX - hidden_row_bytes * hidden ||
                gate_offsets[layer] > UINT64_MAX - hidden_row_bytes * ffn ||
                up_offsets[layer] > UINT64_MAX - hidden_row_bytes * ffn ||
                down_offsets[layer] > UINT64_MAX - ffn_row_bytes * hidden) {
                return TE_STATUS_UNSUPPORTED;
            }
        }

        if (runtime.inputBuffer == nil || runtime.inputCapacity < hidden_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:hidden_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = hidden_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < max_output_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:max_output_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = max_output_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < max_output2_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:max_output2_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = max_output2_bytes;
        }
        if (runtime.output3Buffer == nil || runtime.output3Capacity < hidden_bytes) {
            runtime.output3Buffer = [runtime.device newBufferWithLength:hidden_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output3Capacity = hidden_bytes;
        }
        if (runtime.scratchBuffer == nil || runtime.scratchCapacity < max_scratch_bytes) {
            runtime.scratchBuffer = [runtime.device newBufferWithLength:max_scratch_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.scratchCapacity = max_scratch_bytes;
        }
        if (runtime.normBuffer == nil || runtime.normCapacity < hidden_bytes) {
            runtime.normBuffer = [runtime.device newBufferWithLength:hidden_bytes
                                                             options:MTLResourceStorageModeShared];
            runtime.normCapacity = hidden_bytes;
        }
        if (runtime.weightBuffer == nil || runtime.weightCapacity < all_weight_bytes) {
            runtime.weightBuffer = [runtime.device newBufferWithLength:all_weight_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.weightCapacity = all_weight_bytes;
        }
        if (runtime.weight2Buffer == nil || runtime.weight2Capacity < all_weight_plus_output_bytes) {
            runtime.weight2Buffer = [runtime.device newBufferWithLength:all_weight_plus_output_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.weight2Capacity = all_weight_plus_output_bytes;
        }
        if (runtime.biasBuffer == nil || runtime.biasCapacity < all_weight_bytes) {
            runtime.biasBuffer = [runtime.device newBufferWithLength:all_weight_bytes
                                                             options:MTLResourceStorageModeShared];
            runtime.biasCapacity = all_weight_bytes;
        }
        if (runtime.bias2Buffer == nil || runtime.bias2Capacity < all_kv_bias_bytes) {
            runtime.bias2Buffer = [runtime.device newBufferWithLength:all_kv_bias_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.bias2Capacity = all_kv_bias_bytes;
        }
        if (runtime.bias3Buffer == nil || runtime.bias3Capacity < all_kv_bias_bytes) {
            runtime.bias3Buffer = [runtime.device newBufferWithLength:all_kv_bias_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.bias3Capacity = all_kv_bias_bytes;
        }
        if (runtime.ropeCosBuffer == nil || runtime.ropeCosCapacity < rope_bytes) {
            runtime.ropeCosBuffer = [runtime.device newBufferWithLength:rope_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.ropeCosCapacity = rope_bytes;
        }
        if (runtime.ropeSinBuffer == nil || runtime.ropeSinCapacity < rope_bytes) {
            runtime.ropeSinBuffer = [runtime.device newBufferWithLength:rope_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.ropeSinCapacity = rope_bytes;
        }
        if (runtime.keyCacheBuffer == nil || runtime.keyCacheCapacity < all_cache_bytes) {
            runtime.keyCacheBuffer = [runtime.device newBufferWithLength:all_cache_bytes
                                                                 options:MTLResourceStorageModeShared];
            runtime.keyCacheCapacity = all_cache_bytes;
        }
        if (runtime.valueCacheBuffer == nil || runtime.valueCacheCapacity < all_cache_bytes) {
            runtime.valueCacheBuffer = [runtime.device newBufferWithLength:all_cache_bytes
                                                                  options:MTLResourceStorageModeShared];
            runtime.valueCacheCapacity = all_cache_bytes;
        }
        if (project_argmax && runtime.scalarBuffer == nil) {
            runtime.scalarBuffer = [runtime.device newBufferWithLength:sizeof(uint32_t)
                                                               options:MTLResourceStorageModeShared];
        }
        if (runtime.inputBuffer == nil || runtime.outputBuffer == nil || runtime.output2Buffer == nil ||
            runtime.output3Buffer == nil || runtime.scratchBuffer == nil || runtime.normBuffer == nil ||
            runtime.weightBuffer == nil || runtime.weight2Buffer == nil ||
            runtime.biasBuffer == nil || runtime.bias2Buffer == nil || runtime.bias3Buffer == nil ||
            runtime.ropeCosBuffer == nil || runtime.ropeSinBuffer == nil ||
            runtime.keyCacheBuffer == nil || runtime.valueCacheBuffer == nil ||
            (project_argmax && runtime.scalarBuffer == nil)) {
            return TE_STATUS_OUT_OF_MEMORY;
        }

        memcpy([runtime.inputBuffer contents], hidden_in, hidden_bytes);
        memcpy([runtime.weightBuffer contents], attn_norm_weights, all_weight_bytes);
        memcpy([runtime.weight2Buffer contents], ffn_norm_weights, all_weight_bytes);
        if (project_argmax) {
            memcpy((char *)[runtime.weight2Buffer contents] + all_weight_bytes, output_norm_weight, hidden_bytes);
        }
        memcpy([runtime.biasBuffer contents], q_biases, all_weight_bytes);
        memcpy([runtime.bias2Buffer contents], k_biases, all_kv_bias_bytes);
        memcpy([runtime.bias3Buffer contents], v_biases, all_kv_bias_bytes);
        memcpy([runtime.ropeCosBuffer contents], rope_cos, rope_bytes);
        memcpy([runtime.ropeSinBuffer contents], rope_sin, rope_bytes);
        float *key_contents = (float *)[runtime.keyCacheBuffer contents];
        float *value_contents = (float *)[runtime.valueCacheBuffer contents];
        for (size_t layer = 0; layer < layers; ++layer) {
            memcpy(
                key_contents + layer * cache_tokens * kv,
                key_cache + layer * context_tokens * kv,
                cache_bytes);
            memcpy(
                value_contents + layer * cache_tokens * kv,
                value_cache + layer * context_tokens * kv,
                cache_bytes);
        }
        const uint32_t hidden_len = (uint32_t)hidden;
        const uint32_t ffn_len = (uint32_t)ffn;
        const uint32_t q_dims[3] = {(uint32_t)hidden, (uint32_t)hidden, (uint32_t)hidden_row_bytes};
        const uint32_t kv_dims[3] = {(uint32_t)kv, (uint32_t)hidden, (uint32_t)hidden_row_bytes};
        const uint32_t output_dims[3] = {(uint32_t)hidden, (uint32_t)hidden, (uint32_t)hidden_row_bytes};
        const uint32_t gate_dims[3] = {(uint32_t)ffn, (uint32_t)hidden, (uint32_t)hidden_row_bytes};
        const uint32_t down_dims[3] = {(uint32_t)hidden, (uint32_t)ffn, (uint32_t)ffn_row_bytes};
        const uint32_t norm_dims[2] = {(uint32_t)hidden, 1u};
        const uint32_t bias_dims[6] = {
            (uint32_t)hidden,
            (uint32_t)kv,
            (uint32_t)heads,
            (uint32_t)kv_heads,
            (uint32_t)head_dim,
            (uint32_t)position};
        const uint32_t attention_dims[6] = {
            1u,
            (uint32_t)position,
            (uint32_t)heads,
            (uint32_t)kv_heads,
            (uint32_t)head_dim,
            (uint32_t)cache_tokens};

        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }

        const bool useQLlama = te_metal_use_llama_q4(hidden);
        const bool usePairLlama = te_metal_use_llama_q4_pair(hidden);
        const bool useOutputLlama = te_metal_use_llama_q4(hidden);
        const bool useDownLlama = te_metal_use_llama_q4(ffn);
        const bool useGateLlama = te_metal_use_llama_q4_pair(hidden);
        const NSUInteger qTile = useQLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE;
        const NSUInteger kvTile = usePairLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_PAIR_ROW_TILE;
        const NSUInteger outputTile = useOutputLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE;
        const NSUInteger downTile = useDownLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE;
        const NSUInteger gateTile = useGateLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_PAIR_ROW_TILE;
        const bool useFlashAttention = te_metal_use_flash_attention(runtime, heads, kv_heads, head_dim);
        const MTLSize vectorThreads = MTLSizeMake(TE_METAL_VECTOR_THREADS, 1, 1);
        const MTLSize normThreads = MTLSizeMake(TE_METAL_RMSNORM_THREADS, 1, 1);
        const MTLSize attentionThreads = MTLSizeMake(useFlashAttention ? TE_METAL_FLASH_ATTENTION_THREADS : TE_METAL_ATTENTION_THREADS, 1, 1);
        const MTLSize copyGroups = MTLSizeMake((hidden + TE_METAL_VECTOR_THREADS - 1u) / TE_METAL_VECTOR_THREADS, 1, 1);
        const MTLSize qThreads = MTLSizeMake(useQLlama ? TE_METAL_Q4_LLAMA_THREADS : 128, 1, 1);
        const MTLSize pairThreads = MTLSizeMake(usePairLlama ? TE_METAL_Q4_LLAMA_THREADS : 128, 1, 1);
        const MTLSize outputThreads = MTLSizeMake(useOutputLlama ? TE_METAL_Q4_LLAMA_THREADS : 128, 1, 1);
        const MTLSize downThreads = MTLSizeMake(useDownLlama ? TE_METAL_Q4_LLAMA_THREADS : 128, 1, 1);
        const MTLSize gateThreads = MTLSizeMake(useGateLlama ? TE_METAL_Q4_LLAMA_THREADS : 128, 1, 1);
        const MTLSize qGroups = MTLSizeMake((hidden + qTile - 1u) / qTile, 1, 1);
        const MTLSize kvGroups = MTLSizeMake((kv + kvTile - 1u) / kvTile, 1, 1);
        const MTLSize outputGroups = MTLSizeMake((hidden + outputTile - 1u) / outputTile, 1, 1);
        const MTLSize downGroups = MTLSizeMake((hidden + downTile - 1u) / downTile, 1, 1);
        const MTLSize gateGroups = MTLSizeMake((ffn + gateTile - 1u) / gateTile, 1, 1);
        const MTLSize ffnVectorGroups = MTLSizeMake((ffn + TE_METAL_VECTOR_THREADS - 1u) / TE_METAL_VECTOR_THREADS, 1, 1);
        const NSUInteger bias_items = std::max(std::max(heads * (head_dim / 2u), kv_heads * (head_dim / 2u)), kv);
        const MTLSize biasGroups = MTLSizeMake((bias_items + TE_METAL_VECTOR_THREADS - 1u) / TE_METAL_VECTOR_THREADS, 1, 1);

        for (size_t layer = 0; layer < layers; ++layer) {
            id<MTLBuffer> hiddenIn = (layer % 2u) == 0u ? runtime.inputBuffer : runtime.output3Buffer;
            id<MTLBuffer> hiddenOut = (layer % 2u) == 0u ? runtime.output3Buffer : runtime.inputBuffer;
            const NSUInteger normOffset = (NSUInteger)(layer * hidden_bytes);
            const NSUInteger qBiasOffset = (NSUInteger)(layer * hidden_bytes);
            const NSUInteger kvBiasOffset = (NSUInteger)(layer * kv_bytes);
            const NSUInteger cacheOffset = (NSUInteger)(layer * cache_bytes);

            [encoder setComputePipelineState:runtime.vectorCopyPipeline];
            [encoder setBuffer:hiddenIn offset:0 atIndex:0];
            [encoder setBuffer:hiddenOut offset:0 atIndex:1];
            [encoder setBytes:&hidden_len length:sizeof(hidden_len) atIndex:2];
            [encoder dispatchThreadgroups:copyGroups threadsPerThreadgroup:vectorThreads];

            [encoder setComputePipelineState:runtime.rmsnormPipeline];
            [encoder setBuffer:hiddenIn offset:0 atIndex:0];
            [encoder setBuffer:runtime.weightBuffer offset:normOffset atIndex:1];
            [encoder setBuffer:runtime.normBuffer offset:0 atIndex:2];
            [encoder setBytes:norm_dims length:sizeof(norm_dims) atIndex:3];
            [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:normThreads];

            [encoder setComputePipelineState:useQLlama ? runtime.q4LlamaPipeline : runtime.q4Pipeline];
            [encoder setBuffer:runtime.normBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)q_offsets[layer] atIndex:1];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
            [encoder setBytes:q_dims length:sizeof(q_dims) atIndex:3];
            [encoder dispatchThreadgroups:qGroups threadsPerThreadgroup:qThreads];

            [encoder setComputePipelineState:usePairLlama ? runtime.q4PairLlamaPipeline : runtime.q4PairPipeline];
            [encoder setBuffer:runtime.normBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)k_offsets[layer] atIndex:1];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)v_offsets[layer] atIndex:2];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:3];
            [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:4];
            [encoder setBytes:kv_dims length:sizeof(kv_dims) atIndex:5];
            [encoder dispatchThreadgroups:kvGroups threadsPerThreadgroup:pairThreads];

            [encoder setComputePipelineState:runtime.qkvBiasRopeCachePipeline];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
            [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
            [encoder setBuffer:runtime.biasBuffer offset:qBiasOffset atIndex:3];
            [encoder setBuffer:runtime.bias2Buffer offset:kvBiasOffset atIndex:4];
            [encoder setBuffer:runtime.bias3Buffer offset:kvBiasOffset atIndex:5];
            [encoder setBuffer:runtime.ropeCosBuffer offset:0 atIndex:6];
            [encoder setBuffer:runtime.ropeSinBuffer offset:0 atIndex:7];
            [encoder setBuffer:runtime.keyCacheBuffer offset:cacheOffset atIndex:8];
            [encoder setBuffer:runtime.valueCacheBuffer offset:cacheOffset atIndex:9];
            [encoder setBytes:bias_dims length:sizeof(bias_dims) atIndex:10];
            [encoder dispatchThreadgroups:biasGroups threadsPerThreadgroup:vectorThreads];

            [encoder setComputePipelineState:useFlashAttention ? runtime.attentionFlashPipeline : runtime.attentionDecodePipeline];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.keyCacheBuffer offset:cacheOffset atIndex:1];
            [encoder setBuffer:runtime.valueCacheBuffer offset:cacheOffset atIndex:2];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:3];
            [encoder setBytes:attention_dims length:sizeof(attention_dims) atIndex:4];
            if (!useFlashAttention) {
                [encoder setThreadgroupMemoryLength:cache_tokens * sizeof(float) atIndex:0];
            }
            [encoder dispatchThreadgroups:MTLSizeMake(1, heads, 1) threadsPerThreadgroup:attentionThreads];

            [encoder setComputePipelineState:useOutputLlama ? runtime.q4LlamaPipeline : runtime.q4Pipeline];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)output_offsets[layer] atIndex:1];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
            [encoder setBytes:output_dims length:sizeof(output_dims) atIndex:3];
            [encoder dispatchThreadgroups:outputGroups threadsPerThreadgroup:outputThreads];

            [encoder setComputePipelineState:runtime.addPipeline];
            [encoder setBuffer:hiddenOut offset:0 atIndex:0];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:1];
            [encoder setBuffer:hiddenOut offset:0 atIndex:2];
            [encoder setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
            [encoder dispatchThreadgroups:copyGroups threadsPerThreadgroup:vectorThreads];

            [encoder setComputePipelineState:runtime.rmsnormPipeline];
            [encoder setBuffer:hiddenOut offset:0 atIndex:0];
            [encoder setBuffer:runtime.weight2Buffer offset:normOffset atIndex:1];
            [encoder setBuffer:runtime.normBuffer offset:0 atIndex:2];
            [encoder setBytes:norm_dims length:sizeof(norm_dims) atIndex:3];
            [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:normThreads];

            [encoder setComputePipelineState:useGateLlama ? runtime.q4PairLlamaPipeline : runtime.q4PairPipeline];
            [encoder setBuffer:runtime.normBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)gate_offsets[layer] atIndex:1];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)up_offsets[layer] atIndex:2];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:3];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:4];
            [encoder setBytes:gate_dims length:sizeof(gate_dims) atIndex:5];
            [encoder dispatchThreadgroups:gateGroups threadsPerThreadgroup:gateThreads];

            [encoder setComputePipelineState:runtime.swigluPipeline];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
            [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
            [encoder setBytes:&ffn_len length:sizeof(ffn_len) atIndex:3];
            [encoder dispatchThreadgroups:ffnVectorGroups threadsPerThreadgroup:vectorThreads];

            [encoder setComputePipelineState:useDownLlama ? runtime.q4LlamaPipeline : runtime.q4Pipeline];
            [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)down_offsets[layer] atIndex:1];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
            [encoder setBytes:down_dims length:sizeof(down_dims) atIndex:3];
            [encoder dispatchThreadgroups:downGroups threadsPerThreadgroup:downThreads];

            [encoder setComputePipelineState:runtime.addPipeline];
            [encoder setBuffer:hiddenOut offset:0 atIndex:0];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:1];
            [encoder setBuffer:hiddenOut offset:0 atIndex:2];
            [encoder setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
            [encoder dispatchThreadgroups:copyGroups threadsPerThreadgroup:vectorThreads];
        }
        id<MTLBuffer> finalHidden = (layers % 2u) == 0u ? runtime.inputBuffer : runtime.output3Buffer;
        if (project_argmax) {
            const uint32_t head_dims[3] = {(uint32_t)vocab, (uint32_t)hidden, (uint32_t)head_row_bytes};
            const uint32_t norm_dims_project[2] = {(uint32_t)hidden, 1u};
            const uint32_t vocab_u32 = (uint32_t)vocab;
            const uint32_t argmax_blocks_u32 = (uint32_t)argmax_blocks;
            const bool useQ4HeadLlama = head_ggml_type == TE_METAL_GGML_TYPE_Q4_0 && te_metal_use_llama_q4(hidden);
            const bool useQ8HeadLlama = head_ggml_type == TE_METAL_GGML_TYPE_Q8_0 && te_metal_use_llama_q8(hidden);
            const MTLSize headThreads = useQ4HeadLlama
                ? MTLSizeMake(TE_METAL_Q4_LLAMA_THREADS, 1, 1)
                : MTLSizeMake(useQ8HeadLlama ? TE_METAL_Q8_LLAMA_THREADS : 128, 1, 1);
            const MTLSize headGroups = head_ggml_type == TE_METAL_GGML_TYPE_Q4_0
                ? MTLSizeMake(
                    (vocab + (useQ4HeadLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE) - 1u) /
                        (useQ4HeadLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE),
                    1,
                    1)
                : MTLSizeMake((vocab + (useQ8HeadLlama ? TE_METAL_Q8_LLAMA_ROW_TILE : 1u) - 1u) /
                        (useQ8HeadLlama ? TE_METAL_Q8_LLAMA_ROW_TILE : 1u),
                    1,
                    1);

            [encoder setComputePipelineState:runtime.rmsnormPipeline];
            [encoder setBuffer:finalHidden offset:0 atIndex:0];
            [encoder setBuffer:runtime.weight2Buffer offset:(NSUInteger)all_weight_bytes atIndex:1];
            [encoder setBuffer:finalHidden offset:0 atIndex:2];
            [encoder setBytes:norm_dims_project length:sizeof(norm_dims_project) atIndex:3];
            [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:normThreads];

            if (head_ggml_type == TE_METAL_GGML_TYPE_Q4_0) {
                [encoder setComputePipelineState:useQ4HeadLlama ? runtime.q4LlamaPipeline : runtime.q4Pipeline];
            } else {
                [encoder setComputePipelineState:useQ8HeadLlama ? runtime.q8LlamaPipeline : runtime.q8Pipeline];
            }
            [encoder setBuffer:finalHidden offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)head_offset atIndex:1];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
            [encoder setBytes:head_dims length:sizeof(head_dims) atIndex:3];
            if (useQ8HeadLlama) {
                [encoder setThreadgroupMemoryLength:TE_METAL_Q8_LLAMA_SHMEM atIndex:0];
            }
            [encoder dispatchThreadgroups:headGroups threadsPerThreadgroup:headThreads];

            [encoder setComputePipelineState:runtime.argmaxBlocksPipeline];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:1];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:2];
            [encoder setBytes:&vocab_u32 length:sizeof(vocab_u32) atIndex:3];
            const MTLSize argmaxThreads = MTLSizeMake(TE_METAL_ARGMAX_THREADS, 1, 1);
            [encoder dispatchThreadgroups:MTLSizeMake(argmax_blocks, 1, 1) threadsPerThreadgroup:argmaxThreads];

            [encoder setComputePipelineState:runtime.argmaxFinishPipeline];
            [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
            [encoder setBuffer:runtime.scalarBuffer offset:0 atIndex:2];
            [encoder setBytes:&argmax_blocks_u32 length:sizeof(argmax_blocks_u32) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:argmaxThreads];
        }
        [encoder endEncoding];

        const double profileStart = te_metal_now_ms();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        te_metal_profile_record(
            project_argmax ? TE_METAL_PROFILE_DECODE_PROJECT : TE_METAL_PROFILE_DECODE_ALL_LAYERS,
            commandBuffer,
            te_metal_now_ms() - profileStart);
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }

        if (project_argmax) {
            memcpy(out_token_id, [runtime.scalarBuffer contents], sizeof(*out_token_id));
            if (*out_token_id >= vocab) {
                return TE_STATUS_RUNTIME_ERROR;
            }
        } else {
            memcpy(out, [finalHidden contents], hidden_bytes);
        }
        for (size_t layer = 0; layer < layers; ++layer) {
            memcpy(
                key_cache + (layer * context_tokens + position) * kv,
                key_contents + (layer * cache_tokens + position) * kv,
                kv_bytes);
            memcpy(
                value_cache + (layer * context_tokens + position) * kv,
                value_contents + (layer * cache_tokens + position) * kv,
                kv_bytes);
        }
        return TE_STATUS_OK;
    }
}

// One prompt-prefill layer over a batch of prompt tokens. Used as the fallback
// and profiling-friendly version of the fused all-layer prefill path.
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
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (mapping == nullptr || hidden_in == nullptr || attn_norm_weight == nullptr ||
        ffn_norm_weight == nullptr || q_bias == nullptr || k_bias == nullptr || v_bias == nullptr ||
        rope_cos == nullptr || rope_sin == nullptr || key_cache == nullptr || value_cache == nullptr ||
        out == nullptr || batch == 0 || hidden == 0 || kv == 0 || heads == 0 || kv_heads == 0 ||
        head_dim == 0 || ffn == 0 || epsilon <= 0.0f) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (batch > context_tokens || (head_dim % 2u) != 0u || heads % kv_heads != 0u ||
        batch > UINT32_MAX || heads > UINT32_MAX || kv_heads > UINT32_MAX ||
        head_dim > UINT32_MAX || hidden > UINT32_MAX || kv > UINT32_MAX || ffn > UINT32_MAX ||
        heads * head_dim != hidden || kv_heads * head_dim != kv ||
        q_offset > mapping_len || k_offset > mapping_len || v_offset > mapping_len ||
        output_offset > mapping_len || gate_offset > mapping_len ||
        up_offset > mapping_len || down_offset > mapping_len ||
        ggml_type != TE_METAL_GGML_TYPE_Q4_0) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (batch > 8192u ||
        batch > SIZE_MAX / hidden || batch * hidden > SIZE_MAX / sizeof(float) ||
        batch > SIZE_MAX / kv || batch * kv > SIZE_MAX / sizeof(float) ||
        batch > SIZE_MAX / ffn || batch * ffn > SIZE_MAX / sizeof(float) ||
        batch > SIZE_MAX / (head_dim / 2u) ||
        batch * (head_dim / 2u) > SIZE_MAX / sizeof(float)) {
        return TE_STATUS_UNSUPPORTED;
    }

    @autoreleasepool {
        std::lock_guard<std::mutex> lock(TE_METAL_MUTEX);
        te_status status = te_metal_init_locked();
        if (status != TE_STATUS_OK) {
            return status;
        }
        TEMetalRuntime *runtime = TE_METAL_RUNTIME;
        if (runtime.mappingBuffer == nil || runtime.mappingPtr != mapping || runtime.mappingLen != mapping_len) {
            runtime.mappingBuffer = [runtime.device newBufferWithBytesNoCopy:(void *)mapping
                                                                      length:mapping_len
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
            runtime.mappingPtr = mapping;
            runtime.mappingLen = mapping_len;
            if (runtime.mappingBuffer == nil) {
                return TE_STATUS_UNSUPPORTED;
            }
        }

        const size_t hidden_blocks = (hidden + 31u) / 32u;
        const size_t ffn_blocks = (ffn + 31u) / 32u;
        const size_t hidden_row_bytes = hidden_blocks * 18u;
        const size_t ffn_row_bytes = ffn_blocks * 18u;
        const size_t hidden_values = batch * hidden;
        const size_t kv_values = batch * kv;
        const size_t ffn_values = batch * ffn;
        const size_t rope_values = batch * (head_dim / 2u);
        const size_t hidden_bytes = hidden_values * sizeof(float);
        const size_t kv_bytes = kv_values * sizeof(float);
        const size_t ffn_bytes = ffn_values * sizeof(float);
        const size_t weight_bytes = hidden * sizeof(float);
        const size_t q_bias_bytes = hidden * sizeof(float);
        const size_t kv_bias_bytes = kv * sizeof(float);
        const size_t rope_bytes = rope_values * sizeof(float);
        const size_t max_output_bytes = hidden_bytes > ffn_bytes ? hidden_bytes : ffn_bytes;
        const size_t max_output2_bytes = max_output_bytes > kv_bytes ? max_output_bytes : kv_bytes;
        const size_t max_scratch_bytes = ffn_bytes > kv_bytes ? ffn_bytes : kv_bytes;
        if (hidden_row_bytes > UINT32_MAX || ffn_row_bytes > UINT32_MAX ||
            hidden_values > UINT32_MAX || kv_values > UINT32_MAX || ffn_values > UINT32_MAX ||
            q_offset > UINT64_MAX - hidden_row_bytes * hidden ||
            k_offset > UINT64_MAX - hidden_row_bytes * kv ||
            v_offset > UINT64_MAX - hidden_row_bytes * kv ||
            output_offset > UINT64_MAX - hidden_row_bytes * hidden ||
            gate_offset > UINT64_MAX - hidden_row_bytes * ffn ||
            up_offset > UINT64_MAX - hidden_row_bytes * ffn ||
            down_offset > UINT64_MAX - ffn_row_bytes * hidden) {
            return TE_STATUS_UNSUPPORTED;
        }

        if (runtime.inputBuffer == nil || runtime.inputCapacity < hidden_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:hidden_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = hidden_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < max_output_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:max_output_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = max_output_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < max_output2_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:max_output2_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = max_output2_bytes;
        }
        if (runtime.output3Buffer == nil || runtime.output3Capacity < hidden_bytes) {
            runtime.output3Buffer = [runtime.device newBufferWithLength:hidden_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output3Capacity = hidden_bytes;
        }
        if (runtime.scratchBuffer == nil || runtime.scratchCapacity < max_scratch_bytes) {
            runtime.scratchBuffer = [runtime.device newBufferWithLength:max_scratch_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.scratchCapacity = max_scratch_bytes;
        }
        if (runtime.weightBuffer == nil || runtime.weightCapacity < weight_bytes) {
            runtime.weightBuffer = [runtime.device newBufferWithLength:weight_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.weightCapacity = weight_bytes;
        }
        if (runtime.weight2Buffer == nil || runtime.weight2Capacity < weight_bytes) {
            runtime.weight2Buffer = [runtime.device newBufferWithLength:weight_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.weight2Capacity = weight_bytes;
        }
        if (runtime.biasBuffer == nil || runtime.biasCapacity < q_bias_bytes) {
            runtime.biasBuffer = [runtime.device newBufferWithLength:q_bias_bytes
                                                             options:MTLResourceStorageModeShared];
            runtime.biasCapacity = q_bias_bytes;
        }
        if (runtime.bias2Buffer == nil || runtime.bias2Capacity < kv_bias_bytes) {
            runtime.bias2Buffer = [runtime.device newBufferWithLength:kv_bias_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.bias2Capacity = kv_bias_bytes;
        }
        if (runtime.bias3Buffer == nil || runtime.bias3Capacity < kv_bias_bytes) {
            runtime.bias3Buffer = [runtime.device newBufferWithLength:kv_bias_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.bias3Capacity = kv_bias_bytes;
        }
        if (runtime.ropeCosBuffer == nil || runtime.ropeCosCapacity < rope_bytes) {
            runtime.ropeCosBuffer = [runtime.device newBufferWithLength:rope_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.ropeCosCapacity = rope_bytes;
        }
        if (runtime.ropeSinBuffer == nil || runtime.ropeSinCapacity < rope_bytes) {
            runtime.ropeSinBuffer = [runtime.device newBufferWithLength:rope_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.ropeSinCapacity = rope_bytes;
        }
        if (runtime.keyCacheBuffer == nil || runtime.keyCacheCapacity < kv_bytes) {
            runtime.keyCacheBuffer = [runtime.device newBufferWithLength:kv_bytes
                                                                 options:MTLResourceStorageModeShared];
            runtime.keyCacheCapacity = kv_bytes;
        }
        if (runtime.valueCacheBuffer == nil || runtime.valueCacheCapacity < kv_bytes) {
            runtime.valueCacheBuffer = [runtime.device newBufferWithLength:kv_bytes
                                                                   options:MTLResourceStorageModeShared];
            runtime.valueCacheCapacity = kv_bytes;
        }
        if (runtime.inputBuffer == nil || runtime.outputBuffer == nil || runtime.output2Buffer == nil ||
            runtime.output3Buffer == nil || runtime.scratchBuffer == nil ||
            runtime.weightBuffer == nil || runtime.weight2Buffer == nil ||
            runtime.biasBuffer == nil || runtime.bias2Buffer == nil || runtime.bias3Buffer == nil ||
            runtime.ropeCosBuffer == nil || runtime.ropeSinBuffer == nil ||
            runtime.keyCacheBuffer == nil || runtime.valueCacheBuffer == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }

        memcpy([runtime.inputBuffer contents], hidden_in, hidden_bytes);
        memcpy([runtime.weightBuffer contents], attn_norm_weight, weight_bytes);
        memcpy([runtime.weight2Buffer contents], ffn_norm_weight, weight_bytes);
        memcpy([runtime.biasBuffer contents], q_bias, q_bias_bytes);
        memcpy([runtime.bias2Buffer contents], k_bias, kv_bias_bytes);
        memcpy([runtime.bias3Buffer contents], v_bias, kv_bias_bytes);
        memcpy([runtime.ropeCosBuffer contents], rope_cos, rope_bytes);
        memcpy([runtime.ropeSinBuffer contents], rope_sin, rope_bytes);

        const uint32_t hidden_len = (uint32_t)hidden_values;
        const uint32_t ffn_len = (uint32_t)ffn_values;
        const uint32_t q_dims[4] = {(uint32_t)hidden, (uint32_t)hidden, (uint32_t)hidden_row_bytes, (uint32_t)batch};
        const uint32_t kv_dims[4] = {(uint32_t)kv, (uint32_t)hidden, (uint32_t)hidden_row_bytes, (uint32_t)batch};
        const uint32_t output_dims[4] = {(uint32_t)hidden, (uint32_t)hidden, (uint32_t)hidden_row_bytes, (uint32_t)batch};
        const uint32_t gate_dims[4] = {(uint32_t)ffn, (uint32_t)hidden, (uint32_t)hidden_row_bytes, (uint32_t)batch};
        const uint32_t down_dims[4] = {(uint32_t)hidden, (uint32_t)ffn, (uint32_t)ffn_row_bytes, (uint32_t)batch};
        const uint32_t norm_dims[2] = {(uint32_t)hidden, (uint32_t)batch};
        const uint32_t q_pairs = (uint32_t)(heads * (head_dim / 2u));
        const uint32_t k_pairs = (uint32_t)(kv_heads * (head_dim / 2u));
        const uint32_t max_bias_items = std::max(std::max(q_pairs, k_pairs), (uint32_t)kv);
        const uint32_t bias_dims[8] = {
            (uint32_t)batch,
            0u,
            (uint32_t)hidden,
            (uint32_t)kv,
            (uint32_t)heads,
            (uint32_t)kv_heads,
            (uint32_t)head_dim,
            max_bias_items};
        const uint32_t attention_dims[6] = {
            (uint32_t)batch,
            0u,
            (uint32_t)heads,
            (uint32_t)kv_heads,
            (uint32_t)head_dim,
            (uint32_t)batch};

        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }

        const bool useQMatmul = te_metal_use_q4_matmul(hidden, batch);
        const bool useQLlama = !useQMatmul && te_metal_use_llama_q4_batch(hidden);
        const bool useKVMatmul = te_metal_q4_matmul_kv_requested(batch) && te_metal_use_q4_matmul_pair(hidden, batch);
        const bool useKVLlama = !useKVMatmul && te_metal_use_llama_q4_batch_pair(hidden);
        const bool useOutputMatmul = te_metal_use_q4_matmul(hidden, batch);
        const bool useOutputLlama = !useOutputMatmul && te_metal_use_llama_q4_batch(hidden);
        const bool useGateMatmul = te_metal_q4_matmul_gateup_requested() && te_metal_use_q4_matmul_pair(hidden, batch);
        const bool useGateLlama = !useGateMatmul && te_metal_use_llama_q4_batch_pair(hidden);
        const bool useDownMatmul = te_metal_use_q4_matmul(ffn, batch);
        const bool useDownLlama = !useDownMatmul && te_metal_use_llama_q4_batch(ffn);
        const bool useFfnHalf = te_metal_use_q4_ffn_half(runtime, useGateMatmul, useDownMatmul);
        const bool useFfnGateHalf = te_metal_use_q4_ffn_gate_half(runtime, useFfnHalf, batch);
        const bool useFlashAttention = te_metal_use_flash_attention(runtime, heads, kv_heads, head_dim);
        const bool useFlashGqa = useFlashAttention && te_metal_use_flash_attention_gqa(runtime, batch, heads, kv_heads, head_dim);
        const bool useFlashLongTile = useFlashAttention && !useFlashGqa &&
            te_metal_use_flash_attention_long_tile(runtime, batch, heads, kv_heads, head_dim);
        const NSUInteger qRowTile = useQMatmul ? TE_METAL_Q4_MATMUL_ROW_TILE : (useQLlama ? TE_METAL_Q4_BATCH_LLAMA_ROW_TILE : TE_METAL_Q4_BATCH_ROW_TILE);
        const NSUInteger qBatchTile = useQMatmul ? TE_METAL_Q4_MATMUL_BATCH_TILE : (useQLlama ? TE_METAL_Q4_BATCH_LLAMA_TILE : TE_METAL_Q4_BATCH_TILE);
        const NSUInteger kvRowTile = useKVMatmul ? TE_METAL_Q4_MATMUL_ROW_TILE : (useKVLlama ? TE_METAL_Q4_BATCH_LLAMA_ROW_TILE : TE_METAL_Q4_BATCH_ROW_TILE);
        const NSUInteger kvBatchTile = useKVMatmul ? TE_METAL_Q4_MATMUL_BATCH_TILE : (useKVLlama ? TE_METAL_Q4_BATCH_LLAMA_TILE : TE_METAL_Q4_BATCH_TILE);
        const NSUInteger outputRowTile = useOutputMatmul ? TE_METAL_Q4_MATMUL_ROW_TILE : (useOutputLlama ? TE_METAL_Q4_BATCH_LLAMA_ROW_TILE : TE_METAL_Q4_BATCH_ROW_TILE);
        const NSUInteger outputBatchTile = useOutputMatmul ? TE_METAL_Q4_MATMUL_BATCH_TILE : (useOutputLlama ? TE_METAL_Q4_BATCH_LLAMA_TILE : TE_METAL_Q4_BATCH_TILE);
        const NSUInteger gateRowTile = useGateMatmul ? TE_METAL_Q4_MATMUL_ROW_TILE : (useGateLlama ? TE_METAL_Q4_BATCH_LLAMA_ROW_TILE : TE_METAL_Q4_BATCH_ROW_TILE);
        const NSUInteger gateBatchTile = useGateMatmul ? TE_METAL_Q4_MATMUL_BATCH_TILE : (useGateLlama ? TE_METAL_Q4_BATCH_LLAMA_TILE : TE_METAL_Q4_BATCH_TILE);
        const NSUInteger downRowTile = useDownMatmul ? TE_METAL_Q4_MATMUL_ROW_TILE : (useDownLlama ? TE_METAL_Q4_BATCH_LLAMA_ROW_TILE : TE_METAL_Q4_BATCH_ROW_TILE);
        const NSUInteger downBatchTile = useDownMatmul ? TE_METAL_Q4_MATMUL_BATCH_TILE : (useDownLlama ? TE_METAL_Q4_BATCH_LLAMA_TILE : TE_METAL_Q4_BATCH_TILE);
        const MTLSize vectorThreads = MTLSizeMake(TE_METAL_VECTOR_THREADS, 1, 1);
        const MTLSize normThreads = MTLSizeMake(TE_METAL_RMSNORM_THREADS, 1, 1);
        const MTLSize attentionThreads = MTLSizeMake(useFlashAttention ? TE_METAL_FLASH_ATTENTION_THREADS : TE_METAL_ATTENTION_THREADS, 1, 1);
        const MTLSize qThreads = MTLSizeMake(useQMatmul ? TE_METAL_Q4_MATMUL_THREADS : (useQLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS), 1, 1);
        const MTLSize kvThreads = MTLSizeMake(useKVMatmul ? TE_METAL_Q4_MATMUL_THREADS : (useKVLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS), 1, 1);
        const MTLSize outputThreads = MTLSizeMake(useOutputMatmul ? TE_METAL_Q4_MATMUL_THREADS : (useOutputLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS), 1, 1);
        const MTLSize gateThreads = MTLSizeMake(useGateMatmul ? TE_METAL_Q4_MATMUL_THREADS : (useGateLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS), 1, 1);
        const MTLSize downThreads = MTLSizeMake(useDownMatmul ? TE_METAL_Q4_MATMUL_THREADS : (useDownLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS), 1, 1);
        const MTLSize copyGroups = MTLSizeMake((hidden_values + TE_METAL_VECTOR_THREADS - 1u) / TE_METAL_VECTOR_THREADS, 1, 1);
        const MTLSize normGroups = MTLSizeMake(batch, 1, 1);
        const MTLSize qGroups = useQMatmul
            ? MTLSizeMake((batch + qBatchTile - 1u) / qBatchTile, (hidden + qRowTile - 1u) / qRowTile, 1)
            : MTLSizeMake((hidden + qRowTile - 1u) / qRowTile, (batch + qBatchTile - 1u) / qBatchTile, 1);
        const MTLSize kvGroups = useKVMatmul
            ? MTLSizeMake((batch + kvBatchTile - 1u) / kvBatchTile, (kv + kvRowTile - 1u) / kvRowTile, 1)
            : MTLSizeMake((kv + kvRowTile - 1u) / kvRowTile, (batch + kvBatchTile - 1u) / kvBatchTile, 1);
        const MTLSize outputGroups = useOutputMatmul
            ? MTLSizeMake((batch + outputBatchTile - 1u) / outputBatchTile, (hidden + outputRowTile - 1u) / outputRowTile, 1)
            : MTLSizeMake((hidden + outputRowTile - 1u) / outputRowTile, (batch + outputBatchTile - 1u) / outputBatchTile, 1);
        const MTLSize gateGroups = useGateMatmul
            ? MTLSizeMake((batch + gateBatchTile - 1u) / gateBatchTile, (ffn + gateRowTile - 1u) / gateRowTile, 1)
            : MTLSizeMake((ffn + gateRowTile - 1u) / gateRowTile, (batch + gateBatchTile - 1u) / gateBatchTile, 1);
        const MTLSize downGroups = useDownMatmul
            ? MTLSizeMake((batch + downBatchTile - 1u) / downBatchTile, (hidden + downRowTile - 1u) / downRowTile, 1)
            : MTLSizeMake((hidden + downRowTile - 1u) / downRowTile, (batch + downBatchTile - 1u) / downBatchTile, 1);
        const MTLSize ffnVectorGroups = MTLSizeMake((ffn_values + TE_METAL_VECTOR_THREADS - 1u) / TE_METAL_VECTOR_THREADS, 1, 1);
        const MTLSize biasGroups = MTLSizeMake(((NSUInteger)batch * max_bias_items + TE_METAL_VECTOR_THREADS - 1u) / TE_METAL_VECTOR_THREADS, 1, 1);
        const MTLSize attentionGroups = useFlashGqa ? MTLSizeMake(batch, kv_heads, 1) : MTLSizeMake(batch, heads, 1);

        [encoder setComputePipelineState:runtime.vectorCopyPipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:1];
        [encoder setBytes:&hidden_len length:sizeof(hidden_len) atIndex:2];
        [encoder dispatchThreadgroups:copyGroups threadsPerThreadgroup:vectorThreads];

        [encoder setComputePipelineState:runtime.rmsnormPipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.weightBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:2];
        [encoder setBytes:norm_dims length:sizeof(norm_dims) atIndex:3];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
        [encoder dispatchThreadgroups:normGroups threadsPerThreadgroup:normThreads];

        [encoder setComputePipelineState:useQMatmul ? runtime.q4MatmulPipeline : (useQLlama ? runtime.q4BatchLlamaPipeline : runtime.q4BatchPipeline)];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)q_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBytes:q_dims length:sizeof(q_dims) atIndex:3];
        if (useQMatmul) {
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
        }
        [encoder dispatchThreadgroups:qGroups threadsPerThreadgroup:qThreads];

        if (useKVMatmul) {
            [encoder setComputePipelineState:runtime.q4MatmulPipeline];
            [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)k_offset atIndex:1];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:2];
            [encoder setBytes:kv_dims length:sizeof(kv_dims) atIndex:3];
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
            [encoder dispatchThreadgroups:kvGroups threadsPerThreadgroup:kvThreads];

            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)v_offset atIndex:1];
            [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
            [encoder setBytes:kv_dims length:sizeof(kv_dims) atIndex:3];
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
            [encoder dispatchThreadgroups:kvGroups threadsPerThreadgroup:kvThreads];
        } else {
            [encoder setComputePipelineState:useKVLlama ? runtime.q4BatchPairLlamaPipeline : runtime.q4BatchPairPipeline];
            [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)k_offset atIndex:1];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)v_offset atIndex:2];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:3];
            [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:4];
            [encoder setBytes:kv_dims length:sizeof(kv_dims) atIndex:5];
            [encoder dispatchThreadgroups:kvGroups threadsPerThreadgroup:kvThreads];
        }

        [encoder setComputePipelineState:runtime.qkvBiasRopeCacheBatchPipeline];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.biasBuffer offset:0 atIndex:3];
        [encoder setBuffer:runtime.bias2Buffer offset:0 atIndex:4];
        [encoder setBuffer:runtime.bias3Buffer offset:0 atIndex:5];
        [encoder setBuffer:runtime.ropeCosBuffer offset:0 atIndex:6];
        [encoder setBuffer:runtime.ropeSinBuffer offset:0 atIndex:7];
        [encoder setBuffer:runtime.keyCacheBuffer offset:0 atIndex:8];
        [encoder setBuffer:runtime.valueCacheBuffer offset:0 atIndex:9];
        [encoder setBytes:bias_dims length:sizeof(bias_dims) atIndex:10];
        [encoder dispatchThreadgroups:biasGroups threadsPerThreadgroup:vectorThreads];

        [encoder setComputePipelineState:useFlashGqa ? runtime.attentionFlashGqaPipeline : (useFlashLongTile ? runtime.attentionFlashLongPipeline : (useFlashAttention ? runtime.attentionFlashPipeline : runtime.attentionDecodePipeline))];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.keyCacheBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.valueCacheBuffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:3];
        [encoder setBytes:attention_dims length:sizeof(attention_dims) atIndex:4];
        if (!useFlashAttention) {
            [encoder setThreadgroupMemoryLength:batch * sizeof(float) atIndex:0];
        }
        [encoder dispatchThreadgroups:attentionGroups threadsPerThreadgroup:attentionThreads];

        [encoder setComputePipelineState:useOutputMatmul ? runtime.q4MatmulPipeline : (useOutputLlama ? runtime.q4BatchLlamaPipeline : runtime.q4BatchPipeline)];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)output_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBytes:output_dims length:sizeof(output_dims) atIndex:3];
        if (useOutputMatmul) {
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
        }
        [encoder dispatchThreadgroups:outputGroups threadsPerThreadgroup:outputThreads];

        [encoder setComputePipelineState:runtime.addPipeline];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:2];
        [encoder setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
        [encoder dispatchThreadgroups:copyGroups threadsPerThreadgroup:vectorThreads];

        [encoder setComputePipelineState:runtime.rmsnormPipeline];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.weight2Buffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:2];
        [encoder setBytes:norm_dims length:sizeof(norm_dims) atIndex:3];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
        [encoder dispatchThreadgroups:normGroups threadsPerThreadgroup:normThreads];

        if (useGateMatmul) {
            [encoder setComputePipelineState:useFfnGateHalf ? runtime.q4MatmulStoreHalfPipeline : runtime.q4MatmulPipeline];
            [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)gate_offset atIndex:1];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
            [encoder setBytes:gate_dims length:sizeof(gate_dims) atIndex:3];
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
            [encoder dispatchThreadgroups:gateGroups threadsPerThreadgroup:gateThreads];

            [encoder setComputePipelineState:useFfnGateHalf
                ? runtime.q4MatmulSwigluGateHalfPipeline
                : (useFfnHalf ? runtime.q4MatmulSwigluHalfPipeline : runtime.q4MatmulSwigluPipeline)];
            [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)up_offset atIndex:1];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
            [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:3];
            [encoder setBytes:gate_dims length:sizeof(gate_dims) atIndex:4];
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
            [encoder dispatchThreadgroups:gateGroups threadsPerThreadgroup:gateThreads];
        } else {
            [encoder setComputePipelineState:useGateLlama ? runtime.q4BatchPairLlamaPipeline : runtime.q4BatchPairPipeline];
            [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)gate_offset atIndex:1];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)up_offset atIndex:2];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:3];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:4];
            [encoder setBytes:gate_dims length:sizeof(gate_dims) atIndex:5];
            [encoder dispatchThreadgroups:gateGroups threadsPerThreadgroup:gateThreads];
        }

        if (!useGateMatmul) {
            [encoder setComputePipelineState:runtime.swigluPipeline];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
            [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
            [encoder setBytes:&ffn_len length:sizeof(ffn_len) atIndex:3];
            [encoder dispatchThreadgroups:ffnVectorGroups threadsPerThreadgroup:vectorThreads];
        }

        [encoder setComputePipelineState:useDownMatmul ? (useFfnHalf ? runtime.q4MatmulHalfInputPipeline : runtime.q4MatmulPipeline) : (useDownLlama ? runtime.q4BatchLlamaPipeline : runtime.q4BatchPipeline)];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)down_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBytes:down_dims length:sizeof(down_dims) atIndex:3];
        if (useDownMatmul) {
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
        }
        [encoder dispatchThreadgroups:downGroups threadsPerThreadgroup:downThreads];

        [encoder setComputePipelineState:runtime.addPipeline];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:2];
        [encoder setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
        [encoder dispatchThreadgroups:copyGroups threadsPerThreadgroup:vectorThreads];
        [encoder endEncoding];

        const double profileStart = te_metal_now_ms();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        te_metal_profile_record(
            TE_METAL_PROFILE_PREFILL_LAYER,
            commandBuffer,
            te_metal_now_ms() - profileStart);
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        memcpy(out, [runtime.output3Buffer contents], hidden_bytes);
        memcpy(key_cache, [runtime.keyCacheBuffer contents], kv_bytes);
        memcpy(value_cache, [runtime.valueCacheBuffer contents], kv_bytes);
        return TE_STATUS_OK;
    }
}

// Full prompt-prefill stack. It composes QKV, attention, and FFN stages across
// all layers with minimal host synchronization.
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
) {
    const char *disabled = getenv("TINYENGINE_PREFILL_ALL_LAYERS");
    if (disabled != nullptr && strcmp(disabled, "0") == 0) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (mapping == nullptr || q_offsets == nullptr || k_offsets == nullptr || v_offsets == nullptr ||
        output_offsets == nullptr || gate_offsets == nullptr || up_offsets == nullptr || down_offsets == nullptr ||
        hidden_in == nullptr || attn_norm_weights == nullptr || ffn_norm_weights == nullptr ||
        q_biases == nullptr || k_biases == nullptr || v_biases == nullptr ||
        rope_cos == nullptr || rope_sin == nullptr || key_cache == nullptr || value_cache == nullptr ||
        out == nullptr || layers == 0 || batch == 0 || hidden == 0 || kv == 0 ||
        heads == 0 || kv_heads == 0 || head_dim == 0 || ffn == 0 || epsilon <= 0.0f) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (batch > context_tokens || batch > 8192u || (head_dim % 2u) != 0u ||
        heads % kv_heads != 0u || heads * head_dim != hidden || kv_heads * head_dim != kv ||
        layers > UINT32_MAX || batch > UINT32_MAX || hidden > UINT32_MAX ||
        kv > UINT32_MAX || heads > UINT32_MAX || kv_heads > UINT32_MAX ||
        head_dim > UINT32_MAX || ffn > UINT32_MAX ||
        ggml_type != TE_METAL_GGML_TYPE_Q4_0) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (batch > SIZE_MAX / hidden || batch * hidden > SIZE_MAX / sizeof(float) ||
        batch > SIZE_MAX / kv || batch * kv > SIZE_MAX / sizeof(float) ||
        batch > SIZE_MAX / ffn || batch * ffn > SIZE_MAX / sizeof(float) ||
        layers > SIZE_MAX / hidden || layers * hidden > SIZE_MAX / sizeof(float) ||
        layers > SIZE_MAX / kv || layers * kv > SIZE_MAX / sizeof(float) ||
        layers > SIZE_MAX / batch || layers * batch > SIZE_MAX / kv ||
        layers * batch * kv > SIZE_MAX / sizeof(float) ||
        batch > SIZE_MAX / (head_dim / 2u) ||
        batch * (head_dim / 2u) > SIZE_MAX / sizeof(float)) {
        return TE_STATUS_UNSUPPORTED;
    }
    for (size_t layer = 0; layer < layers; ++layer) {
        if (q_offsets[layer] > mapping_len || k_offsets[layer] > mapping_len ||
            v_offsets[layer] > mapping_len || output_offsets[layer] > mapping_len ||
            gate_offsets[layer] > mapping_len || up_offsets[layer] > mapping_len ||
            down_offsets[layer] > mapping_len) {
            return TE_STATUS_UNSUPPORTED;
        }
    }

    @autoreleasepool {
        std::lock_guard<std::mutex> lock(TE_METAL_MUTEX);
        te_status status = te_metal_init_locked();
        if (status != TE_STATUS_OK) {
            return status;
        }
        TEMetalRuntime *runtime = TE_METAL_RUNTIME;
        if (runtime.mappingBuffer == nil || runtime.mappingPtr != mapping || runtime.mappingLen != mapping_len) {
            runtime.mappingBuffer = [runtime.device newBufferWithBytesNoCopy:(void *)mapping
                                                                      length:mapping_len
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
            runtime.mappingPtr = mapping;
            runtime.mappingLen = mapping_len;
            if (runtime.mappingBuffer == nil) {
                return TE_STATUS_UNSUPPORTED;
            }
        }

        const size_t hidden_blocks = (hidden + 31u) / 32u;
        const size_t ffn_blocks = (ffn + 31u) / 32u;
        const size_t hidden_row_bytes = hidden_blocks * 18u;
        const size_t ffn_row_bytes = ffn_blocks * 18u;
        const size_t hidden_values = batch * hidden;
        const size_t kv_values = batch * kv;
        const size_t ffn_values = batch * ffn;
        const size_t rope_values = batch * (head_dim / 2u);
        const size_t layer_hidden_values = layers * hidden;
        const size_t layer_kv_values = layers * kv;
        const size_t all_cache_values = layers * batch * kv;
        const size_t hidden_bytes = hidden_values * sizeof(float);
        const size_t kv_bytes = kv_values * sizeof(float);
        const size_t ffn_bytes = ffn_values * sizeof(float);
        const size_t rope_bytes = rope_values * sizeof(float);
        const size_t weight_bytes = hidden * sizeof(float);
        const size_t q_bias_bytes = hidden * sizeof(float);
        const size_t kv_bias_bytes = kv * sizeof(float);
        const size_t all_weight_bytes = layer_hidden_values * sizeof(float);
        const size_t all_kv_bias_bytes = layer_kv_values * sizeof(float);
        const size_t all_cache_bytes = all_cache_values * sizeof(float);
        const size_t all_cache_half_bytes = all_cache_values * sizeof(uint16_t);
        const size_t max_output_bytes = hidden_bytes > ffn_bytes ? hidden_bytes : ffn_bytes;
        const size_t max_output2_bytes = max_output_bytes > kv_bytes ? max_output_bytes : kv_bytes;
        const size_t max_scratch_bytes = ffn_bytes > kv_bytes ? ffn_bytes : kv_bytes;
        if (hidden_row_bytes > UINT32_MAX || ffn_row_bytes > UINT32_MAX ||
            hidden_values > UINT32_MAX || kv_values > UINT32_MAX || ffn_values > UINT32_MAX) {
            return TE_STATUS_UNSUPPORTED;
        }
        for (size_t layer = 0; layer < layers; ++layer) {
            if (q_offsets[layer] > UINT64_MAX - hidden_row_bytes * hidden ||
                k_offsets[layer] > UINT64_MAX - hidden_row_bytes * kv ||
                v_offsets[layer] > UINT64_MAX - hidden_row_bytes * kv ||
                output_offsets[layer] > UINT64_MAX - hidden_row_bytes * hidden ||
                gate_offsets[layer] > UINT64_MAX - hidden_row_bytes * ffn ||
                up_offsets[layer] > UINT64_MAX - hidden_row_bytes * ffn ||
                down_offsets[layer] > UINT64_MAX - ffn_row_bytes * hidden) {
                return TE_STATUS_UNSUPPORTED;
            }
        }

        if (runtime.inputBuffer == nil || runtime.inputCapacity < hidden_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:hidden_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = hidden_bytes;
        }
        if (runtime.output3Buffer == nil || runtime.output3Capacity < hidden_bytes) {
            runtime.output3Buffer = [runtime.device newBufferWithLength:hidden_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output3Capacity = hidden_bytes;
        }
        if (runtime.normBuffer == nil || runtime.normCapacity < hidden_bytes) {
            runtime.normBuffer = [runtime.device newBufferWithLength:hidden_bytes
                                                             options:MTLResourceStorageModeShared];
            runtime.normCapacity = hidden_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < max_output_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:max_output_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = max_output_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < max_output2_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:max_output2_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = max_output2_bytes;
        }
        if (runtime.scratchBuffer == nil || runtime.scratchCapacity < max_scratch_bytes) {
            runtime.scratchBuffer = [runtime.device newBufferWithLength:max_scratch_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.scratchCapacity = max_scratch_bytes;
        }
        if (runtime.weightBuffer == nil || runtime.weightCapacity < all_weight_bytes) {
            runtime.weightBuffer = [runtime.device newBufferWithLength:all_weight_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.weightCapacity = all_weight_bytes;
        }
        if (runtime.weight2Buffer == nil || runtime.weight2Capacity < all_weight_bytes) {
            runtime.weight2Buffer = [runtime.device newBufferWithLength:all_weight_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.weight2Capacity = all_weight_bytes;
        }
        if (runtime.biasBuffer == nil || runtime.biasCapacity < all_weight_bytes) {
            runtime.biasBuffer = [runtime.device newBufferWithLength:all_weight_bytes
                                                             options:MTLResourceStorageModeShared];
            runtime.biasCapacity = all_weight_bytes;
        }
        if (runtime.bias2Buffer == nil || runtime.bias2Capacity < all_kv_bias_bytes) {
            runtime.bias2Buffer = [runtime.device newBufferWithLength:all_kv_bias_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.bias2Capacity = all_kv_bias_bytes;
        }
        if (runtime.bias3Buffer == nil || runtime.bias3Capacity < all_kv_bias_bytes) {
            runtime.bias3Buffer = [runtime.device newBufferWithLength:all_kv_bias_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.bias3Capacity = all_kv_bias_bytes;
        }
        if (runtime.ropeCosBuffer == nil || runtime.ropeCosCapacity < rope_bytes) {
            runtime.ropeCosBuffer = [runtime.device newBufferWithLength:rope_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.ropeCosCapacity = rope_bytes;
        }
        if (runtime.ropeSinBuffer == nil || runtime.ropeSinCapacity < rope_bytes) {
            runtime.ropeSinBuffer = [runtime.device newBufferWithLength:rope_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.ropeSinCapacity = rope_bytes;
        }
        if (runtime.keyCacheBuffer == nil || runtime.keyCacheCapacity < all_cache_bytes) {
            runtime.keyCacheBuffer = [runtime.device newBufferWithLength:all_cache_bytes
                                                                 options:MTLResourceStorageModeShared];
            runtime.keyCacheCapacity = all_cache_bytes;
        }
        if (runtime.valueCacheBuffer == nil || runtime.valueCacheCapacity < all_cache_bytes) {
            runtime.valueCacheBuffer = [runtime.device newBufferWithLength:all_cache_bytes
                                                                   options:MTLResourceStorageModeShared];
            runtime.valueCacheCapacity = all_cache_bytes;
        }
        const bool canUseFlashHalfKv = runtime.qkvBiasRopeCacheBatchHalfPipeline != nil &&
            runtime.attentionFlashHalfPipeline != nil && te_metal_flash_attention_half_kv_requested();
        if (canUseFlashHalfKv &&
            (runtime.keyCacheHalfBuffer == nil || runtime.keyCacheHalfCapacity < all_cache_half_bytes)) {
            runtime.keyCacheHalfBuffer = [runtime.device newBufferWithLength:all_cache_half_bytes
                                                                     options:MTLResourceStorageModeShared];
            runtime.keyCacheHalfCapacity = all_cache_half_bytes;
        }
        if (canUseFlashHalfKv &&
            (runtime.valueCacheHalfBuffer == nil || runtime.valueCacheHalfCapacity < all_cache_half_bytes)) {
            runtime.valueCacheHalfBuffer = [runtime.device newBufferWithLength:all_cache_half_bytes
                                                                       options:MTLResourceStorageModeShared];
            runtime.valueCacheHalfCapacity = all_cache_half_bytes;
        }
        if (runtime.inputBuffer == nil || runtime.output3Buffer == nil || runtime.normBuffer == nil ||
            runtime.outputBuffer == nil || runtime.output2Buffer == nil || runtime.scratchBuffer == nil ||
            runtime.weightBuffer == nil || runtime.weight2Buffer == nil ||
            runtime.biasBuffer == nil || runtime.bias2Buffer == nil || runtime.bias3Buffer == nil ||
            runtime.ropeCosBuffer == nil || runtime.ropeSinBuffer == nil ||
            runtime.keyCacheBuffer == nil || runtime.valueCacheBuffer == nil ||
            (canUseFlashHalfKv && (runtime.keyCacheHalfBuffer == nil || runtime.valueCacheHalfBuffer == nil))) {
            return TE_STATUS_OUT_OF_MEMORY;
        }

        memcpy([runtime.inputBuffer contents], hidden_in, hidden_bytes);
        memcpy([runtime.weightBuffer contents], attn_norm_weights, all_weight_bytes);
        memcpy([runtime.weight2Buffer contents], ffn_norm_weights, all_weight_bytes);
        memcpy([runtime.biasBuffer contents], q_biases, all_weight_bytes);
        memcpy([runtime.bias2Buffer contents], k_biases, all_kv_bias_bytes);
        memcpy([runtime.bias3Buffer contents], v_biases, all_kv_bias_bytes);
        memcpy([runtime.ropeCosBuffer contents], rope_cos, rope_bytes);
        memcpy([runtime.ropeSinBuffer contents], rope_sin, rope_bytes);

        const uint32_t hidden_len = (uint32_t)hidden_values;
        const uint32_t ffn_len = (uint32_t)ffn_values;
        const uint32_t q_dims[4] = {(uint32_t)hidden, (uint32_t)hidden, (uint32_t)hidden_row_bytes, (uint32_t)batch};
        const uint32_t kv_dims[4] = {(uint32_t)kv, (uint32_t)hidden, (uint32_t)hidden_row_bytes, (uint32_t)batch};
        const uint32_t output_dims[4] = {(uint32_t)hidden, (uint32_t)hidden, (uint32_t)hidden_row_bytes, (uint32_t)batch};
        const uint32_t gate_dims[4] = {(uint32_t)ffn, (uint32_t)hidden, (uint32_t)hidden_row_bytes, (uint32_t)batch};
        const uint32_t down_dims[4] = {(uint32_t)hidden, (uint32_t)ffn, (uint32_t)ffn_row_bytes, (uint32_t)batch};
        const uint32_t norm_dims[2] = {(uint32_t)hidden, (uint32_t)batch};
        const uint32_t q_pairs = (uint32_t)(heads * (head_dim / 2u));
        const uint32_t k_pairs = (uint32_t)(kv_heads * (head_dim / 2u));
        const uint32_t max_bias_items = std::max(std::max(q_pairs, k_pairs), (uint32_t)kv);
        const uint32_t bias_dims[8] = {
            (uint32_t)batch,
            0u,
            (uint32_t)hidden,
            (uint32_t)kv,
            (uint32_t)heads,
            (uint32_t)kv_heads,
            (uint32_t)head_dim,
            max_bias_items};
        const uint32_t attention_dims[6] = {
            (uint32_t)batch,
            0u,
            (uint32_t)heads,
            (uint32_t)kv_heads,
            (uint32_t)head_dim,
            (uint32_t)batch};

        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }

        const bool useQMatmul = te_metal_use_q4_matmul(hidden, batch);
        const bool useQLlama = !useQMatmul && te_metal_use_llama_q4_batch(hidden);
        const bool useKVMatmul = te_metal_q4_matmul_kv_requested(batch) && te_metal_use_q4_matmul_pair(hidden, batch);
        const bool useKVLlama = !useKVMatmul && te_metal_use_llama_q4_batch_pair(hidden);
        const bool useOutputMatmul = te_metal_use_q4_matmul(hidden, batch);
        const bool useOutputLlama = !useOutputMatmul && te_metal_use_llama_q4_batch(hidden);
        const bool useGateMatmul = te_metal_q4_matmul_gateup_requested() && te_metal_use_q4_matmul_pair(hidden, batch);
        const bool useGateLlama = !useGateMatmul && te_metal_use_llama_q4_batch_pair(hidden);
        const bool useDownMatmul = te_metal_use_q4_matmul(ffn, batch);
        const bool useDownLlama = !useDownMatmul && te_metal_use_llama_q4_batch(ffn);
        const bool useFfnHalf = te_metal_use_q4_ffn_half(runtime, useGateMatmul, useDownMatmul);
        const bool useFfnPairSwiglu = te_metal_use_q4_ffn_pair_swiglu(runtime, useGateMatmul, useFfnHalf, batch);
        const bool useFfnGateHalf = te_metal_use_q4_ffn_gate_half(runtime, useFfnHalf, batch);
        const bool useFlashAttention = te_metal_use_flash_attention(runtime, heads, kv_heads, head_dim);
        const bool useFlashGqa = useFlashAttention && te_metal_use_flash_attention_gqa(runtime, batch, heads, kv_heads, head_dim);
        const bool useFlashLongTile = useFlashAttention && !useFlashGqa &&
            te_metal_use_flash_attention_long_tile(runtime, batch, heads, kv_heads, head_dim);
        const bool useFlashHalfKv = useFlashAttention && !useFlashGqa && !useFlashLongTile && canUseFlashHalfKv;
        const NSUInteger qRowTile = useQMatmul ? TE_METAL_Q4_MATMUL_ROW_TILE : (useQLlama ? TE_METAL_Q4_BATCH_LLAMA_ROW_TILE : TE_METAL_Q4_BATCH_ROW_TILE);
        const NSUInteger qBatchTile = useQMatmul ? TE_METAL_Q4_MATMUL_BATCH_TILE : (useQLlama ? TE_METAL_Q4_BATCH_LLAMA_TILE : TE_METAL_Q4_BATCH_TILE);
        const NSUInteger kvRowTile = useKVMatmul ? TE_METAL_Q4_MATMUL_ROW_TILE : (useKVLlama ? TE_METAL_Q4_BATCH_LLAMA_ROW_TILE : TE_METAL_Q4_BATCH_ROW_TILE);
        const NSUInteger kvBatchTile = useKVMatmul ? TE_METAL_Q4_MATMUL_BATCH_TILE : (useKVLlama ? TE_METAL_Q4_BATCH_LLAMA_TILE : TE_METAL_Q4_BATCH_TILE);
        const NSUInteger outputRowTile = useOutputMatmul ? TE_METAL_Q4_MATMUL_ROW_TILE : (useOutputLlama ? TE_METAL_Q4_BATCH_LLAMA_ROW_TILE : TE_METAL_Q4_BATCH_ROW_TILE);
        const NSUInteger outputBatchTile = useOutputMatmul ? TE_METAL_Q4_MATMUL_BATCH_TILE : (useOutputLlama ? TE_METAL_Q4_BATCH_LLAMA_TILE : TE_METAL_Q4_BATCH_TILE);
        const NSUInteger gateRowTile = useGateMatmul ? TE_METAL_Q4_MATMUL_ROW_TILE : (useGateLlama ? TE_METAL_Q4_BATCH_LLAMA_ROW_TILE : TE_METAL_Q4_BATCH_ROW_TILE);
        const NSUInteger gateBatchTile = useGateMatmul ? TE_METAL_Q4_MATMUL_BATCH_TILE : (useGateLlama ? TE_METAL_Q4_BATCH_LLAMA_TILE : TE_METAL_Q4_BATCH_TILE);
        const NSUInteger downRowTile = useDownMatmul ? TE_METAL_Q4_MATMUL_ROW_TILE : (useDownLlama ? TE_METAL_Q4_BATCH_LLAMA_ROW_TILE : TE_METAL_Q4_BATCH_ROW_TILE);
        const NSUInteger downBatchTile = useDownMatmul ? TE_METAL_Q4_MATMUL_BATCH_TILE : (useDownLlama ? TE_METAL_Q4_BATCH_LLAMA_TILE : TE_METAL_Q4_BATCH_TILE);
        const MTLSize vectorThreads = MTLSizeMake(TE_METAL_VECTOR_THREADS, 1, 1);
        const MTLSize normThreads = MTLSizeMake(TE_METAL_RMSNORM_THREADS, 1, 1);
        const MTLSize attentionThreads = MTLSizeMake(useFlashAttention ? TE_METAL_FLASH_ATTENTION_THREADS : TE_METAL_ATTENTION_THREADS, 1, 1);
        const MTLSize qThreads = MTLSizeMake(useQMatmul ? TE_METAL_Q4_MATMUL_THREADS : (useQLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS), 1, 1);
        const MTLSize kvThreads = MTLSizeMake(useKVMatmul ? TE_METAL_Q4_MATMUL_THREADS : (useKVLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS), 1, 1);
        const MTLSize outputThreads = MTLSizeMake(useOutputMatmul ? TE_METAL_Q4_MATMUL_THREADS : (useOutputLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS), 1, 1);
        const MTLSize gateThreads = MTLSizeMake(useGateMatmul ? TE_METAL_Q4_MATMUL_THREADS : (useGateLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS), 1, 1);
        const MTLSize downThreads = MTLSizeMake(useDownMatmul ? TE_METAL_Q4_MATMUL_THREADS : (useDownLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS), 1, 1);
        const MTLSize copyGroups = MTLSizeMake((hidden_values + TE_METAL_VECTOR_THREADS - 1u) / TE_METAL_VECTOR_THREADS, 1, 1);
        const MTLSize normGroups = MTLSizeMake(batch, 1, 1);
        const MTLSize qGroups = useQMatmul
            ? MTLSizeMake((batch + qBatchTile - 1u) / qBatchTile, (hidden + qRowTile - 1u) / qRowTile, 1)
            : MTLSizeMake((hidden + qRowTile - 1u) / qRowTile, (batch + qBatchTile - 1u) / qBatchTile, 1);
        const MTLSize kvGroups = useKVMatmul
            ? MTLSizeMake((batch + kvBatchTile - 1u) / kvBatchTile, (kv + kvRowTile - 1u) / kvRowTile, 1)
            : MTLSizeMake((kv + kvRowTile - 1u) / kvRowTile, (batch + kvBatchTile - 1u) / kvBatchTile, 1);
        const MTLSize outputGroups = useOutputMatmul
            ? MTLSizeMake((batch + outputBatchTile - 1u) / outputBatchTile, (hidden + outputRowTile - 1u) / outputRowTile, 1)
            : MTLSizeMake((hidden + outputRowTile - 1u) / outputRowTile, (batch + outputBatchTile - 1u) / outputBatchTile, 1);
        const MTLSize gateGroups = useGateMatmul
            ? MTLSizeMake((batch + gateBatchTile - 1u) / gateBatchTile, (ffn + gateRowTile - 1u) / gateRowTile, 1)
            : MTLSizeMake((ffn + gateRowTile - 1u) / gateRowTile, (batch + gateBatchTile - 1u) / gateBatchTile, 1);
        const MTLSize downGroups = useDownMatmul
            ? MTLSizeMake((batch + downBatchTile - 1u) / downBatchTile, (hidden + downRowTile - 1u) / downRowTile, 1)
            : MTLSizeMake((hidden + downRowTile - 1u) / downRowTile, (batch + downBatchTile - 1u) / downBatchTile, 1);
        const MTLSize ffnVectorGroups = MTLSizeMake((ffn_values + TE_METAL_VECTOR_THREADS - 1u) / TE_METAL_VECTOR_THREADS, 1, 1);
        const MTLSize biasGroups = MTLSizeMake(((NSUInteger)batch * max_bias_items + TE_METAL_VECTOR_THREADS - 1u) / TE_METAL_VECTOR_THREADS, 1, 1);
        const MTLSize attentionGroups = useFlashGqa ? MTLSizeMake(batch, kv_heads, 1) : MTLSizeMake(batch, heads, 1);
        const bool stageProfile = te_metal_prefill_stage_profile_enabled();
        auto finishPrefillStage = [&](const char *stage, size_t layer, bool reopen) -> te_status {
            if (!stageProfile) {
                return TE_STATUS_OK;
            }
            [encoder endEncoding];
            const double stageStart = te_metal_now_ms();
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            const double cpu_ms = te_metal_now_ms() - stageStart;
            double gpu_ms = 0.0;
            const CFTimeInterval gpu_start = commandBuffer.GPUStartTime;
            const CFTimeInterval gpu_end = commandBuffer.GPUEndTime;
            if (gpu_end > gpu_start) {
                gpu_ms = (gpu_end - gpu_start) * 1000.0;
            }
            fprintf(
                stderr,
                "tinyengine_metal_stage_profile: scope=prefill_all_layers layer=%zu stage=%s cpu_ms=%.3f gpu_ms=%.3f\n",
                layer,
                stage,
                cpu_ms,
                gpu_ms);
            if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
                return TE_STATUS_RUNTIME_ERROR;
            }
            if (!reopen) {
                encoder = nil;
                commandBuffer = nil;
                return TE_STATUS_OK;
            }
            commandBuffer = [runtime.queue commandBuffer];
            if (commandBuffer == nil) {
                return TE_STATUS_RUNTIME_ERROR;
            }
            encoder = [commandBuffer computeCommandEncoder];
            return encoder == nil ? TE_STATUS_RUNTIME_ERROR : TE_STATUS_OK;
        };
#define TE_PREFILL_STAGE_PROFILE(stage_name, reopen_encoder) do { \
            if (stageProfile) { \
                status = finishPrefillStage((stage_name), layer, (reopen_encoder)); \
                if (status != TE_STATUS_OK) { \
                    return status; \
                } \
            } \
        } while (0)

        for (size_t layer = 0; layer < layers; ++layer) {
            id<MTLBuffer> hiddenIn = (layer % 2u) == 0u ? runtime.inputBuffer : runtime.output3Buffer;
            id<MTLBuffer> hiddenOut = (layer % 2u) == 0u ? runtime.output3Buffer : runtime.inputBuffer;
            const NSUInteger normOffset = (NSUInteger)(layer * weight_bytes);
            const NSUInteger qBiasOffset = (NSUInteger)(layer * q_bias_bytes);
            const NSUInteger kvBiasOffset = (NSUInteger)(layer * kv_bias_bytes);
            const NSUInteger cacheOffset = (NSUInteger)(layer * kv_bytes);
            const NSUInteger halfCacheOffset = (NSUInteger)(layer * batch * kv * sizeof(uint16_t));

            [encoder setComputePipelineState:runtime.vectorCopyPipeline];
            [encoder setBuffer:hiddenIn offset:0 atIndex:0];
            [encoder setBuffer:hiddenOut offset:0 atIndex:1];
            [encoder setBytes:&hidden_len length:sizeof(hidden_len) atIndex:2];
            [encoder dispatchThreadgroups:copyGroups threadsPerThreadgroup:vectorThreads];
            TE_PREFILL_STAGE_PROFILE("residual_copy", true);

            [encoder setComputePipelineState:runtime.rmsnormPipeline];
            [encoder setBuffer:hiddenIn offset:0 atIndex:0];
            [encoder setBuffer:runtime.weightBuffer offset:normOffset atIndex:1];
            [encoder setBuffer:runtime.normBuffer offset:0 atIndex:2];
            [encoder setBytes:norm_dims length:sizeof(norm_dims) atIndex:3];
            [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
            [encoder dispatchThreadgroups:normGroups threadsPerThreadgroup:normThreads];
            TE_PREFILL_STAGE_PROFILE("attn_norm", true);

            [encoder setComputePipelineState:useQMatmul ? runtime.q4MatmulPipeline : (useQLlama ? runtime.q4BatchLlamaPipeline : runtime.q4BatchPipeline)];
            [encoder setBuffer:runtime.normBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)q_offsets[layer] atIndex:1];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
            [encoder setBytes:q_dims length:sizeof(q_dims) atIndex:3];
            if (useQMatmul) {
                [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
            }
            [encoder dispatchThreadgroups:qGroups threadsPerThreadgroup:qThreads];
            TE_PREFILL_STAGE_PROFILE("q_proj", true);

            if (useKVMatmul) {
                [encoder setComputePipelineState:runtime.q4MatmulPipeline];
                [encoder setBuffer:runtime.normBuffer offset:0 atIndex:0];
                [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)k_offsets[layer] atIndex:1];
                [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:2];
                [encoder setBytes:kv_dims length:sizeof(kv_dims) atIndex:3];
                [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
                [encoder dispatchThreadgroups:kvGroups threadsPerThreadgroup:kvThreads];

                [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)v_offsets[layer] atIndex:1];
                [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
                [encoder setBytes:kv_dims length:sizeof(kv_dims) atIndex:3];
                [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
                [encoder dispatchThreadgroups:kvGroups threadsPerThreadgroup:kvThreads];
                TE_PREFILL_STAGE_PROFILE("kv_proj", true);
            } else {
                [encoder setComputePipelineState:useKVLlama ? runtime.q4BatchPairLlamaPipeline : runtime.q4BatchPairPipeline];
                [encoder setBuffer:runtime.normBuffer offset:0 atIndex:0];
                [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)k_offsets[layer] atIndex:1];
                [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)v_offsets[layer] atIndex:2];
                [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:3];
                [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:4];
                [encoder setBytes:kv_dims length:sizeof(kv_dims) atIndex:5];
                [encoder dispatchThreadgroups:kvGroups threadsPerThreadgroup:kvThreads];
                TE_PREFILL_STAGE_PROFILE("kv_proj", true);
            }

            [encoder setComputePipelineState:useFlashHalfKv ? runtime.qkvBiasRopeCacheBatchHalfPipeline : runtime.qkvBiasRopeCacheBatchPipeline];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
            [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
            [encoder setBuffer:runtime.biasBuffer offset:qBiasOffset atIndex:3];
            [encoder setBuffer:runtime.bias2Buffer offset:kvBiasOffset atIndex:4];
            [encoder setBuffer:runtime.bias3Buffer offset:kvBiasOffset atIndex:5];
            [encoder setBuffer:runtime.ropeCosBuffer offset:0 atIndex:6];
            [encoder setBuffer:runtime.ropeSinBuffer offset:0 atIndex:7];
            [encoder setBuffer:runtime.keyCacheBuffer offset:cacheOffset atIndex:8];
            [encoder setBuffer:runtime.valueCacheBuffer offset:cacheOffset atIndex:9];
            if (useFlashHalfKv) {
                [encoder setBuffer:runtime.keyCacheHalfBuffer offset:halfCacheOffset atIndex:10];
                [encoder setBuffer:runtime.valueCacheHalfBuffer offset:halfCacheOffset atIndex:11];
                [encoder setBytes:bias_dims length:sizeof(bias_dims) atIndex:12];
            } else {
                [encoder setBytes:bias_dims length:sizeof(bias_dims) atIndex:10];
            }
            [encoder dispatchThreadgroups:biasGroups threadsPerThreadgroup:vectorThreads];
            TE_PREFILL_STAGE_PROFILE("qkv_bias_rope", true);

            [encoder setComputePipelineState:useFlashGqa ? runtime.attentionFlashGqaPipeline : (useFlashLongTile ? runtime.attentionFlashLongPipeline : (useFlashHalfKv ? runtime.attentionFlashHalfPipeline : (useFlashAttention ? runtime.attentionFlashPipeline : runtime.attentionDecodePipeline)))];
            id<MTLBuffer> attentionKeyCache = useFlashHalfKv ? runtime.keyCacheHalfBuffer : runtime.keyCacheBuffer;
            id<MTLBuffer> attentionValueCache = useFlashHalfKv ? runtime.valueCacheHalfBuffer : runtime.valueCacheBuffer;
            const NSUInteger attentionCacheOffset = useFlashHalfKv ? halfCacheOffset : cacheOffset;
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
            [encoder setBuffer:attentionKeyCache offset:attentionCacheOffset atIndex:1];
            [encoder setBuffer:attentionValueCache offset:attentionCacheOffset atIndex:2];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:3];
            [encoder setBytes:attention_dims length:sizeof(attention_dims) atIndex:4];
            if (!useFlashAttention) {
                [encoder setThreadgroupMemoryLength:batch * sizeof(float) atIndex:0];
            }
            [encoder dispatchThreadgroups:attentionGroups threadsPerThreadgroup:attentionThreads];
            TE_PREFILL_STAGE_PROFILE(useFlashGqa ? "flash_attention_gqa" : (useFlashLongTile ? "flash_attention_t128" : (useFlashHalfKv ? "flash_attention_hkv" : (useFlashAttention ? "flash_attention" : "attention"))), true);

            [encoder setComputePipelineState:useOutputMatmul ? runtime.q4MatmulPipeline : (useOutputLlama ? runtime.q4BatchLlamaPipeline : runtime.q4BatchPipeline)];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)output_offsets[layer] atIndex:1];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
            [encoder setBytes:output_dims length:sizeof(output_dims) atIndex:3];
            if (useOutputMatmul) {
                [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
            }
            [encoder dispatchThreadgroups:outputGroups threadsPerThreadgroup:outputThreads];
            TE_PREFILL_STAGE_PROFILE("output_proj", true);

            [encoder setComputePipelineState:runtime.addPipeline];
            [encoder setBuffer:hiddenOut offset:0 atIndex:0];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:1];
            [encoder setBuffer:hiddenOut offset:0 atIndex:2];
            [encoder setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
            [encoder dispatchThreadgroups:copyGroups threadsPerThreadgroup:vectorThreads];
            TE_PREFILL_STAGE_PROFILE("attn_residual", true);

            [encoder setComputePipelineState:runtime.rmsnormPipeline];
            [encoder setBuffer:hiddenOut offset:0 atIndex:0];
            [encoder setBuffer:runtime.weight2Buffer offset:normOffset atIndex:1];
            [encoder setBuffer:runtime.normBuffer offset:0 atIndex:2];
            [encoder setBytes:norm_dims length:sizeof(norm_dims) atIndex:3];
            [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
            [encoder dispatchThreadgroups:normGroups threadsPerThreadgroup:normThreads];
            TE_PREFILL_STAGE_PROFILE("ffn_norm", true);

            if (useFfnPairSwiglu) {
                [encoder setComputePipelineState:runtime.q4MatmulPairSwigluHalfPipeline];
                [encoder setBuffer:runtime.normBuffer offset:0 atIndex:0];
                [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)gate_offsets[layer] atIndex:1];
                [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)up_offsets[layer] atIndex:2];
                [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:3];
                [encoder setBytes:gate_dims length:sizeof(gate_dims) atIndex:4];
                [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_PAIR_SWIGLU_SHMEM atIndex:0];
                [encoder dispatchThreadgroups:gateGroups threadsPerThreadgroup:gateThreads];
                TE_PREFILL_STAGE_PROFILE("gate_up_swiglu_pair_h", true);
            } else if (useGateMatmul) {
                [encoder setComputePipelineState:useFfnGateHalf ? runtime.q4MatmulStoreHalfPipeline : runtime.q4MatmulPipeline];
                [encoder setBuffer:runtime.normBuffer offset:0 atIndex:0];
                [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)gate_offsets[layer] atIndex:1];
                [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
                [encoder setBytes:gate_dims length:sizeof(gate_dims) atIndex:3];
                [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
                [encoder dispatchThreadgroups:gateGroups threadsPerThreadgroup:gateThreads];
                TE_PREFILL_STAGE_PROFILE(useFfnGateHalf ? "gate_proj_h" : "gate_proj", true);

                [encoder setComputePipelineState:useFfnGateHalf
                    ? runtime.q4MatmulSwigluGateHalfPipeline
                    : (useFfnHalf ? runtime.q4MatmulSwigluHalfPipeline : runtime.q4MatmulSwigluPipeline)];
                [encoder setBuffer:runtime.normBuffer offset:0 atIndex:0];
                [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)up_offsets[layer] atIndex:1];
                [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
                [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:3];
                [encoder setBytes:gate_dims length:sizeof(gate_dims) atIndex:4];
                [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
                [encoder dispatchThreadgroups:gateGroups threadsPerThreadgroup:gateThreads];
                TE_PREFILL_STAGE_PROFILE(useFfnGateHalf ? "up_swiglu_proj_gate_h" : (useFfnHalf ? "up_swiglu_proj_h" : "up_swiglu_proj"), true);
            } else {
                [encoder setComputePipelineState:useGateLlama ? runtime.q4BatchPairLlamaPipeline : runtime.q4BatchPairPipeline];
                [encoder setBuffer:runtime.normBuffer offset:0 atIndex:0];
                [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)gate_offsets[layer] atIndex:1];
                [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)up_offsets[layer] atIndex:2];
                [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:3];
                [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:4];
                [encoder setBytes:gate_dims length:sizeof(gate_dims) atIndex:5];
                [encoder dispatchThreadgroups:gateGroups threadsPerThreadgroup:gateThreads];
                TE_PREFILL_STAGE_PROFILE("gate_up_proj", true);
            }

            if (!useGateMatmul) {
                [encoder setComputePipelineState:runtime.swigluPipeline];
                [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
                [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
                [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
                [encoder setBytes:&ffn_len length:sizeof(ffn_len) atIndex:3];
                [encoder dispatchThreadgroups:ffnVectorGroups threadsPerThreadgroup:vectorThreads];
                TE_PREFILL_STAGE_PROFILE("swiglu", true);
            }

            if (te_metal_use_q4_ffn_down_add(runtime, useFfnHalf, batch)) {
                [encoder setComputePipelineState:runtime.q4MatmulHalfInputAddPipeline];
                [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:0];
                [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)down_offsets[layer] atIndex:1];
                [encoder setBuffer:hiddenOut offset:0 atIndex:2];
                [encoder setBuffer:hiddenOut offset:0 atIndex:3];
                [encoder setBytes:down_dims length:sizeof(down_dims) atIndex:4];
                [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
                [encoder dispatchThreadgroups:downGroups threadsPerThreadgroup:downThreads];
                TE_PREFILL_STAGE_PROFILE("down_proj_h_add", layer + 1u < layers);
            } else {
                [encoder setComputePipelineState:useDownMatmul ? (useFfnHalf ? runtime.q4MatmulHalfInputPipeline : runtime.q4MatmulPipeline) : (useDownLlama ? runtime.q4BatchLlamaPipeline : runtime.q4BatchPipeline)];
                [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:0];
                [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)down_offsets[layer] atIndex:1];
                [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
                [encoder setBytes:down_dims length:sizeof(down_dims) atIndex:3];
                if (useDownMatmul) {
                    [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
                }
                [encoder dispatchThreadgroups:downGroups threadsPerThreadgroup:downThreads];
                TE_PREFILL_STAGE_PROFILE(useFfnHalf ? "down_proj_h" : "down_proj", true);

                [encoder setComputePipelineState:runtime.addPipeline];
                [encoder setBuffer:hiddenOut offset:0 atIndex:0];
                [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:1];
                [encoder setBuffer:hiddenOut offset:0 atIndex:2];
                [encoder setBytes:&hidden_len length:sizeof(hidden_len) atIndex:3];
                [encoder dispatchThreadgroups:copyGroups threadsPerThreadgroup:vectorThreads];
                TE_PREFILL_STAGE_PROFILE("ffn_residual", layer + 1u < layers);
            }
        }
#undef TE_PREFILL_STAGE_PROFILE
        if (!stageProfile) {
            [encoder endEncoding];

            const double profileStart = te_metal_now_ms();
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            te_metal_profile_record(
                TE_METAL_PROFILE_PREFILL_ALL_LAYERS,
                commandBuffer,
                te_metal_now_ms() - profileStart);
            if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
                return TE_STATUS_RUNTIME_ERROR;
            }
        }

        id<MTLBuffer> finalHidden = (layers % 2u) == 0u ? runtime.inputBuffer : runtime.output3Buffer;
        memcpy(out, [finalHidden contents], hidden_bytes);
        const float *key_contents = (const float *)[runtime.keyCacheBuffer contents];
        const float *value_contents = (const float *)[runtime.valueCacheBuffer contents];
        for (size_t layer = 0; layer < layers; ++layer) {
            memcpy(
                key_cache + layer * context_tokens * kv,
                key_contents + layer * batch * kv,
                kv_bytes);
            memcpy(
                value_cache + layer * context_tokens * kv,
                value_contents + layer * batch * kv,
                kv_bytes);
        }
        return TE_STATUS_OK;
    }
}
