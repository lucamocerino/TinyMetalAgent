#include "metal_backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TE_METAL_GGML_TYPE_Q4_0 2u
#define TE_METAL_GGML_TYPE_Q8_0 8u

static NSString *const TE_METAL_SOURCE =
#include "metal/metal_kernels.metal.inc"
;

// Runtime state and tuning gates stay in this translation unit because most
// dispatch entry points use the same static helpers and cached Metal objects.
#include "metal/metal_backend_runtime.mm.inc"

te_status te_metal_attention_qk_mma_f32(
    const float *query,
    const float *key_t,
    size_t q_rows,
    size_t k_cols,
    size_t head_dim,
    float *out
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (query == nullptr || key_t == nullptr || out == nullptr || q_rows == 0 || k_cols == 0 || head_dim == 0) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (q_rows != 8u || k_cols != 8u || head_dim != 64u) {
        return TE_STATUS_UNSUPPORTED;
    }
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(TE_METAL_MUTEX);
        te_status status = te_metal_init_locked();
        if (status != TE_STATUS_OK) {
            return status;
        }
        TEMetalRuntime *runtime = TE_METAL_RUNTIME;
        const size_t query_bytes = q_rows * head_dim * sizeof(float);
        const size_t key_bytes = head_dim * k_cols * sizeof(float);
        const size_t out_bytes = q_rows * k_cols * sizeof(float);
        if (runtime.inputBuffer == nil || runtime.inputCapacity < query_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:query_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = query_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < key_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:key_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = key_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < out_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:out_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = out_bytes;
        }
        if (runtime.inputBuffer == nil || runtime.output2Buffer == nil || runtime.outputBuffer == nil ||
            runtime.attentionQkMmaPipeline == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }
        memcpy([runtime.inputBuffer contents], query, query_bytes);
        memcpy([runtime.output2Buffer contents], key_t, key_bytes);

        const uint32_t dims[3] = {(uint32_t)q_rows, (uint32_t)k_cols, (uint32_t)head_dim};
        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        [encoder setComputePipelineState:runtime.attentionQkMmaPipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBytes:dims length:sizeof(dims) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(TE_METAL_FLASH_ATTENTION_THREADS, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        memcpy(out, [runtime.outputBuffer contents], out_bytes);
        return TE_STATUS_OK;
    }
}

te_status te_metal_attention_tile_mma_f32(
    const float *query,
    const float *key_t,
    const float *value,
    size_t q_rows,
    size_t k_cols,
    size_t head_dim,
    float *out
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (query == nullptr || key_t == nullptr || value == nullptr || out == nullptr ||
        q_rows == 0 || k_cols == 0 || head_dim == 0) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (q_rows != 8u || k_cols != 8u || head_dim != 64u) {
        return TE_STATUS_UNSUPPORTED;
    }

    @autoreleasepool {
        std::lock_guard<std::mutex> lock(TE_METAL_MUTEX);
        te_status status = te_metal_init_locked();
        if (status != TE_STATUS_OK) {
            return status;
        }
        TEMetalRuntime *runtime = TE_METAL_RUNTIME;
        const size_t query_bytes = q_rows * head_dim * sizeof(float);
        const size_t key_bytes = head_dim * k_cols * sizeof(float);
        const size_t value_bytes = k_cols * head_dim * sizeof(float);
        const size_t out_bytes = q_rows * head_dim * sizeof(float);
        if (runtime.inputBuffer == nil || runtime.inputCapacity < query_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:query_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = query_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < key_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:key_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = key_bytes;
        }
        if (runtime.scratchBuffer == nil || runtime.scratchCapacity < value_bytes) {
            runtime.scratchBuffer = [runtime.device newBufferWithLength:value_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.scratchCapacity = value_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < out_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:out_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = out_bytes;
        }
        if (runtime.inputBuffer == nil || runtime.output2Buffer == nil || runtime.scratchBuffer == nil ||
            runtime.outputBuffer == nil || runtime.attentionTileMmaPipeline == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }
        memcpy([runtime.inputBuffer contents], query, query_bytes);
        memcpy([runtime.output2Buffer contents], key_t, key_bytes);
        memcpy([runtime.scratchBuffer contents], value, value_bytes);

        const uint32_t dims[3] = {(uint32_t)q_rows, (uint32_t)k_cols, (uint32_t)head_dim};
        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        [encoder setComputePipelineState:runtime.attentionTileMmaPipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:3];
        [encoder setBytes:dims length:sizeof(dims) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(TE_METAL_FLASH_ATTENTION_THREADS, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        memcpy(out, [runtime.outputBuffer contents], out_bytes);
        return TE_STATUS_OK;
    }
}

te_status te_metal_attention_stream_mma_f32(
    const float *query,
    const float *key_t,
    const float *value,
    size_t q_rows,
    size_t k_cols,
    size_t head_dim,
    float *out
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (query == nullptr || key_t == nullptr || value == nullptr || out == nullptr ||
        q_rows == 0 || k_cols == 0 || head_dim == 0) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (q_rows != 8u || k_cols == 0u || (k_cols % 8u) != 0u || head_dim != 64u) {
        return TE_STATUS_UNSUPPORTED;
    }

    @autoreleasepool {
        std::lock_guard<std::mutex> lock(TE_METAL_MUTEX);
        te_status status = te_metal_init_locked();
        if (status != TE_STATUS_OK) {
            return status;
        }
        TEMetalRuntime *runtime = TE_METAL_RUNTIME;
        const size_t query_bytes = q_rows * head_dim * sizeof(float);
        const size_t key_bytes = head_dim * k_cols * sizeof(float);
        const size_t value_bytes = k_cols * head_dim * sizeof(float);
        const size_t out_bytes = q_rows * head_dim * sizeof(float);
        if (runtime.inputBuffer == nil || runtime.inputCapacity < query_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:query_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = query_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < key_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:key_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = key_bytes;
        }
        if (runtime.scratchBuffer == nil || runtime.scratchCapacity < value_bytes) {
            runtime.scratchBuffer = [runtime.device newBufferWithLength:value_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.scratchCapacity = value_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < out_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:out_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = out_bytes;
        }
        if (runtime.inputBuffer == nil || runtime.output2Buffer == nil || runtime.scratchBuffer == nil ||
            runtime.outputBuffer == nil || runtime.attentionStreamMmaPipeline == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }
        memcpy([runtime.inputBuffer contents], query, query_bytes);
        memcpy([runtime.output2Buffer contents], key_t, key_bytes);
        memcpy([runtime.scratchBuffer contents], value, value_bytes);

        const uint32_t dims[3] = {(uint32_t)q_rows, (uint32_t)k_cols, (uint32_t)head_dim};
        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        [encoder setComputePipelineState:runtime.attentionStreamMmaPipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:3];
        [encoder setBytes:dims length:sizeof(dims) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(TE_METAL_FLASH_ATTENTION_THREADS, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        memcpy(out, [runtime.outputBuffer contents], out_bytes);
        return TE_STATUS_OK;
    }
}

te_status te_metal_attention_causal_mma_f32(
    const float *query,
    const float *key_t,
    const float *value,
    size_t q_rows,
    size_t k_cols,
    size_t head_dim,
    size_t q_base,
    float *out
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (query == nullptr || key_t == nullptr || value == nullptr || out == nullptr ||
        q_rows == 0 || k_cols == 0 || head_dim == 0) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (q_rows != 8u || k_cols == 0u || (k_cols % 8u) != 0u || head_dim != 64u || q_base > k_cols) {
        return TE_STATUS_UNSUPPORTED;
    }

    @autoreleasepool {
        std::lock_guard<std::mutex> lock(TE_METAL_MUTEX);
        te_status status = te_metal_init_locked();
        if (status != TE_STATUS_OK) {
            return status;
        }
        TEMetalRuntime *runtime = TE_METAL_RUNTIME;
        const size_t query_bytes = q_rows * head_dim * sizeof(float);
        const size_t key_bytes = head_dim * k_cols * sizeof(float);
        const size_t value_bytes = k_cols * head_dim * sizeof(float);
        const size_t out_bytes = q_rows * head_dim * sizeof(float);
        if (runtime.inputBuffer == nil || runtime.inputCapacity < query_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:query_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = query_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < key_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:key_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = key_bytes;
        }
        if (runtime.scratchBuffer == nil || runtime.scratchCapacity < value_bytes) {
            runtime.scratchBuffer = [runtime.device newBufferWithLength:value_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.scratchCapacity = value_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < out_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:out_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = out_bytes;
        }
        if (runtime.inputBuffer == nil || runtime.output2Buffer == nil || runtime.scratchBuffer == nil ||
            runtime.outputBuffer == nil || runtime.attentionFlashMmaTransposePipeline == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }
        memcpy([runtime.inputBuffer contents], query, query_bytes);
        memcpy([runtime.output2Buffer contents], key_t, key_bytes);
        memcpy([runtime.scratchBuffer contents], value, value_bytes);

        const uint32_t dims[7] = {
            (uint32_t)q_rows,
            (uint32_t)q_base,
            1u,
            1u,
            (uint32_t)head_dim,
            (uint32_t)k_cols,
            (uint32_t)k_cols};
        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        [encoder setComputePipelineState:runtime.attentionFlashMmaTransposePipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:3];
        [encoder setBytes:dims length:sizeof(dims) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(TE_METAL_FLASH_ATTENTION_THREADS, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        memcpy(out, [runtime.outputBuffer contents], out_bytes);
        return TE_STATUS_OK;
    }
}

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
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (mapping == nullptr || input == nullptr || out == nullptr || batch == 0 || cols == 0 || rows == 0) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (batch > UINT32_MAX || cols > UINT32_MAX || rows > UINT32_MAX ||
        tensor_offset > mapping_len || ggml_type != TE_METAL_GGML_TYPE_Q4_0 ||
        batch > SIZE_MAX / cols || batch * cols > SIZE_MAX / sizeof(float) ||
        batch > SIZE_MAX / rows || batch * rows > SIZE_MAX / sizeof(float)) {
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

        const size_t blocks_per_row = (cols + 31u) / 32u;
        const size_t row_bytes = blocks_per_row * 18u;
        const size_t input_values = batch * cols;
        const size_t output_values = batch * rows;
        const size_t input_bytes = input_values * sizeof(float);
        const size_t output_bytes = output_values * sizeof(float);
        if (row_bytes > UINT32_MAX || tensor_offset > UINT64_MAX - row_bytes * rows) {
            return TE_STATUS_UNSUPPORTED;
        }

        if (runtime.inputBuffer == nil || runtime.inputCapacity < input_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:input_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = input_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < output_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:output_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = output_bytes;
        }
        if (runtime.dimsBuffer == nil) {
            runtime.dimsBuffer = [runtime.device newBufferWithLength:4u * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
        }
        if (runtime.inputBuffer == nil || runtime.outputBuffer == nil || runtime.dimsBuffer == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }

        uint32_t dims[4] = {(uint32_t)rows, (uint32_t)cols, (uint32_t)row_bytes, (uint32_t)batch};
        memcpy([runtime.inputBuffer contents], input, input_bytes);
        memcpy([runtime.dimsBuffer contents], dims, sizeof(dims));

        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (commandBuffer == nil || encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        const bool useQ4Matmul = te_metal_use_q4_matmul(cols, batch);
        const bool useQ4BatchLlama = !useQ4Matmul && te_metal_use_llama_q4_batch(cols);
        const NSUInteger rowTile = useQ4Matmul
            ? TE_METAL_Q4_MATMUL_ROW_TILE
            : (useQ4BatchLlama ? TE_METAL_Q4_BATCH_LLAMA_ROW_TILE : TE_METAL_Q4_BATCH_ROW_TILE);
        const NSUInteger batchTile = useQ4Matmul
            ? TE_METAL_Q4_MATMUL_BATCH_TILE
            : (useQ4BatchLlama ? TE_METAL_Q4_BATCH_LLAMA_TILE : TE_METAL_Q4_BATCH_TILE);
        [encoder setComputePipelineState:useQ4Matmul
            ? runtime.q4MatmulPipeline
            : (useQ4BatchLlama ? runtime.q4BatchLlamaPipeline : runtime.q4BatchPipeline)];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)tensor_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.dimsBuffer offset:0 atIndex:3];

        if (useQ4Matmul) {
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
        }
        const MTLSize threads = MTLSizeMake(
            useQ4Matmul ? TE_METAL_Q4_MATMUL_THREADS : (useQ4BatchLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS),
            1,
            1);
        const MTLSize groups = useQ4Matmul
            ? MTLSizeMake((batch + batchTile - 1u) / batchTile, (rows + rowTile - 1u) / rowTile, 1)
            : MTLSizeMake((rows + rowTile - 1u) / rowTile, (batch + batchTile - 1u) / batchTile, 1);
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        [encoder endEncoding];
        const double profileStart = te_metal_now_ms();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        te_metal_profile_record(
            TE_METAL_PROFILE_MATVEC_BATCH,
            commandBuffer,
            te_metal_now_ms() - profileStart);
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }

        memcpy(out, [runtime.outputBuffer contents], output_bytes);
        return TE_STATUS_OK;
    }
}

te_status te_metal_matvec_argmax_f32(
    const void *mapping,
    size_t mapping_len,
    uint64_t tensor_offset,
    uint32_t ggml_type,
    const float *input,
    size_t cols,
    size_t rows,
    uint32_t *out_index
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (mapping == nullptr || input == nullptr || out_index == nullptr || cols == 0 || rows == 0) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (cols > UINT32_MAX || rows > UINT32_MAX || tensor_offset > mapping_len ||
        rows > SIZE_MAX / cols || rows * cols < 1000000u) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (ggml_type != TE_METAL_GGML_TYPE_Q4_0 && ggml_type != TE_METAL_GGML_TYPE_Q8_0) {
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

        const size_t blocks_per_row = (cols + 31u) / 32u;
        const size_t row_bytes = ggml_type == TE_METAL_GGML_TYPE_Q4_0 ? blocks_per_row * 18u : blocks_per_row * 34u;
        if (row_bytes > UINT32_MAX || tensor_offset > UINT64_MAX - row_bytes * rows) {
            return TE_STATUS_UNSUPPORTED;
        }
        const size_t input_bytes = cols * sizeof(float);
        const size_t output_bytes = rows * sizeof(float);
        const size_t argmax_blocks = (rows + TE_METAL_ARGMAX_THREADS - 1u) / TE_METAL_ARGMAX_THREADS;
        const size_t block_value_bytes = argmax_blocks * sizeof(float);
        const size_t block_index_bytes = argmax_blocks * sizeof(uint32_t);
        if (argmax_blocks == 0 || argmax_blocks > UINT32_MAX) {
            return TE_STATUS_UNSUPPORTED;
        }

        if (runtime.inputBuffer == nil || runtime.inputCapacity < input_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:input_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = input_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < output_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:output_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = output_bytes;
        }
        if (runtime.scratchBuffer == nil || runtime.scratchCapacity < block_value_bytes) {
            runtime.scratchBuffer = [runtime.device newBufferWithLength:block_value_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.scratchCapacity = block_value_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < block_index_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:block_index_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = block_index_bytes;
        }
        if (runtime.dimsBuffer == nil) {
            runtime.dimsBuffer = [runtime.device newBufferWithLength:4u * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
        }
        if (runtime.scalarBuffer == nil) {
            runtime.scalarBuffer = [runtime.device newBufferWithLength:sizeof(uint32_t)
                                                               options:MTLResourceStorageModeShared];
        }
        if (runtime.inputBuffer == nil || runtime.outputBuffer == nil || runtime.scratchBuffer == nil ||
            runtime.output2Buffer == nil || runtime.dimsBuffer == nil || runtime.scalarBuffer == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }

        uint32_t dims[3] = {(uint32_t)rows, (uint32_t)cols, (uint32_t)row_bytes};
        const uint32_t rows_u32 = (uint32_t)rows;
        const uint32_t argmax_blocks_u32 = (uint32_t)argmax_blocks;
        memcpy([runtime.inputBuffer contents], input, input_bytes);
        memcpy([runtime.dimsBuffer contents], dims, sizeof(dims));

        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        const bool useQ4Llama = ggml_type == TE_METAL_GGML_TYPE_Q4_0 && te_metal_use_llama_q4(cols);
        const bool useQ8Llama = ggml_type == TE_METAL_GGML_TYPE_Q8_0 && te_metal_use_llama_q8(cols);
        if (ggml_type == TE_METAL_GGML_TYPE_Q4_0) {
            [encoder setComputePipelineState:useQ4Llama ? runtime.q4LlamaPipeline : runtime.q4Pipeline];
        } else {
            [encoder setComputePipelineState:useQ8Llama ? runtime.q8LlamaPipeline : runtime.q8Pipeline];
        }
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)tensor_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.dimsBuffer offset:0 atIndex:3];
        if (useQ8Llama) {
            [encoder setThreadgroupMemoryLength:TE_METAL_Q8_LLAMA_SHMEM atIndex:0];
        }
        const MTLSize matvecThreads = useQ4Llama
            ? MTLSizeMake(TE_METAL_Q4_LLAMA_THREADS, 1, 1)
            : MTLSizeMake(useQ8Llama ? TE_METAL_Q8_LLAMA_THREADS : 128, 1, 1);
        const MTLSize matvecGroups = ggml_type == TE_METAL_GGML_TYPE_Q4_0
            ? MTLSizeMake(
                (rows + (useQ4Llama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE) - 1u) /
                    (useQ4Llama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE),
                1,
                1)
            : MTLSizeMake((rows + (useQ8Llama ? TE_METAL_Q8_LLAMA_ROW_TILE : 1u) - 1u) /
                    (useQ8Llama ? TE_METAL_Q8_LLAMA_ROW_TILE : 1u),
                1,
                1);
        [encoder dispatchThreadgroups:matvecGroups threadsPerThreadgroup:matvecThreads];
        [encoder endEncoding];

        encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        [encoder setComputePipelineState:runtime.argmaxBlocksPipeline];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:2];
        [encoder setBytes:&rows_u32 length:sizeof(rows_u32) atIndex:3];
        const MTLSize argmaxThreads = MTLSizeMake(TE_METAL_ARGMAX_THREADS, 1, 1);
        const MTLSize argmaxGroups = MTLSizeMake(argmax_blocks, 1, 1);
        [encoder dispatchThreadgroups:argmaxGroups threadsPerThreadgroup:argmaxThreads];
        [encoder endEncoding];

        encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        [encoder setComputePipelineState:runtime.argmaxFinishPipeline];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.scalarBuffer offset:0 atIndex:2];
        [encoder setBytes:&argmax_blocks_u32 length:sizeof(argmax_blocks_u32) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:argmaxThreads];
        [encoder endEncoding];

        const double profileStart = te_metal_now_ms();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        te_metal_profile_record(
            TE_METAL_PROFILE_MATVEC_ARGMAX,
            commandBuffer,
            te_metal_now_ms() - profileStart);
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }

        memcpy(out_index, [runtime.scalarBuffer contents], sizeof(*out_index));
        return *out_index < rows ? TE_STATUS_OK : TE_STATUS_RUNTIME_ERROR;
    }
}

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
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (mapping == nullptr || hidden_in == nullptr || norm_weight == nullptr ||
        out_index == nullptr || cols == 0 || rows == 0 || epsilon <= 0.0f) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (cols > UINT32_MAX || rows > UINT32_MAX || tensor_offset > mapping_len ||
        rows > SIZE_MAX / cols || rows * cols < 1000000u ||
        cols > SIZE_MAX / sizeof(float) || rows > SIZE_MAX / sizeof(float)) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (ggml_type != TE_METAL_GGML_TYPE_Q4_0 && ggml_type != TE_METAL_GGML_TYPE_Q8_0) {
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

        const size_t blocks_per_row = (cols + 31u) / 32u;
        const size_t row_bytes = ggml_type == TE_METAL_GGML_TYPE_Q4_0 ? blocks_per_row * 18u : blocks_per_row * 34u;
        if (row_bytes > UINT32_MAX || tensor_offset > UINT64_MAX - row_bytes * rows) {
            return TE_STATUS_UNSUPPORTED;
        }
        const size_t input_bytes = cols * sizeof(float);
        const size_t output_bytes = rows * sizeof(float);
        const size_t argmax_blocks = (rows + TE_METAL_ARGMAX_THREADS - 1u) / TE_METAL_ARGMAX_THREADS;
        const size_t block_value_bytes = argmax_blocks * sizeof(float);
        const size_t block_index_bytes = argmax_blocks * sizeof(uint32_t);
        if (argmax_blocks == 0 || argmax_blocks > UINT32_MAX) {
            return TE_STATUS_UNSUPPORTED;
        }

        if (runtime.inputBuffer == nil || runtime.inputCapacity < input_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:input_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = input_bytes;
        }
        if (runtime.weightBuffer == nil || runtime.weightCapacity < input_bytes) {
            runtime.weightBuffer = [runtime.device newBufferWithLength:input_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.weightCapacity = input_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < output_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:output_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = output_bytes;
        }
        if (runtime.scratchBuffer == nil || runtime.scratchCapacity < block_value_bytes) {
            runtime.scratchBuffer = [runtime.device newBufferWithLength:block_value_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.scratchCapacity = block_value_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < block_index_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:block_index_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = block_index_bytes;
        }
        if (runtime.scalarBuffer == nil) {
            runtime.scalarBuffer = [runtime.device newBufferWithLength:sizeof(uint32_t)
                                                               options:MTLResourceStorageModeShared];
        }
        if (runtime.inputBuffer == nil || runtime.weightBuffer == nil || runtime.outputBuffer == nil ||
            runtime.scratchBuffer == nil || runtime.output2Buffer == nil || runtime.scalarBuffer == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }

        const uint32_t dims[3] = {(uint32_t)rows, (uint32_t)cols, (uint32_t)row_bytes};
        const uint32_t norm_dims[2] = {(uint32_t)cols, 1u};
        const uint32_t rows_u32 = (uint32_t)rows;
        const uint32_t argmax_blocks_u32 = (uint32_t)argmax_blocks;
        memcpy([runtime.inputBuffer contents], hidden_in, input_bytes);
        memcpy([runtime.weightBuffer contents], norm_weight, input_bytes);

        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        [encoder setComputePipelineState:runtime.rmsnormPipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.weightBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:2];
        [encoder setBytes:norm_dims length:sizeof(norm_dims) atIndex:3];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(TE_METAL_RMSNORM_THREADS, 1, 1)];

        const bool useQ4Llama = ggml_type == TE_METAL_GGML_TYPE_Q4_0 && te_metal_use_llama_q4(cols);
        const bool useQ8Llama = ggml_type == TE_METAL_GGML_TYPE_Q8_0 && te_metal_use_llama_q8(cols);
        if (ggml_type == TE_METAL_GGML_TYPE_Q4_0) {
            [encoder setComputePipelineState:useQ4Llama ? runtime.q4LlamaPipeline : runtime.q4Pipeline];
        } else {
            [encoder setComputePipelineState:useQ8Llama ? runtime.q8LlamaPipeline : runtime.q8Pipeline];
        }
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)tensor_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBytes:dims length:sizeof(dims) atIndex:3];
        if (useQ8Llama) {
            [encoder setThreadgroupMemoryLength:TE_METAL_Q8_LLAMA_SHMEM atIndex:0];
        }
        const MTLSize matvecThreads = useQ4Llama
            ? MTLSizeMake(TE_METAL_Q4_LLAMA_THREADS, 1, 1)
            : MTLSizeMake(useQ8Llama ? TE_METAL_Q8_LLAMA_THREADS : 128, 1, 1);
        const MTLSize matvecGroups = ggml_type == TE_METAL_GGML_TYPE_Q4_0
            ? MTLSizeMake(
                (rows + (useQ4Llama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE) - 1u) /
                    (useQ4Llama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE),
                1,
                1)
            : MTLSizeMake((rows + (useQ8Llama ? TE_METAL_Q8_LLAMA_ROW_TILE : 1u) - 1u) /
                    (useQ8Llama ? TE_METAL_Q8_LLAMA_ROW_TILE : 1u),
                1,
                1);
        [encoder dispatchThreadgroups:matvecGroups threadsPerThreadgroup:matvecThreads];
        [encoder endEncoding];

        encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        [encoder setComputePipelineState:runtime.argmaxBlocksPipeline];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:2];
        [encoder setBytes:&rows_u32 length:sizeof(rows_u32) atIndex:3];
        const MTLSize argmaxThreads = MTLSizeMake(TE_METAL_ARGMAX_THREADS, 1, 1);
        const MTLSize argmaxGroups = MTLSizeMake(argmax_blocks, 1, 1);
        [encoder dispatchThreadgroups:argmaxGroups threadsPerThreadgroup:argmaxThreads];
        [encoder endEncoding];

        encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        [encoder setComputePipelineState:runtime.argmaxFinishPipeline];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.scalarBuffer offset:0 atIndex:2];
        [encoder setBytes:&argmax_blocks_u32 length:sizeof(argmax_blocks_u32) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:argmaxThreads];
        [encoder endEncoding];

        const double profileStart = te_metal_now_ms();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        te_metal_profile_record(
            TE_METAL_PROFILE_MATVEC_ARGMAX,
            commandBuffer,
            te_metal_now_ms() - profileStart);
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }

        memcpy(out_index, [runtime.scalarBuffer contents], sizeof(*out_index));
        return *out_index < rows ? TE_STATUS_OK : TE_STATUS_RUNTIME_ERROR;
    }
}

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
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (mapping == nullptr || input == nullptr || q_out == nullptr || k_out == nullptr || v_out == nullptr ||
        hidden == 0 || kv == 0) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (hidden > UINT32_MAX || kv > UINT32_MAX ||
        q_offset > mapping_len || k_offset > mapping_len || v_offset > mapping_len ||
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

        const size_t blocks_per_row = (hidden + 31u) / 32u;
        const size_t row_bytes = blocks_per_row * 18u;
        const size_t input_bytes = hidden * sizeof(float);
        const size_t q_bytes = hidden * sizeof(float);
        const size_t kv_bytes = kv * sizeof(float);
        if (row_bytes > UINT32_MAX ||
            q_offset > UINT64_MAX - row_bytes * hidden ||
            k_offset > UINT64_MAX - row_bytes * kv ||
            v_offset > UINT64_MAX - row_bytes * kv) {
            return TE_STATUS_UNSUPPORTED;
        }

        if (runtime.inputBuffer == nil || runtime.inputCapacity < input_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:input_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = input_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < q_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:q_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = q_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < kv_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:kv_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = kv_bytes;
        }
        if (runtime.output3Buffer == nil || runtime.output3Capacity < kv_bytes) {
            runtime.output3Buffer = [runtime.device newBufferWithLength:kv_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output3Capacity = kv_bytes;
        }
        if (runtime.dimsBuffer == nil) {
            runtime.dimsBuffer = [runtime.device newBufferWithLength:4u * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
        }
        if (runtime.dims2Buffer == nil) {
            runtime.dims2Buffer = [runtime.device newBufferWithLength:4u * sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        }
        if (runtime.inputBuffer == nil || runtime.outputBuffer == nil || runtime.output2Buffer == nil ||
            runtime.output3Buffer == nil || runtime.dimsBuffer == nil || runtime.dims2Buffer == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }

        uint32_t q_dims[3] = {(uint32_t)hidden, (uint32_t)hidden, (uint32_t)row_bytes};
        uint32_t kv_dims[3] = {(uint32_t)kv, (uint32_t)hidden, (uint32_t)row_bytes};
        memcpy([runtime.inputBuffer contents], input, input_bytes);
        memcpy([runtime.dimsBuffer contents], q_dims, sizeof(q_dims));
        memcpy([runtime.dims2Buffer contents], kv_dims, sizeof(kv_dims));

        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        const bool useQ4Llama = te_metal_use_llama_q4(hidden);
        const bool useQ4PairLlama = te_metal_use_llama_q4_pair(hidden);
        const NSUInteger qTile = useQ4Llama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE;
        const NSUInteger kvTile = useQ4PairLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_PAIR_ROW_TILE;
        const MTLSize qThreads = MTLSizeMake(useQ4Llama ? TE_METAL_Q4_LLAMA_THREADS : 128, 1, 1);
        const MTLSize pairThreads = MTLSizeMake(useQ4PairLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS, 1, 1);
        const MTLSize qGroups = MTLSizeMake((hidden + qTile - 1u) / qTile, 1, 1);
        const MTLSize kvGroups = MTLSizeMake((kv + kvTile - 1u) / kvTile, 1, 1);

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        [encoder setComputePipelineState:useQ4Llama ? runtime.q4LlamaPipeline : runtime.q4Pipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)q_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.dimsBuffer offset:0 atIndex:3];
        [encoder dispatchThreadgroups:qGroups threadsPerThreadgroup:qThreads];

        [encoder setComputePipelineState:useQ4PairLlama ? runtime.q4PairLlamaPipeline : runtime.q4PairPipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)k_offset atIndex:1];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)v_offset atIndex:2];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:3];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:4];
        [encoder setBuffer:runtime.dims2Buffer offset:0 atIndex:5];
        [encoder dispatchThreadgroups:kvGroups threadsPerThreadgroup:pairThreads];
        [encoder endEncoding];

        const double profileStart = te_metal_now_ms();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        te_metal_profile_record(
            TE_METAL_PROFILE_QKV,
            commandBuffer,
            te_metal_now_ms() - profileStart);
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        memcpy(q_out, [runtime.outputBuffer contents], q_bytes);
        memcpy(k_out, [runtime.output2Buffer contents], kv_bytes);
        memcpy(v_out, [runtime.output3Buffer contents], kv_bytes);
        return TE_STATUS_OK;
    }
}

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
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (mapping == nullptr || input == nullptr || q_out == nullptr || k_out == nullptr || v_out == nullptr ||
        batch == 0 || hidden == 0 || kv == 0) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (batch > UINT32_MAX || hidden > UINT32_MAX || kv > UINT32_MAX ||
        q_offset > mapping_len || k_offset > mapping_len || v_offset > mapping_len ||
        ggml_type != TE_METAL_GGML_TYPE_Q4_0 ||
        batch > SIZE_MAX / hidden || batch * hidden > SIZE_MAX / sizeof(float) ||
        batch > SIZE_MAX / kv || batch * kv > SIZE_MAX / sizeof(float)) {
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

        const size_t blocks_per_row = (hidden + 31u) / 32u;
        const size_t row_bytes = blocks_per_row * 18u;
        const size_t input_values = batch * hidden;
        const size_t q_values = batch * hidden;
        const size_t kv_values = batch * kv;
        const size_t input_bytes = input_values * sizeof(float);
        const size_t q_bytes = q_values * sizeof(float);
        const size_t kv_bytes = kv_values * sizeof(float);
        if (row_bytes > UINT32_MAX ||
            q_offset > UINT64_MAX - row_bytes * hidden ||
            k_offset > UINT64_MAX - row_bytes * kv ||
            v_offset > UINT64_MAX - row_bytes * kv) {
            return TE_STATUS_UNSUPPORTED;
        }

        if (runtime.inputBuffer == nil || runtime.inputCapacity < input_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:input_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = input_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < q_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:q_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = q_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < kv_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:kv_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = kv_bytes;
        }
        if (runtime.output3Buffer == nil || runtime.output3Capacity < kv_bytes) {
            runtime.output3Buffer = [runtime.device newBufferWithLength:kv_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output3Capacity = kv_bytes;
        }
        if (runtime.dimsBuffer == nil) {
            runtime.dimsBuffer = [runtime.device newBufferWithLength:4u * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
        }
        if (runtime.dims2Buffer == nil) {
            runtime.dims2Buffer = [runtime.device newBufferWithLength:4u * sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        }
        if (runtime.inputBuffer == nil || runtime.outputBuffer == nil || runtime.output2Buffer == nil ||
            runtime.output3Buffer == nil || runtime.dimsBuffer == nil || runtime.dims2Buffer == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }

        uint32_t q_dims[4] = {(uint32_t)hidden, (uint32_t)hidden, (uint32_t)row_bytes, (uint32_t)batch};
        uint32_t kv_dims[4] = {(uint32_t)kv, (uint32_t)hidden, (uint32_t)row_bytes, (uint32_t)batch};
        memcpy([runtime.inputBuffer contents], input, input_bytes);
        memcpy([runtime.dimsBuffer contents], q_dims, sizeof(q_dims));
        memcpy([runtime.dims2Buffer contents], kv_dims, sizeof(kv_dims));

        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        const bool useQ4Matmul = te_metal_use_q4_matmul(hidden, batch);
        const bool useQ4BatchLlama = !useQ4Matmul && te_metal_use_llama_q4_batch(hidden);
        const bool useKVMatmul = te_metal_q4_matmul_kv_requested(batch) &&
            te_metal_use_q4_matmul_pair(hidden, batch);
        const bool useQ4BatchPairLlama = !useKVMatmul && te_metal_use_llama_q4_batch_pair(hidden);
        const NSUInteger qRowTile = useQ4Matmul
            ? TE_METAL_Q4_MATMUL_ROW_TILE
            : (useQ4BatchLlama ? TE_METAL_Q4_BATCH_LLAMA_ROW_TILE : TE_METAL_Q4_BATCH_ROW_TILE);
        const NSUInteger qBatchTile = useQ4Matmul
            ? TE_METAL_Q4_MATMUL_BATCH_TILE
            : (useQ4BatchLlama ? TE_METAL_Q4_BATCH_LLAMA_TILE : TE_METAL_Q4_BATCH_TILE);
        const NSUInteger kvRowTile = useKVMatmul
            ? TE_METAL_Q4_MATMUL_ROW_TILE
            : (useQ4BatchPairLlama ? TE_METAL_Q4_BATCH_LLAMA_ROW_TILE : TE_METAL_Q4_BATCH_ROW_TILE);
        const NSUInteger kvBatchTile = useKVMatmul
            ? TE_METAL_Q4_MATMUL_BATCH_TILE
            : (useQ4BatchPairLlama ? TE_METAL_Q4_BATCH_LLAMA_TILE : TE_METAL_Q4_BATCH_TILE);
        const MTLSize qThreads = MTLSizeMake(
            useQ4Matmul ? TE_METAL_Q4_MATMUL_THREADS : (useQ4BatchLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS),
            1,
            1);
        const MTLSize kvThreads = MTLSizeMake(
            useKVMatmul ? TE_METAL_Q4_MATMUL_THREADS : (useQ4BatchPairLlama ? TE_METAL_Q4_LLAMA_THREADS : TE_METAL_Q4_BATCH_THREADS),
            1,
            1);
        const MTLSize qGroups = useQ4Matmul
            ? MTLSizeMake((batch + qBatchTile - 1u) / qBatchTile, (hidden + qRowTile - 1u) / qRowTile, 1)
            : MTLSizeMake((hidden + qRowTile - 1u) / qRowTile, (batch + qBatchTile - 1u) / qBatchTile, 1);
        const MTLSize kvGroups = useKVMatmul
            ? MTLSizeMake((batch + kvBatchTile - 1u) / kvBatchTile, (kv + kvRowTile - 1u) / kvRowTile, 1)
            : MTLSizeMake((kv + kvRowTile - 1u) / kvRowTile, (batch + kvBatchTile - 1u) / kvBatchTile, 1);

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        [encoder setComputePipelineState:useQ4Matmul
            ? runtime.q4MatmulPipeline
            : (useQ4BatchLlama ? runtime.q4BatchLlamaPipeline : runtime.q4BatchPipeline)];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)q_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.dimsBuffer offset:0 atIndex:3];
        if (useQ4Matmul) {
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
        }
        [encoder dispatchThreadgroups:qGroups threadsPerThreadgroup:qThreads];

        if (useKVMatmul) {
            [encoder setComputePipelineState:runtime.q4MatmulPipeline];
            [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)k_offset atIndex:1];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:2];
            [encoder setBuffer:runtime.dims2Buffer offset:0 atIndex:3];
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
            [encoder dispatchThreadgroups:kvGroups threadsPerThreadgroup:kvThreads];

            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)v_offset atIndex:1];
            [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:2];
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
            [encoder dispatchThreadgroups:kvGroups threadsPerThreadgroup:kvThreads];
        } else {
            [encoder setComputePipelineState:useQ4BatchPairLlama ? runtime.q4BatchPairLlamaPipeline : runtime.q4BatchPairPipeline];
            [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)k_offset atIndex:1];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)v_offset atIndex:2];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:3];
            [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:4];
            [encoder setBuffer:runtime.dims2Buffer offset:0 atIndex:5];
            [encoder dispatchThreadgroups:kvGroups threadsPerThreadgroup:kvThreads];
        }
        [encoder endEncoding];

        const double profileStart = te_metal_now_ms();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        te_metal_profile_record(
            TE_METAL_PROFILE_QKV_BATCH,
            commandBuffer,
            te_metal_now_ms() - profileStart);
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        memcpy(q_out, [runtime.outputBuffer contents], q_bytes);
        memcpy(k_out, [runtime.output2Buffer contents], kv_bytes);
        memcpy(v_out, [runtime.output3Buffer contents], kv_bytes);
        return TE_STATUS_OK;
    }
}

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
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (mapping == nullptr || input == nullptr || out == nullptr || hidden == 0 || ffn == 0) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (hidden > UINT32_MAX || ffn > UINT32_MAX ||
        gate_offset > mapping_len || up_offset > mapping_len || down_offset > mapping_len ||
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

        const size_t gate_blocks = (hidden + 31u) / 32u;
        const size_t down_blocks = (ffn + 31u) / 32u;
        const size_t gate_row_bytes = gate_blocks * 18u;
        const size_t down_row_bytes = down_blocks * 18u;
        const size_t input_bytes = hidden * sizeof(float);
        const size_t ffn_bytes = ffn * sizeof(float);
        const size_t hidden_bytes = hidden * sizeof(float);
        if (gate_row_bytes > UINT32_MAX || down_row_bytes > UINT32_MAX ||
            gate_offset > UINT64_MAX - gate_row_bytes * ffn ||
            up_offset > UINT64_MAX - gate_row_bytes * ffn ||
            down_offset > UINT64_MAX - down_row_bytes * hidden) {
            return TE_STATUS_UNSUPPORTED;
        }

        if (runtime.inputBuffer == nil || runtime.inputCapacity < input_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:input_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = input_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < ffn_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:ffn_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = ffn_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < ffn_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:ffn_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = ffn_bytes;
        }
        if (runtime.scratchBuffer == nil || runtime.scratchCapacity < ffn_bytes) {
            runtime.scratchBuffer = [runtime.device newBufferWithLength:ffn_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.scratchCapacity = ffn_bytes;
        }
        if (runtime.output3Buffer == nil || runtime.output3Capacity < hidden_bytes) {
            runtime.output3Buffer = [runtime.device newBufferWithLength:hidden_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output3Capacity = hidden_bytes;
        }
        if (runtime.dimsBuffer == nil) {
            runtime.dimsBuffer = [runtime.device newBufferWithLength:4u * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
        }
        if (runtime.dims2Buffer == nil) {
            runtime.dims2Buffer = [runtime.device newBufferWithLength:4u * sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        }
        if (runtime.scalarBuffer == nil) {
            runtime.scalarBuffer = [runtime.device newBufferWithLength:sizeof(uint32_t)
                                                               options:MTLResourceStorageModeShared];
        }
        if (runtime.inputBuffer == nil || runtime.outputBuffer == nil || runtime.output2Buffer == nil ||
            runtime.scratchBuffer == nil || runtime.output3Buffer == nil || runtime.dimsBuffer == nil ||
            runtime.dims2Buffer == nil || runtime.scalarBuffer == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }

        uint32_t gate_dims[3] = {(uint32_t)ffn, (uint32_t)hidden, (uint32_t)gate_row_bytes};
        uint32_t down_dims[3] = {(uint32_t)hidden, (uint32_t)ffn, (uint32_t)down_row_bytes};
        uint32_t swiglu_len = (uint32_t)ffn;
        memcpy([runtime.inputBuffer contents], input, input_bytes);
        memcpy([runtime.dimsBuffer contents], gate_dims, sizeof(gate_dims));
        memcpy([runtime.dims2Buffer contents], down_dims, sizeof(down_dims));
        memcpy([runtime.scalarBuffer contents], &swiglu_len, sizeof(swiglu_len));

        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        const bool useDownLlama = te_metal_use_llama_q4(ffn);
        const bool useQ4PairLlama = te_metal_use_llama_q4_pair(hidden);
        const NSUInteger downTile = useDownLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE;
        const NSUInteger gateTile = useQ4PairLlama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_PAIR_ROW_TILE;
        const MTLSize pairThreads = MTLSizeMake(useQ4PairLlama ? TE_METAL_Q4_LLAMA_THREADS : 128, 1, 1);
        const MTLSize downThreads = MTLSizeMake(useDownLlama ? TE_METAL_Q4_LLAMA_THREADS : 128, 1, 1);
        const MTLSize gateGroups = MTLSizeMake((ffn + gateTile - 1u) / gateTile, 1, 1);
        const MTLSize downGroups = MTLSizeMake((hidden + downTile - 1u) / downTile, 1, 1);
        const MTLSize swigluThreads = MTLSizeMake(256, 1, 1);
        const MTLSize swigluGroups = MTLSizeMake((ffn + 255u) / 256u, 1, 1);

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        [encoder setComputePipelineState:useQ4PairLlama ? runtime.q4PairLlamaPipeline : runtime.q4PairPipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)gate_offset atIndex:1];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)up_offset atIndex:2];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:3];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:4];
        [encoder setBuffer:runtime.dimsBuffer offset:0 atIndex:5];
        [encoder dispatchThreadgroups:gateGroups threadsPerThreadgroup:pairThreads];

        [encoder setComputePipelineState:runtime.swigluPipeline];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.scalarBuffer offset:0 atIndex:3];
        [encoder dispatchThreadgroups:swigluGroups threadsPerThreadgroup:swigluThreads];

        [encoder setComputePipelineState:useDownLlama ? runtime.q4LlamaPipeline : runtime.q4Pipeline];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)down_offset atIndex:1];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.dims2Buffer offset:0 atIndex:3];
        [encoder dispatchThreadgroups:downGroups threadsPerThreadgroup:downThreads];
        [encoder endEncoding];

        const double profileStart = te_metal_now_ms();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        te_metal_profile_record(
            TE_METAL_PROFILE_MLP,
            commandBuffer,
            te_metal_now_ms() - profileStart);
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        memcpy(out, [runtime.output3Buffer contents], hidden_bytes);
        return TE_STATUS_OK;
    }
}

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
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (mapping == nullptr || input == nullptr || out == nullptr || batch == 0 || hidden == 0 || ffn == 0) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (batch > UINT32_MAX || hidden > UINT32_MAX || ffn > UINT32_MAX ||
        gate_offset > mapping_len || up_offset > mapping_len || down_offset > mapping_len ||
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

        const size_t gate_blocks = (hidden + 31u) / 32u;
        const size_t down_blocks = (ffn + 31u) / 32u;
        const size_t gate_row_bytes = gate_blocks * 18u;
        const size_t down_row_bytes = down_blocks * 18u;
        const size_t hidden_values = batch * hidden;
        const size_t ffn_values = batch * ffn;
        const size_t input_bytes = hidden_values * sizeof(float);
        const size_t ffn_bytes = ffn_values * sizeof(float);
        const size_t hidden_bytes = hidden_values * sizeof(float);
        if (gate_row_bytes > UINT32_MAX || down_row_bytes > UINT32_MAX ||
            ffn_values > UINT32_MAX ||
            gate_offset > UINT64_MAX - gate_row_bytes * ffn ||
            up_offset > UINT64_MAX - gate_row_bytes * ffn ||
            down_offset > UINT64_MAX - down_row_bytes * hidden) {
            return TE_STATUS_UNSUPPORTED;
        }

        if (runtime.inputBuffer == nil || runtime.inputCapacity < input_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:input_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = input_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < ffn_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:ffn_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = ffn_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < ffn_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:ffn_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = ffn_bytes;
        }
        if (runtime.scratchBuffer == nil || runtime.scratchCapacity < ffn_bytes) {
            runtime.scratchBuffer = [runtime.device newBufferWithLength:ffn_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.scratchCapacity = ffn_bytes;
        }
        if (runtime.output3Buffer == nil || runtime.output3Capacity < hidden_bytes) {
            runtime.output3Buffer = [runtime.device newBufferWithLength:hidden_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output3Capacity = hidden_bytes;
        }
        if (runtime.dimsBuffer == nil) {
            runtime.dimsBuffer = [runtime.device newBufferWithLength:4u * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
        }
        if (runtime.dims2Buffer == nil) {
            runtime.dims2Buffer = [runtime.device newBufferWithLength:4u * sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        }
        if (runtime.scalarBuffer == nil) {
            runtime.scalarBuffer = [runtime.device newBufferWithLength:sizeof(uint32_t)
                                                               options:MTLResourceStorageModeShared];
        }
        if (runtime.inputBuffer == nil || runtime.outputBuffer == nil || runtime.output2Buffer == nil ||
            runtime.scratchBuffer == nil || runtime.output3Buffer == nil || runtime.dimsBuffer == nil ||
            runtime.dims2Buffer == nil || runtime.scalarBuffer == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }

        uint32_t gate_dims[4] = {(uint32_t)ffn, (uint32_t)hidden, (uint32_t)gate_row_bytes, (uint32_t)batch};
        uint32_t down_dims[4] = {(uint32_t)hidden, (uint32_t)ffn, (uint32_t)down_row_bytes, (uint32_t)batch};
        uint32_t swiglu_len = (uint32_t)ffn_values;
        memcpy([runtime.inputBuffer contents], input, input_bytes);
        memcpy([runtime.dimsBuffer contents], gate_dims, sizeof(gate_dims));
        memcpy([runtime.dims2Buffer contents], down_dims, sizeof(down_dims));
        memcpy([runtime.scalarBuffer contents], &swiglu_len, sizeof(swiglu_len));

        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        const bool useGateMatmul = te_metal_q4_matmul_gateup_requested() &&
            te_metal_use_q4_matmul_pair(hidden, batch);
        const bool useGateLlama = !useGateMatmul && te_metal_use_llama_q4_batch_pair(hidden);
        const bool useDownMatmul = te_metal_use_q4_matmul(ffn, batch);
        const bool useDownLlama = !useDownMatmul && te_metal_use_llama_q4_batch(ffn);
        const bool useFfnHalf = te_metal_use_q4_ffn_half(runtime, useGateMatmul, useDownMatmul);
        const bool useFfnGateHalf = te_metal_use_q4_ffn_gate_half(runtime, useFfnHalf, batch);
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
        const MTLSize gateThreads = MTLSizeMake(
            useGateMatmul ? TE_METAL_Q4_MATMUL_THREADS : (useGateLlama ? TE_METAL_Q4_LLAMA_THREADS : 128),
            1,
            1);
        const MTLSize downThreads = MTLSizeMake(
            useDownMatmul ? TE_METAL_Q4_MATMUL_THREADS : (useDownLlama ? TE_METAL_Q4_LLAMA_THREADS : 128),
            1,
            1);
        const MTLSize gateGroups = useGateMatmul
            ? MTLSizeMake((batch + gateBatchTile - 1u) / gateBatchTile, (ffn + gateRowTile - 1u) / gateRowTile, 1)
            : MTLSizeMake((ffn + gateRowTile - 1u) / gateRowTile, (batch + gateBatchTile - 1u) / gateBatchTile, 1);
        const MTLSize downGroups = useDownMatmul
            ? MTLSizeMake((batch + downBatchTile - 1u) / downBatchTile, (hidden + downRowTile - 1u) / downRowTile, 1)
            : MTLSizeMake((hidden + downRowTile - 1u) / downRowTile, (batch + downBatchTile - 1u) / downBatchTile, 1);
        const MTLSize swigluThreads = MTLSizeMake(256, 1, 1);
        const MTLSize swigluGroups = MTLSizeMake((ffn_values + 255u) / 256u, 1, 1);

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        if (useGateMatmul) {
            [encoder setComputePipelineState:useFfnGateHalf ? runtime.q4MatmulStoreHalfPipeline : runtime.q4MatmulPipeline];
            [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)gate_offset atIndex:1];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
            [encoder setBuffer:runtime.dimsBuffer offset:0 atIndex:3];
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
            [encoder dispatchThreadgroups:gateGroups threadsPerThreadgroup:gateThreads];

            [encoder setComputePipelineState:useFfnGateHalf
                ? runtime.q4MatmulSwigluGateHalfPipeline
                : (useFfnHalf ? runtime.q4MatmulSwigluHalfPipeline : runtime.q4MatmulSwigluPipeline)];
            [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)up_offset atIndex:1];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
            [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:3];
            [encoder setBuffer:runtime.dimsBuffer offset:0 atIndex:4];
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
            [encoder dispatchThreadgroups:gateGroups threadsPerThreadgroup:gateThreads];
        } else {
            [encoder setComputePipelineState:useGateLlama ? runtime.q4BatchPairLlamaPipeline : runtime.q4BatchPairPipeline];
            [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)gate_offset atIndex:1];
            [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)up_offset atIndex:2];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:3];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:4];
            [encoder setBuffer:runtime.dimsBuffer offset:0 atIndex:5];
            [encoder dispatchThreadgroups:gateGroups threadsPerThreadgroup:gateThreads];
        }

        if (!useGateMatmul) {
            [encoder setComputePipelineState:runtime.swigluPipeline];
            [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:0];
            [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:1];
            [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:2];
            [encoder setBuffer:runtime.scalarBuffer offset:0 atIndex:3];
            [encoder dispatchThreadgroups:swigluGroups threadsPerThreadgroup:swigluThreads];
        }

        [encoder setComputePipelineState:useDownMatmul
            ? (useFfnHalf ? runtime.q4MatmulHalfInputPipeline : runtime.q4MatmulPipeline)
            : (useDownLlama ? runtime.q4BatchLlamaPipeline : runtime.q4BatchPipeline)];
        [encoder setBuffer:runtime.scratchBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)down_offset atIndex:1];
        [encoder setBuffer:runtime.output3Buffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.dims2Buffer offset:0 atIndex:3];
        if (useDownMatmul) {
            [encoder setThreadgroupMemoryLength:TE_METAL_Q4_MATMUL_SHMEM atIndex:0];
        }
        [encoder dispatchThreadgroups:downGroups threadsPerThreadgroup:downThreads];
        [encoder endEncoding];

        const double profileStart = te_metal_now_ms();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        te_metal_profile_record(
            TE_METAL_PROFILE_MLP_BATCH,
            commandBuffer,
            te_metal_now_ms() - profileStart);
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        memcpy(out, [runtime.output3Buffer contents], hidden_bytes);
        return TE_STATUS_OK;
    }
}

// Layer orchestration is split out so this file can focus on primitive
// Q4/Q8/QKV/MLP dispatch entry points.
#include "metal/metal_backend_layers.mm.inc"

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
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (mapping == nullptr || input == nullptr || out_a == nullptr || out_b == nullptr || cols == 0 || rows == 0) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (cols > UINT32_MAX || rows > UINT32_MAX ||
        tensor_a_offset > mapping_len || tensor_b_offset > mapping_len ||
        rows > SIZE_MAX / cols || rows * cols < 1000000u) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (ggml_type != TE_METAL_GGML_TYPE_Q4_0 && ggml_type != TE_METAL_GGML_TYPE_Q8_0) {
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

        const size_t blocks_per_row = (cols + 31u) / 32u;
        const size_t row_bytes = ggml_type == TE_METAL_GGML_TYPE_Q4_0 ? blocks_per_row * 18u : blocks_per_row * 34u;
        const size_t input_bytes = cols * sizeof(float);
        const size_t output_bytes = rows * sizeof(float);
        if (row_bytes > UINT32_MAX ||
            tensor_a_offset > UINT64_MAX - row_bytes * rows ||
            tensor_b_offset > UINT64_MAX - row_bytes * rows) {
            return TE_STATUS_UNSUPPORTED;
        }
        if (runtime.inputBuffer == nil || runtime.inputCapacity < input_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:input_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = input_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < output_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:output_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = output_bytes;
        }
        if (runtime.output2Buffer == nil || runtime.output2Capacity < output_bytes) {
            runtime.output2Buffer = [runtime.device newBufferWithLength:output_bytes
                                                                options:MTLResourceStorageModeShared];
            runtime.output2Capacity = output_bytes;
        }
        if (runtime.dimsBuffer == nil) {
            runtime.dimsBuffer = [runtime.device newBufferWithLength:4u * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
        }
        if (runtime.inputBuffer == nil || runtime.outputBuffer == nil ||
            runtime.output2Buffer == nil || runtime.dimsBuffer == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }
        uint32_t dims[3] = {(uint32_t)rows, (uint32_t)cols, (uint32_t)row_bytes};
        memcpy([runtime.inputBuffer contents], input, input_bytes);
        memcpy([runtime.dimsBuffer contents], dims, sizeof(dims));

        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        if (commandBuffer == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }

        const bool useQ4Llama = ggml_type == TE_METAL_GGML_TYPE_Q4_0 && te_metal_use_llama_q4(cols);
        const bool useQ8Llama = ggml_type == TE_METAL_GGML_TYPE_Q8_0 && te_metal_use_llama_q8(cols);
        const MTLSize threads = MTLSizeMake(
            useQ4Llama ? TE_METAL_Q4_LLAMA_THREADS : (useQ8Llama ? TE_METAL_Q8_LLAMA_THREADS : 128),
            1,
            1);
        MTLSize groups;
        id<MTLComputePipelineState> pipeline;
        if (ggml_type == TE_METAL_GGML_TYPE_Q4_0) {
            const NSUInteger rowTile = useQ4Llama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE;
            pipeline = useQ4Llama ? runtime.q4LlamaPipeline : runtime.q4Pipeline;
            groups = MTLSizeMake((rows + rowTile - 1u) / rowTile, 1, 1);
        } else {
            const NSUInteger rowTile = useQ8Llama ? TE_METAL_Q8_LLAMA_ROW_TILE : 1u;
            pipeline = useQ8Llama ? runtime.q8LlamaPipeline : runtime.q8Pipeline;
            groups = MTLSizeMake((rows + rowTile - 1u) / rowTile, 1, 1);
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)tensor_a_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.dimsBuffer offset:0 atIndex:3];
        if (useQ8Llama) {
            [encoder setThreadgroupMemoryLength:TE_METAL_Q8_LLAMA_SHMEM atIndex:0];
        }
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        [encoder endEncoding];

        encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)tensor_b_offset atIndex:1];
        [encoder setBuffer:runtime.output2Buffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.dimsBuffer offset:0 atIndex:3];
        if (useQ8Llama) {
            [encoder setThreadgroupMemoryLength:TE_METAL_Q8_LLAMA_SHMEM atIndex:0];
        }
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        [encoder endEncoding];

        const double profileStart = te_metal_now_ms();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        te_metal_profile_record(
            TE_METAL_PROFILE_MATVEC2,
            commandBuffer,
            te_metal_now_ms() - profileStart);
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        memcpy(out_a, [runtime.outputBuffer contents], output_bytes);
        memcpy(out_b, [runtime.output2Buffer contents], output_bytes);
        return TE_STATUS_OK;
    }
}

te_status te_metal_matvec_f32(
    const void *mapping,
    size_t mapping_len,
    uint64_t tensor_offset,
    uint32_t ggml_type,
    const float *input,
    size_t cols,
    size_t rows,
    float *out
) {
    if (!te_metal_enabled()) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (mapping == nullptr || input == nullptr || out == nullptr || cols == 0 || rows == 0) {
        return TE_STATUS_INVALID_ARGUMENT;
    }
    if (cols > UINT32_MAX || rows > UINT32_MAX || tensor_offset > mapping_len) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (ggml_type != TE_METAL_GGML_TYPE_Q4_0 && ggml_type != TE_METAL_GGML_TYPE_Q8_0) {
        return TE_STATUS_UNSUPPORTED;
    }
    if (rows > SIZE_MAX / cols || rows * cols < 1000000u) {
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

        const size_t blocks_per_row = (cols + 31u) / 32u;
        const size_t row_bytes = ggml_type == TE_METAL_GGML_TYPE_Q4_0 ? blocks_per_row * 18u : blocks_per_row * 34u;
        if (row_bytes > UINT32_MAX || tensor_offset > UINT64_MAX - row_bytes * rows) {
            return TE_STATUS_UNSUPPORTED;
        }

        const size_t input_bytes = cols * sizeof(float);
        const size_t output_bytes = rows * sizeof(float);
        if (runtime.inputBuffer == nil || runtime.inputCapacity < input_bytes) {
            runtime.inputBuffer = [runtime.device newBufferWithLength:input_bytes
                                                              options:MTLResourceStorageModeShared];
            runtime.inputCapacity = input_bytes;
        }
        if (runtime.outputBuffer == nil || runtime.outputCapacity < output_bytes) {
            runtime.outputBuffer = [runtime.device newBufferWithLength:output_bytes
                                                               options:MTLResourceStorageModeShared];
            runtime.outputCapacity = output_bytes;
        }
        if (runtime.dimsBuffer == nil) {
            runtime.dimsBuffer = [runtime.device newBufferWithLength:4u * sizeof(uint32_t)
                                                             options:MTLResourceStorageModeShared];
        }
        uint32_t dims[3] = {(uint32_t)rows, (uint32_t)cols, (uint32_t)row_bytes};
        if (runtime.inputBuffer == nil || runtime.outputBuffer == nil || runtime.dimsBuffer == nil) {
            return TE_STATUS_OUT_OF_MEMORY;
        }
        memcpy([runtime.inputBuffer contents], input, input_bytes);
        memcpy([runtime.dimsBuffer contents], dims, sizeof(dims));

        id<MTLCommandBuffer> commandBuffer = [runtime.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (commandBuffer == nil || encoder == nil) {
            return TE_STATUS_RUNTIME_ERROR;
        }
        const bool useQ4Llama = ggml_type == TE_METAL_GGML_TYPE_Q4_0 && te_metal_use_llama_q4(cols);
        const bool useQ8Llama = ggml_type == TE_METAL_GGML_TYPE_Q8_0 && te_metal_use_llama_q8(cols);
        if (ggml_type == TE_METAL_GGML_TYPE_Q4_0) {
            [encoder setComputePipelineState:useQ4Llama ? runtime.q4LlamaPipeline : runtime.q4Pipeline];
        } else {
            [encoder setComputePipelineState:useQ8Llama ? runtime.q8LlamaPipeline : runtime.q8Pipeline];
        }
        [encoder setBuffer:runtime.inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.mappingBuffer offset:(NSUInteger)tensor_offset atIndex:1];
        [encoder setBuffer:runtime.outputBuffer offset:0 atIndex:2];
        [encoder setBuffer:runtime.dimsBuffer offset:0 atIndex:3];

        if (useQ8Llama) {
            [encoder setThreadgroupMemoryLength:TE_METAL_Q8_LLAMA_SHMEM atIndex:0];
        }
        const MTLSize threads = MTLSizeMake(
            useQ4Llama ? TE_METAL_Q4_LLAMA_THREADS : (useQ8Llama ? TE_METAL_Q8_LLAMA_THREADS : 128),
            1,
            1);
        MTLSize groups;
        if (ggml_type == TE_METAL_GGML_TYPE_Q4_0) {
            const NSUInteger rowTile = useQ4Llama ? TE_METAL_Q4_LLAMA_ROW_TILE : TE_METAL_Q4_ROW_TILE;
            groups = MTLSizeMake((rows + rowTile - 1u) / rowTile, 1, 1);
        } else {
            const NSUInteger rowTile = useQ8Llama ? TE_METAL_Q8_LLAMA_ROW_TILE : 1u;
            groups = MTLSizeMake((rows + rowTile - 1u) / rowTile, 1, 1);
        }
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        [encoder endEncoding];
        const double profileStart = te_metal_now_ms();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        te_metal_profile_record(
            TE_METAL_PROFILE_MATVEC,
            commandBuffer,
            te_metal_now_ms() - profileStart);
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return TE_STATUS_RUNTIME_ERROR;
        }

        memcpy(out, [runtime.outputBuffer contents], output_bytes);
        return TE_STATUS_OK;
    }
}
