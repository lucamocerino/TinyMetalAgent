// Included by metal_backend.mm to initialize TE_METAL_SOURCE. Not a standalone translation unit.
@R"TE_METAL(
#include <metal_stdlib>
using namespace metal;

#define TE_THREADS 128
#define TE_Q4_LLAMA_THREADS 64
#define TE_Q4_LLAMA_NSG 2
#define TE_Q4_LLAMA_ROWS_PER_SIMD 4
#define TE_Q4_LLAMA_ROW_TILE (TE_Q4_LLAMA_NSG * TE_Q4_LLAMA_ROWS_PER_SIMD)
#define TE_Q4_BATCH_LLAMA_ROWS_PER_SIMD 4
#define TE_Q4_BATCH_LLAMA_ROW_TILE (TE_Q4_LLAMA_NSG * TE_Q4_BATCH_LLAMA_ROWS_PER_SIMD)
#define TE_Q4_BATCH_THREADS 128
#define TE_Q4_ROW_TILE 4
#define TE_Q4_PAIR_ROW_TILE 4
#define TE_Q4_BATCH_ROW_TILE 1
#define TE_Q4_BATCH_TILE 7
#define TE_Q4_BATCH_LLAMA_TILE 1
#define TE_Q8_LLAMA_ROW_TILE 2
#define TE_Q8_LLAMA_NSG 4
#define TE_SIMDGROUPS 4
#define TE_Q4_BATCH_SIMDGROUPS 4
#define TE_ARGMAX_THREADS 256
#define TE_FLASH_ATTN_THREADS 128
#define TE_FLASH_ATTN_SIMDGROUPS 4
#define TE_FLASH_ATTN_TILE 64
#define TE_FLASH_ATTN_TILE_LONG 128
#define TE_FLASH_ATTN_GQA_MAX_REPEAT 8
#define TE_METAL_UNROLL _Pragma("clang loop unroll(full)")

static inline float te_load_f16(device const uchar *ptr) {
    const ushort bits = ushort(ptr[0]) | (ushort(ptr[1]) << 8);
    return float(as_type<half>(bits));
}

static inline float te_q4_0_block_dot_llama(device const uchar *block_ptr, float sumy, thread float *yl, uint il) {
    const float scale = te_load_f16(block_ptr);
    device const ushort *qs = (device const ushort *)(block_ptr + 2 + il);
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint i = 0; i < 8; i += 2) {
        const ushort packed = qs[i / 2];
        acc0 += yl[i + 0] * float(packed & 0x000fu);
        acc1 += yl[i + 1] * float(packed & 0x0f00u);
        acc2 += yl[i + 8] * float(packed & 0x00f0u);
        acc3 += yl[i + 9] * float(packed & 0xf000u);
    }
    return scale * (sumy * -8.0f + acc0 + acc1 + acc2 + acc3);
}

struct te_block_q4_0 {
    half d;
    uchar qs[16];
};

static inline void te_dequantize_q4_0_mm(device const te_block_q4_0 *xb, short il, thread half4x4 &reg) {
    device const ushort *qs = ((device const ushort *)xb + 1);
    const float d = float(xb->d);
    const float d1 = il ? d / 16.0f : d;
    const float d2 = d1 / 256.0f;
    const float md = -8.0f * d;
    const ushort mask0 = il ? 0x00f0 : 0x000f;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;
    for (int i = 0; i < 8; ++i) {
        reg_f[i / 2][2 * (i % 2) + 0] = d1 * float(qs[i] & mask0) + md;
        reg_f[i / 2][2 * (i % 2) + 1] = d2 * float(qs[i] & mask1) + md;
    }
    reg = half4x4(reg_f);
}

kernel void te_q4_0_matvec_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint *dims [[buffer(3)]],
    uint row_tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint rows = dims[0];
    const uint cols = dims[1];
    const uint row_bytes = dims[2];
    const uint row_base = row_tile * TE_Q4_ROW_TILE;
    if (row_base >= rows) {
        return;
    }

    const uint row = row_base + simd_group;
    if (row >= rows) {
        return;
    }

    float acc = 0.0f;
    const uint blocks = (cols + 31u) / 32u;
    const uint pairs = blocks * 16u;
    for (uint pair = simd_lane; pair < pairs; pair += 32u) {
        const uint block = pair / 16u;
        const uint inner = pair - block * 16u;
        const uint lo_col = block * 32u + inner;
        const uint hi_col = lo_col + 16u;
        const float x_lo = lo_col < cols ? x[lo_col] : 0.0f;
        const float x_hi = hi_col < cols ? x[hi_col] : 0.0f;
        device const uchar *block_ptr = weights + row * row_bytes + block * 18;
        const float scale = te_load_f16(block_ptr);
        const uchar packed = block_ptr[2 + inner];
        if (lo_col < cols) {
            const uint q = uint(packed) & 0x0fu;
            acc += x_lo * ((float(q) - 8.0f) * scale);
        }
        if (hi_col < cols) {
            const uint q = uint(packed) >> 4u;
            acc += x_hi * ((float(q) - 8.0f) * scale);
        }
    }

    const float sum = simd_sum(acc);
    if (simd_lane == 0u) {
        out[row] = sum;
    }
}

kernel void te_q4_0_matvec_llama_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint *dims [[buffer(3)]],
    uint row_group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint rows = dims[0];
    const uint cols = dims[1];
    const uint row_bytes = dims[2];
    const uint blocks = cols / 32u;
    const uint row_base = (row_group * TE_Q4_LLAMA_NSG + simd_group) * TE_Q4_LLAMA_ROWS_PER_SIMD;
    if (row_base >= rows) {
        return;
    }

    float acc[TE_Q4_LLAMA_ROWS_PER_SIMD];
    for (uint row_lane = 0; row_lane < TE_Q4_LLAMA_ROWS_PER_SIMD; ++row_lane) {
        acc[row_lane] = 0.0f;
    }

    const uint ix = simd_lane / 2u;
    const uint il = (simd_lane & 1u) * 8u;
    device const float *x_block = x + ix * 32u + il;
    for (uint block = ix; block < blocks; block += 16u) {
        float yl[16];
        float sumy = 0.0f;
        for (uint i = 0; i < 8; i += 2) {
            const float y0 = x_block[i + 0];
            const float y1 = x_block[i + 1];
            const float y16 = x_block[i + 16];
            const float y17 = x_block[i + 17];
            sumy += y0 + y1 + y16 + y17;
            yl[i + 0] = y0;
            yl[i + 1] = y1 / 256.0f;
            yl[i + 8] = y16 / 16.0f;
            yl[i + 9] = y17 / 4096.0f;
        }

        for (uint row_lane = 0; row_lane < TE_Q4_LLAMA_ROWS_PER_SIMD; ++row_lane) {
            const uint row = row_base + row_lane;
            if (row < rows) {
                device const uchar *block_ptr = weights + row * row_bytes + block * 18u;
                acc[row_lane] += te_q4_0_block_dot_llama(block_ptr, sumy, yl, il);
            }
        }
        x_block += 16u * 32u;
    }

    for (uint row_lane = 0; row_lane < TE_Q4_LLAMA_ROWS_PER_SIMD; ++row_lane) {
        const float sum = simd_sum(acc[row_lane]);
        const uint row = row_base + row_lane;
        if (simd_lane == 0u && row < rows) {
            out[row] = sum;
        }
    }
}

kernel void te_q4_0_matvec_pair_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weights_a [[buffer(1)]],
    device const uchar *weights_b [[buffer(2)]],
    device float *out_a [[buffer(3)]],
    device float *out_b [[buffer(4)]],
    constant uint *dims [[buffer(5)]],
    uint row_tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint rows = dims[0];
    const uint cols = dims[1];
    const uint row_bytes = dims[2];
    const uint row_base = row_tile * TE_Q4_PAIR_ROW_TILE;
    if (row_base >= rows) {
        return;
    }

    float acc_a[TE_Q4_PAIR_ROW_TILE];
    float acc_b[TE_Q4_PAIR_ROW_TILE];
    for (uint lane = 0; lane < TE_Q4_PAIR_ROW_TILE; ++lane) {
        acc_a[lane] = 0.0f;
        acc_b[lane] = 0.0f;
    }

    const uint blocks = (cols + 31u) / 32u;
    const uint pairs = blocks * 16u;
    for (uint pair = tid; pair < pairs; pair += TE_THREADS) {
        const uint block = pair / 16u;
        const uint inner = pair - block * 16u;
        const uint lo_col = block * 32u + inner;
        const uint hi_col = lo_col + 16u;
        const float x_lo = lo_col < cols ? x[lo_col] : 0.0f;
        const float x_hi = hi_col < cols ? x[hi_col] : 0.0f;
        for (uint lane = 0; lane < TE_Q4_PAIR_ROW_TILE; ++lane) {
            const uint row = row_base + lane;
            if (row < rows) {
                device const uchar *block_a = weights_a + row * row_bytes + block * 18;
                device const uchar *block_b = weights_b + row * row_bytes + block * 18;
                const float scale_a = te_load_f16(block_a);
                const float scale_b = te_load_f16(block_b);
                const uchar packed_a = block_a[2 + inner];
                const uchar packed_b = block_b[2 + inner];
                if (lo_col < cols) {
                    acc_a[lane] += x_lo * ((float(uint(packed_a) & 0x0fu) - 8.0f) * scale_a);
                    acc_b[lane] += x_lo * ((float(uint(packed_b) & 0x0fu) - 8.0f) * scale_b);
                }
                if (hi_col < cols) {
                    acc_a[lane] += x_hi * ((float(uint(packed_a) >> 4u) - 8.0f) * scale_a);
                    acc_b[lane] += x_hi * ((float(uint(packed_b) >> 4u) - 8.0f) * scale_b);
                }
            }
        }
    }

    threadgroup float partial[2 * TE_Q4_PAIR_ROW_TILE * TE_SIMDGROUPS];
    for (uint lane = 0; lane < TE_Q4_PAIR_ROW_TILE; ++lane) {
        const float sum_a = simd_sum(acc_a[lane]);
        const float sum_b = simd_sum(acc_b[lane]);
        if (simd_lane == 0u) {
            const uint offset = lane * TE_SIMDGROUPS + simd_group;
            partial[offset] = sum_a;
            partial[TE_Q4_PAIR_ROW_TILE * TE_SIMDGROUPS + offset] = sum_b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        for (uint lane = 0; lane < TE_Q4_PAIR_ROW_TILE; ++lane) {
            float sum_a = 0.0f;
            float sum_b = 0.0f;
            for (uint group = 0; group < TE_SIMDGROUPS; ++group) {
                const uint offset = lane * TE_SIMDGROUPS + group;
                sum_a += partial[offset];
                sum_b += partial[TE_Q4_PAIR_ROW_TILE * TE_SIMDGROUPS + offset];
            }
            const uint row = row_base + lane;
            if (row < rows) {
                out_a[row] = sum_a;
                out_b[row] = sum_b;
            }
        }
    }
}

kernel void te_q4_0_matvec_pair_llama_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weights_a [[buffer(1)]],
    device const uchar *weights_b [[buffer(2)]],
    device float *out_a [[buffer(3)]],
    device float *out_b [[buffer(4)]],
    constant uint *dims [[buffer(5)]],
    uint row_group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint rows = dims[0];
    const uint cols = dims[1];
    const uint row_bytes = dims[2];
    const uint blocks = cols / 32u;
    const uint row_base = (row_group * TE_Q4_LLAMA_NSG + simd_group) * TE_Q4_LLAMA_ROWS_PER_SIMD;
    if (row_base >= rows) {
        return;
    }

    float acc_a[TE_Q4_LLAMA_ROWS_PER_SIMD];
    float acc_b[TE_Q4_LLAMA_ROWS_PER_SIMD];
    for (uint row_lane = 0; row_lane < TE_Q4_LLAMA_ROWS_PER_SIMD; ++row_lane) {
        acc_a[row_lane] = 0.0f;
        acc_b[row_lane] = 0.0f;
    }

    const uint ix = simd_lane / 2u;
    const uint il = (simd_lane & 1u) * 8u;
    device const float *x_block = x + ix * 32u + il;
    for (uint block = ix; block < blocks; block += 16u) {
        float yl[16];
        float sumy = 0.0f;
        for (uint i = 0; i < 8; i += 2) {
            const float y0 = x_block[i + 0];
            const float y1 = x_block[i + 1];
            const float y16 = x_block[i + 16];
            const float y17 = x_block[i + 17];
            sumy += y0 + y1 + y16 + y17;
            yl[i + 0] = y0;
            yl[i + 1] = y1 / 256.0f;
            yl[i + 8] = y16 / 16.0f;
            yl[i + 9] = y17 / 4096.0f;
        }

        for (uint row_lane = 0; row_lane < TE_Q4_LLAMA_ROWS_PER_SIMD; ++row_lane) {
            const uint row = row_base + row_lane;
            if (row < rows) {
                device const uchar *block_a = weights_a + row * row_bytes + block * 18u;
                device const uchar *block_b = weights_b + row * row_bytes + block * 18u;
                acc_a[row_lane] += te_q4_0_block_dot_llama(block_a, sumy, yl, il);
                acc_b[row_lane] += te_q4_0_block_dot_llama(block_b, sumy, yl, il);
            }
        }
        x_block += 16u * 32u;
    }

    for (uint row_lane = 0; row_lane < TE_Q4_LLAMA_ROWS_PER_SIMD; ++row_lane) {
        const uint row = row_base + row_lane;
        const float sum_a = simd_sum(acc_a[row_lane]);
        const float sum_b = simd_sum(acc_b[row_lane]);
        if (simd_lane == 0u && row < rows) {
            out_a[row] = sum_a;
            out_b[row] = sum_b;
        }
    }
}

kernel void te_q4_0_matvec_batch_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint *dims [[buffer(3)]],
    uint2 grid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint rows = dims[0];
    const uint cols = dims[1];
    const uint row_bytes = dims[2];
    const uint batch_count = dims[3];
    const uint row_base = grid.x * TE_Q4_BATCH_ROW_TILE;
    const uint batch_base = grid.y * TE_Q4_BATCH_TILE;
    if (row_base >= rows || batch_base >= batch_count) {
        return;
    }

    float acc[TE_Q4_BATCH_ROW_TILE][TE_Q4_BATCH_TILE];
    for (uint lane = 0; lane < TE_Q4_BATCH_ROW_TILE; ++lane) {
        for (uint batch_lane = 0; batch_lane < TE_Q4_BATCH_TILE; ++batch_lane) {
            acc[lane][batch_lane] = 0.0f;
        }
    }

    const uint blocks = (cols + 31u) / 32u;
    const uint pairs = blocks * 16u;
    for (uint pair = tid; pair < pairs; pair += TE_Q4_BATCH_THREADS) {
        const uint block = pair / 16u;
        const uint inner = pair - block * 16u;
        const uint lo_col = block * 32u + inner;
        const uint hi_col = lo_col + 16u;
        for (uint lane = 0; lane < TE_Q4_BATCH_ROW_TILE; ++lane) {
            const uint row = row_base + lane;
            if (row < rows) {
                device const uchar *block_ptr = weights + row * row_bytes + block * 18;
                const float scale = te_load_f16(block_ptr);
                const uchar packed = block_ptr[2 + inner];
                const float lo_weight = ((float(uint(packed) & 0x0fu) - 8.0f) * scale);
                const float hi_weight = ((float(uint(packed) >> 4u) - 8.0f) * scale);
                for (uint batch_lane = 0; batch_lane < TE_Q4_BATCH_TILE; ++batch_lane) {
                    const uint batch = batch_base + batch_lane;
                    if (batch < batch_count) {
                        device const float *batch_x = x + batch * cols;
                        if (lo_col < cols) {
                            acc[lane][batch_lane] += batch_x[lo_col] * lo_weight;
                        }
                        if (hi_col < cols) {
                            acc[lane][batch_lane] += batch_x[hi_col] * hi_weight;
                        }
                    }
                }
            }
        }
    }

    threadgroup float partial[TE_Q4_BATCH_ROW_TILE * TE_Q4_BATCH_TILE * TE_Q4_BATCH_SIMDGROUPS];
    for (uint lane = 0; lane < TE_Q4_BATCH_ROW_TILE; ++lane) {
        for (uint batch_lane = 0; batch_lane < TE_Q4_BATCH_TILE; ++batch_lane) {
            const float sum = simd_sum(acc[lane][batch_lane]);
            if (simd_lane == 0u) {
                const uint offset = (lane * TE_Q4_BATCH_TILE + batch_lane) * TE_Q4_BATCH_SIMDGROUPS + simd_group;
                partial[offset] = sum;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        for (uint lane = 0; lane < TE_Q4_BATCH_ROW_TILE; ++lane) {
            for (uint batch_lane = 0; batch_lane < TE_Q4_BATCH_TILE; ++batch_lane) {
                float sum = 0.0f;
                for (uint group = 0; group < TE_Q4_BATCH_SIMDGROUPS; ++group) {
                    const uint offset = (lane * TE_Q4_BATCH_TILE + batch_lane) * TE_Q4_BATCH_SIMDGROUPS + group;
                    sum += partial[offset];
                }
                const uint row = row_base + lane;
                const uint batch = batch_base + batch_lane;
                if (row < rows && batch < batch_count) {
                    out[batch * rows + row] = sum;
                }
            }
        }
    }
}

kernel void te_q4_0_matmat_llama_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint *dims [[buffer(3)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint3 grid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;

    const int rows = int(dims[0]);
    const int cols = int(dims[1]);
    const int row_bytes = int(dims[2]);
    const int batch = int(dims[3]);
    const int r0 = int(grid.y) * NR0;
    const int r1 = int(grid.x) * NR1;

    const short nr0 = short(min(NR0, max(0, rows - r0)));
    const short nr1 = short(min(NR1, max(0, batch - r1)));
    if (nr0 <= 0 || nr1 <= 0) {
        return;
    }

    const short lr0 = ((short)tid / NL0) < nr0 ? ((short)tid / NL0) : nr0 - 1;
    const short lr1 = ((short)tid / NL1) < nr1 ? ((short)tid / NL1) : nr1 - 1;
    const short il0 = tid % NL0;
    short il = il0;

    device const te_block_q4_0 *weight_block =
        (device const te_block_q4_0 *)(weights + row_bytes * (r0 + lr0));
    const short iy = 8 * (tid % NL1);
    device const float *x_ptr = x + (r1 + lr1) * cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 acc[8];
    TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
        acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < cols; loop_k += NK) {
        half4x4 temp_a;
        te_dequantize_q4_0_mm(weight_block, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        TE_METAL_UNROLL for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tid / NL0) / 8;
            const short lx = (tid / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) =
                (loop_k + 16 * il + i < cols) ? temp_a[i / 4][i % 4] : half(0.0f);
        }

        const short sx_b = tid % NL1;
        const short sy_b = (tid / NL1) / 8;
        const short ly_b = (tid / NL1) % 8;
        const short ib_b = 4 * sx_b + sy_b;
        threadgroup half2x4 *dst_b = (threadgroup half2x4 *)(sb + 64 * ib_b + 8 * ly_b);
        *dst_b = half2x4(*((device const float2x4 *)x_ptr));

        il = (il + 2 < 2) ? il + 2 : il % 2;
        if (il < 2) {
            weight_block += 1;
        }
        x_ptr += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + 4 * 64 * (simd_group % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (simd_group / 2);

        TE_METAL_UNROLL for (short ik = 0; ik < NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(acc[i], mb[i / 4], ma[i % 4], acc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + NR0 <= rows && r1 + NR1 <= batch) {
        device float *dst = out + (r0 + 32 * (simd_group & 1)) + (r1 + 16 * (simd_group >> 1)) * rows;
        TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
            simdgroup_store(acc[i], dst + 8 * (i % 4) + 8 * rows * (i / 4), rows, 0, false);
        }
        return;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *tile =
        ((threadgroup float *)shmem) + 32 * (simd_group & 1) + (16 * (simd_group >> 1)) * NR0;
    TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
        simdgroup_store(acc[i], tile + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        for (int batch_col = tid; batch_col < nr1; batch_col += NR1) {
            device float *dst = out + (r1 + batch_col) * rows + r0;
            threadgroup float *src = tile + batch_col * NR0;
            threadgroup float4 *src4 = (threadgroup float4 *)src;
            device float4 *dst4 = (device float4 *)dst;
            int row = 0;
            for (; row < nr0 / 4; ++row) {
                dst4[row] = src4[row];
            }
            row *= 4;
            for (; row < nr0; ++row) {
                dst[row] = src[row];
            }
        }
    }
}

kernel void te_q4_0_matmat_pair_swiglu_llama_h_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weights_gate [[buffer(1)]],
    device const uchar *weights_up [[buffer(2)]],
    device half *out [[buffer(3)]],
    constant uint *dims [[buffer(4)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint3 grid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half *sa_gate = (threadgroup half *)(shmem);
    threadgroup half *sa_up = (threadgroup half *)(shmem + 4096);
    threadgroup half *sb = (threadgroup half *)(shmem + 8192);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;

    const int rows = int(dims[0]);
    const int cols = int(dims[1]);
    const int row_bytes = int(dims[2]);
    const int batch = int(dims[3]);
    const int r0 = int(grid.y) * NR0;
    const int r1 = int(grid.x) * NR1;

    const short nr0 = short(min(NR0, max(0, rows - r0)));
    const short nr1 = short(min(NR1, max(0, batch - r1)));
    if (nr0 <= 0 || nr1 <= 0) {
        return;
    }

    const short lr0 = ((short)tid / NL0) < nr0 ? ((short)tid / NL0) : nr0 - 1;
    const short lr1 = ((short)tid / NL1) < nr1 ? ((short)tid / NL1) : nr1 - 1;
    const short il0 = tid % NL0;
    short il = il0;

    device const te_block_q4_0 *gate_block =
        (device const te_block_q4_0 *)(weights_gate + row_bytes * (r0 + lr0));
    device const te_block_q4_0 *up_block =
        (device const te_block_q4_0 *)(weights_up + row_bytes * (r0 + lr0));
    const short iy = 8 * (tid % NL1);
    device const float *x_ptr = x + (r1 + lr1) * cols + iy;

    simdgroup_half8x8 ma_gate[4];
    simdgroup_half8x8 ma_up[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 acc_gate[8];
    simdgroup_float8x8 acc_up[8];
    TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
        acc_gate[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        acc_up[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < cols; loop_k += NK) {
        half4x4 gate_a;
        half4x4 up_a;
        te_dequantize_q4_0_mm(gate_block, il, gate_a);
        te_dequantize_q4_0_mm(up_block, il, up_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        TE_METAL_UNROLL for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tid / NL0) / 8;
            const short lx = (tid / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            const half gate_value = (loop_k + 16 * il + i < cols) ? gate_a[i / 4][i % 4] : half(0.0f);
            const half up_value = (loop_k + 16 * il + i < cols) ? up_a[i / 4][i % 4] : half(0.0f);
            *(sa_gate + 64 * ib + 8 * ly + lx) = gate_value;
            *(sa_up + 64 * ib + 8 * ly + lx) = up_value;
        }

        const short sx_b = tid % NL1;
        const short sy_b = (tid / NL1) / 8;
        const short ly_b = (tid / NL1) % 8;
        const short ib_b = 4 * sx_b + sy_b;
        threadgroup half2x4 *dst_b = (threadgroup half2x4 *)(sb + 64 * ib_b + 8 * ly_b);
        *dst_b = half2x4(*((device const float2x4 *)x_ptr));

        il = (il + 2 < 2) ? il + 2 : il % 2;
        if (il < 2) {
            gate_block += 1;
            up_block += 1;
        }
        x_ptr += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma_gate = sa_gate + 4 * 64 * (simd_group % 2);
        threadgroup const half *lsma_up = sa_up + 4 * 64 * (simd_group % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (simd_group / 2);

        TE_METAL_UNROLL for (short ik = 0; ik < NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_gate[i], lsma_gate + 64 * i, 8, 0, false);
                simdgroup_load(ma_up[i], lsma_up + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(acc_gate[i], mb[i / 4], ma_gate[i % 4], acc_gate[i]);
                simdgroup_multiply_accumulate(acc_up[i], mb[i / 4], ma_up[i % 4], acc_up[i]);
            }
            lsma_gate += 8 * 64;
            lsma_up += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *gate_tile_base = (threadgroup float *)shmem;
    threadgroup float *up_tile_base = gate_tile_base + NR0 * NR1;
    threadgroup float *gate_tile =
        gate_tile_base + 32 * (simd_group & 1) + (16 * (simd_group >> 1)) * NR0;
    threadgroup float *up_tile =
        up_tile_base + 32 * (simd_group & 1) + (16 * (simd_group >> 1)) * NR0;
    TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
        simdgroup_store(acc_gate[i], gate_tile + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
        simdgroup_store(acc_up[i], up_tile + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        for (int batch_col = tid; batch_col < nr1; batch_col += NR1) {
            const int batch_index = r1 + batch_col;
            threadgroup float *gate_src = gate_tile_base + batch_col * NR0;
            threadgroup float *up_src = up_tile_base + batch_col * NR0;
            device half *dst = out + batch_index * rows + r0;
            int row = 0;
            if ((rows & 3) == 0) {
                threadgroup float4 *gate4 = (threadgroup float4 *)gate_src;
                threadgroup float4 *up4 = (threadgroup float4 *)up_src;
                device half4 *dst4 = (device half4 *)dst;
                for (; row < nr0 / 4; ++row) {
                    const float4 g = gate4[row];
                    dst4[row] = half4((g / (1.0f + exp(-g))) * up4[row]);
                }
                row *= 4;
            }
            for (; row < nr0; ++row) {
                const float g = gate_src[row];
                dst[row] = half((g / (1.0f + exp(-g))) * up_src[row]);
            }
        }
    }
}

kernel void te_q4_0_matmat_h_llama_f32(
    device const half *x [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint *dims [[buffer(3)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint3 grid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;

    const int rows = int(dims[0]);
    const int cols = int(dims[1]);
    const int row_bytes = int(dims[2]);
    const int batch = int(dims[3]);
    const int r0 = int(grid.y) * NR0;
    const int r1 = int(grid.x) * NR1;

    const short nr0 = short(min(NR0, max(0, rows - r0)));
    const short nr1 = short(min(NR1, max(0, batch - r1)));
    if (nr0 <= 0 || nr1 <= 0) {
        return;
    }

    const short lr0 = ((short)tid / NL0) < nr0 ? ((short)tid / NL0) : nr0 - 1;
    const short lr1 = ((short)tid / NL1) < nr1 ? ((short)tid / NL1) : nr1 - 1;
    const short il0 = tid % NL0;
    short il = il0;

    device const te_block_q4_0 *weight_block =
        (device const te_block_q4_0 *)(weights + row_bytes * (r0 + lr0));
    const short iy = 8 * (tid % NL1);
    device const half *x_ptr = x + (r1 + lr1) * cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 acc[8];
    TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
        acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < cols; loop_k += NK) {
        half4x4 temp_a;
        te_dequantize_q4_0_mm(weight_block, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        TE_METAL_UNROLL for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tid / NL0) / 8;
            const short lx = (tid / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) =
                (loop_k + 16 * il + i < cols) ? temp_a[i / 4][i % 4] : half(0.0f);
        }

        const short sx_b = tid % NL1;
        const short sy_b = (tid / NL1) / 8;
        const short ly_b = (tid / NL1) % 8;
        const short ib_b = 4 * sx_b + sy_b;
        threadgroup half2x4 *dst_b = (threadgroup half2x4 *)(sb + 64 * ib_b + 8 * ly_b);
        *dst_b = *((device const half2x4 *)x_ptr);

        il = (il + 2 < 2) ? il + 2 : il % 2;
        if (il < 2) {
            weight_block += 1;
        }
        x_ptr += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + 4 * 64 * (simd_group % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (simd_group / 2);

        TE_METAL_UNROLL for (short ik = 0; ik < NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(acc[i], mb[i / 4], ma[i % 4], acc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (r0 + NR0 <= rows && r1 + NR1 <= batch) {
        device float *dst = out + (r0 + 32 * (simd_group & 1)) + (r1 + 16 * (simd_group >> 1)) * rows;
        TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
            simdgroup_store(acc[i], dst + 8 * (i % 4) + 8 * rows * (i / 4), rows, 0, false);
        }
        return;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *tile =
        ((threadgroup float *)shmem) + 32 * (simd_group & 1) + (16 * (simd_group >> 1)) * NR0;
    TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
        simdgroup_store(acc[i], tile + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        for (int batch_col = tid; batch_col < nr1; batch_col += NR1) {
            device float *dst = out + (r1 + batch_col) * rows + r0;
            threadgroup float *src = tile + batch_col * NR0;
            threadgroup float4 *src4 = (threadgroup float4 *)src;
            device float4 *dst4 = (device float4 *)dst;
            int row = 0;
            for (; row < nr0 / 4; ++row) {
                dst4[row] = src4[row];
            }
            row *= 4;
            for (; row < nr0; ++row) {
                dst[row] = src[row];
            }
        }
    }
}

kernel void te_q4_0_matmat_h_add_llama_f32(
    device const half *x [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *residual [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint *dims [[buffer(4)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint3 grid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;

    const int rows = int(dims[0]);
    const int cols = int(dims[1]);
    const int row_bytes = int(dims[2]);
    const int batch = int(dims[3]);
    const int r0 = int(grid.y) * NR0;
    const int r1 = int(grid.x) * NR1;

    const short nr0 = short(min(NR0, max(0, rows - r0)));
    const short nr1 = short(min(NR1, max(0, batch - r1)));
    if (nr0 <= 0 || nr1 <= 0) {
        return;
    }

    const short lr0 = ((short)tid / NL0) < nr0 ? ((short)tid / NL0) : nr0 - 1;
    const short lr1 = ((short)tid / NL1) < nr1 ? ((short)tid / NL1) : nr1 - 1;
    const short il0 = tid % NL0;
    short il = il0;

    device const te_block_q4_0 *weight_block =
        (device const te_block_q4_0 *)(weights + row_bytes * (r0 + lr0));
    const short iy = 8 * (tid % NL1);
    device const half *x_ptr = x + (r1 + lr1) * cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 acc[8];
    TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
        acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < cols; loop_k += NK) {
        half4x4 temp_a;
        te_dequantize_q4_0_mm(weight_block, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        TE_METAL_UNROLL for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tid / NL0) / 8;
            const short lx = (tid / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) =
                (loop_k + 16 * il + i < cols) ? temp_a[i / 4][i % 4] : half(0.0f);
        }

        const short sx_b = tid % NL1;
        const short sy_b = (tid / NL1) / 8;
        const short ly_b = (tid / NL1) % 8;
        const short ib_b = 4 * sx_b + sy_b;
        threadgroup half2x4 *dst_b = (threadgroup half2x4 *)(sb + 64 * ib_b + 8 * ly_b);
        *dst_b = *((device const half2x4 *)x_ptr);

        il = (il + 2 < 2) ? il + 2 : il % 2;
        if (il < 2) {
            weight_block += 1;
        }
        x_ptr += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + 4 * 64 * (simd_group % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (simd_group / 2);

        TE_METAL_UNROLL for (short ik = 0; ik < NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(acc[i], mb[i / 4], ma[i % 4], acc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *tile =
        ((threadgroup float *)shmem) + 32 * (simd_group & 1) + (16 * (simd_group >> 1)) * NR0;
    TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
        simdgroup_store(acc[i], tile + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        for (int batch_col = tid; batch_col < nr1; batch_col += NR1) {
            const int batch_index = r1 + batch_col;
            device const float *residual_src = residual + batch_index * rows + r0;
            device float *dst = out + batch_index * rows + r0;
            threadgroup float *src = tile + batch_col * NR0;
            int row = 0;
            if ((rows & 3) == 0) {
                threadgroup float4 *src4 = (threadgroup float4 *)src;
                device const float4 *residual4 = (device const float4 *)residual_src;
                device float4 *dst4 = (device float4 *)dst;
                for (; row < nr0 / 4; ++row) {
                    dst4[row] = residual4[row] + src4[row];
                }
                row *= 4;
            }
            for (; row < nr0; ++row) {
                dst[row] = residual_src[row] + src[row];
            }
        }
    }
}

kernel void te_q4_0_matmat_store_h_llama_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant uint *dims [[buffer(3)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint3 grid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;

    const int rows = int(dims[0]);
    const int cols = int(dims[1]);
    const int row_bytes = int(dims[2]);
    const int batch = int(dims[3]);
    const int r0 = int(grid.y) * NR0;
    const int r1 = int(grid.x) * NR1;

    const short nr0 = short(min(NR0, max(0, rows - r0)));
    const short nr1 = short(min(NR1, max(0, batch - r1)));
    if (nr0 <= 0 || nr1 <= 0) {
        return;
    }

    const short lr0 = ((short)tid / NL0) < nr0 ? ((short)tid / NL0) : nr0 - 1;
    const short lr1 = ((short)tid / NL1) < nr1 ? ((short)tid / NL1) : nr1 - 1;
    const short il0 = tid % NL0;
    short il = il0;

    device const te_block_q4_0 *weight_block =
        (device const te_block_q4_0 *)(weights + row_bytes * (r0 + lr0));
    const short iy = 8 * (tid % NL1);
    device const float *x_ptr = x + (r1 + lr1) * cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 acc[8];
    TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
        acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < cols; loop_k += NK) {
        half4x4 temp_a;
        te_dequantize_q4_0_mm(weight_block, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        TE_METAL_UNROLL for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tid / NL0) / 8;
            const short lx = (tid / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) =
                (loop_k + 16 * il + i < cols) ? temp_a[i / 4][i % 4] : half(0.0f);
        }

        const short sx_b = tid % NL1;
        const short sy_b = (tid / NL1) / 8;
        const short ly_b = (tid / NL1) % 8;
        const short ib_b = 4 * sx_b + sy_b;
        threadgroup half2x4 *dst_b = (threadgroup half2x4 *)(sb + 64 * ib_b + 8 * ly_b);
        *dst_b = half2x4(*((device const float2x4 *)x_ptr));

        il = (il + 2 < 2) ? il + 2 : il % 2;
        if (il < 2) {
            weight_block += 1;
        }
        x_ptr += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + 4 * 64 * (simd_group % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (simd_group / 2);

        TE_METAL_UNROLL for (short ik = 0; ik < NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(acc[i], mb[i / 4], ma[i % 4], acc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *tile =
        ((threadgroup float *)shmem) + 32 * (simd_group & 1) + (16 * (simd_group >> 1)) * NR0;
    TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
        simdgroup_store(acc[i], tile + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        for (int batch_col = tid; batch_col < nr1; batch_col += NR1) {
            device half *dst = out + (r1 + batch_col) * rows + r0;
            threadgroup float *src = tile + batch_col * NR0;
            int row = 0;
            if ((rows & 3) == 0) {
                threadgroup float4 *src4 = (threadgroup float4 *)src;
                device half4 *dst4 = (device half4 *)dst;
                for (; row < nr0 / 4; ++row) {
                    dst4[row] = half4(src4[row]);
                }
                row *= 4;
            }
            for (; row < nr0; ++row) {
                dst[row] = half(src[row]);
            }
        }
    }
}

kernel void te_q4_0_matmat_swiglu_llama_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *gate [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint *dims [[buffer(4)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint3 grid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;

    const int rows = int(dims[0]);
    const int cols = int(dims[1]);
    const int row_bytes = int(dims[2]);
    const int batch = int(dims[3]);
    const int r0 = int(grid.y) * NR0;
    const int r1 = int(grid.x) * NR1;

    const short nr0 = short(min(NR0, max(0, rows - r0)));
    const short nr1 = short(min(NR1, max(0, batch - r1)));
    if (nr0 <= 0 || nr1 <= 0) {
        return;
    }

    const short lr0 = ((short)tid / NL0) < nr0 ? ((short)tid / NL0) : nr0 - 1;
    const short lr1 = ((short)tid / NL1) < nr1 ? ((short)tid / NL1) : nr1 - 1;
    const short il0 = tid % NL0;
    short il = il0;

    device const te_block_q4_0 *weight_block =
        (device const te_block_q4_0 *)(weights + row_bytes * (r0 + lr0));
    const short iy = 8 * (tid % NL1);
    device const float *x_ptr = x + (r1 + lr1) * cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 acc[8];
    TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
        acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < cols; loop_k += NK) {
        half4x4 temp_a;
        te_dequantize_q4_0_mm(weight_block, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        TE_METAL_UNROLL for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tid / NL0) / 8;
            const short lx = (tid / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) =
                (loop_k + 16 * il + i < cols) ? temp_a[i / 4][i % 4] : half(0.0f);
        }

        const short sx_b = tid % NL1;
        const short sy_b = (tid / NL1) / 8;
        const short ly_b = (tid / NL1) % 8;
        const short ib_b = 4 * sx_b + sy_b;
        threadgroup half2x4 *dst_b = (threadgroup half2x4 *)(sb + 64 * ib_b + 8 * ly_b);
        *dst_b = half2x4(*((device const float2x4 *)x_ptr));

        il = (il + 2 < 2) ? il + 2 : il % 2;
        if (il < 2) {
            weight_block += 1;
        }
        x_ptr += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + 4 * 64 * (simd_group % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (simd_group / 2);

        TE_METAL_UNROLL for (short ik = 0; ik < NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(acc[i], mb[i / 4], ma[i % 4], acc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *tile =
        ((threadgroup float *)shmem) + 32 * (simd_group & 1) + (16 * (simd_group >> 1)) * NR0;
    TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
        simdgroup_store(acc[i], tile + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        for (int batch_col = tid; batch_col < nr1; batch_col += NR1) {
            const int batch_index = r1 + batch_col;
            threadgroup float *src = tile + batch_col * NR0;
            device const float *gate_src = gate + batch_index * rows + r0;
            device float *dst = out + batch_index * rows + r0;
            int row = 0;
            if ((rows & 3) == 0) {
                threadgroup float4 *src4 = (threadgroup float4 *)src;
                device const float4 *gate4 = (device const float4 *)gate_src;
                device float4 *dst4 = (device float4 *)dst;
                for (; row < nr0 / 4; ++row) {
                    const float4 g = gate4[row];
                    dst4[row] = (g / (1.0f + exp(-g))) * src4[row];
                }
                row *= 4;
            }
            for (; row < nr0; ++row) {
                const float g = gate_src[row];
                dst[row] = (g / (1.0f + exp(-g))) * src[row];
            }
        }
    }
}

kernel void te_q4_0_matmat_swiglu_llama_h_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const float *gate [[buffer(2)]],
    device half *out [[buffer(3)]],
    constant uint *dims [[buffer(4)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint3 grid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;

    const int rows = int(dims[0]);
    const int cols = int(dims[1]);
    const int row_bytes = int(dims[2]);
    const int batch = int(dims[3]);
    const int r0 = int(grid.y) * NR0;
    const int r1 = int(grid.x) * NR1;

    const short nr0 = short(min(NR0, max(0, rows - r0)));
    const short nr1 = short(min(NR1, max(0, batch - r1)));
    if (nr0 <= 0 || nr1 <= 0) {
        return;
    }

    const short lr0 = ((short)tid / NL0) < nr0 ? ((short)tid / NL0) : nr0 - 1;
    const short lr1 = ((short)tid / NL1) < nr1 ? ((short)tid / NL1) : nr1 - 1;
    const short il0 = tid % NL0;
    short il = il0;

    device const te_block_q4_0 *weight_block =
        (device const te_block_q4_0 *)(weights + row_bytes * (r0 + lr0));
    const short iy = 8 * (tid % NL1);
    device const float *x_ptr = x + (r1 + lr1) * cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 acc[8];
    TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
        acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < cols; loop_k += NK) {
        half4x4 temp_a;
        te_dequantize_q4_0_mm(weight_block, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        TE_METAL_UNROLL for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tid / NL0) / 8;
            const short lx = (tid / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) =
                (loop_k + 16 * il + i < cols) ? temp_a[i / 4][i % 4] : half(0.0f);
        }

        const short sx_b = tid % NL1;
        const short sy_b = (tid / NL1) / 8;
        const short ly_b = (tid / NL1) % 8;
        const short ib_b = 4 * sx_b + sy_b;
        threadgroup half2x4 *dst_b = (threadgroup half2x4 *)(sb + 64 * ib_b + 8 * ly_b);
        *dst_b = half2x4(*((device const float2x4 *)x_ptr));

        il = (il + 2 < 2) ? il + 2 : il % 2;
        if (il < 2) {
            weight_block += 1;
        }
        x_ptr += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + 4 * 64 * (simd_group % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (simd_group / 2);

        TE_METAL_UNROLL for (short ik = 0; ik < NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(acc[i], mb[i / 4], ma[i % 4], acc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *tile =
        ((threadgroup float *)shmem) + 32 * (simd_group & 1) + (16 * (simd_group >> 1)) * NR0;
    TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
        simdgroup_store(acc[i], tile + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        for (int batch_col = tid; batch_col < nr1; batch_col += NR1) {
            const int batch_index = r1 + batch_col;
            threadgroup float *src = tile + batch_col * NR0;
            device const float *gate_src = gate + batch_index * rows + r0;
            device half *dst = out + batch_index * rows + r0;
            int row = 0;
            if ((rows & 3) == 0) {
                threadgroup float4 *src4 = (threadgroup float4 *)src;
                device const float4 *gate4 = (device const float4 *)gate_src;
                device half4 *dst4 = (device half4 *)dst;
                for (; row < nr0 / 4; ++row) {
                    const float4 g = gate4[row];
                    const float4 value = (g / (1.0f + exp(-g))) * src4[row];
                    dst4[row] = half4(value);
                }
                row *= 4;
            }
            for (; row < nr0; ++row) {
                const float g = gate_src[row];
                dst[row] = half((g / (1.0f + exp(-g))) * src[row]);
            }
        }
    }
}

kernel void te_q4_0_matmat_swiglu_gate_h_llama_h_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const half *gate [[buffer(2)]],
    device half *out [[buffer(3)]],
    constant uint *dims [[buffer(4)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint3 grid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;

    const int rows = int(dims[0]);
    const int cols = int(dims[1]);
    const int row_bytes = int(dims[2]);
    const int batch = int(dims[3]);
    const int r0 = int(grid.y) * NR0;
    const int r1 = int(grid.x) * NR1;

    const short nr0 = short(min(NR0, max(0, rows - r0)));
    const short nr1 = short(min(NR1, max(0, batch - r1)));
    if (nr0 <= 0 || nr1 <= 0) {
        return;
    }

    const short lr0 = ((short)tid / NL0) < nr0 ? ((short)tid / NL0) : nr0 - 1;
    const short lr1 = ((short)tid / NL1) < nr1 ? ((short)tid / NL1) : nr1 - 1;
    const short il0 = tid % NL0;
    short il = il0;

    device const te_block_q4_0 *weight_block =
        (device const te_block_q4_0 *)(weights + row_bytes * (r0 + lr0));
    const short iy = 8 * (tid % NL1);
    device const float *x_ptr = x + (r1 + lr1) * cols + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 acc[8];
    TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
        acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < cols; loop_k += NK) {
        half4x4 temp_a;
        te_dequantize_q4_0_mm(weight_block, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        TE_METAL_UNROLL for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tid / NL0) / 8;
            const short lx = (tid / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) =
                (loop_k + 16 * il + i < cols) ? temp_a[i / 4][i % 4] : half(0.0f);
        }

        const short sx_b = tid % NL1;
        const short sy_b = (tid / NL1) / 8;
        const short ly_b = (tid / NL1) % 8;
        const short ib_b = 4 * sx_b + sy_b;
        threadgroup half2x4 *dst_b = (threadgroup half2x4 *)(sb + 64 * ib_b + 8 * ly_b);
        *dst_b = half2x4(*((device const float2x4 *)x_ptr));

        il = (il + 2 < 2) ? il + 2 : il % 2;
        if (il < 2) {
            weight_block += 1;
        }
        x_ptr += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + 4 * 64 * (simd_group % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (simd_group / 2);

        TE_METAL_UNROLL for (short ik = 0; ik < NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(acc[i], mb[i / 4], ma[i % 4], acc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *tile =
        ((threadgroup float *)shmem) + 32 * (simd_group & 1) + (16 * (simd_group >> 1)) * NR0;
    TE_METAL_UNROLL for (short i = 0; i < 8; ++i) {
        simdgroup_store(acc[i], tile + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        for (int batch_col = tid; batch_col < nr1; batch_col += NR1) {
            const int batch_index = r1 + batch_col;
            threadgroup float *src = tile + batch_col * NR0;
            device const half *gate_src = gate + batch_index * rows + r0;
            device half *dst = out + batch_index * rows + r0;
            int row = 0;
            if ((rows & 3) == 0) {
                threadgroup float4 *src4 = (threadgroup float4 *)src;
                device const half4 *gate4 = (device const half4 *)gate_src;
                device half4 *dst4 = (device half4 *)dst;
                for (; row < nr0 / 4; ++row) {
                    const float4 g = float4(gate4[row]);
                    const float4 value = (g / (1.0f + exp(-g))) * src4[row];
                    dst4[row] = half4(value);
                }
                row *= 4;
            }
            for (; row < nr0; ++row) {
                const float g = float(gate_src[row]);
                dst[row] = half((g / (1.0f + exp(-g))) * src[row]);
            }
        }
    }
}

kernel void te_q4_0_matvec_batch_llama_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint *dims [[buffer(3)]],
    uint2 grid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint rows = dims[0];
    const uint cols = dims[1];
    const uint row_bytes = dims[2];
    const uint batch_count = dims[3];
    const uint row_base = (grid.x * TE_Q4_LLAMA_NSG + simd_group) * TE_Q4_BATCH_LLAMA_ROWS_PER_SIMD;
    const uint batch_base = grid.y * TE_Q4_BATCH_LLAMA_TILE;
    if (row_base >= rows || batch_base >= batch_count) {
        return;
    }

    float acc[TE_Q4_BATCH_LLAMA_ROWS_PER_SIMD][TE_Q4_BATCH_LLAMA_TILE];
    for (uint row_lane = 0; row_lane < TE_Q4_BATCH_LLAMA_ROWS_PER_SIMD; ++row_lane) {
        for (uint batch_lane = 0; batch_lane < TE_Q4_BATCH_LLAMA_TILE; ++batch_lane) {
            acc[row_lane][batch_lane] = 0.0f;
        }
    }

    const uint blocks = cols / 32u;
    const uint ix = simd_lane / 2u;
    const uint il = (simd_lane & 1u) * 8u;
    for (uint block = ix; block < blocks; block += 16u) {
        float yl[TE_Q4_BATCH_LLAMA_TILE][16];
        float sumy[TE_Q4_BATCH_LLAMA_TILE];
        for (uint batch_lane = 0; batch_lane < TE_Q4_BATCH_LLAMA_TILE; ++batch_lane) {
            const uint batch = batch_base + batch_lane;
            sumy[batch_lane] = 0.0f;
            device const float *x_block = x + batch * cols + block * 32u + il;
            for (uint i = 0; i < 8; i += 2) {
                const float y0 = batch < batch_count ? x_block[i + 0] : 0.0f;
                const float y1 = batch < batch_count ? x_block[i + 1] : 0.0f;
                const float y16 = batch < batch_count ? x_block[i + 16] : 0.0f;
                const float y17 = batch < batch_count ? x_block[i + 17] : 0.0f;
                sumy[batch_lane] += y0 + y1 + y16 + y17;
                yl[batch_lane][i + 0] = y0;
                yl[batch_lane][i + 1] = y1 / 256.0f;
                yl[batch_lane][i + 8] = y16 / 16.0f;
                yl[batch_lane][i + 9] = y17 / 4096.0f;
            }
        }

        for (uint row_lane = 0; row_lane < TE_Q4_BATCH_LLAMA_ROWS_PER_SIMD; ++row_lane) {
            const uint row = row_base + row_lane;
            if (row < rows) {
                device const uchar *block_ptr = weights + row * row_bytes + block * 18u;
                for (uint batch_lane = 0; batch_lane < TE_Q4_BATCH_LLAMA_TILE; ++batch_lane) {
                    acc[row_lane][batch_lane] += te_q4_0_block_dot_llama(
                        block_ptr,
                        sumy[batch_lane],
                        yl[batch_lane],
                        il);
                }
            }
        }
    }

    for (uint row_lane = 0; row_lane < TE_Q4_BATCH_LLAMA_ROWS_PER_SIMD; ++row_lane) {
        const uint row = row_base + row_lane;
        for (uint batch_lane = 0; batch_lane < TE_Q4_BATCH_LLAMA_TILE; ++batch_lane) {
            const uint batch = batch_base + batch_lane;
            const float sum = simd_sum(acc[row_lane][batch_lane]);
            if (simd_lane == 0u && row < rows && batch < batch_count) {
                out[batch * rows + row] = sum;
            }
        }
    }
}

kernel void te_q4_0_matvec_batch_pair_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weights_a [[buffer(1)]],
    device const uchar *weights_b [[buffer(2)]],
    device float *out_a [[buffer(3)]],
    device float *out_b [[buffer(4)]],
    constant uint *dims [[buffer(5)]],
    uint2 grid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint rows = dims[0];
    const uint cols = dims[1];
    const uint row_bytes = dims[2];
    const uint batch_count = dims[3];
    const uint row_base = grid.x * TE_Q4_BATCH_ROW_TILE;
    const uint batch_base = grid.y * TE_Q4_BATCH_TILE;
    if (row_base >= rows || batch_base >= batch_count) {
        return;
    }

    float acc_a[TE_Q4_BATCH_ROW_TILE][TE_Q4_BATCH_TILE];
    float acc_b[TE_Q4_BATCH_ROW_TILE][TE_Q4_BATCH_TILE];
    for (uint lane = 0; lane < TE_Q4_BATCH_ROW_TILE; ++lane) {
        for (uint batch_lane = 0; batch_lane < TE_Q4_BATCH_TILE; ++batch_lane) {
            acc_a[lane][batch_lane] = 0.0f;
            acc_b[lane][batch_lane] = 0.0f;
        }
    }

    const uint blocks = (cols + 31u) / 32u;
    const uint pairs = blocks * 16u;
    for (uint pair = tid; pair < pairs; pair += TE_Q4_BATCH_THREADS) {
        const uint block = pair / 16u;
        const uint inner = pair - block * 16u;
        const uint lo_col = block * 32u + inner;
        const uint hi_col = lo_col + 16u;
        for (uint lane = 0; lane < TE_Q4_BATCH_ROW_TILE; ++lane) {
            const uint row = row_base + lane;
            if (row < rows) {
                device const uchar *block_a = weights_a + row * row_bytes + block * 18;
                device const uchar *block_b = weights_b + row * row_bytes + block * 18;
                const float scale_a = te_load_f16(block_a);
                const float scale_b = te_load_f16(block_b);
                const uchar packed_a = block_a[2 + inner];
                const uchar packed_b = block_b[2 + inner];
                const float lo_weight_a = ((float(uint(packed_a) & 0x0fu) - 8.0f) * scale_a);
                const float hi_weight_a = ((float(uint(packed_a) >> 4u) - 8.0f) * scale_a);
                const float lo_weight_b = ((float(uint(packed_b) & 0x0fu) - 8.0f) * scale_b);
                const float hi_weight_b = ((float(uint(packed_b) >> 4u) - 8.0f) * scale_b);
                for (uint batch_lane = 0; batch_lane < TE_Q4_BATCH_TILE; ++batch_lane) {
                    const uint batch = batch_base + batch_lane;
                    if (batch < batch_count) {
                        device const float *batch_x = x + batch * cols;
                        if (lo_col < cols) {
                            const float x_value = batch_x[lo_col];
                            acc_a[lane][batch_lane] += x_value * lo_weight_a;
                            acc_b[lane][batch_lane] += x_value * lo_weight_b;
                        }
                        if (hi_col < cols) {
                            const float x_value = batch_x[hi_col];
                            acc_a[lane][batch_lane] += x_value * hi_weight_a;
                            acc_b[lane][batch_lane] += x_value * hi_weight_b;
                        }
                    }
                }
            }
        }
    }

    threadgroup float partial[2 * TE_Q4_BATCH_ROW_TILE * TE_Q4_BATCH_TILE * TE_Q4_BATCH_SIMDGROUPS];
    const uint pair_stride = TE_Q4_BATCH_ROW_TILE * TE_Q4_BATCH_TILE * TE_Q4_BATCH_SIMDGROUPS;
    for (uint lane = 0; lane < TE_Q4_BATCH_ROW_TILE; ++lane) {
        for (uint batch_lane = 0; batch_lane < TE_Q4_BATCH_TILE; ++batch_lane) {
            const float sum_a = simd_sum(acc_a[lane][batch_lane]);
            const float sum_b = simd_sum(acc_b[lane][batch_lane]);
            if (simd_lane == 0u) {
                const uint offset = (lane * TE_Q4_BATCH_TILE + batch_lane) * TE_Q4_BATCH_SIMDGROUPS + simd_group;
                partial[offset] = sum_a;
                partial[pair_stride + offset] = sum_b;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        for (uint lane = 0; lane < TE_Q4_BATCH_ROW_TILE; ++lane) {
            for (uint batch_lane = 0; batch_lane < TE_Q4_BATCH_TILE; ++batch_lane) {
                float sum_a = 0.0f;
                float sum_b = 0.0f;
                for (uint group = 0; group < TE_Q4_BATCH_SIMDGROUPS; ++group) {
                    const uint offset = (lane * TE_Q4_BATCH_TILE + batch_lane) * TE_Q4_BATCH_SIMDGROUPS + group;
                    sum_a += partial[offset];
                    sum_b += partial[pair_stride + offset];
                }
                const uint row = row_base + lane;
                const uint batch = batch_base + batch_lane;
                if (row < rows && batch < batch_count) {
                    out_a[batch * rows + row] = sum_a;
                    out_b[batch * rows + row] = sum_b;
                }
            }
        }
    }
}

kernel void te_q4_0_matvec_batch_pair_llama_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weights_a [[buffer(1)]],
    device const uchar *weights_b [[buffer(2)]],
    device float *out_a [[buffer(3)]],
    device float *out_b [[buffer(4)]],
    constant uint *dims [[buffer(5)]],
    uint2 grid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint rows = dims[0];
    const uint cols = dims[1];
    const uint row_bytes = dims[2];
    const uint batch_count = dims[3];
    const uint row_base = (grid.x * TE_Q4_LLAMA_NSG + simd_group) * TE_Q4_BATCH_LLAMA_ROWS_PER_SIMD;
    const uint batch_base = grid.y * TE_Q4_BATCH_LLAMA_TILE;
    if (row_base >= rows || batch_base >= batch_count) {
        return;
    }

    float acc_a[TE_Q4_BATCH_LLAMA_ROWS_PER_SIMD][TE_Q4_BATCH_LLAMA_TILE];
    float acc_b[TE_Q4_BATCH_LLAMA_ROWS_PER_SIMD][TE_Q4_BATCH_LLAMA_TILE];
    for (uint row_lane = 0; row_lane < TE_Q4_BATCH_LLAMA_ROWS_PER_SIMD; ++row_lane) {
        for (uint batch_lane = 0; batch_lane < TE_Q4_BATCH_LLAMA_TILE; ++batch_lane) {
            acc_a[row_lane][batch_lane] = 0.0f;
            acc_b[row_lane][batch_lane] = 0.0f;
        }
    }

    const uint blocks = cols / 32u;
    const uint ix = simd_lane / 2u;
    const uint il = (simd_lane & 1u) * 8u;
    for (uint block = ix; block < blocks; block += 16u) {
        float yl[TE_Q4_BATCH_LLAMA_TILE][16];
        float sumy[TE_Q4_BATCH_LLAMA_TILE];
        for (uint batch_lane = 0; batch_lane < TE_Q4_BATCH_LLAMA_TILE; ++batch_lane) {
            const uint batch = batch_base + batch_lane;
            sumy[batch_lane] = 0.0f;
            device const float *x_block = x + batch * cols + block * 32u + il;
            for (uint i = 0; i < 8; i += 2) {
                const float y0 = batch < batch_count ? x_block[i + 0] : 0.0f;
                const float y1 = batch < batch_count ? x_block[i + 1] : 0.0f;
                const float y16 = batch < batch_count ? x_block[i + 16] : 0.0f;
                const float y17 = batch < batch_count ? x_block[i + 17] : 0.0f;
                sumy[batch_lane] += y0 + y1 + y16 + y17;
                yl[batch_lane][i + 0] = y0;
                yl[batch_lane][i + 1] = y1 / 256.0f;
                yl[batch_lane][i + 8] = y16 / 16.0f;
                yl[batch_lane][i + 9] = y17 / 4096.0f;
            }
        }

        for (uint row_lane = 0; row_lane < TE_Q4_BATCH_LLAMA_ROWS_PER_SIMD; ++row_lane) {
            const uint row = row_base + row_lane;
            if (row < rows) {
                device const uchar *block_a = weights_a + row * row_bytes + block * 18u;
                device const uchar *block_b = weights_b + row * row_bytes + block * 18u;
                for (uint batch_lane = 0; batch_lane < TE_Q4_BATCH_LLAMA_TILE; ++batch_lane) {
                    acc_a[row_lane][batch_lane] += te_q4_0_block_dot_llama(
                        block_a,
                        sumy[batch_lane],
                        yl[batch_lane],
                        il);
                    acc_b[row_lane][batch_lane] += te_q4_0_block_dot_llama(
                        block_b,
                        sumy[batch_lane],
                        yl[batch_lane],
                        il);
                }
            }
        }
    }

    for (uint row_lane = 0; row_lane < TE_Q4_BATCH_LLAMA_ROWS_PER_SIMD; ++row_lane) {
        const uint row = row_base + row_lane;
        for (uint batch_lane = 0; batch_lane < TE_Q4_BATCH_LLAMA_TILE; ++batch_lane) {
            const uint batch = batch_base + batch_lane;
            const float sum_a = simd_sum(acc_a[row_lane][batch_lane]);
            const float sum_b = simd_sum(acc_b[row_lane][batch_lane]);
            if (simd_lane == 0u && row < rows && batch < batch_count) {
                out_a[batch * rows + row] = sum_a;
                out_b[batch * rows + row] = sum_b;
            }
        }
    }
}

kernel void te_q8_0_matvec_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint *dims [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint rows = dims[0];
    const uint cols = dims[1];
    const uint row_bytes = dims[2];
    if (row >= rows) {
        return;
    }

    float acc = 0.0f;
    for (uint col = tid; col < cols; col += TE_THREADS) {
        const uint block = col / 32;
        const uint inner = col - block * 32;
        device const uchar *block_ptr = weights + row * row_bytes + block * 34;
        const float scale = te_load_f16(block_ptr);
        const char q = *(device const char *)(block_ptr + 2 + inner);
        acc += x[col] * float(q) * scale;
    }
    threadgroup float partial[TE_SIMDGROUPS];
    const float sum = simd_sum(acc);
    if (simd_lane == 0u) {
        partial[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float row_sum = 0.0f;
        for (uint group = 0; group < TE_SIMDGROUPS; ++group) {
            row_sum += partial[group];
        }
        out[row] = row_sum;
    }
}

kernel void te_q8_0_matvec_llama_f32(
    device const float *x [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint *dims [[buffer(3)]],
    threadgroup char *shmem [[threadgroup(0)]],
    uint row_group [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint rows = dims[0];
    const uint cols = dims[1];
    const uint row_bytes = dims[2];
    const uint row_base = row_group * TE_Q8_LLAMA_ROW_TILE;
    if (row_base >= rows) {
        return;
    }

    constexpr ushort NW = 32;
    constexpr ushort NQ = 8;
    const uint block_count = cols / 32;
    const ushort ix = simd_lane / (NW / NQ);
    const ushort il = simd_lane % (NW / NQ);
    const uint first_block = simd_group * NQ + ix;
    device const float *x_block = x + first_block * 32u + il * NQ;
    float sumf[TE_Q8_LLAMA_ROW_TILE] = {0.0f, 0.0f};

    for (uint block = first_block; block < block_count; block += TE_Q8_LLAMA_NSG * NQ) {
        float yl[NQ];
        for (ushort i = 0; i < NQ; ++i) {
            yl[i] = x_block[i];
        }
        for (ushort row_lane = 0; row_lane < TE_Q8_LLAMA_ROW_TILE; ++row_lane) {
            const uint row = row_base + row_lane;
            if (row >= rows) {
                continue;
            }
            device const uchar *block_ptr = weights + row * row_bytes + block * 34u;
            const float scale = te_load_f16(block_ptr);
            device const char *qs = (device const char *)(block_ptr + 2u + il * NQ);
            float sumq = 0.0f;
            for (ushort i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row_lane] += sumq * scale;
        }
        x_block += TE_Q8_LLAMA_NSG * NQ * 32u;
    }

    threadgroup float *partial = (threadgroup float *)shmem;
    for (ushort row_lane = 0; row_lane < TE_Q8_LLAMA_ROW_TILE; ++row_lane) {
        threadgroup float *row_partial = partial + row_lane * NW;
        if (simd_group == 0) {
            row_partial[simd_lane] = 0.0f;
        }
        sumf[row_lane] = simd_sum(sumf[row_lane]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort row_lane = 0; row_lane < TE_Q8_LLAMA_ROW_TILE; ++row_lane) {
        if (simd_lane == 0) {
            partial[row_lane * NW + simd_group] = sumf[row_lane];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort row_lane = 0; row_lane < TE_Q8_LLAMA_ROW_TILE; ++row_lane) {
        const uint row = row_base + row_lane;
        if (row >= rows) {
            continue;
        }
        const float total = simd_sum(partial[row_lane * NW + simd_lane]);
        if (simd_lane == 0 && simd_group == 0) {
            out[row] = total;
        }
    }
}

kernel void te_argmax_blocks_f32(
    device const float *values [[buffer(0)]],
    device float *block_values [[buffer(1)]],
    device uint *block_indices [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint index = block * TE_ARGMAX_THREADS + tid;
    float best_value = -3.402823466e+38f;
    uint best_index = 0xffffffffu;
    if (index < len) {
        best_value = values[index];
        best_index = index;
    }

    threadgroup float local_values[TE_ARGMAX_THREADS];
    threadgroup uint local_indices[TE_ARGMAX_THREADS];
    local_values[tid] = best_value;
    local_indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = TE_ARGMAX_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float rhs_value = local_values[tid + stride];
            const uint rhs_index = local_indices[tid + stride];
            const float lhs_value = local_values[tid];
            const uint lhs_index = local_indices[tid];
            if (rhs_value > lhs_value || (rhs_value == lhs_value && rhs_index < lhs_index)) {
                local_values[tid] = rhs_value;
                local_indices[tid] = rhs_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        block_values[block] = local_values[0];
        block_indices[block] = local_indices[0];
    }
}

kernel void te_argmax_finish_f32(
    device const float *block_values [[buffer(0)]],
    device const uint *block_indices [[buffer(1)]],
    device uint *out_index [[buffer(2)]],
    constant uint &block_count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    float best_value = -3.402823466e+38f;
    uint best_index = 0xffffffffu;
    for (uint index = tid; index < block_count; index += TE_ARGMAX_THREADS) {
        const float value = block_values[index];
        const uint candidate = block_indices[index];
        if (value > best_value || (value == best_value && candidate < best_index)) {
            best_value = value;
            best_index = candidate;
        }
    }

    threadgroup float local_values[TE_ARGMAX_THREADS];
    threadgroup uint local_indices[TE_ARGMAX_THREADS];
    local_values[tid] = best_value;
    local_indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = TE_ARGMAX_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float rhs_value = local_values[tid + stride];
            const uint rhs_index = local_indices[tid + stride];
            const float lhs_value = local_values[tid];
            const uint lhs_index = local_indices[tid];
            if (rhs_value > lhs_value || (rhs_value == lhs_value && rhs_index < lhs_index)) {
                local_values[tid] = rhs_value;
                local_indices[tid] = rhs_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        out_index[0] = local_indices[0];
    }
}

kernel void te_swiglu_f32(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < len) {
        const float g = gate[gid];
        out[gid] = (g / (1.0f + exp(-g))) * up[gid];
    }
}

kernel void te_add_f32_kernel(
    device const float *lhs [[buffer(0)]],
    device const float *rhs [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < len) {
        out[gid] = lhs[gid] + rhs[gid];
    }
}

kernel void te_rmsnorm_batch_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint *dims [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    uint batch [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint hidden = dims[0];
    const uint batch_count = dims[1];
    if (batch >= batch_count) {
        return;
    }
    device const float *batch_input = input + batch * hidden;
    device float *batch_out = out + batch * hidden;
    float acc = 0.0f;
    for (uint index = tid; index < hidden; index += TE_THREADS) {
        const float value = batch_input[index];
        acc += value * value;
    }

    threadgroup float partial[TE_SIMDGROUPS];
    const float sum = simd_sum(acc);
    if (simd_lane == 0u) {
        partial[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint group = 0; group < TE_SIMDGROUPS; ++group) {
            total += partial[group];
        }
        partial[0] = 1.0f / sqrt((total / float(hidden)) + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = partial[0];
    for (uint index = tid; index < hidden; index += TE_THREADS) {
        batch_out[index] = batch_input[index] * scale * weight[index];
    }
}

kernel void te_copy_f32_kernel(
    device const float *input [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < len) {
        out[gid] = input[gid];
    }
}

kernel void te_qkv_bias_rope_cache_f32(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device float *v [[buffer(2)]],
    device const float *q_bias [[buffer(3)]],
    device const float *k_bias [[buffer(4)]],
    device const float *v_bias [[buffer(5)]],
    device const float *rope_cos [[buffer(6)]],
    device const float *rope_sin [[buffer(7)]],
    device float *key_cache [[buffer(8)]],
    device float *value_cache [[buffer(9)]],
    constant uint *dims [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint hidden = dims[0];
    const uint kv = dims[1];
    const uint heads = dims[2];
    const uint kv_heads = dims[3];
    const uint head_dim = dims[4];
    const uint position = dims[5];
    const uint rope_half = head_dim / 2u;
    const uint q_pairs = heads * rope_half;
    const uint k_pairs = kv_heads * rope_half;
    const uint cache_offset = position * kv;

    if (gid < q_pairs) {
        const uint head = gid / rope_half;
        const uint dim = gid - head * rope_half;
        const uint offset = head * head_dim + dim;
        if (offset + rope_half < hidden) {
            const float a = q[offset] + q_bias[offset];
            const float b = q[offset + rope_half] + q_bias[offset + rope_half];
            const float c = rope_cos[dim];
            const float s = rope_sin[dim];
            q[offset] = a * c - b * s;
            q[offset + rope_half] = b * c + a * s;
        }
    }

    if (gid < k_pairs) {
        const uint head = gid / rope_half;
        const uint dim = gid - head * rope_half;
        const uint offset = head * head_dim + dim;
        if (offset + rope_half < kv) {
            const float a = k[offset] + k_bias[offset];
            const float b = k[offset + rope_half] + k_bias[offset + rope_half];
            const float c = rope_cos[dim];
            const float s = rope_sin[dim];
            const float lo = a * c - b * s;
            const float hi = b * c + a * s;
            k[offset] = lo;
            k[offset + rope_half] = hi;
            key_cache[cache_offset + offset] = lo;
            key_cache[cache_offset + offset + rope_half] = hi;
        }
    }

    if (gid < kv) {
        const float value = v[gid] + v_bias[gid];
        v[gid] = value;
        value_cache[cache_offset + gid] = value;
    }
}

kernel void te_qkv_bias_rope_cache_batch_f32(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device float *v [[buffer(2)]],
    device const float *q_bias [[buffer(3)]],
    device const float *k_bias [[buffer(4)]],
    device const float *v_bias [[buffer(5)]],
    device const float *rope_cos [[buffer(6)]],
    device const float *rope_sin [[buffer(7)]],
    device float *key_cache [[buffer(8)]],
    device float *value_cache [[buffer(9)]],
    constant uint *dims [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint batch_count = dims[0];
    const uint base_position = dims[1];
    const uint hidden = dims[2];
    const uint kv = dims[3];
    const uint heads = dims[4];
    const uint kv_heads = dims[5];
    const uint head_dim = dims[6];
    const uint max_items = dims[7];
    if (max_items == 0u) {
        return;
    }
    const uint batch = gid / max_items;
    const uint item = gid - batch * max_items;
    if (batch >= batch_count) {
        return;
    }

    const uint rope_half = head_dim / 2u;
    const uint q_pairs = heads * rope_half;
    const uint k_pairs = kv_heads * rope_half;
    const uint position = base_position + batch;
    const uint q_batch_offset = batch * hidden;
    const uint kv_batch_offset = batch * kv;
    const uint cache_offset = position * kv;
    device const float *cos_row = rope_cos + position * rope_half;
    device const float *sin_row = rope_sin + position * rope_half;

    if (item < q_pairs) {
        const uint head = item / rope_half;
        const uint dim = item - head * rope_half;
        const uint offset = head * head_dim + dim;
        if (offset + rope_half < hidden) {
            const uint index = q_batch_offset + offset;
            const float a = q[index] + q_bias[offset];
            const float b = q[index + rope_half] + q_bias[offset + rope_half];
            const float c = cos_row[dim];
            const float s = sin_row[dim];
            q[index] = a * c - b * s;
            q[index + rope_half] = b * c + a * s;
        }
    }

    if (item < k_pairs) {
        const uint head = item / rope_half;
        const uint dim = item - head * rope_half;
        const uint offset = head * head_dim + dim;
        if (offset + rope_half < kv) {
            const uint index = kv_batch_offset + offset;
            const float a = k[index] + k_bias[offset];
            const float b = k[index + rope_half] + k_bias[offset + rope_half];
            const float c = cos_row[dim];
            const float s = sin_row[dim];
            const float lo = a * c - b * s;
            const float hi = b * c + a * s;
            k[index] = lo;
            k[index + rope_half] = hi;
            key_cache[cache_offset + offset] = lo;
            key_cache[cache_offset + offset + rope_half] = hi;
        }
    }

    if (item < kv) {
        const float value = v[kv_batch_offset + item] + v_bias[item];
        v[kv_batch_offset + item] = value;
        value_cache[cache_offset + item] = value;
    }
}

kernel void te_qkv_bias_rope_cache_batch_h_f32(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device float *v [[buffer(2)]],
    device const float *q_bias [[buffer(3)]],
    device const float *k_bias [[buffer(4)]],
    device const float *v_bias [[buffer(5)]],
    device const float *rope_cos [[buffer(6)]],
    device const float *rope_sin [[buffer(7)]],
    device float *key_cache [[buffer(8)]],
    device float *value_cache [[buffer(9)]],
    device half *key_cache_h [[buffer(10)]],
    device half *value_cache_h [[buffer(11)]],
    constant uint *dims [[buffer(12)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint batch_count = dims[0];
    const uint base_position = dims[1];
    const uint hidden = dims[2];
    const uint kv = dims[3];
    const uint heads = dims[4];
    const uint kv_heads = dims[5];
    const uint head_dim = dims[6];
    const uint max_items = dims[7];
    if (max_items == 0u) {
        return;
    }
    const uint batch = gid / max_items;
    const uint item = gid - batch * max_items;
    if (batch >= batch_count) {
        return;
    }

    const uint rope_half = head_dim / 2u;
    const uint q_pairs = heads * rope_half;
    const uint k_pairs = kv_heads * rope_half;
    const uint position = base_position + batch;
    const uint q_batch_offset = batch * hidden;
    const uint kv_batch_offset = batch * kv;
    const uint cache_offset = position * kv;
    device const float *cos_row = rope_cos + position * rope_half;
    device const float *sin_row = rope_sin + position * rope_half;

    if (item < q_pairs) {
        const uint head = item / rope_half;
        const uint dim = item - head * rope_half;
        const uint offset = head * head_dim + dim;
        if (offset + rope_half < hidden) {
            const uint index = q_batch_offset + offset;
            const float a = q[index] + q_bias[offset];
            const float b = q[index + rope_half] + q_bias[offset + rope_half];
            const float c = cos_row[dim];
            const float s = sin_row[dim];
            q[index] = a * c - b * s;
            q[index + rope_half] = b * c + a * s;
        }
    }

    if (item < k_pairs) {
        const uint head = item / rope_half;
        const uint dim = item - head * rope_half;
        const uint offset = head * head_dim + dim;
        if (offset + rope_half < kv) {
            const uint index = kv_batch_offset + offset;
            const float a = k[index] + k_bias[offset];
            const float b = k[index + rope_half] + k_bias[offset + rope_half];
            const float c = cos_row[dim];
            const float s = sin_row[dim];
            const float lo = a * c - b * s;
            const float hi = b * c + a * s;
            k[index] = lo;
            k[index + rope_half] = hi;
            key_cache[cache_offset + offset] = lo;
            key_cache[cache_offset + offset + rope_half] = hi;
            key_cache_h[cache_offset + offset] = half(lo);
            key_cache_h[cache_offset + offset + rope_half] = half(hi);
        }
    }

    if (item < kv) {
        const float value = v[kv_batch_offset + item] + v_bias[item];
        v[kv_batch_offset + item] = value;
        value_cache[cache_offset + item] = value;
        value_cache_h[cache_offset + item] = half(value);
    }
}

kernel void te_attention_decode_f32_kernel(
    device const float *query [[buffer(0)]],
    device const float *key_cache [[buffer(1)]],
    device const float *value_cache [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint *dims [[buffer(4)]],
    threadgroup float *scores [[threadgroup(0)]],
    uint2 grid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint batch_count = dims[0];
    const uint base_position = dims[1];
    const uint heads = dims[2];
    const uint kv_heads = dims[3];
    const uint head_dim = dims[4];
    const uint cache_tokens = dims[5];
    const uint batch = grid.x;
    const uint head = grid.y;
    if (batch >= batch_count || head >= heads || kv_heads == 0u) {
        return;
    }
    const uint position = base_position + batch;
    if (position >= cache_tokens) {
        return;
    }

    const uint kv_repeat = heads / kv_heads;
    const uint kv_head = head / kv_repeat;
    const float scale = rsqrt(float(head_dim));
    device const float *head_q = query + batch * heads * head_dim + head * head_dim;
    for (uint past = tid; past <= position; past += TE_THREADS) {
        device const float *past_k = key_cache + (past * kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (uint dim = 0u; dim < head_dim; ++dim) {
            dot += head_q[dim] * past_k[dim];
        }
        scores[past] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float max_score = -3.402823466e+38F;
        for (uint past = 0u; past <= position; ++past) {
            max_score = max(max_score, scores[past]);
        }
        float sum = 0.0f;
        for (uint past = 0u; past <= position; ++past) {
            const float value = exp(scores[past] - max_score);
            scores[past] = value;
            sum += value;
        }
        if (sum > 0.0f) {
            const float inv_sum = 1.0f / sum;
            for (uint past = 0u; past <= position; ++past) {
                scores[past] *= inv_sum;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *head_out = out + batch * heads * head_dim + head * head_dim;
    for (uint dim = tid; dim < head_dim; dim += TE_THREADS) {
        float value = 0.0f;
        for (uint past = 0u; past <= position; ++past) {
            device const float *past_v = value_cache + (past * kv_heads + kv_head) * head_dim;
            value += scores[past] * past_v[dim];
        }
        head_out[dim] = value;
    }
}

kernel void te_attention_flash_f32_kernel(
    device const float *query [[buffer(0)]],
    device const float *key_cache [[buffer(1)]],
    device const float *value_cache [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint *dims [[buffer(4)]],
    uint2 grid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint batch_count = dims[0];
    const uint base_position = dims[1];
    const uint heads = dims[2];
    const uint kv_heads = dims[3];
    const uint head_dim = dims[4];
    const uint cache_tokens = dims[5];
    const uint batch = grid.x;
    const uint head = grid.y;
    if (batch >= batch_count || head >= heads || kv_heads == 0u || (heads % kv_heads) != 0u) {
        return;
    }
    const uint position = base_position + batch;
    if (position >= cache_tokens || head_dim == 0u || head_dim > TE_FLASH_ATTN_THREADS) {
        return;
    }

    threadgroup float scores[TE_FLASH_ATTN_TILE];
    threadgroup float partial[TE_FLASH_ATTN_SIMDGROUPS];
    threadgroup float shared[2];

    const uint kv_repeat = heads / kv_heads;
    const uint kv_head = head / kv_repeat;
    const float scale = rsqrt(float(head_dim));
    device const float *head_q = query + batch * heads * head_dim + head * head_dim;
    device float *head_out = out + batch * heads * head_dim + head * head_dim;

    float m = -3.402823466e+38F;
    float l = 0.0f;
    float acc = 0.0f;

    for (uint block_start = 0u; block_start <= position; block_start += TE_FLASH_ATTN_TILE) {
        for (uint local = simd_group; local < TE_FLASH_ATTN_TILE; local += TE_FLASH_ATTN_SIMDGROUPS) {
            const uint past = block_start + local;
            float dot = 0.0f;
            if (past <= position && past < cache_tokens) {
                device const float *past_k = key_cache + (past * kv_heads + kv_head) * head_dim;
                for (uint dim = simd_lane; dim < head_dim; dim += 32u) {
                    dot += head_q[dim] * past_k[dim];
                }
                dot = simd_sum(dot) * scale;
            } else {
                dot = -3.402823466e+38F;
            }
            if (simd_lane == 0u) {
                scores[local] = dot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_max = -3.402823466e+38F;
        for (uint local = tid; local < TE_FLASH_ATTN_TILE; local += TE_FLASH_ATTN_THREADS) {
            local_max = max(local_max, scores[local]);
        }
        const float group_max = simd_max(local_max);
        if (simd_lane == 0u) {
            partial[simd_group] = group_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float block_max = partial[0];
            for (uint group = 1u; group < TE_FLASH_ATTN_SIMDGROUPS; ++group) {
                block_max = max(block_max, partial[group]);
            }
            shared[0] = max(m, block_max);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float new_m = shared[0];
        float local_sum = 0.0f;
        for (uint local = tid; local < TE_FLASH_ATTN_TILE; local += TE_FLASH_ATTN_THREADS) {
            const float weight = exp(scores[local] - new_m);
            scores[local] = weight;
            local_sum += weight;
        }
        const float group_sum = simd_sum(local_sum);
        if (simd_lane == 0u) {
            partial[simd_group] = group_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float block_sum = 0.0f;
            for (uint group = 0u; group < TE_FLASH_ATTN_SIMDGROUPS; ++group) {
                block_sum += partial[group];
            }
            shared[1] = block_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float alpha = exp(m - new_m);
        if (tid < head_dim) {
            float block_acc = 0.0f;
            for (uint local = 0u; local < TE_FLASH_ATTN_TILE; ++local) {
                const uint past = block_start + local;
                if (past <= position && past < cache_tokens) {
                    device const float *past_v = value_cache + (past * kv_heads + kv_head) * head_dim;
                    block_acc += scores[local] * past_v[tid];
                }
            }
            acc = acc * alpha + block_acc;
        }
        l = l * alpha + shared[1];
        m = new_m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < head_dim) {
        head_out[tid] = l > 0.0f ? acc / l : 0.0f;
    }
}

kernel void te_attention_flash_t128_f32_kernel(
    device const float *query [[buffer(0)]],
    device const float *key_cache [[buffer(1)]],
    device const float *value_cache [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint *dims [[buffer(4)]],
    uint2 grid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint batch_count = dims[0];
    const uint base_position = dims[1];
    const uint heads = dims[2];
    const uint kv_heads = dims[3];
    const uint head_dim = dims[4];
    const uint cache_tokens = dims[5];
    const uint batch = grid.x;
    const uint head = grid.y;
    if (batch >= batch_count || head >= heads || kv_heads == 0u || (heads % kv_heads) != 0u) {
        return;
    }
    const uint position = base_position + batch;
    if (position >= cache_tokens || head_dim == 0u || head_dim > TE_FLASH_ATTN_THREADS) {
        return;
    }

    threadgroup float scores[TE_FLASH_ATTN_TILE_LONG];
    threadgroup float partial[TE_FLASH_ATTN_SIMDGROUPS];
    threadgroup float shared[2];

    const uint kv_repeat = heads / kv_heads;
    const uint kv_head = head / kv_repeat;
    const float scale = rsqrt(float(head_dim));
    device const float *head_q = query + batch * heads * head_dim + head * head_dim;
    device float *head_out = out + batch * heads * head_dim + head * head_dim;

    float m = -3.402823466e+38F;
    float l = 0.0f;
    float acc = 0.0f;

    for (uint block_start = 0u; block_start <= position; block_start += TE_FLASH_ATTN_TILE_LONG) {
        for (uint local = simd_group; local < TE_FLASH_ATTN_TILE_LONG; local += TE_FLASH_ATTN_SIMDGROUPS) {
            const uint past = block_start + local;
            float dot = 0.0f;
            if (past <= position && past < cache_tokens) {
                device const float *past_k = key_cache + (past * kv_heads + kv_head) * head_dim;
                for (uint dim = simd_lane; dim < head_dim; dim += 32u) {
                    dot += head_q[dim] * past_k[dim];
                }
                dot = simd_sum(dot) * scale;
            } else {
                dot = -3.402823466e+38F;
            }
            if (simd_lane == 0u) {
                scores[local] = dot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_max = -3.402823466e+38F;
        for (uint local = tid; local < TE_FLASH_ATTN_TILE_LONG; local += TE_FLASH_ATTN_THREADS) {
            local_max = max(local_max, scores[local]);
        }
        const float group_max = simd_max(local_max);
        if (simd_lane == 0u) {
            partial[simd_group] = group_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float block_max = partial[0];
            for (uint group = 1u; group < TE_FLASH_ATTN_SIMDGROUPS; ++group) {
                block_max = max(block_max, partial[group]);
            }
            shared[0] = max(m, block_max);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float new_m = shared[0];
        float local_sum = 0.0f;
        for (uint local = tid; local < TE_FLASH_ATTN_TILE_LONG; local += TE_FLASH_ATTN_THREADS) {
            const float weight = exp(scores[local] - new_m);
            scores[local] = weight;
            local_sum += weight;
        }
        const float group_sum = simd_sum(local_sum);
        if (simd_lane == 0u) {
            partial[simd_group] = group_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float block_sum = 0.0f;
            for (uint group = 0u; group < TE_FLASH_ATTN_SIMDGROUPS; ++group) {
                block_sum += partial[group];
            }
            shared[1] = block_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float alpha = exp(m - new_m);
        if (tid < head_dim) {
            float block_acc = 0.0f;
            for (uint local = 0u; local < TE_FLASH_ATTN_TILE_LONG; ++local) {
                const uint past = block_start + local;
                if (past <= position && past < cache_tokens) {
                    device const float *past_v = value_cache + (past * kv_heads + kv_head) * head_dim;
                    block_acc += scores[local] * past_v[tid];
                }
            }
            acc = acc * alpha + block_acc;
        }
        l = l * alpha + shared[1];
        m = new_m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < head_dim) {
        head_out[tid] = l > 0.0f ? acc / l : 0.0f;
    }
}

kernel void te_attention_flash_gqa_f32_kernel(
    device const float *query [[buffer(0)]],
    device const float *key_cache [[buffer(1)]],
    device const float *value_cache [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint *dims [[buffer(4)]],
    uint2 grid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint batch_count = dims[0];
    const uint base_position = dims[1];
    const uint heads = dims[2];
    const uint kv_heads = dims[3];
    const uint head_dim = dims[4];
    const uint cache_tokens = dims[5];
    const uint batch = grid.x;
    const uint kv_head = grid.y;
    if (batch >= batch_count || kv_head >= kv_heads || kv_heads == 0u || (heads % kv_heads) != 0u) {
        return;
    }
    const uint kv_repeat = heads / kv_heads;
    const uint position = base_position + batch;
    if (kv_repeat == 0u || kv_repeat > TE_FLASH_ATTN_GQA_MAX_REPEAT ||
        position >= cache_tokens || head_dim == 0u || head_dim > TE_FLASH_ATTN_TILE) {
        return;
    }

    threadgroup float q_shared[TE_FLASH_ATTN_GQA_MAX_REPEAT][TE_FLASH_ATTN_TILE];
    threadgroup float scores[TE_FLASH_ATTN_GQA_MAX_REPEAT][TE_FLASH_ATTN_TILE];
    threadgroup float partial[TE_FLASH_ATTN_GQA_MAX_REPEAT][TE_FLASH_ATTN_SIMDGROUPS];
    threadgroup float shared_m[TE_FLASH_ATTN_GQA_MAX_REPEAT];
    threadgroup float shared_l[TE_FLASH_ATTN_GQA_MAX_REPEAT];

    if (tid < head_dim) {
        for (uint r = 0u; r < kv_repeat; ++r) {
            const uint head = kv_head * kv_repeat + r;
            q_shared[r][tid] = query[(batch * heads + head) * head_dim + tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = rsqrt(float(head_dim));
    float m[TE_FLASH_ATTN_GQA_MAX_REPEAT];
    float l[TE_FLASH_ATTN_GQA_MAX_REPEAT];
    float acc[TE_FLASH_ATTN_GQA_MAX_REPEAT];
    for (uint r = 0u; r < kv_repeat; ++r) {
        m[r] = -3.402823466e+38F;
        l[r] = 0.0f;
        acc[r] = 0.0f;
    }

    for (uint block_start = 0u; block_start <= position; block_start += TE_FLASH_ATTN_TILE) {
        for (uint local = simd_group; local < TE_FLASH_ATTN_TILE; local += TE_FLASH_ATTN_SIMDGROUPS) {
            const uint past = block_start + local;
            float dot[TE_FLASH_ATTN_GQA_MAX_REPEAT];
            for (uint r = 0u; r < kv_repeat; ++r) {
                dot[r] = 0.0f;
            }
            if (past <= position && past < cache_tokens) {
                device const float *past_k = key_cache + (past * kv_heads + kv_head) * head_dim;
                for (uint dim = simd_lane; dim < head_dim; dim += 32u) {
                    const float k_value = past_k[dim];
                    for (uint r = 0u; r < kv_repeat; ++r) {
                        dot[r] += q_shared[r][dim] * k_value;
                    }
                }
                for (uint r = 0u; r < kv_repeat; ++r) {
                    dot[r] = simd_sum(dot[r]) * scale;
                }
            } else {
                for (uint r = 0u; r < kv_repeat; ++r) {
                    dot[r] = -3.402823466e+38F;
                }
            }
            if (simd_lane == 0u) {
                for (uint r = 0u; r < kv_repeat; ++r) {
                    scores[r][local] = dot[r];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0u; r < kv_repeat; ++r) {
            float local_max = -3.402823466e+38F;
            for (uint local = tid; local < TE_FLASH_ATTN_TILE; local += TE_FLASH_ATTN_THREADS) {
                local_max = max(local_max, scores[r][local]);
            }
            const float group_max = simd_max(local_max);
            if (simd_lane == 0u) {
                partial[r][simd_group] = group_max;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < kv_repeat) {
            float block_max = partial[tid][0];
            for (uint group = 1u; group < TE_FLASH_ATTN_SIMDGROUPS; ++group) {
                block_max = max(block_max, partial[tid][group]);
            }
            shared_m[tid] = max(m[tid], block_max);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint r = 0u; r < kv_repeat; ++r) {
            const float new_m = shared_m[r];
            float local_sum = 0.0f;
            for (uint local = tid; local < TE_FLASH_ATTN_TILE; local += TE_FLASH_ATTN_THREADS) {
                const float weight = exp(scores[r][local] - new_m);
                scores[r][local] = weight;
                local_sum += weight;
            }
            const float group_sum = simd_sum(local_sum);
            if (simd_lane == 0u) {
                partial[r][simd_group] = group_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < kv_repeat) {
            float block_sum = 0.0f;
            for (uint group = 0u; group < TE_FLASH_ATTN_SIMDGROUPS; ++group) {
                block_sum += partial[tid][group];
            }
            shared_l[tid] = block_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_acc[TE_FLASH_ATTN_GQA_MAX_REPEAT];
        for (uint r = 0u; r < kv_repeat; ++r) {
            block_acc[r] = 0.0f;
        }
        if (tid < head_dim) {
            for (uint local = 0u; local < TE_FLASH_ATTN_TILE; ++local) {
                const uint past = block_start + local;
                if (past <= position && past < cache_tokens) {
                    device const float *past_v = value_cache + (past * kv_heads + kv_head) * head_dim;
                    const float v_value = past_v[tid];
                    for (uint r = 0u; r < kv_repeat; ++r) {
                        block_acc[r] += scores[r][local] * v_value;
                    }
                }
            }
            for (uint r = 0u; r < kv_repeat; ++r) {
                const float alpha = exp(m[r] - shared_m[r]);
                acc[r] = acc[r] * alpha + block_acc[r];
            }
        }
        for (uint r = 0u; r < kv_repeat; ++r) {
            const float alpha = exp(m[r] - shared_m[r]);
            l[r] = l[r] * alpha + shared_l[r];
            m[r] = shared_m[r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < head_dim) {
        for (uint r = 0u; r < kv_repeat; ++r) {
            const uint head = kv_head * kv_repeat + r;
            out[(batch * heads + head) * head_dim + tid] = l[r] > 0.0f ? acc[r] / l[r] : 0.0f;
        }
    }
}

kernel void te_attention_flash_h_f32_kernel(
    device const float *query [[buffer(0)]],
    device const half *key_cache [[buffer(1)]],
    device const half *value_cache [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint *dims [[buffer(4)]],
    uint2 grid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint batch_count = dims[0];
    const uint base_position = dims[1];
    const uint heads = dims[2];
    const uint kv_heads = dims[3];
    const uint head_dim = dims[4];
    const uint cache_tokens = dims[5];
    const uint batch = grid.x;
    const uint head = grid.y;
    if (batch >= batch_count || head >= heads || kv_heads == 0u || (heads % kv_heads) != 0u) {
        return;
    }
    const uint position = base_position + batch;
    if (position >= cache_tokens || head_dim == 0u || head_dim > TE_FLASH_ATTN_THREADS) {
        return;
    }

    threadgroup float scores[TE_FLASH_ATTN_TILE];
    threadgroup float partial[TE_FLASH_ATTN_SIMDGROUPS];
    threadgroup float shared[2];

    const uint kv_repeat = heads / kv_heads;
    const uint kv_head = head / kv_repeat;
    const float scale = rsqrt(float(head_dim));
    device const float *head_q = query + batch * heads * head_dim + head * head_dim;
    device float *head_out = out + batch * heads * head_dim + head * head_dim;

    float m = -3.402823466e+38F;
    float l = 0.0f;
    float acc = 0.0f;

    for (uint block_start = 0u; block_start <= position; block_start += TE_FLASH_ATTN_TILE) {
        for (uint local = simd_group; local < TE_FLASH_ATTN_TILE; local += TE_FLASH_ATTN_SIMDGROUPS) {
            const uint past = block_start + local;
            float dot = 0.0f;
            if (past <= position && past < cache_tokens) {
                device const half *past_k = key_cache + (past * kv_heads + kv_head) * head_dim;
                for (uint dim = simd_lane; dim < head_dim; dim += 32u) {
                    dot += head_q[dim] * float(past_k[dim]);
                }
                dot = simd_sum(dot) * scale;
            } else {
                dot = -3.402823466e+38F;
            }
            if (simd_lane == 0u) {
                scores[local] = dot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_max = -3.402823466e+38F;
        for (uint local = tid; local < TE_FLASH_ATTN_TILE; local += TE_FLASH_ATTN_THREADS) {
            local_max = max(local_max, scores[local]);
        }
        const float group_max = simd_max(local_max);
        if (simd_lane == 0u) {
            partial[simd_group] = group_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float block_max = partial[0];
            for (uint group = 1u; group < TE_FLASH_ATTN_SIMDGROUPS; ++group) {
                block_max = max(block_max, partial[group]);
            }
            shared[0] = max(m, block_max);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float new_m = shared[0];
        float local_sum = 0.0f;
        for (uint local = tid; local < TE_FLASH_ATTN_TILE; local += TE_FLASH_ATTN_THREADS) {
            const float weight = exp(scores[local] - new_m);
            scores[local] = weight;
            local_sum += weight;
        }
        const float group_sum = simd_sum(local_sum);
        if (simd_lane == 0u) {
            partial[simd_group] = group_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float block_sum = 0.0f;
            for (uint group = 0u; group < TE_FLASH_ATTN_SIMDGROUPS; ++group) {
                block_sum += partial[group];
            }
            shared[1] = block_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float alpha = exp(m - new_m);
        if (tid < head_dim) {
            float block_acc = 0.0f;
            for (uint local = 0u; local < TE_FLASH_ATTN_TILE; ++local) {
                const uint past = block_start + local;
                if (past <= position && past < cache_tokens) {
                    device const half *past_v = value_cache + (past * kv_heads + kv_head) * head_dim;
                    block_acc += scores[local] * float(past_v[tid]);
                }
            }
            acc = acc * alpha + block_acc;
        }
        l = l * alpha + shared[1];
        m = new_m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < head_dim) {
        head_out[tid] = l > 0.0f ? acc / l : 0.0f;
    }
}
)TE_METAL"
