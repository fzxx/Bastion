// Bastion SIMD 硬件加速实现（SSE2/AVX2）：把 N 块的状态字打包到同一向量寄存器里，一次处理 N 块
// 列混合用右旋（rotr32），旋转工具按右旋来写

#include "Bastion_internal.h"

#include <cstring>

// SIMD intrinsics：immintrin.h 包含 SSE2/AVX2 全部，intrin.h 提供 __cpuid/_xgetbv
#include <intrin.h>
#include <immintrin.h>

namespace {

// SIMD 能力检测
// 返回 0=不支持，1=SSE2，2=AVX2
// x64 下 SSE2 必然可用，AVX2 需要 cpuid 第 7 叶子 + OSXSAVE + XCR0.AVX/MASK
inline int detectSIMDLevel() {
    int level = 0;
    int cpuInfo[4];

#if defined(_M_X64) || defined(__x86_64__)
    // x64 默认就有 SSE2
    level = 1;
#else
    // x86 通过 cpuid 第 1 叶子 EDX bit 26 检测 SSE2
    __cpuid(cpuInfo, 1);
    if (cpuInfo[3] & (1 << 26)) {
        level = 1;
    }
#endif

    if (level >= 1) {
        // AVX2 检测：cpuid(1).ECX bit 27 (OSXSAVE) + XCR0 bit 1/2 + cpuid(7,0).EBX bit 5
        __cpuid(cpuInfo, 1);
        if ((cpuInfo[2] & (1 << 27)) != 0) {
            unsigned long long xcr0 = _xgetbv(0);
            if ((xcr0 & 0x6) == 0x6) {
                __cpuid(cpuInfo, 0);
                if (cpuInfo[0] >= 7) {
                    __cpuidex(cpuInfo, 7, 0);
                    if (cpuInfo[1] & (1 << 5)) {
                        level = 2;
                    }
                }
            }
        }
    }
    return level;
}

} // namespace

// 进程启动时检测一次，之后只读
const int g_simdLevel = detectSIMDLevel();

namespace {

// SIMD 旋转工具

// SSE2 右旋模板：rotr32(x, R) = srli(x,R) | slli(x, 32-R)
template <unsigned int R>
inline __m128i rotr32_sse2(__m128i x) {
    return _mm_or_si128(_mm_srli_epi32(x, R), _mm_slli_epi32(x, 32 - R));
}

// AVX2 右旋模板，逻辑与 SSE2 相同，宽度 128→256
template <unsigned int R>
inline __m256i rotr32_avx2(__m256i x) {
    return _mm256_or_si256(_mm256_srli_epi32(x, R), _mm256_slli_epi32(x, 32 - R));
}

// SSE2 左旋模板：rotl32(x, R) = slli(x,R) | srli(x, 32-R)，解密逆 ColumnMix 用
template <unsigned int R>
inline __m128i rotl32_sse2(__m128i x) {
    return _mm_or_si128(_mm_slli_epi32(x, R), _mm_srli_epi32(x, 32 - R));
}

// AVX2 左旋模板，宽度 128→256
template <unsigned int R>
inline __m256i rotl32_avx2(__m256i x) {
    return _mm256_or_si256(_mm256_slli_epi32(x, R), _mm256_srli_epi32(x, 32 - R));
}

// 4x4 转置：把 4 块各自的 s0..s3 向量重排为块0..3 的完整状态，unpacklo/hi 解交织
inline void transpose4x4_sse2(__m128i r0, __m128i r1, __m128i r2, __m128i r3,
                              __m128i& o0, __m128i& o1, __m128i& o2, __m128i& o3) {
    __m128i t0 = _mm_unpacklo_epi32(r0, r1);
    __m128i t1 = _mm_unpackhi_epi32(r0, r1);
    __m128i t2 = _mm_unpacklo_epi32(r2, r3);
    __m128i t3 = _mm_unpackhi_epi32(r2, r3);
    o0 = _mm_unpacklo_epi64(t0, t2);
    o1 = _mm_unpackhi_epi64(t0, t2);
    o2 = _mm_unpacklo_epi64(t1, t3);
    o3 = _mm_unpackhi_epi64(t1, t3);
}

// G_Mix/H_Mix 宏
// 语句与 encryptBlockInline 逐行对应，仅变量名 s→r

// G_Mix 宏（SSE2）
#define BASTION_GMIX_SSE2(r0,r1,r2,r3,r4,r5,r6,r7) do { \
    r0 = _mm_add_epi32(r0, rotr32_sse2<GM_R5>(r2)); \
    r1 = _mm_add_epi32(r1, rotr32_sse2<GM_R5>(r3)); \
    r6 = _mm_xor_si128(r6, r0); \
    r7 = _mm_xor_si128(r7, r1); \
    r6 = rotr32_sse2<GM_R1>(r6); \
    r7 = rotr32_sse2<GM_R1>(r7); \
    r4 = _mm_add_epi32(r4, rotr32_sse2<GM_R6>(r6)); \
    r5 = _mm_add_epi32(r5, rotr32_sse2<GM_R6>(r7)); \
    r2 = _mm_xor_si128(r2, r4); \
    r3 = _mm_xor_si128(r3, r5); \
    r2 = rotr32_sse2<GM_R2>(r2); \
    r3 = rotr32_sse2<GM_R2>(r3); \
    r0 = _mm_xor_si128(r0, rotr32_sse2<GM_R9>(r4)); \
    r1 = _mm_xor_si128(r1, rotr32_sse2<GM_R9>(r5)); \
    r0 = _mm_add_epi32(r0, rotr32_sse2<GM_R7>(r2)); \
    r1 = _mm_add_epi32(r1, rotr32_sse2<GM_R7>(r3)); \
    r6 = _mm_xor_si128(r6, r0); \
    r7 = _mm_xor_si128(r7, r1); \
    r6 = rotr32_sse2<GM_R3>(r6); \
    r7 = rotr32_sse2<GM_R3>(r7); \
    r4 = _mm_add_epi32(r4, rotr32_sse2<GM_R8>(r6)); \
    r5 = _mm_add_epi32(r5, rotr32_sse2<GM_R8>(r7)); \
    r2 = _mm_xor_si128(r2, r4); \
    r3 = _mm_xor_si128(r3, r5); \
    r2 = rotr32_sse2<GM_R4>(r2); \
    r3 = rotr32_sse2<GM_R4>(r3); \
} while(0)

// H_Mix 宏（SSE2），旋转常数换 HM_R1..R9
#define BASTION_HMIX_SSE2(r0,r1,r2,r3,r4,r5,r6,r7) do { \
    r0 = _mm_add_epi32(r0, rotr32_sse2<HM_R5>(r2)); \
    r1 = _mm_add_epi32(r1, rotr32_sse2<HM_R5>(r3)); \
    r6 = _mm_xor_si128(r6, r0); \
    r7 = _mm_xor_si128(r7, r1); \
    r6 = rotr32_sse2<HM_R1>(r6); \
    r7 = rotr32_sse2<HM_R1>(r7); \
    r4 = _mm_add_epi32(r4, rotr32_sse2<HM_R6>(r6)); \
    r5 = _mm_add_epi32(r5, rotr32_sse2<HM_R6>(r7)); \
    r2 = _mm_xor_si128(r2, r4); \
    r3 = _mm_xor_si128(r3, r5); \
    r2 = rotr32_sse2<HM_R2>(r2); \
    r3 = rotr32_sse2<HM_R2>(r3); \
    r0 = _mm_xor_si128(r0, rotr32_sse2<HM_R9>(r4)); \
    r1 = _mm_xor_si128(r1, rotr32_sse2<HM_R9>(r5)); \
    r0 = _mm_add_epi32(r0, rotr32_sse2<HM_R7>(r2)); \
    r1 = _mm_add_epi32(r1, rotr32_sse2<HM_R7>(r3)); \
    r6 = _mm_xor_si128(r6, r0); \
    r7 = _mm_xor_si128(r7, r1); \
    r6 = rotr32_sse2<HM_R3>(r6); \
    r7 = rotr32_sse2<HM_R3>(r7); \
    r4 = _mm_add_epi32(r4, rotr32_sse2<HM_R8>(r6)); \
    r5 = _mm_add_epi32(r5, rotr32_sse2<HM_R8>(r7)); \
    r2 = _mm_xor_si128(r2, r4); \
    r3 = _mm_xor_si128(r3, r5); \
    r2 = rotr32_sse2<HM_R4>(r2); \
    r3 = rotr32_sse2<HM_R4>(r3); \
} while(0)

// G_Mix/H_Mix 宏（AVX2），宽度 128→256
#define BASTION_GMIX_AVX2(r0,r1,r2,r3,r4,r5,r6,r7) do { \
    r0 = _mm256_add_epi32(r0, rotr32_avx2<GM_R5>(r2)); \
    r1 = _mm256_add_epi32(r1, rotr32_avx2<GM_R5>(r3)); \
    r6 = _mm256_xor_si256(r6, r0); \
    r7 = _mm256_xor_si256(r7, r1); \
    r6 = rotr32_avx2<GM_R1>(r6); \
    r7 = rotr32_avx2<GM_R1>(r7); \
    r4 = _mm256_add_epi32(r4, rotr32_avx2<GM_R6>(r6)); \
    r5 = _mm256_add_epi32(r5, rotr32_avx2<GM_R6>(r7)); \
    r2 = _mm256_xor_si256(r2, r4); \
    r3 = _mm256_xor_si256(r3, r5); \
    r2 = rotr32_avx2<GM_R2>(r2); \
    r3 = rotr32_avx2<GM_R2>(r3); \
    r0 = _mm256_xor_si256(r0, rotr32_avx2<GM_R9>(r4)); \
    r1 = _mm256_xor_si256(r1, rotr32_avx2<GM_R9>(r5)); \
    r0 = _mm256_add_epi32(r0, rotr32_avx2<GM_R7>(r2)); \
    r1 = _mm256_add_epi32(r1, rotr32_avx2<GM_R7>(r3)); \
    r6 = _mm256_xor_si256(r6, r0); \
    r7 = _mm256_xor_si256(r7, r1); \
    r6 = rotr32_avx2<GM_R3>(r6); \
    r7 = rotr32_avx2<GM_R3>(r7); \
    r4 = _mm256_add_epi32(r4, rotr32_avx2<GM_R8>(r6)); \
    r5 = _mm256_add_epi32(r5, rotr32_avx2<GM_R8>(r7)); \
    r2 = _mm256_xor_si256(r2, r4); \
    r3 = _mm256_xor_si256(r3, r5); \
    r2 = rotr32_avx2<GM_R4>(r2); \
    r3 = rotr32_avx2<GM_R4>(r3); \
} while(0)

#define BASTION_HMIX_AVX2(r0,r1,r2,r3,r4,r5,r6,r7) do { \
    r0 = _mm256_add_epi32(r0, rotr32_avx2<HM_R5>(r2)); \
    r1 = _mm256_add_epi32(r1, rotr32_avx2<HM_R5>(r3)); \
    r6 = _mm256_xor_si256(r6, r0); \
    r7 = _mm256_xor_si256(r7, r1); \
    r6 = rotr32_avx2<HM_R1>(r6); \
    r7 = rotr32_avx2<HM_R1>(r7); \
    r4 = _mm256_add_epi32(r4, rotr32_avx2<HM_R6>(r6)); \
    r5 = _mm256_add_epi32(r5, rotr32_avx2<HM_R6>(r7)); \
    r2 = _mm256_xor_si256(r2, r4); \
    r3 = _mm256_xor_si256(r3, r5); \
    r2 = rotr32_avx2<HM_R2>(r2); \
    r3 = rotr32_avx2<HM_R2>(r3); \
    r0 = _mm256_xor_si256(r0, rotr32_avx2<HM_R9>(r4)); \
    r1 = _mm256_xor_si256(r1, rotr32_avx2<HM_R9>(r5)); \
    r0 = _mm256_add_epi32(r0, rotr32_avx2<HM_R7>(r2)); \
    r1 = _mm256_add_epi32(r1, rotr32_avx2<HM_R7>(r3)); \
    r6 = _mm256_xor_si256(r6, r0); \
    r7 = _mm256_xor_si256(r7, r1); \
    r6 = rotr32_avx2<HM_R3>(r6); \
    r7 = rotr32_avx2<HM_R3>(r7); \
    r4 = _mm256_add_epi32(r4, rotr32_avx2<HM_R8>(r6)); \
    r5 = _mm256_add_epi32(r5, rotr32_avx2<HM_R8>(r7)); \
    r2 = _mm256_xor_si256(r2, r4); \
    r3 = _mm256_xor_si256(r3, r5); \
    r2 = rotr32_avx2<HM_R4>(r2); \
    r3 = rotr32_avx2<HM_R4>(r3); \
} while(0)

// 逆 G_Mix/H_Mix 宏：解密用左旋 + 减法，顺序严格按标量逆运算（每步用前一步新值）

// 逆 G_Mix 宏（SSE2）
#define BASTION_INV_GMIX_SSE2(r0,r1,r2,r3,r4,r5,r6,r7) do { \
    r2 = _mm_xor_si128(rotl32_sse2<GM_R4>(r2), r4); \
    r3 = _mm_xor_si128(rotl32_sse2<GM_R4>(r3), r5); \
    r4 = _mm_sub_epi32(r4, rotr32_sse2<GM_R8>(r6)); \
    r5 = _mm_sub_epi32(r5, rotr32_sse2<GM_R8>(r7)); \
    r6 = _mm_xor_si128(rotl32_sse2<GM_R3>(r6), r0); \
    r7 = _mm_xor_si128(rotl32_sse2<GM_R3>(r7), r1); \
    r0 = _mm_sub_epi32(r0, rotr32_sse2<GM_R7>(r2)); \
    r1 = _mm_sub_epi32(r1, rotr32_sse2<GM_R7>(r3)); \
    r0 = _mm_xor_si128(r0, rotr32_sse2<GM_R9>(r4)); \
    r1 = _mm_xor_si128(r1, rotr32_sse2<GM_R9>(r5)); \
    r2 = _mm_xor_si128(rotl32_sse2<GM_R2>(r2), r4); \
    r3 = _mm_xor_si128(rotl32_sse2<GM_R2>(r3), r5); \
    r4 = _mm_sub_epi32(r4, rotr32_sse2<GM_R6>(r6)); \
    r5 = _mm_sub_epi32(r5, rotr32_sse2<GM_R6>(r7)); \
    r6 = _mm_xor_si128(rotl32_sse2<GM_R1>(r6), r0); \
    r7 = _mm_xor_si128(rotl32_sse2<GM_R1>(r7), r1); \
    r0 = _mm_sub_epi32(r0, rotr32_sse2<GM_R5>(r2)); \
    r1 = _mm_sub_epi32(r1, rotr32_sse2<GM_R5>(r3)); \
} while(0)

// 逆 H_Mix 宏（SSE2），旋转常数换 HM_R1..R9
#define BASTION_INV_HMIX_SSE2(r0,r1,r2,r3,r4,r5,r6,r7) do { \
    r2 = _mm_xor_si128(rotl32_sse2<HM_R4>(r2), r4); \
    r3 = _mm_xor_si128(rotl32_sse2<HM_R4>(r3), r5); \
    r4 = _mm_sub_epi32(r4, rotr32_sse2<HM_R8>(r6)); \
    r5 = _mm_sub_epi32(r5, rotr32_sse2<HM_R8>(r7)); \
    r6 = _mm_xor_si128(rotl32_sse2<HM_R3>(r6), r0); \
    r7 = _mm_xor_si128(rotl32_sse2<HM_R3>(r7), r1); \
    r0 = _mm_sub_epi32(r0, rotr32_sse2<HM_R7>(r2)); \
    r1 = _mm_sub_epi32(r1, rotr32_sse2<HM_R7>(r3)); \
    r0 = _mm_xor_si128(r0, rotr32_sse2<HM_R9>(r4)); \
    r1 = _mm_xor_si128(r1, rotr32_sse2<HM_R9>(r5)); \
    r2 = _mm_xor_si128(rotl32_sse2<HM_R2>(r2), r4); \
    r3 = _mm_xor_si128(rotl32_sse2<HM_R2>(r3), r5); \
    r4 = _mm_sub_epi32(r4, rotr32_sse2<HM_R6>(r6)); \
    r5 = _mm_sub_epi32(r5, rotr32_sse2<HM_R6>(r7)); \
    r6 = _mm_xor_si128(rotl32_sse2<HM_R1>(r6), r0); \
    r7 = _mm_xor_si128(rotl32_sse2<HM_R1>(r7), r1); \
    r0 = _mm_sub_epi32(r0, rotr32_sse2<HM_R5>(r2)); \
    r1 = _mm_sub_epi32(r1, rotr32_sse2<HM_R5>(r3)); \
} while(0)

// 逆 G_Mix 宏（AVX2），宽度 128→256
#define BASTION_INV_GMIX_AVX2(r0,r1,r2,r3,r4,r5,r6,r7) do { \
    r2 = _mm256_xor_si256(rotl32_avx2<GM_R4>(r2), r4); \
    r3 = _mm256_xor_si256(rotl32_avx2<GM_R4>(r3), r5); \
    r4 = _mm256_sub_epi32(r4, rotr32_avx2<GM_R8>(r6)); \
    r5 = _mm256_sub_epi32(r5, rotr32_avx2<GM_R8>(r7)); \
    r6 = _mm256_xor_si256(rotl32_avx2<GM_R3>(r6), r0); \
    r7 = _mm256_xor_si256(rotl32_avx2<GM_R3>(r7), r1); \
    r0 = _mm256_sub_epi32(r0, rotr32_avx2<GM_R7>(r2)); \
    r1 = _mm256_sub_epi32(r1, rotr32_avx2<GM_R7>(r3)); \
    r0 = _mm256_xor_si256(r0, rotr32_avx2<GM_R9>(r4)); \
    r1 = _mm256_xor_si256(r1, rotr32_avx2<GM_R9>(r5)); \
    r2 = _mm256_xor_si256(rotl32_avx2<GM_R2>(r2), r4); \
    r3 = _mm256_xor_si256(rotl32_avx2<GM_R2>(r3), r5); \
    r4 = _mm256_sub_epi32(r4, rotr32_avx2<GM_R6>(r6)); \
    r5 = _mm256_sub_epi32(r5, rotr32_avx2<GM_R6>(r7)); \
    r6 = _mm256_xor_si256(rotl32_avx2<GM_R1>(r6), r0); \
    r7 = _mm256_xor_si256(rotl32_avx2<GM_R1>(r7), r1); \
    r0 = _mm256_sub_epi32(r0, rotr32_avx2<GM_R5>(r2)); \
    r1 = _mm256_sub_epi32(r1, rotr32_avx2<GM_R5>(r3)); \
} while(0)

// 逆 H_Mix 宏（AVX2），旋转常数换 HM_R1..R9
#define BASTION_INV_HMIX_AVX2(r0,r1,r2,r3,r4,r5,r6,r7) do { \
    r2 = _mm256_xor_si256(rotl32_avx2<HM_R4>(r2), r4); \
    r3 = _mm256_xor_si256(rotl32_avx2<HM_R4>(r3), r5); \
    r4 = _mm256_sub_epi32(r4, rotr32_avx2<HM_R8>(r6)); \
    r5 = _mm256_sub_epi32(r5, rotr32_avx2<HM_R8>(r7)); \
    r6 = _mm256_xor_si256(rotl32_avx2<HM_R3>(r6), r0); \
    r7 = _mm256_xor_si256(rotl32_avx2<HM_R3>(r7), r1); \
    r0 = _mm256_sub_epi32(r0, rotr32_avx2<HM_R7>(r2)); \
    r1 = _mm256_sub_epi32(r1, rotr32_avx2<HM_R7>(r3)); \
    r0 = _mm256_xor_si256(r0, rotr32_avx2<HM_R9>(r4)); \
    r1 = _mm256_xor_si256(r1, rotr32_avx2<HM_R9>(r5)); \
    r2 = _mm256_xor_si256(rotl32_avx2<HM_R2>(r2), r4); \
    r3 = _mm256_xor_si256(rotl32_avx2<HM_R2>(r3), r5); \
    r4 = _mm256_sub_epi32(r4, rotr32_avx2<HM_R6>(r6)); \
    r5 = _mm256_sub_epi32(r5, rotr32_avx2<HM_R6>(r7)); \
    r6 = _mm256_xor_si256(rotl32_avx2<HM_R1>(r6), r0); \
    r7 = _mm256_xor_si256(rotl32_avx2<HM_R1>(r7), r1); \
    r0 = _mm256_sub_epi32(r0, rotr32_avx2<HM_R5>(r2)); \
    r1 = _mm256_sub_epi32(r1, rotr32_avx2<HM_R5>(r3)); \
} while(0)

// CTR 批量加密，从 counter 加载初始状态

// SSE2 4 块批量加密，向量化版 encryptBlocksBatch4；in/out 长度必须是 32 的整数倍
inline void encryptBlocksBatch4_SSE2(const BlockCipher& c, const uint8_t* counter,
                                     const uint8_t* in, uint8_t* out, int len) {
    int n = len / 32;
    if (n < 4) {
        // 不足 4 块直接走标量
        encryptBlocksBatch4(c, counter, in, out, len);
        return;
    }

    uint32_t n0 = readU32LE(counter);
    uint32_t n1 = readU32LE(counter + 4);
    uint32_t n2 = readU32LE(counter + 8);
    uint32_t n3 = readU32LE(counter + 12);
    uint32_t n4 = readU32LE(counter + 16);
    uint32_t n5 = readU32LE(counter + 20);
    uint64_t baseCtr = readU64LE(counter + 24);
    const uint64_t* keys = c.roundKeys;
    int rounds = c.rounds;

    int i = 0;
    // 4 块批量主循环
    for (; i + 3 < n; i += 4) {
        uint64_t c0 = baseCtr + (uint64_t)i;
        uint64_t c1 = baseCtr + (uint64_t)(i + 1);
        uint64_t c2 = baseCtr + (uint64_t)(i + 2);
        uint64_t c3 = baseCtr + (uint64_t)(i + 3);

        // 8 个 __m128i 状态字：r_k = [块0.s_k, 块1.s_k, 块2.s_k, 块3.s_k]
        __m128i r0 = _mm_set1_epi32(n0);
        __m128i r1 = _mm_set1_epi32(n1);
        __m128i r2 = _mm_set1_epi32(n2);
        __m128i r3 = _mm_set1_epi32(n3);
        __m128i r4 = _mm_set1_epi32(n4);
        __m128i r5 = _mm_set1_epi32(n5);
        // s6 = 计数器低 32 位，s7 = 计数器高 32 位
        // _mm_set_epi32 参数顺序 lane3..lane0，lane0 对应块0
        __m128i r6 = _mm_set_epi32((uint32_t)c3, (uint32_t)c2, (uint32_t)c1, (uint32_t)c0);
        __m128i r7 = _mm_set_epi32((uint32_t)(c3 >> 32), (uint32_t)(c2 >> 32),
                                    (uint32_t)(c1 >> 32), (uint32_t)(c0 >> 32));

        // 输入白化（4 块共享轮密钥，广播到所有 lane）
        r0 = _mm_xor_si128(r0, _mm_set1_epi32((uint32_t)keys[0]));
        r1 = _mm_xor_si128(r1, _mm_set1_epi32((uint32_t)(keys[0] >> 32)));
        r2 = _mm_xor_si128(r2, _mm_set1_epi32((uint32_t)keys[1]));
        r3 = _mm_xor_si128(r3, _mm_set1_epi32((uint32_t)(keys[1] >> 32)));
        r4 = _mm_xor_si128(r4, _mm_set1_epi32((uint32_t)keys[2]));
        r5 = _mm_xor_si128(r5, _mm_set1_epi32((uint32_t)(keys[2] >> 32)));
        r6 = _mm_xor_si128(r6, _mm_set1_epi32((uint32_t)keys[3]));
        r7 = _mm_xor_si128(r7, _mm_set1_epi32((uint32_t)(keys[3] >> 32)));

        int ki = 4;
        for (int r = 0; r < rounds; r++) {
            if ((r & 1) == 0) {
                BASTION_GMIX_SSE2(r0, r1, r2, r3, r4, r5, r6, r7);
            } else {
                BASTION_HMIX_SSE2(r0, r1, r2, r3, r4, r5, r6, r7);
            }

            // ShiftRows：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)（__m128i 整体重排）
            { __m128i t = r0; r0 = r2; r2 = r7; r7 = r1; r1 = r5; r5 = r4; r4 = r3; r3 = r6; r6 = t; }

            // AddRoundKey
            r0 = _mm_xor_si128(r0, _mm_set1_epi32((uint32_t)keys[ki]));
            r1 = _mm_xor_si128(r1, _mm_set1_epi32((uint32_t)(keys[ki] >> 32)));
            r2 = _mm_xor_si128(r2, _mm_set1_epi32((uint32_t)keys[ki + 1]));
            r3 = _mm_xor_si128(r3, _mm_set1_epi32((uint32_t)(keys[ki + 1] >> 32)));
            r4 = _mm_xor_si128(r4, _mm_set1_epi32((uint32_t)keys[ki + 2]));
            r5 = _mm_xor_si128(r5, _mm_set1_epi32((uint32_t)(keys[ki + 2] >> 32)));
            r6 = _mm_xor_si128(r6, _mm_set1_epi32((uint32_t)keys[ki + 3]));
            r7 = _mm_xor_si128(r7, _mm_set1_epi32((uint32_t)(keys[ki + 3] >> 32)));
            ki += 4;
        }

        // 输出白化
        r0 = _mm_xor_si128(r0, _mm_set1_epi32((uint32_t)keys[ki]));
        r1 = _mm_xor_si128(r1, _mm_set1_epi32((uint32_t)(keys[ki] >> 32)));
        r2 = _mm_xor_si128(r2, _mm_set1_epi32((uint32_t)keys[ki + 1]));
        r3 = _mm_xor_si128(r3, _mm_set1_epi32((uint32_t)(keys[ki + 1] >> 32)));
        r4 = _mm_xor_si128(r4, _mm_set1_epi32((uint32_t)keys[ki + 2]));
        r5 = _mm_xor_si128(r5, _mm_set1_epi32((uint32_t)(keys[ki + 2] >> 32)));
        r6 = _mm_xor_si128(r6, _mm_set1_epi32((uint32_t)keys[ki + 3]));
        r7 = _mm_xor_si128(r7, _mm_set1_epi32((uint32_t)(keys[ki + 3] >> 32)));

        // 4x4 转置：r0..r3 → 块0..3 的 s0..s3（前 16 字节）
        __m128i out0_lo, out1_lo, out2_lo, out3_lo;
        transpose4x4_sse2(r0, r1, r2, r3, out0_lo, out1_lo, out2_lo, out3_lo);
        // r4..r7 → 块0..3 的 s4..s7（后 16 字节）
        __m128i out0_hi, out1_hi, out2_hi, out3_hi;
        transpose4x4_sse2(r4, r5, r6, r7, out0_hi, out1_hi, out2_hi, out3_hi);

        // 写出 4 块，每块 2 个 __m128i（前 16B + 后 16B），与 src 异或后写入 dst
        const __m128i* s = reinterpret_cast<const __m128i*>(in + i * 32);
        __m128i* o = reinterpret_cast<__m128i*>(out + i * 32);
        __m128i d;

        d = _mm_loadu_si128(s + 0);
        _mm_storeu_si128(o + 0, _mm_xor_si128(d, out0_lo));
        d = _mm_loadu_si128(s + 1);
        _mm_storeu_si128(o + 1, _mm_xor_si128(d, out0_hi));

        d = _mm_loadu_si128(s + 2);
        _mm_storeu_si128(o + 2, _mm_xor_si128(d, out1_lo));
        d = _mm_loadu_si128(s + 3);
        _mm_storeu_si128(o + 3, _mm_xor_si128(d, out1_hi));

        d = _mm_loadu_si128(s + 4);
        _mm_storeu_si128(o + 4, _mm_xor_si128(d, out2_lo));
        d = _mm_loadu_si128(s + 5);
        _mm_storeu_si128(o + 5, _mm_xor_si128(d, out2_hi));

        d = _mm_loadu_si128(s + 6);
        _mm_storeu_si128(o + 6, _mm_xor_si128(d, out3_lo));
        d = _mm_loadu_si128(s + 7);
        _mm_storeu_si128(o + 7, _mm_xor_si128(d, out3_hi));
    }

    // 尾部不足 4 块：逐块标量加密 + xorWords
    if (i < n) {
        uint8_t ctrBuf[32];
        std::memcpy(ctrBuf, counter, 24);
        for (; i < n; i++) {
            writeU64LE(ctrBuf + 24, baseCtr + (uint64_t)i);
            uint8_t ks[32];
            blockEncrypt(ctrBuf, ks, c);
            int off = i * 32;
            xorWords(in + off, ks, out + off, 32);
        }
    }
}

// AVX2 8 块批量加密，8 个 __m256i 状态字，每个含 8 块的同一个 s_k；in/out 长度必须是 32 的整数倍
inline void encryptBlocksBatch8_AVX2(const BlockCipher& c, const uint8_t* counter,
                                     const uint8_t* in, uint8_t* out, int len) {
    int n = len / 32;
    if (n < 8) {
        // 不足 8 块，回退 SSE2（内部会判断 <4 块再回退标量）
        encryptBlocksBatch4_SSE2(c, counter, in, out, len);
        return;
    }

    uint32_t n0 = readU32LE(counter);
    uint32_t n1 = readU32LE(counter + 4);
    uint32_t n2 = readU32LE(counter + 8);
    uint32_t n3 = readU32LE(counter + 12);
    uint32_t n4 = readU32LE(counter + 16);
    uint32_t n5 = readU32LE(counter + 20);
    uint64_t baseCtr = readU64LE(counter + 24);
    const uint64_t* keys = c.roundKeys;
    int rounds = c.rounds;

    int i = 0;
    // 8 块批量主循环
    for (; i + 7 < n; i += 8) {
        uint64_t c0 = baseCtr + (uint64_t)i;
        uint64_t c1 = baseCtr + (uint64_t)(i + 1);
        uint64_t c2 = baseCtr + (uint64_t)(i + 2);
        uint64_t c3 = baseCtr + (uint64_t)(i + 3);
        uint64_t c4 = baseCtr + (uint64_t)(i + 4);
        uint64_t c5 = baseCtr + (uint64_t)(i + 5);
        uint64_t c6 = baseCtr + (uint64_t)(i + 6);
        uint64_t c7 = baseCtr + (uint64_t)(i + 7);

        // 8 个 __m256i 状态字：r_k = [块0..7 的 s_k]
        __m256i r0 = _mm256_set1_epi32(n0);
        __m256i r1 = _mm256_set1_epi32(n1);
        __m256i r2 = _mm256_set1_epi32(n2);
        __m256i r3 = _mm256_set1_epi32(n3);
        __m256i r4 = _mm256_set1_epi32(n4);
        __m256i r5 = _mm256_set1_epi32(n5);
        // s6 = 计数器低 32 位，s7 = 计数器高 32 位
        // _mm256_set_epi32 参数顺序 lane7..lane0，lane0 对应块0
        __m256i r6 = _mm256_set_epi32(
            (uint32_t)c7, (uint32_t)c6, (uint32_t)c5, (uint32_t)c4,
            (uint32_t)c3, (uint32_t)c2, (uint32_t)c1, (uint32_t)c0);
        __m256i r7 = _mm256_set_epi32(
            (uint32_t)(c7 >> 32), (uint32_t)(c6 >> 32),
            (uint32_t)(c5 >> 32), (uint32_t)(c4 >> 32),
            (uint32_t)(c3 >> 32), (uint32_t)(c2 >> 32),
            (uint32_t)(c1 >> 32), (uint32_t)(c0 >> 32));

        // 输入白化（8 块共享轮密钥，广播到所有 lane）
        r0 = _mm256_xor_si256(r0, _mm256_set1_epi32((uint32_t)keys[0]));
        r1 = _mm256_xor_si256(r1, _mm256_set1_epi32((uint32_t)(keys[0] >> 32)));
        r2 = _mm256_xor_si256(r2, _mm256_set1_epi32((uint32_t)keys[1]));
        r3 = _mm256_xor_si256(r3, _mm256_set1_epi32((uint32_t)(keys[1] >> 32)));
        r4 = _mm256_xor_si256(r4, _mm256_set1_epi32((uint32_t)keys[2]));
        r5 = _mm256_xor_si256(r5, _mm256_set1_epi32((uint32_t)(keys[2] >> 32)));
        r6 = _mm256_xor_si256(r6, _mm256_set1_epi32((uint32_t)keys[3]));
        r7 = _mm256_xor_si256(r7, _mm256_set1_epi32((uint32_t)(keys[3] >> 32)));

        int ki = 4;
        for (int r = 0; r < rounds; r++) {
            if ((r & 1) == 0) {
                BASTION_GMIX_AVX2(r0, r1, r2, r3, r4, r5, r6, r7);
            } else {
                BASTION_HMIX_AVX2(r0, r1, r2, r3, r4, r5, r6, r7);
            }

            // ShiftRows：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
            { __m256i t = r0; r0 = r2; r2 = r7; r7 = r1; r1 = r5; r5 = r4; r4 = r3; r3 = r6; r6 = t; }

            // AddRoundKey
            r0 = _mm256_xor_si256(r0, _mm256_set1_epi32((uint32_t)keys[ki]));
            r1 = _mm256_xor_si256(r1, _mm256_set1_epi32((uint32_t)(keys[ki] >> 32)));
            r2 = _mm256_xor_si256(r2, _mm256_set1_epi32((uint32_t)keys[ki + 1]));
            r3 = _mm256_xor_si256(r3, _mm256_set1_epi32((uint32_t)(keys[ki + 1] >> 32)));
            r4 = _mm256_xor_si256(r4, _mm256_set1_epi32((uint32_t)keys[ki + 2]));
            r5 = _mm256_xor_si256(r5, _mm256_set1_epi32((uint32_t)(keys[ki + 2] >> 32)));
            r6 = _mm256_xor_si256(r6, _mm256_set1_epi32((uint32_t)keys[ki + 3]));
            r7 = _mm256_xor_si256(r7, _mm256_set1_epi32((uint32_t)(keys[ki + 3] >> 32)));
            ki += 4;
        }

        // 输出白化
        r0 = _mm256_xor_si256(r0, _mm256_set1_epi32((uint32_t)keys[ki]));
        r1 = _mm256_xor_si256(r1, _mm256_set1_epi32((uint32_t)(keys[ki] >> 32)));
        r2 = _mm256_xor_si256(r2, _mm256_set1_epi32((uint32_t)keys[ki + 1]));
        r3 = _mm256_xor_si256(r3, _mm256_set1_epi32((uint32_t)(keys[ki + 1] >> 32)));
        r4 = _mm256_xor_si256(r4, _mm256_set1_epi32((uint32_t)keys[ki + 2]));
        r5 = _mm256_xor_si256(r5, _mm256_set1_epi32((uint32_t)(keys[ki + 2] >> 32)));
        r6 = _mm256_xor_si256(r6, _mm256_set1_epi32((uint32_t)keys[ki + 3]));
        r7 = _mm256_xor_si256(r7, _mm256_set1_epi32((uint32_t)(keys[ki + 3] >> 32)));

        // 8x8 转置：AVX2 的 unpack 在每个 128 位通道内独立执行，需 permute2x128 跨通道重组
        __m256i t0 = _mm256_unpacklo_epi32(r0, r1);
        __m256i t1 = _mm256_unpackhi_epi32(r0, r1);
        __m256i t2 = _mm256_unpacklo_epi32(r2, r3);
        __m256i t3 = _mm256_unpackhi_epi32(r2, r3);

        __m256i u0 = _mm256_unpacklo_epi64(t0, t2);
        __m256i u1 = _mm256_unpackhi_epi64(t0, t2);
        __m256i u2 = _mm256_unpacklo_epi64(t1, t3);
        __m256i u3 = _mm256_unpackhi_epi64(t1, t3);

        t0 = _mm256_unpacklo_epi32(r4, r5);
        t1 = _mm256_unpackhi_epi32(r4, r5);
        t2 = _mm256_unpacklo_epi32(r6, r7);
        t3 = _mm256_unpackhi_epi32(r6, r7);

        __m256i v0 = _mm256_unpacklo_epi64(t0, t2);
        __m256i v1 = _mm256_unpackhi_epi64(t0, t2);
        __m256i v2 = _mm256_unpacklo_epi64(t1, t3);
        __m256i v3 = _mm256_unpackhi_epi64(t1, t3);

        // permute2x128 跨 128 位通道：u.low+v.low → 块i 的前 16B，u.high+v.high → 块i+4 的前 16B
        // ks_k 是块 k 的完整 32 字节密钥流（前 16B = s0..s3，后 16B = s4..s7）
        __m256i ks0 = _mm256_permute2x128_si256(u0, v0, 0x20);
        __m256i ks4 = _mm256_permute2x128_si256(u0, v0, 0x31);
        __m256i ks1 = _mm256_permute2x128_si256(u1, v1, 0x20);
        __m256i ks5 = _mm256_permute2x128_si256(u1, v1, 0x31);
        __m256i ks2 = _mm256_permute2x128_si256(u2, v2, 0x20);
        __m256i ks6 = _mm256_permute2x128_si256(u2, v2, 0x31);
        __m256i ks3 = _mm256_permute2x128_si256(u3, v3, 0x20);
        __m256i ks7 = _mm256_permute2x128_si256(u3, v3, 0x31);

        // 写出 8 块，每块 1 个 __m256i（32 字节），与 src 异或后写入 dst
        const __m256i* s = reinterpret_cast<const __m256i*>(in + i * 32);
        __m256i* o = reinterpret_cast<__m256i*>(out + i * 32);
        __m256i d;

        d = _mm256_loadu_si256(s + 0);
        _mm256_storeu_si256(o + 0, _mm256_xor_si256(d, ks0));
        d = _mm256_loadu_si256(s + 1);
        _mm256_storeu_si256(o + 1, _mm256_xor_si256(d, ks1));
        d = _mm256_loadu_si256(s + 2);
        _mm256_storeu_si256(o + 2, _mm256_xor_si256(d, ks2));
        d = _mm256_loadu_si256(s + 3);
        _mm256_storeu_si256(o + 3, _mm256_xor_si256(d, ks3));
        d = _mm256_loadu_si256(s + 4);
        _mm256_storeu_si256(o + 4, _mm256_xor_si256(d, ks4));
        d = _mm256_loadu_si256(s + 5);
        _mm256_storeu_si256(o + 5, _mm256_xor_si256(d, ks5));
        d = _mm256_loadu_si256(s + 6);
        _mm256_storeu_si256(o + 6, _mm256_xor_si256(d, ks6));
        d = _mm256_loadu_si256(s + 7);
        _mm256_storeu_si256(o + 7, _mm256_xor_si256(d, ks7));
    }

    // 清理 AVX 状态，避免 AVX→SSE 转换惩罚
    _mm256_zeroupper();

    // 尾部不足 8 块：构造推进后的 counter，回退 SSE2（内部继续判断 <4 块再回退标量）
    if (i < n) {
        uint8_t ctrBuf[32];
        std::memcpy(ctrBuf, counter, 24);
        writeU64LE(ctrBuf + 24, baseCtr + (uint64_t)i);
        int remain = (n - i) * 32;
        encryptBlocksBatch4_SSE2(c, ctrBuf, in + i * 32, out + i * 32, remain);
    }
}

// 从输入加载初始状态的批量加解密
// CTR 的初始状态是 nonce+counter（只需 set1 广播 nonce + set_epi32 打包 counter）
// CBC 解密 / XTS 加解密的初始状态是 4/8 块各不相同的输入数据，需要加载 + 转置
// 加解密核心（16 轮 Mix + ShiftRows + AddRoundKey + 白化）与 CTR 版完全相同

// 带 tweak 的单块加解密：加密 C = E(P^T)^T，解密 P = D(C^T)^T
// XTS 的尾巴块和不足一批的零头用它
inline void xtsBlockCrypt(const BlockCipher& c, const uint8_t* in, uint8_t* out,
                          const uint8_t* tweak, bool encrypt) {
    uint8_t blk[32];
    xorWords(in, tweak, blk, 32);
    if (encrypt) blockEncrypt(blk, blk, c);
    else blockDecrypt(blk, blk, c);
    xorWords(blk, tweak, out, 32);
}

// SSE2 4块批量加密，从输入数据加载初始状态（XTS 加密用）
// in/out 长度 len 必须是 32 的整数倍，in 可等于 out（原地）
// tweaks 非空时按块做 XTS 的两次 tweak 异或（C = E(P^T)^T），省掉外面两遍全缓冲读写
inline void encryptBlocksFromInputBatch4_SSE2(const BlockCipher& c,
                                              const uint8_t* in, uint8_t* out, int len,
                                              const uint8_t* tweaks = nullptr) {
    int n = len / 32;
    if (n < 4) {
        // 不足 4 块走标量，带 tweak 时把 tweak 揉进加解密
        for (int i = 0; i < n; i++) {
            if (tweaks == nullptr) blockEncrypt(in + i * 32, out + i * 32, c);
            else xtsBlockCrypt(c, in + i * 32, out + i * 32, tweaks + i * 32, true);
        }
        return;
    }

    const uint64_t* keys = c.roundKeys;
    int rounds = c.rounds;
    const __m128i* in128 = reinterpret_cast<const __m128i*>(in);
    __m128i* out128 = reinterpret_cast<__m128i*>(out);

    int i = 0;
    for (; i + 3 < n; i += 4) {
        // 加载 4 块的前 16 字节（s0..s3），转置成 r0..r3
        __m128i a0 = _mm_loadu_si128(in128 + (i + 0) * 2);
        __m128i a1 = _mm_loadu_si128(in128 + (i + 1) * 2);
        __m128i a2 = _mm_loadu_si128(in128 + (i + 2) * 2);
        __m128i a3 = _mm_loadu_si128(in128 + (i + 3) * 2);
        // 带 tweak 时先把 tweak 揉进明文，这是 C = E(P^T)^T 的前半步
        if (tweaks != nullptr) {
            const __m128i* tw = reinterpret_cast<const __m128i*>(tweaks + i * 32);
            a0 = _mm_xor_si128(a0, _mm_loadu_si128(tw + 0));
            a1 = _mm_xor_si128(a1, _mm_loadu_si128(tw + 2));
            a2 = _mm_xor_si128(a2, _mm_loadu_si128(tw + 4));
            a3 = _mm_xor_si128(a3, _mm_loadu_si128(tw + 6));
        }
        __m128i r0, r1, r2, r3;
        transpose4x4_sse2(a0, a1, a2, a3, r0, r1, r2, r3);

        // 加载 4 块的后 16 字节（s4..s7），转置成 r4..r7
        __m128i b0 = _mm_loadu_si128(in128 + (i + 0) * 2 + 1);
        __m128i b1 = _mm_loadu_si128(in128 + (i + 1) * 2 + 1);
        __m128i b2 = _mm_loadu_si128(in128 + (i + 2) * 2 + 1);
        __m128i b3 = _mm_loadu_si128(in128 + (i + 3) * 2 + 1);
        if (tweaks != nullptr) {
            const __m128i* tw = reinterpret_cast<const __m128i*>(tweaks + i * 32);
            b0 = _mm_xor_si128(b0, _mm_loadu_si128(tw + 1));
            b1 = _mm_xor_si128(b1, _mm_loadu_si128(tw + 3));
            b2 = _mm_xor_si128(b2, _mm_loadu_si128(tw + 5));
            b3 = _mm_xor_si128(b3, _mm_loadu_si128(tw + 7));
        }
        __m128i r4, r5, r6, r7;
        transpose4x4_sse2(b0, b1, b2, b3, r4, r5, r6, r7);

        // 输入白化
        r0 = _mm_xor_si128(r0, _mm_set1_epi32((uint32_t)keys[0]));
        r1 = _mm_xor_si128(r1, _mm_set1_epi32((uint32_t)(keys[0] >> 32)));
        r2 = _mm_xor_si128(r2, _mm_set1_epi32((uint32_t)keys[1]));
        r3 = _mm_xor_si128(r3, _mm_set1_epi32((uint32_t)(keys[1] >> 32)));
        r4 = _mm_xor_si128(r4, _mm_set1_epi32((uint32_t)keys[2]));
        r5 = _mm_xor_si128(r5, _mm_set1_epi32((uint32_t)(keys[2] >> 32)));
        r6 = _mm_xor_si128(r6, _mm_set1_epi32((uint32_t)keys[3]));
        r7 = _mm_xor_si128(r7, _mm_set1_epi32((uint32_t)(keys[3] >> 32)));

        int ki = 4;
        for (int r = 0; r < rounds; r++) {
            if ((r & 1) == 0) BASTION_GMIX_SSE2(r0, r1, r2, r3, r4, r5, r6, r7);
            else BASTION_HMIX_SSE2(r0, r1, r2, r3, r4, r5, r6, r7);
            // ShiftRows：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
            { __m128i t = r0; r0 = r2; r2 = r7; r7 = r1; r1 = r5; r5 = r4; r4 = r3; r3 = r6; r6 = t; }
            r0 = _mm_xor_si128(r0, _mm_set1_epi32((uint32_t)keys[ki]));
            r1 = _mm_xor_si128(r1, _mm_set1_epi32((uint32_t)(keys[ki] >> 32)));
            r2 = _mm_xor_si128(r2, _mm_set1_epi32((uint32_t)keys[ki + 1]));
            r3 = _mm_xor_si128(r3, _mm_set1_epi32((uint32_t)(keys[ki + 1] >> 32)));
            r4 = _mm_xor_si128(r4, _mm_set1_epi32((uint32_t)keys[ki + 2]));
            r5 = _mm_xor_si128(r5, _mm_set1_epi32((uint32_t)(keys[ki + 2] >> 32)));
            r6 = _mm_xor_si128(r6, _mm_set1_epi32((uint32_t)keys[ki + 3]));
            r7 = _mm_xor_si128(r7, _mm_set1_epi32((uint32_t)(keys[ki + 3] >> 32)));
            ki += 4;
        }

        // 输出白化
        r0 = _mm_xor_si128(r0, _mm_set1_epi32((uint32_t)keys[ki]));
        r1 = _mm_xor_si128(r1, _mm_set1_epi32((uint32_t)(keys[ki] >> 32)));
        r2 = _mm_xor_si128(r2, _mm_set1_epi32((uint32_t)keys[ki + 1]));
        r3 = _mm_xor_si128(r3, _mm_set1_epi32((uint32_t)(keys[ki + 1] >> 32)));
        r4 = _mm_xor_si128(r4, _mm_set1_epi32((uint32_t)keys[ki + 2]));
        r5 = _mm_xor_si128(r5, _mm_set1_epi32((uint32_t)(keys[ki + 2] >> 32)));
        r6 = _mm_xor_si128(r6, _mm_set1_epi32((uint32_t)keys[ki + 3]));
        r7 = _mm_xor_si128(r7, _mm_set1_epi32((uint32_t)(keys[ki + 3] >> 32)));

        // 转置写出
        __m128i o0_lo, o1_lo, o2_lo, o3_lo;
        transpose4x4_sse2(r0, r1, r2, r3, o0_lo, o1_lo, o2_lo, o3_lo);
        __m128i o0_hi, o1_hi, o2_hi, o3_hi;
        transpose4x4_sse2(r4, r5, r6, r7, o0_hi, o1_hi, o2_hi, o3_hi);

        // 带 tweak 时输出前再异或一次，凑齐 C = E(P^T)^T
        if (tweaks != nullptr) {
            const __m128i* tw = reinterpret_cast<const __m128i*>(tweaks + i * 32);
            o0_lo = _mm_xor_si128(o0_lo, _mm_loadu_si128(tw + 0));
            o0_hi = _mm_xor_si128(o0_hi, _mm_loadu_si128(tw + 1));
            o1_lo = _mm_xor_si128(o1_lo, _mm_loadu_si128(tw + 2));
            o1_hi = _mm_xor_si128(o1_hi, _mm_loadu_si128(tw + 3));
            o2_lo = _mm_xor_si128(o2_lo, _mm_loadu_si128(tw + 4));
            o2_hi = _mm_xor_si128(o2_hi, _mm_loadu_si128(tw + 5));
            o3_lo = _mm_xor_si128(o3_lo, _mm_loadu_si128(tw + 6));
            o3_hi = _mm_xor_si128(o3_hi, _mm_loadu_si128(tw + 7));
        }

        __m128i* o = out128 + i * 2;
        _mm_storeu_si128(o + 0, o0_lo);
        _mm_storeu_si128(o + 1, o0_hi);
        _mm_storeu_si128(o + 2, o1_lo);
        _mm_storeu_si128(o + 3, o1_hi);
        _mm_storeu_si128(o + 4, o2_lo);
        _mm_storeu_si128(o + 5, o2_hi);
        _mm_storeu_si128(o + 6, o3_lo);
        _mm_storeu_si128(o + 7, o3_hi);
    }

    // 尾部不足 4 块，带 tweak 时把 tweak 揉进加解密
    for (; i < n; i++) {
        if (tweaks == nullptr) blockEncrypt(in + i * 32, out + i * 32, c);
        else xtsBlockCrypt(c, in + i * 32, out + i * 32, tweaks + i * 32, true);
    }
}

// AVX2 8块批量加密，从输入数据加载初始状态（XTS 加密用）
// tweaks 非空时按块做 XTS 的两次 tweak 异或（C = E(P^T)^T），省掉外面两遍全缓冲读写
inline void encryptBlocksFromInputBatch8_AVX2(const BlockCipher& c,
                                              const uint8_t* in, uint8_t* out, int len,
                                              const uint8_t* tweaks = nullptr) {
    int n = len / 32;
    if (n < 8) {
        encryptBlocksFromInputBatch4_SSE2(c, in, out, len, tweaks);
        return;
    }

    const uint64_t* keys = c.roundKeys;
    int rounds = c.rounds;
    const __m256i* in256 = reinterpret_cast<const __m256i*>(in);
    __m256i* out256 = reinterpret_cast<__m256i*>(out);

    int i = 0;
    for (; i + 7 < n; i += 8) {
        // 加载 8 块（每块 32 字节 = 1 个 __m256i），转置成 8 个 __m256i 状态字
        // 8x8 转置：先按 4+4 分两组加载，unpack + permute
        __m256i v0 = _mm256_loadu_si256(in256 + i + 0);  // 块0: s0..s7
        __m256i v1 = _mm256_loadu_si256(in256 + i + 1);  // 块1
        __m256i v2 = _mm256_loadu_si256(in256 + i + 2);  // 块2
        __m256i v3 = _mm256_loadu_si256(in256 + i + 3);  // 块3
        __m256i v4 = _mm256_loadu_si256(in256 + i + 4);  // 块4
        __m256i v5 = _mm256_loadu_si256(in256 + i + 5);  // 块5
        __m256i v6 = _mm256_loadu_si256(in256 + i + 6);  // 块6
        __m256i v7 = _mm256_loadu_si256(in256 + i + 7);  // 块7

        // 带 tweak 时先把 tweak 揉进明文，这是 C = E(P^T)^T 的前半步
        if (tweaks != nullptr) {
            const __m256i* tw = reinterpret_cast<const __m256i*>(tweaks + i * 32);
            v0 = _mm256_xor_si256(v0, _mm256_loadu_si256(tw + 0));
            v1 = _mm256_xor_si256(v1, _mm256_loadu_si256(tw + 1));
            v2 = _mm256_xor_si256(v2, _mm256_loadu_si256(tw + 2));
            v3 = _mm256_xor_si256(v3, _mm256_loadu_si256(tw + 3));
            v4 = _mm256_xor_si256(v4, _mm256_loadu_si256(tw + 4));
            v5 = _mm256_xor_si256(v5, _mm256_loadu_si256(tw + 5));
            v6 = _mm256_xor_si256(v6, _mm256_loadu_si256(tw + 6));
            v7 = _mm256_xor_si256(v7, _mm256_loadu_si256(tw + 7));
        }

        // 8x8 转置：8 块 → 8 个 __m256i 各含 8 块的同一状态字
        __m256i t0 = _mm256_unpacklo_epi32(v0, v1);
        __m256i t1 = _mm256_unpackhi_epi32(v0, v1);
        __m256i t2 = _mm256_unpacklo_epi32(v2, v3);
        __m256i t3 = _mm256_unpackhi_epi32(v2, v3);
        __m256i u0 = _mm256_unpacklo_epi64(t0, t2);
        __m256i u1 = _mm256_unpackhi_epi64(t0, t2);
        __m256i u2 = _mm256_unpacklo_epi64(t1, t3);
        __m256i u3 = _mm256_unpackhi_epi64(t1, t3);

        t0 = _mm256_unpacklo_epi32(v4, v5);
        t1 = _mm256_unpackhi_epi32(v4, v5);
        t2 = _mm256_unpacklo_epi32(v6, v7);
        t3 = _mm256_unpackhi_epi32(v6, v7);
        __m256i v0_ = _mm256_unpacklo_epi64(t0, t2);
        __m256i v1_ = _mm256_unpackhi_epi64(t0, t2);
        __m256i v2_ = _mm256_unpacklo_epi64(t1, t3);
        __m256i v3_ = _mm256_unpackhi_epi64(t1, t3);

        // 跨 128 位通道重组：permute2x128 把 u/v 的低/高 128 组合成 8 块的状态字
        // r_k 的 low 128 = 块0..3 的 s_k，high 128 = 块4..7 的 s_k
        __m256i r0 = _mm256_permute2x128_si256(u0, v0_, 0x20);
        __m256i r1 = _mm256_permute2x128_si256(u1, v1_, 0x20);
        __m256i r2 = _mm256_permute2x128_si256(u2, v2_, 0x20);
        __m256i r3 = _mm256_permute2x128_si256(u3, v3_, 0x20);
        __m256i r4 = _mm256_permute2x128_si256(u0, v0_, 0x31);
        __m256i r5 = _mm256_permute2x128_si256(u1, v1_, 0x31);
        __m256i r6 = _mm256_permute2x128_si256(u2, v2_, 0x31);
        __m256i r7 = _mm256_permute2x128_si256(u3, v3_, 0x31);

        // 输入白化
        r0 = _mm256_xor_si256(r0, _mm256_set1_epi32((uint32_t)keys[0]));
        r1 = _mm256_xor_si256(r1, _mm256_set1_epi32((uint32_t)(keys[0] >> 32)));
        r2 = _mm256_xor_si256(r2, _mm256_set1_epi32((uint32_t)keys[1]));
        r3 = _mm256_xor_si256(r3, _mm256_set1_epi32((uint32_t)(keys[1] >> 32)));
        r4 = _mm256_xor_si256(r4, _mm256_set1_epi32((uint32_t)keys[2]));
        r5 = _mm256_xor_si256(r5, _mm256_set1_epi32((uint32_t)(keys[2] >> 32)));
        r6 = _mm256_xor_si256(r6, _mm256_set1_epi32((uint32_t)keys[3]));
        r7 = _mm256_xor_si256(r7, _mm256_set1_epi32((uint32_t)(keys[3] >> 32)));

        int ki = 4;
        for (int r = 0; r < rounds; r++) {
            if ((r & 1) == 0) BASTION_GMIX_AVX2(r0, r1, r2, r3, r4, r5, r6, r7);
            else BASTION_HMIX_AVX2(r0, r1, r2, r3, r4, r5, r6, r7);
            // ShiftRows：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
            { __m256i t = r0; r0 = r2; r2 = r7; r7 = r1; r1 = r5; r5 = r4; r4 = r3; r3 = r6; r6 = t; }
            r0 = _mm256_xor_si256(r0, _mm256_set1_epi32((uint32_t)keys[ki]));
            r1 = _mm256_xor_si256(r1, _mm256_set1_epi32((uint32_t)(keys[ki] >> 32)));
            r2 = _mm256_xor_si256(r2, _mm256_set1_epi32((uint32_t)keys[ki + 1]));
            r3 = _mm256_xor_si256(r3, _mm256_set1_epi32((uint32_t)(keys[ki + 1] >> 32)));
            r4 = _mm256_xor_si256(r4, _mm256_set1_epi32((uint32_t)keys[ki + 2]));
            r5 = _mm256_xor_si256(r5, _mm256_set1_epi32((uint32_t)(keys[ki + 2] >> 32)));
            r6 = _mm256_xor_si256(r6, _mm256_set1_epi32((uint32_t)keys[ki + 3]));
            r7 = _mm256_xor_si256(r7, _mm256_set1_epi32((uint32_t)(keys[ki + 3] >> 32)));
            ki += 4;
        }

        // 输出白化
        r0 = _mm256_xor_si256(r0, _mm256_set1_epi32((uint32_t)keys[ki]));
        r1 = _mm256_xor_si256(r1, _mm256_set1_epi32((uint32_t)(keys[ki] >> 32)));
        r2 = _mm256_xor_si256(r2, _mm256_set1_epi32((uint32_t)keys[ki + 1]));
        r3 = _mm256_xor_si256(r3, _mm256_set1_epi32((uint32_t)(keys[ki + 1] >> 32)));
        r4 = _mm256_xor_si256(r4, _mm256_set1_epi32((uint32_t)keys[ki + 2]));
        r5 = _mm256_xor_si256(r5, _mm256_set1_epi32((uint32_t)(keys[ki + 2] >> 32)));
        r6 = _mm256_xor_si256(r6, _mm256_set1_epi32((uint32_t)keys[ki + 3]));
        r7 = _mm256_xor_si256(r7, _mm256_set1_epi32((uint32_t)(keys[ki + 3] >> 32)));

        // 8x8 逆转置：8 个 __m256i 状态字 → 8 块的 32 字节输出
        // 与加密批量函数的写出转置逻辑相同
        t0 = _mm256_unpacklo_epi32(r0, r1);
        t1 = _mm256_unpackhi_epi32(r0, r1);
        t2 = _mm256_unpacklo_epi32(r2, r3);
        t3 = _mm256_unpackhi_epi32(r2, r3);
        u0 = _mm256_unpacklo_epi64(t0, t2);
        u1 = _mm256_unpackhi_epi64(t0, t2);
        u2 = _mm256_unpacklo_epi64(t1, t3);
        u3 = _mm256_unpackhi_epi64(t1, t3);

        t0 = _mm256_unpacklo_epi32(r4, r5);
        t1 = _mm256_unpackhi_epi32(r4, r5);
        t2 = _mm256_unpacklo_epi32(r6, r7);
        t3 = _mm256_unpackhi_epi32(r6, r7);
        v0_ = _mm256_unpacklo_epi64(t0, t2);
        v1_ = _mm256_unpackhi_epi64(t0, t2);
        v2_ = _mm256_unpacklo_epi64(t1, t3);
        v3_ = _mm256_unpackhi_epi64(t1, t3);

        __m256i ks0 = _mm256_permute2x128_si256(u0, v0_, 0x20);
        __m256i ks1 = _mm256_permute2x128_si256(u1, v1_, 0x20);
        __m256i ks2 = _mm256_permute2x128_si256(u2, v2_, 0x20);
        __m256i ks3 = _mm256_permute2x128_si256(u3, v3_, 0x20);
        __m256i ks4 = _mm256_permute2x128_si256(u0, v0_, 0x31);
        __m256i ks5 = _mm256_permute2x128_si256(u1, v1_, 0x31);
        __m256i ks6 = _mm256_permute2x128_si256(u2, v2_, 0x31);
        __m256i ks7 = _mm256_permute2x128_si256(u3, v3_, 0x31);

        // 带 tweak 时输出前再异或一次，凑齐 C = E(P^T)^T
        if (tweaks != nullptr) {
            const __m256i* tw = reinterpret_cast<const __m256i*>(tweaks + i * 32);
            ks0 = _mm256_xor_si256(ks0, _mm256_loadu_si256(tw + 0));
            ks1 = _mm256_xor_si256(ks1, _mm256_loadu_si256(tw + 1));
            ks2 = _mm256_xor_si256(ks2, _mm256_loadu_si256(tw + 2));
            ks3 = _mm256_xor_si256(ks3, _mm256_loadu_si256(tw + 3));
            ks4 = _mm256_xor_si256(ks4, _mm256_loadu_si256(tw + 4));
            ks5 = _mm256_xor_si256(ks5, _mm256_loadu_si256(tw + 5));
            ks6 = _mm256_xor_si256(ks6, _mm256_loadu_si256(tw + 6));
            ks7 = _mm256_xor_si256(ks7, _mm256_loadu_si256(tw + 7));
        }

        __m256i* o = out256 + i;
        _mm256_storeu_si256(o + 0, ks0);
        _mm256_storeu_si256(o + 1, ks1);
        _mm256_storeu_si256(o + 2, ks2);
        _mm256_storeu_si256(o + 3, ks3);
        _mm256_storeu_si256(o + 4, ks4);
        _mm256_storeu_si256(o + 5, ks5);
        _mm256_storeu_si256(o + 6, ks6);
        _mm256_storeu_si256(o + 7, ks7);
    }

    _mm256_zeroupper();

    // 尾部不足 8 块，回退 SSE2，tweak 指针跟着往后挪
    if (i < n) {
        encryptBlocksFromInputBatch4_SSE2(c, in + i * 32, out + i * 32, (n - i) * 32,
                                          tweaks ? tweaks + i * 32 : nullptr);
    }
}

// SSE2 4块批量解密，从输入数据加载初始状态（CBC/XTS 解密用）
// 解密是加密的逆：先解除输出白化，倒序轮（解除 AddRoundKey → 逆 ShiftRows → 逆 ColumnMix），解除输入白化
inline void decryptBlocksBatch4_SSE2(const BlockCipher& c,
                                     const uint8_t* in, uint8_t* out, int len,
                                     const uint8_t* tweaks = nullptr) {
    int n = len / 32;
    if (n < 4) {
        // 不足 4 块走标量，带 tweak 时把 tweak 揉进加解密
        for (int i = 0; i < n; i++) {
            if (tweaks == nullptr) blockDecrypt(in + i * 32, out + i * 32, c);
            else xtsBlockCrypt(c, in + i * 32, out + i * 32, tweaks + i * 32, false);
        }
        return;
    }

    const uint64_t* keys = c.roundKeys;
    int rounds = c.rounds;
    const __m128i* in128 = reinterpret_cast<const __m128i*>(in);
    __m128i* out128 = reinterpret_cast<__m128i*>(out);

    int i = 0;
    for (; i + 3 < n; i += 4) {
        // 加载 + 转置（同 encryptBlocksFromInputBatch4_SSE2）
        __m128i a0 = _mm_loadu_si128(in128 + (i + 0) * 2);
        __m128i a1 = _mm_loadu_si128(in128 + (i + 1) * 2);
        __m128i a2 = _mm_loadu_si128(in128 + (i + 2) * 2);
        __m128i a3 = _mm_loadu_si128(in128 + (i + 3) * 2);
        // 带 tweak 时先把 tweak 揉进密文，这是 P = D(C^T)^T 的前半步
        if (tweaks != nullptr) {
            const __m128i* tw = reinterpret_cast<const __m128i*>(tweaks + i * 32);
            a0 = _mm_xor_si128(a0, _mm_loadu_si128(tw + 0));
            a1 = _mm_xor_si128(a1, _mm_loadu_si128(tw + 2));
            a2 = _mm_xor_si128(a2, _mm_loadu_si128(tw + 4));
            a3 = _mm_xor_si128(a3, _mm_loadu_si128(tw + 6));
        }
        __m128i r0, r1, r2, r3;
        transpose4x4_sse2(a0, a1, a2, a3, r0, r1, r2, r3);

        __m128i b0 = _mm_loadu_si128(in128 + (i + 0) * 2 + 1);
        __m128i b1 = _mm_loadu_si128(in128 + (i + 1) * 2 + 1);
        __m128i b2 = _mm_loadu_si128(in128 + (i + 2) * 2 + 1);
        __m128i b3 = _mm_loadu_si128(in128 + (i + 3) * 2 + 1);
        if (tweaks != nullptr) {
            const __m128i* tw = reinterpret_cast<const __m128i*>(tweaks + i * 32);
            b0 = _mm_xor_si128(b0, _mm_loadu_si128(tw + 1));
            b1 = _mm_xor_si128(b1, _mm_loadu_si128(tw + 3));
            b2 = _mm_xor_si128(b2, _mm_loadu_si128(tw + 5));
            b3 = _mm_xor_si128(b3, _mm_loadu_si128(tw + 7));
        }
        __m128i r4, r5, r6, r7;
        transpose4x4_sse2(b0, b1, b2, b3, r4, r5, r6, r7);

        // 解除输出白化（最后一组轮密钥）
        int ki = 4 + rounds * 4;
        r0 = _mm_xor_si128(r0, _mm_set1_epi32((uint32_t)keys[ki]));
        r1 = _mm_xor_si128(r1, _mm_set1_epi32((uint32_t)(keys[ki] >> 32)));
        r2 = _mm_xor_si128(r2, _mm_set1_epi32((uint32_t)keys[ki + 1]));
        r3 = _mm_xor_si128(r3, _mm_set1_epi32((uint32_t)(keys[ki + 1] >> 32)));
        r4 = _mm_xor_si128(r4, _mm_set1_epi32((uint32_t)keys[ki + 2]));
        r5 = _mm_xor_si128(r5, _mm_set1_epi32((uint32_t)(keys[ki + 2] >> 32)));
        r6 = _mm_xor_si128(r6, _mm_set1_epi32((uint32_t)keys[ki + 3]));
        r7 = _mm_xor_si128(r7, _mm_set1_epi32((uint32_t)(keys[ki + 3] >> 32)));

        // 倒序轮
        for (int r = rounds - 1; r >= 0; r--) {
            ki = 4 + r * 4;
            // 解除 AddRoundKey
            r0 = _mm_xor_si128(r0, _mm_set1_epi32((uint32_t)keys[ki]));
            r1 = _mm_xor_si128(r1, _mm_set1_epi32((uint32_t)(keys[ki] >> 32)));
            r2 = _mm_xor_si128(r2, _mm_set1_epi32((uint32_t)keys[ki + 1]));
            r3 = _mm_xor_si128(r3, _mm_set1_epi32((uint32_t)(keys[ki + 1] >> 32)));
            r4 = _mm_xor_si128(r4, _mm_set1_epi32((uint32_t)keys[ki + 2]));
            r5 = _mm_xor_si128(r5, _mm_set1_epi32((uint32_t)(keys[ki + 2] >> 32)));
            r6 = _mm_xor_si128(r6, _mm_set1_epi32((uint32_t)keys[ki + 3]));
            r7 = _mm_xor_si128(r7, _mm_set1_epi32((uint32_t)(keys[ki + 3] >> 32)));

            // 逆 ShiftRows：8-循环置换的逆
            { __m128i t = r0; r0 = r6; r6 = r3; r3 = r4; r4 = r5; r5 = r1; r1 = r7; r7 = r2; r2 = t; }

            // 逆 ColumnMix
            if ((r & 1) == 0) BASTION_INV_GMIX_SSE2(r0, r1, r2, r3, r4, r5, r6, r7);
            else BASTION_INV_HMIX_SSE2(r0, r1, r2, r3, r4, r5, r6, r7);
        }

        // 解除输入白化
        r0 = _mm_xor_si128(r0, _mm_set1_epi32((uint32_t)keys[0]));
        r1 = _mm_xor_si128(r1, _mm_set1_epi32((uint32_t)(keys[0] >> 32)));
        r2 = _mm_xor_si128(r2, _mm_set1_epi32((uint32_t)keys[1]));
        r3 = _mm_xor_si128(r3, _mm_set1_epi32((uint32_t)(keys[1] >> 32)));
        r4 = _mm_xor_si128(r4, _mm_set1_epi32((uint32_t)keys[2]));
        r5 = _mm_xor_si128(r5, _mm_set1_epi32((uint32_t)(keys[2] >> 32)));
        r6 = _mm_xor_si128(r6, _mm_set1_epi32((uint32_t)keys[3]));
        r7 = _mm_xor_si128(r7, _mm_set1_epi32((uint32_t)(keys[3] >> 32)));

        // 转置写出
        __m128i o0_lo, o1_lo, o2_lo, o3_lo;
        transpose4x4_sse2(r0, r1, r2, r3, o0_lo, o1_lo, o2_lo, o3_lo);
        __m128i o0_hi, o1_hi, o2_hi, o3_hi;
        transpose4x4_sse2(r4, r5, r6, r7, o0_hi, o1_hi, o2_hi, o3_hi);

        // 带 tweak 时输出前再异或一次，凑齐 P = D(C^T)^T
        if (tweaks != nullptr) {
            const __m128i* tw = reinterpret_cast<const __m128i*>(tweaks + i * 32);
            o0_lo = _mm_xor_si128(o0_lo, _mm_loadu_si128(tw + 0));
            o0_hi = _mm_xor_si128(o0_hi, _mm_loadu_si128(tw + 1));
            o1_lo = _mm_xor_si128(o1_lo, _mm_loadu_si128(tw + 2));
            o1_hi = _mm_xor_si128(o1_hi, _mm_loadu_si128(tw + 3));
            o2_lo = _mm_xor_si128(o2_lo, _mm_loadu_si128(tw + 4));
            o2_hi = _mm_xor_si128(o2_hi, _mm_loadu_si128(tw + 5));
            o3_lo = _mm_xor_si128(o3_lo, _mm_loadu_si128(tw + 6));
            o3_hi = _mm_xor_si128(o3_hi, _mm_loadu_si128(tw + 7));
        }

        __m128i* o = out128 + i * 2;
        _mm_storeu_si128(o + 0, o0_lo);
        _mm_storeu_si128(o + 1, o0_hi);
        _mm_storeu_si128(o + 2, o1_lo);
        _mm_storeu_si128(o + 3, o1_hi);
        _mm_storeu_si128(o + 4, o2_lo);
        _mm_storeu_si128(o + 5, o2_hi);
        _mm_storeu_si128(o + 6, o3_lo);
        _mm_storeu_si128(o + 7, o3_hi);
    }

    // 尾部不足 4 块，带 tweak 时把 tweak 揉进加解密
    for (; i < n; i++) {
        if (tweaks == nullptr) blockDecrypt(in + i * 32, out + i * 32, c);
        else xtsBlockCrypt(c, in + i * 32, out + i * 32, tweaks + i * 32, false);
    }
}

// AVX2 8块批量解密，从输入数据加载初始状态（CBC/XTS 解密用）
// tweaks 非空时按块做 XTS 的两次 tweak 异或（P = D(C^T)^T），省掉外面两遍全缓冲读写
inline void decryptBlocksBatch8_AVX2(const BlockCipher& c,
                                     const uint8_t* in, uint8_t* out, int len,
                                     const uint8_t* tweaks = nullptr) {
    int n = len / 32;
    if (n < 8) {
        decryptBlocksBatch4_SSE2(c, in, out, len, tweaks);
        return;
    }

    const uint64_t* keys = c.roundKeys;
    int rounds = c.rounds;
    const __m256i* in256 = reinterpret_cast<const __m256i*>(in);
    __m256i* out256 = reinterpret_cast<__m256i*>(out);

    int i = 0;
    for (; i + 7 < n; i += 8) {
        // 加载 + 8x8 转置（同 encryptBlocksFromInputBatch8_AVX2）
        __m256i v0 = _mm256_loadu_si256(in256 + i + 0);
        __m256i v1 = _mm256_loadu_si256(in256 + i + 1);
        __m256i v2 = _mm256_loadu_si256(in256 + i + 2);
        __m256i v3 = _mm256_loadu_si256(in256 + i + 3);
        __m256i v4 = _mm256_loadu_si256(in256 + i + 4);
        __m256i v5 = _mm256_loadu_si256(in256 + i + 5);
        __m256i v6 = _mm256_loadu_si256(in256 + i + 6);
        __m256i v7 = _mm256_loadu_si256(in256 + i + 7);

        // 带 tweak 时先把 tweak 揉进密文，这是 P = D(C^T)^T 的前半步
        if (tweaks != nullptr) {
            const __m256i* tw = reinterpret_cast<const __m256i*>(tweaks + i * 32);
            v0 = _mm256_xor_si256(v0, _mm256_loadu_si256(tw + 0));
            v1 = _mm256_xor_si256(v1, _mm256_loadu_si256(tw + 1));
            v2 = _mm256_xor_si256(v2, _mm256_loadu_si256(tw + 2));
            v3 = _mm256_xor_si256(v3, _mm256_loadu_si256(tw + 3));
            v4 = _mm256_xor_si256(v4, _mm256_loadu_si256(tw + 4));
            v5 = _mm256_xor_si256(v5, _mm256_loadu_si256(tw + 5));
            v6 = _mm256_xor_si256(v6, _mm256_loadu_si256(tw + 6));
            v7 = _mm256_xor_si256(v7, _mm256_loadu_si256(tw + 7));
        }

        __m256i t0 = _mm256_unpacklo_epi32(v0, v1);
        __m256i t1 = _mm256_unpackhi_epi32(v0, v1);
        __m256i t2 = _mm256_unpacklo_epi32(v2, v3);
        __m256i t3 = _mm256_unpackhi_epi32(v2, v3);
        __m256i u0 = _mm256_unpacklo_epi64(t0, t2);
        __m256i u1 = _mm256_unpackhi_epi64(t0, t2);
        __m256i u2 = _mm256_unpacklo_epi64(t1, t3);
        __m256i u3 = _mm256_unpackhi_epi64(t1, t3);

        t0 = _mm256_unpacklo_epi32(v4, v5);
        t1 = _mm256_unpackhi_epi32(v4, v5);
        t2 = _mm256_unpacklo_epi32(v6, v7);
        t3 = _mm256_unpackhi_epi32(v6, v7);
        __m256i v0_ = _mm256_unpacklo_epi64(t0, t2);
        __m256i v1_ = _mm256_unpackhi_epi64(t0, t2);
        __m256i v2_ = _mm256_unpacklo_epi64(t1, t3);
        __m256i v3_ = _mm256_unpackhi_epi64(t1, t3);

        __m256i r0 = _mm256_permute2x128_si256(u0, v0_, 0x20);
        __m256i r1 = _mm256_permute2x128_si256(u1, v1_, 0x20);
        __m256i r2 = _mm256_permute2x128_si256(u2, v2_, 0x20);
        __m256i r3 = _mm256_permute2x128_si256(u3, v3_, 0x20);
        __m256i r4 = _mm256_permute2x128_si256(u0, v0_, 0x31);
        __m256i r5 = _mm256_permute2x128_si256(u1, v1_, 0x31);
        __m256i r6 = _mm256_permute2x128_si256(u2, v2_, 0x31);
        __m256i r7 = _mm256_permute2x128_si256(u3, v3_, 0x31);

        // 解除输出白化
        int ki = 4 + rounds * 4;
        r0 = _mm256_xor_si256(r0, _mm256_set1_epi32((uint32_t)keys[ki]));
        r1 = _mm256_xor_si256(r1, _mm256_set1_epi32((uint32_t)(keys[ki] >> 32)));
        r2 = _mm256_xor_si256(r2, _mm256_set1_epi32((uint32_t)keys[ki + 1]));
        r3 = _mm256_xor_si256(r3, _mm256_set1_epi32((uint32_t)(keys[ki + 1] >> 32)));
        r4 = _mm256_xor_si256(r4, _mm256_set1_epi32((uint32_t)keys[ki + 2]));
        r5 = _mm256_xor_si256(r5, _mm256_set1_epi32((uint32_t)(keys[ki + 2] >> 32)));
        r6 = _mm256_xor_si256(r6, _mm256_set1_epi32((uint32_t)keys[ki + 3]));
        r7 = _mm256_xor_si256(r7, _mm256_set1_epi32((uint32_t)(keys[ki + 3] >> 32)));

        // 倒序轮
        for (int r = rounds - 1; r >= 0; r--) {
            ki = 4 + r * 4;
            r0 = _mm256_xor_si256(r0, _mm256_set1_epi32((uint32_t)keys[ki]));
            r1 = _mm256_xor_si256(r1, _mm256_set1_epi32((uint32_t)(keys[ki] >> 32)));
            r2 = _mm256_xor_si256(r2, _mm256_set1_epi32((uint32_t)keys[ki + 1]));
            r3 = _mm256_xor_si256(r3, _mm256_set1_epi32((uint32_t)(keys[ki + 1] >> 32)));
            r4 = _mm256_xor_si256(r4, _mm256_set1_epi32((uint32_t)keys[ki + 2]));
            r5 = _mm256_xor_si256(r5, _mm256_set1_epi32((uint32_t)(keys[ki + 2] >> 32)));
            r6 = _mm256_xor_si256(r6, _mm256_set1_epi32((uint32_t)keys[ki + 3]));
            r7 = _mm256_xor_si256(r7, _mm256_set1_epi32((uint32_t)(keys[ki + 3] >> 32)));

            // 逆 ShiftRows：8-循环置换的逆
            { __m256i t = r0; r0 = r6; r6 = r3; r3 = r4; r4 = r5; r5 = r1; r1 = r7; r7 = r2; r2 = t; }

            if ((r & 1) == 0) BASTION_INV_GMIX_AVX2(r0, r1, r2, r3, r4, r5, r6, r7);
            else BASTION_INV_HMIX_AVX2(r0, r1, r2, r3, r4, r5, r6, r7);
        }

        // 解除输入白化
        r0 = _mm256_xor_si256(r0, _mm256_set1_epi32((uint32_t)keys[0]));
        r1 = _mm256_xor_si256(r1, _mm256_set1_epi32((uint32_t)(keys[0] >> 32)));
        r2 = _mm256_xor_si256(r2, _mm256_set1_epi32((uint32_t)keys[1]));
        r3 = _mm256_xor_si256(r3, _mm256_set1_epi32((uint32_t)(keys[1] >> 32)));
        r4 = _mm256_xor_si256(r4, _mm256_set1_epi32((uint32_t)keys[2]));
        r5 = _mm256_xor_si256(r5, _mm256_set1_epi32((uint32_t)(keys[2] >> 32)));
        r6 = _mm256_xor_si256(r6, _mm256_set1_epi32((uint32_t)keys[3]));
        r7 = _mm256_xor_si256(r7, _mm256_set1_epi32((uint32_t)(keys[3] >> 32)));

        // 8x8 逆转置写出
        t0 = _mm256_unpacklo_epi32(r0, r1);
        t1 = _mm256_unpackhi_epi32(r0, r1);
        t2 = _mm256_unpacklo_epi32(r2, r3);
        t3 = _mm256_unpackhi_epi32(r2, r3);
        u0 = _mm256_unpacklo_epi64(t0, t2);
        u1 = _mm256_unpackhi_epi64(t0, t2);
        u2 = _mm256_unpacklo_epi64(t1, t3);
        u3 = _mm256_unpackhi_epi64(t1, t3);

        t0 = _mm256_unpacklo_epi32(r4, r5);
        t1 = _mm256_unpackhi_epi32(r4, r5);
        t2 = _mm256_unpacklo_epi32(r6, r7);
        t3 = _mm256_unpackhi_epi32(r6, r7);
        v0_ = _mm256_unpacklo_epi64(t0, t2);
        v1_ = _mm256_unpackhi_epi64(t0, t2);
        v2_ = _mm256_unpacklo_epi64(t1, t3);
        v3_ = _mm256_unpackhi_epi64(t1, t3);

        __m256i ks0 = _mm256_permute2x128_si256(u0, v0_, 0x20);
        __m256i ks1 = _mm256_permute2x128_si256(u1, v1_, 0x20);
        __m256i ks2 = _mm256_permute2x128_si256(u2, v2_, 0x20);
        __m256i ks3 = _mm256_permute2x128_si256(u3, v3_, 0x20);
        __m256i ks4 = _mm256_permute2x128_si256(u0, v0_, 0x31);
        __m256i ks5 = _mm256_permute2x128_si256(u1, v1_, 0x31);
        __m256i ks6 = _mm256_permute2x128_si256(u2, v2_, 0x31);
        __m256i ks7 = _mm256_permute2x128_si256(u3, v3_, 0x31);

        // 带 tweak 时输出前再异或一次，凑齐 P = D(C^T)^T
        if (tweaks != nullptr) {
            const __m256i* tw = reinterpret_cast<const __m256i*>(tweaks + i * 32);
            ks0 = _mm256_xor_si256(ks0, _mm256_loadu_si256(tw + 0));
            ks1 = _mm256_xor_si256(ks1, _mm256_loadu_si256(tw + 1));
            ks2 = _mm256_xor_si256(ks2, _mm256_loadu_si256(tw + 2));
            ks3 = _mm256_xor_si256(ks3, _mm256_loadu_si256(tw + 3));
            ks4 = _mm256_xor_si256(ks4, _mm256_loadu_si256(tw + 4));
            ks5 = _mm256_xor_si256(ks5, _mm256_loadu_si256(tw + 5));
            ks6 = _mm256_xor_si256(ks6, _mm256_loadu_si256(tw + 6));
            ks7 = _mm256_xor_si256(ks7, _mm256_loadu_si256(tw + 7));
        }

        __m256i* o = out256 + i;
        _mm256_storeu_si256(o + 0, ks0);
        _mm256_storeu_si256(o + 1, ks1);
        _mm256_storeu_si256(o + 2, ks2);
        _mm256_storeu_si256(o + 3, ks3);
        _mm256_storeu_si256(o + 4, ks4);
        _mm256_storeu_si256(o + 5, ks5);
        _mm256_storeu_si256(o + 6, ks6);
        _mm256_storeu_si256(o + 7, ks7);
    }

    _mm256_zeroupper();

    // 尾部不足 8 块，回退 SSE2，tweak 指针跟着往后挪
    if (i < n) {
        decryptBlocksBatch4_SSE2(c, in + i * 32, out + i * 32, (n - i) * 32,
                                 tweaks ? tweaks + i * 32 : nullptr);
    }
}

// CBC 解密专用：8 块 AVX2 批量解密 + XOR prev 合并写出
// 解密结果 D(C) 直接 XOR prev 写到 out，省去 decBuf 中间缓冲的 100MB 写 + 100MB 读
// in: ciphertext（原始密文，本函数不修改），out: plaintext
// firstPrev: block[0] 的 prev（IV 或前一段最后一块密文）
// 倒序存储保证 in-place（out == in）安全：写出 out[j] 前先读 in[j-1] 作为 p[j]
inline void cbcDecryptBatch8_AVX2_XorPrev(const BlockCipher& c,
                                          const uint8_t* in, uint8_t* out,
                                          const uint8_t* firstPrev) {
    const uint64_t* keys = c.roundKeys;
    int rounds = c.rounds;
    const __m256i* in256 = reinterpret_cast<const __m256i*>(in);
    __m256i* out256 = reinterpret_cast<__m256i*>(out);

    // 加载 8 块 + 8x8 转置（同 decryptBlocksBatch8_AVX2）
    __m256i v0 = _mm256_loadu_si256(in256 + 0);
    __m256i v1 = _mm256_loadu_si256(in256 + 1);
    __m256i v2 = _mm256_loadu_si256(in256 + 2);
    __m256i v3 = _mm256_loadu_si256(in256 + 3);
    __m256i v4 = _mm256_loadu_si256(in256 + 4);
    __m256i v5 = _mm256_loadu_si256(in256 + 5);
    __m256i v6 = _mm256_loadu_si256(in256 + 6);
    __m256i v7 = _mm256_loadu_si256(in256 + 7);

    __m256i t0 = _mm256_unpacklo_epi32(v0, v1);
    __m256i t1 = _mm256_unpackhi_epi32(v0, v1);
    __m256i t2 = _mm256_unpacklo_epi32(v2, v3);
    __m256i t3 = _mm256_unpackhi_epi32(v2, v3);
    __m256i u0 = _mm256_unpacklo_epi64(t0, t2);
    __m256i u1 = _mm256_unpackhi_epi64(t0, t2);
    __m256i u2 = _mm256_unpacklo_epi64(t1, t3);
    __m256i u3 = _mm256_unpackhi_epi64(t1, t3);

    t0 = _mm256_unpacklo_epi32(v4, v5);
    t1 = _mm256_unpackhi_epi32(v4, v5);
    t2 = _mm256_unpacklo_epi32(v6, v7);
    t3 = _mm256_unpackhi_epi32(v6, v7);
    __m256i v0_ = _mm256_unpacklo_epi64(t0, t2);
    __m256i v1_ = _mm256_unpackhi_epi64(t0, t2);
    __m256i v2_ = _mm256_unpacklo_epi64(t1, t3);
    __m256i v3_ = _mm256_unpackhi_epi64(t1, t3);

    __m256i r0 = _mm256_permute2x128_si256(u0, v0_, 0x20);
    __m256i r1 = _mm256_permute2x128_si256(u1, v1_, 0x20);
    __m256i r2 = _mm256_permute2x128_si256(u2, v2_, 0x20);
    __m256i r3 = _mm256_permute2x128_si256(u3, v3_, 0x20);
    __m256i r4 = _mm256_permute2x128_si256(u0, v0_, 0x31);
    __m256i r5 = _mm256_permute2x128_si256(u1, v1_, 0x31);
    __m256i r6 = _mm256_permute2x128_si256(u2, v2_, 0x31);
    __m256i r7 = _mm256_permute2x128_si256(u3, v3_, 0x31);

    // 解除输出白化
    int ki = 4 + rounds * 4;
    r0 = _mm256_xor_si256(r0, _mm256_set1_epi32((uint32_t)keys[ki]));
    r1 = _mm256_xor_si256(r1, _mm256_set1_epi32((uint32_t)(keys[ki] >> 32)));
    r2 = _mm256_xor_si256(r2, _mm256_set1_epi32((uint32_t)keys[ki + 1]));
    r3 = _mm256_xor_si256(r3, _mm256_set1_epi32((uint32_t)(keys[ki + 1] >> 32)));
    r4 = _mm256_xor_si256(r4, _mm256_set1_epi32((uint32_t)keys[ki + 2]));
    r5 = _mm256_xor_si256(r5, _mm256_set1_epi32((uint32_t)(keys[ki + 2] >> 32)));
    r6 = _mm256_xor_si256(r6, _mm256_set1_epi32((uint32_t)keys[ki + 3]));
    r7 = _mm256_xor_si256(r7, _mm256_set1_epi32((uint32_t)(keys[ki + 3] >> 32)));

    // 倒序轮
    for (int r = rounds - 1; r >= 0; r--) {
        ki = 4 + r * 4;
        r0 = _mm256_xor_si256(r0, _mm256_set1_epi32((uint32_t)keys[ki]));
        r1 = _mm256_xor_si256(r1, _mm256_set1_epi32((uint32_t)(keys[ki] >> 32)));
        r2 = _mm256_xor_si256(r2, _mm256_set1_epi32((uint32_t)keys[ki + 1]));
        r3 = _mm256_xor_si256(r3, _mm256_set1_epi32((uint32_t)(keys[ki + 1] >> 32)));
        r4 = _mm256_xor_si256(r4, _mm256_set1_epi32((uint32_t)keys[ki + 2]));
        r5 = _mm256_xor_si256(r5, _mm256_set1_epi32((uint32_t)(keys[ki + 2] >> 32)));
        r6 = _mm256_xor_si256(r6, _mm256_set1_epi32((uint32_t)keys[ki + 3]));
        r7 = _mm256_xor_si256(r7, _mm256_set1_epi32((uint32_t)(keys[ki + 3] >> 32)));

        // 逆 ShiftRows：8-循环置换的逆
        { __m256i t = r0; r0 = r6; r6 = r3; r3 = r4; r4 = r5; r5 = r1; r1 = r7; r7 = r2; r2 = t; }

        if ((r & 1) == 0) BASTION_INV_GMIX_AVX2(r0, r1, r2, r3, r4, r5, r6, r7);
        else BASTION_INV_HMIX_AVX2(r0, r1, r2, r3, r4, r5, r6, r7);
    }

    // 解除输入白化
    r0 = _mm256_xor_si256(r0, _mm256_set1_epi32((uint32_t)keys[0]));
    r1 = _mm256_xor_si256(r1, _mm256_set1_epi32((uint32_t)(keys[0] >> 32)));
    r2 = _mm256_xor_si256(r2, _mm256_set1_epi32((uint32_t)keys[1]));
    r3 = _mm256_xor_si256(r3, _mm256_set1_epi32((uint32_t)(keys[1] >> 32)));
    r4 = _mm256_xor_si256(r4, _mm256_set1_epi32((uint32_t)keys[2]));
    r5 = _mm256_xor_si256(r5, _mm256_set1_epi32((uint32_t)(keys[2] >> 32)));
    r6 = _mm256_xor_si256(r6, _mm256_set1_epi32((uint32_t)keys[3]));
    r7 = _mm256_xor_si256(r7, _mm256_set1_epi32((uint32_t)(keys[3] >> 32)));

    // 8x8 逆转置得到 ks0..ks7（解密结果 D(C)）
    t0 = _mm256_unpacklo_epi32(r0, r1);
    t1 = _mm256_unpackhi_epi32(r0, r1);
    t2 = _mm256_unpacklo_epi32(r2, r3);
    t3 = _mm256_unpackhi_epi32(r2, r3);
    u0 = _mm256_unpacklo_epi64(t0, t2);
    u1 = _mm256_unpackhi_epi64(t0, t2);
    u2 = _mm256_unpacklo_epi64(t1, t3);
    u3 = _mm256_unpackhi_epi64(t1, t3);

    t0 = _mm256_unpacklo_epi32(r4, r5);
    t1 = _mm256_unpackhi_epi32(r4, r5);
    t2 = _mm256_unpacklo_epi32(r6, r7);
    t3 = _mm256_unpackhi_epi32(r6, r7);
    v0_ = _mm256_unpacklo_epi64(t0, t2);
    v1_ = _mm256_unpackhi_epi64(t0, t2);
    v2_ = _mm256_unpacklo_epi64(t1, t3);
    v3_ = _mm256_unpackhi_epi64(t1, t3);

    __m256i ks0 = _mm256_permute2x128_si256(u0, v0_, 0x20);
    __m256i ks1 = _mm256_permute2x128_si256(u1, v1_, 0x20);
    __m256i ks2 = _mm256_permute2x128_si256(u2, v2_, 0x20);
    __m256i ks3 = _mm256_permute2x128_si256(u3, v3_, 0x20);
    __m256i ks4 = _mm256_permute2x128_si256(u0, v0_, 0x31);
    __m256i ks5 = _mm256_permute2x128_si256(u1, v1_, 0x31);
    __m256i ks6 = _mm256_permute2x128_si256(u2, v2_, 0x31);
    __m256i ks7 = _mm256_permute2x128_si256(u3, v3_, 0x31);

    // 倒序存储 + XOR prev：p[0]=firstPrev，p[j]=in[j-1] for j=1..7
    // 先写 out[7]（读 in[6]），再 out[6]（读 in[5]）... 最后 out[0]（读 firstPrev）
    // 这样 in-place 时写出不会覆盖待读的 prev
    __m256i p7 = _mm256_loadu_si256(in256 + 6);
    _mm256_storeu_si256(out256 + 7, _mm256_xor_si256(ks7, p7));
    __m256i p6 = _mm256_loadu_si256(in256 + 5);
    _mm256_storeu_si256(out256 + 6, _mm256_xor_si256(ks6, p6));
    __m256i p5 = _mm256_loadu_si256(in256 + 4);
    _mm256_storeu_si256(out256 + 5, _mm256_xor_si256(ks5, p5));
    __m256i p4 = _mm256_loadu_si256(in256 + 3);
    _mm256_storeu_si256(out256 + 4, _mm256_xor_si256(ks4, p4));
    __m256i p3 = _mm256_loadu_si256(in256 + 2);
    _mm256_storeu_si256(out256 + 3, _mm256_xor_si256(ks3, p3));
    __m256i p2 = _mm256_loadu_si256(in256 + 1);
    _mm256_storeu_si256(out256 + 2, _mm256_xor_si256(ks2, p2));
    __m256i p1 = _mm256_loadu_si256(in256 + 0);
    _mm256_storeu_si256(out256 + 1, _mm256_xor_si256(ks1, p1));
    __m256i p0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(firstPrev));
    _mm256_storeu_si256(out256 + 0, _mm256_xor_si256(ks0, p0));

    _mm256_zeroupper();
}

// CBC 解密专用：4 块 SSE2 批量解密 + XOR prev 合并写出（无 decBuf）
// 倒序存储保证 in-place 安全
inline void cbcDecryptBatch4_SSE2_XorPrev(const BlockCipher& c,
                                          const uint8_t* in, uint8_t* out,
                                          const uint8_t* firstPrev) {
    const uint64_t* keys = c.roundKeys;
    int rounds = c.rounds;
    const __m128i* in128 = reinterpret_cast<const __m128i*>(in);
    __m128i* out128 = reinterpret_cast<__m128i*>(out);

    // 加载 + 转置（同 decryptBlocksBatch4_SSE2）
    __m128i a0 = _mm_loadu_si128(in128 + 0);
    __m128i a1 = _mm_loadu_si128(in128 + 2);
    __m128i a2 = _mm_loadu_si128(in128 + 4);
    __m128i a3 = _mm_loadu_si128(in128 + 6);
    __m128i r0, r1, r2, r3;
    transpose4x4_sse2(a0, a1, a2, a3, r0, r1, r2, r3);

    __m128i b0 = _mm_loadu_si128(in128 + 1);
    __m128i b1 = _mm_loadu_si128(in128 + 3);
    __m128i b2 = _mm_loadu_si128(in128 + 5);
    __m128i b3 = _mm_loadu_si128(in128 + 7);
    __m128i r4, r5, r6, r7;
    transpose4x4_sse2(b0, b1, b2, b3, r4, r5, r6, r7);

    // 解除输出白化
    int ki = 4 + rounds * 4;
    r0 = _mm_xor_si128(r0, _mm_set1_epi32((uint32_t)keys[ki]));
    r1 = _mm_xor_si128(r1, _mm_set1_epi32((uint32_t)(keys[ki] >> 32)));
    r2 = _mm_xor_si128(r2, _mm_set1_epi32((uint32_t)keys[ki + 1]));
    r3 = _mm_xor_si128(r3, _mm_set1_epi32((uint32_t)(keys[ki + 1] >> 32)));
    r4 = _mm_xor_si128(r4, _mm_set1_epi32((uint32_t)keys[ki + 2]));
    r5 = _mm_xor_si128(r5, _mm_set1_epi32((uint32_t)(keys[ki + 2] >> 32)));
    r6 = _mm_xor_si128(r6, _mm_set1_epi32((uint32_t)keys[ki + 3]));
    r7 = _mm_xor_si128(r7, _mm_set1_epi32((uint32_t)(keys[ki + 3] >> 32)));

    // 倒序轮
    for (int r = rounds - 1; r >= 0; r--) {
        ki = 4 + r * 4;
        r0 = _mm_xor_si128(r0, _mm_set1_epi32((uint32_t)keys[ki]));
        r1 = _mm_xor_si128(r1, _mm_set1_epi32((uint32_t)(keys[ki] >> 32)));
        r2 = _mm_xor_si128(r2, _mm_set1_epi32((uint32_t)keys[ki + 1]));
        r3 = _mm_xor_si128(r3, _mm_set1_epi32((uint32_t)(keys[ki + 1] >> 32)));
        r4 = _mm_xor_si128(r4, _mm_set1_epi32((uint32_t)keys[ki + 2]));
        r5 = _mm_xor_si128(r5, _mm_set1_epi32((uint32_t)(keys[ki + 2] >> 32)));
        r6 = _mm_xor_si128(r6, _mm_set1_epi32((uint32_t)keys[ki + 3]));
        r7 = _mm_xor_si128(r7, _mm_set1_epi32((uint32_t)(keys[ki + 3] >> 32)));

        // 逆 ShiftRows：8-循环置换的逆
        { __m128i t = r0; r0 = r6; r6 = r3; r3 = r4; r4 = r5; r5 = r1; r1 = r7; r7 = r2; r2 = t; }

        if ((r & 1) == 0) BASTION_INV_GMIX_SSE2(r0, r1, r2, r3, r4, r5, r6, r7);
        else BASTION_INV_HMIX_SSE2(r0, r1, r2, r3, r4, r5, r6, r7);
    }

    // 解除输入白化
    r0 = _mm_xor_si128(r0, _mm_set1_epi32((uint32_t)keys[0]));
    r1 = _mm_xor_si128(r1, _mm_set1_epi32((uint32_t)(keys[0] >> 32)));
    r2 = _mm_xor_si128(r2, _mm_set1_epi32((uint32_t)keys[1]));
    r3 = _mm_xor_si128(r3, _mm_set1_epi32((uint32_t)(keys[1] >> 32)));
    r4 = _mm_xor_si128(r4, _mm_set1_epi32((uint32_t)keys[2]));
    r5 = _mm_xor_si128(r5, _mm_set1_epi32((uint32_t)(keys[2] >> 32)));
    r6 = _mm_xor_si128(r6, _mm_set1_epi32((uint32_t)keys[3]));
    r7 = _mm_xor_si128(r7, _mm_set1_epi32((uint32_t)(keys[3] >> 32)));

    // 转置得到 4 块的 D(C)
    __m128i o0_lo, o1_lo, o2_lo, o3_lo;
    transpose4x4_sse2(r0, r1, r2, r3, o0_lo, o1_lo, o2_lo, o3_lo);
    __m128i o0_hi, o1_hi, o2_hi, o3_hi;
    transpose4x4_sse2(r4, r5, r6, r7, o0_hi, o1_hi, o2_hi, o3_hi);

    // 倒序存储 + XOR prev：p[0]=firstPrev，p[j]=in[j-1] for j=1..3
    // block j 占 in128 + j*2（low）和 in128 + j*2 + 1（high）
    __m128i p3_lo = _mm_loadu_si128(in128 + 4);  // in[2]
    __m128i p3_hi = _mm_loadu_si128(in128 + 5);
    _mm_storeu_si128(out128 + 6, _mm_xor_si128(o3_lo, p3_lo));
    _mm_storeu_si128(out128 + 7, _mm_xor_si128(o3_hi, p3_hi));

    __m128i p2_lo = _mm_loadu_si128(in128 + 2);  // in[1]
    __m128i p2_hi = _mm_loadu_si128(in128 + 3);
    _mm_storeu_si128(out128 + 4, _mm_xor_si128(o2_lo, p2_lo));
    _mm_storeu_si128(out128 + 5, _mm_xor_si128(o2_hi, p2_hi));

    __m128i p1_lo = _mm_loadu_si128(in128 + 0);  // in[0]
    __m128i p1_hi = _mm_loadu_si128(in128 + 1);
    _mm_storeu_si128(out128 + 2, _mm_xor_si128(o1_lo, p1_lo));
    _mm_storeu_si128(out128 + 3, _mm_xor_si128(o1_hi, p1_hi));

    __m128i p0_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(firstPrev));
    __m128i p0_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(firstPrev + 16));
    _mm_storeu_si128(out128 + 0, _mm_xor_si128(o0_lo, p0_lo));
    _mm_storeu_si128(out128 + 1, _mm_xor_si128(o0_hi, p0_hi));
}

// XTS GF(2^256) 乘 2

// GF(2^256) 乘 2 寄存器版：tweak 保持在 lo/hi 寄存器中，预计算 tweak 链时省掉每次 load
// 逻辑跟 gfMul2_256 一致，只是输入输出在寄存器而不是内存
inline void gfMul2_reg(__m128i& lo, __m128i& hi) {
    static const __m128i maskFE = _mm_set1_epi16((short)(unsigned short)0xFEFE);
    static const __m128i mask01 = _mm_set1_epi16((short)(unsigned short)0x0101);
    static const __m128i polyMask = _mm_set_epi8(0x25, 0x04, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    static const __m128i crossMask = _mm_set_epi8(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    // 无分支：carry/cross 位扩展成全 1 掩码，用 AND 控制是否注入
    // 进位位是随机的，分支预测器会误预测，掩码方式免去分支惩罚
    int carry = _mm_movemask_epi8(lo) & 1;
    int cross = _mm_movemask_epi8(hi) & 1;

    __m128i shiftsLo = _mm_and_si128(_mm_slli_epi16(lo, 1), maskFE);
    __m128i shiftsHi = _mm_and_si128(_mm_slli_epi16(hi, 1), maskFE);
    __m128i carriesLo = _mm_and_si128(_mm_srli_epi16(lo, 7), mask01);
    __m128i carriesHi = _mm_and_si128(_mm_srli_epi16(hi, 7), mask01);
    __m128i carriesLoShifted = _mm_srli_si128(carriesLo, 1);
    __m128i carriesHiShifted = _mm_srli_si128(carriesHi, 1);

    // cross=1 时注入 crossMask（x[16] 高位进位到 x[15] 的 bit0），否则不注入
    __m128i crossSel = _mm_set1_epi8((char)(-cross));
    lo = _mm_or_si128(shiftsLo, _mm_or_si128(carriesLoShifted, _mm_and_si128(crossSel, crossMask)));
    hi = _mm_or_si128(shiftsHi, carriesHiShifted);
    // carry=1 时异或多项式（x[0] 高位进位到 x[30]/x[31]），否则不变
    __m128i carrySel = _mm_set1_epi8((char)(-carry));
    hi = _mm_xor_si128(hi, _mm_and_si128(carrySel, polyMask));
}

} // namespace

// 对外暴露的 SIMD 函数

// 按 SIMD 级别分发整块加密（len 必须是 32 的整数倍且 > 0）
// 调度策略：AVX2 一次 8 块，SSE2 一次 4 块，标量一次 4 块（回退）
// 各批量函数内部会处理尾部不足批次的情况，调用方无需关心
void encryptBlocksDispatched(const BlockCipher& c, const uint8_t* counter,
                             const uint8_t* in, uint8_t* out, int len) {
    if (g_simdLevel >= 2 && len >= 8 * 32) {
        encryptBlocksBatch8_AVX2(c, counter, in, out, len);
    } else if (g_simdLevel >= 1 && len >= 4 * 32) {
        encryptBlocksBatch4_SSE2(c, counter, in, out, len);
    } else {
        encryptBlocksBatch4(c, counter, in, out, len);
    }
}

// 按 SIMD 级别分发从输入加载的批量加密（XTS 加密用）
void encryptBlocksFromInputDispatched(const BlockCipher& c,
                                      const uint8_t* in, uint8_t* out, int len) {
    if (g_simdLevel >= 2 && len >= 8 * 32) {
        encryptBlocksFromInputBatch8_AVX2(c, in, out, len);
    } else if (g_simdLevel >= 1 && len >= 4 * 32) {
        encryptBlocksFromInputBatch4_SSE2(c, in, out, len);
    } else {
        int n = len / 32;
        for (int i = 0; i < n; i++) blockEncrypt(in + i * 32, out + i * 32, c);
    }
}

// 按 SIMD 级别分发批量解密（CBC/XTS 解密用）
void decryptBlocksDispatched(const BlockCipher& c,
                             const uint8_t* in, uint8_t* out, int len) {
    if (g_simdLevel >= 2 && len >= 8 * 32) {
        decryptBlocksBatch8_AVX2(c, in, out, len);
    } else if (g_simdLevel >= 1 && len >= 4 * 32) {
        decryptBlocksBatch4_SSE2(c, in, out, len);
    } else {
        int n = len / 32;
        for (int i = 0; i < n; i++) blockDecrypt(in + i * 32, out + i * 32, c);
    }
}

// CBC 解密一段连续块：解密 + XOR prev 合并，无 decBuf 中间缓冲
// 8/4 块走 AVX2/SSE2 批量合并函数，尾部走单块标量
void cbcDecryptRange(const BlockCipher& c,
                     const uint8_t* in, uint8_t* out, int blocks,
                     const uint8_t* firstPrev) {
    uint8_t savedPrev[32];  // 保存下一批的 firstPrev（in-place 时避免覆盖）
    uint8_t curPrev[32];    // 当前批的 prev 副本，与 savedPrev 分开存放，避免提前复制时互相覆盖
    std::memcpy(curPrev, firstPrev, 32);
    int i = 0;

    // savedPrev 必须在批量写出之前复制：in == out 时批量会覆盖本批最后一块密文
    while (g_simdLevel >= 2 && i + 8 <= blocks) {
        std::memcpy(savedPrev, in + (i + 7) * 32, 32);
        cbcDecryptBatch8_AVX2_XorPrev(c, in + i * 32, out + i * 32, curPrev);
        std::memcpy(curPrev, savedPrev, 32);
        i += 8;
    }
    while (g_simdLevel >= 1 && i + 4 <= blocks) {
        std::memcpy(savedPrev, in + (i + 3) * 32, 32);
        cbcDecryptBatch4_SSE2_XorPrev(c, in + i * 32, out + i * 32, curPrev);
        std::memcpy(curPrev, savedPrev, 32);
        i += 4;
    }
    while (i < blocks) {
        uint8_t decBlock[32];
        blockDecrypt(in + i * 32, decBlock, c);
        // 先复制本块原始密文再做 XOR 写出，in == out 时写出会覆盖待读的密文
        std::memcpy(savedPrev, in + i * 32, 32);
        // 标量 XOR prev：结果与 SIMD 版本一致，且不依赖 SSE2（g_simdLevel==0 时也能跑）
        xorWords(decBlock, curPrev, out + i * 32, 32);
        std::memcpy(curPrev, savedPrev, 32);
        i++;
    }
}

// XTS 辅助函数

// 原地 XOR 一个 32 字节块，用两条 128 位 XOR 替代 xorWords 的 4 次 uint64 操作
void xorBlock32(uint8_t* blk, const uint8_t* tweak) {
    __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk));
    __m128i t0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tweak));
    __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 16));
    __m128i t1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tweak + 16));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(blk), _mm_xor_si128(v0, t0));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(blk + 16), _mm_xor_si128(v1, t1));
}

// GF(2^256) 乘 2，多项式 0x0425，进位时异或到 x[30] 和 x[31]
// tweak 是大端字节序，乘 2 = 整体左移 1 位
// MSVC 不支持字节级移位，用 16 位移位 + mask 模拟：_mm_slli_epi16 & 0xFEFE 左移，_mm_srli_epi16 & 0x0101 取进位
void gfMul2_256(uint8_t x[32]) {
    static const __m128i maskFE = _mm_set1_epi16((short)(unsigned short)0xFEFE);  // 字节左移后清跨字节进位
    static const __m128i mask01 = _mm_set1_epi16((short)(unsigned short)0x0101);  // 字节右移后只保留 bit0
    // 多项式掩码：x[30] ^= 0x04, x[31] ^= 0x25（hi 的 byte14/byte15）
    static const __m128i polyMask = _mm_set_epi8(0x25, 0x04, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    // 跨边界进位掩码：x[16] 的高位放到 lo 的 byte15
    static const __m128i crossMask = _mm_set_epi8(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x));
    __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + 16));

    // movemask 返回 16 位，bit0 = byte0 最高位
    int carry = _mm_movemask_epi8(lo) & 1;   // x[0] >> 7
    int cross = _mm_movemask_epi8(hi) & 1;   // x[16] >> 7

    // 每字节左移 1 位：16 位左移后 mask 掉跨字节进位
    __m128i shiftsLo = _mm_and_si128(_mm_slli_epi16(lo, 1), maskFE);
    __m128i shiftsHi = _mm_and_si128(_mm_slli_epi16(hi, 1), maskFE);

    // 每字节高位（进位源：x[i] >> 7）：16 位右移后只保留每字节 bit0
    __m128i carriesLo = _mm_and_si128(_mm_srli_epi16(lo, 7), mask01);
    __m128i carriesHi = _mm_and_si128(_mm_srli_epi16(hi, 7), mask01);

    // 进位向低地址移 1 字节：byte i = 旧 byte i+1，byte15 = 0
    __m128i carriesLoShifted = _mm_srli_si128(carriesLo, 1);
    __m128i carriesHiShifted = _mm_srli_si128(carriesHi, 1);

    // 跨 128 位边界：新 x[15] 低位 = x[16] 高位
    if (cross) carriesLoShifted = _mm_or_si128(carriesLoShifted, crossMask);

    __m128i newLo = _mm_or_si128(shiftsLo, carriesLoShifted);
    __m128i newHi = _mm_or_si128(shiftsHi, carriesHiShifted);

    // 最高位进位异或多项式
    if (carry) newHi = _mm_xor_si128(newHi, polyMask);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(x), newLo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(x + 16), newHi);
}

// XTS 处理一段连续完整块，从 startTweak 开始，处理完后把推进后的 tweak 写到 outTweak
// tweak 异或已经融合进批量加解密（加密 C = E(P^T)^T，解密 P = D(C^T)^T），
// tweak 链在寄存器里按批推进，省掉先整段预计算 tweak 再整段 XOR 两遍的访存
void xtsProcessSegment(const BlockCipher& c1, const uint8_t* startTweak,
                       uint8_t* out, int blocks, bool encrypt, uint8_t* outTweak) {
    __m128i tLo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(startTweak));
    __m128i tHi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(startTweak + 16));

    int b = 0;

    // AVX2 一批 64 块（8 个 8 块批次），tweak 落进小数组交给批量函数一起异或
    if (g_simdLevel >= 2) {
        alignas(16) uint8_t tweaks[64][32];
        while (blocks - b >= 64) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(tweaks[0]), tLo);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(tweaks[0] + 16), tHi);
            for (int j = 1; j < 64; j++) {
                gfMul2_reg(tLo, tHi);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(tweaks[j]), tLo);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(tweaks[j] + 16), tHi);
            }
            // 推进到下一批的起始 tweak（本批最后一个的下一个）
            gfMul2_reg(tLo, tHi);

            if (encrypt) {
                encryptBlocksFromInputBatch8_AVX2(c1, out + b * 32, out + b * 32, 64 * 32, &tweaks[0][0]);
            } else {
                decryptBlocksBatch8_AVX2(c1, out + b * 32, out + b * 32, 64 * 32, &tweaks[0][0]);
            }
            b += 64;
        }
    }

    // SSE2 一批 32 块
    if (g_simdLevel >= 1) {
        alignas(16) uint8_t tweaks[32][32];
        while (blocks - b >= 32) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(tweaks[0]), tLo);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(tweaks[0] + 16), tHi);
            for (int j = 1; j < 32; j++) {
                gfMul2_reg(tLo, tHi);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(tweaks[j]), tLo);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(tweaks[j] + 16), tHi);
            }
            gfMul2_reg(tLo, tHi);

            if (encrypt) {
                encryptBlocksFromInputBatch4_SSE2(c1, out + b * 32, out + b * 32, 32 * 32, &tweaks[0][0]);
            } else {
                decryptBlocksBatch4_SSE2(c1, out + b * 32, out + b * 32, 32 * 32, &tweaks[0][0]);
            }
            b += 32;
        }
    }

    // 尾巴不足一批：按最多 8 块一组交给批量函数，内部还会再退到 4 块 / 单块
    while (b < blocks) {
        int cnt = blocks - b;
        if (cnt > 8) cnt = 8;

        alignas(16) uint8_t tweaks[8][32];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(tweaks[0]), tLo);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(tweaks[0] + 16), tHi);
        for (int j = 1; j < cnt; j++) {
            gfMul2_reg(tLo, tHi);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(tweaks[j]), tLo);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(tweaks[j] + 16), tHi);
        }
        // 推进到下一组的起始 tweak
        gfMul2_reg(tLo, tHi);

        if (g_simdLevel >= 2) {
            if (encrypt) {
                encryptBlocksFromInputBatch8_AVX2(c1, out + b * 32, out + b * 32, cnt * 32, &tweaks[0][0]);
            } else {
                decryptBlocksBatch8_AVX2(c1, out + b * 32, out + b * 32, cnt * 32, &tweaks[0][0]);
            }
        } else if (g_simdLevel >= 1) {
            if (encrypt) {
                encryptBlocksFromInputBatch4_SSE2(c1, out + b * 32, out + b * 32, cnt * 32, &tweaks[0][0]);
            } else {
                decryptBlocksBatch4_SSE2(c1, out + b * 32, out + b * 32, cnt * 32, &tweaks[0][0]);
            }
        } else {
            for (int j = 0; j < cnt; j++) {
                xtsBlockCrypt(c1, out + (b + j) * 32, out + (b + j) * 32, tweaks[j], encrypt);
            }
        }
        b += cnt;
    }

    // 把最终 tweak 写回内存
    if (outTweak) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(outTweak), tLo);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(outTweak + 16), tHi);
    }
}

// 4 条独立链并行哈希压缩（SSE2）
// 每 XMM 装 4 条链的同一个状态字（与 CTR 批量加密的按块并行布局一致），
// 16 轮 G/H Mix 是纯 lane 运算（无 shuffle），Davies-Meyer 前馈逐 lane 独立
// stT 为状态字序（stT[k][i]=链 i 的 s_k），状态直接 load/store，消息经 4x4 转置
void arxCompress4xSSE2(uint32_t stT[8][4], const uint8_t* blk[4], bool last) {
    // 加载状态（状态字序，每行 4 链的同一个字，直接 16 字节 load）
    __m128i S0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&stT[0]));
    __m128i S1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&stT[1]));
    __m128i S2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&stT[2]));
    __m128i S3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&stT[3]));
    __m128i S4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&stT[4]));
    __m128i S5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&stT[5]));
    __m128i S6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&stT[6]));
    __m128i S7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&stT[7]));

    // 旧状态与消息按状态字序存栈（前馈用）
    alignas(16) __m128i oldT[8] = { S0, S1, S2, S3, S4, S5, S6, S7 };
    alignas(16) __m128i msgT[8];

    // 加载 4 块消息 → M0-M7（块序经 4x4 转置成状态字序）
    __m128i M0, M1, M2, M3, M4, M5, M6, M7;
    {
        __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk[0]));
        __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk[1]));
        __m128i r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk[2]));
        __m128i r3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk[3]));
        transpose4x4_sse2(r0, r1, r2, r3, M0, M1, M2, M3);
        r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk[0] + 16));
        r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk[1] + 16));
        r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk[2] + 16));
        r3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk[3] + 16));
        transpose4x4_sse2(r0, r1, r2, r3, M4, M5, M6, M7);
    }
    msgT[0] = M0; msgT[1] = M1; msgT[2] = M2; msgT[3] = M3;
    msgT[4] = M4; msgT[5] = M5; msgT[6] = M6; msgT[7] = M7;

    // 末块封口常数：只加在吸收阶段，不参与末尾前馈，与标量实现一致
    __m128i F0 = _mm_set1_epi32(last ? (int)kFinalConst[0] : 0);
    __m128i F1 = _mm_set1_epi32(last ? (int)kFinalConst[1] : 0);
    __m128i F2 = _mm_set1_epi32(last ? (int)kFinalConst[2] : 0);
    __m128i F3 = _mm_set1_epi32(last ? (int)kFinalConst[3] : 0);
    __m128i F4 = _mm_set1_epi32(last ? (int)kFinalConst[4] : 0);
    __m128i F5 = _mm_set1_epi32(last ? (int)kFinalConst[5] : 0);
    __m128i F6 = _mm_set1_epi32(last ? (int)kFinalConst[6] : 0);
    __m128i F7 = _mm_set1_epi32(last ? (int)kFinalConst[7] : 0);

    // 消息模加注入（末块把封口常数一并加进去，加在 16 轮之前）
    S0 = _mm_add_epi32(_mm_add_epi32(S0, M0), F0); S1 = _mm_add_epi32(_mm_add_epi32(S1, M1), F1);
    S2 = _mm_add_epi32(_mm_add_epi32(S2, M2), F2); S3 = _mm_add_epi32(_mm_add_epi32(S3, M3), F3);
    S4 = _mm_add_epi32(_mm_add_epi32(S4, M4), F4); S5 = _mm_add_epi32(_mm_add_epi32(S5, M5), F5);
    S6 = _mm_add_epi32(_mm_add_epi32(S6, M6), F6); S7 = _mm_add_epi32(_mm_add_epi32(S7, M7), F7);

    // 16 轮 G/H Mix + ShiftRows + 轮常数
    for (int r = 0; r < ARX_ROUNDS; r++) {
        if ((r & 1) == 0) {
            BASTION_GMIX_SSE2(S0, S1, S2, S3, S4, S5, S6, S7);
        } else {
            BASTION_HMIX_SSE2(S0, S1, S2, S3, S4, S5, S6, S7);
        }
        // ShiftRows：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
        { __m128i t = S0; S0 = S2; S2 = S7; S7 = S1; S1 = S5; S5 = S4; S4 = S3; S3 = S6; S6 = t; }
        S0 = _mm_xor_si128(S0, _mm_set1_epi32(kRoundConstants[r][0]));
        S1 = _mm_xor_si128(S1, _mm_set1_epi32(kRoundConstants[r][1]));
        S2 = _mm_xor_si128(S2, _mm_set1_epi32(kRoundConstants[r][2]));
        S3 = _mm_xor_si128(S3, _mm_set1_epi32(kRoundConstants[r][3]));
        S4 = _mm_xor_si128(S4, _mm_set1_epi32(kRoundConstants[r][4]));
        S5 = _mm_xor_si128(S5, _mm_set1_epi32(kRoundConstants[r][5]));
        S6 = _mm_xor_si128(S6, _mm_set1_epi32(kRoundConstants[r][6]));
        S7 = _mm_xor_si128(S7, _mm_set1_epi32(kRoundConstants[r][7]));
    }

    // Davies-Meyer 前馈：S_k = (S_k ^ old_k) + msg_k
    S0 = _mm_add_epi32(_mm_xor_si128(S0, oldT[0]), msgT[0]);
    S1 = _mm_add_epi32(_mm_xor_si128(S1, oldT[1]), msgT[1]);
    S2 = _mm_add_epi32(_mm_xor_si128(S2, oldT[2]), msgT[2]);
    S3 = _mm_add_epi32(_mm_xor_si128(S3, oldT[3]), msgT[3]);
    S4 = _mm_add_epi32(_mm_xor_si128(S4, oldT[4]), msgT[4]);
    S5 = _mm_add_epi32(_mm_xor_si128(S5, oldT[5]), msgT[5]);
    S6 = _mm_add_epi32(_mm_xor_si128(S6, oldT[6]), msgT[6]);
    S7 = _mm_add_epi32(_mm_xor_si128(S7, oldT[7]), msgT[7]);

    // 写回（状态字序）
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&stT[0]), S0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&stT[1]), S1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&stT[2]), S2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&stT[3]), S3);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&stT[4]), S4);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&stT[5]), S5);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&stT[6]), S6);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&stT[7]), S7);
}
