// 平台化安全清零的能力探测，必须放在任何标准库头文件之前：
// macOS 先请求 C11 附录 K，<string.h> 才会暴露 memset_s
#if defined(__APPLE__) && !defined(__STDC_WANT_LIB_EXT1__)
#define __STDC_WANT_LIB_EXT1__ 1
#endif
// glibc 在 -std=c++17（__STRICT_ANSI__）下默认关掉 _DEFAULT_SOURCE，补开后才有 explicit_bzero 声明
#if defined(__linux__) && !defined(_DEFAULT_SOURCE)
#define _DEFAULT_SOURCE 1
#endif
// explicit_bzero 在 glibc 2.25 起提供，BSD 系原生提供
#if (defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 25))) || \
    defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__) || defined(__DragonFly__)
#define BASTION_HAVE_EXPLICIT_BZERO 1
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>    // SecureZeroMemory
#endif

#include "Bastion.h"
#include "Bastion_internal.h"
#include <cstring>
#include <memory>
#include <mutex>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>
#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>

namespace {

// 压缩函数是手动展开 16 轮，编译期核对常量，改轮数时不会漏同步
static_assert(ARX_ROUNDS == 16, "ARX_ROUNDS must match the unrolled rounds");
// GF(2^8) 用的不可约多项式 0x12D
constexpr uint8_t GF8_POLY = 0x2D;

// 每个 S 盒异或的 delta 值，0-3 号盒各一个
const uint8_t SBOX_DELTAS[4] = {0x00, 0x5A, 0xB4, 0x2D};

// 预移位好的 S 盒表，查表直接拿结果，省得每次移位
uint32_t sbox0T0[256], sbox1T8[256], sbox2T16[256], sbox3T24[256];

// 16 轮轮常数、每轮 8 个 uint32 定义见 Bastion_internal.h（inline constexpr，供标量与 SIMD 共用）

// 统一用右旋，加密方向全用这个
inline uint32_t rotr32(uint32_t x, int r) {
    return (x >> r) | (x << (32 - r));
}

// 左旋是右旋的逆，解密时才用到
inline uint32_t rotl32(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

// MurmurHash3 finalizer 改的，给常数掺非线性
inline uint32_t xmxFinalize(uint32_t x) {
    x ^= x >> 16;
    x *= 0x45d9f3b;
    x ^= x >> 16;
    x *= 0x45d9f3b;
    x ^= x >> 16;
    return x;
}

// GF(2^8) 乘法，多项式 0x12D
uint8_t gf8Mul(uint8_t a, uint8_t b) {
    uint8_t result = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1) result ^= a;
        uint8_t carry = a & 0x80;
        a <<= 1;
        if (carry) a ^= GF8_POLY;
        b >>= 1;
    }
    return result;
}

// GF(2^8) 求逆，费马小定理硬算，0 的逆特殊处理
uint8_t gf8Inv(uint8_t a) {
    if (a == 0) return 0;
    uint8_t result = 1;
    uint8_t base = a;
    // a^254 = a^-1，因为 a^255 = 1
    for (int exp = 254; exp > 0; exp >>= 1) {
        if (exp & 1) result = gf8Mul(result, base);
        base = gf8Mul(base, base);
    }
    return result;
}

// Bastion 自己的仿射变换，不是 AES 那种循环矩阵
uint8_t affineBastion(uint8_t x) {
    int b0 = (x >> 0) & 1, b1 = (x >> 1) & 1, b2 = (x >> 2) & 1, b3 = (x >> 3) & 1;
    int b4 = (x >> 4) & 1, b5 = (x >> 5) & 1, b6 = (x >> 6) & 1, b7 = (x >> 7) & 1;

    int y0 = b7 ^ b5 ^ b4 ^ b3 ^ b1 ^ 1;
    int y1 = b7 ^ b6 ^ b4 ^ b3 ^ b2;
    int y2 = b6 ^ b5 ^ b3 ^ b2 ^ b1 ^ 1;
    int y3 = b5 ^ b4 ^ b2 ^ b1 ^ b0 ^ 1;
    int y4 = b7 ^ b4 ^ b3 ^ b1 ^ b0;
    int y5 = b7 ^ b6 ^ b3 ^ b2 ^ b0 ^ 1;
    int y6 = b6 ^ b5 ^ b2 ^ b1 ^ b0;
    int y7 = b7 ^ b5 ^ b4 ^ b0 ^ 1;

    return (uint8_t)((y0 & 1) | ((y1 & 1) << 1) | ((y2 & 1) << 2) | ((y3 & 1) << 3) |
                     ((y4 & 1) << 4) | ((y5 & 1) << 5) | ((y6 & 1) << 6) | ((y7 & 1) << 7));
}

// 基础 S 盒 = 仿射变换(求逆(x))
void genSbox(uint8_t out[256]) {
    for (int i = 0; i < 256; i++) {
        out[i] = affineBastion(gf8Inv((uint8_t)i));
    }
}

// 用 4 个 S 盒替换一个 uint32 的 4 个字节，密钥扩展用的
inline uint32_t sboxWordApply32(uint32_t v) {
    return sbox3T24[(uint8_t)(v >> 24)] |
           sbox2T16[(uint8_t)(v >> 16)] |
           sbox1T8[(uint8_t)(v >> 8)] |
           sbox0T0[(uint8_t)v];
}

// 库加载时初始化 S 盒
struct Initializer {
    Initializer() {
        uint8_t base[256];
        genSbox(base);

        // 4 个 S 盒分别异或不同的 delta
        uint8_t sboxes[4][256];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 256; j++) {
                sboxes[i][j] = base[j] ^ SBOX_DELTAS[i];
            }
        }

        // 预先移位到对应字节位置
        for (int i = 0; i < 256; i++) {
            sbox0T0[i]  = (uint32_t)sboxes[0][i];
            sbox1T8[i]  = (uint32_t)sboxes[1][i] << 8;
            sbox2T16[i] = (uint32_t)sboxes[2][i] << 16;
            sbox3T24[i] = (uint32_t)sboxes[3][i] << 24;
        }
    }
};
static Initializer g_init;

// 密钥扩展，输出 144 个 uint32（当 72 个 uint64 用）
void expandKey(BlockCipher* c, const uint8_t* key) {
    int totalWords = (c->rounds + 2) * 8;  // 144 个 uint32
    // uint64 数组当 uint32 用，小端平台布局一样
    uint32_t* w = reinterpret_cast<uint32_t*>(c->roundKeys);

    for (int i = 0; i < 8; i++) {
        w[i] = readU32LE(key + i * 4);
    }

    for (int i = 8; i < totalWords; i++) {
        uint32_t temp = w[i - 1];
        if (i % 8 == 0) {
            temp = rotl32(temp, 8);
            temp = sboxWordApply32(temp);
        } else if (i % 4 == 0) {
            temp = sboxWordApply32(temp);
            temp = rotl32(temp, 16);
            temp += rotl32(temp, 7);
        } else {
            temp = xmxFinalize(temp);
        }
        w[i] = w[i - 8] ^ temp;
    }

    // 轮常数预合并进轮密钥，运行时少一次异或
    for (int r = 0; r < c->rounds; r++) {
        int base = (r + 1) * 8;  // 跳过输入白化 w[0..7]
        for (int k = 0; k < 8; k++) {
            w[base + k] ^= kRoundConstants[r][k];
        }
    }
}

// 16 轮加密，全在寄存器里跑
__forceinline void encryptBlockInline(uint32_t& s0, uint32_t& s1, uint32_t& s2, uint32_t& s3,
                                      uint32_t& s4, uint32_t& s5, uint32_t& s6, uint32_t& s7,
                                      int rounds, const uint64_t* keys) {
    // 输入白化
    s0 ^= (uint32_t)keys[0]; s1 ^= (uint32_t)(keys[0] >> 32);
    s2 ^= (uint32_t)keys[1]; s3 ^= (uint32_t)(keys[1] >> 32);
    s4 ^= (uint32_t)keys[2]; s5 ^= (uint32_t)(keys[2] >> 32);
    s6 ^= (uint32_t)keys[3]; s7 ^= (uint32_t)(keys[3] >> 32);

    int ki = 4;
    for (int r = 0; r < rounds; r++) {
        // 偶数轮用 G_Mix，奇数轮用 H_Mix
        if ((r & 1) == 0) {
            // G_Mix
            s0 += rotr32(s2, GM_R5); s1 += rotr32(s3, GM_R5);
            s6 ^= s0; s7 ^= s1;
            s6 = rotr32(s6, GM_R1); s7 = rotr32(s7, GM_R1);
            s4 += rotr32(s6, GM_R6); s5 += rotr32(s7, GM_R6);
            s2 ^= s4; s3 ^= s5;
            s2 = rotr32(s2, GM_R2); s3 = rotr32(s3, GM_R2);
            // 折回 XOR
            s0 ^= rotr32(s4, GM_R9); s1 ^= rotr32(s5, GM_R9);
            s0 += rotr32(s2, GM_R7); s1 += rotr32(s3, GM_R7);
            s6 ^= s0; s7 ^= s1;
            s6 = rotr32(s6, GM_R3); s7 = rotr32(s7, GM_R3);
            s4 += rotr32(s6, GM_R8); s5 += rotr32(s7, GM_R8);
            s2 ^= s4; s3 ^= s5;
            s2 = rotr32(s2, GM_R4); s3 = rotr32(s3, GM_R4);
        } else {
            // H_Mix，旋转量不同
            s0 += rotr32(s2, HM_R5); s1 += rotr32(s3, HM_R5);
            s6 ^= s0; s7 ^= s1;
            s6 = rotr32(s6, HM_R1); s7 = rotr32(s7, HM_R1);
            s4 += rotr32(s6, HM_R6); s5 += rotr32(s7, HM_R6);
            s2 ^= s4; s3 ^= s5;
            s2 = rotr32(s2, HM_R2); s3 = rotr32(s3, HM_R2);
            s0 ^= rotr32(s4, HM_R9); s1 ^= rotr32(s5, HM_R9);
            s0 += rotr32(s2, HM_R7); s1 += rotr32(s3, HM_R7);
            s6 ^= s0; s7 ^= s1;
            s6 = rotr32(s6, HM_R3); s7 = rotr32(s7, HM_R3);
            s4 += rotr32(s6, HM_R8); s5 += rotr32(s7, HM_R8);
            s2 ^= s4; s3 ^= s5;
            s2 = rotr32(s2, HM_R4); s3 = rotr32(s3, HM_R4);
        }

        // ShiftRows：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
        { uint32_t t = s0; s0 = s2; s2 = s7; s7 = s1; s1 = s5; s5 = s4; s4 = s3; s3 = s6; s6 = t; }

        // AddRoundKey
        s0 ^= (uint32_t)keys[ki]; s1 ^= (uint32_t)(keys[ki] >> 32);
        s2 ^= (uint32_t)keys[ki + 1]; s3 ^= (uint32_t)(keys[ki + 1] >> 32);
        s4 ^= (uint32_t)keys[ki + 2]; s5 ^= (uint32_t)(keys[ki + 2] >> 32);
        s6 ^= (uint32_t)keys[ki + 3]; s7 ^= (uint32_t)(keys[ki + 3] >> 32);
        ki += 4;
    }

    // 输出白化
    s0 ^= (uint32_t)keys[ki]; s1 ^= (uint32_t)(keys[ki] >> 32);
    s2 ^= (uint32_t)keys[ki + 1]; s3 ^= (uint32_t)(keys[ki + 1] >> 32);
    s4 ^= (uint32_t)keys[ki + 2]; s5 ^= (uint32_t)(keys[ki + 2] >> 32);
    s6 ^= (uint32_t)keys[ki + 3]; s7 ^= (uint32_t)(keys[ki + 3] >> 32);
}

// 16 轮解密
__forceinline void decryptBlockInline(uint32_t& s0, uint32_t& s1, uint32_t& s2, uint32_t& s3,
                                      uint32_t& s4, uint32_t& s5, uint32_t& s6, uint32_t& s7,
                                      int rounds, const uint64_t* keys) {
    // 先解除输出白化
    int ki = 4 + rounds * 4;
    s0 ^= (uint32_t)keys[ki]; s1 ^= (uint32_t)(keys[ki] >> 32);
    s2 ^= (uint32_t)keys[ki + 1]; s3 ^= (uint32_t)(keys[ki + 1] >> 32);
    s4 ^= (uint32_t)keys[ki + 2]; s5 ^= (uint32_t)(keys[ki + 2] >> 32);
    s6 ^= (uint32_t)keys[ki + 3]; s7 ^= (uint32_t)(keys[ki + 3] >> 32);

    for (int r = rounds - 1; r >= 0; r--) {
        ki = 4 + r * 4;
        // 逆 AddRoundKey
        s0 ^= (uint32_t)keys[ki]; s1 ^= (uint32_t)(keys[ki] >> 32);
        s2 ^= (uint32_t)keys[ki + 1]; s3 ^= (uint32_t)(keys[ki + 1] >> 32);
        s4 ^= (uint32_t)keys[ki + 2]; s5 ^= (uint32_t)(keys[ki + 2] >> 32);
        s6 ^= (uint32_t)keys[ki + 3]; s7 ^= (uint32_t)(keys[ki + 3] >> 32);

        // 逆 ShiftRows：8-循环置换的逆
        { uint32_t t = s0; s0 = s6; s6 = s3; s3 = s4; s4 = s5; s5 = s1; s1 = s7; s7 = s2; s2 = t; }

        // 逆 ColumnMix
        if ((r & 1) == 0) {
            // 逆 G_Mix
            s2 = rotl32(s2, GM_R4) ^ s4; s3 = rotl32(s3, GM_R4) ^ s5;
            s4 -= rotr32(s6, GM_R8); s5 -= rotr32(s7, GM_R8);
            s6 = rotl32(s6, GM_R3) ^ s0; s7 = rotl32(s7, GM_R3) ^ s1;
            s0 -= rotr32(s2, GM_R7); s1 -= rotr32(s3, GM_R7);
            s0 ^= rotr32(s4, GM_R9); s1 ^= rotr32(s5, GM_R9);
            s2 = rotl32(s2, GM_R2) ^ s4; s3 = rotl32(s3, GM_R2) ^ s5;
            s4 -= rotr32(s6, GM_R6); s5 -= rotr32(s7, GM_R6);
            s6 = rotl32(s6, GM_R1) ^ s0; s7 = rotl32(s7, GM_R1) ^ s1;
            s0 -= rotr32(s2, GM_R5); s1 -= rotr32(s3, GM_R5);
        } else {
            // 逆 H_Mix
            s2 = rotl32(s2, HM_R4) ^ s4; s3 = rotl32(s3, HM_R4) ^ s5;
            s4 -= rotr32(s6, HM_R8); s5 -= rotr32(s7, HM_R8);
            s6 = rotl32(s6, HM_R3) ^ s0; s7 = rotl32(s7, HM_R3) ^ s1;
            s0 -= rotr32(s2, HM_R7); s1 -= rotr32(s3, HM_R7);
            s0 ^= rotr32(s4, HM_R9); s1 ^= rotr32(s5, HM_R9);
            s2 = rotl32(s2, HM_R2) ^ s4; s3 = rotl32(s3, HM_R2) ^ s5;
            s4 -= rotr32(s6, HM_R6); s5 -= rotr32(s7, HM_R6);
            s6 = rotl32(s6, HM_R1) ^ s0; s7 = rotl32(s7, HM_R1) ^ s1;
            s0 -= rotr32(s2, HM_R5); s1 -= rotr32(s3, HM_R5);
        }
    }

    // 解除输入白化
    s0 ^= (uint32_t)keys[0]; s1 ^= (uint32_t)(keys[0] >> 32);
    s2 ^= (uint32_t)keys[1]; s3 ^= (uint32_t)(keys[1] >> 32);
    s4 ^= (uint32_t)keys[2]; s5 ^= (uint32_t)(keys[2] >> 32);
    s6 ^= (uint32_t)keys[3]; s7 ^= (uint32_t)(keys[3] >> 32);
}

// ARX 压缩函数

// ARX 一轮变换，IsG=true 走 G_Mix，false 走 H_Mix
template<bool IsG>
inline void arxRoundT(uint32_t& s0, uint32_t& s1, uint32_t& s2, uint32_t& s3,
                      uint32_t& s4, uint32_t& s5, uint32_t& s6, uint32_t& s7) {
    if constexpr (IsG) {
        s0 += rotr32(s2, GM_R5); s1 += rotr32(s3, GM_R5);
        s6 ^= s0; s7 ^= s1;
        s6 = rotr32(s6, GM_R1); s7 = rotr32(s7, GM_R1);
        s4 += rotr32(s6, GM_R6); s5 += rotr32(s7, GM_R6);
        s2 ^= s4; s3 ^= s5;
        s2 = rotr32(s2, GM_R2); s3 = rotr32(s3, GM_R2);
        s0 ^= rotr32(s4, GM_R9); s1 ^= rotr32(s5, GM_R9);
        s0 += rotr32(s2, GM_R7); s1 += rotr32(s3, GM_R7);
        s6 ^= s0; s7 ^= s1;
        s6 = rotr32(s6, GM_R3); s7 = rotr32(s7, GM_R3);
        s4 += rotr32(s6, GM_R8); s5 += rotr32(s7, GM_R8);
        s2 ^= s4; s3 ^= s5;
        s2 = rotr32(s2, GM_R4); s3 = rotr32(s3, GM_R4);
    } else {
        s0 += rotr32(s2, HM_R5); s1 += rotr32(s3, HM_R5);
        s6 ^= s0; s7 ^= s1;
        s6 = rotr32(s6, HM_R1); s7 = rotr32(s7, HM_R1);
        s4 += rotr32(s6, HM_R6); s5 += rotr32(s7, HM_R6);
        s2 ^= s4; s3 ^= s5;
        s2 = rotr32(s2, HM_R2); s3 = rotr32(s3, HM_R2);
        s0 ^= rotr32(s4, HM_R9); s1 ^= rotr32(s5, HM_R9);
        s0 += rotr32(s2, HM_R7); s1 += rotr32(s3, HM_R7);
        s6 ^= s0; s7 ^= s1;
        s6 = rotr32(s6, HM_R3); s7 = rotr32(s7, HM_R3);
        s4 += rotr32(s6, HM_R8); s5 += rotr32(s7, HM_R8);
        s2 ^= s4; s3 ^= s5;
        s2 = rotr32(s2, HM_R4); s3 = rotr32(s3, HM_R4);
    }
    // ShiftRows：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
    { uint32_t t = s0; s0 = s2; s2 = s7; s7 = s1; s1 = s5; s5 = s4; s4 = s3; s3 = s6; s6 = t; }
}

// 压缩一个 32 字节消息块，16 轮 ARX，Davies-Meyer 前馈；last=true 时注入封口常数
void arxCompress(uint32_t state[8], const uint8_t* block, bool last) {
    uint32_t m[8];
    for (int i = 0; i < 8; i++) m[i] = readU32LE(block + i * 4);

    // 保存旧状态，前馈用
    uint32_t o0 = state[0], o1 = state[1], o2 = state[2], o3 = state[3];
    uint32_t o4 = state[4], o5 = state[5], o6 = state[6], o7 = state[7];

    // 封口常数只加在吸收阶段，不参与末尾前馈
    uint32_t f0 = last ? kFinalConst[0] : 0u, f1 = last ? kFinalConst[1] : 0u;
    uint32_t f2 = last ? kFinalConst[2] : 0u, f3 = last ? kFinalConst[3] : 0u;
    uint32_t f4 = last ? kFinalConst[4] : 0u, f5 = last ? kFinalConst[5] : 0u;
    uint32_t f6 = last ? kFinalConst[6] : 0u, f7 = last ? kFinalConst[7] : 0u;

    // 消息模加注入（末块含封口常数）
    uint32_t s0 = o0 + m[0] + f0, s1 = o1 + m[1] + f1, s2 = o2 + m[2] + f2, s3 = o3 + m[3] + f3;
    uint32_t s4 = o4 + m[4] + f4, s5 = o5 + m[5] + f5, s6 = o6 + m[6] + f6, s7 = o7 + m[7] + f7;

    arxRoundT<true>(s0, s1, s2, s3, s4, s5, s6, s7);   // r0 G
    s0 ^= kRoundConstants[0][0]; s1 ^= kRoundConstants[0][1]; s2 ^= kRoundConstants[0][2]; s3 ^= kRoundConstants[0][3];
    s4 ^= kRoundConstants[0][4]; s5 ^= kRoundConstants[0][5]; s6 ^= kRoundConstants[0][6]; s7 ^= kRoundConstants[0][7];
    arxRoundT<false>(s0, s1, s2, s3, s4, s5, s6, s7);  // r1 H
    s0 ^= kRoundConstants[1][0]; s1 ^= kRoundConstants[1][1]; s2 ^= kRoundConstants[1][2]; s3 ^= kRoundConstants[1][3];
    s4 ^= kRoundConstants[1][4]; s5 ^= kRoundConstants[1][5]; s6 ^= kRoundConstants[1][6]; s7 ^= kRoundConstants[1][7];
    arxRoundT<true>(s0, s1, s2, s3, s4, s5, s6, s7);   // r2 G
    s0 ^= kRoundConstants[2][0]; s1 ^= kRoundConstants[2][1]; s2 ^= kRoundConstants[2][2]; s3 ^= kRoundConstants[2][3];
    s4 ^= kRoundConstants[2][4]; s5 ^= kRoundConstants[2][5]; s6 ^= kRoundConstants[2][6]; s7 ^= kRoundConstants[2][7];
    arxRoundT<false>(s0, s1, s2, s3, s4, s5, s6, s7);  // r3 H
    s0 ^= kRoundConstants[3][0]; s1 ^= kRoundConstants[3][1]; s2 ^= kRoundConstants[3][2]; s3 ^= kRoundConstants[3][3];
    s4 ^= kRoundConstants[3][4]; s5 ^= kRoundConstants[3][5]; s6 ^= kRoundConstants[3][6]; s7 ^= kRoundConstants[3][7];
    arxRoundT<true>(s0, s1, s2, s3, s4, s5, s6, s7);   // r4 G
    s0 ^= kRoundConstants[4][0]; s1 ^= kRoundConstants[4][1]; s2 ^= kRoundConstants[4][2]; s3 ^= kRoundConstants[4][3];
    s4 ^= kRoundConstants[4][4]; s5 ^= kRoundConstants[4][5]; s6 ^= kRoundConstants[4][6]; s7 ^= kRoundConstants[4][7];
    arxRoundT<false>(s0, s1, s2, s3, s4, s5, s6, s7);  // r5 H
    s0 ^= kRoundConstants[5][0]; s1 ^= kRoundConstants[5][1]; s2 ^= kRoundConstants[5][2]; s3 ^= kRoundConstants[5][3];
    s4 ^= kRoundConstants[5][4]; s5 ^= kRoundConstants[5][5]; s6 ^= kRoundConstants[5][6]; s7 ^= kRoundConstants[5][7];
    arxRoundT<true>(s0, s1, s2, s3, s4, s5, s6, s7);   // r6 G
    s0 ^= kRoundConstants[6][0]; s1 ^= kRoundConstants[6][1]; s2 ^= kRoundConstants[6][2]; s3 ^= kRoundConstants[6][3];
    s4 ^= kRoundConstants[6][4]; s5 ^= kRoundConstants[6][5]; s6 ^= kRoundConstants[6][6]; s7 ^= kRoundConstants[6][7];
    arxRoundT<false>(s0, s1, s2, s3, s4, s5, s6, s7);  // r7 H
    s0 ^= kRoundConstants[7][0]; s1 ^= kRoundConstants[7][1]; s2 ^= kRoundConstants[7][2]; s3 ^= kRoundConstants[7][3];
    s4 ^= kRoundConstants[7][4]; s5 ^= kRoundConstants[7][5]; s6 ^= kRoundConstants[7][6]; s7 ^= kRoundConstants[7][7];
    arxRoundT<true>(s0, s1, s2, s3, s4, s5, s6, s7);   // r8 G
    s0 ^= kRoundConstants[8][0]; s1 ^= kRoundConstants[8][1]; s2 ^= kRoundConstants[8][2]; s3 ^= kRoundConstants[8][3];
    s4 ^= kRoundConstants[8][4]; s5 ^= kRoundConstants[8][5]; s6 ^= kRoundConstants[8][6]; s7 ^= kRoundConstants[8][7];
    arxRoundT<false>(s0, s1, s2, s3, s4, s5, s6, s7);  // r9 H
    s0 ^= kRoundConstants[9][0]; s1 ^= kRoundConstants[9][1]; s2 ^= kRoundConstants[9][2]; s3 ^= kRoundConstants[9][3];
    s4 ^= kRoundConstants[9][4]; s5 ^= kRoundConstants[9][5]; s6 ^= kRoundConstants[9][6]; s7 ^= kRoundConstants[9][7];
    arxRoundT<true>(s0, s1, s2, s3, s4, s5, s6, s7);   // r10 G
    s0 ^= kRoundConstants[10][0]; s1 ^= kRoundConstants[10][1]; s2 ^= kRoundConstants[10][2]; s3 ^= kRoundConstants[10][3];
    s4 ^= kRoundConstants[10][4]; s5 ^= kRoundConstants[10][5]; s6 ^= kRoundConstants[10][6]; s7 ^= kRoundConstants[10][7];
    arxRoundT<false>(s0, s1, s2, s3, s4, s5, s6, s7); // r11 H
    s0 ^= kRoundConstants[11][0]; s1 ^= kRoundConstants[11][1]; s2 ^= kRoundConstants[11][2]; s3 ^= kRoundConstants[11][3];
    s4 ^= kRoundConstants[11][4]; s5 ^= kRoundConstants[11][5]; s6 ^= kRoundConstants[11][6]; s7 ^= kRoundConstants[11][7];
    arxRoundT<true>(s0, s1, s2, s3, s4, s5, s6, s7);   // r12 G
    s0 ^= kRoundConstants[12][0]; s1 ^= kRoundConstants[12][1]; s2 ^= kRoundConstants[12][2]; s3 ^= kRoundConstants[12][3];
    s4 ^= kRoundConstants[12][4]; s5 ^= kRoundConstants[12][5]; s6 ^= kRoundConstants[12][6]; s7 ^= kRoundConstants[12][7];
    arxRoundT<false>(s0, s1, s2, s3, s4, s5, s6, s7);  // r13 H
    s0 ^= kRoundConstants[13][0]; s1 ^= kRoundConstants[13][1]; s2 ^= kRoundConstants[13][2]; s3 ^= kRoundConstants[13][3];
    s4 ^= kRoundConstants[13][4]; s5 ^= kRoundConstants[13][5]; s6 ^= kRoundConstants[13][6]; s7 ^= kRoundConstants[13][7];
    arxRoundT<true>(s0, s1, s2, s3, s4, s5, s6, s7);   // r14 G
    s0 ^= kRoundConstants[14][0]; s1 ^= kRoundConstants[14][1]; s2 ^= kRoundConstants[14][2]; s3 ^= kRoundConstants[14][3];
    s4 ^= kRoundConstants[14][4]; s5 ^= kRoundConstants[14][5]; s6 ^= kRoundConstants[14][6]; s7 ^= kRoundConstants[14][7];
    arxRoundT<false>(s0, s1, s2, s3, s4, s5, s6, s7);  // r15 H
    s0 ^= kRoundConstants[15][0]; s1 ^= kRoundConstants[15][1]; s2 ^= kRoundConstants[15][2]; s3 ^= kRoundConstants[15][3];
    s4 ^= kRoundConstants[15][4]; s5 ^= kRoundConstants[15][5]; s6 ^= kRoundConstants[15][6]; s7 ^= kRoundConstants[15][7];

    // Davies-Meyer 前馈
    state[0] = (s0 ^ o0) + m[0]; state[1] = (s1 ^ o1) + m[1];
    state[2] = (s2 ^ o2) + m[2]; state[3] = (s3 ^ o3) + m[3];
    state[4] = (s4 ^ o4) + m[4]; state[5] = (s5 ^ o5) + m[5];
    state[6] = (s6 ^ o6) + m[6]; state[7] = (s7 ^ o7) + m[7];
}

// SHA-256 初始 IV，素数平方根小数部分
const uint32_t ARX_INIT_STATE[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
};

// 哈希函数

struct Hash {
    uint32_t state[8];
    uint8_t buf[32];
    int bufLen;
    uint64_t total;
};

void hashInit(Hash* h) {
    std::memcpy(h->state, ARX_INIT_STATE, sizeof(ARX_INIT_STATE));
    h->bufLen = 0;
    h->total = 0;
}

inline void arxCompressDispatch(uint32_t state[8], const uint8_t* block, bool last) {
    arxCompress(state, block, last);
}

void hashWrite(Hash* h, const uint8_t* data, int len) {
    // len 为负会变成巨大的 memcpy 拷贝量，先挡掉
    if (len <= 0) return;
    h->total += (uint64_t)len;
    // 先把缓冲区填满
    if (h->bufLen > 0) {
        int copied = 32 - h->bufLen;
        if (copied > len) copied = len;
        std::memcpy(h->buf + h->bufLen, data, copied);
        h->bufLen += copied;
        data += copied;
        len -= copied;
        if (h->bufLen < 32) return;
        arxCompressDispatch(h->state, h->buf, false);
        h->bufLen = 0;
    }
    // 整块批量压缩
    while (len >= 32) {
        arxCompressDispatch(h->state, data, false);
        data += 32;
        len -= 32;
    }
    // 尾部存到缓冲区
    if (len > 0) {
        std::memcpy(h->buf, data, len);
        h->bufLen = len;
    }
}

// 算最终摘要，注意长度域是小端，跟标准 SHA-256 不一样
void hashSum(Hash* h, uint8_t out[32]) {
    uint32_t state[8];
    std::memcpy(state, h->state, sizeof(state));

    int dataLen = h->bufLen;
    int padNeed = 32 - dataLen % 32;
    if (padNeed < 9) padNeed += 32;  // 至少要放 1 字节 0x80 + 8 字节长度

    uint8_t padded[64] = {0};
    std::memcpy(padded, h->buf, dataLen);
    padded[dataLen] = 0x80;
    // 比特长度，小端
    writeU64LE(padded + dataLen + padNeed - 8, h->total * 8);

    int totalBlockBytes = dataLen + padNeed;
    for (int i = 0; i < totalBlockBytes; i += 32) {
        // 最后一个填充块带封口标志
        bool last = (i + 32 >= totalBlockBytes);
        arxCompressDispatch(state, padded + i, last);
    }

    for (int i = 0; i < 8; i++) {
        writeU32LE(out + i * 4, state[i]);
    }
}

// HMAC

// int64 长度分块写入，单笔数据超过 int 范围时按 1GB 分块喂给 hashWrite
void hashWrite64(Hash* h, const uint8_t* data, int64_t len) {
    while (len > 0) {
        int chunk = (len > 0x40000000) ? 0x40000000 : (int)len;
        hashWrite(h, data, chunk);
        data += chunk;
        len -= chunk;
    }
}

// HMAC 用两把密钥哈希：inner 异或 ipad，outer 异或 opad
void hmacCompute(const uint8_t* key, int keyLen, const uint8_t* data, int64_t dataLen, uint8_t out[32]) {
    uint8_t k[32] = {0};
    if (keyLen > 32) {
        // 密钥超过 32 字节就先哈希
        Hash kh;
        hashInit(&kh);
        hashWrite(&kh, key, keyLen);
        hashSum(&kh, k);
    } else {
        std::memcpy(k, key, keyLen);
    }

    uint8_t ipad[32], opad[32];
    for (int i = 0; i < 32; i++) {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5C;
    }

    // inner: ipad 异或到初始状态，再写 data
    Hash innerH;
    hashInit(&innerH);
    for (int i = 0; i < 8; i++) innerH.state[i] ^= readU32LE(ipad + i * 4);
    hashWrite64(&innerH, data, dataLen);
    uint8_t innerHash[32];
    hashSum(&innerH, innerHash);

    // outer: opad 异或到初始状态，再写 innerHash
    Hash outerH;
    hashInit(&outerH);
    for (int i = 0; i < 8; i++) outerH.state[i] ^= readU32LE(opad + i * 4);
    hashWrite(&outerH, innerHash, 32);
    hashSum(&outerH, out);
}

// 4 条链并行 HMAC：消息各 32 字节、密钥相同
void hmacBlock4x(const uint8_t* key, int keyLen, const uint8_t* msg[4], uint8_t out[4][32]) {
    uint8_t k[32] = {0};
    if (keyLen > 32) {
        Hash kh;
        hashInit(&kh);
        hashWrite(&kh, key, keyLen);
        hashSum(&kh, k);
    } else {
        std::memcpy(k, key, keyLen);
    }

    uint8_t ipad[32], opad[32];
    for (int i = 0; i < 32; i++) {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5C;
    }

    // stT[k][i] = 链 i 的 s_k，保持状态字序
    uint32_t stT[8][4];
    // inner 初始状态（4 链相同）
    for (int j = 0; j < 8; j++) {
        uint32_t iv = ARX_INIT_STATE[j] ^ readU32LE(ipad + j * 4);
        for (int i = 0; i < 4; i++) stT[j][i] = iv;
    }
    // 压缩 32 字节消息（4 链并行）
    arxCompress4xSSE2(stT, msg, false);

    // 补丁块：dataLen=32，padNeed=32，内容 = 0x80 | 0x00*23 | le64(32*8)
    uint8_t pad[32] = {0};
    pad[0] = 0x80;
    writeU64LE(pad + 24, 32 * 8);
    const uint8_t* pads[4] = { pad, pad, pad, pad };
    arxCompress4xSSE2(stT, pads, true);

    // innerHash：状态字序 → 链序
    uint8_t innerHash[4][32];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) writeU32LE(innerHash[i] + j * 4, stT[j][i]);
    }

    // outer 初始状态
    for (int j = 0; j < 8; j++) {
        uint32_t ov = ARX_INIT_STATE[j] ^ readU32LE(opad + j * 4);
        for (int i = 0; i < 4; i++) stT[j][i] = ov;
    }
    // 压缩 innerHash（4 链并行）+ 补丁块（total 同为 32）
    const uint8_t* ih[4] = { innerHash[0], innerHash[1], innerHash[2], innerHash[3] };
    arxCompress4xSSE2(stT, ih, false);
    arxCompress4xSSE2(stT, pads, true);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) writeU32LE(out[i] + j * 4, stT[j][i]);
    }
}

// HMAC 增量更新上下文
struct HMACCtx {
    Hash inner;
    uint8_t opad[32];
};

void hmacInit(HMACCtx* ctx, const uint8_t* key, int keyLen) {
    uint8_t k[32] = {0};
    if (keyLen > 32) {
        Hash kh;
        hashInit(&kh);
        hashWrite(&kh, key, keyLen);
        hashSum(&kh, k);
    } else {
        std::memcpy(k, key, keyLen);
    }

    uint8_t ipad[32];
    for (int i = 0; i < 32; i++) {
        ipad[i] = k[i] ^ 0x36;
        ctx->opad[i] = k[i] ^ 0x5C;
    }

    hashInit(&ctx->inner);
    for (int i = 0; i < 8; i++) ctx->inner.state[i] ^= readU32LE(ipad + i * 4);
}

inline void hmacWrite(HMACCtx* ctx, const uint8_t* data, int len) {
    hashWrite(&ctx->inner, data, len);
}

void hmacFinal(HMACCtx* ctx, uint8_t out[32]) {
    uint8_t innerHash[32];
    hashSum(&ctx->inner, innerHash);

    Hash outerH;
    hashInit(&outerH);
    for (int i = 0; i < 8; i++) outerH.state[i] ^= readU32LE(ctx->opad + i * 4);
    hashWrite(&outerH, innerHash, 32);
    hashSum(&outerH, out);
}

// HKDF

// HKDF 提取: PRK = HMAC(salt, ikm)
void hkdfExtract(const uint8_t* ikm, int ikmLen, const uint8_t* salt, int saltLen, uint8_t prk[32]) {
    hmacCompute(salt, saltLen, ikm, ikmLen, prk);
}

// HKDF 扩展，输出指定长度
bool hkdfExpand(const uint8_t prk[32], const uint8_t* info, int infoLen, int length, uint8_t* out) {
    if (length <= 0 || length > 255 * 32) return false;
    // info 长度接近 int 上限时下面的 32 + infoLen + 1 会溢出成负数，提前拒绝
    if (infoLen < 0 || infoLen > (std::numeric_limits<int>::max)() - 33) return false;

    // 堆分配，避免 8KB 栈占用；按块对齐分配，最后一轮写 32 字节不会越界
    int numBlocks = (length + 31) / 32;
    std::unique_ptr<uint8_t[]> result(new uint8_t[numBlocks * 32]);
    uint8_t prev[32];
    int prevLen = 0;
    int offset = 0;
    uint8_t counter = 1;

    // innerBuf 要放 prev(32) + info + counter(1)，info 长度不定，动态分配防栈溢出
    int innerCap = 32 + infoLen + 1;
    std::unique_ptr<uint8_t[]> innerBuf(new uint8_t[innerCap]);

    while (offset < length) {
        // T(i) = HMAC(prk, T(i-1) || info || counter)，直接复用 hmacCompute
        int innerLen = 0;
        if (prevLen > 0) {
            std::memcpy(innerBuf.get(), prev, prevLen);
            innerLen = prevLen;
        }
        std::memcpy(innerBuf.get() + innerLen, info, infoLen);
        innerLen += infoLen;
        innerBuf[innerLen++] = counter;

        hmacCompute(prk, 32, innerBuf.get(), innerLen, prev);
        prevLen = 32;

        std::memcpy(result.get() + offset, prev, 32);
        offset += 32;
        counter++;
    }

    std::memcpy(out, result.get(), length);
    return true;
}

// CTR 模式

// 根据数据量算 worker 数量
inline int getOptimalWorkerCount(int dataSize) {
    unsigned cpuCount = std::thread::hardware_concurrency();
    if (cpuCount == 0) cpuCount = 1;
    if (cpuCount <= 1) return 1;
    int blocks = dataSize / 32;
    if (blocks < (int)cpuCount * MIN_BLOCKS_PER_WORKER) {
        int w = blocks / MIN_BLOCKS_PER_WORKER;
        if (w < 1) w = 1;
        return w;
    }
    if ((int)cpuCount > MAX_WORKERS) return MAX_WORKERS;
    return (int)cpuCount;
}

// 常驻并行池：后台线程第一次用到时才创建，最多 MAX_WORKERS 个，析构时唤醒并收掉
class ParallelPool {
public:
    // 进程内唯一实例，函数内静态对象保证只构造一次
    static ParallelPool& instance() {
        static ParallelPool pool;
        return pool;
    }

    // 把作业切成 taskCount 个任务并行跑完再返回；fn 会被多个线程同时调用，各任务不能有依赖
    void run(int taskCount, const std::function<void(int)>& fn) {
        if (taskCount <= 0) return;
        // 单任务，或者后台线程一个都没起来，当前线程自己跑完
        if (taskCount == 1 || !ensureWorkers()) {
            for (int i = 0; i < taskCount; i++) fn(i);
            return;
        }

        std::shared_ptr<Job> job(new Job());
        job->fn = &fn;
        job->taskCount = taskCount;

        {
            std::lock_guard<std::mutex> lk(m_mtx);
            m_jobs.push_back(job);
        }
        m_cvTask.notify_all();

        // 调用线程先抢一批干，剩下的留给后台线程
        drainRest(job.get());

        // 等后台线程把剩下的任务清空
        std::unique_lock<std::mutex> lk(m_mtx);
        m_cvDone.wait(lk, [&job] { return job->done.load(std::memory_order_acquire) >= job->taskCount; });
        // 从队列里摘掉自己的作业，不影响后面提交的作业
        for (size_t i = 0; i < m_jobs.size(); i++) {
            if (m_jobs[i] == job) {
                m_jobs.erase(m_jobs.begin() + i);
                break;
            }
        }
    }

private:
    // 一个作业：任务下标 0 到 taskCount-1，哪个线程抢到哪个线程干
    struct Job {
        const std::function<void(int)>* fn = nullptr;
        int taskCount = 0;
        std::atomic<int> next{0};   // 下一个还没人认领的任务下标
        std::atomic<int> done{0};   // 已经干完的任务数
    };

    ParallelPool() = default;

    // 退出流程：叫醒所有后台线程让它们自己结束，再挨个 join
    ~ParallelPool() {
        {
            std::lock_guard<std::mutex> lk(m_mtx);
            m_quit = true;
        }
        m_cvTask.notify_all();
        for (auto& t : m_workers) {
            if (t.joinable()) t.join();
        }
    }

    ParallelPool(const ParallelPool&) = delete;
    ParallelPool& operator=(const ParallelPool&) = delete;

    // 第一次用的时候把后台线程拉起来；返回 false 表示一个都没起来，调用方退回单线程
    bool ensureWorkers() {
        std::lock_guard<std::mutex> lk(m_mtx);
        if (m_started) return !m_workers.empty();
        m_started = true;

        unsigned hw = std::thread::hardware_concurrency();
        if (hw < 2) hw = 2;
        // 调用线程自己也当一名执行者，所以后台线程最多 MAX_WORKERS 个
        unsigned n = hw - 1;
        if (n > (unsigned)MAX_WORKERS) n = (unsigned)MAX_WORKERS;
        try {
            for (unsigned i = 0; i < n; i++) {
                m_workers.emplace_back([this] { workerLoop(); });
            }
        } catch (...) {
            // 线程创建失败就少起几个，剩下的任务调用线程自己扛
        }
        return !m_workers.empty();
    }

    // 干完任务记数，最后一个干完的持锁叫醒等待方（锁外唤醒有漏醒风险）
    void runTask(Job* job, int idx) {
        (*job->fn)(idx);
        if (job->done.fetch_add(1, std::memory_order_acq_rel) + 1 >= job->taskCount) {
            std::lock_guard<std::mutex> lk(m_mtx);
            m_cvDone.notify_all();
        }
    }

    // 把作业里还没人认领的任务一口气抢完
    void drainRest(Job* job) {
        int idx;
        while ((idx = job->next.fetch_add(1, std::memory_order_relaxed)) < job->taskCount) {
            runTask(job, idx);
        }
    }

    // 后台线程主循环：等活、认领任务、干完接着等
    void workerLoop() {
        std::unique_lock<std::mutex> lk(m_mtx);
        while (true) {
            m_cvTask.wait(lk, [this] {
                if (m_quit) return true;
                // 队列里还有没人认领的任务才算有活
                for (const auto& j : m_jobs) {
                    if (j->next.load(std::memory_order_relaxed) < j->taskCount) return true;
                }
                return false;
            });
            if (m_quit) break;

            // 持锁认领一个任务，被叫醒的线程就一定有活干
            std::shared_ptr<Job> job;
            int idx = -1;
            for (auto& j : m_jobs) {
                int cur = j->next.fetch_add(1, std::memory_order_relaxed);
                if (cur < j->taskCount) {
                    job = j;
                    idx = cur;
                    break;
                }
            }
            lk.unlock();
            if (job) {
                runTask(job.get(), idx);
                // 顺手把这个作业里剩下的任务也抢走
                drainRest(job.get());
            }
            lk.lock();
        }
    }

    std::mutex m_mtx;
    std::condition_variable m_cvTask;   // 后台线程在这里等新活
    std::condition_variable m_cvDone;   // 调用线程在这里等作业完工
    std::vector<std::shared_ptr<Job>> m_jobs;
    std::vector<std::thread> m_workers;
    bool m_started = false;
    bool m_quit = false;
};

// 并行工具的小包装：把 taskCount 个任务分给常驻线程池和当前线程
inline void parallelFor(int taskCount, const std::function<void(int)>& fn) {
    ParallelPool::instance().run(taskCount, fn);
}

// 多核并行 CTR：数据按块分给 worker，每个 worker 独立计数器初值，区间不重叠；roundKeys 跨线程只读
inline void ctrCryptParallel(const BlockCipher& c, const uint8_t* counter,
                             const uint8_t* in, uint8_t* out, int len) {
    int workers = getOptimalWorkerCount(len);
    if (workers <= 1) {
        encryptBlocksDispatched(c, counter, in, out, len);
        return;
    }

    int blocks = len / 32;
    int blocksPerWorker = (blocks + workers - 1) / workers;
    uint64_t baseCtr = readU64LE(counter + 24);

    parallelFor(workers, [&](int w) {
        int startBlock = w * blocksPerWorker;
        if (startBlock >= blocks) return;
        int endBlock = startBlock + blocksPerWorker;
        if (endBlock > blocks) endBlock = blocks;

        // 每个 worker 独立 counter 副本（前 24B nonce + 推进后的计数器）
        uint8_t workerCtr[32];
        std::memcpy(workerCtr, counter, 24);
        writeU64LE(workerCtr + 24, baseCtr + (uint64_t)startBlock);

        int startOff = startBlock * 32;
        int endOff = endBlock * 32;
        encryptBlocksDispatched(c, workerCtr, in + startOff, out + startOff, endOff - startOff);
    });
}

// CTR 批量处理，counter 前 24 字节 nonce，后 8 字节小端计数器
// 三段调度：>=64KB 多核并行；0<整块<64KB 走 SIMD 分发；尾部不足一块单块加密
void ctrCrypt(const BlockCipher& c, const uint8_t* counter, const uint8_t* in, uint8_t* out, int len) {
    if (len <= 0) return;

    int fullBytes = (len / 32) * 32;  // 整块部分

    if (fullBytes >= PARALLEL_THRESHOLD) {
        // 大数据量多核并行，worker 内部按 SIMD 分发
        ctrCryptParallel(c, counter, in, out, fullBytes);
    } else if (fullBytes > 0) {
        // 小数据量按 SIMD 分发（AVX2/SSE2/标量）
        encryptBlocksDispatched(c, counter, in, out, fullBytes);
    }

    // 尾部不足一块，单块加密 + 逐字节异或
    if (fullBytes < len) {
        uint8_t ctrBuf[32];
        std::memcpy(ctrBuf, counter, 24);
        // 计数器推进到第 fullBytes/32 块
        uint64_t tailCtr = readU64LE(counter + 24) + (uint64_t)(fullBytes / 32);
        writeU64LE(ctrBuf + 24, tailCtr);
        uint8_t ks[32];
        blockEncrypt(ctrBuf, ks, c);
        int remain = len - fullBytes;
        for (int i = 0; i < remain; i++) out[fullBytes + i] = in[fullBytes + i] ^ ks[i];
    }
}

// CBC 模式

// CBC 加密内存版，串行依赖每块依赖前一块密文，不能并行；prev 用寄存器，直接调 encryptBlockInline
bool cbcEncryptBuffer(const BlockCipher& c, const uint8_t* iv, const uint8_t* plaintext, int plaintextLen,
                      uint8_t* out, int* outLen) {
    int fullBlocks = plaintextLen / 32;
    int lastLen = plaintextLen - fullBlocks * 32;

    uint32_t p0 = readU32LE(iv), p1 = readU32LE(iv + 4);
    uint32_t p2 = readU32LE(iv + 8), p3 = readU32LE(iv + 12);
    uint32_t p4 = readU32LE(iv + 16), p5 = readU32LE(iv + 20);
    uint32_t p6 = readU32LE(iv + 24), p7 = readU32LE(iv + 28);

    // 完整块：P XOR prev -> E -> out
    for (int b = 0; b < fullBlocks; b++) {
        const uint8_t* pt = plaintext + b * 32;
        uint32_t s0 = readU32LE(pt)      ^ p0;
        uint32_t s1 = readU32LE(pt + 4)  ^ p1;
        uint32_t s2 = readU32LE(pt + 8)  ^ p2;
        uint32_t s3 = readU32LE(pt + 12) ^ p3;
        uint32_t s4 = readU32LE(pt + 16) ^ p4;
        uint32_t s5 = readU32LE(pt + 20) ^ p5;
        uint32_t s6 = readU32LE(pt + 24) ^ p6;
        uint32_t s7 = readU32LE(pt + 28) ^ p7;
        encryptBlockInline(s0, s1, s2, s3, s4, s5, s6, s7, c.rounds, c.roundKeys);
        uint8_t* ct = out + b * 32;
        writeU32LE(ct, s0);      p0 = s0;
        writeU32LE(ct + 4, s1);  p1 = s1;
        writeU32LE(ct + 8, s2);  p2 = s2;
        writeU32LE(ct + 12, s3); p3 = s3;
        writeU32LE(ct + 16, s4); p4 = s4;
        writeU32LE(ct + 20, s5); p5 = s5;
        writeU32LE(ct + 24, s6); p6 = s6;
        writeU32LE(ct + 28, s7); p7 = s7;
    }

    // 最后一块带 PKCS#7 填充，空数据也要补一个完整块
    int padLen = 32 - lastLen;
    uint8_t last[32];
    std::memcpy(last, plaintext + fullBlocks * 32, lastLen);
    for (int i = lastLen; i < 32; i++) last[i] = (uint8_t)padLen;
    uint32_t s0 = readU32LE(last)      ^ p0;
    uint32_t s1 = readU32LE(last + 4)  ^ p1;
    uint32_t s2 = readU32LE(last + 8)  ^ p2;
    uint32_t s3 = readU32LE(last + 12) ^ p3;
    uint32_t s4 = readU32LE(last + 16) ^ p4;
    uint32_t s5 = readU32LE(last + 20) ^ p5;
    uint32_t s6 = readU32LE(last + 24) ^ p6;
    uint32_t s7 = readU32LE(last + 28) ^ p7;
    encryptBlockInline(s0, s1, s2, s3, s4, s5, s6, s7, c.rounds, c.roundKeys);
    uint8_t* ct = out + fullBlocks * 32;
    writeU32LE(ct, s0);
    writeU32LE(ct + 4, s1);
    writeU32LE(ct + 8, s2);
    writeU32LE(ct + 12, s3);
    writeU32LE(ct + 16, s4);
    writeU32LE(ct + 20, s5);
    writeU32LE(ct + 24, s6);
    writeU32LE(ct + 28, s7);

    *outLen = (fullBlocks + 1) * 32;
    return true;
}

// CBC 解密内存版，常量时间校验填充；P[i] = D(C[i]) XOR C[i-1]，各 worker 独立可并行
bool cbcDecryptBuffer(const BlockCipher& c, const uint8_t* iv, const uint8_t* ciphertext, int ciphertextLen,
                      uint8_t* out, int* outLen) {
    if (ciphertextLen == 0 || ciphertextLen % 32 != 0) return false;
    int blocks = ciphertextLen / 32;

    if (ciphertextLen >= PARALLEL_THRESHOLD) {
        // getOptimalWorkerCount 返回值不会超过 MAX_WORKERS，无需再夹一遍
        int workers = getOptimalWorkerCount(ciphertextLen);
        if (workers > 1) {
            int blocksPerWorker = (blocks + workers - 1) / workers;
            // prev 快照必须在任务开跑前全部拿到：in == out 时任务会就地覆盖密文
            std::array<std::array<uint8_t, 32>, MAX_WORKERS> prevSnaps;
            for (int w = 0; w < workers; w++) {
                int startBlock = w * blocksPerWorker;
                if (startBlock == 0) {
                    std::memcpy(prevSnaps[w].data(), iv, 32);
                } else {
                    std::memcpy(prevSnaps[w].data(), ciphertext + (startBlock - 1) * 32, 32);
                }
            }

            parallelFor(workers, [&](int w) {
                int startBlock = w * blocksPerWorker;
                if (startBlock >= blocks) return;
                int endBlock = startBlock + blocksPerWorker;
                if (endBlock > blocks) endBlock = blocks;
                cbcDecryptRange(c, ciphertext + startBlock * 32,
                                out + startBlock * 32, endBlock - startBlock, prevSnaps[w].data());
            });
        } else {
            cbcDecryptRange(c, ciphertext, out, blocks, iv);
        }
    } else {
        cbcDecryptRange(c, ciphertext, out, blocks, iv);
    }

    // 常量时间校验 PKCS#7 填充
    uint8_t* last = out + (blocks - 1) * 32;
    int padLen = last[31];
    uint32_t pad = (uint32_t)padLen;
    uint8_t bad = (uint8_t)((((pad - 1) >> 31) | ((32u - pad) >> 31)) & 1);
    for (int i = 0; i < 32; i++) {
        uint8_t b = last[31 - i];
        uint32_t idxDiff = (uint32_t)((int32_t)(padLen - i - 1));
        uint8_t mask = (uint8_t)0 - (uint8_t)((idxDiff >> 31) ^ 1);
        bad |= (b ^ (uint8_t)padLen) & mask;
    }
    if (bad != 0) return false;

    *outLen = ciphertextLen - padLen;
    return true;
}

// XTS 模式

// XTS 密文窃取，处理最后不完整块。tweak 是最后一个完整块的 tweak
void xtsCiphertextSteal(const BlockCipher& c1, uint8_t* out, int fullBlocks, int partial,
                        uint8_t* tweak, bool encrypt) {
    int prevOff = (fullBlocks - 1) * 32;  // 倒数第二个完整块
    int lastOff = fullBlocks * 32;        // 最后不完整块的偏移

    uint8_t Ppartial[32] = {0}, Pprev[32], CC[32], PP[32];
    std::memcpy(Ppartial, out + lastOff, partial);
    std::memcpy(Pprev, out + prevOff, 32);

    if (encrypt) {
        // 加密倒数第二块得 CC，CC 前 m 字节当最后一块密文
        std::memcpy(CC, Pprev, 32);
        xorBlock32(CC, tweak);
        blockEncrypt(CC, CC, c1);
        xorBlock32(CC, tweak);
        std::memcpy(out + lastOff, CC, partial);
        // PP = CC 尾部 + Ppartial，用下一个 tweak 加密后写到倒数第二块
        std::memcpy(PP, CC + partial, 32 - partial);
        std::memcpy(PP + 32 - partial, Ppartial, partial);
        gfMul2_256(tweak);
        xorBlock32(PP, tweak);
        blockEncrypt(PP, PP, c1);
        xorBlock32(PP, tweak);
        std::memcpy(out + prevOff, PP, 32);
    } else {
        // 解密：先算 tweakN1（最后一块的 tweak）
        uint8_t tweakN1[32];
        std::memcpy(tweakN1, tweak, 32);
        gfMul2_256(tweakN1);

        uint8_t Cpartial[32] = {0};
        std::memcpy(Cpartial, out + lastOff, partial);

        // 解密倒数第二块（用 tweakN1），PP 尾部 m 字节是原明文尾部
        std::memcpy(PP, out + prevOff, 32);
        xorBlock32(PP, tweakN1);
        blockDecrypt(PP, PP, c1);
        xorBlock32(PP, tweakN1);
        std::memcpy(out + lastOff, PP + 32 - partial, partial);

        // CC = Cpartial + PP 前部，用 tweak 解密得倒数第二块明文
        std::memcpy(CC, Cpartial, partial);
        std::memcpy(CC + partial, PP, 32 - partial);
        xorBlock32(CC, tweak);
        blockDecrypt(CC, CC, c1);
        xorBlock32(CC, tweak);
        std::memcpy(out + prevOff, CC, 32);
    }
}

// XTS 处理，支持密文窃取和多核并行
void xtsProcess(const BlockCipher& c1, const BlockCipher& c2, const uint8_t* data, int len,
                uint64_t sector, bool encrypt, uint8_t* out) {
    std::memcpy(out, data, len);

    // tweak = E_key2(sector 小端填充 32 字节)
    uint8_t tweak[32] = {0};
    writeU64LE(tweak + 24, sector);
    blockEncrypt(tweak, tweak, c2);

    int fullBlocks = len / 32;
    int partial = len % 32;
    int processBlocks = fullBlocks;
    if (partial > 0) processBlocks = fullBlocks - 1;  // 留一块做密文窃取

    // 大数据量多核，每个 worker 处理一段连续块，独立 tweak 链
    if (processBlocks * 32 >= PARALLEL_THRESHOLD) {
        int workers = getOptimalWorkerCount(processBlocks * 32);
        if (workers > 1) {
            int blocksPerWorker = (processBlocks + workers - 1) / workers;
            // 串行预计算每个 worker 的起始 tweak
            std::vector<std::array<uint8_t, 32>> startTweaks(workers);
            std::memcpy(startTweaks[0].data(), tweak, 32);
            for (int w = 1; w < workers; w++) {
                std::memcpy(startTweaks[w].data(), startTweaks[w - 1].data(), 32);
                for (int i = 0; i < blocksPerWorker; i++) gfMul2_256(startTweaks[w].data());
            }

            // 每段把自己收尾的 tweak 写进各自槽位，密文窃取用最后一段的那个
            std::vector<std::array<uint8_t, 32>> finalTweaks(workers);

            parallelFor(workers, [&](int w) {
                int startBlock = w * blocksPerWorker;
                if (startBlock >= processBlocks) return;
                int endBlock = startBlock + blocksPerWorker;
                if (endBlock > processBlocks) endBlock = processBlocks;
                bool isLast = (endBlock == processBlocks);
                xtsProcessSegment(c1, startTweaks[w].data(), out + startBlock * 32,
                                  endBlock - startBlock, encrypt,
                                  isLast ? finalTweaks[w].data() : nullptr);
            });

            // 用最后一段跑完的 tweak 做密文窃取
            int lastSeg = (processBlocks - 1) / blocksPerWorker;
            std::memcpy(tweak, finalTweaks[lastSeg].data(), 32);

            if (partial > 0 && fullBlocks >= 1) {
                xtsCiphertextSteal(c1, out, fullBlocks, partial, tweak, encrypt);
            }
            return;
        }
    }

    // 小数据量单线程处理
    uint8_t finalTweak[32];
    xtsProcessSegment(c1, tweak, out, processBlocks, encrypt, finalTweak);
    std::memcpy(tweak, finalTweak, 32);

    // 密文窃取收尾
    if (partial > 0 && fullBlocks >= 1) {
        xtsCiphertextSteal(c1, out, fullBlocks, partial, tweak, encrypt);
    }
}

// AEAD

// 从主密钥和 nonce 派生加密密钥和认证密钥，nonce 当 HKDF 的 salt
void deriveAEADKeys(const uint8_t* master, int masterLen, const uint8_t* nonce, int nonceLen,
                    uint8_t encKey[32], uint8_t macKey[32]) {
    uint8_t prk[32];
    hkdfExtract(master, masterLen, nonce, nonceLen, prk);
    hkdfExpand(prk, (const uint8_t*)"aead-key", 8, 32, encKey);
    hkdfExpand(prk, (const uint8_t*)"tag-key", 7, 32, macKey);
}

// 算认证标签：HMAC(macKey, aad || ct || le64(aadLen) || le64(ctLen))，取前 16 字节
void aeadComputeTag(const uint8_t macKey[32], const uint8_t* aad, int aadLen,
                    const uint8_t* ct, int ctLen, uint8_t tag[16]) {
    HMACCtx hmac;
    hmacInit(&hmac, macKey, 32);
    if (aadLen > 0) hmacWrite(&hmac, aad, aadLen);
    if (ctLen > 0) hmacWrite(&hmac, ct, ctLen);
    // 尾部追加长度域
    uint8_t lenBuf[16];
    writeU64LE(lenBuf, (uint64_t)aadLen);
    writeU64LE(lenBuf + 8, (uint64_t)ctLen);
    hmacWrite(&hmac, lenBuf, 16);

    uint8_t fullTag[32];
    hmacFinal(&hmac, fullTag);
    std::memcpy(tag, fullTag, 16);
}

// 常量时间比较，防时序攻击
bool constantTimeCompare(const uint8_t* a, const uint8_t* b, int len) {
    uint8_t result = 0;
    for (int i = 0; i < len; i++) result |= a[i] ^ b[i];
    return result == 0;
}

// 安全清零：优先平台保证不被优化掉的清零 API，最后回退 volatile 逐字节写
void secureZero(void* ptr, size_t len) {
#if defined(_WIN32)
    SecureZeroMemory(ptr, len);
#elif defined(__STDC_LIB_EXT1__) && __STDC_LIB_EXT1__ >= 201112L
    memset_s(ptr, len, 0, len);
#elif defined(BASTION_HAVE_EXPLICIT_BZERO)
    explicit_bzero(ptr, len);
#else
    // volatile 阻止编译器把死存储优化掉
    volatile uint8_t* p = static_cast<volatile uint8_t*>(ptr);
    while (len--) *p++ = 0;
#endif
}

} // namespace

// 标量加解密函数（全局命名空间，供 Bastion_simd.cpp 调用）

// 从字节加载 8 个 uint32 加密后写回
void blockEncrypt(const uint8_t* src, uint8_t* dst, const BlockCipher& c) {
    uint32_t s0 = readU32LE(src), s1 = readU32LE(src + 4);
    uint32_t s2 = readU32LE(src + 8), s3 = readU32LE(src + 12);
    uint32_t s4 = readU32LE(src + 16), s5 = readU32LE(src + 20);
    uint32_t s6 = readU32LE(src + 24), s7 = readU32LE(src + 28);
    encryptBlockInline(s0, s1, s2, s3, s4, s5, s6, s7, c.rounds, c.roundKeys);
    writeU32LE(dst, s0); writeU32LE(dst + 4, s1);
    writeU32LE(dst + 8, s2); writeU32LE(dst + 12, s3);
    writeU32LE(dst + 16, s4); writeU32LE(dst + 20, s5);
    writeU32LE(dst + 24, s6); writeU32LE(dst + 28, s7);
}

void blockDecrypt(const uint8_t* src, uint8_t* dst, const BlockCipher& c) {
    uint32_t s0 = readU32LE(src), s1 = readU32LE(src + 4);
    uint32_t s2 = readU32LE(src + 8), s3 = readU32LE(src + 12);
    uint32_t s4 = readU32LE(src + 16), s5 = readU32LE(src + 20);
    uint32_t s6 = readU32LE(src + 24), s7 = readU32LE(src + 28);
    decryptBlockInline(s0, s1, s2, s3, s4, s5, s6, s7, c.rounds, c.roundKeys);
    writeU32LE(dst, s0); writeU32LE(dst + 4, s1);
    writeU32LE(dst + 8, s2); writeU32LE(dst + 12, s3);
    writeU32LE(dst + 16, s4); writeU32LE(dst + 20, s5);
    writeU32LE(dst + 24, s6); writeU32LE(dst + 28, s7);
}

// uint64 批量异或
void xorWords(const uint8_t* in, const uint8_t* key, uint8_t* out, int len) {
    int n = len & ~7;  // 8 字节对齐
    int i = 0;
    for (; i < n; i += 8) {
        writeU64LE(out + i, readU64LE(in + i) ^ readU64LE(key + i));
    }
    for (; i < len; i++) {
        out[i] = in[i] ^ key[i];
    }
}

// 4 块批量软件流水线加密：4 组独立状态字交错执行 16 轮，in/out 长度必须是 32 的整数倍
void encryptBlocksBatch4(const BlockCipher& c, const uint8_t* counter,
                         const uint8_t* in, uint8_t* out, int len) {
    int n = len / 32;  // 块数
    if (n <= 0) return;

    uint32_t n0 = readU32LE(counter);
    uint32_t n1 = readU32LE(counter + 4);
    uint32_t n2 = readU32LE(counter + 8);
    uint32_t n3 = readU32LE(counter + 12);
    uint32_t n4 = readU32LE(counter + 16);
    uint32_t n5 = readU32LE(counter + 20);
    uint64_t baseCtr = readU64LE(counter + 24);
    const uint64_t* keys = c.roundKeys;
    int rounds = c.rounds;

    uint32_t s0_0, s1_0, s2_0, s3_0, s4_0, s5_0, s6_0, s7_0;
    uint32_t s0_1, s1_1, s2_1, s3_1, s4_1, s5_1, s6_1, s7_1;
    uint32_t s0_2, s1_2, s2_2, s3_2, s4_2, s5_2, s6_2, s7_2;
    uint32_t s0_3, s1_3, s2_3, s3_3, s4_3, s5_3, s6_3, s7_3;

    int i = 0;
    // 4 块批量主循环
    for (; i + 3 < n; i += 4) {
        // 4 组计数器初值：baseCtr + i + 组号
        uint64_t c0 = baseCtr + (uint64_t)i;
        uint64_t c1 = baseCtr + (uint64_t)(i + 1);
        uint64_t c2 = baseCtr + (uint64_t)(i + 2);
        uint64_t c3 = baseCtr + (uint64_t)(i + 3);

        // 初始化 4 组状态字（前 6 个字是 nonce，后 2 个字是计数器）
        s0_0 = n0; s1_0 = n1; s2_0 = n2; s3_0 = n3; s4_0 = n4; s5_0 = n5; s6_0 = (uint32_t)c0; s7_0 = (uint32_t)(c0 >> 32);
        s0_1 = n0; s1_1 = n1; s2_1 = n2; s3_1 = n3; s4_1 = n4; s5_1 = n5; s6_1 = (uint32_t)c1; s7_1 = (uint32_t)(c1 >> 32);
        s0_2 = n0; s1_2 = n1; s2_2 = n2; s3_2 = n3; s4_2 = n4; s5_2 = n5; s6_2 = (uint32_t)c2; s7_2 = (uint32_t)(c2 >> 32);
        s0_3 = n0; s1_3 = n1; s2_3 = n2; s3_3 = n3; s4_3 = n4; s5_3 = n5; s6_3 = (uint32_t)c3; s7_3 = (uint32_t)(c3 >> 32);

        // 输入白化，4 组共享轮密钥，只加载一次
        uint32_t k0 = (uint32_t)keys[0], k1 = (uint32_t)(keys[0] >> 32);
        uint32_t k2 = (uint32_t)keys[1], k3 = (uint32_t)(keys[1] >> 32);
        uint32_t k4 = (uint32_t)keys[2], k5 = (uint32_t)(keys[2] >> 32);
        uint32_t k6 = (uint32_t)keys[3], k7 = (uint32_t)(keys[3] >> 32);
        s0_0 ^= k0; s1_0 ^= k1; s2_0 ^= k2; s3_0 ^= k3; s4_0 ^= k4; s5_0 ^= k5; s6_0 ^= k6; s7_0 ^= k7;
        s0_1 ^= k0; s1_1 ^= k1; s2_1 ^= k2; s3_1 ^= k3; s4_1 ^= k4; s5_1 ^= k5; s6_1 ^= k6; s7_1 ^= k7;
        s0_2 ^= k0; s1_2 ^= k1; s2_2 ^= k2; s3_2 ^= k3; s4_2 ^= k4; s5_2 ^= k5; s6_2 ^= k6; s7_2 ^= k7;
        s0_3 ^= k0; s1_3 ^= k1; s2_3 ^= k2; s3_3 ^= k3; s4_3 ^= k4; s5_3 ^= k5; s6_3 ^= k6; s7_3 ^= k7;

        int ki = 4;
        for (int r = 0; r < rounds; r++) {
            // 本轮轮密钥，4 组共享
            k0 = (uint32_t)keys[ki]; k1 = (uint32_t)(keys[ki] >> 32);
            k2 = (uint32_t)keys[ki + 1]; k3 = (uint32_t)(keys[ki + 1] >> 32);
            k4 = (uint32_t)keys[ki + 2]; k5 = (uint32_t)(keys[ki + 2] >> 32);
            k6 = (uint32_t)keys[ki + 3]; k7 = (uint32_t)(keys[ki + 3] >> 32);

            if ((r & 1) == 0) {
                // G_Mix 4 组
                s0_0 += rotr32(s2_0, GM_R5); s1_0 += rotr32(s3_0, GM_R5);
                s6_0 ^= s0_0; s7_0 ^= s1_0;
                s6_0 = rotr32(s6_0, GM_R1); s7_0 = rotr32(s7_0, GM_R1);
                s4_0 += rotr32(s6_0, GM_R6); s5_0 += rotr32(s7_0, GM_R6);
                s2_0 ^= s4_0; s3_0 ^= s5_0;
                s2_0 = rotr32(s2_0, GM_R2); s3_0 = rotr32(s3_0, GM_R2);
                s0_0 ^= rotr32(s4_0, GM_R9); s1_0 ^= rotr32(s5_0, GM_R9);
                s0_0 += rotr32(s2_0, GM_R7); s1_0 += rotr32(s3_0, GM_R7);
                s6_0 ^= s0_0; s7_0 ^= s1_0;
                s6_0 = rotr32(s6_0, GM_R3); s7_0 = rotr32(s7_0, GM_R3);
                s4_0 += rotr32(s6_0, GM_R8); s5_0 += rotr32(s7_0, GM_R8);
                s2_0 ^= s4_0; s3_0 ^= s5_0;
                s2_0 = rotr32(s2_0, GM_R4); s3_0 = rotr32(s3_0, GM_R4);
                s0_1 += rotr32(s2_1, GM_R5); s1_1 += rotr32(s3_1, GM_R5);
                s6_1 ^= s0_1; s7_1 ^= s1_1;
                s6_1 = rotr32(s6_1, GM_R1); s7_1 = rotr32(s7_1, GM_R1);
                s4_1 += rotr32(s6_1, GM_R6); s5_1 += rotr32(s7_1, GM_R6);
                s2_1 ^= s4_1; s3_1 ^= s5_1;
                s2_1 = rotr32(s2_1, GM_R2); s3_1 = rotr32(s3_1, GM_R2);
                s0_1 ^= rotr32(s4_1, GM_R9); s1_1 ^= rotr32(s5_1, GM_R9);
                s0_1 += rotr32(s2_1, GM_R7); s1_1 += rotr32(s3_1, GM_R7);
                s6_1 ^= s0_1; s7_1 ^= s1_1;
                s6_1 = rotr32(s6_1, GM_R3); s7_1 = rotr32(s7_1, GM_R3);
                s4_1 += rotr32(s6_1, GM_R8); s5_1 += rotr32(s7_1, GM_R8);
                s2_1 ^= s4_1; s3_1 ^= s5_1;
                s2_1 = rotr32(s2_1, GM_R4); s3_1 = rotr32(s3_1, GM_R4);
                s0_2 += rotr32(s2_2, GM_R5); s1_2 += rotr32(s3_2, GM_R5);
                s6_2 ^= s0_2; s7_2 ^= s1_2;
                s6_2 = rotr32(s6_2, GM_R1); s7_2 = rotr32(s7_2, GM_R1);
                s4_2 += rotr32(s6_2, GM_R6); s5_2 += rotr32(s7_2, GM_R6);
                s2_2 ^= s4_2; s3_2 ^= s5_2;
                s2_2 = rotr32(s2_2, GM_R2); s3_2 = rotr32(s3_2, GM_R2);
                s0_2 ^= rotr32(s4_2, GM_R9); s1_2 ^= rotr32(s5_2, GM_R9);
                s0_2 += rotr32(s2_2, GM_R7); s1_2 += rotr32(s3_2, GM_R7);
                s6_2 ^= s0_2; s7_2 ^= s1_2;
                s6_2 = rotr32(s6_2, GM_R3); s7_2 = rotr32(s7_2, GM_R3);
                s4_2 += rotr32(s6_2, GM_R8); s5_2 += rotr32(s7_2, GM_R8);
                s2_2 ^= s4_2; s3_2 ^= s5_2;
                s2_2 = rotr32(s2_2, GM_R4); s3_2 = rotr32(s3_2, GM_R4);
                s0_3 += rotr32(s2_3, GM_R5); s1_3 += rotr32(s3_3, GM_R5);
                s6_3 ^= s0_3; s7_3 ^= s1_3;
                s6_3 = rotr32(s6_3, GM_R1); s7_3 = rotr32(s7_3, GM_R1);
                s4_3 += rotr32(s6_3, GM_R6); s5_3 += rotr32(s7_3, GM_R6);
                s2_3 ^= s4_3; s3_3 ^= s5_3;
                s2_3 = rotr32(s2_3, GM_R2); s3_3 = rotr32(s3_3, GM_R2);
                s0_3 ^= rotr32(s4_3, GM_R9); s1_3 ^= rotr32(s5_3, GM_R9);
                s0_3 += rotr32(s2_3, GM_R7); s1_3 += rotr32(s3_3, GM_R7);
                s6_3 ^= s0_3; s7_3 ^= s1_3;
                s6_3 = rotr32(s6_3, GM_R3); s7_3 = rotr32(s7_3, GM_R3);
                s4_3 += rotr32(s6_3, GM_R8); s5_3 += rotr32(s7_3, GM_R8);
                s2_3 ^= s4_3; s3_3 ^= s5_3;
                s2_3 = rotr32(s2_3, GM_R4); s3_3 = rotr32(s3_3, GM_R4);
            } else {
                // H_Mix 4 组
                s0_0 += rotr32(s2_0, HM_R5); s1_0 += rotr32(s3_0, HM_R5);
                s6_0 ^= s0_0; s7_0 ^= s1_0;
                s6_0 = rotr32(s6_0, HM_R1); s7_0 = rotr32(s7_0, HM_R1);
                s4_0 += rotr32(s6_0, HM_R6); s5_0 += rotr32(s7_0, HM_R6);
                s2_0 ^= s4_0; s3_0 ^= s5_0;
                s2_0 = rotr32(s2_0, HM_R2); s3_0 = rotr32(s3_0, HM_R2);
                s0_0 ^= rotr32(s4_0, HM_R9); s1_0 ^= rotr32(s5_0, HM_R9);
                s0_0 += rotr32(s2_0, HM_R7); s1_0 += rotr32(s3_0, HM_R7);
                s6_0 ^= s0_0; s7_0 ^= s1_0;
                s6_0 = rotr32(s6_0, HM_R3); s7_0 = rotr32(s7_0, HM_R3);
                s4_0 += rotr32(s6_0, HM_R8); s5_0 += rotr32(s7_0, HM_R8);
                s2_0 ^= s4_0; s3_0 ^= s5_0;
                s2_0 = rotr32(s2_0, HM_R4); s3_0 = rotr32(s3_0, HM_R4);
                s0_1 += rotr32(s2_1, HM_R5); s1_1 += rotr32(s3_1, HM_R5);
                s6_1 ^= s0_1; s7_1 ^= s1_1;
                s6_1 = rotr32(s6_1, HM_R1); s7_1 = rotr32(s7_1, HM_R1);
                s4_1 += rotr32(s6_1, HM_R6); s5_1 += rotr32(s7_1, HM_R6);
                s2_1 ^= s4_1; s3_1 ^= s5_1;
                s2_1 = rotr32(s2_1, HM_R2); s3_1 = rotr32(s3_1, HM_R2);
                s0_1 ^= rotr32(s4_1, HM_R9); s1_1 ^= rotr32(s5_1, HM_R9);
                s0_1 += rotr32(s2_1, HM_R7); s1_1 += rotr32(s3_1, HM_R7);
                s6_1 ^= s0_1; s7_1 ^= s1_1;
                s6_1 = rotr32(s6_1, HM_R3); s7_1 = rotr32(s7_1, HM_R3);
                s4_1 += rotr32(s6_1, HM_R8); s5_1 += rotr32(s7_1, HM_R8);
                s2_1 ^= s4_1; s3_1 ^= s5_1;
                s2_1 = rotr32(s2_1, HM_R4); s3_1 = rotr32(s3_1, HM_R4);
                s0_2 += rotr32(s2_2, HM_R5); s1_2 += rotr32(s3_2, HM_R5);
                s6_2 ^= s0_2; s7_2 ^= s1_2;
                s6_2 = rotr32(s6_2, HM_R1); s7_2 = rotr32(s7_2, HM_R1);
                s4_2 += rotr32(s6_2, HM_R6); s5_2 += rotr32(s7_2, HM_R6);
                s2_2 ^= s4_2; s3_2 ^= s5_2;
                s2_2 = rotr32(s2_2, HM_R2); s3_2 = rotr32(s3_2, HM_R2);
                s0_2 ^= rotr32(s4_2, HM_R9); s1_2 ^= rotr32(s5_2, HM_R9);
                s0_2 += rotr32(s2_2, HM_R7); s1_2 += rotr32(s3_2, HM_R7);
                s6_2 ^= s0_2; s7_2 ^= s1_2;
                s6_2 = rotr32(s6_2, HM_R3); s7_2 = rotr32(s7_2, HM_R3);
                s4_2 += rotr32(s6_2, HM_R8); s5_2 += rotr32(s7_2, HM_R8);
                s2_2 ^= s4_2; s3_2 ^= s5_2;
                s2_2 = rotr32(s2_2, HM_R4); s3_2 = rotr32(s3_2, HM_R4);
                s0_3 += rotr32(s2_3, HM_R5); s1_3 += rotr32(s3_3, HM_R5);
                s6_3 ^= s0_3; s7_3 ^= s1_3;
                s6_3 = rotr32(s6_3, HM_R1); s7_3 = rotr32(s7_3, HM_R1);
                s4_3 += rotr32(s6_3, HM_R6); s5_3 += rotr32(s7_3, HM_R6);
                s2_3 ^= s4_3; s3_3 ^= s5_3;
                s2_3 = rotr32(s2_3, HM_R2); s3_3 = rotr32(s3_3, HM_R2);
                s0_3 ^= rotr32(s4_3, HM_R9); s1_3 ^= rotr32(s5_3, HM_R9);
                s0_3 += rotr32(s2_3, HM_R7); s1_3 += rotr32(s3_3, HM_R7);
                s6_3 ^= s0_3; s7_3 ^= s1_3;
                s6_3 = rotr32(s6_3, HM_R3); s7_3 = rotr32(s7_3, HM_R3);
                s4_3 += rotr32(s6_3, HM_R8); s5_3 += rotr32(s7_3, HM_R8);
                s2_3 ^= s4_3; s3_3 ^= s5_3;
                s2_3 = rotr32(s2_3, HM_R4); s3_3 = rotr32(s3_3, HM_R4);
            }

            // ShiftRows 4 组：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
            { uint32_t t = s0_0; s0_0 = s2_0; s2_0 = s7_0; s7_0 = s1_0; s1_0 = s5_0; s5_0 = s4_0; s4_0 = s3_0; s3_0 = s6_0; s6_0 = t; }
            { uint32_t t = s0_1; s0_1 = s2_1; s2_1 = s7_1; s7_1 = s1_1; s1_1 = s5_1; s5_1 = s4_1; s4_1 = s3_1; s3_1 = s6_1; s6_1 = t; }
            { uint32_t t = s0_2; s0_2 = s2_2; s2_2 = s7_2; s7_2 = s1_2; s1_2 = s5_2; s5_2 = s4_2; s4_2 = s3_2; s3_2 = s6_2; s6_2 = t; }
            { uint32_t t = s0_3; s0_3 = s2_3; s2_3 = s7_3; s7_3 = s1_3; s1_3 = s5_3; s5_3 = s4_3; s4_3 = s3_3; s3_3 = s6_3; s6_3 = t; }

            // AddRoundKey 4 组
            s0_0 ^= k0; s1_0 ^= k1; s2_0 ^= k2; s3_0 ^= k3; s4_0 ^= k4; s5_0 ^= k5; s6_0 ^= k6; s7_0 ^= k7;
            s0_1 ^= k0; s1_1 ^= k1; s2_1 ^= k2; s3_1 ^= k3; s4_1 ^= k4; s5_1 ^= k5; s6_1 ^= k6; s7_1 ^= k7;
            s0_2 ^= k0; s1_2 ^= k1; s2_2 ^= k2; s3_2 ^= k3; s4_2 ^= k4; s5_2 ^= k5; s6_2 ^= k6; s7_2 ^= k7;
            s0_3 ^= k0; s1_3 ^= k1; s2_3 ^= k2; s3_3 ^= k3; s4_3 ^= k4; s5_3 ^= k5; s6_3 ^= k6; s7_3 ^= k7;
            ki += 4;
        }

        // 输出白化
        {
            int ki_out = 4 + rounds * 4;
            uint32_t ok0 = (uint32_t)keys[ki_out], ok1 = (uint32_t)(keys[ki_out] >> 32);
            uint32_t ok2 = (uint32_t)keys[ki_out + 1], ok3 = (uint32_t)(keys[ki_out + 1] >> 32);
            uint32_t ok4 = (uint32_t)keys[ki_out + 2], ok5 = (uint32_t)(keys[ki_out + 2] >> 32);
            uint32_t ok6 = (uint32_t)keys[ki_out + 3], ok7 = (uint32_t)(keys[ki_out + 3] >> 32);
            s0_0 ^= ok0; s1_0 ^= ok1; s2_0 ^= ok2; s3_0 ^= ok3; s4_0 ^= ok4; s5_0 ^= ok5; s6_0 ^= ok6; s7_0 ^= ok7;
            s0_1 ^= ok0; s1_1 ^= ok1; s2_1 ^= ok2; s3_1 ^= ok3; s4_1 ^= ok4; s5_1 ^= ok5; s6_1 ^= ok6; s7_1 ^= ok7;
            s0_2 ^= ok0; s1_2 ^= ok1; s2_2 ^= ok2; s3_2 ^= ok3; s4_2 ^= ok4; s5_2 ^= ok5; s6_2 ^= ok6; s7_2 ^= ok7;
            s0_3 ^= ok0; s1_3 ^= ok1; s2_3 ^= ok2; s3_3 ^= ok3; s4_3 ^= ok4; s5_3 ^= ok5; s6_3 ^= ok6; s7_3 ^= ok7;
        }

        // 写出 4 块密钥流并 XOR src
        int off = i * 32;
        writeU32LE(out + off, s0_0 ^ readU32LE(in + off));
        writeU32LE(out + off + 4, s1_0 ^ readU32LE(in + off + 4));
        writeU32LE(out + off + 8, s2_0 ^ readU32LE(in + off + 8));
        writeU32LE(out + off + 12, s3_0 ^ readU32LE(in + off + 12));
        writeU32LE(out + off + 16, s4_0 ^ readU32LE(in + off + 16));
        writeU32LE(out + off + 20, s5_0 ^ readU32LE(in + off + 20));
        writeU32LE(out + off + 24, s6_0 ^ readU32LE(in + off + 24));
        writeU32LE(out + off + 28, s7_0 ^ readU32LE(in + off + 28));
        off += 32;
        writeU32LE(out + off, s0_1 ^ readU32LE(in + off));
        writeU32LE(out + off + 4, s1_1 ^ readU32LE(in + off + 4));
        writeU32LE(out + off + 8, s2_1 ^ readU32LE(in + off + 8));
        writeU32LE(out + off + 12, s3_1 ^ readU32LE(in + off + 12));
        writeU32LE(out + off + 16, s4_1 ^ readU32LE(in + off + 16));
        writeU32LE(out + off + 20, s5_1 ^ readU32LE(in + off + 20));
        writeU32LE(out + off + 24, s6_1 ^ readU32LE(in + off + 24));
        writeU32LE(out + off + 28, s7_1 ^ readU32LE(in + off + 28));
        off += 32;
        writeU32LE(out + off, s0_2 ^ readU32LE(in + off));
        writeU32LE(out + off + 4, s1_2 ^ readU32LE(in + off + 4));
        writeU32LE(out + off + 8, s2_2 ^ readU32LE(in + off + 8));
        writeU32LE(out + off + 12, s3_2 ^ readU32LE(in + off + 12));
        writeU32LE(out + off + 16, s4_2 ^ readU32LE(in + off + 16));
        writeU32LE(out + off + 20, s5_2 ^ readU32LE(in + off + 20));
        writeU32LE(out + off + 24, s6_2 ^ readU32LE(in + off + 24));
        writeU32LE(out + off + 28, s7_2 ^ readU32LE(in + off + 28));
        off += 32;
        writeU32LE(out + off, s0_3 ^ readU32LE(in + off));
        writeU32LE(out + off + 4, s1_3 ^ readU32LE(in + off + 4));
        writeU32LE(out + off + 8, s2_3 ^ readU32LE(in + off + 8));
        writeU32LE(out + off + 12, s3_3 ^ readU32LE(in + off + 12));
        writeU32LE(out + off + 16, s4_3 ^ readU32LE(in + off + 16));
        writeU32LE(out + off + 20, s5_3 ^ readU32LE(in + off + 20));
        writeU32LE(out + off + 24, s6_3 ^ readU32LE(in + off + 24));
        writeU32LE(out + off + 28, s7_3 ^ readU32LE(in + off + 28));
    }

    // 尾部不足 4 块，逐块标量加密 + xorWords
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

// C API 实现

extern "C" {

BASTION_API int Bastion_BlockEncrypt(const uint8_t* plaintext,
                                     const uint8_t* key, int keyLen,
                                     uint8_t* ciphertext) {
    if (keyLen != Bastion::KEY_SIZE) return -1;
    BlockCipher c;
    c.rounds = DEFAULT_ROUNDS;
    expandKey(&c, key);
    blockEncrypt(plaintext, ciphertext, c);
    secureZero(&c, sizeof(c));
    return 0;
}

BASTION_API int Bastion_BlockDecrypt(const uint8_t* ciphertext,
                                     const uint8_t* key, int keyLen,
                                     uint8_t* plaintext) {
    if (keyLen != Bastion::KEY_SIZE) return -1;
    BlockCipher c;
    c.rounds = DEFAULT_ROUNDS;
    expandKey(&c, key);
    blockDecrypt(ciphertext, plaintext, c);
    secureZero(&c, sizeof(c));
    return 0;
}

BASTION_API int Bastion_CTRCryptBuffer(const uint8_t* key, int keyLen,
                                       const uint8_t* nonce, int nonceLen,
                                       const uint8_t* in, int inLen,
                                       uint8_t* out) {
    if (keyLen != Bastion::KEY_SIZE || nonceLen != Bastion::NONCE_SIZE) return -1;
    if (inLen < 0) return -1;
    // 有数据时输入输出指针不能为空（长度为 0 时允许传空指针）
    if (inLen > 0 && (in == nullptr || out == nullptr)) return -1;
    BlockCipher c;
    c.rounds = DEFAULT_ROUNDS;
    expandKey(&c, key);
    // 构造 32 字节 counter 块：前 24 字节 nonce + 后 8 字节计数器初值 0
    uint8_t counter[32] = {0};
    std::memcpy(counter, nonce, 24);
    ctrCrypt(c, counter, in, out, inLen);
    secureZero(&c, sizeof(c));
    return 0;
}

BASTION_API int Bastion_CTRCrypt(const uint8_t* key, int keyLen,
                                 const uint8_t* nonce, int nonceLen,
                                 ReadCallback readCb, void* readCtx,
                                 WriteCallback writeCb, void* writeCtx) {
    if (keyLen != Bastion::KEY_SIZE || nonceLen != Bastion::NONCE_SIZE) return -1;
    // 读写回调不能为空，否则循环里直接调用会崩溃
    if (readCb == nullptr || writeCb == nullptr) return -1;
    BlockCipher c;
    c.rounds = DEFAULT_ROUNDS;
    expandKey(&c, key);

    // 用 4MB 缓冲区分块处理，避免一次性加载大文件
    const int BUF = 4 * 1024 * 1024;
    std::unique_ptr<uint8_t[]> inBuf(new uint8_t[BUF]);
    std::unique_ptr<uint8_t[]> outBuf(new uint8_t[BUF]);
    // 构造 32 字节 counter 块
    uint8_t ctr[32] = {0};
    std::memcpy(ctr, nonce, 24);
    uint64_t baseCtr = 0;

    while (true) {
        int n = readCb(readCtx, inBuf.get(), BUF);
        if (n < 0) { secureZero(&c, sizeof(c)); return -1; }
        // 回调声称读了超过请求长度的数据，后续按 n 读写会越界，按错误处理
        if (n > BUF) { secureZero(&c, sizeof(c)); return -1; }
        if (n == 0) break;
        writeU64LE(ctr + 24, baseCtr);
        ctrCrypt(c, ctr, inBuf.get(), outBuf.get(), n);
        int blocks = n / 32;
        baseCtr += (uint64_t)blocks;
        // 尾部不足块也消耗了一个计数器值
        if (n % 32 != 0) baseCtr++;
        if (writeCb(writeCtx, outBuf.get(), n) != n) {
            secureZero(&c, sizeof(c));
            return -1;
        }
        if (n < BUF) break;
    }
    secureZero(&c, sizeof(c));
    return 0;
}

BASTION_API int Bastion_CBCEncryptBuffer(const uint8_t* key, int keyLen,
                                         const uint8_t* iv, int ivLen,
                                         const uint8_t* plaintext, int plaintextLen,
                                         uint8_t* out, int* outLen) {
    if (keyLen != Bastion::KEY_SIZE || ivLen != Bastion::BLOCK_SIZE) return -1;
    if (plaintextLen < 0) return -1;
    // 有数据时明文指针不能为空；输出缓冲与长度指针无论如何都要有效
    if (plaintextLen > 0 && plaintext == nullptr) return -1;
    if (out == nullptr || outLen == nullptr) return -1;
    BlockCipher c;
    c.rounds = DEFAULT_ROUNDS;
    expandKey(&c, key);
    bool ok = cbcEncryptBuffer(c, iv, plaintext, plaintextLen, out, outLen);
    secureZero(&c, sizeof(c));
    return ok ? 0 : -1;
}

BASTION_API int Bastion_CBCDecryptBuffer(const uint8_t* key, int keyLen,
                                         const uint8_t* iv, int ivLen,
                                         const uint8_t* ciphertext, int ciphertextLen,
                                         uint8_t* plaintext, int* plaintextLen) {
    if (keyLen != Bastion::KEY_SIZE || ivLen != Bastion::BLOCK_SIZE) return -1;
    if (ciphertextLen < 0) return -1;
    // 有数据时密文指针不能为空；明文输出缓冲与长度指针无论如何都要有效
    if (ciphertextLen > 0 && ciphertext == nullptr) return -1;
    if (plaintext == nullptr || plaintextLen == nullptr) return -1;
    BlockCipher c;
    c.rounds = DEFAULT_ROUNDS;
    expandKey(&c, key);
    bool ok = cbcDecryptBuffer(c, iv, ciphertext, ciphertextLen, plaintext, plaintextLen);
    secureZero(&c, sizeof(c));
    return ok ? 0 : -1;
}

BASTION_API int Bastion_CBCEncrypt(const uint8_t* key, int keyLen,
                                   const uint8_t* iv, int ivLen,
                                   ReadCallback readCb, void* readCtx,
                                   WriteCallback writeCb, void* writeCtx) {
    if (keyLen != Bastion::KEY_SIZE || ivLen != Bastion::BLOCK_SIZE) return -1;
    // 读写回调不能为空，否则循环里直接调用会崩溃
    if (readCb == nullptr || writeCb == nullptr) return -1;
    BlockCipher c;
    c.rounds = DEFAULT_ROUNDS;
    expandKey(&c, key);

    const int BUF = 4 * 1024 * 1024;
    std::unique_ptr<uint8_t[]> buf(new uint8_t[BUF + 32]);
    // prev 用 uint32 寄存器存，省掉每块 memcpy 和逐字节 XOR
    uint32_t p0 = readU32LE(iv), p1 = readU32LE(iv + 4);
    uint32_t p2 = readU32LE(iv + 8), p3 = readU32LE(iv + 12);
    uint32_t p4 = readU32LE(iv + 16), p5 = readU32LE(iv + 20);
    uint32_t p6 = readU32LE(iv + 24), p7 = readU32LE(iv + 28);
    int carry = 0;

    while (true) {
        int n = readCb(readCtx, buf.get() + carry, BUF);
        if (n < 0) { secureZero(&c, sizeof(c)); return -1; }
        // 缓冲区预留了 carry 个字节的空间，超过 BUF 就会写越界
        if (n > BUF) { secureZero(&c, sizeof(c)); return -1; }
        int end = carry + n;
        int fullEnd = (end / 32) * 32;
        // 逐块加密：P XOR prev -> E -> out
        for (int i = 0; i < fullEnd; i += 32) {
            uint8_t* bp = buf.get() + i;
            uint32_t s0 = readU32LE(bp)      ^ p0;
            uint32_t s1 = readU32LE(bp + 4)  ^ p1;
            uint32_t s2 = readU32LE(bp + 8)  ^ p2;
            uint32_t s3 = readU32LE(bp + 12) ^ p3;
            uint32_t s4 = readU32LE(bp + 16) ^ p4;
            uint32_t s5 = readU32LE(bp + 20) ^ p5;
            uint32_t s6 = readU32LE(bp + 24) ^ p6;
            uint32_t s7 = readU32LE(bp + 28) ^ p7;
            encryptBlockInline(s0, s1, s2, s3, s4, s5, s6, s7, c.rounds, c.roundKeys);
            writeU32LE(bp, s0);      p0 = s0;
            writeU32LE(bp + 4, s1);  p1 = s1;
            writeU32LE(bp + 8, s2);  p2 = s2;
            writeU32LE(bp + 12, s3); p3 = s3;
            writeU32LE(bp + 16, s4); p4 = s4;
            writeU32LE(bp + 20, s5); p5 = s5;
            writeU32LE(bp + 24, s6); p6 = s6;
            writeU32LE(bp + 28, s7); p7 = s7;
        }
        if (fullEnd > 0) {
            if (writeCb(writeCtx, buf.get(), fullEnd) != fullEnd) {
                secureZero(&c, sizeof(c));
                return -1;
            }
        }
        // 不足一块的尾部挪到开头
        int leftover = end - fullEnd;
        std::memmove(buf.get(), buf.get() + fullEnd, leftover);
        carry = leftover;
        if (n == 0 || n < BUF) {
            // EOF，补 PKCS#7 填充
            int padLen = 32 - carry;
            uint8_t last[32];
            std::memcpy(last, buf.get(), carry);
            for (int i = carry; i < 32; i++) last[i] = (uint8_t)padLen;
            uint32_t s0 = readU32LE(last)      ^ p0;
            uint32_t s1 = readU32LE(last + 4)  ^ p1;
            uint32_t s2 = readU32LE(last + 8)  ^ p2;
            uint32_t s3 = readU32LE(last + 12) ^ p3;
            uint32_t s4 = readU32LE(last + 16) ^ p4;
            uint32_t s5 = readU32LE(last + 20) ^ p5;
            uint32_t s6 = readU32LE(last + 24) ^ p6;
            uint32_t s7 = readU32LE(last + 28) ^ p7;
            encryptBlockInline(s0, s1, s2, s3, s4, s5, s6, s7, c.rounds, c.roundKeys);
            writeU32LE(last, s0);
            writeU32LE(last + 4, s1);
            writeU32LE(last + 8, s2);
            writeU32LE(last + 12, s3);
            writeU32LE(last + 16, s4);
            writeU32LE(last + 20, s5);
            writeU32LE(last + 24, s6);
            writeU32LE(last + 28, s7);
            if (writeCb(writeCtx, last, 32) != 32) {
                secureZero(&c, sizeof(c));
                return -1;
            }
            break;
        }
    }
    secureZero(&c, sizeof(c));
    return 0;
}

BASTION_API int Bastion_CBCDecrypt(const uint8_t* key, int keyLen,
                                   const uint8_t* iv, int ivLen,
                                   ReadCallback readCb, void* readCtx,
                                   WriteCallback writeCb, void* writeCtx) {
    if (keyLen != Bastion::KEY_SIZE || ivLen != Bastion::BLOCK_SIZE) return -1;
    // 读写回调不能为空，否则循环里直接调用会崩溃
    if (readCb == nullptr || writeCb == nullptr) return -1;
    BlockCipher c;
    c.rounds = DEFAULT_ROUNDS;
    expandKey(&c, key);

    const int BUF = 4 * 1024 * 1024;
    std::unique_ptr<uint8_t[]> buf(new uint8_t[BUF + 32]);
    uint8_t prev[32];
    std::memcpy(prev, iv, 32);
    int carry = 0;
    bool hasPending = false;
    uint8_t pending[32];

    while (true) {
        int n = readCb(readCtx, buf.get() + carry, BUF);
        if (n < 0) { secureZero(&c, sizeof(c)); return -1; }
        // 缓冲区预留了 carry 个字节的空间，超过 BUF 就会写越界
        if (n > BUF) { secureZero(&c, sizeof(c)); return -1; }
        int end = carry + n;
        int fullEnd = (end / 32) * 32;
        for (int i = 0; i < fullEnd; i += 32) {
            if (hasPending) {
                uint8_t dec[32];
                blockDecrypt(pending, dec, c);
                for (int j = 0; j < 32; j++) dec[j] ^= prev[j];
                std::memcpy(prev, pending, 32);
                if (writeCb(writeCtx, dec, 32) != 32) {
                    secureZero(&c, sizeof(c));
                    return -1;
                }
            }
            std::memcpy(pending, buf.get() + i, 32);
            hasPending = true;
        }
        int leftover = end - fullEnd;
        std::memmove(buf.get(), buf.get() + fullEnd, leftover);
        carry = leftover;
        if (n == 0 || n < BUF) break;
    }

    if (!hasPending) { secureZero(&c, sizeof(c)); return -1; }
    // 解密最后一块并校验填充
    uint8_t plain[32];
    blockDecrypt(pending, plain, c);
    for (int i = 0; i < 32; i++) plain[i] ^= prev[i];

    int padLen = plain[31];
    uint32_t pad = (uint32_t)padLen;
    uint8_t bad = (uint8_t)((((pad - 1) >> 31) | ((32u - pad) >> 31)) & 1);
    for (int i = 0; i < 32; i++) {
        uint8_t b = plain[31 - i];
        uint32_t idxDiff = (uint32_t)((int32_t)(padLen - i - 1));
        uint8_t mask = (uint8_t)0 - (uint8_t)((idxDiff >> 31) ^ 1);
        bad |= (b ^ (uint8_t)padLen) & mask;
    }
    if (bad != 0) { secureZero(&c, sizeof(c)); return -1; }

    int writeLen = 32 - padLen;
    if (writeLen > 0) {
        if (writeCb(writeCtx, plain, writeLen) != writeLen) {
            secureZero(&c, sizeof(c));
            return -1;
        }
    }
    secureZero(&c, sizeof(c));
    return 0;
}

BASTION_API int Bastion_XTSEncrypt(const uint8_t* key1, int key1Len,
                                   const uint8_t* key2, int key2Len,
                                   uint64_t sectorNum,
                                   const uint8_t* plaintext, int plaintextLen,
                                   uint8_t* ciphertext) {
    if (key1Len != Bastion::KEY_SIZE || key2Len != Bastion::KEY_SIZE) return -1;
    if (plaintextLen < Bastion::BLOCK_SIZE) return -1;
    // 数据长度至少一块，明文指针必然要有效
    if (plaintext == nullptr) return -1;
    BlockCipher c1, c2;
    c1.rounds = DEFAULT_ROUNDS; c2.rounds = DEFAULT_ROUNDS;
    expandKey(&c1, key1);
    expandKey(&c2, key2);
    xtsProcess(c1, c2, plaintext, plaintextLen, sectorNum, true, ciphertext);
    secureZero(&c1, sizeof(c1));
    secureZero(&c2, sizeof(c2));
    return 0;
}

BASTION_API int Bastion_XTSDecrypt(const uint8_t* key1, int key1Len,
                                   const uint8_t* key2, int key2Len,
                                   uint64_t sectorNum,
                                   const uint8_t* ciphertext, int ciphertextLen,
                                   uint8_t* plaintext) {
    if (key1Len != Bastion::KEY_SIZE || key2Len != Bastion::KEY_SIZE) return -1;
    if (ciphertextLen < Bastion::BLOCK_SIZE) return -1;
    // 数据长度至少一块，密文指针必然要有效
    if (ciphertext == nullptr) return -1;
    BlockCipher c1, c2;
    c1.rounds = DEFAULT_ROUNDS; c2.rounds = DEFAULT_ROUNDS;
    expandKey(&c1, key1);
    expandKey(&c2, key2);
    xtsProcess(c1, c2, ciphertext, ciphertextLen, sectorNum, false, plaintext);
    secureZero(&c1, sizeof(c1));
    secureZero(&c2, sizeof(c2));
    return 0;
}

BASTION_API int Bastion_AEADSeal(const uint8_t* key, int keyLen,
                                 const uint8_t* nonce, int nonceLen,
                                 const uint8_t* ad, int adLen,
                                 const uint8_t* plaintext, int plaintextLen,
                                 uint8_t* out, int* outLen) {
    if (keyLen != Bastion::KEY_SIZE || nonceLen != Bastion::NONCE_SIZE) return -1;
    if (ad == nullptr && adLen > 0) return -1;
    if (adLen < 0 || plaintextLen < 0) return -1;
    // 有数据时指针不能为空（长度为 0 时允许传空指针），输出指针与长度必须有效
    if ((plaintext == nullptr && plaintextLen > 0) || out == nullptr || outLen == nullptr) return -1;

    uint8_t encKey[32], macKey[32];
    deriveAEADKeys(key, keyLen, nonce, nonceLen, encKey, macKey);

    // CTR 加密，nonce 当计数器块前 24 字节，后 8 字节从 0 开始
    BlockCipher c;
    c.rounds = DEFAULT_ROUNDS;
    expandKey(&c, encKey);
    uint8_t counter[32] = {0};
    std::memcpy(counter, nonce, 24);
    ctrCrypt(c, counter, plaintext, out, plaintextLen);
    secureZero(&c, sizeof(c));

    // 算认证标签
    uint8_t tag[16];
    aeadComputeTag(macKey, ad, adLen, out, plaintextLen, tag);
    std::memcpy(out + plaintextLen, tag, 16);
    *outLen = plaintextLen + 16;

    secureZero(encKey, 32);
    secureZero(macKey, 32);
    return 0;
}

BASTION_API int Bastion_AEADOpen(const uint8_t* key, int keyLen,
                                 const uint8_t* nonce, int nonceLen,
                                 const uint8_t* ad, int adLen,
                                 const uint8_t* ciphertext, int ciphertextLen,
                                 uint8_t* plaintext, int* plaintextLen) {
    if (keyLen != Bastion::KEY_SIZE || nonceLen != Bastion::NONCE_SIZE) return -1;
    if (ciphertextLen < 16) return -1;
    if (ad == nullptr && adLen > 0) return -1;
    if (adLen < 0) return -1;
    // ciphertext 至少 16 字节所以不能为空；明文长度指针必须有效，解密结果为空时允许明文缓冲为空指针
    if (ciphertext == nullptr || plaintextLen == nullptr) return -1;
    if (plaintext == nullptr && ciphertextLen > 16) return -1;

    int ctLen = ciphertextLen - 16;
    const uint8_t* expectedTag = ciphertext + ctLen;

    uint8_t encKey[32], macKey[32];
    deriveAEADKeys(key, keyLen, nonce, nonceLen, encKey, macKey);

    // 先验证标签，过了再解密
    uint8_t actualTag[16];
    aeadComputeTag(macKey, ad, adLen, ciphertext, ctLen, actualTag);
    if (!constantTimeCompare(actualTag, expectedTag, 16)) {
        secureZero(encKey, 32);
        secureZero(macKey, 32);
        return 1;  // 验证失败
    }

    BlockCipher c;
    c.rounds = DEFAULT_ROUNDS;
    expandKey(&c, encKey);
    uint8_t counter[32] = {0};
    std::memcpy(counter, nonce, 24);
    ctrCrypt(c, counter, ciphertext, plaintext, ctLen);
    secureZero(&c, sizeof(c));

    *plaintextLen = ctLen;
    secureZero(encKey, 32);
    secureZero(macKey, 32);
    return 0;
}

BASTION_API int Bastion_AEADSealStream(const uint8_t* key, int keyLen,
                                       const uint8_t* nonce, int nonceLen,
                                       const uint8_t* ad, int adLen,
                                       ReadCallback readCb, void* readCtx,
                                       WriteCallback writeCb, void* writeCtx,
                                       uint8_t outTag[Bastion::TAG_SIZE]) {
    if (keyLen != Bastion::KEY_SIZE || nonceLen != Bastion::NONCE_SIZE) return -1;
    if (ad == nullptr && adLen > 0) return -1;
    if (adLen < 0) return -1;
    // 读写回调与标签输出缓冲不能为空，否则直接调用会崩溃
    if (readCb == nullptr || writeCb == nullptr || outTag == nullptr) return -1;

    uint8_t encKey[32], macKey[32];
    deriveAEADKeys(key, keyLen, nonce, nonceLen, encKey, macKey);

    BlockCipher c;
    c.rounds = DEFAULT_ROUNDS;
    expandKey(&c, encKey);

    // HMAC 边读边算，CTR 边读边加密
    Hash innerH, outerH;
    hashInit(&innerH);
    hashInit(&outerH);
    uint8_t ipad[32], opad[32];
    for (int i = 0; i < 32; i++) { ipad[i] = macKey[i] ^ 0x36; opad[i] = macKey[i] ^ 0x5C; }
    // ipad/opad 都是 32 字节，直接异或到初始状态构造带密钥的哈希状态
    for (int i = 0; i < 8; i++) {
        innerH.state[i] = ARX_INIT_STATE[i] ^ readU32LE(ipad + i * 4);
        outerH.state[i] = ARX_INIT_STATE[i] ^ readU32LE(opad + i * 4);
    }

    // aad 进 inner
    if (adLen > 0) hashWrite(&innerH, ad, adLen);

    const int BUF = 4 * 1024 * 1024;
    // 双缓冲：主线程 CTR 加密时，HMAC 线程并行消费上一块密文算 inner 哈希
    std::unique_ptr<uint8_t[]> inBufs[2];
    std::unique_ptr<uint8_t[]> outBufs[2];
    inBufs[0].reset(new uint8_t[BUF]);
    inBufs[1].reset(new uint8_t[BUF]);
    outBufs[0].reset(new uint8_t[BUF]);
    outBufs[1].reset(new uint8_t[BUF]);

    // 槽位状态：0=空闲，1=密文已就绪待 HMAC 消费
    int slotState[2] = {0, 0};
    int slotLen[2] = {0, 0};
    std::mutex mtx;
    std::condition_variable cvProduce;   // 主线程等待槽位空闲
    std::condition_variable cvConsume;   // HMAC 线程等待密文就绪
    bool macDone = false;
    bool ioError = false;

    std::thread macThread([&] {
        int slot = 0;
        while (true) {
            std::unique_lock<std::mutex> lk(mtx);
            cvConsume.wait(lk, [&] { return slotState[slot] == 1 || macDone; });
            if (slotState[slot] == 1) {
                hashWrite(&innerH, outBufs[slot].get(), slotLen[slot]);
                slotState[slot] = 0;
                lk.unlock();
                cvProduce.notify_one();
                slot = 1 - slot;
            } else if (macDone) {
                // 收尾：处理主线程退出时可能残留的密文槽位
                for (int k = 0; k < 2; k++) {
                    if (slotState[k] == 1) {
                        hashWrite(&innerH, outBufs[k].get(), slotLen[k]);
                        slotState[k] = 0;
                    }
                }
                break;
            }
        }
    });

    uint8_t ctr[32] = {0};
    std::memcpy(ctr, nonce, 24);
    uint64_t baseCtr = 0;
    int64_t ctLen = 0;
    int cur = 0;

    while (true) {
        {
            std::unique_lock<std::mutex> lk(mtx);
            cvProduce.wait(lk, [&] { return slotState[cur] == 0; });
        }
        int n = readCb(readCtx, inBufs[cur].get(), BUF);
        if (n < 0) { ioError = true; break; }
        // 回调声称读了超过请求长度的数据，后续按 n 加解密会越界，按 I/O 错误处理
        if (n > BUF) { ioError = true; break; }
        if (n == 0) break;
        writeU64LE(ctr + 24, baseCtr);
        ctrCrypt(c, ctr, inBufs[cur].get(), outBufs[cur].get(), n);
        int blocks = n / 32;
        baseCtr += (uint64_t)blocks;
        if (n % 32 != 0) baseCtr++;
        if (writeCb(writeCtx, outBufs[cur].get(), n) != n) { ioError = true; break; }
        ctLen += n;
        {
            std::unique_lock<std::mutex> lk(mtx);
            slotLen[cur] = n;
            slotState[cur] = 1;
        }
        cvConsume.notify_one();
        cur = 1 - cur;
        if (n < BUF) break;
    }

    // 通知 HMAC 线程收尾并等待其消费完残留密文
    {
        std::unique_lock<std::mutex> lk(mtx);
        macDone = true;
    }
    cvConsume.notify_all();
    macThread.join();

    secureZero(&c, sizeof(c));
    if (ioError) {
        secureZero(encKey, 32);
        secureZero(macKey, 32);
        return -1;
    }

    // 追加长度域（aadLen、ctLen 均为主线程写入的量，与 HMAC 消费顺序一致）
    uint8_t lenBuf[16];
    writeU64LE(lenBuf, (uint64_t)adLen);
    writeU64LE(lenBuf + 8, (uint64_t)ctLen);
    hashWrite(&innerH, lenBuf, 16);

    uint8_t innerHash[32];
    hashSum(&innerH, innerHash);
    hashWrite(&outerH, innerHash, 32);
    uint8_t fullTag[32];
    hashSum(&outerH, fullTag);
    std::memcpy(outTag, fullTag, 16);

    secureZero(encKey, 32);
    secureZero(macKey, 32);
    return 0;
}

BASTION_API int Bastion_Hash(const uint8_t* input, int64_t inputLen,
                            uint8_t out[Bastion::BLOCK_SIZE]) {
    // 长度不能为负；有数据时指针不能为空（空输入的指针允许为空）
    if (inputLen < 0 || (input == nullptr && inputLen > 0)) return -1;
    Hash h;
    hashInit(&h);
    hashWrite64(&h, input, inputLen);
    hashSum(&h, out);
    return 0;
}

BASTION_API int Bastion_HashStream(ReadCallback readCb, void* ctx,
                                   uint8_t out[Bastion::BLOCK_SIZE]) {
    // 读回调不能为空，否则循环里直接调用会崩溃
    if (readCb == nullptr) return -1;
    Hash h;
    hashInit(&h);
    const int BUF = 4 * 1024 * 1024;
    std::unique_ptr<uint8_t[]> buf(new uint8_t[BUF]);
    while (true) {
        int n = readCb(ctx, buf.get(), BUF);
        if (n < 0) return -1;
        // 回调声称读了超过请求长度的数据，hashWrite 按 n 读取会越界
        if (n > BUF) return -1;
        if (n == 0) break;
        hashWrite(&h, buf.get(), n);
        if (n < BUF) break;
    }
    hashSum(&h, out);
    return 0;
}

BASTION_API int Bastion_XOF(const uint8_t* input, int64_t inputLen,
                           int length, uint8_t* out) {
    if (length <= 0 || out == nullptr) return -1;
    if (inputLen < 0 || (input == nullptr && inputLen > 0)) return -1;
    // 先算哈希，再用哈希值当密钥对零块做 CTR 扩展
    uint8_t state[32];
    Hash h;
    hashInit(&h);
    hashWrite64(&h, input, inputLen);
    hashSum(&h, state);

    BlockCipher c;
    c.rounds = DEFAULT_ROUNDS;
    expandKey(&c, state);

    // counter 全 0，零块当输入走批量加密出密钥流，一次 4KB
    uint8_t counter[32] = {0};
    static const uint8_t zeros[4096] = {0};
    uint64_t baseCtr = 0;
    int written = 0;
    while (written < length) {
        int remain = length - written;
        int chunk = remain >= (int)sizeof(zeros) ? (int)sizeof(zeros) : (remain / 32) * 32;
        if (chunk <= 0) break;
        writeU64LE(counter + 24, baseCtr);
        encryptBlocksDispatched(c, counter, zeros, out + written, chunk);
        written += chunk;
        baseCtr += (uint64_t)(chunk / 32);
    }
    // 不足一块的尾巴，单块加密取前几个字节
    if (written < length) {
        writeU64LE(counter + 24, baseCtr);
        uint8_t ks[32];
        blockEncrypt(counter, ks, c);
        for (int i = written; i < length; i++) out[i] = ks[i - written];
    }
    secureZero(&c, sizeof(c));
    return 0;
}

BASTION_API int Bastion_HMAC(const uint8_t* key, int keyLen,
                            const uint8_t* data, int64_t dataLen,
                            uint8_t out[Bastion::BLOCK_SIZE]) {
    if (keyLen < 0 || dataLen < 0) return -1;
    // 有数据时指针不能为空（空输入的指针允许为空）
    if ((key == nullptr && keyLen > 0) || (data == nullptr && dataLen > 0)) return -1;
    hmacCompute(key, keyLen, data, dataLen, out);
    return 0;
}

BASTION_API int Bastion_HKDF(const uint8_t* masterKey, int masterKeyLen,
                             const uint8_t* salt, int saltLen,
                             const uint8_t* info, int infoLen,
                             int length, uint8_t* out) {
    if (length <= 0 || length > 255 * 32 || out == nullptr) return -1;
    if (masterKeyLen < 0 || saltLen < 0 || infoLen < 0) return -1;
    // 有数据时指针不能为空（长度为 0 时允许传空指针，对应"无 salt/无 info"）
    if ((masterKey == nullptr && masterKeyLen > 0) ||
        (salt == nullptr && saltLen > 0) ||
        (info == nullptr && infoLen > 0)) return -1;
    uint8_t prk[32];
    hkdfExtract(masterKey, masterKeyLen, salt, saltLen, prk);
    bool ok = hkdfExpand(prk, info, infoLen, length, out);
    // prk 是主密钥派生的中间密钥，用完清掉
    secureZero(prk, 32);
    if (!ok) return -1;
    return 0;
}

BASTION_API int Bastion_PBKDF2(const uint8_t* password, int passwordLen,
                               const uint8_t* salt, int saltLen,
                               int iterations, int keyLength,
                               uint8_t* out) {
    if (iterations < 1 || keyLength < 1 || passwordLen <= 0) return -1;
    if (password == nullptr || (salt == nullptr && saltLen > 0)) return -1;
    if (saltLen < 0) return -1;
    // keyLength 接近 int 上限时下面的 keyLength + 31 会溢出，提前拒绝
    if (keyLength > (std::numeric_limits<int>::max)() - 31) return -1;
    // salt 长度接近 int 上限时下面的 saltLen + 4 会溢出，提前拒绝
    if (saltLen > (std::numeric_limits<int>::max)() - 4) return -1;
    int numBlocks = (keyLength + 31) / 32;
    int offset = 0;
    // salt 长度不定，动态分配防栈溢出
    std::unique_ptr<uint8_t[]> inputBuf(new uint8_t[saltLen + 4]);
    uint8_t* input = inputBuf.get();

    // 块数不足 4 时 4 路并行只有开销没有收益，直接串行
    if (numBlocks < 4) {
        for (int blockIdx = 1; blockIdx <= numBlocks; blockIdx++) {
            // U1 = HMAC(password, salt || BE(blockIdx))
            int inputLen = 0;
            if (saltLen > 0) {
                std::memcpy(input, salt, saltLen);
                inputLen = saltLen;
            }
            // 块索引大端，RFC 2898 标准
            input[inputLen] = (uint8_t)(blockIdx >> 24);
            input[inputLen + 1] = (uint8_t)(blockIdx >> 16);
            input[inputLen + 2] = (uint8_t)(blockIdx >> 8);
            input[inputLen + 3] = (uint8_t)blockIdx;
            inputLen += 4;

            uint8_t uPrev[32];
            hmacCompute(password, passwordLen, input, inputLen, uPrev);
            uint8_t accumulator[32];
            std::memcpy(accumulator, uPrev, 32);

            // 迭代 c-1 次
            for (int j = 2; j <= iterations; j++) {
                uint8_t uCurr[32];
                hmacCompute(password, passwordLen, uPrev, 32, uCurr);
                for (int k = 0; k < 32; k++) accumulator[k] ^= uCurr[k];
                std::memcpy(uPrev, uCurr, 32);
            }

            int copyLen = 32;
            if (offset + copyLen > keyLength) copyLen = keyLength - offset;
            std::memcpy(out + offset, accumulator, copyLen);
            offset += copyLen;

            // 迭代中间值就是派生密钥材料，写完马上清掉
            secureZero(uPrev, 32);
            secureZero(accumulator, 32);
        }
        return 0;
    }

    // 4 个输出块一组并行（RFC 2898 各块链独立），迭代热点的 HMAC 走 4 链 SIMD
    for (int base = 1; base <= numBlocks; base += 4) {
        int n = numBlocks - base + 1;
        if (n > 4) n = 4;

        uint8_t accum[4][32];
        uint8_t uPrev[4][32];

        // U1 = HMAC(password, salt || BE(blockIdx))
        for (int i = 0; i < n; i++) {
            int blockIdx = base + i;
            int inputLen = 0;
            if (saltLen > 0) {
                std::memcpy(input, salt, saltLen);
                inputLen = saltLen;
            }
            // 块索引大端，RFC 2898 标准
            input[inputLen] = (uint8_t)(blockIdx >> 24);
            input[inputLen + 1] = (uint8_t)(blockIdx >> 16);
            input[inputLen + 2] = (uint8_t)(blockIdx >> 8);
            input[inputLen + 3] = (uint8_t)blockIdx;
            inputLen += 4;

            hmacCompute(password, passwordLen, input, inputLen, uPrev[i]);
            std::memcpy(accum[i], uPrev[i], 32);
        }

        // 迭代 c-1 次：n 条链并行（n==4 时走 4 链 SIMD）
        for (int j = 2; j <= iterations; j++) {
            uint8_t uCurr[4][32];
            if (n == 4) {
                const uint8_t* msgs[4] = { uPrev[0], uPrev[1], uPrev[2], uPrev[3] };
                hmacBlock4x(password, passwordLen, msgs, uCurr);
            } else {
                for (int i = 0; i < n; i++) {
                    hmacCompute(password, passwordLen, uPrev[i], 32, uCurr[i]);
                }
            }
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < 32; k++) accum[i][k] ^= uCurr[i][k];
                std::memcpy(uPrev[i], uCurr[i], 32);
            }
        }

        for (int i = 0; i < n; i++) {
            int copyLen = 32;
            if (offset + copyLen > keyLength) copyLen = keyLength - offset;
            std::memcpy(out + offset, accum[i], copyLen);
            offset += copyLen;

            // 迭代中间值就是派生密钥材料，写完马上清掉
            secureZero(uPrev[i], 32);
            secureZero(accum[i], 32);
        }
    }
    return 0;
}

BASTION_API int Bastion_DeriveKey(const char* context,
                                  const uint8_t* keyMaterial, int keyMaterialLen,
                                  int length, uint8_t* out) {
    if (length <= 0 || out == nullptr) return -1;
    if (context == nullptr || keyMaterialLen < 0) return -1;
    if (keyMaterial == nullptr && keyMaterialLen > 0) return -1;
    // input = context + 0x00 + keyMaterial
    int ctxLen = (int)std::strlen(context);
    // 两段长度相加超过 int 上限时总长度会算成负数，提前拒绝
    if (keyMaterialLen > (std::numeric_limits<int>::max)() - 1 - ctxLen) return -1;
    std::unique_ptr<uint8_t[]> input(new uint8_t[ctxLen + 1 + keyMaterialLen]);
    std::memcpy(input.get(), context, ctxLen);
    input[ctxLen] = 0x00;
    std::memcpy(input.get() + ctxLen + 1, keyMaterial, keyMaterialLen);

    uint8_t baseKey[32];
    Hash h;
    hashInit(&h);
    hashWrite(&h, input.get(), ctxLen + 1 + keyMaterialLen);
    hashSum(&h, baseKey);
    // input 里含原始密钥材料，哈希完就没用了
    secureZero(input.get(), ctxLen + 1 + keyMaterialLen);

    if (length <= 32) {
        std::memcpy(out, baseKey, length);
        secureZero(baseKey, 32);
        return 0;
    }

    // 超过 32 字节用 CTR 扩展，零计数器
    BlockCipher c;
    c.rounds = DEFAULT_ROUNDS;
    expandKey(&c, baseKey);
    uint8_t counter[32] = {0};
    std::unique_ptr<uint8_t[]> zeros(new uint8_t[length]);
    std::memset(zeros.get(), 0, length);
    ctrCrypt(c, counter, zeros.get(), out, length);
    secureZero(&c, sizeof(c));
    secureZero(baseKey, 32);
    return 0;
}

} // extern "C"
