#pragma once

// Bastion.cpp 与 Bastion_simd.cpp 共享的内部定义

#include <cstdint>
#include <cstddef>
#include <cstring>

// 算法常数

constexpr int GM_R1 = 13, GM_R2 = 19, GM_R3 = 23, GM_R4 = 29;
constexpr int GM_R5 = 7,  GM_R6 = 11, GM_R7 = 5,  GM_R8 = 15;
constexpr int GM_R9 = 17;  // 折回 XOR 用的旋转量
constexpr int HM_R1 = 9,  HM_R2 = 17, HM_R3 = 21, HM_R4 = 27;
constexpr int HM_R5 = 3,  HM_R6 = 11, HM_R7 = 7,  HM_R8 = 13;
constexpr int HM_R9 = 25;

constexpr int DEFAULT_ROUNDS = 16;       // 默认轮数
constexpr int ARX_ROUNDS = DEFAULT_ROUNDS;  // ARX 压缩函数轮数，与分组密码轮数一致
constexpr int MAX_ROUND_KEYS = (DEFAULT_ROUNDS + 2) * 4;  // 72 个 uint64

// CTR 多核并行阈值
constexpr int PARALLEL_THRESHOLD = 64 * 1024;  // 64KB 以上启用多核并行
constexpr int MIN_BLOCKS_PER_WORKER = 1024;    // 每 worker 至少 1024 块（32KB）
constexpr int MAX_WORKERS = 8;                 // 最多 8 个 worker

// 分组密码上下文

// 密钥扩展后的轮密钥
struct BlockCipher {
    int rounds;
    uint64_t roundKeys[MAX_ROUND_KEYS];
};

// 小端读写，memcpy 避免对齐未定义行为

inline uint32_t readU32LE(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

inline void writeU32LE(uint8_t* p, uint32_t v) {
    std::memcpy(p, &v, 4);
}

inline uint64_t readU64LE(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

inline void writeU64LE(uint8_t* p, uint64_t v) {
    std::memcpy(p, &v, 8);
}

// Bastion.cpp 中定义的函数

// 单块加解密（标量实现）
void blockEncrypt(const uint8_t* src, uint8_t* dst, const BlockCipher& c);
void blockDecrypt(const uint8_t* src, uint8_t* dst, const BlockCipher& c);

// 哈希压缩轮常数，inline constexpr 供标量与 SIMD 共用
inline constexpr uint32_t kRoundConstants[DEFAULT_ROUNDS][8] = {
    { 0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89 },
    { 0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C, 0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917 },
    { 0x9216D5D9, 0x8979FB1B, 0xD1310BA6, 0x98DFB5AC, 0x2FFD72DB, 0xD01ADFB7, 0xB8E1AFED, 0x6A267E96 },
    { 0xBA7C9045, 0xF12C7F99, 0x24A19947, 0xB3916CF7, 0x0801F2E2, 0x858EFC16, 0x636920D8, 0x71574E69 },
    { 0xA458FEA3, 0xF4933D7E, 0x0D95748F, 0x728EB658, 0x718BCD58, 0x82154AEE, 0x7B54A41D, 0xC25A59B5 },
    { 0x9C30D539, 0x2AF26013, 0xC5D1B023, 0x286085F0, 0xCA417918, 0xB8DB38EF, 0x8E79DCB0, 0x603A180E },
    { 0x6C9E0E8B, 0xB01E8A3E, 0xD71577C1, 0xBD314B27, 0x78AF2FDA, 0x55605C60, 0xE65525F3, 0xAA55AB94 },
    { 0x57489862, 0x63E81440, 0x55CA396A, 0x2AAB10B6, 0xB4CC5C34, 0x1141E8CE, 0xA15486AF, 0x7C72E993 },
    { 0xB3EE1411, 0x636FBC2A, 0x2BA9C55D, 0x741831F6, 0xCE5C3E16, 0x9B87931E, 0xAFD6BA33, 0x6C24CF5C },
    { 0x7A325381, 0x28958677, 0x3B8F4898, 0x6B4BB9AF, 0xC4BFE81B, 0x66282193, 0x61D809CC, 0xFB21A991 },
    { 0x487CAC60, 0x5DEC8032, 0xEF845D5D, 0xE98575B1, 0xDC262302, 0xEB651B88, 0x23893E81, 0xD396ACC5 },
    { 0x0F6D6FF3, 0x83F44239, 0x2E0B4482, 0xA4842004, 0x69C8F04A, 0x9E1F9B5E, 0x21C66842, 0xF6E96C9A },
    { 0x670C9C61, 0xABD388F0, 0x6A51A0D2, 0xD8542F68, 0x960FA728, 0xAB5133A3, 0x6EEF0B6C, 0x137A3BE4 },
    { 0xBA3BF050, 0x7EFB2A98, 0xA1F1651D, 0x39AF0176, 0x66CA593E, 0x82430E88, 0x8CEE8619, 0x456F9FB4 },
    { 0x7D84A5C3, 0x3B8B5EBE, 0xE06F75D8, 0x85C12073, 0x401A449F, 0x56C16AA6, 0x4ED3AA62, 0x363F7706 },
    { 0x1BFEDF72, 0x429B023D, 0x37D0D724, 0xD00A1248, 0xDB0FEAD3, 0x49F1C09B, 0x075372C9, 0x80991B7B },
};

// 末块封口常数，标记最后一块，阻断长度扩展
inline constexpr uint32_t kFinalConst[8] = {
    0x8F02C1BA, 0xB7ED2F35, 0x61711891, 0xDE7A9041, 0x758EDD75, 0x47708A47, 0x07B0E7A5, 0x4B91FB25,
};

// 4 块批量加密（标量实现，SIMD 回退用）
void encryptBlocksBatch4(const BlockCipher& c, const uint8_t* counter,
                         const uint8_t* in, uint8_t* out, int len);

// uint64 批量异或：out = in ^ key，len 是字节数
void xorWords(const uint8_t* in, const uint8_t* key, uint8_t* out, int len);

// Bastion_simd.cpp 中定义的函数

// SIMD 级别（0=不支持，1=SSE2，2=AVX2），进程启动时检测一次
extern const int g_simdLevel;

// 按 SIMD 级别分发整块加密（len 必须是 32 的整数倍）
void encryptBlocksDispatched(const BlockCipher& c, const uint8_t* counter,
                             const uint8_t* in, uint8_t* out, int len);

// 按 SIMD 级别分发从输入加载的批量加密（XTS 加密用）
void encryptBlocksFromInputDispatched(const BlockCipher& c,
                                      const uint8_t* in, uint8_t* out, int len);

// 按 SIMD 级别分发批量解密（CBC/XTS 解密用）
void decryptBlocksDispatched(const BlockCipher& c,
                             const uint8_t* in, uint8_t* out, int len);

// 4 条独立链并行哈希压缩（SSE2），stT 为状态字序，blk 为各链消息块，last=true 时注入封口常数
void arxCompress4xSSE2(uint32_t stT[8][4], const uint8_t* blk[4], bool last);

// CBC 解密一段连续块：解密 + XOR prev 合并，无中间缓冲
void cbcDecryptRange(const BlockCipher& c,
                     const uint8_t* in, uint8_t* out, int blocks,
                     const uint8_t* firstPrev);

// XTS 辅助函数

// 原地 XOR 一个 32 字节块
void xorBlock32(uint8_t* blk, const uint8_t* tweak);
// GF(2^256) 乘 2
void gfMul2_256(uint8_t x[32]);
// 处理一段连续完整块，从 startTweak 开始，处理完后把推进后的 tweak 写到 outTweak
void xtsProcessSegment(const BlockCipher& c1, const uint8_t* startTweak,
                       uint8_t* out, int blocks, bool encrypt, uint8_t* outTweak);