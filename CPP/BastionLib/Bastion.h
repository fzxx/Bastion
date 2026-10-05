#pragma once

#include <cstdint>
#include <cstddef>

// 编译库时定义 BASTIONLIB_EXPORTS 导出符号
#if defined(_WIN32) || defined(_WIN64)
    #ifdef BASTIONLIB_EXPORTS
        #define BASTION_API __declspec(dllexport)
    #else
        #define BASTION_API __declspec(dllimport)
    #endif
#else
    #define BASTION_API
#endif

namespace Bastion {
    constexpr size_t BLOCK_SIZE = 32;   // 分组大小，256 位
    constexpr size_t KEY_SIZE = 32;     // 密钥固定 32 字节
    constexpr size_t NONCE_SIZE = 24;   // nonce，CTR/AEAD 都用这个，192 位
    constexpr size_t TAG_SIZE = 16;     // AEAD 认证标签，128 位
    constexpr int ROUNDS = 16;          // 默认轮数，别随便动
}

// 读回调返回实际读到的字节数，0 表示没数据了，-1 表示出错
typedef int (*ReadCallback)(void* ctx, uint8_t* buf, int len);
// 写回调返回实际写入的字节数，-1 表示出错
typedef int (*WriteCallback)(void* ctx, const uint8_t* buf, int len);

#ifdef __cplusplus
extern "C" {
#endif

// 分组密码

// 单块加密，输入输出各 32 字节
BASTION_API int Bastion_BlockEncrypt(const uint8_t* plaintext,
                                     const uint8_t* key, int keyLen,
                                     uint8_t* ciphertext);

// 单块解密
BASTION_API int Bastion_BlockDecrypt(const uint8_t* ciphertext,
                                     const uint8_t* key, int keyLen,
                                     uint8_t* plaintext);

// CTR 模式

// CTR 加解密，加解密是同一个操作，计数器从 0 开始递增
BASTION_API int Bastion_CTRCrypt(const uint8_t* key, int keyLen,
                                 const uint8_t* nonce, int nonceLen,
                                 ReadCallback readCb, void* readCtx,
                                 WriteCallback writeCb, void* writeCtx);

// 内存版 CTR，方便一次性处理小块数据
BASTION_API int Bastion_CTRCryptBuffer(const uint8_t* key, int keyLen,
                                       const uint8_t* nonce, int nonceLen,
                                       const uint8_t* in, int inLen,
                                       uint8_t* out);

// CBC 模式

// CBC 加密，PKCS#7 填充，iv 是 32 字节
BASTION_API int Bastion_CBCEncrypt(const uint8_t* key, int keyLen,
                                   const uint8_t* iv, int ivLen,
                                   ReadCallback readCb, void* readCtx,
                                   WriteCallback writeCb, void* writeCtx);

// CBC 解密，会校验填充
BASTION_API int Bastion_CBCDecrypt(const uint8_t* key, int keyLen,
                                   const uint8_t* iv, int ivLen,
                                   ReadCallback readCb, void* readCtx,
                                   WriteCallback writeCb, void* writeCtx);

// 内存版 CBC 加密，输出长度 = 明文长度向上取整到块 + 一个完整填充块
BASTION_API int Bastion_CBCEncryptBuffer(const uint8_t* key, int keyLen,
                                         const uint8_t* iv, int ivLen,
                                         const uint8_t* plaintext, int plaintextLen,
                                         uint8_t* out, int* outLen);

// 内存版 CBC 解密，返回明文长度，填充不对返回 -1
BASTION_API int Bastion_CBCDecryptBuffer(const uint8_t* key, int keyLen,
                                         const uint8_t* iv, int ivLen,
                                         const uint8_t* ciphertext, int ciphertextLen,
                                         uint8_t* plaintext, int* plaintextLen);

// XTS 模式

// XTS 加密，磁盘加密用，支持密文窃取；key1 数据密钥，key2 tweak 密钥
BASTION_API int Bastion_XTSEncrypt(const uint8_t* key1, int key1Len,
                                   const uint8_t* key2, int key2Len,
                                   uint64_t sectorNum,
                                   const uint8_t* plaintext, int plaintextLen,
                                   uint8_t* ciphertext);

// XTS 解密
BASTION_API int Bastion_XTSDecrypt(const uint8_t* key1, int key1Len,
                                   const uint8_t* key2, int key2Len,
                                   uint64_t sectorNum,
                                   const uint8_t* ciphertext, int ciphertextLen,
                                   uint8_t* plaintext);

// AEAD 模式

// AEAD 加密，输出 ciphertext || tag
BASTION_API int Bastion_AEADSeal(const uint8_t* key, int keyLen,
                                 const uint8_t* nonce, int nonceLen,
                                 const uint8_t* ad, int adLen,
                                 const uint8_t* plaintext, int plaintextLen,
                                 uint8_t* out, int* outLen);

// AEAD 解密，输入是 ciphertext || tag，验证不过返回 1
BASTION_API int Bastion_AEADOpen(const uint8_t* key, int keyLen,
                                 const uint8_t* nonce, int nonceLen,
                                 const uint8_t* ad, int adLen,
                                 const uint8_t* ciphertext, int ciphertextLen,
                                 uint8_t* plaintext, int* plaintextLen);

// 流式 AEAD 加密，密文写到 writeCb，tag 通过 outTag 返回
BASTION_API int Bastion_AEADSealStream(const uint8_t* key, int keyLen,
                                       const uint8_t* nonce, int nonceLen,
                                       const uint8_t* ad, int adLen,
                                       ReadCallback readCb, void* readCtx,
                                       WriteCallback writeCb, void* writeCtx,
                                       uint8_t outTag[Bastion::TAG_SIZE]);

// 哈希与消息认证

// ARX 哈希，32 字节摘要
BASTION_API int Bastion_Hash(const uint8_t* input, int64_t inputLen,
                             uint8_t out[Bastion::BLOCK_SIZE]);

// 流式哈希
BASTION_API int Bastion_HashStream(ReadCallback readCb, void* ctx,
                                   uint8_t out[Bastion::BLOCK_SIZE]);

// 可变长度输出 XOF，length 指定输出字节数
BASTION_API int Bastion_XOF(const uint8_t* input, int64_t inputLen,
                            int length, uint8_t* out);

// HMAC，32 字节输出
BASTION_API int Bastion_HMAC(const uint8_t* key, int keyLen,
                             const uint8_t* data, int64_t dataLen,
                             uint8_t out[Bastion::BLOCK_SIZE]);

// 密钥派生

// HKDF，length 是输出长度
BASTION_API int Bastion_HKDF(const uint8_t* masterKey, int masterKeyLen,
                             const uint8_t* salt, int saltLen,
                             const uint8_t* info, int infoLen,
                             int length, uint8_t* out);

// PBKDF2，iterations 是迭代次数
BASTION_API int Bastion_PBKDF2(const uint8_t* password, int passwordLen,
                               const uint8_t* salt, int saltLen,
                               int iterations, int keyLength,
                               uint8_t* out);

// DeriveKey，context 区分不同用途的派生结果
BASTION_API int Bastion_DeriveKey(const char* context,
                                  const uint8_t* keyMaterial, int keyMaterialLen,
                                  int length, uint8_t* out);

#ifdef __cplusplus
}
#endif
