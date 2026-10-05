![Version](https://img.shields.io/badge/作者-风之暇想-blue.svg)   ![Languages](https://img.shields.io/badge/Languages-C++%20|%20Go%20|%20JavaScript%20|%20Rust-orange.svg)

![app.ico](app.ico)

**Bastion 棱堡**，一款高性能的自研对称加密算法库，提供 CTR 流加密、CBC、XTS、AEAD 认证加密、HASH、HMAC、HKDF、PBKDF2 模式。

---

### 概述

Bastion 棱堡是基于 SPN（代换-置换网络）结构设计的 256 位分组密码算法库，融合 ARX（Add-Rotate-XOR）扩散组件与非循环仿射变换 S 盒；加密库提供分组密码（16 轮，支持 CTR/CBC/XTS 模式）、AEAD 认证加密（Encrypt-then-MAC）、HASH、HMAC、HKDF 密钥派生、PBKDF2 密钥派生、DeriveKey 密钥派生的密码学功能。
### 特性

- **多种加密模式**：支持CTR 流加密、CBC（PKCS#7 填充）、XTS（磁盘加密，含密文窃取）、AEAD 认证加密
- **高性能设计**：批量处理（4 块批量）、多核并行（64KB 以上启用，最多 8 worker）、SIMD 硬件加速
- **跨平台支持**：提供 C++、Go、JavaScript、Rust 四种语言代码
- **丰富原语**：除分组密码外还提供哈希、HMAC、HKDF、PBKDF2、DeriveKey、XOF 可扩展输出

### 算法规范

[算法规范文档](SPECIFICATION.md)

---

## 库调用说明

### C++ 版本

**使用示例：**

```cpp
#include "BastionLib/Bastion.h"
#include <windows.h>
#include <bcrypt.h>
#include <cstring>
#include <cstdlib>
#include <vector>

// 用系统安全随机源填充缓冲区（BCryptGenRandom）
void random_bytes(uint8_t* buf, size_t len) {
    NTSTATUS st = BCryptGenRandom(nullptr, buf, (ULONG)len,
                                  BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (st != 0) abort(); // 系统随机源失败直接终止，避免用弱随机数
}

int main() {
    // 准备密钥和 Nonce（调用系统密码学安全随机源生成）
    uint8_t key[32];
    random_bytes(key, sizeof(key));
    uint8_t nonce[Bastion::NONCE_SIZE]; // 24字节
    random_bytes(nonce, sizeof(nonce));

    // ===== 1. 分组密码 - 单块加密/解密 =====
    uint8_t blockPlain[32] = { /* 32字节明文块 */ };
    uint8_t blockCipher[32];
    Bastion_BlockEncrypt(blockPlain, key, 32, blockCipher);
    uint8_t blockDecrypted[32];
    Bastion_BlockDecrypt(blockCipher, key, 32, blockDecrypted);

    // ===== 2. CTR流加密（加解密同一操作） =====
    std::vector<uint8_t> data = {'H', 'e', 'l', 'l', 'o'};
    std::vector<uint8_t> ctrOut(data.size());
    Bastion_CTRCryptBuffer(key, 32, nonce, 24, data.data(), data.size(), ctrOut.data());
    // ctrOut为密文，再调用一次即还原明文
    Bastion_CTRCryptBuffer(key, 32, nonce, 24, ctrOut.data(), ctrOut.size(), data.data());

    // ===== 3. CBC模式（PKCS#7填充） =====
    uint8_t iv[32];
    random_bytes(iv, sizeof(iv));
    std::vector<uint8_t> cbcOut(data.size() + 32);
    int cbcOutLen = (int)cbcOut.size();
    Bastion_CBCEncryptBuffer(key, 32, iv, 32, data.data(), data.size(), cbcOut.data(), &cbcOutLen);
    std::vector<uint8_t> cbcPlain(cbcOutLen);
    int cbcPlainLen = (int)cbcPlain.size();
    Bastion_CBCDecryptBuffer(key, 32, iv, 32, cbcOut.data(), cbcOutLen, cbcPlain.data(), &cbcPlainLen);

    // ===== 4. XTS模式（磁盘加密，支持密文窃取） =====
    uint8_t key2[32];
    random_bytes(key2, sizeof(key2));
    std::vector<uint8_t> xtsCipher(data.size());
    Bastion_XTSEncrypt(key, 32, key2, 32, 0, data.data(), data.size(), xtsCipher.data());
    std::vector<uint8_t> xtsPlain(data.size());
    Bastion_XTSDecrypt(key, 32, key2, 32, 0, xtsCipher.data(), xtsCipher.size(), xtsPlain.data());

    // ===== 5. AEAD认证加密（输出 ciphertext || tag） =====
    const uint8_t ad[] = "additional-data";
    uint8_t aeadOut[256];
    int aeadOutLen = sizeof(aeadOut);
    Bastion_AEADSeal(key, 32, nonce, 24, ad, sizeof(ad) - 1,
                     data.data(), data.size(), aeadOut, &aeadOutLen);
    uint8_t aeadPlain[256];
    int aeadPlainLen = sizeof(aeadPlain);
    Bastion_AEADOpen(key, 32, nonce, 24, ad, sizeof(ad) - 1,
                     aeadOut, aeadOutLen, aeadPlain, &aeadPlainLen);

    // ===== 6. HASH =====
    const char* message = "Hello, Bastion!";
    uint8_t hash[32];
    Bastion_Hash((const uint8_t*)message, strlen(message), hash);

    // ===== 7. HMAC =====
    uint8_t hmacKey[] = "secret-key";
    uint8_t hmacOut[32];
    Bastion_HMAC(hmacKey, sizeof(hmacKey) - 1,
                 (const uint8_t*)message, strlen(message), hmacOut);

    // ===== 8. HKDF密钥派生 =====
    uint8_t salt[32] = { /* 盐值 */ };
    uint8_t info[] = "my-app";
    uint8_t derivedKey[32];
    Bastion_HKDF(key, 32, salt, 32, info, sizeof(info) - 1, 32, derivedKey);

    // ===== 9. PBKDF2密钥派生 =====
    const char* password = "user-password";
    uint8_t pbkdf2Salt[] = "random-salt";
    uint8_t keyFromPassword[32];
    Bastion_PBKDF2((const uint8_t*)password, strlen(password),
                   pbkdf2Salt, sizeof(pbkdf2Salt) - 1,
                   10000, 32, keyFromPassword);

    // ===== 10. 密码学安全随机数（系统安全随机源） =====
    uint8_t randomBytes[64];
    random_bytes(randomBytes, sizeof(randomBytes));

    return 0;
}
```

---

### Go 版本

[https://pkg.go.dev/github.com/fzxx/Bastion/GO/bastion](https://pkg.go.dev/github.com/fzxx/Bastion/GO/bastion)

**使用示例：**

```go
package main

import (
    "bytes"
    "crypto/rand"
    "fmt"

    bastion "bastion/bastion"
)

// 从系统安全随机源取 n 字节，失败直接终止
func mustRandom(n int) []byte {
    b := make([]byte, n)
    if _, err := rand.Read(b); err != nil {
        panic(err)
    }
    return b
}

func main() {
    // 准备密钥和Nonce（调用系统安全随机源）
    key := mustRandom(32)
    nonce := mustRandom(24) // 24字节

    // ===== 1. 分组密码 - 单块加密/解密 =====
    blockPlain := make([]byte, 32)
    c, _ := bastion.NewCipher(key)
    blockCipher := make([]byte, 32)
    c.Encrypt(blockCipher, blockPlain)
    blockDecrypted := make([]byte, 32)
    c.Decrypt(blockDecrypted, blockCipher)
    fmt.Printf("Block decrypted match: %v\n", bytes.Equal(blockPlain, blockDecrypted))

    // ===== 2. CTR流加密（加解密同一操作） =====
    plaintext := []byte("Secret message")
    ciphertext, _ := bastion.CTREncrypt(key, nonce, plaintext)
    decrypted, _ := bastion.CTRDecrypt(key, nonce, ciphertext)
    fmt.Printf("Decrypted: %s\n", decrypted)

    // ===== 3. CBC模式（PKCS#7填充） =====
    iv := mustRandom(32)
    cbcCiphertext, _ := bastion.CBCEncryptBytes(key, iv, plaintext)
    cbcDecrypted, _ := bastion.CBCDecryptBytes(key, iv, cbcCiphertext)
    fmt.Printf("CBC Decrypted: %s\n", cbcDecrypted)

    // ===== 4. XTS模式（key为64字节：前32字节key1 + 后32字节key2） =====
    xtsKey := mustRandom(64)
    xtsCiphertext, _ := bastion.XTSEncrypt(xtsKey, plaintext, 0)
    xtsDecrypted, _ := bastion.XTSDecrypt(xtsKey, xtsCiphertext, 0)
    fmt.Printf("XTS Decrypted: %s\n", xtsDecrypted)

    // ===== 5. AEAD认证加密（输出 ciphertext || tag） =====
    aad := []byte("context info")
    sealed, _ := bastion.GCMSeal(key, nonce, plaintext, aad)
    opened, _ := bastion.GCMOpen(key, nonce, sealed, aad)
    fmt.Printf("AEAD Decrypted: %s\n", opened)

    // ===== 6. HASH =====
    message := []byte("Hello, Bastion!")
    hash := bastion.Sum(message)
    fmt.Printf("Hash: %x\n", hash)

    // ===== 7. HMAC =====
    hmacKey := []byte("secret-key")
    hmacResult := bastion.Compute(hmacKey, message)
    fmt.Printf("HMAC: %x\n", hmacResult)

    // ===== 8. HKDF密钥派生 =====
    salt := []byte("random-salt")
    info := []byte("my-app")
    derivedKey, _ := bastion.HKDF(key, salt, info, 32)
    fmt.Printf("Derived Key: %x\n", derivedKey)

    // ===== 9. PBKDF2密钥派生 =====
    password := []byte("user-password")
    pbkdf2Salt := []byte("random-salt")
    keyFromPassword, _ := bastion.PBKDF2(password, pbkdf2Salt, 10000, 32)
    fmt.Printf("Key from password: %x\n", keyFromPassword)

    // ===== 10. DeriveKey密钥派生（BLAKE3风格） =====
    derived := bastion.DeriveKey("my-app", key, 32)
    fmt.Printf("DeriveKey: %x\n", derived)

    // ===== 11. 密码学安全随机数（系统安全随机源） =====
    randomBytes := mustRandom(64)
    fmt.Printf("Random bytes: %x\n", randomBytes)
}
```

---

### JavaScript 版本

[https://fzxx.github.io/Bastion/](https://fzxx.github.io/Bastion/)

**CDN引用**


```
<script src="https://cdn.jsdelivr.net/gh/fzxx/Bastion/JS/Bastion.js"></script>
```

```
<script src="https://cdn.statically.io/gh/fzxx/Bastion@main/JS/Bastion.js"></script>
```

**使用示例：**

```javascript
// 浏览器环境
// <script src="Bastion.js"></script>

// Node.js环境
// const Bastion = require('./Bastion.js');
// const crypto = require('crypto').webcrypto; // Node.js 19以下无全局crypto，需这样显式引入

// 准备密钥和Nonce
const key = crypto.getRandomValues(new Uint8Array(32));
const nonce = crypto.getRandomValues(new Uint8Array(24));

// ===== 1. 分组密码 - 单块加密/解密 =====
const blockPlain = new Uint8Array(32); // 32字节明文块
const blockCipher = Bastion.blockEncrypt(key, blockPlain);
const blockDecrypted = Bastion.blockDecrypt(key, blockCipher);

// ===== 2. CTR流加密（加解密同一操作） =====
const plaintext = new TextEncoder().encode("Secret message");
const ciphertext = Bastion.ctrCrypt(key, nonce, plaintext);
const decrypted = Bastion.ctrCrypt(key, nonce, ciphertext);
console.log("Decrypted:", new TextDecoder().decode(decrypted));

// ===== 3. CBC模式（PKCS#7填充） =====
const iv = crypto.getRandomValues(new Uint8Array(32));
const cbcCiphertext = Bastion.cbcEncrypt(key, iv, plaintext);
const cbcDecrypted = Bastion.cbcDecrypt(key, iv, cbcCiphertext);
console.log("CBC Decrypted:", new TextDecoder().decode(cbcDecrypted));

// ===== 4. XTS模式（磁盘加密，支持密文窃取） =====
const key1 = crypto.getRandomValues(new Uint8Array(32)); // 数据加密密钥
const key2 = crypto.getRandomValues(new Uint8Array(32)); // Tweak密钥
const xtsCiphertext = Bastion.xtsEncrypt(key1, key2, 0, plaintext);
const xtsDecrypted = Bastion.xtsDecrypt(key1, key2, 0, xtsCiphertext);
console.log("XTS Decrypted:", new TextDecoder().decode(xtsDecrypted));

// ===== 5. AEAD认证加密（输出 ciphertext || tag） =====
const ad = new TextEncoder().encode("context info");
const aeadCiphertext = Bastion.aeadEncrypt(key, nonce, plaintext, ad);
const aeadDecrypted = Bastion.aeadDecrypt(key, nonce, aeadCiphertext, ad);
console.log("AEAD Decrypted:", new TextDecoder().decode(aeadDecrypted));

// ===== 6. HASH =====
const message = new TextEncoder().encode("Hello, Bastion!");
const hash = Bastion.hash(message);
console.log("Hash:", Array.from(hash).map(b => b.toString(16).padStart(2, '0')).join(''));

// ===== 7. HMAC =====
const hmacKey = new TextEncoder().encode("secret-key");
const hmacResult = Bastion.hmac(hmacKey, message);
console.log("HMAC:", Array.from(hmacResult).map(b => b.toString(16).padStart(2, '0')).join(''));

// ===== 8. HKDF密钥派生 =====
const salt = new TextEncoder().encode("random-salt");
const info = new TextEncoder().encode("my-app");
const derivedKey = Bastion.hkdf(key, salt, info, 32);
console.log("Derived Key:", Array.from(derivedKey).map(b => b.toString(16).padStart(2, '0')).join(''));

// ===== 9. PBKDF2密钥派生 =====
const password = new TextEncoder().encode("user-password");
const pbkdf2Salt = new TextEncoder().encode("random-salt");
const keyFromPassword = Bastion.pbkdf2(password, pbkdf2Salt, 10000, 32);
console.log("Key from password:", Array.from(keyFromPassword).map(b => b.toString(16).padStart(2, '0')).join(''));

// ===== 10. DeriveKey密钥派生（BLAKE3风格） =====
const derived = Bastion.deriveKey("my-app", key, 32);
console.log("DeriveKey:", Array.from(derived).map(b => b.toString(16).padStart(2, '0')).join(''));

// ===== 11. 密码学安全随机数（Web Crypto 标准随机源） =====
const randomBytes = crypto.getRandomValues(new Uint8Array(64));
console.log("Random bytes:", Array.from(randomBytes).map(b => b.toString(16).padStart(2, '0')).join(''));
```

---

### Rust 版本

**使用示例：**

```rust
use std::io::Cursor;
use getrandom::getrandom; // 系统安全随机源

fn main() -> std::io::Result<()> {
    // 准备密钥和Nonce（调用系统密码学安全随机数）
    let mut key = [0u8; bastion::KEY_SIZE];     // 32字节
    getrandom(&mut key).expect("RNG 失败");
    let mut nonce = [0u8; bastion::NONCE_SIZE]; // 24字节，CTR/AEAD要求严格24字节
    getrandom(&mut nonce).expect("RNG 失败");

    // ===== 1. 分组密码 - 单块加密/解密 =====
    let block_plain = [7u8; 32]; // 32字节明文块
    let cipher = bastion::BlockCipher::new(&key);
    let block_cipher = cipher.encrypt_block(&block_plain);
    let block_decrypted = cipher.decrypt_block(&block_cipher);
    assert_eq!(block_plain, block_decrypted);

    // ===== 2. CTR流加密（加解密同一操作，输入输出走流式） =====
    let plaintext = b"Secret message";
    let mut ciphertext = Vec::new();
    bastion::ctr_crypt(&key, &nonce, &mut Cursor::new(plaintext), &mut ciphertext)?;
    let mut decrypted = Vec::new();
    bastion::ctr_crypt(&key, &nonce, &mut Cursor::new(&ciphertext[..]), &mut decrypted)?;
    assert_eq!(plaintext, &decrypted[..]);

    // ===== 3. CBC模式（PKCS#7填充，IV固定32字节） =====
    let mut iv = [0u8; bastion::BLOCK_SIZE];
    getrandom(&mut iv).expect("RNG 失败");
    let mut cbc_ciphertext = Vec::new();
    bastion::cbc_encrypt(&key, &iv, &mut Cursor::new(plaintext), &mut cbc_ciphertext)?;
    let mut cbc_decrypted = Vec::new();
    bastion::cbc_decrypt(&key, &iv, &mut Cursor::new(&cbc_ciphertext[..]), &mut cbc_decrypted)?;
    assert_eq!(plaintext, &cbc_decrypted[..]);

    // ===== 4. XTS模式（磁盘加密，key为64字节：前32字节key1 + 后32字节key2，数据至少一个完整块32字节） =====
    let mut xts_key = [0u8; 64];
    getrandom(&mut xts_key).expect("RNG 失败");
    let sector_data = [42u8; 64];
    let xts_ciphertext = bastion::xts_encrypt(&xts_key, &sector_data, 0);
    let xts_decrypted = bastion::xts_decrypt(&xts_key, &xts_ciphertext, 0);
    assert_eq!(sector_data, &xts_decrypted[..]);

    // ===== 5. AEAD认证加密（输出 密文 || tag，tag固定16字节） =====
    let aad = b"context info";
    let mut sealed = Vec::new();
    bastion::aead_seal(&key, &nonce, &mut Cursor::new(plaintext), &mut sealed, aad)?;
    let mut opened = Vec::new();
    bastion::aead_open(&key, &nonce, &mut Cursor::new(&sealed[..]), &mut opened, aad)?;
    assert_eq!(plaintext, &opened[..]);

    // ===== 6. HASH =====
    let message = b"Hello, Bastion!";
    let hash = bastion::hash(&mut Cursor::new(message))?;

    // ===== 7. HMAC =====
    let hmac_key = b"secret-key";
    let hmac_result = bastion::hmac(hmac_key, &mut Cursor::new(message))?;

    // ===== 8. HKDF密钥派生 =====
    let salt = b"random-salt";
    let info = b"my-app";
    let derived_key = bastion::hkdf(&key, salt, info, 32);

    // ===== 9. PBKDF2密钥派生 =====
    let password = b"user-password";
    let pbkdf2_salt = b"random-salt";
    let key_from_password = bastion::pbkdf2(password, pbkdf2_salt, 10000, 32);

    // ===== 10. DeriveKey密钥派生（BLAKE3风格） =====
    let derived = bastion::derive_key("my-app", &key, 32);

    // ===== 11. 密码学安全随机数（系统安全随机源） =====
    let mut random_bytes = [0u8; 64];
    getrandom(&mut random_bytes).expect("RNG 失败");

    Ok(())
}
```

---

## 📖 更新日志

[更新日志](CHANGELOG.md)

## 许可证

[GNU General Public License 3.0](https://github.com/fzxx/Bastion/blob/main/LICENSE)
