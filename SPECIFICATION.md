# Bastion 棱堡算法规范

**版本**: 1.0
**日期**: 2026-10-05


---

## 1. 概述

Bastion 棱堡是一种基于 SPN（代换-置换网络）结构设计的 256 位分组密码算法，融合 ARX（Add-Rotate-XOR）扩散组件与非循环仿射变换 S 盒。

### 1.1 设计目标

- **高性能**: 批量处理、多核并行、SIMD 硬件加速（SSE2/AVX2）、内存池优化
- **高安全性**: 256 位密钥、256 位分组、16 轮迭代，S 盒差分均匀度达 8 位双射函数差分均匀度的理论下界
- **多功能**: 支持分组密码（CTR/CBC/XTS 模式）、AEAD 认证加密、哈希、HMAC、密钥派生
- **跨平台**: C++、Go、JavaScript、Rust 四种语言实现，接口一致、测试向量互通

### 1.2 算法特点

| 特性 | 描述 |
|------|------|
| 结构 | SPN（代换-置换网络）+ ARX 扩散 |
| 状态大小 | 256位 (8×32位字) |
| 块大小 | 256位 (32字节) |
| 密钥长度 | 256位 (32字节) |
| Nonce长度 | 192位 (24字节) |
| AEAD标签 | 128位 (16字节) |
| 轮数 | 16轮（固定） |
| S 盒 | 4个8位S盒 S0..S3（GF(2^8)求逆+非循环仿射变换） |
| 字节序 | 小端序 (Little-Endian) |

---

## 2. 算法参数

### 2.1 常量定义

```c
#define BLOCK_SIZE         32       // 分组大小，256 位
#define KEY_SIZE           32       // 密钥长度，256 位
#define NONCE_SIZE         24       // nonce 长度，192 位
#define TAG_SIZE           16       // AEAD 认证标签长度，128 位
#define ROUNDS             16       // 分组密码默认轮数
#define ARX_ROUNDS         16       // ARX 哈希压缩函数轮数，与分组密码轮数一致
```

### 2.2 旋转常数

G_Mix 与 H_Mix 使用两套独立的旋转常数，主扩散与加法绑定互补配对：

```
G_Mix:  R1=13, R2=19, R3=23, R4=29   // 主扩散旋转
        R5=7,  R6=11, R7=5,  R8=15   // 加法绑定（与 R1..R4 互补）
        R9=17                          // 折回 XOR（s4→s0 短路反馈）
H_Mix:  R1=9,  R2=17, R3=21, R4=27
        R5=3,  R6=11, R7=7,  R8=13
        R9=25
```

### 2.3 S 盒构造

S 盒用于密钥扩展（主循环非线性来自模加法，不使用 S 盒），基于 GF(2^8) 有限域求逆与非循环仿射变换：

```
不可约多项式: x^8 + x^5 + x^3 + x^2 + 1 (0x12D)

基础S盒: S(x) = affine(inv(x))
  inv(x):  在 GF(2^8) 上求逆元（0 → 0，Fermat 小定理）
  affine:  非循环仿射矩阵变换（见 affineBastion）

派生S盒: S0 = S(x)         delta = 0x00
         S1 = S(x) ^ 0x5A
         S2 = S(x) ^ 0xB4
         S3 = S(x) ^ 0x2D
```

仿射变换输出位（y0 为最低位）：

```
y0 = b7 ^ b5 ^ b4 ^ b3 ^ b1 ^ 1
y1 = b7 ^ b6 ^ b4 ^ b3 ^ b2
y2 = b6 ^ b5 ^ b3 ^ b2 ^ b1 ^ 1
y3 = b5 ^ b4 ^ b2 ^ b1 ^ b0 ^ 1
y4 = b7 ^ b4 ^ b3 ^ b1 ^ b0
y5 = b7 ^ b6 ^ b3 ^ b2 ^ b0 ^ 1
y6 = b6 ^ b5 ^ b2 ^ b1 ^ b0
y7 = b7 ^ b5 ^ b4 ^ b0 ^ 1
```

4 个 S 盒按字节位置并行作用于 32 位字（sboxWordApply32），S 盒差分均匀度为 4/256，达到 8 位双射函数差分均匀度的理论下界。四个变体只在输出端异或固定常数 δ_k，差分分布表与线性近似表的数值与基础盒一致，差异仅体现在常数平移上：sboxWordApply32 等价于对 32 位字的四个字节统一施加基础盒后，再异或固定字常数 0x2DB45A00（字节 0..3 分别对应 δ = 0x00、0x5A、0xB4、0x2D）。

### 2.4 SubConstants 轮常数

16 轮 × 8 个 32 位固定轮常数，取自圆周率 π 的十六进制小数部分（π = 3.243F6A8885A308D3...），自小数点后连续截取 1024 个十六进制字符，每 8 个字符为一个 32 位字，共 16 轮 × 8 = 128 个字，全局固定、公开、确定：

```
subConstants[0] = { 0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89 }
subConstants[1] = { 0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C, 0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917 }
subConstants[2] = { 0x9216D5D9, 0x8979FB1B, 0xD1310BA6, 0x98DFB5AC, 0x2FFD72DB, 0xD01ADFB7, 0xB8E1AFED, 0x6A267E96 }
subConstants[3] = { 0xBA7C9045, 0xF12C7F99, 0x24A19947, 0xB3916CF7, 0x0801F2E2, 0x858EFC16, 0x636920D8, 0x71574E69 }
subConstants[4] = { 0xA458FEA3, 0xF4933D7E, 0x0D95748F, 0x728EB658, 0x718BCD58, 0x82154AEE, 0x7B54A41D, 0xC25A59B5 }
subConstants[5] = { 0x9C30D539, 0x2AF26013, 0xC5D1B023, 0x286085F0, 0xCA417918, 0xB8DB38EF, 0x8E79DCB0, 0x603A180E }
subConstants[6] = { 0x6C9E0E8B, 0xB01E8A3E, 0xD71577C1, 0xBD314B27, 0x78AF2FDA, 0x55605C60, 0xE65525F3, 0xAA55AB94 }
subConstants[7] = { 0x57489862, 0x63E81440, 0x55CA396A, 0x2AAB10B6, 0xB4CC5C34, 0x1141E8CE, 0xA15486AF, 0x7C72E993 }
subConstants[8] = { 0xB3EE1411, 0x636FBC2A, 0x2BA9C55D, 0x741831F6, 0xCE5C3E16, 0x9B87931E, 0xAFD6BA33, 0x6C24CF5C }
subConstants[9] = { 0x7A325381, 0x28958677, 0x3B8F4898, 0x6B4BB9AF, 0xC4BFE81B, 0x66282193, 0x61D809CC, 0xFB21A991 }
subConstants[10] = { 0x487CAC60, 0x5DEC8032, 0xEF845D5D, 0xE98575B1, 0xDC262302, 0xEB651B88, 0x23893E81, 0xD396ACC5 }
subConstants[11] = { 0x0F6D6FF3, 0x83F44239, 0x2E0B4482, 0xA4842004, 0x69C8F04A, 0x9E1F9B5E, 0x21C66842, 0xF6E96C9A }
subConstants[12] = { 0x670C9C61, 0xABD388F0, 0x6A51A0D2, 0xD8542F68, 0x960FA728, 0xAB5133A3, 0x6EEF0B6C, 0x137A3BE4 }
subConstants[13] = { 0xBA3BF050, 0x7EFB2A98, 0xA1F1651D, 0x39AF0176, 0x66CA593E, 0x82430E88, 0x8CEE8619, 0x456F9FB4 }
subConstants[14] = { 0x7D84A5C3, 0x3B8B5EBE, 0xE06F75D8, 0x85C12073, 0x401A449F, 0x56C16AA6, 0x4ED3AA62, 0x363F7706 }
subConstants[15] = { 0x1BFEDF72, 0x429B023D, 0x37D0D724, 0xD00A1248, 0xDB0FEAD3, 0x49F1C09B, 0x075372C9, 0x80991B7B }
```

用无理数小数做固定常数，可审计、无隐藏结构，符合主流算法惯例（如 BLAKE2 用 π 小数）。轮常数在密钥扩展时预合并进轮密钥，运行时无需再查找异或。

---

## 3. 数据结构

### 3.1 状态表示

算法内部状态由 8 个 32 位无符号整数组成：

```
状态: [s0, s1, s2, s3, s4, s5, s6, s7]
      每个 si 是 32位无符号整数 (uint32)
```

状态内存布局（小端序）：

```
字节偏移:  0  1  2  3   4  5  6  7   8  9 10 11  12 13 14 15
          +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
          |     s0      |     s1      |     s2      |     s3      |
          +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

字节偏移: 16 17 18 19  20 21 22 23  24 25 26 27  28 29 30 31
          +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
          |     s4      |     s5      |     s6      |     s7      |
          +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
```

### 3.2 密钥格式

密钥为 32 字节数组，按小端序读取为 8 个 32 位字：

```
K = [k0, k1, k2, ..., k31]
s_i = LE32(K[4i .. 4i+3])
```

### 3.3 Nonce格式

Nonce 为 24 字节数组，CTR 与 AEAD 共用：

```
N = [n0, n1, n2, ..., n23]
```

**重要**: Nonce 必须使用密码学安全的随机数生成器生成，且每个密钥只能使用一次。

---

## 4. 核心运算

### 4.1 G_Mix（偶数列混合）

G_Mix 作用于 8 个字状态（偶数通道 s0,s2,s4,s6 与奇数通道 s1,s3,s5,s7 同构并行），共 13 步：

```
// 偶数通道（奇数通道用 s1,s3,s5,s7 同样操作）
s0 += rotr(s2, R5)
s6 ^= s0;   s6 = rotr(s6, R1)
s4 += rotr(s6, R6)
s2 ^= s4;   s2 = rotr(s2, R2)
s0 ^= rotr(s4, R9)          // 折回XOR：s4→s0 短路反馈
s0 += rotr(s2, R7)
s6 ^= s0;   s6 = rotr(s6, R3)
s4 += rotr(s6, R8)
s2 ^= s4;   s2 = rotr(s2, R4)
```

其中 `+` 为模 2^32 加法，`rotr(x, n)` 为循环右移 n 位。

### 4.2 H_Mix（奇数列混合）

H_Mix 与 G_Mix 结构相同，仅旋转常数不同（使用 HM_R1..R9），与 G_Mix 在奇偶轮交替使用。

### 4.3 ShiftRows

轮内字置换操作：8 个字构成一个 8-循环，循环为 (0 2 7 1 5 4 3 6)，无不动点：

```
s0, s1, s2, s3, s4, s5, s6, s7 = s2, s5, s7, s6, s3, s4, s0, s1
```

等价写法（就地 8-循环）：

```
t = s0
s0 = s2; s2 = s7; s7 = s1; s1 = s5
s5 = s4; s4 = s3; s3 = s6; s6 = t
```

逆 ShiftRows 为上述置换的逆：

```
t = s0
s0 = s6; s6 = s3; s3 = s4; s4 = s5
s5 = s1; s1 = s7; s7 = s2; s2 = t
```

ShiftRows 不再自逆，解密需用逆置换。

### 4.4 轮函数

一轮完整变换包含三个步骤：

```
1. ColumnMix: 偶数轮用 G_Mix，奇数轮用 H_Mix
2. ShiftRows:  单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
3. AddRoundKey: s[i] ^= 轮密钥（SubConstants 已预合并）
```

### 4.5 逆轮函数

解密时按相反顺序执行逆操作：先逆 AddRoundKey，再逆 ShiftRows（8-循环置换的逆），最后执行 InvG_Mix/InvH_Mix。逆 ColumnMix 按正运算的 13 步逆序执行，加变减、旋转方向取反、XOR 自逆：

```
// InvG_Mix 偶数通道
s2 = rotl(s2, R4) ^ s4
s4 -= rotr(s6, R8)
s6 = rotl(s6, R3) ^ s0
s0 -= rotr(s2, R7)
s0 ^= rotr(s4, R9)          // 折回XOR逆运算（XOR自逆）
s2 = rotl(s2, R2) ^ s4
s4 -= rotr(s6, R6)
s6 = rotl(s6, R1) ^ s0
s0 -= rotr(s2, R5)
```

### 4.6 轮密钥生成

密钥扩展生成 144 个 32 位字（输入白化 8 + 16 轮 × 8 + 输出白化 8），三分支非线性变换：

```
w[0..7] = 密钥按小端序拆分的 8 个字

for i = 8 to 143:
    temp = w[i-1]
    if i % 8 == 0:                       // 周期边界：旋转+S盒
        temp = rotl(temp, 8)
        temp = sboxWordApply32(temp)     // S0..S3 并行替换
    else if i % 4 == 0:                  // 半周期：S盒+旋转+自加
        temp = sboxWordApply32(temp)
        temp = rotl(temp, 16)
        temp += rotl(temp, 7)
    else:                                // 普通步：xmx非线性
        temp = xmxFinalize(temp)
    w[i] = w[i-8] ^ temp
```

其中：

```
xmxFinalize(x):                                   // MurmurHash3 finalizer
    x ^= x >> 16
    x *= 0x45d9f3b
    x ^= x >> 16
    x *= 0x45d9f3b
    x ^= x >> 16
    return x
```

扩展完成后，将 SubConstants 逐轮预合并到轮密钥：

```
for r = 0 to 15:
    base = (r + 1) * 8      // 跳过输入白化 w[0..7]
    w[base .. base+7] ^= subConstants[r][0..7]
```

### 4.7 分组加密

```
输入: 明文块 P[32], 轮密钥 w[144]
输出: 密文块 C[32]

state = 小端序加载 P 为 8 个字

// 输入白化
state[i] ^= w[i]                    (i = 0..7)

// 16 轮迭代
for r = 0 to 15:
    ColumnMix(state, r)             // 偶数轮 G_Mix，奇数轮 H_Mix
    ShiftRows(state)
    state[i] ^= w[8 + r*8 + i]      (i = 0..7)

// 输出白化
state[i] ^= w[136 + i]              (i = 0..7)

C = 小端序存储 state
```

### 4.8 分组解密

解密按加密的严格逆序执行：先解除输出白化，再逐轮执行（逆 AddRoundKey → 逆 ShiftRows → InvColumnMix），最后解除输入白化。加解密互逆性可形式证明（13 步 ARX 路径逐条数据依赖验证）。

---

## 5. CTR 模式

### 5.1 计数器块构造

CTR 模式使用 24 字节 nonce 加内部 8 字节计数器（计数器不暴露到外部接口）：

```
counter块[32]:
    counter[0..23]  = nonce[0..23]        // 24字节 nonce
    counter[24..31] = 计数器 (uint64 小端序，从 0 开始)
```

### 5.2 密钥流生成

```
输入: 密钥 K, 计数器块 counter[32]
输出: 密钥流块 keystream[32]

对每个块:
    keystream[i] = BlockEncrypt(K, counter)
    counter 后 8 字节小端递增 1
```

### 5.3 加解密过程

CTR 加解密为同一操作（XOR 自反）：

```
C[i] = P[i] ^ keystream[i]
P[i] = C[i] ^ keystream[i]
```

性能优化：4 块批量加密 + 大数据量（≥64KB）多核并行，每 worker 至少 1024 块、最多 8 worker（C++/Go/Rust 原生实现；JavaScript 为单线程逐块处理）。

---

## 6. CBC 模式

### 6.1 加密

```
输入: 密钥 K, IV[32], 明文 P
输出: 密文 C

P = PKCS7Pad(P, 32)          // 填充到 32 字节整数倍
prevBlock = IV

for each 32-byte block of P:
    block = plaintextBlock XOR prevBlock
    ciphertextBlock = BlockEncrypt(K, block)
    output(ciphertextBlock)
    prevBlock = ciphertextBlock
```

### 6.2 解密

```
prevBlock = IV

for each 32-byte block of C:
    decryptedBlock = BlockDecrypt(K, ciphertextBlock)
    plaintextBlock = decryptedBlock XOR prevBlock
    output(plaintextBlock)
    prevBlock = ciphertextBlock

P = PKCS7Unpad(P, 32)        // 填充校验失败返回错误
```

### 6.3 PKCS#7 填充

填充值等于填充字节数（1..32），解密时常量时间校验填充。

性能优化：解密与 XOR prev 可合并执行（C++/JS 逐块合并），也可先批量解密到缓冲区再统一 XOR prev（Go/Rust）；大数据量（≥64KB）时 C++/Go/Rust 按块区间划分多核并行，各区间以前一块密文（或 IV）的快照作为起始 prev，结果与串行 XOR 链一致。

---

## 7. XTS 模式

### 7.1 GF(2^256) 乘法

XTS 使用 GF(2^256) 域上的乘 2 运算，约化多项式为 x^256 + x^10 + x^5 + x^2 + 1（字节级约化 0x0425），已形式验证不可约。

tweak 按大端序存储（byte[0] 为最高有效字节），乘 2 运算从左端（byte[0]）向右端（byte[31]）移位，进位从 byte[0] 的 MSB 捕获，溢出时在数组高位端异或约化多项式：

```
gfMul2_256(tweak[32]):
    carry = tweak[0] >> 7
    for i = 0 to 30:
        tweak[i] = (tweak[i] << 1) | (tweak[i+1] >> 7)
    tweak[31] <<= 1
    if carry != 0:
        tweak[30] ^= 0x04
        tweak[31] ^= 0x25
```

### 7.2 Tweak 生成

```
输入: 扇区号 sectorNum (uint64), tweak 密钥 key2
输出: 初始 tweak[32]

tweak = 32字节全零
tweak[24..31] = sectorNum (64位小端序)
tweak = BlockEncrypt(key2, tweak)
```

### 7.3 加密（含密文窃取）

XTS 使用两个独立密钥：key1 用于数据加密，key2 用于 tweak 加密。输入数据必须至少一个完整块（32 字节），不足一块按错误处理。

```
输入: key1, key2, 扇区号 sectorNum, 明文 P
输出: 密文 C

tweak = generateXTSTweak(sectorNum, key2)

if len(P) % 32 == 0:
    // 全部为完整块，逐个加密后 tweak 乘 2
    for i = 0 to numFullBlocks - 1:
        C[i] = BlockEncrypt(key1, P[i] ^ tweak) ^ tweak
        tweak = gfMul2_256(tweak)
else:
    // 密文窃取：最后一个完整块与部分块对调位置
    加密前 numFullBlocks-1 个完整块（同上）
    最后完整块加密后截取 remaining 字节放在密文尾部，
    部分块与最后一个完整块的剩余密文拼接后加密，
    整体结果写入密文倒数第二块位置
```

### 7.4 解密

按加密的逆过程执行，密文窃取时先解密部分块所在的组合块再还原最后完整块。加解密对合性已验证。

性能优化：tweak 链串行预计算后批量并行加解密，GF(2^256) 乘 2 在 SIMD 路径使用无分支掩码实现。

---

## 8. AEAD 认证加密

### 8.1 密钥派生

AEAD 使用 HKDF 从主密钥与 nonce 派生加密/认证子密钥，实现域分离与 nonce 绑定：

```
prk    ← HKDF-Extract(master, nonce)          // HMAC(nonce, master)
encKey ← HKDF-Expand(prk, "aead-key", 32)     // CTR 加密密钥
macKey ← HKDF-Expand(prk, "tag-key",  32)     // ARX HMAC 认证密钥
```

nonce 通过 HKDF salt 参与派生，不同 nonce 产生完全独立的子密钥对。

### 8.2 加密（Encrypt-then-MAC）

```
输入: 密钥 K, nonce N, 明文 P, 附加数据 AD
输出: 密文 C || 认证标签 Tag

encKey, macKey = deriveAEADKeys(K, N)

// 1. CTR 加密
C = CTREncrypt(encKey, N, P)

// 2. ARX HMAC 认证（输入依次写入）
mac = NewHMAC(macKey)
mac.update(AD)
mac.update(C)
mac.update(LE64(len(AD)) || LE64(len(C)))     // 双长度域
Tag = mac.finalize()[0..15]                   // 取前 16 字节

输出 C || Tag
```

### 8.3 解密

```
输入: 密钥 K, nonce N, 密文 C || Tag, 附加数据 AD
输出: 明文 P 或 验证失败

encKey, macKey = deriveAEADKeys(K, N)
expectedTag = 按 8.2 相同过程计算
if not constantTimeCompare(Tag, expectedTag):
    返回 验证失败

P = CTRDecrypt(encKey, N, C)
```

### 8.4 安全性

- Encrypt-then-MAC 构造可证明满足 IND-CPA + INT-CTXT（Bellare-Namprempre 定理）
- 认证标签 128 位，密钥域分离防止跨模式攻击
- 双长度域防止长度扩展攻击
- 标签比较与填充校验均为常量时间

---

## 9. 哈希函数

### 9.1 初始状态

ARX Hash 使用 SHA-256 风格素数平方根小数部分作为固定初始 IV：

```
arxInitialState[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
}
```

### 9.2 压缩函数

压缩函数复用 Bastion 的 G_Mix/H_Mix 组件（无 S 盒、无密钥扩展），固定 16 轮，每轮异或圆周率派生的轮常数做域分离，Davies-Meyer 前馈，末块把封口常数加进消息模加注入：

```
输入: 状态 state[8], 消息块 block[32], 最后一块标志 last
输出: 新状态

m[0..7] = 小端序读取 block
original = state

// 消息模加权入（末块把封口常数一并加进去）
s[i] = original[i] + m[i] + (last ? finalConstants[i] : 0)      (mod 2^32)

// 16 轮 ARX（偶数轮 G_Mix，奇数轮 H_Mix，含 ShiftRows）
for r = 0 to 15:
    s = arxRound(s, r)
    s[i] ^= roundConstants[r][i]   // 每轮异或圆周率派生的轮常数

// 状态加入：新状态与旧状态异或后模加消息块，破坏公开固定置换的代数逆
state[i] = (s[i] ^ original[i]) + m[i]
```

末块封口常数（π 派生的固定常数，紧邻轮常数之后）：
```
finalConstants[8] = {
    0x8F02C1BA, 0xB7ED2F35, 0x61711891, 0xDE7A9041,
    0x758EDD75, 0x47708A47, 0x07B0E7A5, 0x4B91FB25,
}
```

该结构在 Davies-Meyer 前馈基础上追加模加消息块，破坏公开固定置换的可逆代数链，状态不可逆。`last` 为真（最后一块）时，`finalConstants` 加在 16 轮 ARX 之前的吸收阶段，使"中间块"与"末块"的压缩之间隔着一次公开置换：由摘要反算可继续吸收数据的链值必须求逆该置换，因此拼接续块的长度扩展不成立。轮常数取自第 2.4 节 subConstants 常数表。

### 9.3 填充

Merkle-Damgård 填充（最小填充，至少 9 字节），最后一个填充块以 `last=true` 压缩（即把封口常数加进吸收阶段）：

```
M || 0x80 || 0x00...0x00 || LE64(bitlen)
```

填充块是消息的最后一个压缩块，必须带封口标志；若填充恰好跨两个块（极端情况下），仅最后一块带标志。

### 9.4 哈希计算

```
输入: 消息 M
输出: 哈希值 H[32]

state = arxInitialState
逐块执行压缩函数（中间块 last=false）
最终填充块以 last=true 压缩后输出 state 的小端序字节
```

### 9.5 XOF 可扩展输出

以哈希摘要作为密钥，用全零 nonce 的 CTR 模式对零块加密生成任意长度的扩展输出。

---

## 10. HMAC

基于 RFC 2104 结构，使用 ARX Hash，ipad/opad 直接异或到初始状态构造带密钥哈希：

```
输入: 密钥 K, 消息 M
输出: HMAC 值 [32]

if len(K) > 32:
    K = Hash(K)

ipad[i] = K[i] ^ 0x36
opad[i] = K[i] ^ 0x5C

// keyed IV 构造：ipad/opad 直接异或进初始状态，不经过压缩函数
inner.state = arxInitialState XOR LE32(ipad)
outer.state = arxInitialState XOR LE32(opad)

innerHash = hashWithState(inner, M)
outerHash = hashWithState(outer, innerHash)
返回 outerHash
```

安全性：ipad/opad 异或进初始状态，得到内外两把不同的链值密钥，目标安全强度与 HMAC 相当。该构造不把 pad 块送进压缩函数，因此不套用 NMAC/HMAC 的可证明归约，其 PRF 性质依赖「以链值为密钥的压缩函数是 PRF」这一假设。

---

## 11. 密钥派生

### 11.1 HKDF (RFC 5869)

```
输入: 输入密钥材料 IKM, 盐 salt, 信息 info, 输出长度 length
输出: 派生密钥

// 提取阶段
if salt 为空:
    salt = 32个零字节
PRK = HMAC(salt, IKM)

// 扩展阶段
N = ceil(length / 32)
T = 空
for i = 1 to N:
    T_i = HMAC(PRK, T_{i-1} || info || i)
    T = T || T_i
返回 T[0..length-1]
```

### 11.2 PBKDF2 (RFC 2898)

```
输入: 密码 password, 盐 salt, 迭代次数 iterations, 密钥长度 keyLen
输出: 派生密钥

N = ceil(keyLen / 32)
for i = 1 to N:
    U_1 = HMAC(password, salt || BE32(i))    // 块索引大端序，RFC 2898 兼容
    F = U_1
    for j = 2 to iterations:
        U_j = HMAC(password, U_{j-1})
        F = F XOR U_j
    DK = DK || F
返回 DK[0..keyLen-1]
```

### 11.3 DeriveKey（BLAKE3 风格）

```
输入: context 字符串, 密钥材料 keyMaterial, 输出长度 length
输出: 派生密钥

input = context || 0x00 || keyMaterial
baseKey = Hash(input)

if length <= 32:
    返回 baseKey[0..length-1]
else:
    // 确定性扩展：零 nonce 的 CTR 对零块加密
    返回 CTREncrypt(baseKey, 零nonce, 零块[length])
```

---

## 12. 测试向量

各语言实现共享相同的测试向量文件，验证跨语言一致性：

- [C++ 测试向量](CPP/test_vectors.json)
- [Go 测试向量](GO/test_vectors.json)
- [JavaScript 测试向量](JS/test_vectors.json)
- [Rust 测试向量](Rust/test_vectors.json)
---

## 13. 安全性分析

### 13.1 安全强度

| 参数 | 安全级别 |
|------|----------|
| 密钥长度 | 256位（抵抗暴力破解） |
| 分组长度 | 256位（抵抗生日攻击） |
| Nonce长度 | 192位（防止 Nonce 碰撞） |
| AEAD标签 | 128位（认证强度） |
| 轮数 | 16轮（差分/线性安全余量的严格下界尚未建立，见 13.3） |

### 13.2 设计原理

1. **SPN + ARX 混合结构**: G_Mix/H_Mix 中的模加法提供字级非线性，旋转与异或提供字级扩散；S 盒仅用于密钥扩展
2. **非循环仿射 S 盒（密钥扩展用）**: 打破循环矩阵的代数规律，S 盒差分均匀度 4/256，达 8 位双射函数差分均匀度的理论下界
3. **G/H 双 Mix 交替**: 奇偶轮使用不同旋转常数，破坏轮间同构，抑制单轮差分权重为 0 的路径跨轮累积
4. **折回 XOR 反馈**: s4→s0 短路反馈打破单向数据流，增加回旋镖攻击难度
5. **旋转加法绑定**: 旋转常数与加法绑定互补配对，降低单轮差分传播的零权重路径出现概率
6. **ARX 前馈 + 末块封口**: 哈希压缩函数异或旧状态后模加消息块，阻断公开固定置换的代数逆，保证单向性；末块把封口常数加进吸收阶段，使跨长度的压缩域分离，阻断长度扩展攻击
7. **HMAC keyed IV**: ipad/opad 直接异或初始状态，派生内外两把链值密钥
8. **AEAD Encrypt-then-MAC**: 可证明 IND-CPA + INT-CTXT，密钥域分离防跨模式攻击
9. **GF(2^256) 不可约多项式**: XTS 代数基础形式验证通过

### 13.3 安全性评估口径

| 攻击类型 | 16 轮分析状态 |
|----------|---------------|
| 差分攻击 | 无严格安全下界；全部单比特候选差分与多密钥深度采样的实测口径见密码学正确性分析报告（第 2 轮起无可观测信号，保守外推约 307 位） |
| 线性攻击 | 无严格安全下界；单次模加法的相关性已用逐比特动态规划精确计算，整条线性轨迹在异或相等约束下按候选掩码集确定性传播 |
| 旋转攻击 | 纯旋转区分器在有轮常数时实测存活率为 0（2^21 次采样）；128 个轮常数中对任意旋转量保持不变的有 0 个，互为旋转关系的常数对 0 个 |
| 不可能差分 | 字级截断模型下未发现覆盖 16 轮的区分器 |
| 积分攻击 | 字级统计探测（4096 采样点）显示区分器在第 2 轮起不再存活，属探测口径 |
| 代数攻击 | 代数次数上界 min(2^r, 32) 在第 5 轮饱和至 32，属理论上界口径 |
| 回旋镖攻击 | 实测覆盖：回旋镖 4~8 轮零命中、放大回旋镖 2~5 轮零命中（3 轮最强信号约 2^-8.40）；BCT / BDDT 精确建模留作后续 |

严格边界需要比特级 ARX 差分/线性 MILP 或 SAT 模型，本规范不引用启发式外推值作为安全边界。

关于旋转攻击：其防线是每轮异或的 π 轮常数。去掉轮常数后按攻击者最优旋转量 r = 1 计算，128 次模加的旋转轨迹权重约 181.1 位；实际每轮异或轮常数后旋转对称被破坏，区分器存活率实测归零。设计中的小旋转量（GM_R5=7、GM_R7=5、HM_R5=3、HM_R7=7）影响的是字内扩散快慢，不是旋转攻击的入口——攻击者的旋转量由自己选定，与设计常数无关。

### 13.4 使用建议

1. **Nonce 管理**: 每个密钥必须使用唯一 nonce，推荐密码学安全随机数生成器
2. **密钥派生**: 从口令派生密钥使用 PBKDF2，迭代次数建议 ≥ 10000
3. **XTS 用途**: XTS 适用于磁盘扇区加密，不适用于通用消息加密（无认证）
4. **常量时间**: 填充校验与标签比较均为常量时间实现，不泄露时序信息

### 13.5 已知限制

1. **16 轮差分/线性无严格下界**：现有结论来自启发式轨迹搜索与降轮实测，不能作为安全边界（见 13.3）。
2. **S 盒变体为常数平移**：四个 S 盒是同一基础盒的输出异或变形，字节位置之间没有差异化的非线性；S 盒仅用于密钥扩展，主循环的非线性来自模加法，因此该限制不影响主循环强度。

---

## 14. 参考实现

本项目提供四种语言的参考实现：

- **C++**: [CPP/BastionLib/](CPP/BastionLib/)
- **Go**: [GO/bastion/](GO/bastion/)
- **JavaScript**: [JS/Bastion.js](JS/Bastion.js)
- **Rust**: [Rust/BastionLib/](Rust/BastionLib/)

所有实现都遵循本规范，并通过了相同的测试向量验证。

---

*本规范遵循开源原则，欢迎社区审查和改进。*
