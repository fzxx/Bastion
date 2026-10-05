//! Bastion 棱堡加密算法库（Rust 实现）
//! 跟 Go 版逐行对应翻译，保证测试向量一致

use std::cell::RefCell;
use std::io::{self, Cursor, Read, Write};
use std::sync::mpsc;
use std::sync::Mutex;

// 基础常量

pub const BLOCK_SIZE: usize = 32;
pub const KEY_SIZE: usize = 32;
pub const HASH_SIZE: usize = 32;
pub const HASH_BLOCK_SIZE: usize = 32;
pub const HMAC_BLOCK_SIZE: usize = 32;
pub const KDF_HASH_LEN: usize = 32;
pub const CTR_COUNTER_SIZE: usize = 24;
pub const NONCE_SIZE: usize = 24;
pub const GCM_TAG_SIZE: usize = 16;
pub const DEFAULT_ROUNDS: usize = 16;
pub const MAX_ROUND_KEYS: usize = (DEFAULT_ROUNDS + 2) * 4; // 72 个 uint64，这个数够装所有轮密钥了
pub const ARX_ROUNDS: usize = DEFAULT_ROUNDS; // ARX 压缩函数轮数，与分组密码轮数一致

// GM/HM 列混淆用的旋转位数
pub(crate) const GM_R1: u32 = 13;
pub(crate) const GM_R2: u32 = 19;
pub(crate) const GM_R3: u32 = 23;
pub(crate) const GM_R4: u32 = 29;
pub(crate) const GM_R5: u32 = 7;
pub(crate) const GM_R6: u32 = 11;
pub(crate) const GM_R7: u32 = 5;
pub(crate) const GM_R8: u32 = 15;
pub(crate) const GM_R9: u32 = 17;

pub(crate) const HM_R1: u32 = 9;
pub(crate) const HM_R2: u32 = 17;
pub(crate) const HM_R3: u32 = 21;
pub(crate) const HM_R4: u32 = 27;
pub(crate) const HM_R5: u32 = 3;
pub(crate) const HM_R6: u32 = 11;
pub(crate) const HM_R7: u32 = 7;
pub(crate) const HM_R8: u32 = 13;
pub(crate) const HM_R9: u32 = 25;

// SIMD 加速模块（x86_64 平台）
#[cfg(target_arch = "x86_64")]
pub(crate) mod simd;

// 流式处理缓冲大小，跟 C++/Go/闪旋 对齐（4MB）
const STREAM_BUFFER: usize = 4 << 20;

// AEAD 分块缓冲大小（1MB）：够流水线重叠用，又不用每次调用都清零几 MB
const SEAL_SLOT_SIZE: usize = 1 << 20;

// 流式缓冲按线程复用：几 MB 的临时缓冲只建一次，之后同一线程反复用，
// 省掉每次调用重新分配并清零几 MB 的开销，小消息时这块比加解密本身还费时间
thread_local! {
    static STREAM_BUF_A: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static STREAM_BUF_B: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

// 借出两个流式缓冲给闭包用；缓冲不够长就按需扩容，之后不再动
fn with_stream_bufs<R>(f: impl FnOnce(&mut [u8], &mut [u8]) -> R) -> R {
    STREAM_BUF_A.with(|a| {
        STREAM_BUF_B.with(|b| {
            let mut a = a.borrow_mut();
            let mut b = b.borrow_mut();
            if a.len() < STREAM_BUFFER {
                a.resize(STREAM_BUFFER, 0);
            }
            if b.len() < STREAM_BUFFER {
                b.resize(STREAM_BUFFER, 0);
            }
            f(&mut a[..], &mut b[..])
        })
    })
}

// 借出 A 缓冲，只要一块缓冲的（哈希、AEAD 读缓冲）用它
fn with_stream_buf_a<R>(f: impl FnOnce(&mut [u8]) -> R) -> R {
    STREAM_BUF_A.with(|a| {
        let mut a = a.borrow_mut();
        if a.len() < STREAM_BUFFER {
            a.resize(STREAM_BUFFER, 0);
        }
        f(&mut a[..])
    })
}

// 借出 B 缓冲，HMAC 用它；HMAC 内部遇到超长密钥会再调哈希（用 A 缓冲），两块分开不会互相顶掉
fn with_stream_buf_b<R>(f: impl FnOnce(&mut [u8]) -> R) -> R {
    STREAM_BUF_B.with(|b| {
        let mut b = b.borrow_mut();
        if b.len() < STREAM_BUFFER {
            b.resize(STREAM_BUFFER, 0);
        }
        f(&mut b[..])
    })
}

/// ARX 哈希初始 IV，直接拿 SHA-256 的初始值来用
const ARX_INITIAL_STATE: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

// GF(2^8) 与 S 盒

const GF8_POLY: u8 = 0x2D;

#[inline]
const fn gf8_mul(a: u8, b: u8) -> u8 {
    let mut p = 0u8;
    let mut a = a;
    let mut b = b;
    let mut i = 0;
    while i < 8 {
        if b & 1 != 0 {
            p ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= GF8_POLY;
        }
        b >>= 1;
        i += 1;
    }
    p
}

#[inline]
const fn gf8_inv(b: u8) -> u8 {
    if b == 0 {
        return 0;
    }
    // 用 Fermat 小定理求逆：a^254 = a^(-1)，因为 GF(2^8) 里 a^255 = 1
    let mut result = 1u8;
    let mut base = b;
    let mut exp = 254u8;
    while exp > 0 {
        if exp & 1 != 0 {
            result = gf8_mul(result, base);
        }
        base = gf8_mul(base, base);
        exp >>= 1;
    }
    result
}

#[inline]
const fn affine_bastion(x: u8) -> u8 {
    let b0 = x & 1;
    let b1 = (x >> 1) & 1;
    let b2 = (x >> 2) & 1;
    let b3 = (x >> 3) & 1;
    let b4 = (x >> 4) & 1;
    let b5 = (x >> 5) & 1;
    let b6 = (x >> 6) & 1;
    let b7 = (x >> 7) & 1;

    let y0 = b7 ^ b5 ^ b4 ^ b3 ^ b1 ^ 1;
    let y1 = b7 ^ b6 ^ b4 ^ b3 ^ b2;
    let y2 = b6 ^ b5 ^ b3 ^ b2 ^ b1 ^ 1;
    let y3 = b5 ^ b4 ^ b2 ^ b1 ^ b0 ^ 1;
    let y4 = b7 ^ b4 ^ b3 ^ b1 ^ b0;
    let y5 = b7 ^ b6 ^ b3 ^ b2 ^ b0 ^ 1;
    let y6 = b6 ^ b5 ^ b2 ^ b1 ^ b0;
    let y7 = b7 ^ b5 ^ b4 ^ b0 ^ 1;

    (y0 & 1)
        | ((y1 & 1) << 1)
        | ((y2 & 1) << 2)
        | ((y3 & 1) << 3)
        | ((y4 & 1) << 4)
        | ((y5 & 1) << 5)
        | ((y6 & 1) << 6)
        | ((y7 & 1) << 7)
}

pub const SBOX: [u8; 256] = build_sbox();

const fn build_sbox() -> [u8; 256] {
    let mut s = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        s[i] = affine_bastion(gf8_inv(i as u8));
        i += 1;
    }
    s
}

// 旋转

#[inline]
fn rotl32(x: u32, n: usize) -> u32 {
    x.rotate_left(n as u32)
}

#[inline]
fn rotr32(x: u32, n: usize) -> u32 {
    x.rotate_right(n as u32)
}

// S 盒转成 T 表，32 位字查表用

type SBoxTables = ([u32; 256], [u32; 256], [u32; 256], [u32; 256]);

fn build_sbox_tables() -> SBoxTables {
    let mut s0 = [0u32; 256];
    let mut s1 = [0u32; 256];
    let mut s2 = [0u32; 256];
    let mut s3 = [0u32; 256];
    for i in 0..256u32 {
        let b = i as u8;
        let v = SBOX[b as usize] as u32;
        s0[i as usize] = v;
        s1[i as usize] = (v ^ 0x5A) << 8;
        s2[i as usize] = (v ^ 0xB4) << 16;
        s3[i as usize] = (v ^ 0x2D) << 24;
    }
    (s0, s1, s2, s3)
}

use std::sync::OnceLock;
static SBOX_TABLES: OnceLock<SBoxTables> = OnceLock::new();

fn sbox_tables() -> &'static SBoxTables {
    SBOX_TABLES.get_or_init(build_sbox_tables)
}

#[inline]
fn sbox_word_apply32(v: u32) -> u32 {
    let (s0, s1, s2, s3) = sbox_tables();
    s3[(v >> 24) as usize]
        | s2[((v >> 16) & 0xFF) as usize]
        | s1[((v >> 8) & 0xFF) as usize]
        | s0[(v & 0xFF) as usize]
}

// 圆周率 π 派生的固定轮常数，取自 π 十六进制小数部分连续截取，每 8 个字符一个 32 位字
const ROUND_CONSTANTS: [[u32; 8]; DEFAULT_ROUNDS] = [
    [0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89],
    [0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C, 0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917],
    [0x9216D5D9, 0x8979FB1B, 0xD1310BA6, 0x98DFB5AC, 0x2FFD72DB, 0xD01ADFB7, 0xB8E1AFED, 0x6A267E96],
    [0xBA7C9045, 0xF12C7F99, 0x24A19947, 0xB3916CF7, 0x0801F2E2, 0x858EFC16, 0x636920D8, 0x71574E69],
    [0xA458FEA3, 0xF4933D7E, 0x0D95748F, 0x728EB658, 0x718BCD58, 0x82154AEE, 0x7B54A41D, 0xC25A59B5],
    [0x9C30D539, 0x2AF26013, 0xC5D1B023, 0x286085F0, 0xCA417918, 0xB8DB38EF, 0x8E79DCB0, 0x603A180E],
    [0x6C9E0E8B, 0xB01E8A3E, 0xD71577C1, 0xBD314B27, 0x78AF2FDA, 0x55605C60, 0xE65525F3, 0xAA55AB94],
    [0x57489862, 0x63E81440, 0x55CA396A, 0x2AAB10B6, 0xB4CC5C34, 0x1141E8CE, 0xA15486AF, 0x7C72E993],
    [0xB3EE1411, 0x636FBC2A, 0x2BA9C55D, 0x741831F6, 0xCE5C3E16, 0x9B87931E, 0xAFD6BA33, 0x6C24CF5C],
    [0x7A325381, 0x28958677, 0x3B8F4898, 0x6B4BB9AF, 0xC4BFE81B, 0x66282193, 0x61D809CC, 0xFB21A991],
    [0x487CAC60, 0x5DEC8032, 0xEF845D5D, 0xE98575B1, 0xDC262302, 0xEB651B88, 0x23893E81, 0xD396ACC5],
    [0x0F6D6FF3, 0x83F44239, 0x2E0B4482, 0xA4842004, 0x69C8F04A, 0x9E1F9B5E, 0x21C66842, 0xF6E96C9A],
    [0x670C9C61, 0xABD388F0, 0x6A51A0D2, 0xD8542F68, 0x960FA728, 0xAB5133A3, 0x6EEF0B6C, 0x137A3BE4],
    [0xBA3BF050, 0x7EFB2A98, 0xA1F1651D, 0x39AF0176, 0x66CA593E, 0x82430E88, 0x8CEE8619, 0x456F9FB4],
    [0x7D84A5C3, 0x3B8B5EBE, 0xE06F75D8, 0x85C12073, 0x401A449F, 0x56C16AA6, 0x4ED3AA62, 0x363F7706],
    [0x1BFEDF72, 0x429B023D, 0x37D0D724, 0xD00A1248, 0xDB0FEAD3, 0x49F1C09B, 0x075372C9, 0x80991B7B],
];

// 末块封口常数（π 派生），标记最后一块，阻断长度扩展
const FINAL_CONST: [u32; 8] = [
    0x8F02C1BA, 0xB7ED2F35, 0x61711891, 0xDE7A9041, 0x758EDD75, 0x47708A47, 0x07B0E7A5, 0x4B91FB25,
];

/// XOR-Multiply-XOR，参考 MurmurHash3 那个 finalizer，用来搅匀 u32
fn xmx_finalize(x: u32) -> u32 {
    let mut x = x;
    x ^= x >> 16;
    x = x.wrapping_mul(0x045D_9F3B);
    x ^= x >> 16;
    x = x.wrapping_mul(0x045D_9F3B);
    x ^= x >> 16;
    x
}

// 字级工具：加载/存储 32 字节块到 8 个 u32

#[inline]
pub(crate) fn load_words(block: &[u8]) -> [u32; 8] {
    let mut w = [0u32; 8];
    for i in 0..8 {
        w[i] = u32::from_le_bytes([
            block[4 * i],
            block[4 * i + 1],
            block[4 * i + 2],
            block[4 * i + 3],
        ]);
    }
    w
}

#[inline]
pub(crate) fn store_words(w: &[u32; 8]) -> [u8; BLOCK_SIZE] {
    let mut out = [0u8; BLOCK_SIZE];
    for i in 0..8 {
        out[4 * i..4 * i + 4].copy_from_slice(&w[i].to_le_bytes());
    }
    out
}

// GM/HM 列混淆，跟块密码轮函数共用

/// 正向 GM 列混淆，偶数轮用
fn g_mix8(a: &mut [u32; 8]) {
    let s0 = a[0];
    let s1 = a[1];
    let s2 = a[2];
    let s3 = a[3];
    let s4 = a[4];
    let s5 = a[5];

    a[0] = s0.wrapping_add(rotr32(s2, GM_R5 as usize));
    a[1] = s1.wrapping_add(rotr32(s3, GM_R5 as usize));
    a[6] ^= a[0];
    a[7] ^= a[1];
    a[6] = rotr32(a[6], GM_R1 as usize);
    a[7] = rotr32(a[7], GM_R1 as usize);
    a[4] = s4.wrapping_add(rotr32(a[6], GM_R6 as usize));
    a[5] = s5.wrapping_add(rotr32(a[7], GM_R6 as usize));
    a[2] ^= a[4];
    a[3] ^= a[5];
    a[2] = rotr32(a[2], GM_R2 as usize);
    a[3] = rotr32(a[3], GM_R2 as usize);
    a[0] ^= rotr32(a[4], GM_R9 as usize);
    a[1] ^= rotr32(a[5], GM_R9 as usize);
    a[0] = a[0].wrapping_add(rotr32(a[2], GM_R7 as usize));
    a[1] = a[1].wrapping_add(rotr32(a[3], GM_R7 as usize));
    a[6] ^= a[0];
    a[7] ^= a[1];
    a[6] = rotr32(a[6], GM_R3 as usize);
    a[7] = rotr32(a[7], GM_R3 as usize);
    a[4] = a[4].wrapping_add(rotr32(a[6], GM_R8 as usize));
    a[5] = a[5].wrapping_add(rotr32(a[7], GM_R8 as usize));
    a[2] ^= a[4];
    a[3] ^= a[5];
    a[2] = rotr32(a[2], GM_R4 as usize);
    a[3] = rotr32(a[3], GM_R4 as usize);
}

/// 正向 HM 列混淆，奇数轮用
fn h_mix8(a: &mut [u32; 8]) {
    let s0 = a[0];
    let s1 = a[1];
    let s2 = a[2];
    let s3 = a[3];
    let s4 = a[4];
    let s5 = a[5];

    a[0] = s0.wrapping_add(rotr32(s2, HM_R5 as usize));
    a[1] = s1.wrapping_add(rotr32(s3, HM_R5 as usize));
    a[6] ^= a[0];
    a[7] ^= a[1];
    a[6] = rotr32(a[6], HM_R1 as usize);
    a[7] = rotr32(a[7], HM_R1 as usize);
    a[4] = s4.wrapping_add(rotr32(a[6], HM_R6 as usize));
    a[5] = s5.wrapping_add(rotr32(a[7], HM_R6 as usize));
    a[2] ^= a[4];
    a[3] ^= a[5];
    a[2] = rotr32(a[2], HM_R2 as usize);
    a[3] = rotr32(a[3], HM_R2 as usize);
    a[0] ^= rotr32(a[4], HM_R9 as usize);
    a[1] ^= rotr32(a[5], HM_R9 as usize);
    a[0] = a[0].wrapping_add(rotr32(a[2], HM_R7 as usize));
    a[1] = a[1].wrapping_add(rotr32(a[3], HM_R7 as usize));
    a[6] ^= a[0];
    a[7] ^= a[1];
    a[6] = rotr32(a[6], HM_R3 as usize);
    a[7] = rotr32(a[7], HM_R3 as usize);
    a[4] = a[4].wrapping_add(rotr32(a[6], HM_R8 as usize));
    a[5] = a[5].wrapping_add(rotr32(a[7], HM_R8 as usize));
    a[2] ^= a[4];
    a[3] ^= a[5];
    a[2] = rotr32(a[2], HM_R4 as usize);
    a[3] = rotr32(a[3], HM_R4 as usize);
}

// ShiftRows：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
fn shift_rows(a: &mut [u32; 8]) {
    let t = a[0]; a[0] = a[2]; a[2] = a[7]; a[7] = a[1];
    a[1] = a[5]; a[5] = a[4]; a[4] = a[3]; a[3] = a[6]; a[6] = t;
}

// 逆 ShiftRows：8-循环置换的逆
fn inv_shift_rows(a: &mut [u32; 8]) {
    let t = a[0]; a[0] = a[6]; a[6] = a[3]; a[3] = a[4];
    a[4] = a[5]; a[5] = a[1]; a[1] = a[7]; a[7] = a[2]; a[2] = t;
}

// 块密码

pub struct BlockCipher {
    pub(crate) round_keys: [u32; MAX_ROUND_KEYS * 2], // 72 个 uint64 = 144 个 u32
    pub(crate) rounds: usize,
}

impl BlockCipher {
    pub fn new(key: &[u8]) -> Self {
        // 密钥必须是 32 字节，短密钥补零会静默派生出与调用方预期不符的轮密钥，直接拒绝
        assert_eq!(
            key.len(),
            KEY_SIZE,
            "bastion: key must be {} bytes, got {}",
            KEY_SIZE,
            key.len()
        );
        let mut rk = [0u32; MAX_ROUND_KEYS * 2];
        expand_key(key, &mut rk);
        BlockCipher {
            round_keys: rk,
            rounds: DEFAULT_ROUNDS,
        }
    }

    /// 取第 k 个 uint64 轮密钥，拆成两个 u32（低 32 位和高 32 位）
    #[inline]
    fn key_at(&self, k: usize) -> (u32, u32) {
        (self.round_keys[2 * k], self.round_keys[2 * k + 1])
    }

    pub fn encrypt_block(&self, src: &[u8]) -> [u8; BLOCK_SIZE] {
        let mut a = load_words(src);
        self.crypt_core(&mut a, true);
        store_words(&a)
    }

    pub fn decrypt_block(&self, src: &[u8]) -> [u8; BLOCK_SIZE] {
        let mut a = load_words(src);
        self.crypt_core(&mut a, false);
        store_words(&a)
    }

    // 单块加解密核心，encrypt=true 走正向轮，false 走逆轮
    #[inline]
    fn crypt_core(&self, a: &mut [u32; 8], encrypt: bool) {
        if encrypt {
            // 输入白化，用 round_keys[0..8] 异或
            for (j, aj) in a.iter_mut().enumerate() {
                *aj ^= self.round_keys[j];
            }
            let mut ki = 4usize;
            for r in 0..self.rounds {
                if r % 2 == 0 {
                    g_mix8(a);
                } else {
                    h_mix8(a);
                }
                shift_rows(a);
                // 轮常数已在 expand_key 里预合并进轮密钥
                let (k0, k1) = self.key_at(ki);
                let (k2, k3) = self.key_at(ki + 1);
                let (k4, k5) = self.key_at(ki + 2);
                let (k6, k7) = self.key_at(ki + 3);
                a[0] ^= k0;
                a[1] ^= k1;
                a[2] ^= k2;
                a[3] ^= k3;
                a[4] ^= k4;
                a[5] ^= k5;
                a[6] ^= k6;
                a[7] ^= k7;
                ki += 4;
            }
            // 输出白化
            ki = 4 + self.rounds * 4;
            let (k0, k1) = self.key_at(ki);
            let (k2, k3) = self.key_at(ki + 1);
            let (k4, k5) = self.key_at(ki + 2);
            let (k6, k7) = self.key_at(ki + 3);
            a[0] ^= k0;
            a[1] ^= k1;
            a[2] ^= k2;
            a[3] ^= k3;
            a[4] ^= k4;
            a[5] ^= k5;
            a[6] ^= k6;
            a[7] ^= k7;
        } else {
            // 输出白化解除
            let mut ki = 4 + self.rounds * 4;
            let (k0, k1) = self.key_at(ki);
            let (k2, k3) = self.key_at(ki + 1);
            let (k4, k5) = self.key_at(ki + 2);
            let (k6, k7) = self.key_at(ki + 3);
            a[0] ^= k0;
            a[1] ^= k1;
            a[2] ^= k2;
            a[3] ^= k3;
            a[4] ^= k4;
            a[5] ^= k5;
            a[6] ^= k6;
            a[7] ^= k7;
            for r in (0..self.rounds).rev() {
                ki = 4 + r * 4;
                let (k0, k1) = self.key_at(ki);
                let (k2, k3) = self.key_at(ki + 1);
                let (k4, k5) = self.key_at(ki + 2);
                let (k6, k7) = self.key_at(ki + 3);
                a[0] ^= k0;
                a[1] ^= k1;
                a[2] ^= k2;
                a[3] ^= k3;
                a[4] ^= k4;
                a[5] ^= k5;
                a[6] ^= k6;
                a[7] ^= k7;
                // InvShiftRows：奇偶解交织的逆
                inv_shift_rows(a);
                if r % 2 == 0 {
                    g_mix8_inv(a);
                } else {
                    h_mix8_inv(a);
                }
            }
            // 输入白化解除
            for (j, aj) in a.iter_mut().enumerate() {
                *aj ^= self.round_keys[j];
            }
        }
    }
}

// 平台化安全清零用到的系统 API：Windows 的 SecureZeroMemory 只是 winnt.h 内联实现，没有 DLL 导出符号可 FFI
// Windows 走下面的 write_volatile 兜底，与 SDK 内联实现的易失写入语义一致

// Apple：C11 附录 K 的 memset_s，系统保证写入不被优化
#[cfg(target_vendor = "apple")]
extern "C" {
    fn memset_s(dst: *mut std::ffi::c_void, dst_size: usize, ch: i32, count: usize) -> i32;
}

// Linux/BSD/Android：explicit_bzero，系统保证写入不被优化
#[cfg(all(
    unix,
    not(target_vendor = "apple"),
    any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "dragonfly",
        target_os = "android"
    )
))]
extern "C" {
    fn explicit_bzero(s: *mut std::ffi::c_void, n: usize);
}

// 平台清零入口：上面声明的 API 各写一份实现

#[cfg(target_vendor = "apple")]
unsafe fn platform_zero(ptr: *mut u8, len: usize) {
    let _ = memset_s(ptr as *mut std::ffi::c_void, len, 0, len);
}

#[cfg(all(
    unix,
    not(target_vendor = "apple"),
    any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "dragonfly",
        target_os = "android"
    )
))]
unsafe fn platform_zero(ptr: *mut u8, len: usize) {
    explicit_bzero(ptr as *mut std::ffi::c_void, len);
}

// 兜底（含 Windows）：write_volatile 阻止编译器优化掉写入，写完加编译器屏障
#[cfg(not(any(
    target_vendor = "apple",
    all(
        unix,
        not(target_vendor = "apple"),
        any(
            target_os = "linux",
            target_os = "freebsd",
            target_os = "openbsd",
            target_os = "netbsd",
            target_os = "dragonfly",
            target_os = "android"
        )
    )
)))]
unsafe fn platform_zero(ptr: *mut u8, len: usize) {
    for i in 0..len {
        std::ptr::write_volatile(ptr.add(i), 0);
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

// 安全清零内存：优先平台 API，没有平台 API 时走 platform_zero 里的 volatile 兜底
fn secure_zero(buf: &mut [u8]) {
    unsafe { platform_zero(buf.as_mut_ptr(), buf.len()) }
}

// Drop 时自动清零轮密钥，不管函数怎么返回都能擦除
impl Drop for BlockCipher {
    fn drop(&mut self) {
        // 轮密钥按字节整体清零，走平台安全清零路径
        unsafe {
            platform_zero(
                self.round_keys.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(&self.round_keys),
            );
        }
        self.rounds = 0;
    }
}

// 密钥包装器，Drop 时自动清零，用在 AEAD 的子密钥上
struct ZeroizingKey([u8; KEY_SIZE]);

impl AsRef<[u8]> for ZeroizingKey {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl Drop for ZeroizingKey {
    fn drop(&mut self) {
        secure_zero(&mut self.0);
    }
}

// 逆混合（跟 Go 的 InvG/InvH 轮逐行对应）

fn g_mix8_inv(a: &mut [u32; 8]) {
    let mut s0 = a[0];
    let mut s1 = a[1];
    let mut s2 = a[2];
    let mut s3 = a[3];
    let mut s4 = a[4];
    let mut s5 = a[5];
    let mut s6 = a[6];
    let mut s7 = a[7];

    s2 = rotl32(s2, GM_R4 as usize) ^ s4;
    s3 = rotl32(s3, GM_R4 as usize) ^ s5;
    s4 = s4.wrapping_sub(rotr32(s6, GM_R8 as usize));
    s5 = s5.wrapping_sub(rotr32(s7, GM_R8 as usize));
    s6 = rotl32(s6, GM_R3 as usize) ^ s0;
    s7 = rotl32(s7, GM_R3 as usize) ^ s1;
    s0 = s0.wrapping_sub(rotr32(s2, GM_R7 as usize));
    s1 = s1.wrapping_sub(rotr32(s3, GM_R7 as usize));
    s0 ^= rotr32(s4, GM_R9 as usize);
    s1 ^= rotr32(s5, GM_R9 as usize);
    s2 = rotl32(s2, GM_R2 as usize) ^ s4;
    s3 = rotl32(s3, GM_R2 as usize) ^ s5;
    s4 = s4.wrapping_sub(rotr32(s6, GM_R6 as usize));
    s5 = s5.wrapping_sub(rotr32(s7, GM_R6 as usize));
    s6 = rotl32(s6, GM_R1 as usize) ^ s0;
    s7 = rotl32(s7, GM_R1 as usize) ^ s1;
    s0 = s0.wrapping_sub(rotr32(s2, GM_R5 as usize));
    s1 = s1.wrapping_sub(rotr32(s3, GM_R5 as usize));

    a[0] = s0;
    a[1] = s1;
    a[2] = s2;
    a[3] = s3;
    a[4] = s4;
    a[5] = s5;
    a[6] = s6;
    a[7] = s7;
}

fn h_mix8_inv(a: &mut [u32; 8]) {
    let mut s0 = a[0];
    let mut s1 = a[1];
    let mut s2 = a[2];
    let mut s3 = a[3];
    let mut s4 = a[4];
    let mut s5 = a[5];
    let mut s6 = a[6];
    let mut s7 = a[7];

    s2 = rotl32(s2, HM_R4 as usize) ^ s4;
    s3 = rotl32(s3, HM_R4 as usize) ^ s5;
    s4 = s4.wrapping_sub(rotr32(s6, HM_R8 as usize));
    s5 = s5.wrapping_sub(rotr32(s7, HM_R8 as usize));
    s6 = rotl32(s6, HM_R3 as usize) ^ s0;
    s7 = rotl32(s7, HM_R3 as usize) ^ s1;
    s0 = s0.wrapping_sub(rotr32(s2, HM_R7 as usize));
    s1 = s1.wrapping_sub(rotr32(s3, HM_R7 as usize));
    s0 ^= rotr32(s4, HM_R9 as usize);
    s1 ^= rotr32(s5, HM_R9 as usize);
    s2 = rotl32(s2, HM_R2 as usize) ^ s4;
    s3 = rotl32(s3, HM_R2 as usize) ^ s5;
    s4 = s4.wrapping_sub(rotr32(s6, HM_R6 as usize));
    s5 = s5.wrapping_sub(rotr32(s7, HM_R6 as usize));
    s6 = rotl32(s6, HM_R1 as usize) ^ s0;
    s7 = rotl32(s7, HM_R1 as usize) ^ s1;
    s0 = s0.wrapping_sub(rotr32(s2, HM_R5 as usize));
    s1 = s1.wrapping_sub(rotr32(s3, HM_R5 as usize));

    a[0] = s0;
    a[1] = s1;
    a[2] = s2;
    a[3] = s3;
    a[4] = s4;
    a[5] = s5;
    a[6] = s6;
    a[7] = s7;
}

fn expand_key(key: &[u8], rk: &mut [u32; MAX_ROUND_KEYS * 2]) {
    // 装载密钥，小端序 8 个 u32（入口已校验 key 长度为 32 字节）
    let mut kb = [0u8; KEY_SIZE];
    kb.copy_from_slice(key);
    for i in 0..8 {
        rk[i] = u32::from_le_bytes([kb[4 * i], kb[4 * i + 1], kb[4 * i + 2], kb[4 * i + 3]]);
    }
    let n = MAX_ROUND_KEYS * 2;
    let mut i = 8usize;
    while i < n {
        let mut temp = rk[i - 1];
        if i.is_multiple_of(8) {
            temp = rotl32(temp, 8);
            temp = sbox_word_apply32(temp);
        } else if i.is_multiple_of(4) {
            temp = sbox_word_apply32(temp);
            temp = rotl32(temp, 16);
            temp = temp.wrapping_add(rotl32(temp, 7));
        } else {
            temp = xmx_finalize(temp);
        }
        rk[i] = rk[i - 8] ^ temp;
        i += 1;
    }
    // 轮常数异或到每轮基址上
    for (r, sc_row) in ROUND_CONSTANTS.iter().enumerate() {
        let base = (r + 1) * 8;
        for j in 0..8 {
            rk[base + j] ^= sc_row[j];
        }
    }
    // kb 是主密钥的栈副本，装载完就清掉
    secure_zero(&mut kb);
}

// ARX 哈希

/// ARX 压缩，Davies-Meyer 结构
fn arx_round(s: &mut [u32; 8], round: usize) {
    let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5, mut s6, mut s7) =
        (s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
    if round.is_multiple_of(2) {
        s0 = s0.wrapping_add(rotr32(s2, GM_R5 as usize));
        s1 = s1.wrapping_add(rotr32(s3, GM_R5 as usize));
        s6 ^= s0;
        s7 ^= s1;
        s6 = rotr32(s6, GM_R1 as usize);
        s7 = rotr32(s7, GM_R1 as usize);
        s4 = s4.wrapping_add(rotr32(s6, GM_R6 as usize));
        s5 = s5.wrapping_add(rotr32(s7, GM_R6 as usize));
        s2 ^= s4;
        s3 ^= s5;
        s2 = rotr32(s2, GM_R2 as usize);
        s3 = rotr32(s3, GM_R2 as usize);
        s0 ^= rotr32(s4, GM_R9 as usize);
        s1 ^= rotr32(s5, GM_R9 as usize);
        s0 = s0.wrapping_add(rotr32(s2, GM_R7 as usize));
        s1 = s1.wrapping_add(rotr32(s3, GM_R7 as usize));
        s6 ^= s0;
        s7 ^= s1;
        s6 = rotr32(s6, GM_R3 as usize);
        s7 = rotr32(s7, GM_R3 as usize);
        s4 = s4.wrapping_add(rotr32(s6, GM_R8 as usize));
        s5 = s5.wrapping_add(rotr32(s7, GM_R8 as usize));
        s2 ^= s4;
        s3 ^= s5;
        s2 = rotr32(s2, GM_R4 as usize);
        s3 = rotr32(s3, GM_R4 as usize);
    } else {
        s0 = s0.wrapping_add(rotr32(s2, HM_R5 as usize));
        s1 = s1.wrapping_add(rotr32(s3, HM_R5 as usize));
        s6 ^= s0;
        s7 ^= s1;
        s6 = rotr32(s6, HM_R1 as usize);
        s7 = rotr32(s7, HM_R1 as usize);
        s4 = s4.wrapping_add(rotr32(s6, HM_R6 as usize));
        s5 = s5.wrapping_add(rotr32(s7, HM_R6 as usize));
        s2 ^= s4;
        s3 ^= s5;
        s2 = rotr32(s2, HM_R2 as usize);
        s3 = rotr32(s3, HM_R2 as usize);
        s0 ^= rotr32(s4, HM_R9 as usize);
        s1 ^= rotr32(s5, HM_R9 as usize);
        s0 = s0.wrapping_add(rotr32(s2, HM_R7 as usize));
        s1 = s1.wrapping_add(rotr32(s3, HM_R7 as usize));
        s6 ^= s0;
        s7 ^= s1;
        s6 = rotr32(s6, HM_R3 as usize);
        s7 = rotr32(s7, HM_R3 as usize);
        s4 = s4.wrapping_add(rotr32(s6, HM_R8 as usize));
        s5 = s5.wrapping_add(rotr32(s7, HM_R8 as usize));
        s2 ^= s4;
        s3 ^= s5;
        s2 = rotr32(s2, HM_R4 as usize);
        s3 = rotr32(s3, HM_R4 as usize);
    }
    // ShiftRows：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
    let t = s0; s0 = s2; s2 = s7; s7 = s1;
    s1 = s5; s5 = s4; s4 = s3; s3 = s6; s6 = t;
    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
    s[4] = s4;
    s[5] = s5;
    s[6] = s6;
    s[7] = s7;
}

fn arx_compress(state: &mut [u32; 8], block: &[u8], last: bool) {
    let mut m = [0u32; 8];
    for i in 0..8 {
        m[i] = u32::from_le_bytes([
            block[4 * i],
            block[4 * i + 1],
            block[4 * i + 2],
            block[4 * i + 3],
        ]);
    }
    // 封口常数只加在吸收阶段，不参与末尾前馈
    let seal: [u32; 8] = if last { FINAL_CONST } else { [0u32; 8] };
    let mut s = *state;
    // 模加注入消息（末块含封口常数）
    for i in 0..8 {
        s[i] = s[i].wrapping_add(m[i]).wrapping_add(seal[i]);
    }
    for (r, c) in ROUND_CONSTANTS.iter().enumerate() {
        arx_round(&mut s, r);
        // 每轮异或圆周率派生的轮常数做域分离
        for i in 0..8 {
            s[i] ^= c[i];
        }
    }
    for i in 0..8 {
        // Davies-Meyer 前馈
        state[i] = (state[i] ^ s[i]).wrapping_add(m[i]);
    }
}

/// 算 ARX 哈希，输入走流式，边读边压缩，不整块载入内存
pub fn hash<R: Read>(input: &mut R) -> io::Result<[u8; HASH_SIZE]> {
    let mut h = HashState::new(ARX_INITIAL_STATE);
    // 缓冲按线程复用，小输入不必每次开 4MB 再清零
    with_stream_buf_a(|buf| -> io::Result<()> {
        loop {
            let n = input.read(buf)?;
            if n == 0 {
                break;
            }
            h.write(&buf[..n]);
        }
        Ok(())
    })?;
    Ok(h.sum())
}

// HMAC

pub struct Hmac {
    inner: HashState,
    outer: HashState,
}

// ARX 压缩的独立增量状态，哈希与 HMAC 共用的压缩内核
struct HashState {
    state: [u32; 8],
    buf: [u8; HASH_BLOCK_SIZE],
    buf_len: usize,
    total: u64,
}

impl HashState {
    // 以指定初始状态新建增量状态，HMAC 的 ipad/opad 就体现在初始状态上
    fn new(init_state: [u32; 8]) -> Self {
        HashState {
            state: init_state,
            buf: [0u8; HASH_BLOCK_SIZE],
            buf_len: 0,
            total: 0,
        }
    }
}

// 状态可能含 HMAC 的 ipad/opad 派生值或密钥材料缓冲，丢弃时走平台安全清零
impl Drop for HashState {
    fn drop(&mut self) {
        unsafe {
            platform_zero(
                self.state.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(&self.state),
            );
            platform_zero(self.buf.as_mut_ptr(), self.buf.len());
        }
        self.buf_len = 0;
    }
}

fn derive_hmac_states(k: &[u8; HMAC_BLOCK_SIZE]) -> (HashState, HashState) {
    let mut ipad = [0u8; HMAC_BLOCK_SIZE];
    let mut opad = [0u8; HMAC_BLOCK_SIZE];
    for i in 0..HMAC_BLOCK_SIZE {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5C;
    }
    let mut ipad_state = [0u32; 8];
    let mut opad_state = [0u32; 8];
    for i in 0..8 {
        ipad_state[i] = u32::from_le_bytes([
            ipad[4 * i],
            ipad[4 * i + 1],
            ipad[4 * i + 2],
            ipad[4 * i + 3],
        ]);
        opad_state[i] = u32::from_le_bytes([
            opad[4 * i],
            opad[4 * i + 1],
            opad[4 * i + 2],
            opad[4 * i + 3],
        ]);
    }
    let mut inner_state = ARX_INITIAL_STATE;
    let mut outer_state = ARX_INITIAL_STATE;
    for i in 0..8 {
        inner_state[i] ^= ipad_state[i];
        outer_state[i] ^= opad_state[i];
    }
    (HashState::new(inner_state), HashState::new(outer_state))
}

impl HashState {
    fn write(&mut self, data: &[u8]) {
        self.total += data.len() as u64;
        if self.buf_len > 0 {
            let space = HASH_BLOCK_SIZE - self.buf_len;
            let take = space.min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            if self.buf_len < HASH_BLOCK_SIZE {
                return;
            }
            arx_compress(&mut self.state, &self.buf, false);
            self.buf_len = 0;
            self.process_remaining(&data[take..]);
        } else {
            self.process_remaining(data);
        }
    }

    fn process_remaining(&mut self, mut data: &[u8]) {
        while data.len() >= HASH_BLOCK_SIZE {
            let mut block = [0u8; HASH_BLOCK_SIZE];
            block.copy_from_slice(&data[..HASH_BLOCK_SIZE]);
            arx_compress(&mut self.state, &block, false);
            data = &data[HASH_BLOCK_SIZE..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn sum(&self) -> [u8; HASH_SIZE] {
        let mut state = self.state;
        let data_len = self.buf_len;
        let mut padded = [0u8; HASH_BLOCK_SIZE * 2];
        padded[..data_len].copy_from_slice(&self.buf[..data_len]);
        padded[data_len] = 0x80;
        let mut pad_need = HASH_BLOCK_SIZE - (data_len % HASH_BLOCK_SIZE);
        if pad_need < 9 {
            pad_need += HASH_BLOCK_SIZE;
        }
        let len = (self.total.wrapping_mul(8)).to_le_bytes();
        padded[data_len + pad_need - 8..data_len + pad_need].copy_from_slice(&len);
        let total_block_bytes = data_len + pad_need;
        let mut j = 0;
        while j < total_block_bytes {
            let mut block = [0u8; HASH_BLOCK_SIZE];
            block.copy_from_slice(&padded[j..j + HASH_BLOCK_SIZE]);
            // 最后一个填充块带封口标志，标记这是消息末尾
            let last = j + HASH_BLOCK_SIZE >= total_block_bytes;
            arx_compress(&mut state, &block, last);
            j += HASH_BLOCK_SIZE;
        }
        let mut out = [0u8; HASH_SIZE];
        for i in 0..8 {
            out[4 * i..4 * i + 4].copy_from_slice(&state[i].to_le_bytes());
        }
        out
    }
}

impl Hmac {
    pub fn new(key: &[u8]) -> Self {
        let mut k = [0u8; HMAC_BLOCK_SIZE];
        if key.len() > HMAC_BLOCK_SIZE {
            // 密钥超块长先哈希截短，Cursor 只是适配流式哈希，不会出错
            let h = hash(&mut Cursor::new(key)).unwrap_or_default();
            k[..HASH_SIZE].copy_from_slice(&h);
        } else {
            k[..key.len()].copy_from_slice(key);
        }
        let (inner, outer) = derive_hmac_states(&k);
        Hmac { inner, outer }
    }

    pub fn write(&mut self, data: &[u8]) {
        self.inner.write(data);
    }

    pub fn sum(&self) -> [u8; HASH_SIZE] {
        let inner_copy = self.inner.clone_state();
        let inner_hash = inner_copy.sum();
        let mut outer_copy = self.outer.clone_state();
        outer_copy.write(&inner_hash);
        outer_copy.sum()
    }
}

impl HashState {
    fn clone_state(&self) -> HashState {
        HashState {
            state: self.state,
            buf: self.buf,
            buf_len: self.buf_len,
            total: self.total,
        }
    }
}

// 内存版 HMAC，供 HKDF/PBKDF2 等高频小数据内部调用
fn hmac_slice(key: &[u8], data: &[u8]) -> [u8; HASH_SIZE] {
    let mut h = Hmac::new(key);
    h.write(data);
    h.sum()
}

/// HMAC，输入走流式
pub fn hmac<R: Read>(key: &[u8], input: &mut R) -> io::Result<[u8; HASH_SIZE]> {
    let mut h = Hmac::new(key);
    // 缓冲按线程复用，小输入不必每次开 4MB 再清零
    with_stream_buf_b(|buf| -> io::Result<()> {
        loop {
            let n = input.read(buf)?;
            if n == 0 {
                break;
            }
            h.write(&buf[..n]);
        }
        Ok(())
    })?;
    Ok(h.sum())
}

// HKDF

pub fn hkdf_extract(ikm: &[u8], salt: &[u8]) -> [u8; KDF_HASH_LEN] {
    // 空 salt 经 HMAC 密钥规范化（补零到块长）后等价于标准缺省的全零 salt，无需特判
    hmac_slice(salt, ikm)
}

pub fn hkdf_expand(prk: &[u8; KDF_HASH_LEN], info: &[u8], length: usize) -> Vec<u8> {
    // 输出长度上限 255 块：块计数器是 u8，超限会回绕导致派生输入重复
    if length > 255 * KDF_HASH_LEN {
        panic!(
            "bastion HKDF expand: length {} exceeds max {}",
            length,
            255 * KDF_HASH_LEN
        );
    }
    let mut result = Vec::with_capacity(length);
    let mut prev = Vec::new();
    let mut counter: u8 = 1;
    while result.len() < length {
        let mut h = Hmac::new(prk);
        if !prev.is_empty() {
            h.write(&prev);
        }
        h.write(info);
        h.write(&[counter]);
        prev = h.sum().to_vec();
        result.extend_from_slice(&prev);
        // 长度上限正好 255 块，最后一轮后计数到 256，用回绕加法避免调试构建的溢出 panic
        counter = counter.wrapping_add(1);
    }
    result.truncate(length);
    result
}

pub fn hkdf(ikm: &[u8], salt: &[u8], info: &[u8], length: usize) -> Vec<u8> {
    let prk = hkdf_extract(ikm, salt);
    hkdf_expand(&prk, info, length)
}

// PBKDF2

// pbkdf2_block 计算 PBKDF2 的第 block_idx 个输出块（各块链独立，可并行）
fn pbkdf2_block(password: &[u8], salt: &[u8], iterations: u32, block_idx: usize) -> [u8; KDF_HASH_LEN] {
    let mut h = Hmac::new(password);
    h.write(salt);
    h.write(&(block_idx as u32).to_be_bytes());
    let mut u = h.sum();
    let mut acc = u;
    for _ in 1..iterations {
        let mut h2 = Hmac::new(password);
        h2.write(&u);
        u = h2.sum();
        for k in 0..KDF_HASH_LEN {
            acc[k] ^= u[k];
        }
    }
    // 最后一轮的 u 已并入 acc，迭代链中间值用完清掉
    secure_zero(&mut u);
    acc
}

pub fn pbkdf2(password: &[u8], salt: &[u8], iterations: u32, key_len: usize) -> Vec<u8> {
    // 空口令与迭代次数 0 直接拒绝，避免静默按缺省值派生出弱密钥
    assert!(!password.is_empty(), "bastion PBKDF2: password must not be empty");
    assert!(iterations >= 1, "bastion PBKDF2: iterations must be >= 1");
    let num_blocks = key_len.div_ceil(KDF_HASH_LEN);
    let mut result = vec![0u8; num_blocks * KDF_HASH_LEN];

    // 块数不足 4 时并行只有开销没有收益，直接串行
    if num_blocks < 4 {
        for block_idx in 1..=num_blocks {
            let acc = pbkdf2_block(password, salt, iterations, block_idx);
            result[(block_idx - 1) * KDF_HASH_LEN..block_idx * KDF_HASH_LEN].copy_from_slice(&acc);
        }
        result.truncate(key_len);
        return result;
    }

    // RFC 2898 各输出块链完全独立，多核并行每个 worker 处理一段块区间
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(num_blocks);
    // 先用 split_at_mut 一次性切分 result，避免循环内多次可变借用
    let (chunk, rem) = (num_blocks / workers, num_blocks % workers);
    let mut segs: Vec<(usize, &mut [u8])> = Vec::with_capacity(workers);
    let mut rest: &mut [u8] = &mut result[..];
    let mut cursor = 0usize;
    for w in 0..workers {
        let cnt = chunk + if w < rem { 1 } else { 0 };
        if cnt == 0 {
            continue;
        }
        let byte_cnt = cnt * KDF_HASH_LEN;
        let (head, tail) = rest.split_at_mut(byte_cnt);
        segs.push((cursor, head));
        rest = tail;
        cursor += cnt;
    }
    std::thread::scope(|s| {
        for (s_idx, seg) in segs {
            let cnt = seg.len() / KDF_HASH_LEN;
            s.spawn(move || {
                for bi in 0..cnt {
                    let acc = pbkdf2_block(password, salt, iterations, s_idx + bi + 1);
                    seg[bi * KDF_HASH_LEN..(bi + 1) * KDF_HASH_LEN].copy_from_slice(&acc);
                }
            });
        }
    });
    result.truncate(key_len);
    result
}

/// BLAKE3 风格的密钥派生
/// 先 hash(context || 0x00 || keyMaterial) 拿到 baseKey
/// 长度 <= 32 直接截取，超过就用 baseKey 做 CTR 加密零明文扩展
pub fn derive_key(context: &str, key_material: &[u8], length: usize) -> Vec<u8> {
    if length == 0 {
        return Vec::new();
    }
    let mut input = Vec::with_capacity(context.len() + 1 + key_material.len());
    input.extend_from_slice(context.as_bytes());
    input.push(0x00);
    input.extend_from_slice(key_material);

    let base_key = hash(&mut Cursor::new(&input)).unwrap_or_default();

    if length <= HASH_SIZE {
        return base_key[..length].to_vec();
    }

    let cipher = BlockCipher::new(&base_key);
    // 计数器布局：前 24 字节 nonce + 后 8 字节小端计数器
    let mut counter = [0u8; BLOCK_SIZE];
    let mut out = vec![0u8; length];
    let mut offset = 0usize;
    while offset < length {
        let ks = cipher.encrypt_block(&counter);
        let take = (BLOCK_SIZE).min(length - offset);
        out[offset..offset + take].copy_from_slice(&ks[..take]);
        // 后 8 字节小端递增
        let mut ctr = u64::from_le_bytes(counter[24..32].try_into().unwrap());
        ctr = ctr.wrapping_add(1);
        counter[24..32].copy_from_slice(&ctr.to_le_bytes());
        offset += take;
    }
    out
}

// CTR 模式

// 64KB 以上才开多核
const CTR_PARALLEL_THRESHOLD: usize = 64 * 1024;
// 每 worker 至少 1024 块（32KB）
const CTR_MIN_BLOCKS_PER_WORKER: usize = 1024;
// 最多 8 个 worker
const CTR_MAX_WORKERS: usize = 8;

// 根据数据量算合适的 worker 数
fn get_optimal_worker_count(data_len: usize) -> usize {
    let cpu = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if cpu <= 1 {
        return 1;
    }
    let blocks = data_len / BLOCK_SIZE;
    if blocks < cpu * CTR_MIN_BLOCKS_PER_WORKER {
        let w = blocks / CTR_MIN_BLOCKS_PER_WORKER;
        let w = if w < 1 { 1 } else { w };
        return w.min(CTR_MAX_WORKERS);
    }
    if cpu > CTR_MAX_WORKERS {
        return CTR_MAX_WORKERS;
    }
    cpu
}

// 单轮正向混淆，跟 BlockCipher::encrypt_block 里的轮函数一样
#[inline(always)]
fn encrypt_one_round(a: &mut [u32; 8], keys: &[u32; MAX_ROUND_KEYS * 2], ki: usize, even: bool) {
    if even {
        g_mix8(a);
    } else {
        h_mix8(a);
    }
    shift_rows(a);
    let (k0, k1) = (keys[2 * ki], keys[2 * ki + 1]);
    let (k2, k3) = (keys[2 * (ki + 1)], keys[2 * (ki + 1) + 1]);
    let (k4, k5) = (keys[2 * (ki + 2)], keys[2 * (ki + 2) + 1]);
    let (k6, k7) = (keys[2 * (ki + 3)], keys[2 * (ki + 3) + 1]);
    a[0] ^= k0;
    a[1] ^= k1;
    a[2] ^= k2;
    a[3] ^= k3;
    a[4] ^= k4;
    a[5] ^= k5;
    a[6] ^= k6;
    a[7] ^= k7;
}

// 单轮逆向混淆，跟 BlockCipher::decrypt_block 的轮函数一样
#[inline(always)]
#[allow(dead_code)]
fn decrypt_one_round(a: &mut [u32; 8], keys: &[u32; MAX_ROUND_KEYS * 2], ki: usize, even: bool) {
    let (k0, k1) = (keys[2 * ki], keys[2 * ki + 1]);
    let (k2, k3) = (keys[2 * (ki + 1)], keys[2 * (ki + 1) + 1]);
    let (k4, k5) = (keys[2 * (ki + 2)], keys[2 * (ki + 2) + 1]);
    let (k6, k7) = (keys[2 * (ki + 3)], keys[2 * (ki + 3) + 1]);
    a[0] ^= k0;
    a[1] ^= k1;
    a[2] ^= k2;
    a[3] ^= k3;
    a[4] ^= k4;
    a[5] ^= k5;
    a[6] ^= k6;
    a[7] ^= k7;
    inv_shift_rows(a);
    if even {
        g_mix8_inv(a);
    } else {
        h_mix8_inv(a);
    }
}

// 4 组状态字分别做完整解密（输入白化在外面已经做完了），4 组独立可交错流水线
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn decrypt_states_4way(s: &mut [[u32; 8]; 4], keys: &[u32; MAX_ROUND_KEYS * 2], rounds: usize) {
    // 输出白化解除
    let ki = 4 + rounds * 4;
    for ss in s.iter_mut() {
        for j in 0..8 {
            ss[j] ^= keys[2 * ki + j];
        }
    }
    for r in (0..rounds).rev() {
        let ki = 4 + r * 4;
        let even = r % 2 == 0;
        for ss in s.iter_mut() {
            decrypt_one_round(ss, keys, ki, even);
        }
    }
    // 输入白化解除
    for ss in s.iter_mut() {
        for j in 0..8 {
            ss[j] ^= keys[j];
        }
    }
}

// 4 组状态字分别做完整加密（输入白化在外面已经做完了）
#[inline(always)]
pub(crate) fn encrypt_states_4way(s: &mut [[u32; 8]; 4], keys: &[u32; MAX_ROUND_KEYS * 2], rounds: usize) {
    for ss in s.iter_mut() {
        for j in 0..8 {
            ss[j] ^= keys[j];
        }
    }
    let mut ki = 4usize;
    for r in 0..rounds {
        let even = r % 2 == 0;
        for ss in s.iter_mut() {
            encrypt_one_round(ss, keys, ki, even);
        }
        ki += 4;
    }
    let ki = 4 + rounds * 4;
    for ss in s.iter_mut() {
        for j in 0..8 {
            ss[j] ^= keys[2 * ki + j];
        }
    }
}

// 4 块批量加密 CTR 密钥流并 XOR 写入，软件 4 路 ILP 流水线；counter 前 24 字节 nonce 后 8 字节小端计数器
pub(crate) fn encrypt_blocks_batch4(cipher: &BlockCipher, dst: &mut [u8], src: &[u8], counter: &[u8; BLOCK_SIZE]) {
    let n = src.len() / BLOCK_SIZE;
    if n == 0 {
        return;
    }
    // nonce 前 24 字节拆成 6 个 u32，小端序
    let n0 = u32::from_le_bytes([counter[0], counter[1], counter[2], counter[3]]);
    let n1 = u32::from_le_bytes([counter[4], counter[5], counter[6], counter[7]]);
    let n2 = u32::from_le_bytes([counter[8], counter[9], counter[10], counter[11]]);
    let n3 = u32::from_le_bytes([counter[12], counter[13], counter[14], counter[15]]);
    let n4 = u32::from_le_bytes([counter[16], counter[17], counter[18], counter[19]]);
    let n5 = u32::from_le_bytes([counter[20], counter[21], counter[22], counter[23]]);
    let base_ctr = u64::from_le_bytes(counter[24..32].try_into().unwrap());
    let keys = &cipher.round_keys;
    let rounds = cipher.rounds;

    let mut i = 0usize;
    while i + 3 < n {
        let c0 = base_ctr + i as u64;
        let c1 = base_ctr + i as u64 + 1;
        let c2 = base_ctr + i as u64 + 2;
        let c3 = base_ctr + i as u64 + 3;
        // 4 组独立状态，展开成数组让编译器方便交错指令
        let mut s: [[u32; 8]; 4] = [
            [n0, n1, n2, n3, n4, n5, c0 as u32, (c0 >> 32) as u32],
            [n0, n1, n2, n3, n4, n5, c1 as u32, (c1 >> 32) as u32],
            [n0, n1, n2, n3, n4, n5, c2 as u32, (c2 >> 32) as u32],
            [n0, n1, n2, n3, n4, n5, c3 as u32, (c3 >> 32) as u32],
        ];

        // 4 路 ILP 完整加密
        encrypt_states_4way(&mut s, keys, rounds);

        // 写出 4 块密钥流并 XOR 到明文
        for (g, sg) in s.iter().enumerate() {
            let off = (i + g) * BLOCK_SIZE;
            let d = &mut dst[off..off + BLOCK_SIZE];
            let sblk = &src[off..off + BLOCK_SIZE];
            for j in 0..8 {
                let v =
                    u32::from_le_bytes([sblk[4 * j], sblk[4 * j + 1], sblk[4 * j + 2], sblk[4 * j + 3]])
                        ^ sg[j];
                d[4 * j..4 * j + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        i += 4;
    }

    // 尾部不足 4 块，用单块 Encrypt 兜底
    if i < n {
        let mut ctr = *counter;
        for blk in i..n {
            ctr[CTR_COUNTER_SIZE..].copy_from_slice(&(base_ctr + blk as u64).to_le_bytes());
            let ks = cipher.encrypt_block(&ctr);
            let off = blk * BLOCK_SIZE;
            for j in 0..BLOCK_SIZE {
                dst[off + j] = src[off + j] ^ ks[j];
            }
        }
    }
}

// 多核并行 CTR 加解密：把完整块分给多个 worker，各自推进计数器
// 只处理完整块，不足一块的尾部由调用方顺序处理
fn ctr_crypt_blocks(cipher: &BlockCipher, dst: &mut [u8], src: &[u8], counter: &[u8; BLOCK_SIZE]) {
    let len = src.len();
    if len == 0 {
        return;
    }
    if len < CTR_PARALLEL_THRESHOLD {
        #[cfg(target_arch = "x86_64")]
        simd::encrypt_blocks_dispatched(cipher, dst, src, counter);
        #[cfg(not(target_arch = "x86_64"))]
        encrypt_blocks_batch4(cipher, dst, src, counter);
        return;
    }
    let workers = get_optimal_worker_count(len);
    if workers <= 1 {
        #[cfg(target_arch = "x86_64")]
        simd::encrypt_blocks_dispatched(cipher, dst, src, counter);
        #[cfg(not(target_arch = "x86_64"))]
        encrypt_blocks_batch4(cipher, dst, src, counter);
        return;
    }

    let base_ctr = u64::from_le_bytes(counter[24..32].try_into().unwrap());
    let blocks = len / BLOCK_SIZE;
    let blocks_per_worker = blocks.div_ceil(workers);
    // 用 split_at_mut 把 dst 切成各 worker 独立的可变切片，避免所有权麻烦
    std::thread::scope(|s| {
        let mut dst_rem = dst;
        let mut src_off = 0usize;
        for w in 0..workers {
            let start_block = w * blocks_per_worker;
            if start_block >= blocks {
                break;
            }
            let mut end_block = start_block + blocks_per_worker;
            if end_block > blocks {
                end_block = blocks;
            }
            let end_off = end_block * BLOCK_SIZE;
            let this_len = end_off - src_off;
            let (this_dst, rest_dst) = dst_rem.split_at_mut(this_len);
            let this_src = &src[src_off..end_off];
            // 每个 worker 自己拿一份计数器副本，把 nonce 和推进后的计数器填好
            let mut worker_ctr = *counter;
            worker_ctr[CTR_COUNTER_SIZE..]
                .copy_from_slice(&(base_ctr + start_block as u64).to_le_bytes());
            s.spawn(move || {
                #[cfg(target_arch = "x86_64")]
                simd::encrypt_blocks_dispatched(cipher, this_dst, this_src, &worker_ctr);
                #[cfg(not(target_arch = "x86_64"))]
                encrypt_blocks_batch4(cipher, this_dst, this_src, &worker_ctr);
            });
            dst_rem = rest_dst;
            src_off = end_off;
        }
    });
}

/// CTR 加解密（同一种操作），输入输出走流式；分块处理保持多核批量加速
pub fn ctr_crypt<R: Read, W: Write>(
    key: &[u8],
    nonce: &[u8],
    input: &mut R,
    output: &mut W,
) -> io::Result<()> {
    // nonce 长度必须严格等于 24：短了会被静默补零、长了后 8 字节会被丢掉，
    // 两种都会让不同调用落到同一个计数器块上造成密钥流重用，直接拒绝
    if nonce.len() != CTR_COUNTER_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("bastion CTR: nonce must be {} bytes", CTR_COUNTER_SIZE),
        ));
    }
    let cipher = BlockCipher::new(key);
    let mut counter = [0u8; BLOCK_SIZE];
    counter[..CTR_COUNTER_SIZE].copy_from_slice(&nonce[..CTR_COUNTER_SIZE]);
    // 读写缓冲按线程复用，小数据不必每次开 8MB 再清零
    with_stream_bufs(|read_buf, write_buf| -> io::Result<()> {
        let mut block_idx: u64 = 0;
        loop {
            let n = input.read(read_buf)?;
            if n == 0 {
                break;
            }
            let full_blocks = n / BLOCK_SIZE;
            let block_bytes = full_blocks * BLOCK_SIZE;
            if full_blocks > 0 {
                // 把当前块序号写进计数器，批量函数据此独立推进各区间的密钥流
                counter[CTR_COUNTER_SIZE..].copy_from_slice(&block_idx.to_le_bytes());
                ctr_crypt_blocks(
                    &cipher,
                    &mut write_buf[..block_bytes],
                    &read_buf[..block_bytes],
                    &counter,
                );
                block_idx += full_blocks as u64;
            }
            // 尾部不足一块，顺序单块处理
            let tail = block_bytes;
            let mut idx = block_idx;
            let mut pos = tail;
            while pos < n {
                counter[CTR_COUNTER_SIZE..].copy_from_slice(&idx.to_le_bytes());
                let ks = cipher.encrypt_block(&counter);
                let take = (BLOCK_SIZE).min(n - pos);
                for i in 0..take {
                    write_buf[pos + i] = read_buf[pos + i] ^ ks[i];
                }
                pos += take;
                idx += 1;
            }
            block_idx = idx;
            output.write_all(&write_buf[..n])?;
        }
        Ok(())
    })
}

// CBC 模式

/// CBC 加密，PKCS#7 填充，输入输出走流式；EOF 时补最后一个填充块
pub fn cbc_encrypt<R: Read, W: Write>(
    key: &[u8],
    iv: &[u8; BLOCK_SIZE],
    input: &mut R,
    output: &mut W,
) -> io::Result<()> {
    let cipher = BlockCipher::new(key);
    let mut prev = *iv;
    let mut pending: Vec<u8> = Vec::with_capacity(STREAM_BUFFER + BLOCK_SIZE);
    // consumed 记录 pending 中已加密的字节偏移，避免每块 drain 造成 O(n²) 搬移
    let mut consumed = 0usize;
    // 密文先攒进缓冲，凑一批一起写出，省掉每块一次 write_all
    with_stream_bufs(|read_buf, write_buf| -> io::Result<()> {
        let mut out_len = 0usize;
        loop {
            let n = input.read(read_buf)?;
            if n == 0 {
                break;
            }
            pending.extend_from_slice(&read_buf[..n]);
            while consumed + BLOCK_SIZE <= pending.len() {
                let mut block = [0u8; BLOCK_SIZE];
                block.copy_from_slice(&pending[consumed..consumed + BLOCK_SIZE]);
                for j in 0..BLOCK_SIZE {
                    block[j] ^= prev[j];
                }
                let ct = cipher.encrypt_block(&block);
                // 缓冲装不下就先冲出去，避免积压超过缓冲长度
                if out_len + BLOCK_SIZE > STREAM_BUFFER {
                    output.write_all(&write_buf[..out_len])?;
                    out_len = 0;
                }
                write_buf[out_len..out_len + BLOCK_SIZE].copy_from_slice(&ct);
                out_len += BLOCK_SIZE;
                prev = ct;
                consumed += BLOCK_SIZE;
            }
            if consumed > 0 {
                pending.drain(..consumed);
                consumed = 0;
            }
            output.write_all(&write_buf[..out_len])?;
            out_len = 0;
        }
        Ok(())
    })?;
    // EOF 按 PKCS#7 补足最后一块并输出
    let pad_len = BLOCK_SIZE - pending.len();
    let mut block = [0u8; BLOCK_SIZE];
    block[..pending.len()].copy_from_slice(&pending);
    for item in block.iter_mut().take(BLOCK_SIZE).skip(pending.len()) {
        *item = pad_len as u8;
    }
    for j in 0..BLOCK_SIZE {
        block[j] ^= prev[j];
    }
    let ct = cipher.encrypt_block(&block);
    output.write_all(&ct)?;
    Ok(())
}

/// CBC 解密若干完整块：块数够多就多核分片，片内走 SIMD 批量解密 + 前块密文异或
fn cbc_decrypt_blocks(cipher: &BlockCipher, dst: &mut [u8], ct: &[u8], prev: &[u8; BLOCK_SIZE]) {
    let len = ct.len();
    let blocks = len / BLOCK_SIZE;
    if blocks == 0 {
        return;
    }
    let workers = if len >= CTR_PARALLEL_THRESHOLD {
        get_optimal_worker_count(len)
    } else {
        1
    };
    if workers <= 1 {
        cbc_decrypt_range(cipher, dst, ct, prev);
        return;
    }

    let per = blocks.div_ceil(workers);
    std::thread::scope(|s| {
        let mut dst_rem = dst;
        let mut off = 0usize;
        for w in 0..workers {
            let sb = w * per;
            if sb >= blocks {
                break;
            }
            let eb = (sb + per).min(blocks);
            let end = eb * BLOCK_SIZE;
            let this_len = end - off;
            let (this_dst, rest_dst) = dst_rem.split_at_mut(this_len);
            let this_ct = &ct[off..end];
            // 每段首块的前块密文直接从密文里取，段与段之间互不依赖
            let mut p = *prev;
            if sb > 0 {
                p.copy_from_slice(&ct[(sb - 1) * BLOCK_SIZE..sb * BLOCK_SIZE]);
            }
            s.spawn(move || {
                cbc_decrypt_range(cipher, this_dst, this_ct, &p);
            });
            dst_rem = rest_dst;
            off = end;
        }
    });
}

// 单段：先整段批量解密（SIMD 分派），再串上"明文 = 解密块 ^ 前块密文"
fn cbc_decrypt_range(cipher: &BlockCipher, dst: &mut [u8], ct: &[u8], prev: &[u8; BLOCK_SIZE]) {
    #[cfg(target_arch = "x86_64")]
    simd::decrypt_blocks_dispatched(cipher, dst, ct);
    #[cfg(not(target_arch = "x86_64"))]
    {
        for i in (0..ct.len()).step_by(BLOCK_SIZE) {
            let dec = cipher.decrypt_block(&ct[i..i + BLOCK_SIZE]);
            dst[i..i + BLOCK_SIZE].copy_from_slice(&dec);
        }
    }
    let mut p = *prev;
    for off in (0..ct.len()).step_by(BLOCK_SIZE) {
        for j in 0..BLOCK_SIZE {
            dst[off + j] ^= p[j];
        }
        p.copy_from_slice(&ct[off..off + BLOCK_SIZE]);
    }
}

/// CBC 解密，PKCS#7 去填充，输入输出走流式；填充无效返回错误
pub fn cbc_decrypt<R: Read, W: Write>(
    key: &[u8],
    iv: &[u8; BLOCK_SIZE],
    input: &mut R,
    output: &mut W,
) -> io::Result<()> {
    let cipher = BlockCipher::new(key);
    let mut prev = *iv;
    // 明文字节先攒这里，除了末尾留一块等填充分辨，其余凑一批一起写出，
    // 不要每 32 字节调一次 write_all
    let mut plain_buf: Vec<u8> = Vec::with_capacity(STREAM_BUFFER + BLOCK_SIZE);
    // 密文零头跨读保留，不足一整块的尾巴最后报长度错
    let mut pending: Vec<u8> = Vec::with_capacity(STREAM_BUFFER + BLOCK_SIZE);
    with_stream_buf_a(|read_buf| -> io::Result<()> {
        loop {
            let n = input.read(read_buf)?;
            if n == 0 {
                break;
            }
            pending.extend_from_slice(&read_buf[..n]);
            let full_bytes = (pending.len() / BLOCK_SIZE) * BLOCK_SIZE;
            if full_bytes > 0 {
                let start = plain_buf.len();
                plain_buf.resize(start + full_bytes, 0);
                // 块间独立，够多就多核 + SIMD 批量解密
                cbc_decrypt_blocks(&cipher, &mut plain_buf[start..], &pending[..full_bytes], &prev);
                prev.copy_from_slice(&pending[full_bytes - BLOCK_SIZE..full_bytes]);
                pending.drain(..full_bytes);
            }
            // 最后一块明文留在缓冲里不写：它可能是填充分块，EOF 时才定夺
            if plain_buf.len() > BLOCK_SIZE {
                let flush = plain_buf.len() - BLOCK_SIZE;
                output.write_all(&plain_buf[..flush])?;
                plain_buf.drain(..flush);
            }
        }
        Ok(())
    })?;
    // 密文长度必须是块大小的整数倍，空密文同样无效
    if !pending.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "CBC密文长度必须是块大小的整数倍",
        ));
    }
    if plain_buf.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "CBC密文长度必须是块大小的整数倍",
        ));
    }
    // 最后一块去 PKCS#7 填充
    {
        let hp = &plain_buf[..BLOCK_SIZE];
        let pad = hp[BLOCK_SIZE - 1] as u32;
        // 常量时间验证 PKCS#7 填充：循环次数固定防计时侧信道；bad 分别拦截 pad<1 与 pad>32
        let mut bad = (((pad.wrapping_sub(1)) >> 31) | (((BLOCK_SIZE as u32).wrapping_sub(pad)) >> 31)) & 1;
        for i in 0..BLOCK_SIZE {
            let b = hp[BLOCK_SIZE - 1 - i];
            let idx_diff = (pad as i32).wrapping_sub(i as i32).wrapping_sub(1);
            let mask = 0u8.wrapping_sub((((idx_diff as u32) >> 31) ^ 1) as u8);
            bad |= ((b ^ pad as u8) & mask) as u32;
        }
        if bad != 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "CBC填充无效"));
        }
        output.write_all(&hp[..BLOCK_SIZE - pad as usize])?;
    }
    Ok(())
}

// XTS 模式

pub(crate) fn gf_mul2_256(x: &mut [u8; BLOCK_SIZE]) {
    let carry = x[0] >> 7;
    for i in 0..BLOCK_SIZE - 1 {
        x[i] = (x[i] << 1) | (x[i + 1] >> 7);
    }
    x[BLOCK_SIZE - 1] <<= 1;
    if carry != 0 {
        x[BLOCK_SIZE - 2] ^= 0x04;
        x[BLOCK_SIZE - 1] ^= 0x25;
    }
}

// 逐字节异或两个 32 字节块，XTS 的 tweak 异或用
#[allow(dead_code)]
pub(crate) fn xor_block(a: &[u8], b: &[u8; BLOCK_SIZE]) -> [u8; BLOCK_SIZE] {
    let mut out = [0u8; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        out[i] = a[i] ^ b[i];
    }
    out
}

pub fn xts_encrypt(key: &[u8; 64], data: &[u8], sector: u64) -> Vec<u8> {
    // 接口约定：XTS 以完整块为最小处理单位，不足一块的输入返回空结果，长度由调用方保证
    if data.len() < BLOCK_SIZE {
        return Vec::new();
    }
    xts_process(key, data, sector, true)
}

pub fn xts_decrypt(key: &[u8; 64], data: &[u8], sector: u64) -> Vec<u8> {
    // 接口约定：XTS 以完整块为最小处理单位，不足一块的输入返回空结果，长度由调用方保证
    if data.len() < BLOCK_SIZE {
        return Vec::new();
    }
    xts_process(key, data, sector, false)
}

// 由基础 tweak 推出第 k 块的 tweak：tweak_k = base · α^k（GF(2²⁵⁶) 倍乘）
fn xts_tweak_at(base: &[u8; BLOCK_SIZE], k: usize) -> [u8; BLOCK_SIZE] {
    let mut t = *base;
    for _ in 0..k {
        gf_mul2_256(&mut t);
    }
    t
}

// 4 路 ILP 串行处理 count 个完整块，起始 tweak 是区间首块的 tweak
// 调用方传本区间的子切片，区间之间互不依赖（tweak 由块序号独立推出），可以并行调度
fn xts_process_range(
    cipher: &BlockCipher,
    result: &mut [u8],
    count: usize,
    start_tweak: &[u8; BLOCK_SIZE],
    encrypt: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        simd::xts_process_segment_dispatched(cipher, result, count, start_tweak, encrypt);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let keys = &cipher.round_keys;
        let rounds = cipher.rounds;
        let mut t = *start_tweak;
        let mut b = 0usize;
        while b + 4 <= count {
            let off = b * BLOCK_SIZE;
            let t0 = t;
            let mut t1 = t;
            gf_mul2_256(&mut t1);
            let mut t2 = t1;
            gf_mul2_256(&mut t2);
            let mut t3 = t2;
            gf_mul2_256(&mut t3);
            let mut s: [[u32; 8]; 4] = [
                load_words(&xor_block(&result[off..off + BLOCK_SIZE], &t0)),
                load_words(&xor_block(&result[off + BLOCK_SIZE..off + 2 * BLOCK_SIZE], &t1)),
                load_words(&xor_block(&result[off + 2 * BLOCK_SIZE..off + 3 * BLOCK_SIZE], &t2)),
                load_words(&xor_block(&result[off + 3 * BLOCK_SIZE..off + 4 * BLOCK_SIZE], &t3)),
            ];
            if encrypt {
                encrypt_states_4way(&mut s, keys, rounds);
            } else {
                decrypt_states_4way(&mut s, keys, rounds);
            }
            let sb0 = store_words(&s[0]);
            let sb1 = store_words(&s[1]);
            let sb2 = store_words(&s[2]);
            let sb3 = store_words(&s[3]);
            for i in 0..BLOCK_SIZE {
                result[off + i] = sb0[i] ^ t0[i];
                result[off + BLOCK_SIZE + i] = sb1[i] ^ t1[i];
                result[off + 2 * BLOCK_SIZE + i] = sb2[i] ^ t2[i];
                result[off + 3 * BLOCK_SIZE + i] = sb3[i] ^ t3[i];
            }
            t = t3;
            gf_mul2_256(&mut t);
            b += 4;
        }
        while b < count {
            let off = b * BLOCK_SIZE;
            let mut block = [0u8; BLOCK_SIZE];
            for i in 0..BLOCK_SIZE {
                block[i] = result[off + i] ^ t[i];
            }
            let enc = if encrypt {
                cipher.encrypt_block(&block)
            } else {
                cipher.decrypt_block(&block)
            };
            for i in 0..BLOCK_SIZE {
                result[off + i] = enc[i] ^ t[i];
            }
            gf_mul2_256(&mut t);
            b += 1;
        }
    }
}

fn xts_process(key: &[u8; 64], data: &[u8], sector: u64, encrypt: bool) -> Vec<u8> {
    let key1 = &key[..KEY_SIZE];
    let key2 = &key[KEY_SIZE..];
    let block1 = BlockCipher::new(key1);
    let block2 = BlockCipher::new(key2);
    let n = data.len();
    let mut result = data.to_vec();

    let mut base_tweak = [0u8; BLOCK_SIZE];
    base_tweak[24..32].copy_from_slice(&sector.to_le_bytes());
    base_tweak = block2.encrypt_block(&base_tweak);

    let full_blocks = n / BLOCK_SIZE;
    let partial = n % BLOCK_SIZE;
    let process_blocks = if partial > 0 {
        full_blocks.saturating_sub(1)
    } else {
        full_blocks
    };

    if process_blocks > 0 {
        let data_len = process_blocks * BLOCK_SIZE;
        if data_len >= CTR_PARALLEL_THRESHOLD && process_blocks >= 256 {
            let workers = get_optimal_worker_count(data_len);
            if workers > 1 {
                let blocks_per_worker = process_blocks.div_ceil(workers);
                // 闭包只借 cipher 引用，不转移所有权，后面 CTS 还要用
                let block1_ref = &block1;
                std::thread::scope(|s| {
                    let mut res_rem = result.as_mut_slice();
                    let mut off = 0usize;
                    for w in 0..workers {
                        let start_block = w * blocks_per_worker;
                        if start_block >= process_blocks {
                            break;
                        }
                        let mut end_block = start_block + blocks_per_worker;
                        if end_block > process_blocks {
                            end_block = process_blocks;
                        }
                        let end_off = end_block * BLOCK_SIZE;
                        let this_len = end_off - off;
                        let (this_res, rest_res) = res_rem.split_at_mut(this_len);
                        // 区间首块的 tweak 由基础 tweak 按块序号独立推出，不依赖前一区间
                        let start_tweak = xts_tweak_at(&base_tweak, start_block);
                        s.spawn(move || {
                            xts_process_range(
                                block1_ref,
                                this_res,
                                end_block - start_block,
                                &start_tweak,
                                encrypt,
                            );
                        });
                        res_rem = rest_res;
                        off = end_off;
                    }
                });
            } else {
                xts_process_range(&block1, &mut result, process_blocks, &base_tweak, encrypt);
            }
        } else {
            xts_process_range(&block1, &mut result, process_blocks, &base_tweak, encrypt);
        }
    }

    // CTS 末尾块依赖整段最后一个完整块的 tweak，并行处理完再顺序做
    if partial > 0 && full_blocks >= 1 {
        let tweak = xts_tweak_at(&base_tweak, process_blocks);
        let prev_off = (full_blocks - 1) * BLOCK_SIZE;
        let last_off = full_blocks * BLOCK_SIZE;
        if encrypt {
            xts_cts_encrypt(&block1, &mut result, prev_off, last_off, partial, &tweak);
        } else {
            xts_cts_decrypt(&block1, &mut result, prev_off, last_off, partial, &tweak);
        }
    }
    result
}

fn xts_cts_encrypt(
    block: &BlockCipher,
    data: &mut [u8],
    prev_off: usize,
    last_off: usize,
    m: usize,
    tweak: &[u8; BLOCK_SIZE],
) {
    let mut p_partial = [0u8; BLOCK_SIZE];
    p_partial[..m].copy_from_slice(&data[last_off..last_off + m]);
    let mut p_prev = [0u8; BLOCK_SIZE];
    p_prev.copy_from_slice(&data[prev_off..prev_off + BLOCK_SIZE]);

    let mut cc = p_prev;
    for i in 0..BLOCK_SIZE {
        cc[i] ^= tweak[i];
    }
    cc = block.encrypt_block(&cc);
    for i in 0..BLOCK_SIZE {
        cc[i] ^= tweak[i];
    }
    data[last_off..last_off + m].copy_from_slice(&cc[..m]);

    let mut pp = [0u8; BLOCK_SIZE];
    pp[..BLOCK_SIZE - m].copy_from_slice(&cc[m..BLOCK_SIZE]);
    pp[BLOCK_SIZE - m..].copy_from_slice(&p_partial[..m]);

    let mut tweak_n = *tweak;
    gf_mul2_256(&mut tweak_n);
    for i in 0..BLOCK_SIZE {
        pp[i] ^= tweak_n[i];
    }
    pp = block.encrypt_block(&pp);
    for i in 0..BLOCK_SIZE {
        pp[i] ^= tweak_n[i];
    }
    data[prev_off..prev_off + BLOCK_SIZE].copy_from_slice(&pp);
}

fn xts_cts_decrypt(
    block: &BlockCipher,
    data: &mut [u8],
    prev_off: usize,
    last_off: usize,
    m: usize,
    tweak: &[u8; BLOCK_SIZE],
) {
    let mut c_partial = [0u8; BLOCK_SIZE];
    c_partial[..m].copy_from_slice(&data[last_off..last_off + m]);

    let mut tweak_n1 = *tweak;
    gf_mul2_256(&mut tweak_n1);

    let mut pp = [0u8; BLOCK_SIZE];
    pp.copy_from_slice(&data[prev_off..prev_off + BLOCK_SIZE]);
    for i in 0..BLOCK_SIZE {
        pp[i] ^= tweak_n1[i];
    }
    pp = block.decrypt_block(&pp);
    for i in 0..BLOCK_SIZE {
        pp[i] ^= tweak_n1[i];
    }
    data[last_off..last_off + m].copy_from_slice(&pp[BLOCK_SIZE - m..BLOCK_SIZE]);

    let mut cc = [0u8; BLOCK_SIZE];
    cc[..m].copy_from_slice(&c_partial[..m]);
    cc[m..].copy_from_slice(&pp[..BLOCK_SIZE - m]);
    for i in 0..BLOCK_SIZE {
        cc[i] ^= tweak[i];
    }
    cc = block.decrypt_block(&cc);
    for i in 0..BLOCK_SIZE {
        cc[i] ^= tweak[i];
    }
    data[prev_off..prev_off + BLOCK_SIZE].copy_from_slice(&cc);
}

// AEAD

fn derive_aead_keys(
    master: &[u8; KEY_SIZE],
    nonce: &[u8; NONCE_SIZE],
) -> ([u8; KEY_SIZE], [u8; KEY_SIZE]) {
    let mut prk = hkdf_extract(master, nonce);
    let mut enc = hkdf_expand(&prk, b"aead-key", KEY_SIZE);
    let mut mac = hkdf_expand(&prk, b"tag-key", KEY_SIZE);
    let mut enc_k = [0u8; KEY_SIZE];
    let mut mac_k = [0u8; KEY_SIZE];
    enc_k.copy_from_slice(&enc);
    mac_k.copy_from_slice(&mac);
    // 中间量都是密钥材料，复制完就抹掉，别留在堆上
    secure_zero(&mut prk);
    secure_zero(&mut enc);
    secure_zero(&mut mac);
    (enc_k, mac_k)
}

/// AEAD 认证加密，输入走流式，输出为 密文 || tag
pub fn aead_seal<R: Read, W: Write>(
    key: &[u8; KEY_SIZE],
    nonce: &[u8; NONCE_SIZE],
    input: &mut R,
    output: &mut W,
    aad: &[u8],
) -> io::Result<()> {
    let (enc_key, mac_key) = derive_aead_keys(key, nonce);
    let enc_key = ZeroizingKey(enc_key);
    let mac_key = ZeroizingKey(mac_key);
    let cipher = BlockCipher::new(enc_key.as_ref());

    let mut counter = [0u8; BLOCK_SIZE];
    counter[..CTR_COUNTER_SIZE].copy_from_slice(&nonce[..CTR_COUNTER_SIZE]);

    // 双缓冲：主线程 CTR 加密，后台线程并行认证密文算 inner MAC。
    // mac_tx 传 (槽位, 长度)，free_rx 回传被认证完的槽位供复用。
    // 槽位缓冲用 Mutex 保护，主线程加密与 MAC 线程读取不同步访问同一槽
    let ct_bufs: [Mutex<Vec<u8>>; 2] = [
        Mutex::new(vec![0u8; SEAL_SLOT_SIZE]),
        Mutex::new(vec![0u8; SEAL_SLOT_SIZE]),
    ];
    let (mac_tx, mac_rx) = mpsc::channel::<(usize, usize)>();
    let (free_tx, free_rx) = mpsc::channel::<usize>();

    // 读缓冲按线程复用，小消息不必每次开 4MB 再清零
    let result = with_stream_buf_a(|read_buf| std::thread::scope(|s| {
        // mac_rx 必须 move 进 MAC 线程；其余共享对象用引用捕获
        let ct_bufs_ref = &ct_bufs;
        let mac_key_ref = &mac_key;
        let aad_ref = &aad;
        let free_tx_ref = &free_tx;
        let mac_handle = s.spawn(move || {
            let mut mac = Hmac::new(mac_key_ref.as_ref());
            mac.write(aad_ref);
            while let Ok((slot, len)) = mac_rx.recv() {
                let buf = ct_bufs_ref[slot].lock().unwrap();
                mac.write(&buf[..len]);
                drop(buf);
                let _ = free_tx_ref.send(slot);
            }
            mac
        });

        // 主线程：读明文 → CTR 加密到 ct_bufs[slot] → 发 MAC 线程 → 写出密文
        let mut block_idx: u64 = 0;
        let mut ct_len: u64 = 0;
        let mut free_slots: Vec<usize> = vec![1, 0];
        let mut io_result: io::Result<()> = Ok(());
        'main: loop {
            let slot = match free_slots.pop() {
                Some(s) => s,
                None => match free_rx.recv() {
                    Ok(s) => s,
                    Err(_) => break,
                },
            };
            // 聚合读取到槽位读满或 EOF：io::Read 允许单次短读，管道等输入
            // 若按单次返回量判 EOF 会把后续数据截掉，必须以 0 字节返回为准
            let mut n = 0usize;
            while n < SEAL_SLOT_SIZE {
                match input.read(&mut read_buf[n..SEAL_SLOT_SIZE]) {
                    Ok(0) => break,
                    Ok(k) => n += k,
                    Err(e) => {
                        io_result = Err(e);
                        break 'main;
                    }
                }
            }
            if n == 0 {
                break;
            }
            let mut buf = ct_bufs[slot].lock().unwrap();
            // 完整块批量加密
            let full_blocks = n / BLOCK_SIZE;
            let block_bytes = full_blocks * BLOCK_SIZE;
            if full_blocks > 0 {
                counter[CTR_COUNTER_SIZE..].copy_from_slice(&block_idx.to_le_bytes());
                ctr_crypt_blocks(
                    &cipher,
                    &mut buf[..block_bytes],
                    &read_buf[..block_bytes],
                    &counter,
                );
                block_idx += full_blocks as u64;
            }
            // 尾部不足一块，顺序单块处理
            let mut idx = block_idx;
            let mut pos = block_bytes;
            while pos < n {
                counter[CTR_COUNTER_SIZE..].copy_from_slice(&idx.to_le_bytes());
                let ks = cipher.encrypt_block(&counter);
                let take = (BLOCK_SIZE).min(n - pos);
                for i in 0..take {
                    buf[pos + i] = read_buf[pos + i] ^ ks[i];
                }
                pos += take;
                idx += 1;
            }
            block_idx = idx;
            ct_len += n as u64;
            drop(buf);
            if mac_tx.send((slot, n)).is_err() {
                break;
            }
            if let Err(e) = output.write_all(&ct_bufs[slot].lock().unwrap()[..n]) {
                io_result = Err(e);
                break 'main;
            }
            // 本次没读满一个槽位说明输入到末尾了，收尾
            if n < SEAL_SLOT_SIZE {
                break;
            }
        }
        drop(mac_tx);
        let mut mac = mac_handle.join().unwrap_or_else(|_| unreachable!());
        // 收尾：把 aad/密文长度写进 MAC 算出 tag 拼到末尾
        let mut len_buf = [0u8; 16];
        len_buf[..8].copy_from_slice(&(aad.len() as u64).to_le_bytes());
        len_buf[8..].copy_from_slice(&ct_len.to_le_bytes());
        mac.write(&len_buf);
        let sum = mac.sum();
        if io_result.is_ok() {
            output.write_all(&sum[..GCM_TAG_SIZE])?;
        }
        io_result
    }));
    result
}

/// AEAD 认证解密，输入走流式（密文 || tag），返回 tag 校验结果
pub fn aead_open<R: Read, W: Write>(
    key: &[u8; KEY_SIZE],
    nonce: &[u8; NONCE_SIZE],
    input: &mut R,
    output: &mut W,
    aad: &[u8],
) -> io::Result<bool> {
    let (enc_key, mac_key) = derive_aead_keys(key, nonce);
    let enc_key = ZeroizingKey(enc_key);
    let mac_key = ZeroizingKey(mac_key);
    let cipher = BlockCipher::new(enc_key.as_ref());
    // 认证用的 MAC，先把 aad 写进去
    let mut mac = Hmac::new(mac_key.as_ref());
    mac.write(aad);

    let mut counter = [0u8; BLOCK_SIZE];
    counter[..CTR_COUNTER_SIZE].copy_from_slice(&nonce[..CTR_COUNTER_SIZE]);
    let mut block_idx: u64 = 0;
    let mut ct_len: u64 = 0;
    // 读写缓冲 1MB 一块：配合分块批处理够用，又不用每次调用都清零几 MB
    let mut read_buf = vec![0u8; SEAL_SLOT_SIZE];
    let mut write_buf = vec![0u8; SEAL_SLOT_SIZE];
    // 末尾 16 字节是 tag，先留在缓冲里不解密，等输入读完再比对
    let mut pending: Vec<u8> = Vec::with_capacity(BLOCK_SIZE + GCM_TAG_SIZE);

    loop {
        let n = input.read(&mut read_buf)?;
        if n == 0 {
            break;
        }
        pending.extend_from_slice(&read_buf[..n]);
        // 只解掉不包含 tag 的完整块，认证明文密文并写出
        let avail = pending.len().saturating_sub(GCM_TAG_SIZE);
        let processable = (avail / BLOCK_SIZE) * BLOCK_SIZE;
        if processable > 0 {
            mac.write(&pending[..processable]);
            let full = processable / BLOCK_SIZE;
            counter[CTR_COUNTER_SIZE..].copy_from_slice(&block_idx.to_le_bytes());
            ctr_crypt_blocks(
                &cipher,
                &mut write_buf[..processable],
                &pending[..processable],
                &counter,
            );
            block_idx += full as u64;
            ct_len += processable as u64;
            output.write_all(&write_buf[..processable])?;
            pending.drain(..processable);
        }
    }

    // 密文不够一个 tag 长度就没法校验，按数据非法报错而不是返回"未通过"
    if pending.len() < GCM_TAG_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bastion AEAD: data too short",
        ));
    }
    // 尾声不足一块的部分顺次解密并认证
    let tail_ct = pending.len() - GCM_TAG_SIZE;
    if tail_ct > 0 {
        mac.write(&pending[..tail_ct]);
        let mut idx = block_idx;
        let mut pos = 0usize;
        while pos < tail_ct {
            counter[CTR_COUNTER_SIZE..].copy_from_slice(&idx.to_le_bytes());
            let ks = cipher.encrypt_block(&counter);
            let take = (BLOCK_SIZE).min(tail_ct - pos);
            for i in 0..take {
                write_buf[pos + i] = pending[pos + i] ^ ks[i];
            }
            pos += take;
            idx += 1;
        }
        ct_len += tail_ct as u64;
        output.write_all(&write_buf[..tail_ct])?;
    }
    let expected = &pending[tail_ct..];
    let mut len_buf = [0u8; 16];
    len_buf[..8].copy_from_slice(&(aad.len() as u64).to_le_bytes());
    len_buf[8..].copy_from_slice(&ct_len.to_le_bytes());
    mac.write(&len_buf);
    let sum = mac.sum();
    // 常量时间比较标签，防止计时侧信道泄露匹配进度
    let mut diff = 0u8;
    for i in 0..GCM_TAG_SIZE {
        diff |= sum[i] ^ expected[i];
    }
    // 校验失败按错误返回：明文在上面已经写进输出，返回 Ok(false) 很容易被调用方
    // 当成"成功但内容为空"，必须给出明确的失败信号
    if diff != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bastion AEAD: authentication failed",
        ));
    }
    Ok(true)
}

// 便捷封装：encrypt/decrypt 等价于 CTR 流式加解密

pub fn encrypt<R: Read, W: Write>(
    key: &[u8],
    nonce: &[u8],
    input: &mut R,
    output: &mut W,
) -> io::Result<()> {
    ctr_crypt(key, nonce, input, output)
}

pub fn decrypt<R: Read, W: Write>(
    key: &[u8],
    nonce: &[u8],
    input: &mut R,
    output: &mut W,
) -> io::Result<()> {
    ctr_crypt(key, nonce, input, output)
}
