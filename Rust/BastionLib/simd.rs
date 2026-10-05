//! Bastion SIMD 硬件加速（x86_64）：N 块的同一状态字放到同一向量寄存器，一次处理 N 块
//! SSE2 在 x86_64 平台默认可用，AVX2 需运行时检测

// SIMD 批量函数需要操作 8 个并行向量寄存器，参数数量超出常规限制
#![allow(clippy::too_many_arguments)]

use core::arch::x86_64::*;
use core::sync::atomic::{AtomicI32, Ordering};

use crate::{BlockCipher, BLOCK_SIZE, MAX_ROUND_KEYS};

// ===== SIMD 级别检测 =====

static SIMD_LEVEL: AtomicI32 = AtomicI32::new(0);
static SIMD_INIT: std::sync::Once = std::sync::Once::new();

fn get_simd_level() -> i32 {
    SIMD_INIT.call_once(|| {
        let level = detect_simd_level();
        SIMD_LEVEL.store(level, Ordering::Relaxed);
    });
    SIMD_LEVEL.load(Ordering::Relaxed)
}

fn detect_simd_level() -> i32 {
    let mut level = 1;
    if is_x86_feature_detected!("avx2") {
        level = 2;
    }
    level
}

// ===== SSE2 旋转辅助（宏生成各旋转量的独立函数） =====

macro_rules! define_rotr32_sse2 {
    ($($name:ident = $r:expr),* $(,)?) => {
        $(
            #[inline]
            unsafe fn $name(x: __m128i) -> __m128i {
                _mm_or_si128(_mm_srli_epi32(x, $r), _mm_slli_epi32(x, 32 - $r))
            }
        )*
    }
}
macro_rules! define_rotl32_sse2 {
    ($($name:ident = $r:expr),* $(,)?) => {
        $(
            #[inline]
            unsafe fn $name(x: __m128i) -> __m128i {
                _mm_or_si128(_mm_slli_epi32(x, $r), _mm_srli_epi32(x, 32 - $r))
            }
        )*
    }
}

define_rotr32_sse2! {
    rotr32_sse2_r5 = 7,  rotr32_sse2_r6 = 11, rotr32_sse2_r7 = 5,  rotr32_sse2_r8 = 15,
    rotr32_sse2_r1 = 13, rotr32_sse2_r2 = 19, rotr32_sse2_r3 = 23, rotr32_sse2_r4 = 29,
    rotr32_sse2_r9 = 17,
    rotr32_sse2_hr5 = 3, rotr32_sse2_hr6 = 11, rotr32_sse2_hr7 = 7, rotr32_sse2_hr8 = 13,
    rotr32_sse2_hr1 = 9, rotr32_sse2_hr2 = 17, rotr32_sse2_hr3 = 21, rotr32_sse2_hr4 = 27,
    rotr32_sse2_hr9 = 25,
}
define_rotl32_sse2! {
    rotl32_sse2_r1 = 13, rotl32_sse2_r2 = 19, rotl32_sse2_r3 = 23, rotl32_sse2_r4 = 29,
    rotl32_sse2_hr1 = 9, rotl32_sse2_hr2 = 17, rotl32_sse2_hr3 = 21, rotl32_sse2_hr4 = 27,
}

// ===== SSE2 G_Mix / H_Mix =====

#[inline]
unsafe fn g_mix_sse2(
    r0: &mut __m128i, r1: &mut __m128i, r2: &mut __m128i, r3: &mut __m128i,
    r4: &mut __m128i, r5: &mut __m128i, r6: &mut __m128i, r7: &mut __m128i,
) {
    *r0 = _mm_add_epi32(*r0, rotr32_sse2_r5(*r2));
    *r1 = _mm_add_epi32(*r1, rotr32_sse2_r5(*r3));
    *r6 = _mm_xor_si128(*r6, *r0);
    *r7 = _mm_xor_si128(*r7, *r1);
    *r6 = rotr32_sse2_r1(*r6); *r7 = rotr32_sse2_r1(*r7);
    *r4 = _mm_add_epi32(*r4, rotr32_sse2_r6(*r6));
    *r5 = _mm_add_epi32(*r5, rotr32_sse2_r6(*r7));
    *r2 = _mm_xor_si128(*r2, *r4); *r3 = _mm_xor_si128(*r3, *r5);
    *r2 = rotr32_sse2_r2(*r2); *r3 = rotr32_sse2_r2(*r3);
    *r0 = _mm_xor_si128(*r0, rotr32_sse2_r9(*r4));
    *r1 = _mm_xor_si128(*r1, rotr32_sse2_r9(*r5));
    *r0 = _mm_add_epi32(*r0, rotr32_sse2_r7(*r2));
    *r1 = _mm_add_epi32(*r1, rotr32_sse2_r7(*r3));
    *r6 = _mm_xor_si128(*r6, *r0); *r7 = _mm_xor_si128(*r7, *r1);
    *r6 = rotr32_sse2_r3(*r6); *r7 = rotr32_sse2_r3(*r7);
    *r4 = _mm_add_epi32(*r4, rotr32_sse2_r8(*r6));
    *r5 = _mm_add_epi32(*r5, rotr32_sse2_r8(*r7));
    *r2 = _mm_xor_si128(*r2, *r4); *r3 = _mm_xor_si128(*r3, *r5);
    *r2 = rotr32_sse2_r4(*r2); *r3 = rotr32_sse2_r4(*r3);
}

#[inline]
unsafe fn h_mix_sse2(
    r0: &mut __m128i, r1: &mut __m128i, r2: &mut __m128i, r3: &mut __m128i,
    r4: &mut __m128i, r5: &mut __m128i, r6: &mut __m128i, r7: &mut __m128i,
) {
    *r0 = _mm_add_epi32(*r0, rotr32_sse2_hr5(*r2));
    *r1 = _mm_add_epi32(*r1, rotr32_sse2_hr5(*r3));
    *r6 = _mm_xor_si128(*r6, *r0); *r7 = _mm_xor_si128(*r7, *r1);
    *r6 = rotr32_sse2_hr1(*r6); *r7 = rotr32_sse2_hr1(*r7);
    *r4 = _mm_add_epi32(*r4, rotr32_sse2_hr6(*r6));
    *r5 = _mm_add_epi32(*r5, rotr32_sse2_hr6(*r7));
    *r2 = _mm_xor_si128(*r2, *r4); *r3 = _mm_xor_si128(*r3, *r5);
    *r2 = rotr32_sse2_hr2(*r2); *r3 = rotr32_sse2_hr2(*r3);
    *r0 = _mm_xor_si128(*r0, rotr32_sse2_hr9(*r4));
    *r1 = _mm_xor_si128(*r1, rotr32_sse2_hr9(*r5));
    *r0 = _mm_add_epi32(*r0, rotr32_sse2_hr7(*r2));
    *r1 = _mm_add_epi32(*r1, rotr32_sse2_hr7(*r3));
    *r6 = _mm_xor_si128(*r6, *r0); *r7 = _mm_xor_si128(*r7, *r1);
    *r6 = rotr32_sse2_hr3(*r6); *r7 = rotr32_sse2_hr3(*r7);
    *r4 = _mm_add_epi32(*r4, rotr32_sse2_hr8(*r6));
    *r5 = _mm_add_epi32(*r5, rotr32_sse2_hr8(*r7));
    *r2 = _mm_xor_si128(*r2, *r4); *r3 = _mm_xor_si128(*r3, *r5);
    *r2 = rotr32_sse2_hr4(*r2); *r3 = rotr32_sse2_hr4(*r3);
}

// ===== SSE2 逆 G_Mix / H_Mix =====

#[inline]
unsafe fn inv_g_mix_sse2(
    r0: &mut __m128i, r1: &mut __m128i, r2: &mut __m128i, r3: &mut __m128i,
    r4: &mut __m128i, r5: &mut __m128i, r6: &mut __m128i, r7: &mut __m128i,
) {
    *r2 = _mm_xor_si128(rotl32_sse2_r4(*r2), *r4);
    *r3 = _mm_xor_si128(rotl32_sse2_r4(*r3), *r5);
    *r4 = _mm_sub_epi32(*r4, rotr32_sse2_r8(*r6));
    *r5 = _mm_sub_epi32(*r5, rotr32_sse2_r8(*r7));
    *r6 = _mm_xor_si128(rotl32_sse2_r3(*r6), *r0);
    *r7 = _mm_xor_si128(rotl32_sse2_r3(*r7), *r1);
    *r0 = _mm_sub_epi32(*r0, rotr32_sse2_r7(*r2));
    *r1 = _mm_sub_epi32(*r1, rotr32_sse2_r7(*r3));
    *r0 = _mm_xor_si128(*r0, rotr32_sse2_r9(*r4));
    *r1 = _mm_xor_si128(*r1, rotr32_sse2_r9(*r5));
    *r2 = _mm_xor_si128(rotl32_sse2_r2(*r2), *r4);
    *r3 = _mm_xor_si128(rotl32_sse2_r2(*r3), *r5);
    *r4 = _mm_sub_epi32(*r4, rotr32_sse2_r6(*r6));
    *r5 = _mm_sub_epi32(*r5, rotr32_sse2_r6(*r7));
    *r6 = _mm_xor_si128(rotl32_sse2_r1(*r6), *r0);
    *r7 = _mm_xor_si128(rotl32_sse2_r1(*r7), *r1);
    *r0 = _mm_sub_epi32(*r0, rotr32_sse2_r5(*r2));
    *r1 = _mm_sub_epi32(*r1, rotr32_sse2_r5(*r3));
}

#[inline]
unsafe fn inv_h_mix_sse2(
    r0: &mut __m128i, r1: &mut __m128i, r2: &mut __m128i, r3: &mut __m128i,
    r4: &mut __m128i, r5: &mut __m128i, r6: &mut __m128i, r7: &mut __m128i,
) {
    *r2 = _mm_xor_si128(rotl32_sse2_hr4(*r2), *r4);
    *r3 = _mm_xor_si128(rotl32_sse2_hr4(*r3), *r5);
    *r4 = _mm_sub_epi32(*r4, rotr32_sse2_hr8(*r6));
    *r5 = _mm_sub_epi32(*r5, rotr32_sse2_hr8(*r7));
    *r6 = _mm_xor_si128(rotl32_sse2_hr3(*r6), *r0);
    *r7 = _mm_xor_si128(rotl32_sse2_hr3(*r7), *r1);
    *r0 = _mm_sub_epi32(*r0, rotr32_sse2_hr7(*r2));
    *r1 = _mm_sub_epi32(*r1, rotr32_sse2_hr7(*r3));
    *r0 = _mm_xor_si128(*r0, rotr32_sse2_hr9(*r4));
    *r1 = _mm_xor_si128(*r1, rotr32_sse2_hr9(*r5));
    *r2 = _mm_xor_si128(rotl32_sse2_hr2(*r2), *r4);
    *r3 = _mm_xor_si128(rotl32_sse2_hr2(*r3), *r5);
    *r4 = _mm_sub_epi32(*r4, rotr32_sse2_hr6(*r6));
    *r5 = _mm_sub_epi32(*r5, rotr32_sse2_hr6(*r7));
    *r6 = _mm_xor_si128(rotl32_sse2_hr1(*r6), *r0);
    *r7 = _mm_xor_si128(rotl32_sse2_hr1(*r7), *r1);
    *r0 = _mm_sub_epi32(*r0, rotr32_sse2_hr5(*r2));
    *r1 = _mm_sub_epi32(*r1, rotr32_sse2_hr5(*r3));
}

// ===== SSE2 4x4 转置 =====

#[inline]
unsafe fn transpose4x4_sse2(
    r0: __m128i, r1: __m128i, r2: __m128i, r3: __m128i,
    o0: &mut __m128i, o1: &mut __m128i, o2: &mut __m128i, o3: &mut __m128i,
) {
    let t0 = _mm_unpacklo_epi32(r0, r1);
    let t1 = _mm_unpackhi_epi32(r0, r1);
    let t2 = _mm_unpacklo_epi32(r2, r3);
    let t3 = _mm_unpackhi_epi32(r2, r3);
    *o0 = _mm_unpacklo_epi64(t0, t2);
    *o1 = _mm_unpackhi_epi64(t0, t2);
    *o2 = _mm_unpacklo_epi64(t1, t3);
    *o3 = _mm_unpackhi_epi64(t1, t3);
}

// ===== SSE2 16 轮批量加密核心 =====

#[inline]
unsafe fn sse2_rounds_encrypt(
    r0: &mut __m128i, r1: &mut __m128i, r2: &mut __m128i, r3: &mut __m128i,
    r4: &mut __m128i, r5: &mut __m128i, r6: &mut __m128i, r7: &mut __m128i,
    keys: &[u32; MAX_ROUND_KEYS * 2], rounds: usize,
) {
    *r0 = _mm_xor_si128(*r0, _mm_set1_epi32(keys[0] as i32));
    *r1 = _mm_xor_si128(*r1, _mm_set1_epi32(keys[1] as i32));
    *r2 = _mm_xor_si128(*r2, _mm_set1_epi32(keys[2] as i32));
    *r3 = _mm_xor_si128(*r3, _mm_set1_epi32(keys[3] as i32));
    *r4 = _mm_xor_si128(*r4, _mm_set1_epi32(keys[4] as i32));
    *r5 = _mm_xor_si128(*r5, _mm_set1_epi32(keys[5] as i32));
    *r6 = _mm_xor_si128(*r6, _mm_set1_epi32(keys[6] as i32));
    *r7 = _mm_xor_si128(*r7, _mm_set1_epi32(keys[7] as i32));

    let mut ki = 4usize;
    for r in 0..rounds {
        if (r & 1) == 0 {
            g_mix_sse2(r0, r1, r2, r3, r4, r5, r6, r7);
        } else {
            h_mix_sse2(r0, r1, r2, r3, r4, r5, r6, r7);
        }
        // ShiftRows：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
        let t = *r0; *r0 = *r2; *r2 = *r7; *r7 = *r1;
        *r1 = *r5; *r5 = *r4; *r4 = *r3; *r3 = *r6; *r6 = t;
        *r0 = _mm_xor_si128(*r0, _mm_set1_epi32(keys[2 * ki] as i32));
        *r1 = _mm_xor_si128(*r1, _mm_set1_epi32(keys[2 * ki + 1] as i32));
        *r2 = _mm_xor_si128(*r2, _mm_set1_epi32(keys[2 * (ki + 1)] as i32));
        *r3 = _mm_xor_si128(*r3, _mm_set1_epi32(keys[2 * (ki + 1) + 1] as i32));
        *r4 = _mm_xor_si128(*r4, _mm_set1_epi32(keys[2 * (ki + 2)] as i32));
        *r5 = _mm_xor_si128(*r5, _mm_set1_epi32(keys[2 * (ki + 2) + 1] as i32));
        *r6 = _mm_xor_si128(*r6, _mm_set1_epi32(keys[2 * (ki + 3)] as i32));
        *r7 = _mm_xor_si128(*r7, _mm_set1_epi32(keys[2 * (ki + 3) + 1] as i32));
        ki += 4;
    }
    let ki_out = 4 + rounds * 4;
    *r0 = _mm_xor_si128(*r0, _mm_set1_epi32(keys[2 * ki_out] as i32));
    *r1 = _mm_xor_si128(*r1, _mm_set1_epi32(keys[2 * ki_out + 1] as i32));
    *r2 = _mm_xor_si128(*r2, _mm_set1_epi32(keys[2 * (ki_out + 1)] as i32));
    *r3 = _mm_xor_si128(*r3, _mm_set1_epi32(keys[2 * (ki_out + 1) + 1] as i32));
    *r4 = _mm_xor_si128(*r4, _mm_set1_epi32(keys[2 * (ki_out + 2)] as i32));
    *r5 = _mm_xor_si128(*r5, _mm_set1_epi32(keys[2 * (ki_out + 2) + 1] as i32));
    *r6 = _mm_xor_si128(*r6, _mm_set1_epi32(keys[2 * (ki_out + 3)] as i32));
    *r7 = _mm_xor_si128(*r7, _mm_set1_epi32(keys[2 * (ki_out + 3) + 1] as i32));
}

// ===== SSE2 16 轮批量解密核心 =====

#[inline]
unsafe fn sse2_rounds_decrypt(
    r0: &mut __m128i, r1: &mut __m128i, r2: &mut __m128i, r3: &mut __m128i,
    r4: &mut __m128i, r5: &mut __m128i, r6: &mut __m128i, r7: &mut __m128i,
    keys: &[u32; MAX_ROUND_KEYS * 2], rounds: usize,
) {
    let ki = 4 + rounds * 4;
    *r0 = _mm_xor_si128(*r0, _mm_set1_epi32(keys[2 * ki] as i32));
    *r1 = _mm_xor_si128(*r1, _mm_set1_epi32(keys[2 * ki + 1] as i32));
    *r2 = _mm_xor_si128(*r2, _mm_set1_epi32(keys[2 * (ki + 1)] as i32));
    *r3 = _mm_xor_si128(*r3, _mm_set1_epi32(keys[2 * (ki + 1) + 1] as i32));
    *r4 = _mm_xor_si128(*r4, _mm_set1_epi32(keys[2 * (ki + 2)] as i32));
    *r5 = _mm_xor_si128(*r5, _mm_set1_epi32(keys[2 * (ki + 2) + 1] as i32));
    *r6 = _mm_xor_si128(*r6, _mm_set1_epi32(keys[2 * (ki + 3)] as i32));
    *r7 = _mm_xor_si128(*r7, _mm_set1_epi32(keys[2 * (ki + 3) + 1] as i32));

    for r in (0..rounds).rev() {
        let ki = 4 + r * 4;
        *r0 = _mm_xor_si128(*r0, _mm_set1_epi32(keys[2 * ki] as i32));
        *r1 = _mm_xor_si128(*r1, _mm_set1_epi32(keys[2 * ki + 1] as i32));
        *r2 = _mm_xor_si128(*r2, _mm_set1_epi32(keys[2 * (ki + 1)] as i32));
        *r3 = _mm_xor_si128(*r3, _mm_set1_epi32(keys[2 * (ki + 1) + 1] as i32));
        *r4 = _mm_xor_si128(*r4, _mm_set1_epi32(keys[2 * (ki + 2)] as i32));
        *r5 = _mm_xor_si128(*r5, _mm_set1_epi32(keys[2 * (ki + 2) + 1] as i32));
        *r6 = _mm_xor_si128(*r6, _mm_set1_epi32(keys[2 * (ki + 3)] as i32));
        *r7 = _mm_xor_si128(*r7, _mm_set1_epi32(keys[2 * (ki + 3) + 1] as i32));
        // 逆 ShiftRows：8-循环置换的逆
        let t = *r0; *r0 = *r6; *r6 = *r3; *r3 = *r4;
        *r4 = *r5; *r5 = *r1; *r1 = *r7; *r7 = *r2; *r2 = t;
        if (r & 1) == 0 {
            inv_g_mix_sse2(r0, r1, r2, r3, r4, r5, r6, r7);
        } else {
            inv_h_mix_sse2(r0, r1, r2, r3, r4, r5, r6, r7);
        }
    }
    *r0 = _mm_xor_si128(*r0, _mm_set1_epi32(keys[0] as i32));
    *r1 = _mm_xor_si128(*r1, _mm_set1_epi32(keys[1] as i32));
    *r2 = _mm_xor_si128(*r2, _mm_set1_epi32(keys[2] as i32));
    *r3 = _mm_xor_si128(*r3, _mm_set1_epi32(keys[3] as i32));
    *r4 = _mm_xor_si128(*r4, _mm_set1_epi32(keys[4] as i32));
    *r5 = _mm_xor_si128(*r5, _mm_set1_epi32(keys[5] as i32));
    *r6 = _mm_xor_si128(*r6, _mm_set1_epi32(keys[6] as i32));
    *r7 = _mm_xor_si128(*r7, _mm_set1_epi32(keys[7] as i32));
}

// ===== SSE2 CTR 批量加密（4块） =====

unsafe fn encrypt_blocks_batch4_sse2(
    keys: &[u32; MAX_ROUND_KEYS * 2], rounds: usize,
    dst: &mut [u8], src: &[u8], counter: &[u8; 32],
) {
    let n = src.len() / BLOCK_SIZE;
    if n < 4 {
        return;
    }

    let n0 = u32::from_le_bytes([counter[0], counter[1], counter[2], counter[3]]);
    let n1 = u32::from_le_bytes([counter[4], counter[5], counter[6], counter[7]]);
    let n2 = u32::from_le_bytes([counter[8], counter[9], counter[10], counter[11]]);
    let n3 = u32::from_le_bytes([counter[12], counter[13], counter[14], counter[15]]);
    let n4 = u32::from_le_bytes([counter[16], counter[17], counter[18], counter[19]]);
    let n5 = u32::from_le_bytes([counter[20], counter[21], counter[22], counter[23]]);
    let base_ctr = u64::from_le_bytes([counter[24], counter[25], counter[26], counter[27],
                                       counter[28], counter[29], counter[30], counter[31]]);

    let v_n0 = _mm_set1_epi32(n0 as i32);
    let v_n1 = _mm_set1_epi32(n1 as i32);
    let v_n2 = _mm_set1_epi32(n2 as i32);
    let v_n3 = _mm_set1_epi32(n3 as i32);
    let v_n4 = _mm_set1_epi32(n4 as i32);
    let v_n5 = _mm_set1_epi32(n5 as i32);

    let mut i = 0usize;
    while i + 3 < n {
        let c0 = base_ctr + i as u64;
        let c1 = base_ctr + i as u64 + 1;
        let c2 = base_ctr + i as u64 + 2;
        let c3 = base_ctr + i as u64 + 3;

        let mut r0 = v_n0; let mut r1 = v_n1;
        let mut r2 = v_n2; let mut r3 = v_n3;
        let mut r4 = v_n4; let mut r5 = v_n5;
        let mut r6 = _mm_set_epi32(c3 as i32, c2 as i32, c1 as i32, c0 as i32);
        let mut r7 = _mm_set_epi32((c3 >> 32) as i32, (c2 >> 32) as i32,
                                    (c1 >> 32) as i32, (c0 >> 32) as i32);

        sse2_rounds_encrypt(&mut r0, &mut r1, &mut r2, &mut r3,
                            &mut r4, &mut r5, &mut r6, &mut r7, keys, rounds);

        let mut out0_lo = _mm_setzero_si128();
        let mut out1_lo = _mm_setzero_si128();
        let mut out2_lo = _mm_setzero_si128();
        let mut out3_lo = _mm_setzero_si128();
        transpose4x4_sse2(r0, r1, r2, r3, &mut out0_lo, &mut out1_lo, &mut out2_lo, &mut out3_lo);
        let mut out0_hi = _mm_setzero_si128();
        let mut out1_hi = _mm_setzero_si128();
        let mut out2_hi = _mm_setzero_si128();
        let mut out3_hi = _mm_setzero_si128();
        transpose4x4_sse2(r4, r5, r6, r7, &mut out0_hi, &mut out1_hi, &mut out2_hi, &mut out3_hi);

        let s = src.as_ptr().add(i * BLOCK_SIZE) as *const __m128i;
        let d = dst.as_mut_ptr().add(i * BLOCK_SIZE) as *mut __m128i;

        let d0 = _mm_loadu_si128(s.add(0));
        _mm_storeu_si128(d.add(0), _mm_xor_si128(d0, out0_lo));
        let d0 = _mm_loadu_si128(s.add(1));
        _mm_storeu_si128(d.add(1), _mm_xor_si128(d0, out0_hi));
        let d0 = _mm_loadu_si128(s.add(2));
        _mm_storeu_si128(d.add(2), _mm_xor_si128(d0, out1_lo));
        let d0 = _mm_loadu_si128(s.add(3));
        _mm_storeu_si128(d.add(3), _mm_xor_si128(d0, out1_hi));
        let d0 = _mm_loadu_si128(s.add(4));
        _mm_storeu_si128(d.add(4), _mm_xor_si128(d0, out2_lo));
        let d0 = _mm_loadu_si128(s.add(5));
        _mm_storeu_si128(d.add(5), _mm_xor_si128(d0, out2_hi));
        let d0 = _mm_loadu_si128(s.add(6));
        _mm_storeu_si128(d.add(6), _mm_xor_si128(d0, out3_lo));
        let d0 = _mm_loadu_si128(s.add(7));
        _mm_storeu_si128(d.add(7), _mm_xor_si128(d0, out3_hi));

        i += 4;
    }
}

// ======================================================================
// AVX2 实现
// ======================================================================

macro_rules! define_rotr32_avx2 {
    ($($name:ident = $r:expr),* $(,)?) => {
        $(
            #[target_feature(enable = "avx2")]
            #[inline]
            unsafe fn $name(x: __m256i) -> __m256i {
                _mm256_or_si256(_mm256_srli_epi32(x, $r), _mm256_slli_epi32(x, 32 - $r))
            }
        )*
    }
}
macro_rules! define_rotl32_avx2 {
    ($($name:ident = $r:expr),* $(,)?) => {
        $(
            #[target_feature(enable = "avx2")]
            #[inline]
            unsafe fn $name(x: __m256i) -> __m256i {
                _mm256_or_si256(_mm256_slli_epi32(x, $r), _mm256_srli_epi32(x, 32 - $r))
            }
        )*
    }
}

define_rotr32_avx2! {
    rotr32_avx2_r5 = 7,  rotr32_avx2_r6 = 11, rotr32_avx2_r7 = 5,  rotr32_avx2_r8 = 15,
    rotr32_avx2_r1 = 13, rotr32_avx2_r2 = 19, rotr32_avx2_r3 = 23, rotr32_avx2_r4 = 29,
    rotr32_avx2_r9 = 17,
    rotr32_avx2_hr5 = 3, rotr32_avx2_hr6 = 11, rotr32_avx2_hr7 = 7, rotr32_avx2_hr8 = 13,
    rotr32_avx2_hr1 = 9, rotr32_avx2_hr2 = 17, rotr32_avx2_hr3 = 21, rotr32_avx2_hr4 = 27,
    rotr32_avx2_hr9 = 25,
}
define_rotl32_avx2! {
    rotl32_avx2_r1 = 13, rotl32_avx2_r2 = 19, rotl32_avx2_r3 = 23, rotl32_avx2_r4 = 29,
    rotl32_avx2_hr1 = 9, rotl32_avx2_hr2 = 17, rotl32_avx2_hr3 = 21, rotl32_avx2_hr4 = 27,
}

// ===== AVX2 G_Mix / H_Mix =====

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn g_mix_avx2(
    r0: &mut __m256i, r1: &mut __m256i, r2: &mut __m256i, r3: &mut __m256i,
    r4: &mut __m256i, r5: &mut __m256i, r6: &mut __m256i, r7: &mut __m256i,
) {
    *r0 = _mm256_add_epi32(*r0, rotr32_avx2_r5(*r2));
    *r1 = _mm256_add_epi32(*r1, rotr32_avx2_r5(*r3));
    *r6 = _mm256_xor_si256(*r6, *r0); *r7 = _mm256_xor_si256(*r7, *r1);
    *r6 = rotr32_avx2_r1(*r6); *r7 = rotr32_avx2_r1(*r7);
    *r4 = _mm256_add_epi32(*r4, rotr32_avx2_r6(*r6));
    *r5 = _mm256_add_epi32(*r5, rotr32_avx2_r6(*r7));
    *r2 = _mm256_xor_si256(*r2, *r4); *r3 = _mm256_xor_si256(*r3, *r5);
    *r2 = rotr32_avx2_r2(*r2); *r3 = rotr32_avx2_r2(*r3);
    *r0 = _mm256_xor_si256(*r0, rotr32_avx2_r9(*r4));
    *r1 = _mm256_xor_si256(*r1, rotr32_avx2_r9(*r5));
    *r0 = _mm256_add_epi32(*r0, rotr32_avx2_r7(*r2));
    *r1 = _mm256_add_epi32(*r1, rotr32_avx2_r7(*r3));
    *r6 = _mm256_xor_si256(*r6, *r0); *r7 = _mm256_xor_si256(*r7, *r1);
    *r6 = rotr32_avx2_r3(*r6); *r7 = rotr32_avx2_r3(*r7);
    *r4 = _mm256_add_epi32(*r4, rotr32_avx2_r8(*r6));
    *r5 = _mm256_add_epi32(*r5, rotr32_avx2_r8(*r7));
    *r2 = _mm256_xor_si256(*r2, *r4); *r3 = _mm256_xor_si256(*r3, *r5);
    *r2 = rotr32_avx2_r4(*r2); *r3 = rotr32_avx2_r4(*r3);
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn h_mix_avx2(
    r0: &mut __m256i, r1: &mut __m256i, r2: &mut __m256i, r3: &mut __m256i,
    r4: &mut __m256i, r5: &mut __m256i, r6: &mut __m256i, r7: &mut __m256i,
) {
    *r0 = _mm256_add_epi32(*r0, rotr32_avx2_hr5(*r2));
    *r1 = _mm256_add_epi32(*r1, rotr32_avx2_hr5(*r3));
    *r6 = _mm256_xor_si256(*r6, *r0); *r7 = _mm256_xor_si256(*r7, *r1);
    *r6 = rotr32_avx2_hr1(*r6); *r7 = rotr32_avx2_hr1(*r7);
    *r4 = _mm256_add_epi32(*r4, rotr32_avx2_hr6(*r6));
    *r5 = _mm256_add_epi32(*r5, rotr32_avx2_hr6(*r7));
    *r2 = _mm256_xor_si256(*r2, *r4); *r3 = _mm256_xor_si256(*r3, *r5);
    *r2 = rotr32_avx2_hr2(*r2); *r3 = rotr32_avx2_hr2(*r3);
    *r0 = _mm256_xor_si256(*r0, rotr32_avx2_hr9(*r4));
    *r1 = _mm256_xor_si256(*r1, rotr32_avx2_hr9(*r5));
    *r0 = _mm256_add_epi32(*r0, rotr32_avx2_hr7(*r2));
    *r1 = _mm256_add_epi32(*r1, rotr32_avx2_hr7(*r3));
    *r6 = _mm256_xor_si256(*r6, *r0); *r7 = _mm256_xor_si256(*r7, *r1);
    *r6 = rotr32_avx2_hr3(*r6); *r7 = rotr32_avx2_hr3(*r7);
    *r4 = _mm256_add_epi32(*r4, rotr32_avx2_hr8(*r6));
    *r5 = _mm256_add_epi32(*r5, rotr32_avx2_hr8(*r7));
    *r2 = _mm256_xor_si256(*r2, *r4); *r3 = _mm256_xor_si256(*r3, *r5);
    *r2 = rotr32_avx2_hr4(*r2); *r3 = rotr32_avx2_hr4(*r3);
}

// ===== AVX2 逆 G_Mix / H_Mix =====

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn inv_g_mix_avx2(
    r0: &mut __m256i, r1: &mut __m256i, r2: &mut __m256i, r3: &mut __m256i,
    r4: &mut __m256i, r5: &mut __m256i, r6: &mut __m256i, r7: &mut __m256i,
) {
    *r2 = _mm256_xor_si256(rotl32_avx2_r4(*r2), *r4);
    *r3 = _mm256_xor_si256(rotl32_avx2_r4(*r3), *r5);
    *r4 = _mm256_sub_epi32(*r4, rotr32_avx2_r8(*r6));
    *r5 = _mm256_sub_epi32(*r5, rotr32_avx2_r8(*r7));
    *r6 = _mm256_xor_si256(rotl32_avx2_r3(*r6), *r0);
    *r7 = _mm256_xor_si256(rotl32_avx2_r3(*r7), *r1);
    *r0 = _mm256_sub_epi32(*r0, rotr32_avx2_r7(*r2));
    *r1 = _mm256_sub_epi32(*r1, rotr32_avx2_r7(*r3));
    *r0 = _mm256_xor_si256(*r0, rotr32_avx2_r9(*r4));
    *r1 = _mm256_xor_si256(*r1, rotr32_avx2_r9(*r5));
    *r2 = _mm256_xor_si256(rotl32_avx2_r2(*r2), *r4);
    *r3 = _mm256_xor_si256(rotl32_avx2_r2(*r3), *r5);
    *r4 = _mm256_sub_epi32(*r4, rotr32_avx2_r6(*r6));
    *r5 = _mm256_sub_epi32(*r5, rotr32_avx2_r6(*r7));
    *r6 = _mm256_xor_si256(rotl32_avx2_r1(*r6), *r0);
    *r7 = _mm256_xor_si256(rotl32_avx2_r1(*r7), *r1);
    *r0 = _mm256_sub_epi32(*r0, rotr32_avx2_r5(*r2));
    *r1 = _mm256_sub_epi32(*r1, rotr32_avx2_r5(*r3));
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn inv_h_mix_avx2(
    r0: &mut __m256i, r1: &mut __m256i, r2: &mut __m256i, r3: &mut __m256i,
    r4: &mut __m256i, r5: &mut __m256i, r6: &mut __m256i, r7: &mut __m256i,
) {
    *r2 = _mm256_xor_si256(rotl32_avx2_hr4(*r2), *r4);
    *r3 = _mm256_xor_si256(rotl32_avx2_hr4(*r3), *r5);
    *r4 = _mm256_sub_epi32(*r4, rotr32_avx2_hr8(*r6));
    *r5 = _mm256_sub_epi32(*r5, rotr32_avx2_hr8(*r7));
    *r6 = _mm256_xor_si256(rotl32_avx2_hr3(*r6), *r0);
    *r7 = _mm256_xor_si256(rotl32_avx2_hr3(*r7), *r1);
    *r0 = _mm256_sub_epi32(*r0, rotr32_avx2_hr7(*r2));
    *r1 = _mm256_sub_epi32(*r1, rotr32_avx2_hr7(*r3));
    *r0 = _mm256_xor_si256(*r0, rotr32_avx2_hr9(*r4));
    *r1 = _mm256_xor_si256(*r1, rotr32_avx2_hr9(*r5));
    *r2 = _mm256_xor_si256(rotl32_avx2_hr2(*r2), *r4);
    *r3 = _mm256_xor_si256(rotl32_avx2_hr2(*r3), *r5);
    *r4 = _mm256_sub_epi32(*r4, rotr32_avx2_hr6(*r6));
    *r5 = _mm256_sub_epi32(*r5, rotr32_avx2_hr6(*r7));
    *r6 = _mm256_xor_si256(rotl32_avx2_hr1(*r6), *r0);
    *r7 = _mm256_xor_si256(rotl32_avx2_hr1(*r7), *r1);
    *r0 = _mm256_sub_epi32(*r0, rotr32_avx2_hr5(*r2));
    *r1 = _mm256_sub_epi32(*r1, rotr32_avx2_hr5(*r3));
}

// ===== AVX2 16 轮批量加密核心 =====

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn avx2_rounds_encrypt(
    r0: &mut __m256i, r1: &mut __m256i, r2: &mut __m256i, r3: &mut __m256i,
    r4: &mut __m256i, r5: &mut __m256i, r6: &mut __m256i, r7: &mut __m256i,
    keys: &[u32; MAX_ROUND_KEYS * 2], rounds: usize,
) {
    *r0 = _mm256_xor_si256(*r0, _mm256_set1_epi32(keys[0] as i32));
    *r1 = _mm256_xor_si256(*r1, _mm256_set1_epi32(keys[1] as i32));
    *r2 = _mm256_xor_si256(*r2, _mm256_set1_epi32(keys[2] as i32));
    *r3 = _mm256_xor_si256(*r3, _mm256_set1_epi32(keys[3] as i32));
    *r4 = _mm256_xor_si256(*r4, _mm256_set1_epi32(keys[4] as i32));
    *r5 = _mm256_xor_si256(*r5, _mm256_set1_epi32(keys[5] as i32));
    *r6 = _mm256_xor_si256(*r6, _mm256_set1_epi32(keys[6] as i32));
    *r7 = _mm256_xor_si256(*r7, _mm256_set1_epi32(keys[7] as i32));

    let mut ki = 4usize;
    for r in 0..rounds {
        if (r & 1) == 0 { g_mix_avx2(r0, r1, r2, r3, r4, r5, r6, r7); }
        else { h_mix_avx2(r0, r1, r2, r3, r4, r5, r6, r7); }
        // ShiftRows：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
        let t = *r0; *r0 = *r2; *r2 = *r7; *r7 = *r1;
        *r1 = *r5; *r5 = *r4; *r4 = *r3; *r3 = *r6; *r6 = t;
        *r0 = _mm256_xor_si256(*r0, _mm256_set1_epi32(keys[2 * ki] as i32));
        *r1 = _mm256_xor_si256(*r1, _mm256_set1_epi32(keys[2 * ki + 1] as i32));
        *r2 = _mm256_xor_si256(*r2, _mm256_set1_epi32(keys[2 * (ki + 1)] as i32));
        *r3 = _mm256_xor_si256(*r3, _mm256_set1_epi32(keys[2 * (ki + 1) + 1] as i32));
        *r4 = _mm256_xor_si256(*r4, _mm256_set1_epi32(keys[2 * (ki + 2)] as i32));
        *r5 = _mm256_xor_si256(*r5, _mm256_set1_epi32(keys[2 * (ki + 2) + 1] as i32));
        *r6 = _mm256_xor_si256(*r6, _mm256_set1_epi32(keys[2 * (ki + 3)] as i32));
        *r7 = _mm256_xor_si256(*r7, _mm256_set1_epi32(keys[2 * (ki + 3) + 1] as i32));
        ki += 4;
    }
    let ki_out = 4 + rounds * 4;
    *r0 = _mm256_xor_si256(*r0, _mm256_set1_epi32(keys[2 * ki_out] as i32));
    *r1 = _mm256_xor_si256(*r1, _mm256_set1_epi32(keys[2 * ki_out + 1] as i32));
    *r2 = _mm256_xor_si256(*r2, _mm256_set1_epi32(keys[2 * (ki_out + 1)] as i32));
    *r3 = _mm256_xor_si256(*r3, _mm256_set1_epi32(keys[2 * (ki_out + 1) + 1] as i32));
    *r4 = _mm256_xor_si256(*r4, _mm256_set1_epi32(keys[2 * (ki_out + 2)] as i32));
    *r5 = _mm256_xor_si256(*r5, _mm256_set1_epi32(keys[2 * (ki_out + 2) + 1] as i32));
    *r6 = _mm256_xor_si256(*r6, _mm256_set1_epi32(keys[2 * (ki_out + 3)] as i32));
    *r7 = _mm256_xor_si256(*r7, _mm256_set1_epi32(keys[2 * (ki_out + 3) + 1] as i32));
}

// ===== AVX2 16 轮批量解密核心 =====

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn avx2_rounds_decrypt(
    r0: &mut __m256i, r1: &mut __m256i, r2: &mut __m256i, r3: &mut __m256i,
    r4: &mut __m256i, r5: &mut __m256i, r6: &mut __m256i, r7: &mut __m256i,
    keys: &[u32; MAX_ROUND_KEYS * 2], rounds: usize,
) {
    let ki = 4 + rounds * 4;
    *r0 = _mm256_xor_si256(*r0, _mm256_set1_epi32(keys[2 * ki] as i32));
    *r1 = _mm256_xor_si256(*r1, _mm256_set1_epi32(keys[2 * ki + 1] as i32));
    *r2 = _mm256_xor_si256(*r2, _mm256_set1_epi32(keys[2 * (ki + 1)] as i32));
    *r3 = _mm256_xor_si256(*r3, _mm256_set1_epi32(keys[2 * (ki + 1) + 1] as i32));
    *r4 = _mm256_xor_si256(*r4, _mm256_set1_epi32(keys[2 * (ki + 2)] as i32));
    *r5 = _mm256_xor_si256(*r5, _mm256_set1_epi32(keys[2 * (ki + 2) + 1] as i32));
    *r6 = _mm256_xor_si256(*r6, _mm256_set1_epi32(keys[2 * (ki + 3)] as i32));
    *r7 = _mm256_xor_si256(*r7, _mm256_set1_epi32(keys[2 * (ki + 3) + 1] as i32));

    for r in (0..rounds).rev() {
        let ki = 4 + r * 4;
        *r0 = _mm256_xor_si256(*r0, _mm256_set1_epi32(keys[2 * ki] as i32));
        *r1 = _mm256_xor_si256(*r1, _mm256_set1_epi32(keys[2 * ki + 1] as i32));
        *r2 = _mm256_xor_si256(*r2, _mm256_set1_epi32(keys[2 * (ki + 1)] as i32));
        *r3 = _mm256_xor_si256(*r3, _mm256_set1_epi32(keys[2 * (ki + 1) + 1] as i32));
        *r4 = _mm256_xor_si256(*r4, _mm256_set1_epi32(keys[2 * (ki + 2)] as i32));
        *r5 = _mm256_xor_si256(*r5, _mm256_set1_epi32(keys[2 * (ki + 2) + 1] as i32));
        *r6 = _mm256_xor_si256(*r6, _mm256_set1_epi32(keys[2 * (ki + 3)] as i32));
        *r7 = _mm256_xor_si256(*r7, _mm256_set1_epi32(keys[2 * (ki + 3) + 1] as i32));
        // 逆 ShiftRows：8-循环置换的逆
        let t = *r0; *r0 = *r6; *r6 = *r3; *r3 = *r4;
        *r4 = *r5; *r5 = *r1; *r1 = *r7; *r7 = *r2; *r2 = t;
        if (r & 1) == 0 { inv_g_mix_avx2(r0, r1, r2, r3, r4, r5, r6, r7); }
        else { inv_h_mix_avx2(r0, r1, r2, r3, r4, r5, r6, r7); }
    }
    *r0 = _mm256_xor_si256(*r0, _mm256_set1_epi32(keys[0] as i32));
    *r1 = _mm256_xor_si256(*r1, _mm256_set1_epi32(keys[1] as i32));
    *r2 = _mm256_xor_si256(*r2, _mm256_set1_epi32(keys[2] as i32));
    *r3 = _mm256_xor_si256(*r3, _mm256_set1_epi32(keys[3] as i32));
    *r4 = _mm256_xor_si256(*r4, _mm256_set1_epi32(keys[4] as i32));
    *r5 = _mm256_xor_si256(*r5, _mm256_set1_epi32(keys[5] as i32));
    *r6 = _mm256_xor_si256(*r6, _mm256_set1_epi32(keys[6] as i32));
    *r7 = _mm256_xor_si256(*r7, _mm256_set1_epi32(keys[7] as i32));
}

// ===== AVX2 CTR 批量加密（8块） =====

#[target_feature(enable = "avx2")]
unsafe fn encrypt_blocks_batch8_avx2(
    keys: &[u32; MAX_ROUND_KEYS * 2], rounds: usize,
    dst: &mut [u8], src: &[u8], counter: &[u8; 32],
) {
    let n = src.len() / BLOCK_SIZE;
    if n < 8 { return; }

    let n0 = u32::from_le_bytes([counter[0], counter[1], counter[2], counter[3]]);
    let n1 = u32::from_le_bytes([counter[4], counter[5], counter[6], counter[7]]);
    let n2 = u32::from_le_bytes([counter[8], counter[9], counter[10], counter[11]]);
    let n3 = u32::from_le_bytes([counter[12], counter[13], counter[14], counter[15]]);
    let n4 = u32::from_le_bytes([counter[16], counter[17], counter[18], counter[19]]);
    let n5 = u32::from_le_bytes([counter[20], counter[21], counter[22], counter[23]]);
    let base_ctr = u64::from_le_bytes([counter[24], counter[25], counter[26], counter[27],
                                       counter[28], counter[29], counter[30], counter[31]]);

    let v_n0 = _mm256_set1_epi32(n0 as i32);
    let v_n1 = _mm256_set1_epi32(n1 as i32);
    let v_n2 = _mm256_set1_epi32(n2 as i32);
    let v_n3 = _mm256_set1_epi32(n3 as i32);
    let v_n4 = _mm256_set1_epi32(n4 as i32);
    let v_n5 = _mm256_set1_epi32(n5 as i32);

    let mut i = 0usize;
    while i + 7 < n {
        let c0 = base_ctr + i as u64;
        let c1 = base_ctr + i as u64 + 1;
        let c2 = base_ctr + i as u64 + 2;
        let c3 = base_ctr + i as u64 + 3;
        let c4 = base_ctr + i as u64 + 4;
        let c5 = base_ctr + i as u64 + 5;
        let c6 = base_ctr + i as u64 + 6;
        let c7 = base_ctr + i as u64 + 7;

        let mut r0 = v_n0; let mut r1 = v_n1;
        let mut r2 = v_n2; let mut r3 = v_n3;
        let mut r4 = v_n4; let mut r5 = v_n5;
        let mut r6 = _mm256_set_epi32(
            c7 as i32, c6 as i32, c5 as i32, c4 as i32,
            c3 as i32, c2 as i32, c1 as i32, c0 as i32);
        let mut r7 = _mm256_set_epi32(
            (c7 >> 32) as i32, (c6 >> 32) as i32, (c5 >> 32) as i32, (c4 >> 32) as i32,
            (c3 >> 32) as i32, (c2 >> 32) as i32, (c1 >> 32) as i32, (c0 >> 32) as i32);

        avx2_rounds_encrypt(&mut r0, &mut r1, &mut r2, &mut r3,
                            &mut r4, &mut r5, &mut r6, &mut r7, keys, rounds);

        let t0 = _mm256_unpacklo_epi32(r0, r1);
        let t1 = _mm256_unpackhi_epi32(r0, r1);
        let t2 = _mm256_unpacklo_epi32(r2, r3);
        let t3 = _mm256_unpackhi_epi32(r2, r3);
        let u0 = _mm256_unpacklo_epi64(t0, t2);
        let u1 = _mm256_unpackhi_epi64(t0, t2);
        let u2 = _mm256_unpacklo_epi64(t1, t3);
        let u3 = _mm256_unpackhi_epi64(t1, t3);

        let t0 = _mm256_unpacklo_epi32(r4, r5);
        let t1 = _mm256_unpackhi_epi32(r4, r5);
        let t2 = _mm256_unpacklo_epi32(r6, r7);
        let t3 = _mm256_unpackhi_epi32(r6, r7);
        let v0 = _mm256_unpacklo_epi64(t0, t2);
        let v1 = _mm256_unpackhi_epi64(t0, t2);
        let v2 = _mm256_unpacklo_epi64(t1, t3);
        let v3 = _mm256_unpackhi_epi64(t1, t3);

        let ks0 = _mm256_permute2x128_si256(u0, v0, 0x20);
        let ks4 = _mm256_permute2x128_si256(u0, v0, 0x31);
        let ks1 = _mm256_permute2x128_si256(u1, v1, 0x20);
        let ks5 = _mm256_permute2x128_si256(u1, v1, 0x31);
        let ks2 = _mm256_permute2x128_si256(u2, v2, 0x20);
        let ks6 = _mm256_permute2x128_si256(u2, v2, 0x31);
        let ks3 = _mm256_permute2x128_si256(u3, v3, 0x20);
        let ks7 = _mm256_permute2x128_si256(u3, v3, 0x31);

        let s = src.as_ptr().add(i * BLOCK_SIZE) as *const __m256i;
        let d = dst.as_mut_ptr().add(i * BLOCK_SIZE) as *mut __m256i;

        let d0 = _mm256_loadu_si256(s.add(0));
        _mm256_storeu_si256(d.add(0), _mm256_xor_si256(d0, ks0));
        let d0 = _mm256_loadu_si256(s.add(1));
        _mm256_storeu_si256(d.add(1), _mm256_xor_si256(d0, ks1));
        let d0 = _mm256_loadu_si256(s.add(2));
        _mm256_storeu_si256(d.add(2), _mm256_xor_si256(d0, ks2));
        let d0 = _mm256_loadu_si256(s.add(3));
        _mm256_storeu_si256(d.add(3), _mm256_xor_si256(d0, ks3));
        let d0 = _mm256_loadu_si256(s.add(4));
        _mm256_storeu_si256(d.add(4), _mm256_xor_si256(d0, ks4));
        let d0 = _mm256_loadu_si256(s.add(5));
        _mm256_storeu_si256(d.add(5), _mm256_xor_si256(d0, ks5));
        let d0 = _mm256_loadu_si256(s.add(6));
        _mm256_storeu_si256(d.add(6), _mm256_xor_si256(d0, ks6));
        let d0 = _mm256_loadu_si256(s.add(7));
        _mm256_storeu_si256(d.add(7), _mm256_xor_si256(d0, ks7));

        i += 8;
    }
    _mm256_zeroupper();
}

// ===== SSE2 从输入加载的批量加密（4块，XTS 加密用） =====

unsafe fn encrypt_blocks_from_input_batch4_sse2(
    keys: &[u32; MAX_ROUND_KEYS * 2], rounds: usize,
    dst: &mut [u8], src: &[u8],
) {
    let n = src.len() / BLOCK_SIZE;
    if n < 4 { return; }

    let in128 = src.as_ptr() as *const __m128i;
    let out128 = dst.as_mut_ptr() as *mut __m128i;

    let mut i = 0usize;
    while i + 3 < n {
        let a0 = _mm_loadu_si128(in128.add(i * 2));
        let a1 = _mm_loadu_si128(in128.add((i + 1) * 2));
        let a2 = _mm_loadu_si128(in128.add((i + 2) * 2));
        let a3 = _mm_loadu_si128(in128.add((i + 3) * 2));
        let mut r0 = _mm_setzero_si128();
        let mut r1 = _mm_setzero_si128();
        let mut r2 = _mm_setzero_si128();
        let mut r3 = _mm_setzero_si128();
        transpose4x4_sse2(a0, a1, a2, a3, &mut r0, &mut r1, &mut r2, &mut r3);

        let b0 = _mm_loadu_si128(in128.add(i * 2 + 1));
        let b1 = _mm_loadu_si128(in128.add((i + 1) * 2 + 1));
        let b2 = _mm_loadu_si128(in128.add((i + 2) * 2 + 1));
        let b3 = _mm_loadu_si128(in128.add((i + 3) * 2 + 1));
        let mut r4 = _mm_setzero_si128();
        let mut r5 = _mm_setzero_si128();
        let mut r6 = _mm_setzero_si128();
        let mut r7 = _mm_setzero_si128();
        transpose4x4_sse2(b0, b1, b2, b3, &mut r4, &mut r5, &mut r6, &mut r7);

        sse2_rounds_encrypt(&mut r0, &mut r1, &mut r2, &mut r3,
                            &mut r4, &mut r5, &mut r6, &mut r7, keys, rounds);

        let mut o0_lo = _mm_setzero_si128();
        let mut o1_lo = _mm_setzero_si128();
        let mut o2_lo = _mm_setzero_si128();
        let mut o3_lo = _mm_setzero_si128();
        transpose4x4_sse2(r0, r1, r2, r3, &mut o0_lo, &mut o1_lo, &mut o2_lo, &mut o3_lo);
        let mut o0_hi = _mm_setzero_si128();
        let mut o1_hi = _mm_setzero_si128();
        let mut o2_hi = _mm_setzero_si128();
        let mut o3_hi = _mm_setzero_si128();
        transpose4x4_sse2(r4, r5, r6, r7, &mut o0_hi, &mut o1_hi, &mut o2_hi, &mut o3_hi);

        let o = out128.add(i * 2);
        _mm_storeu_si128(o.add(0), o0_lo);
        _mm_storeu_si128(o.add(1), o0_hi);
        _mm_storeu_si128(o.add(2), o1_lo);
        _mm_storeu_si128(o.add(3), o1_hi);
        _mm_storeu_si128(o.add(4), o2_lo);
        _mm_storeu_si128(o.add(5), o2_hi);
        _mm_storeu_si128(o.add(6), o3_lo);
        _mm_storeu_si128(o.add(7), o3_hi);

        i += 4;
    }
}

// ===== SSE2 通用批量解密（4块，从输入加载，CBC/XTS 解密用） =====

unsafe fn decrypt_blocks_batch4_sse2(
    keys: &[u32; MAX_ROUND_KEYS * 2], rounds: usize,
    dst: &mut [u8], src: &[u8],
) {
    let n = src.len() / BLOCK_SIZE;
    if n < 4 { return; }

    let in128 = src.as_ptr() as *const __m128i;
    let out128 = dst.as_mut_ptr() as *mut __m128i;

    let mut i = 0usize;
    while i + 3 < n {
        let a0 = _mm_loadu_si128(in128.add(i * 2));
        let a1 = _mm_loadu_si128(in128.add((i + 1) * 2));
        let a2 = _mm_loadu_si128(in128.add((i + 2) * 2));
        let a3 = _mm_loadu_si128(in128.add((i + 3) * 2));
        let mut r0 = _mm_setzero_si128();
        let mut r1 = _mm_setzero_si128();
        let mut r2 = _mm_setzero_si128();
        let mut r3 = _mm_setzero_si128();
        transpose4x4_sse2(a0, a1, a2, a3, &mut r0, &mut r1, &mut r2, &mut r3);

        let b0 = _mm_loadu_si128(in128.add(i * 2 + 1));
        let b1 = _mm_loadu_si128(in128.add((i + 1) * 2 + 1));
        let b2 = _mm_loadu_si128(in128.add((i + 2) * 2 + 1));
        let b3 = _mm_loadu_si128(in128.add((i + 3) * 2 + 1));
        let mut r4 = _mm_setzero_si128();
        let mut r5 = _mm_setzero_si128();
        let mut r6 = _mm_setzero_si128();
        let mut r7 = _mm_setzero_si128();
        transpose4x4_sse2(b0, b1, b2, b3, &mut r4, &mut r5, &mut r6, &mut r7);

        sse2_rounds_decrypt(&mut r0, &mut r1, &mut r2, &mut r3,
                            &mut r4, &mut r5, &mut r6, &mut r7, keys, rounds);

        let mut o0_lo = _mm_setzero_si128();
        let mut o1_lo = _mm_setzero_si128();
        let mut o2_lo = _mm_setzero_si128();
        let mut o3_lo = _mm_setzero_si128();
        transpose4x4_sse2(r0, r1, r2, r3, &mut o0_lo, &mut o1_lo, &mut o2_lo, &mut o3_lo);
        let mut o0_hi = _mm_setzero_si128();
        let mut o1_hi = _mm_setzero_si128();
        let mut o2_hi = _mm_setzero_si128();
        let mut o3_hi = _mm_setzero_si128();
        transpose4x4_sse2(r4, r5, r6, r7, &mut o0_hi, &mut o1_hi, &mut o2_hi, &mut o3_hi);

        let o = out128.add(i * 2);
        _mm_storeu_si128(o.add(0), o0_lo);
        _mm_storeu_si128(o.add(1), o0_hi);
        _mm_storeu_si128(o.add(2), o1_lo);
        _mm_storeu_si128(o.add(3), o1_hi);
        _mm_storeu_si128(o.add(4), o2_lo);
        _mm_storeu_si128(o.add(5), o2_hi);
        _mm_storeu_si128(o.add(6), o3_lo);
        _mm_storeu_si128(o.add(7), o3_hi);

        i += 4;
    }
}

// ===== AVX2 从输入加载的批量加密（8块，XTS 加密用） =====

#[target_feature(enable = "avx2")]
unsafe fn encrypt_blocks_from_input_batch8_avx2(
    keys: &[u32; MAX_ROUND_KEYS * 2], rounds: usize,
    dst: &mut [u8], src: &[u8],
) {
    let n = src.len() / BLOCK_SIZE;
    if n < 8 { return; }

    let in256 = src.as_ptr() as *const __m256i;
    let out256 = dst.as_mut_ptr() as *mut __m256i;

    let mut i = 0usize;
    while i + 7 < n {
        let v0 = _mm256_loadu_si256(in256.add(i));
        let v1 = _mm256_loadu_si256(in256.add(i + 1));
        let v2 = _mm256_loadu_si256(in256.add(i + 2));
        let v3 = _mm256_loadu_si256(in256.add(i + 3));
        let v4 = _mm256_loadu_si256(in256.add(i + 4));
        let v5 = _mm256_loadu_si256(in256.add(i + 5));
        let v6 = _mm256_loadu_si256(in256.add(i + 6));
        let v7 = _mm256_loadu_si256(in256.add(i + 7));

        let t0a = _mm256_unpacklo_epi32(v0, v1);
        let t1a = _mm256_unpackhi_epi32(v0, v1);
        let t2a = _mm256_unpacklo_epi32(v2, v3);
        let t3a = _mm256_unpackhi_epi32(v2, v3);
        let u0 = _mm256_unpacklo_epi64(t0a, t2a);
        let u1 = _mm256_unpackhi_epi64(t0a, t2a);
        let u2 = _mm256_unpacklo_epi64(t1a, t3a);
        let u3 = _mm256_unpackhi_epi64(t1a, t3a);

        let t0b = _mm256_unpacklo_epi32(v4, v5);
        let t1b = _mm256_unpackhi_epi32(v4, v5);
        let t2b = _mm256_unpacklo_epi32(v6, v7);
        let t3b = _mm256_unpackhi_epi32(v6, v7);
        let v0_ = _mm256_unpacklo_epi64(t0b, t2b);
        let v1_ = _mm256_unpackhi_epi64(t0b, t2b);
        let v2_ = _mm256_unpacklo_epi64(t1b, t3b);
        let v3_ = _mm256_unpackhi_epi64(t1b, t3b);

        let mut r0 = _mm256_permute2x128_si256(u0, v0_, 0x20);
        let mut r1 = _mm256_permute2x128_si256(u1, v1_, 0x20);
        let mut r2 = _mm256_permute2x128_si256(u2, v2_, 0x20);
        let mut r3 = _mm256_permute2x128_si256(u3, v3_, 0x20);
        let mut r4 = _mm256_permute2x128_si256(u0, v0_, 0x31);
        let mut r5 = _mm256_permute2x128_si256(u1, v1_, 0x31);
        let mut r6 = _mm256_permute2x128_si256(u2, v2_, 0x31);
        let mut r7 = _mm256_permute2x128_si256(u3, v3_, 0x31);

        avx2_rounds_encrypt(&mut r0, &mut r1, &mut r2, &mut r3,
                            &mut r4, &mut r5, &mut r6, &mut r7, keys, rounds);

        let t0 = _mm256_unpacklo_epi32(r0, r1);
        let t1 = _mm256_unpackhi_epi32(r0, r1);
        let t2 = _mm256_unpacklo_epi32(r2, r3);
        let t3 = _mm256_unpackhi_epi32(r2, r3);
        let u0 = _mm256_unpacklo_epi64(t0, t2);
        let u1 = _mm256_unpackhi_epi64(t0, t2);
        let u2 = _mm256_unpacklo_epi64(t1, t3);
        let u3 = _mm256_unpackhi_epi64(t1, t3);

        let t0 = _mm256_unpacklo_epi32(r4, r5);
        let t1 = _mm256_unpackhi_epi32(r4, r5);
        let t2 = _mm256_unpacklo_epi32(r6, r7);
        let t3 = _mm256_unpackhi_epi32(r6, r7);
        let v0_ = _mm256_unpacklo_epi64(t0, t2);
        let v1_ = _mm256_unpackhi_epi64(t0, t2);
        let v2_ = _mm256_unpacklo_epi64(t1, t3);
        let v3_ = _mm256_unpackhi_epi64(t1, t3);

        let ks0 = _mm256_permute2x128_si256(u0, v0_, 0x20);
        let ks1 = _mm256_permute2x128_si256(u1, v1_, 0x20);
        let ks2 = _mm256_permute2x128_si256(u2, v2_, 0x20);
        let ks3 = _mm256_permute2x128_si256(u3, v3_, 0x20);
        let ks4 = _mm256_permute2x128_si256(u0, v0_, 0x31);
        let ks5 = _mm256_permute2x128_si256(u1, v1_, 0x31);
        let ks6 = _mm256_permute2x128_si256(u2, v2_, 0x31);
        let ks7 = _mm256_permute2x128_si256(u3, v3_, 0x31);

        let o = out256.add(i);
        _mm256_storeu_si256(o.add(0), ks0);
        _mm256_storeu_si256(o.add(1), ks1);
        _mm256_storeu_si256(o.add(2), ks2);
        _mm256_storeu_si256(o.add(3), ks3);
        _mm256_storeu_si256(o.add(4), ks4);
        _mm256_storeu_si256(o.add(5), ks5);
        _mm256_storeu_si256(o.add(6), ks6);
        _mm256_storeu_si256(o.add(7), ks7);

        i += 8;
    }
    _mm256_zeroupper();
}

// ===== AVX2 通用批量解密（8块，从输入加载，CBC/XTS 解密用） =====

#[target_feature(enable = "avx2")]
unsafe fn decrypt_blocks_batch8_avx2(
    keys: &[u32; MAX_ROUND_KEYS * 2], rounds: usize,
    dst: &mut [u8], src: &[u8],
) {
    let n = src.len() / BLOCK_SIZE;
    if n < 8 { return; }

    let in256 = src.as_ptr() as *const __m256i;
    let out256 = dst.as_mut_ptr() as *mut __m256i;

    let mut i = 0usize;
    while i + 7 < n {
        let v0 = _mm256_loadu_si256(in256.add(i));
        let v1 = _mm256_loadu_si256(in256.add(i + 1));
        let v2 = _mm256_loadu_si256(in256.add(i + 2));
        let v3 = _mm256_loadu_si256(in256.add(i + 3));
        let v4 = _mm256_loadu_si256(in256.add(i + 4));
        let v5 = _mm256_loadu_si256(in256.add(i + 5));
        let v6 = _mm256_loadu_si256(in256.add(i + 6));
        let v7 = _mm256_loadu_si256(in256.add(i + 7));

        let t0a = _mm256_unpacklo_epi32(v0, v1);
        let t1a = _mm256_unpackhi_epi32(v0, v1);
        let t2a = _mm256_unpacklo_epi32(v2, v3);
        let t3a = _mm256_unpackhi_epi32(v2, v3);
        let u0 = _mm256_unpacklo_epi64(t0a, t2a);
        let u1 = _mm256_unpackhi_epi64(t0a, t2a);
        let u2 = _mm256_unpacklo_epi64(t1a, t3a);
        let u3 = _mm256_unpackhi_epi64(t1a, t3a);

        let t0b = _mm256_unpacklo_epi32(v4, v5);
        let t1b = _mm256_unpackhi_epi32(v4, v5);
        let t2b = _mm256_unpacklo_epi32(v6, v7);
        let t3b = _mm256_unpackhi_epi32(v6, v7);
        let v0_ = _mm256_unpacklo_epi64(t0b, t2b);
        let v1_ = _mm256_unpackhi_epi64(t0b, t2b);
        let v2_ = _mm256_unpacklo_epi64(t1b, t3b);
        let v3_ = _mm256_unpackhi_epi64(t1b, t3b);

        let mut r0 = _mm256_permute2x128_si256(u0, v0_, 0x20);
        let mut r1 = _mm256_permute2x128_si256(u1, v1_, 0x20);
        let mut r2 = _mm256_permute2x128_si256(u2, v2_, 0x20);
        let mut r3 = _mm256_permute2x128_si256(u3, v3_, 0x20);
        let mut r4 = _mm256_permute2x128_si256(u0, v0_, 0x31);
        let mut r5 = _mm256_permute2x128_si256(u1, v1_, 0x31);
        let mut r6 = _mm256_permute2x128_si256(u2, v2_, 0x31);
        let mut r7 = _mm256_permute2x128_si256(u3, v3_, 0x31);

        avx2_rounds_decrypt(&mut r0, &mut r1, &mut r2, &mut r3,
                            &mut r4, &mut r5, &mut r6, &mut r7, keys, rounds);

        let t0 = _mm256_unpacklo_epi32(r0, r1);
        let t1 = _mm256_unpackhi_epi32(r0, r1);
        let t2 = _mm256_unpacklo_epi32(r2, r3);
        let t3 = _mm256_unpackhi_epi32(r2, r3);
        let u0 = _mm256_unpacklo_epi64(t0, t2);
        let u1 = _mm256_unpackhi_epi64(t0, t2);
        let u2 = _mm256_unpacklo_epi64(t1, t3);
        let u3 = _mm256_unpackhi_epi64(t1, t3);

        let t0 = _mm256_unpacklo_epi32(r4, r5);
        let t1 = _mm256_unpackhi_epi32(r4, r5);
        let t2 = _mm256_unpacklo_epi32(r6, r7);
        let t3 = _mm256_unpackhi_epi32(r6, r7);
        let v0_ = _mm256_unpacklo_epi64(t0, t2);
        let v1_ = _mm256_unpackhi_epi64(t0, t2);
        let v2_ = _mm256_unpacklo_epi64(t1, t3);
        let v3_ = _mm256_unpackhi_epi64(t1, t3);

        let ks0 = _mm256_permute2x128_si256(u0, v0_, 0x20);
        let ks1 = _mm256_permute2x128_si256(u1, v1_, 0x20);
        let ks2 = _mm256_permute2x128_si256(u2, v2_, 0x20);
        let ks3 = _mm256_permute2x128_si256(u3, v3_, 0x20);
        let ks4 = _mm256_permute2x128_si256(u0, v0_, 0x31);
        let ks5 = _mm256_permute2x128_si256(u1, v1_, 0x31);
        let ks6 = _mm256_permute2x128_si256(u2, v2_, 0x31);
        let ks7 = _mm256_permute2x128_si256(u3, v3_, 0x31);

        let o = out256.add(i);
        _mm256_storeu_si256(o.add(0), ks0);
        _mm256_storeu_si256(o.add(1), ks1);
        _mm256_storeu_si256(o.add(2), ks2);
        _mm256_storeu_si256(o.add(3), ks3);
        _mm256_storeu_si256(o.add(4), ks4);
        _mm256_storeu_si256(o.add(5), ks5);
        _mm256_storeu_si256(o.add(6), ks6);
        _mm256_storeu_si256(o.add(7), ks7);

        i += 8;
    }
    _mm256_zeroupper();
}

// ===== SIMD 辅助函数：GF(2^256) 乘 2 寄存器版、XOR 32 字节块 =====

// GF(2^256) 乘 2 寄存器版（SSE2），tweak 保持在 lo/hi 寄存器中
// 逻辑跟标量 gf_mul2_256 一致，用 SIMD 无分支掩码方式避免分支预测惩罚
unsafe fn gf_mul2_reg(lo: &mut __m128i, hi: &mut __m128i) {
    let mask_fe = _mm_set1_epi16(-258i16);
    let mask_01 = _mm_set1_epi16(0x0101i16);
    let poly_mask = _mm_set_epi8(0x25, 0x04, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    let cross_mask = _mm_set_epi8(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    let carry = _mm_movemask_epi8(*lo) & 1;
    let cross = _mm_movemask_epi8(*hi) & 1;

    let shifts_lo = _mm_and_si128(_mm_slli_epi16(*lo, 1), mask_fe);
    let shifts_hi = _mm_and_si128(_mm_slli_epi16(*hi, 1), mask_fe);
    let carries_lo = _mm_and_si128(_mm_srli_epi16(*lo, 7), mask_01);
    let carries_hi = _mm_and_si128(_mm_srli_epi16(*hi, 7), mask_01);
    let carries_lo_shifted = _mm_srli_si128(carries_lo, 1);
    let carries_hi_shifted = _mm_srli_si128(carries_hi, 1);

    let cross_sel = _mm_set1_epi8((-cross) as i8);
    *lo = _mm_or_si128(shifts_lo, _mm_or_si128(carries_lo_shifted, _mm_and_si128(cross_sel, cross_mask)));
    *hi = _mm_or_si128(shifts_hi, carries_hi_shifted);
    let carry_sel = _mm_set1_epi8((-carry) as i8);
    *hi = _mm_xor_si128(*hi, _mm_and_si128(carry_sel, poly_mask));
}

// SIMD 加速的 32 字节块 XOR，用两条 128 位 XOR 替代逐字节循环
unsafe fn xor_block32_simd(blk: &mut [u8], tweak: &[u8; BLOCK_SIZE]) {
    let v0 = _mm_loadu_si128(blk.as_ptr() as *const __m128i);
    let t0 = _mm_loadu_si128(tweak.as_ptr() as *const __m128i);
    let v1 = _mm_loadu_si128(blk.as_ptr().add(16) as *const __m128i);
    let t1 = _mm_loadu_si128(tweak.as_ptr().add(16) as *const __m128i);
    _mm_storeu_si128(blk.as_mut_ptr() as *mut __m128i, _mm_xor_si128(v0, t0));
    _mm_storeu_si128(blk.as_mut_ptr().add(16) as *mut __m128i, _mm_xor_si128(v1, t1));
}

// ===== 调度器函数 =====

/// CTR 加密调度器：AVX2 → SSE2 → 标量 4 路 ILP
pub fn encrypt_blocks_dispatched(
    cipher: &BlockCipher, dst: &mut [u8], src: &[u8], counter: &[u8; BLOCK_SIZE],
) {
    let len = src.len();
    if len == 0 { return; }
    let level = get_simd_level();
    let keys = &cipher.round_keys;
    let rounds = cipher.rounds;

    unsafe {
        if level >= 2 && len >= 8 * BLOCK_SIZE {
            let n8 = (len / (8 * BLOCK_SIZE)) * 8;
            let simd_len = n8 * BLOCK_SIZE;
            encrypt_blocks_batch8_avx2(keys, rounds, &mut dst[..simd_len], &src[..simd_len], counter);
            if simd_len < len {
                let ctr = counter_from(simd_len, counter);
                crate::encrypt_blocks_batch4(cipher, &mut dst[simd_len..], &src[simd_len..], &ctr);
            }
        } else if level >= 1 && len >= 4 * BLOCK_SIZE {
            let n4 = (len / (4 * BLOCK_SIZE)) * 4;
            let simd_len = n4 * BLOCK_SIZE;
            encrypt_blocks_batch4_sse2(keys, rounds, &mut dst[..simd_len], &src[..simd_len], counter);
            if simd_len < len {
                let ctr = counter_from(simd_len, counter);
                crate::encrypt_blocks_batch4(cipher, &mut dst[simd_len..], &src[simd_len..], &ctr);
            }
        } else {
            crate::encrypt_blocks_batch4(cipher, dst, src, counter);
        }
    }
}

fn counter_from(offset: usize, base: &[u8; BLOCK_SIZE]) -> [u8; BLOCK_SIZE] {
    let mut ctr = *base;
    let base_ctr = u64::from_le_bytes([base[24], base[25], base[26], base[27],
                                        base[28], base[29], base[30], base[31]]);
    ctr[24..32].copy_from_slice(&(base_ctr + (offset / BLOCK_SIZE) as u64).to_le_bytes());
    ctr
}

/// 从输入加载的批量加密调度器：AVX2 → SSE2 → 标量
pub fn encrypt_blocks_from_input_dispatched(
    cipher: &BlockCipher, dst: &mut [u8], src: &[u8],
) {
    let len = src.len();
    if len == 0 { return; }
    let level = get_simd_level();
    let keys = &cipher.round_keys;
    let rounds = cipher.rounds;

    unsafe {
        if level >= 2 && len >= 8 * BLOCK_SIZE {
            let n8 = (len / (8 * BLOCK_SIZE)) * 8;
            let simd_len = n8 * BLOCK_SIZE;
            encrypt_blocks_from_input_batch8_avx2(keys, rounds, &mut dst[..simd_len], &src[..simd_len]);
            if simd_len < len {
                for i in (n8 * BLOCK_SIZE..len).step_by(BLOCK_SIZE) {
                    let enc = cipher.encrypt_block(&src[i..i + BLOCK_SIZE]);
                    dst[i..i + BLOCK_SIZE].copy_from_slice(&enc);
                }
            }
        } else if level >= 1 && len >= 4 * BLOCK_SIZE {
            let n4 = (len / (4 * BLOCK_SIZE)) * 4;
            let simd_len = n4 * BLOCK_SIZE;
            encrypt_blocks_from_input_batch4_sse2(keys, rounds, &mut dst[..simd_len], &src[..simd_len]);
            if simd_len < len {
                for i in (n4 * BLOCK_SIZE..len).step_by(BLOCK_SIZE) {
                    let enc = cipher.encrypt_block(&src[i..i + BLOCK_SIZE]);
                    dst[i..i + BLOCK_SIZE].copy_from_slice(&enc);
                }
            }
        } else {
            for i in (0..len).step_by(BLOCK_SIZE) {
                let enc = cipher.encrypt_block(&src[i..i + BLOCK_SIZE]);
                dst[i..i + BLOCK_SIZE].copy_from_slice(&enc);
            }
        }
    }
}

/// 批量解密调度器：AVX2 → SSE2 → 标量
pub fn decrypt_blocks_dispatched(
    cipher: &BlockCipher, dst: &mut [u8], src: &[u8],
) {
    let len = src.len();
    if len == 0 { return; }
    let level = get_simd_level();
    let keys = &cipher.round_keys;
    let rounds = cipher.rounds;

    unsafe {
        if level >= 2 && len >= 8 * BLOCK_SIZE {
            let n8 = (len / (8 * BLOCK_SIZE)) * 8;
            let simd_len = n8 * BLOCK_SIZE;
            decrypt_blocks_batch8_avx2(keys, rounds, &mut dst[..simd_len], &src[..simd_len]);
            if simd_len < len {
                for i in (n8 * BLOCK_SIZE..len).step_by(BLOCK_SIZE) {
                    let dec = cipher.decrypt_block(&src[i..i + BLOCK_SIZE]);
                    dst[i..i + BLOCK_SIZE].copy_from_slice(&dec);
                }
            }
        } else if level >= 1 && len >= 4 * BLOCK_SIZE {
            let n4 = (len / (4 * BLOCK_SIZE)) * 4;
            let simd_len = n4 * BLOCK_SIZE;
            decrypt_blocks_batch4_sse2(keys, rounds, &mut dst[..simd_len], &src[..simd_len]);
            if simd_len < len {
                for i in (n4 * BLOCK_SIZE..len).step_by(BLOCK_SIZE) {
                    let dec = cipher.decrypt_block(&src[i..i + BLOCK_SIZE]);
                    dst[i..i + BLOCK_SIZE].copy_from_slice(&dec);
                }
            }
        } else {
            for i in (0..len).step_by(BLOCK_SIZE) {
                let dec = cipher.decrypt_block(&src[i..i + BLOCK_SIZE]);
                dst[i..i + BLOCK_SIZE].copy_from_slice(&dec);
            }
        }
    }
}

/// XTS 段处理调度器（SIMD 加速）：AVX2 → SSE2 → 标量 4 路 ILP
pub fn xts_process_segment_dispatched(
    cipher: &BlockCipher, result: &mut [u8], blocks: usize,
    start_tweak: &[u8; BLOCK_SIZE], encrypt: bool,
) {
    if blocks == 0 { return; }
    let level = get_simd_level();

    unsafe {
        // 用寄存器版 GF 乘 2 预计算 tweak 链
        let mut t_lo = _mm_loadu_si128(start_tweak.as_ptr() as *const __m128i);
        let mut t_hi = _mm_loadu_si128(start_tweak.as_ptr().add(16) as *const __m128i);

        let mut b = 0usize;
        // 整批处理
        while b < blocks {
            let remain = blocks - b;
            let cur_batch;
            if level >= 2 && remain >= 64 { cur_batch = 64; }
            else if level >= 2 && remain >= 32 { cur_batch = 32; }
            else if level >= 2 && remain >= 8 { cur_batch = 8; }
            else if level >= 1 && remain >= 4 { cur_batch = 4; }
            else { cur_batch = 1; }

            // 预计算 cur_batch 个 tweak
            let mut tweaks = [[0u8; BLOCK_SIZE]; 64];
            let mut lt = t_lo;
            let mut ht = t_hi;
            for tweak in tweaks.iter_mut().take(cur_batch) {
                _mm_storeu_si128(tweak.as_mut_ptr() as *mut __m128i, lt);
                _mm_storeu_si128(tweak.as_mut_ptr().add(16) as *mut __m128i, ht);
                gf_mul2_reg(&mut lt, &mut ht);
            }
            // 推进到下一个 tweak
            t_lo = lt;
            t_hi = ht;

            let off = b * BLOCK_SIZE;

            // 原地 XOR tweak
            for j in 0..cur_batch {
                xor_block32_simd(&mut result[off + j * BLOCK_SIZE..off + (j + 1) * BLOCK_SIZE], &tweaks[j]);
            }

            // 批量加解密（in-place：拷贝一份 src，加密到 dst 覆盖原数据）
            let blk_len = cur_batch * BLOCK_SIZE;
            let src_copy = result[off..off + blk_len].to_vec();
            if encrypt {
                encrypt_blocks_from_input_dispatched(cipher, &mut result[off..off + blk_len], &src_copy);
            } else {
                decrypt_blocks_dispatched(cipher, &mut result[off..off + blk_len], &src_copy);
            }

            // 原地 XOR tweak
            for j in 0..cur_batch {
                xor_block32_simd(&mut result[off + j * BLOCK_SIZE..off + (j + 1) * BLOCK_SIZE], &tweaks[j]);
            }

            b += cur_batch;
        }
    }
}