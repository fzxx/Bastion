// amd64 SIMD 批量加密：SSE2 一次 4 块，寄存器布局 r_k = [N 块的同一个 s_k]
// 汇编函数只使用 caller-saved 寄存器（AX/CX/DX/SI/DI/R8-R11），避免保存/恢复开销
// 指令名采用 plan9 助记符：PSRLL/PSLLL=32 位移位，PUNPCKLLQ/HLQ=解包低/高 32 位

#include "textflag.h"

// rotr32(x, R) = (x >> R) | (x << (32-R))
#define ROTR32_SSE(x, R, t0, t1) \
	MOVOU x, t0; \
	PSRLL $R, t0; \
	MOVOU x, t1; \
	PSLLL $(32-R), t1; \
	POR t1, t0

// G_Mix：主扩散（旋转常数 GM_R1..GM_R9）
#define GMIX_SSE(r0,r1,r2,r3,r4,r5,r6,r7) \
	ROTR32_SSE(r2, 7, X8, X9); \
	PADDL X8, r0; \
	ROTR32_SSE(r3, 7, X8, X9); \
	PADDL X8, r1; \
	PXOR r0, r6; \
	PXOR r1, r7; \
	ROTR32_SSE(r6, 13, X8, X9); \
	MOVOU X8, r6; \
	ROTR32_SSE(r7, 13, X8, X9); \
	MOVOU X8, r7; \
	ROTR32_SSE(r6, 11, X8, X9); \
	PADDL X8, r4; \
	ROTR32_SSE(r7, 11, X8, X9); \
	PADDL X8, r5; \
	PXOR r4, r2; \
	PXOR r5, r3; \
	ROTR32_SSE(r2, 19, X8, X9); \
	MOVOU X8, r2; \
	ROTR32_SSE(r3, 19, X8, X9); \
	MOVOU X8, r3; \
	ROTR32_SSE(r4, 17, X8, X9); \
	PXOR X8, r0; \
	ROTR32_SSE(r5, 17, X8, X9); \
	PXOR X8, r1; \
	ROTR32_SSE(r2, 5, X8, X9); \
	PADDL X8, r0; \
	ROTR32_SSE(r3, 5, X8, X9); \
	PADDL X8, r1; \
	PXOR r0, r6; \
	PXOR r1, r7; \
	ROTR32_SSE(r6, 23, X8, X9); \
	MOVOU X8, r6; \
	ROTR32_SSE(r7, 23, X8, X9); \
	MOVOU X8, r7; \
	ROTR32_SSE(r6, 15, X8, X9); \
	PADDL X8, r4; \
	ROTR32_SSE(r7, 15, X8, X9); \
	PADDL X8, r5; \
	PXOR r4, r2; \
	PXOR r5, r3; \
	ROTR32_SSE(r2, 29, X8, X9); \
	MOVOU X8, r2; \
	ROTR32_SSE(r3, 29, X8, X9); \
	MOVOU X8, r3

// H_Mix：辅助扩散（旋转常数 HM_R1..HM_R9）
#define HMIX_SSE(r0,r1,r2,r3,r4,r5,r6,r7) \
	ROTR32_SSE(r2, 3, X8, X9); \
	PADDL X8, r0; \
	ROTR32_SSE(r3, 3, X8, X9); \
	PADDL X8, r1; \
	PXOR r0, r6; \
	PXOR r1, r7; \
	ROTR32_SSE(r6, 9, X8, X9); \
	MOVOU X8, r6; \
	ROTR32_SSE(r7, 9, X8, X9); \
	MOVOU X8, r7; \
	ROTR32_SSE(r6, 11, X8, X9); \
	PADDL X8, r4; \
	ROTR32_SSE(r7, 11, X8, X9); \
	PADDL X8, r5; \
	PXOR r4, r2; \
	PXOR r5, r3; \
	ROTR32_SSE(r2, 17, X8, X9); \
	MOVOU X8, r2; \
	ROTR32_SSE(r3, 17, X8, X9); \
	MOVOU X8, r3; \
	ROTR32_SSE(r4, 25, X8, X9); \
	PXOR X8, r0; \
	ROTR32_SSE(r5, 25, X8, X9); \
	PXOR X8, r1; \
	ROTR32_SSE(r2, 7, X8, X9); \
	PADDL X8, r0; \
	ROTR32_SSE(r3, 7, X8, X9); \
	PADDL X8, r1; \
	PXOR r0, r6; \
	PXOR r1, r7; \
	ROTR32_SSE(r6, 21, X8, X9); \
	MOVOU X8, r6; \
	ROTR32_SSE(r7, 21, X8, X9); \
	MOVOU X8, r7; \
	ROTR32_SSE(r6, 13, X8, X9); \
	PADDL X8, r4; \
	ROTR32_SSE(r7, 13, X8, X9); \
	PADDL X8, r5; \
	PXOR r4, r2; \
	PXOR r5, r3; \
	ROTR32_SSE(r2, 27, X8, X9); \
	MOVOU X8, r2; \
	ROTR32_SSE(r3, 27, X8, X9); \
	MOVOU X8, r3

// detectSIMD 返回 1=SSE2，2=AVX2（amd64 必然有 SSE2）
// 检测流程：CPUID(1) 的 OSXSAVE+AVX 位 → XGETBV 确认 XCR0 已启用 XMM/YMM 保存 → CPUID(7,0) 的 AVX2 位
// XCR0 检查不能省：OSXSAVE=1 只代表 CPU 支持，OS 没启用 YMM 状态保存时直接跑 YMM 指令会触发 #UD 崩溃
TEXT ·detectSIMD(SB), NOSPLIT, $8-8
	MOVQ BX, 0(SP)                  // CPUID 会破坏 EBX，先存栈
	MOVL $1, AX
	CPUID
	TESTL $0x08000000, CX           // ECX bit27 = OSXSAVE
	JZ    sse2_only
	TESTL $0x18000000, CX           // ECX bit28 = AVX
	JZ    sse2_only
	XORL CX, CX                     // XGETBV(0)：读 XCR0，结果在 EDX:EAX
	BYTE $0x0F
	BYTE $0x01
	BYTE $0xD0                      // XGETBV
	ANDL $0x6, AX                   // bit1 = XMM 状态启用，bit2 = YMM 状态启用
	CMPL AX, $0x6
	JNE   sse2_only
	MOVL $7, AX
	XORL CX, CX
	CPUID
	TESTL $0x20, BX                 // EBX bit5 = AVX2
	JZ    sse2_only
	MOVQ 0(SP), BX
	MOVQ $2, AX
	MOVQ AX, ret+0(FP)
	RET
sse2_only:
	MOVQ 0(SP), BX
	MOVQ $1, AX
	MOVQ AX, ret+0(FP)
	RET

// asmEncryptBlocks4CTR SSE2 一次加密 blocks 块（4 的整数倍）
// 参数：keys(0) rounds(8) counter(16) in(24) out(32) blocks(40)，共 48 字节参数区
// 栈：0(SP)=rounds，8(SP)=4+4*rounds（AddRoundKey 结束偏移），16..64(SP)=nonce 广播后的 X0-X5
TEXT ·asmEncryptBlocks4CTR(SB), NOSPLIT, $112-48
	MOVQ keys+0(FP), AX         // AX = roundKeys 基址
	MOVQ rounds+8(FP), R11      // R11 = rounds
	MOVQ counter+16(FP), CX     // CX = counter
	MOVQ in+24(FP), DX          // DX = 输入
	MOVQ out+32(FP), SI         // SI = 输出
	MOVQ blocks+40(FP), DI      // DI = 块数

	// 保存 rounds，并计算 AddRoundKey 结束偏移 ki_end = 4+4*rounds
	MOVQ R11, 0(SP)
	SHLQ $2, R11
	ADDQ $4, R11
	MOVQ R11, 8(SP)

	// baseCtr = counter 尾部 8 字节小端计数器
	MOVQ 24(CX), R8

	// nonce 前 6 个字广播到 X0-X5（每批会重新恢复，因为轮循环会修改这些寄存器）
	MOVL 0(CX), R10
	MOVL R10, X0
	PSHUFD $0, X0, X0
	MOVL 4(CX), R10
	MOVL R10, X1
	PSHUFD $0, X1, X1
	MOVL 8(CX), R10
	MOVL R10, X2
	PSHUFD $0, X2, X2
	MOVL 12(CX), R10
	MOVL R10, X3
	PSHUFD $0, X3, X3
	MOVL 16(CX), R10
	MOVL R10, X4
	PSHUFD $0, X4, X4
	MOVL 20(CX), R10
	MOVL R10, X5
	PSHUFD $0, X5, X5
	// 广播后的 nonce 状态存栈，每批开始时恢复
	MOVOU X0, 16(SP)
	MOVOU X1, 32(SP)
	MOVOU X2, 48(SP)
	MOVOU X3, 64(SP)
	MOVOU X4, 80(SP)
	MOVOU X5, 96(SP)

	XORQ R9, R9                 // R9 = 已处理块数 i

loop:
	CMPQ R9, DI
	JGE  done

	// 恢复本批 nonce 广播状态
	MOVOU 16(SP), X0
	MOVOU 32(SP), X1
	MOVOU 48(SP), X2
	MOVOU 64(SP), X3
	MOVOU 80(SP), X4
	MOVOU 96(SP), X5

	// 组装 X6=[c0lo,c1lo,c2lo,c3lo]，X7=[c0hi,c1hi,c2hi,c3hi]
	MOVQ R8, R10
	ADDQ R9, R10                // c0 = baseCtr + i
	// 低 32 位
	MOVL R10, X6
	LEAQ 1(R10), R11
	MOVL R11, X7
	PUNPCKLLQ X7, X6            // [c0lo, c1lo]
	LEAQ 2(R10), R11
	MOVL R11, X7
	LEAQ 3(R10), R11
	MOVL R11, X9
	PUNPCKLLQ X9, X7            // [c2lo, c3lo]
	PUNPCKLQDQ X7, X6           // X6 = [c0lo,c1lo,c2lo,c3lo]
	// 高 32 位
	MOVQ R10, R11
	SHRQ $32, R11
	MOVL R11, X7
	LEAQ 1(R10), R11
	SHRQ $32, R11
	MOVL R11, X9
	PUNPCKLLQ X9, X7            // [c0hi, c1hi]
	LEAQ 2(R10), R11
	SHRQ $32, R11
	MOVL R11, X9
	LEAQ 3(R10), R11
	SHRQ $32, R11
	MOVL R11, X10
	PUNPCKLLQ X10, X9           // [c2hi, c3hi]
	PUNPCKLQDQ X9, X7           // X7 = [c0hi,c1hi,c2hi,c3hi]

	// 输入白化：keys[0..3] 的 8 个 uint32 广播 XOR
	MOVL 0(AX), R10
	MOVL R10, X8
	PSHUFD $0, X8, X8
	PXOR X8, X0
	MOVL 4(AX), R10
	MOVL R10, X8
	PSHUFD $0, X8, X8
	PXOR X8, X1
	MOVL 8(AX), R10
	MOVL R10, X8
	PSHUFD $0, X8, X8
	PXOR X8, X2
	MOVL 12(AX), R10
	MOVL R10, X8
	PSHUFD $0, X8, X8
	PXOR X8, X3
	MOVL 16(AX), R10
	MOVL R10, X8
	PSHUFD $0, X8, X8
	PXOR X8, X4
	MOVL 20(AX), R10
	MOVL R10, X8
	PSHUFD $0, X8, X8
	PXOR X8, X5
	MOVL 24(AX), R10
	MOVL R10, X8
	PSHUFD $0, X8, X8
	PXOR X8, X6
	MOVL 28(AX), R10
	MOVL R10, X8
	PSHUFD $0, X8, X8
	PXOR X8, X7

	// 16 轮：r = (ki-4)/4，偶轮 G_Mix，奇轮 H_Mix，每轮 ShiftRows + AddRoundKey
	MOVQ $4, R10                // ki = 4
round_loop:
	MOVQ 8(SP), R11
	CMPQ R10, R11               // ki < ki_end?
	JGE  round_done
	MOVQ R10, R11
	SUBQ $4, R11
	SHRQ $2, R11                // R11 = r
	TESTQ $1, R11
	JZ    use_gmix
	HMIX_SSE(X0, X1, X2, X3, X4, X5, X6, X7)
	JMP   after_mix
use_gmix:
	GMIX_SSE(X0, X1, X2, X3, X4, X5, X6, X7)
after_mix:
	// ShiftRows：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
	MOVOU X0, X8
	MOVOU X2, X0
	MOVOU X7, X2
	MOVOU X1, X7
	MOVOU X5, X1
	MOVOU X4, X5
	MOVOU X3, X4
	MOVOU X6, X3
	MOVOU X8, X6
	// AddRoundKey：keys[ki..ki+3] 的 8 个 uint32 广播 XOR
	MOVL 0(AX)(R10*8), R11
	MOVL R11, X8
	PSHUFD $0, X8, X8
	PXOR X8, X0
	MOVL 4(AX)(R10*8), R11
	MOVL R11, X8
	PSHUFD $0, X8, X8
	PXOR X8, X1
	MOVL 8(AX)(R10*8), R11
	MOVL R11, X8
	PSHUFD $0, X8, X8
	PXOR X8, X2
	MOVL 12(AX)(R10*8), R11
	MOVL R11, X8
	PSHUFD $0, X8, X8
	PXOR X8, X3
	MOVL 16(AX)(R10*8), R11
	MOVL R11, X8
	PSHUFD $0, X8, X8
	PXOR X8, X4
	MOVL 20(AX)(R10*8), R11
	MOVL R11, X8
	PSHUFD $0, X8, X8
	PXOR X8, X5
	MOVL 24(AX)(R10*8), R11
	MOVL R11, X8
	PSHUFD $0, X8, X8
	PXOR X8, X6
	MOVL 28(AX)(R10*8), R11
	MOVL R11, X8
	PSHUFD $0, X8, X8
	PXOR X8, X7
	ADDQ $4, R10
	JMP  round_loop
round_done:
	// 输出白化：keys[ki..ki+3]（ki 此时 = ki_end）
	MOVL 0(AX)(R10*8), R11
	MOVL R11, X8
	PSHUFD $0, X8, X8
	PXOR X8, X0
	MOVL 4(AX)(R10*8), R11
	MOVL R11, X8
	PSHUFD $0, X8, X8
	PXOR X8, X1
	MOVL 8(AX)(R10*8), R11
	MOVL R11, X8
	PSHUFD $0, X8, X8
	PXOR X8, X2
	MOVL 12(AX)(R10*8), R11
	MOVL R11, X8
	PSHUFD $0, X8, X8
	PXOR X8, X3
	MOVL 16(AX)(R10*8), R11
	MOVL R11, X8
	PSHUFD $0, X8, X8
	PXOR X8, X4
	MOVL 20(AX)(R10*8), R11
	MOVL R11, X8
	PSHUFD $0, X8, X8
	PXOR X8, X5
	MOVL 24(AX)(R10*8), R11
	MOVL R11, X8
	PSHUFD $0, X8, X8
	PXOR X8, X6
	MOVL 28(AX)(R10*8), R11
	MOVL R11, X8
	PSHUFD $0, X8, X8
	PXOR X8, X7

	// 4x4 转置：r0-r3 → 块0..3 前 16B（X0-X3），r4-r7 → 块0..3 后 16B（X4-X7）
	MOVOU X0, X8
	PUNPCKLLQ X1, X8
	MOVOU X0, X9
	PUNPCKHLQ X1, X9
	MOVOU X2, X10
	PUNPCKLLQ X3, X10
	MOVOU X2, X11
	PUNPCKHLQ X3, X11
	MOVOU X8, X0
	PUNPCKLQDQ X10, X0
	MOVOU X8, X1
	PUNPCKHQDQ X10, X1
	MOVOU X9, X2
	PUNPCKLQDQ X11, X2
	MOVOU X9, X3
	PUNPCKHQDQ X11, X3
	MOVOU X4, X8
	PUNPCKLLQ X5, X8
	MOVOU X4, X9
	PUNPCKHLQ X5, X9
	MOVOU X6, X10
	PUNPCKLLQ X7, X10
	MOVOU X6, X11
	PUNPCKHLQ X7, X11
	MOVOU X8, X4
	PUNPCKLQDQ X10, X4
	MOVOU X8, X5
	PUNPCKHQDQ X10, X5
	MOVOU X9, X6
	PUNPCKLQDQ X11, X6
	MOVOU X9, X7
	PUNPCKHQDQ X11, X7

	// 与输入异或写出 4 块
	MOVOU 0(DX), X8
	PXOR X0, X8
	MOVOU X8, 0(SI)
	MOVOU 16(DX), X8
	PXOR X4, X8
	MOVOU X8, 16(SI)
	MOVOU 32(DX), X8
	PXOR X1, X8
	MOVOU X8, 32(SI)
	MOVOU 48(DX), X8
	PXOR X5, X8
	MOVOU X8, 48(SI)
	MOVOU 64(DX), X8
	PXOR X2, X8
	MOVOU X8, 64(SI)
	MOVOU 80(DX), X8
	PXOR X6, X8
	MOVOU X8, 80(SI)
	MOVOU 96(DX), X8
	PXOR X3, X8
	MOVOU X8, 96(SI)
	MOVOU 112(DX), X8
	PXOR X7, X8
	MOVOU X8, 112(SI)

	ADDQ $128, DX
	ADDQ $128, SI
	ADDQ $4, R9
	JMP  loop
done:
	RET

// ============ AVX2 8 块批量加密 ============
// 与 SSE2 版同构：8 个 YMM 状态字各存 8 块的同一个 s_k，一次 8 块
// 指令用 Intel 风格 VEX 助记符；GPR 仍只用 caller-saved（AX/CX/DX/SI/DI/R8-R11）

// rotr32_avx(x, R) = (x >> R) | (x << (32-R))
#define ROTR32_AVX(x, R, t0, t1) \
	VPSRLD $R, x, t0; \
	VPSLLD $(32-R), x, t1; \
	VPOR t1, t0, t0

// G_Mix（AVX2）
#define GMIX_AVX(r0,r1,r2,r3,r4,r5,r6,r7) \
	ROTR32_AVX(r2, 7, Y12, Y13); \
	VPADDD Y12, r0, r0; \
	ROTR32_AVX(r3, 7, Y12, Y13); \
	VPADDD Y12, r1, r1; \
	VPXOR r0, r6, r6; \
	VPXOR r1, r7, r7; \
	ROTR32_AVX(r6, 13, Y12, Y13); \
	VMOVDQA Y12, r6; \
	ROTR32_AVX(r7, 13, Y12, Y13); \
	VMOVDQA Y12, r7; \
	ROTR32_AVX(r6, 11, Y12, Y13); \
	VPADDD Y12, r4, r4; \
	ROTR32_AVX(r7, 11, Y12, Y13); \
	VPADDD Y12, r5, r5; \
	VPXOR r4, r2, r2; \
	VPXOR r5, r3, r3; \
	ROTR32_AVX(r2, 19, Y12, Y13); \
	VMOVDQA Y12, r2; \
	ROTR32_AVX(r3, 19, Y12, Y13); \
	VMOVDQA Y12, r3; \
	ROTR32_AVX(r4, 17, Y12, Y13); \
	VPXOR Y12, r0, r0; \
	ROTR32_AVX(r5, 17, Y12, Y13); \
	VPXOR Y12, r1, r1; \
	ROTR32_AVX(r2, 5, Y12, Y13); \
	VPADDD Y12, r0, r0; \
	ROTR32_AVX(r3, 5, Y12, Y13); \
	VPADDD Y12, r1, r1; \
	VPXOR r0, r6, r6; \
	VPXOR r1, r7, r7; \
	ROTR32_AVX(r6, 23, Y12, Y13); \
	VMOVDQA Y12, r6; \
	ROTR32_AVX(r7, 23, Y12, Y13); \
	VMOVDQA Y12, r7; \
	ROTR32_AVX(r6, 15, Y12, Y13); \
	VPADDD Y12, r4, r4; \
	ROTR32_AVX(r7, 15, Y12, Y13); \
	VPADDD Y12, r5, r5; \
	VPXOR r4, r2, r2; \
	VPXOR r5, r3, r3; \
	ROTR32_AVX(r2, 29, Y12, Y13); \
	VMOVDQA Y12, r2; \
	ROTR32_AVX(r3, 29, Y12, Y13); \
	VMOVDQA Y12, r3

// H_Mix（AVX2）
#define HMIX_AVX(r0,r1,r2,r3,r4,r5,r6,r7) \
	ROTR32_AVX(r2, 3, Y12, Y13); \
	VPADDD Y12, r0, r0; \
	ROTR32_AVX(r3, 3, Y12, Y13); \
	VPADDD Y12, r1, r1; \
	VPXOR r0, r6, r6; \
	VPXOR r1, r7, r7; \
	ROTR32_AVX(r6, 9, Y12, Y13); \
	VMOVDQA Y12, r6; \
	ROTR32_AVX(r7, 9, Y12, Y13); \
	VMOVDQA Y12, r7; \
	ROTR32_AVX(r6, 11, Y12, Y13); \
	VPADDD Y12, r4, r4; \
	ROTR32_AVX(r7, 11, Y12, Y13); \
	VPADDD Y12, r5, r5; \
	VPXOR r4, r2, r2; \
	VPXOR r5, r3, r3; \
	ROTR32_AVX(r2, 17, Y12, Y13); \
	VMOVDQA Y12, r2; \
	ROTR32_AVX(r3, 17, Y12, Y13); \
	VMOVDQA Y12, r3; \
	ROTR32_AVX(r4, 25, Y12, Y13); \
	VPXOR Y12, r0, r0; \
	ROTR32_AVX(r5, 25, Y12, Y13); \
	VPXOR Y12, r1, r1; \
	ROTR32_AVX(r2, 7, Y12, Y13); \
	VPADDD Y12, r0, r0; \
	ROTR32_AVX(r3, 7, Y12, Y13); \
	VPADDD Y12, r1, r1; \
	VPXOR r0, r6, r6; \
	VPXOR r1, r7, r7; \
	ROTR32_AVX(r6, 21, Y12, Y13); \
	VMOVDQA Y12, r6; \
	ROTR32_AVX(r7, 21, Y12, Y13); \
	VMOVDQA Y12, r7; \
	ROTR32_AVX(r6, 13, Y12, Y13); \
	VPADDD Y12, r4, r4; \
	ROTR32_AVX(r7, 13, Y12, Y13); \
	VPADDD Y12, r5, r5; \
	VPXOR r4, r2, r2; \
	VPXOR r5, r3, r3; \
	ROTR32_AVX(r2, 27, Y12, Y13); \
	VMOVDQA Y12, r2; \
	ROTR32_AVX(r3, 27, Y12, Y13); \
	VMOVDQA Y12, r3

// asmEncryptBlocks8CTR AVX2 一次加密 blocks 块（8 的整数倍）
// 参数：keys(0) rounds(8) counter(16) in(24) out(32) blocks(40)，共 48 字节参数区
// 栈：0(SP)=8 个计数器低 32 位，32(SP)=高 32 位，64(SP)=rounds，72(SP)=4+4*rounds
TEXT ·asmEncryptBlocks8CTR(SB), NOSPLIT, $96-48
	MOVQ keys+0(FP), AX
	MOVQ rounds+8(FP), R11
	MOVQ counter+16(FP), CX
	MOVQ in+24(FP), DX
	MOVQ out+32(FP), SI
	MOVQ blocks+40(FP), DI

	// 保存 rounds 并计算 ki_end = 4+4*rounds
	MOVQ R11, 64(SP)
	SHLQ $2, R11
	ADDQ $4, R11
	MOVQ R11, 72(SP)

	MOVQ 24(CX), R8            // baseCtr
	XORQ R9, R9                // i

loop8:
	CMPQ R9, DI
	JGE  done8

	// nonce 6 个字直接广播到 Y0-Y5（每批重新广播，轮循环会改这些寄存器）
	VPBROADCASTD 0(CX), Y0
	VPBROADCASTD 4(CX), Y1
	VPBROADCASTD 8(CX), Y2
	VPBROADCASTD 12(CX), Y3
	VPBROADCASTD 16(CX), Y4
	VPBROADCASTD 20(CX), Y5

	// 组装 8 个计数器：低 32 位写 0(SP)，高 32 位写 32(SP)
	MOVQ R8, R10
	ADDQ R9, R10               // c0 = baseCtr + i
	MOVL R10, 0(SP)
	LEAQ 1(R10), R11
	MOVL R11, 4(SP)
	LEAQ 2(R10), R11
	MOVL R11, 8(SP)
	LEAQ 3(R10), R11
	MOVL R11, 12(SP)
	LEAQ 4(R10), R11
	MOVL R11, 16(SP)
	LEAQ 5(R10), R11
	MOVL R11, 20(SP)
	LEAQ 6(R10), R11
	MOVL R11, 24(SP)
	LEAQ 7(R10), R11
	MOVL R11, 28(SP)
	MOVQ R10, R11
	SHRQ $32, R11
	MOVL R11, 32(SP)
	LEAQ 1(R10), R11
	SHRQ $32, R11
	MOVL R11, 36(SP)
	LEAQ 2(R10), R11
	SHRQ $32, R11
	MOVL R11, 40(SP)
	LEAQ 3(R10), R11
	SHRQ $32, R11
	MOVL R11, 44(SP)
	LEAQ 4(R10), R11
	SHRQ $32, R11
	MOVL R11, 48(SP)
	LEAQ 5(R10), R11
	SHRQ $32, R11
	MOVL R11, 52(SP)
	LEAQ 6(R10), R11
	SHRQ $32, R11
	MOVL R11, 56(SP)
	LEAQ 7(R10), R11
	SHRQ $32, R11
	MOVL R11, 60(SP)
	VMOVDQU 0(SP), Y6
	VMOVDQU 32(SP), Y7

	// 输入白化：keys[0..3] 8 个 uint32 广播 XOR
	VPBROADCASTD 0(AX), Y8
	VPXOR Y8, Y0, Y0
	VPBROADCASTD 4(AX), Y8
	VPXOR Y8, Y1, Y1
	VPBROADCASTD 8(AX), Y8
	VPXOR Y8, Y2, Y2
	VPBROADCASTD 12(AX), Y8
	VPXOR Y8, Y3, Y3
	VPBROADCASTD 16(AX), Y8
	VPXOR Y8, Y4, Y4
	VPBROADCASTD 20(AX), Y8
	VPXOR Y8, Y5, Y5
	VPBROADCASTD 24(AX), Y8
	VPXOR Y8, Y6, Y6
	VPBROADCASTD 28(AX), Y8
	VPXOR Y8, Y7, Y7

	// 16 轮
	MOVQ $4, R10               // ki = 4
round8_loop:
	MOVQ 72(SP), R11
	CMPQ R10, R11
	JGE  round8_done
	MOVQ R10, R11
	SUBQ $4, R11
	SHRQ $2, R11
	TESTQ $1, R11
	JZ    use_gmix8
	HMIX_AVX(Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7)
	JMP   after_mix8
use_gmix8:
	GMIX_AVX(Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7)
after_mix8:
	// ShiftRows：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
	VMOVDQA Y0, Y8
	VMOVDQA Y2, Y0
	VMOVDQA Y7, Y2
	VMOVDQA Y1, Y7
	VMOVDQA Y5, Y1
	VMOVDQA Y4, Y5
	VMOVDQA Y3, Y4
	VMOVDQA Y6, Y3
	VMOVDQA Y8, Y6
	// AddRoundKey：keys[ki..ki+3] 8 个 uint32 广播 XOR
	VPBROADCASTD 0(AX)(R10*8), Y8
	VPXOR Y8, Y0, Y0
	VPBROADCASTD 4(AX)(R10*8), Y8
	VPXOR Y8, Y1, Y1
	VPBROADCASTD 8(AX)(R10*8), Y8
	VPXOR Y8, Y2, Y2
	VPBROADCASTD 12(AX)(R10*8), Y8
	VPXOR Y8, Y3, Y3
	VPBROADCASTD 16(AX)(R10*8), Y8
	VPXOR Y8, Y4, Y4
	VPBROADCASTD 20(AX)(R10*8), Y8
	VPXOR Y8, Y5, Y5
	VPBROADCASTD 24(AX)(R10*8), Y8
	VPXOR Y8, Y6, Y6
	VPBROADCASTD 28(AX)(R10*8), Y8
	VPXOR Y8, Y7, Y7
	ADDQ $4, R10
	JMP  round8_loop
round8_done:
	// 输出白化
	VPBROADCASTD 0(AX)(R10*8), Y8
	VPXOR Y8, Y0, Y0
	VPBROADCASTD 4(AX)(R10*8), Y8
	VPXOR Y8, Y1, Y1
	VPBROADCASTD 8(AX)(R10*8), Y8
	VPXOR Y8, Y2, Y2
	VPBROADCASTD 12(AX)(R10*8), Y8
	VPXOR Y8, Y3, Y3
	VPBROADCASTD 16(AX)(R10*8), Y8
	VPXOR Y8, Y4, Y4
	VPBROADCASTD 20(AX)(R10*8), Y8
	VPXOR Y8, Y5, Y5
	VPBROADCASTD 24(AX)(R10*8), Y8
	VPXOR Y8, Y6, Y6
	VPBROADCASTD 28(AX)(R10*8), Y8
	VPXOR Y8, Y7, Y7

	// 8x8 转置：unpack 在 128 位通道内独立，最后 VPERM2I128 跨通道组合
	// t0..t3 = unpacklo/hi32(r0..r3)，u0..u3 = unpacklo/hi64
	VPUNPCKLDQ Y1, Y0, Y8
	VPUNPCKHDQ Y1, Y0, Y9
	VPUNPCKLDQ Y3, Y2, Y10
	VPUNPCKHDQ Y3, Y2, Y11
	VPUNPCKLQDQ Y10, Y8, Y0
	VPUNPCKHQDQ Y10, Y8, Y1
	VPUNPCKLQDQ Y11, Y9, Y2
	VPUNPCKHQDQ Y11, Y9, Y3
	VPUNPCKLDQ Y5, Y4, Y8
	VPUNPCKHDQ Y5, Y4, Y9
	VPUNPCKLDQ Y7, Y6, Y10
	VPUNPCKHDQ Y7, Y6, Y11
	VPUNPCKLQDQ Y10, Y8, Y4
	VPUNPCKHQDQ Y10, Y8, Y5
	VPUNPCKLQDQ Y11, Y9, Y6
	VPUNPCKHQDQ Y11, Y9, Y7
	// ks0..ks7：每块完整 32 字节
	// plan9 的 VPERM2I128 第一操作数充当 b：0x20 → [a.low,b.low]，0x31 → [a.high,b.high]
	// 这里 a=Yu(前16B)，b=Yv(后16B)，故写 Yv 在前
	VPERM2I128 $0x20, Y4, Y0, Y8
	VPERM2I128 $0x31, Y4, Y0, Y9
	VPERM2I128 $0x20, Y5, Y1, Y10
	VPERM2I128 $0x31, Y5, Y1, Y11
	VPERM2I128 $0x20, Y6, Y2, Y12
	VPERM2I128 $0x31, Y6, Y2, Y13
	VPERM2I128 $0x20, Y7, Y3, Y14
	VPERM2I128 $0x31, Y7, Y3, Y15

	// 与输入异或写出 8 块
	// 注意 Y8..Y15 顺序是 [ks0,ks4,ks1,ks5,ks2,ks6,ks3,ks7]
	VMOVDQU 0(DX), Y0
	VPXOR Y8, Y0, Y0
	VMOVDQU Y0, 0(SI)
	VMOVDQU 32(DX), Y1
	VPXOR Y10, Y1, Y1
	VMOVDQU Y1, 32(SI)
	VMOVDQU 64(DX), Y2
	VPXOR Y12, Y2, Y2
	VMOVDQU Y2, 64(SI)
	VMOVDQU 96(DX), Y3
	VPXOR Y14, Y3, Y3
	VMOVDQU Y3, 96(SI)
	VMOVDQU 128(DX), Y4
	VPXOR Y9, Y4, Y4
	VMOVDQU Y4, 128(SI)
	VMOVDQU 160(DX), Y5
	VPXOR Y11, Y5, Y5
	VMOVDQU Y5, 160(SI)
	VMOVDQU 192(DX), Y6
	VPXOR Y13, Y6, Y6
	VMOVDQU Y6, 192(SI)
	VMOVDQU 224(DX), Y7
	VPXOR Y15, Y7, Y7
	VMOVDQU Y7, 224(SI)

	ADDQ $256, DX
	ADDQ $256, SI
	ADDQ $8, R9
	JMP  loop8
done8:
	VZEROUPPER                      // 清 upper YMM，避免返回后执行 SSE 指令触发 AVX-SSE 切换惩罚
	RET

// ============ 逆 Mix 宏（解密用） ============

// rotl32(x, R) = (x << R) | (x >> (32-R))
#define ROTL32_SSE(x, R, t0, t1) \
	MOVOU x, t0; \
	PSLLL $R, t0; \
	MOVOU x, t1; \
	PSRLL $(32-R), t1; \
	POR t1, t0

// 逆 G_Mix（SSE2）：解密逆 ColumnMix，左旋 + 减法，顺序严格按标量逆运算
#define INV_GMIX_SSE(r0,r1,r2,r3,r4,r5,r6,r7) \
	ROTL32_SSE(r2, 29, X8, X9); \
	PXOR r4, X8; \
	MOVOU X8, r2; \
	ROTL32_SSE(r3, 29, X8, X9); \
	PXOR r5, X8; \
	MOVOU X8, r3; \
	ROTR32_SSE(r6, 15, X8, X9); \
	PSUBL X8, r4; \
	ROTR32_SSE(r7, 15, X8, X9); \
	PSUBL X8, r5; \
	ROTL32_SSE(r6, 23, X8, X9); \
	PXOR r0, X8; \
	MOVOU X8, r6; \
	ROTL32_SSE(r7, 23, X8, X9); \
	PXOR r1, X8; \
	MOVOU X8, r7; \
	ROTR32_SSE(r2, 5, X8, X9); \
	PSUBL X8, r0; \
	ROTR32_SSE(r3, 5, X8, X9); \
	PSUBL X8, r1; \
	ROTR32_SSE(r4, 17, X8, X9); \
	PXOR X8, r0; \
	ROTR32_SSE(r5, 17, X8, X9); \
	PXOR X8, r1; \
	ROTL32_SSE(r2, 19, X8, X9); \
	PXOR r4, X8; \
	MOVOU X8, r2; \
	ROTL32_SSE(r3, 19, X8, X9); \
	PXOR r5, X8; \
	MOVOU X8, r3; \
	ROTR32_SSE(r6, 11, X8, X9); \
	PSUBL X8, r4; \
	ROTR32_SSE(r7, 11, X8, X9); \
	PSUBL X8, r5; \
	ROTL32_SSE(r6, 13, X8, X9); \
	PXOR r0, X8; \
	MOVOU X8, r6; \
	ROTL32_SSE(r7, 13, X8, X9); \
	PXOR r1, X8; \
	MOVOU X8, r7; \
	ROTR32_SSE(r2, 7, X8, X9); \
	PSUBL X8, r0; \
	ROTR32_SSE(r3, 7, X8, X9); \
	PSUBL X8, r1

// 逆 H_Mix（SSE2），旋转常数换 HM_R1..R9
#define INV_HMIX_SSE(r0,r1,r2,r3,r4,r5,r6,r7) \
	ROTL32_SSE(r2, 27, X8, X9); \
	PXOR r4, X8; \
	MOVOU X8, r2; \
	ROTL32_SSE(r3, 27, X8, X9); \
	PXOR r5, X8; \
	MOVOU X8, r3; \
	ROTR32_SSE(r6, 13, X8, X9); \
	PSUBL X8, r4; \
	ROTR32_SSE(r7, 13, X8, X9); \
	PSUBL X8, r5; \
	ROTL32_SSE(r6, 21, X8, X9); \
	PXOR r0, X8; \
	MOVOU X8, r6; \
	ROTL32_SSE(r7, 21, X8, X9); \
	PXOR r1, X8; \
	MOVOU X8, r7; \
	ROTR32_SSE(r2, 7, X8, X9); \
	PSUBL X8, r0; \
	ROTR32_SSE(r3, 7, X8, X9); \
	PSUBL X8, r1; \
	ROTR32_SSE(r4, 25, X8, X9); \
	PXOR X8, r0; \
	ROTR32_SSE(r5, 25, X8, X9); \
	PXOR X8, r1; \
	ROTL32_SSE(r2, 17, X8, X9); \
	PXOR r4, X8; \
	MOVOU X8, r2; \
	ROTL32_SSE(r3, 17, X8, X9); \
	PXOR r5, X8; \
	MOVOU X8, r3; \
	ROTR32_SSE(r6, 11, X8, X9); \
	PSUBL X8, r4; \
	ROTR32_SSE(r7, 11, X8, X9); \
	PSUBL X8, r5; \
	ROTL32_SSE(r6, 9, X8, X9); \
	PXOR r0, X8; \
	MOVOU X8, r6; \
	ROTL32_SSE(r7, 9, X8, X9); \
	PXOR r1, X8; \
	MOVOU X8, r7; \
	ROTR32_SSE(r2, 3, X8, X9); \
	PSUBL X8, r0; \
	ROTR32_SSE(r3, 3, X8, X9); \
	PSUBL X8, r1

// 4x4 转置：块序 X0-X3 ↔ 状态字序，转置两次互为逆
#define TRANSPOSE4_SSE(lo0,lo1,lo2,lo3,hi0,hi1,hi2,hi3) \
	MOVOU lo0, X8; \
	PUNPCKLLQ lo1, X8; \
	MOVOU lo0, X9; \
	PUNPCKHLQ lo1, X9; \
	MOVOU lo2, X10; \
	PUNPCKLLQ lo3, X10; \
	MOVOU lo2, X11; \
	PUNPCKHLQ lo3, X11; \
	MOVOU X8, lo0; \
	PUNPCKLQDQ X10, lo0; \
	MOVOU X8, lo1; \
	PUNPCKHQDQ X10, lo1; \
	MOVOU X9, lo2; \
	PUNPCKLQDQ X11, lo2; \
	MOVOU X9, lo3; \
	PUNPCKHQDQ X11, lo3; \
	MOVOU hi0, X8; \
	PUNPCKLLQ hi1, X8; \
	MOVOU hi0, X9; \
	PUNPCKHLQ hi1, X9; \
	MOVOU hi2, X10; \
	PUNPCKLLQ hi3, X10; \
	MOVOU hi2, X11; \
	PUNPCKHLQ hi3, X11; \
	MOVOU X8, hi0; \
	PUNPCKLQDQ X10, hi0; \
	MOVOU X8, hi1; \
	PUNPCKHQDQ X10, hi1; \
	MOVOU X9, hi2; \
	PUNPCKLQDQ X11, hi2; \
	MOVOU X9, hi3; \
	PUNPCKHQDQ X11, hi3

// 密钥白化/AddRoundKey：AX=keys，R10=ki（uint64 索引）
#define ADDKEY8_SSE(r0,r1,r2,r3,r4,r5,r6,r7) \
	MOVL 0(AX)(R10*8), R11; \
	MOVL R11, X8; \
	PSHUFD $0, X8, X8; \
	PXOR X8, r0; \
	MOVL 4(AX)(R10*8), R11; \
	MOVL R11, X8; \
	PSHUFD $0, X8, X8; \
	PXOR X8, r1; \
	MOVL 8(AX)(R10*8), R11; \
	MOVL R11, X8; \
	PSHUFD $0, X8, X8; \
	PXOR X8, r2; \
	MOVL 12(AX)(R10*8), R11; \
	MOVL R11, X8; \
	PSHUFD $0, X8, X8; \
	PXOR X8, r3; \
	MOVL 16(AX)(R10*8), R11; \
	MOVL R11, X8; \
	PSHUFD $0, X8, X8; \
	PXOR X8, r4; \
	MOVL 20(AX)(R10*8), R11; \
	MOVL R11, X8; \
	PSHUFD $0, X8, X8; \
	PXOR X8, r5; \
	MOVL 24(AX)(R10*8), R11; \
	MOVL R11, X8; \
	PSHUFD $0, X8, X8; \
	PXOR X8, r6; \
	MOVL 28(AX)(R10*8), R11; \
	MOVL R11, X8; \
	PSHUFD $0, X8, X8; \
	PXOR X8, r7

// asmEncryptBlocks4FromInput SSE2 从输入加载 4 块批量加密（XTS 加密用）
// 参数：keys(0) rounds(8) in(16) out(24) blocks(32)，共 40 字节参数区
// 栈：0(SP)=rounds，8(SP)=4+4*rounds
TEXT ·asmEncryptBlocks4FromInput(SB), NOSPLIT, $16-40
	MOVQ keys+0(FP), AX
	MOVQ rounds+8(FP), R11
	MOVQ in+16(FP), DX
	MOVQ out+24(FP), SI
	MOVQ blocks+32(FP), DI

	MOVQ R11, 0(SP)
	SHLQ $2, R11
	ADDQ $4, R11
	MOVQ R11, 8(SP)

	XORQ R9, R9
loop_enc_in:
	CMPQ R9, DI
	JGE  done_enc_in
	// 加载 4 块：前 16B 到 X0-X3，后 16B 到 X4-X7
	MOVOU 0(DX), X0
	MOVOU 32(DX), X1
	MOVOU 64(DX), X2
	MOVOU 96(DX), X3
	MOVOU 16(DX), X4
	MOVOU 48(DX), X5
	MOVOU 80(DX), X6
	MOVOU 112(DX), X7
	TRANSPOSE4_SSE(X0, X1, X2, X3, X4, X5, X6, X7)

	// 输入白化
	MOVQ $0, R10
	ADDKEY8_SSE(X0, X1, X2, X3, X4, X5, X6, X7)

	// 16 轮正向
	MOVQ $4, R10
round_enc_in:
	MOVQ 8(SP), R11
	CMPQ R10, R11
	JGE  round_done_enc_in
	MOVQ R10, R11
	SUBQ $4, R11
	SHRQ $2, R11
	TESTQ $1, R11
	JZ    use_gmix_enc_in
	HMIX_SSE(X0, X1, X2, X3, X4, X5, X6, X7)
	JMP   after_mix_enc_in
use_gmix_enc_in:
	GMIX_SSE(X0, X1, X2, X3, X4, X5, X6, X7)
after_mix_enc_in:
	// ShiftRows：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
	MOVOU X0, X8
	MOVOU X2, X0
	MOVOU X7, X2
	MOVOU X1, X7
	MOVOU X5, X1
	MOVOU X4, X5
	MOVOU X3, X4
	MOVOU X6, X3
	MOVOU X8, X6
	ADDKEY8_SSE(X0, X1, X2, X3, X4, X5, X6, X7)
	ADDQ $4, R10
	JMP  round_enc_in
round_done_enc_in:
	// 输出白化（ki 已到 ki_end）
	ADDKEY8_SSE(X0, X1, X2, X3, X4, X5, X6, X7)

	// 转置回块序
	TRANSPOSE4_SSE(X0, X1, X2, X3, X4, X5, X6, X7)
	// 写出 4 块
	MOVOU X0, 0(SI)
	MOVOU X4, 16(SI)
	MOVOU X1, 32(SI)
	MOVOU X5, 48(SI)
	MOVOU X2, 64(SI)
	MOVOU X6, 80(SI)
	MOVOU X3, 96(SI)
	MOVOU X7, 112(SI)

	ADDQ $128, DX
	ADDQ $128, SI
	ADDQ $4, R9
	JMP  loop_enc_in
done_enc_in:
	RET

// asmDecryptBlocks4FromInput SSE2 从输入加载 4 块批量解密（CBC/XTS 解密用）
TEXT ·asmDecryptBlocks4FromInput(SB), NOSPLIT, $16-40
	MOVQ keys+0(FP), AX
	MOVQ rounds+8(FP), R11
	MOVQ in+16(FP), DX
	MOVQ out+24(FP), SI
	MOVQ blocks+32(FP), DI

	MOVQ R11, 0(SP)
	SHLQ $2, R11
	ADDQ $4, R11
	MOVQ R11, 8(SP)

	XORQ R9, R9
loop_dec_in:
	CMPQ R9, DI
	JGE  done_dec_in
	MOVOU 0(DX), X0
	MOVOU 32(DX), X1
	MOVOU 64(DX), X2
	MOVOU 96(DX), X3
	MOVOU 16(DX), X4
	MOVOU 48(DX), X5
	MOVOU 80(DX), X6
	MOVOU 112(DX), X7
	TRANSPOSE4_SSE(X0, X1, X2, X3, X4, X5, X6, X7)

	// 解除输出白化
	MOVQ 8(SP), R10
	ADDKEY8_SSE(X0, X1, X2, X3, X4, X5, X6, X7)

	// 倒序轮：r = rounds-1 .. 0，每轮 AddRoundKey → ShiftRows → 逆 Mix
	MOVQ 8(SP), R10
	SUBQ $4, R10
round_dec_in:
	// AddRoundKey（解密第一步，XOR 自逆）
	ADDKEY8_SSE(X0, X1, X2, X3, X4, X5, X6, X7)
	// ShiftRows（置换，逆）：8-循环置换的逆
	MOVOU X0, X8
	MOVOU X6, X0
	MOVOU X3, X6
	MOVOU X4, X3
	MOVOU X5, X4
	MOVOU X1, X5
	MOVOU X7, X1
	MOVOU X2, X7
	MOVOU X8, X2
	// 逆 Mix：r = ki/4 - 1，偶数 r 用逆 G，奇数用逆 H
	MOVQ R10, R11
	SUBQ $4, R11
	SHRQ $2, R11
	TESTQ $1, R11
	JZ    use_inv_g
	INV_HMIX_SSE(X0, X1, X2, X3, X4, X5, X6, X7)
	JMP   after_inv
use_inv_g:
	INV_GMIX_SSE(X0, X1, X2, X3, X4, X5, X6, X7)
after_inv:
	SUBQ $4, R10
	MOVQ $4, R11
	CMPQ R10, R11
	JGE  round_dec_in

	// 解除输入白化
	MOVQ $0, R10
	ADDKEY8_SSE(X0, X1, X2, X3, X4, X5, X6, X7)

	// 转置回块序并写出
	TRANSPOSE4_SSE(X0, X1, X2, X3, X4, X5, X6, X7)
	MOVOU X0, 0(SI)
	MOVOU X4, 16(SI)
	MOVOU X1, 32(SI)
	MOVOU X5, 48(SI)
	MOVOU X2, 64(SI)
	MOVOU X6, 80(SI)
	MOVOU X3, 96(SI)
	MOVOU X7, 112(SI)

	ADDQ $128, DX
	ADDQ $128, SI
	ADDQ $4, R9
	JMP  loop_dec_in
done_dec_in:
	RET
