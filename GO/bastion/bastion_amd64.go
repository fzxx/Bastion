//go:build amd64

// amd64 平台 SIMD 批量加密入口：SSE2 一次 4 块、AVX2 一次 8 块，寄存器布局 r_k = [N 块的同一个 s_k]
package bastion

import (
	"encoding/binary"
	"unsafe"
)

// simdLevel 进程启动时用 CPUID 检测一次：1=SSE2，2=AVX2（amd64 必然支持 SSE2）
var simdLevel = detectSIMD()

//go:noescape
func detectSIMD() int

// asmEncryptBlocks4CTR 用 SSE2 一次加密 blocks 块（blocks 必须是 4 的整数倍）
// counter 指向首块的 32 字节计数器块，in/out 数据区长度 = blocks*32，允许 in == out
//go:noescape
func asmEncryptBlocks4CTR(keys *[maxRoundKeys]uint64, rounds int, counter *[32]byte, in, out unsafe.Pointer, blocks int)

// asmEncryptBlocks8CTR 用 AVX2 一次加密 blocks 块（blocks 必须是 8 的整数倍），语义同 4 块版
//go:noescape
func asmEncryptBlocks8CTR(keys *[maxRoundKeys]uint64, rounds int, counter *[32]byte, in, out unsafe.Pointer, blocks int)

// asmEncryptBlocks4FromInput SSE2 从输入加载 4 块批量加密（XTS 加密用），允许 in == out
//go:noescape
func asmEncryptBlocks4FromInput(keys *[maxRoundKeys]uint64, rounds int, in, out unsafe.Pointer, blocks int)

// asmDecryptBlocks4FromInput SSE2 从输入加载 4 块批量解密（CBC/XTS 解密用），允许 in == out
//go:noescape
func asmDecryptBlocks4FromInput(keys *[maxRoundKeys]uint64, rounds int, in, out unsafe.Pointer, blocks int)

// encryptFromInputDispatch 从输入加载的批量加密（XTS 加密用），处理完（含尾部回退单块）返回 true，块数不足 4 返回 false
func encryptFromInputDispatch(c *BlockCipher, dst, src []byte) bool {
	n := len(src) / BlockSize
	if n == 0 {
		return false
	}
	if simdLevel >= 1 && n >= 4 {
		full := n &^ 3
		asmEncryptBlocks4FromInput(&c.roundKeys, c.rounds,
			unsafe.Pointer(&src[0]), unsafe.Pointer(&dst[0]), full)
		for i := full; i < n; i++ {
			off := i * BlockSize
			c.Encrypt(dst[off:off+BlockSize], src[off:off+BlockSize])
		}
		return true
	}
	return false
}

// decryptFromInputDispatch 从输入加载的批量解密（CBC/XTS 解密用），尾部不足 4 块回退单块
func decryptFromInputDispatch(c *BlockCipher, dst, src []byte) bool {
	n := len(src) / BlockSize
	if n == 0 {
		return false
	}
	if simdLevel >= 1 && n >= 4 {
		full := n &^ 3
		asmDecryptBlocks4FromInput(&c.roundKeys, c.rounds,
			unsafe.Pointer(&src[0]), unsafe.Pointer(&dst[0]), full)
		for i := full; i < n; i++ {
			off := i * BlockSize
			c.Decrypt(dst[off:off+BlockSize], src[off:off+BlockSize])
		}
		return true
	}
	return false
}

// encryptBlocksDispatch 优先走 SIMD 批量加密，处理完（含尾部回退）返回 true，无法用 SIMD 返回 false
func encryptBlocksDispatch(c *BlockCipher, dst, src, counter []byte) bool {
	n := len(src) / BlockSize
	if n == 0 {
		return false
	}
	if simdLevel >= 2 && n >= 8 {
		full := n &^ 7
		asmEncryptBlocks8CTR(&c.roundKeys, c.rounds,
			(*[32]byte)(unsafe.Pointer(&counter[0])),
			unsafe.Pointer(&src[0]), unsafe.Pointer(&dst[0]), full)
		if n > full {
			encryptBlocksBatch4(c, dst[full*BlockSize:], src[full*BlockSize:], nextCounter(counter, full))
		}
		return true
	}
	if simdLevel >= 1 && n >= 4 {
		full := n &^ 3
		asmEncryptBlocks4CTR(&c.roundKeys, c.rounds,
			(*[32]byte)(unsafe.Pointer(&counter[0])),
			unsafe.Pointer(&src[0]), unsafe.Pointer(&dst[0]), full)
		if n > full {
			encryptBlocksBatch4(c, dst[full*BlockSize:], src[full*BlockSize:], nextCounter(counter, full))
		}
		return true
	}
	return false
}

// nextCounter 从 counter 推进 full 块，返回新 counter（前 24B nonce 不变，计数器加 full）
func nextCounter(counter []byte, full int) []byte {
	ctr := make([]byte, BlockSize)
	copy(ctr[:24], counter[:24])
	binary.LittleEndian.PutUint64(ctr[24:], binary.LittleEndian.Uint64(counter[24:])+uint64(full))
	return ctr
}
