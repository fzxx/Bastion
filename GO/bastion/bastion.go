// Bastion 分组密码库

package bastion

import (
	"bytes"
	"crypto/subtle"
	"encoding/binary"
	"fmt"
	"io"
	"math/bits"
	"runtime"
	"sync"
	"unsafe"
)

// 常量

const BlockSize = 32 // 256 位分组

const NonceSize = 24 // 192 位 nonce，CTR/AEAD 用

const ctrCounterSize = 24 // counter 块中 nonce 占 24 字节，后 8 字节为计数器

const (
	gcmNonceSize = 24 // 192 位 nonce
	GCMTagSize   = 16 // 128 位认证标签，半轮 Bastion 置换 HMAC
)

const hmacBlockSize = 32 // 256 位块

const (
	hashBlockSize = 32 // 256 位块
	hashSize      = 32 // 256 位摘要
)

const (
	kdfHashLen = 32 // HMAC-Bastion 输出 32 字节
)

const defaultRounds = 16 // 默认轮数

// 32字节块拆成8个uint32
const blockWords = BlockSize / 4

// maxRoundKeys 够放下输入白化 + 16轮 + 输出白化
const maxRoundKeys = (defaultRounds + 2) * 4

// 并行加密相关常量：64KB 以上启用多核，每 worker 至少 1024 块，最多 8 个 worker
const (
	parallelThreshold  = 64 << 10 // 64KB 以上启用多核并行
	minBlocksPerWorker = 1024     // 每 worker 至少 1024 块（32KB），避免 goroutine 调度开销超过收益
	maxWorkers         = 8        // 最多 8 个 worker，防止过度并行
)

// maxEmptyReads 流式读取时容忍的连续空读次数：io.Reader 允许返回 (0, nil)，
// 但连续空读不代表 EOF，一直读下去就是死循环，超过次数按 io.ErrNoProgress 报错
const maxEmptyReads = 100

// GF(2^8) 不可约多项式: x^8 + x^5 + x^3 + x^2 + 1 (0x12D)
const gf8Poly byte = 0x2D

// 固定旋转常数，改一个安全性就变了
// G_Mix: 主扩散旋转
const (
	GM_R1 = 13
	GM_R2 = 19
	GM_R3 = 23
	GM_R4 = 29
)

// G_Mix: 加法旋转绑定
const (
	GM_R5 = 7  // 与GM_R1=13互补
	GM_R6 = 11
	GM_R7 = 5
	GM_R8 = 15
)

// H_Mix: 辅助扩散旋转
const (
	HM_R1 = 9
	HM_R2 = 17
	HM_R3 = 21
	HM_R4 = 27
)

// H_Mix: 加法旋转绑定
const (
	HM_R5 = 3  // 与HM_R1=9互补
	HM_R6 = 11
	HM_R7 = 7
	HM_R8 = 13
)

// G_Mix: 折回XOR旋转常数
const GM_R9 = 17

// H_Mix: 折回XOR旋转常数
const HM_R9 = 25

// 全局变量与初始化

// streamBufPool 4MB 缓冲池，流式加解密分块用
var streamBufPool = sync.Pool{
	New: func() interface{} {
		b := make([]byte, 4*1024*1024) // 4MB
		return &b
	},
}

// sboxDeltas 4 个 S 盒的 delta 常数（密钥扩展用前 4 个 S0-S3）
var sboxDeltas = [4]byte{0x00, 0x5A, 0xB4, 0x2D}

// 预移位 S 盒查找表，消除运行时移位，init 后 S 盒原始数据不再需要
var sbox0T0, sbox1T8, sbox2T16, sbox3T24 [256]uint32

// 圆周率 π 派生的固定轮常数，16 轮 × 8 个 uint32
// 取自 π 的十六进制小数部分连续截取，每 8 个字符一个 32 位字，全局固定公开
var roundConstants = [defaultRounds][8]uint32{
	{0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89},
	{0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C, 0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917},
	{0x9216D5D9, 0x8979FB1B, 0xD1310BA6, 0x98DFB5AC, 0x2FFD72DB, 0xD01ADFB7, 0xB8E1AFED, 0x6A267E96},
	{0xBA7C9045, 0xF12C7F99, 0x24A19947, 0xB3916CF7, 0x0801F2E2, 0x858EFC16, 0x636920D8, 0x71574E69},
	{0xA458FEA3, 0xF4933D7E, 0x0D95748F, 0x728EB658, 0x718BCD58, 0x82154AEE, 0x7B54A41D, 0xC25A59B5},
	{0x9C30D539, 0x2AF26013, 0xC5D1B023, 0x286085F0, 0xCA417918, 0xB8DB38EF, 0x8E79DCB0, 0x603A180E},
	{0x6C9E0E8B, 0xB01E8A3E, 0xD71577C1, 0xBD314B27, 0x78AF2FDA, 0x55605C60, 0xE65525F3, 0xAA55AB94},
	{0x57489862, 0x63E81440, 0x55CA396A, 0x2AAB10B6, 0xB4CC5C34, 0x1141E8CE, 0xA15486AF, 0x7C72E993},
	{0xB3EE1411, 0x636FBC2A, 0x2BA9C55D, 0x741831F6, 0xCE5C3E16, 0x9B87931E, 0xAFD6BA33, 0x6C24CF5C},
	{0x7A325381, 0x28958677, 0x3B8F4898, 0x6B4BB9AF, 0xC4BFE81B, 0x66282193, 0x61D809CC, 0xFB21A991},
	{0x487CAC60, 0x5DEC8032, 0xEF845D5D, 0xE98575B1, 0xDC262302, 0xEB651B88, 0x23893E81, 0xD396ACC5},
	{0x0F6D6FF3, 0x83F44239, 0x2E0B4482, 0xA4842004, 0x69C8F04A, 0x9E1F9B5E, 0x21C66842, 0xF6E96C9A},
	{0x670C9C61, 0xABD388F0, 0x6A51A0D2, 0xD8542F68, 0x960FA728, 0xAB5133A3, 0x6EEF0B6C, 0x137A3BE4},
	{0xBA3BF050, 0x7EFB2A98, 0xA1F1651D, 0x39AF0176, 0x66CA593E, 0x82430E88, 0x8CEE8619, 0x456F9FB4},
	{0x7D84A5C3, 0x3B8B5EBE, 0xE06F75D8, 0x85C12073, 0x401A449F, 0x56C16AA6, 0x4ED3AA62, 0x363F7706},
	{0x1BFEDF72, 0x429B023D, 0x37D0D724, 0xD00A1248, 0xDB0FEAD3, 0x49F1C09B, 0x075372C9, 0x80991B7B},
}

// 末块封口常数（π 派生），标记最后一块，阻断长度扩展
var finalConst = [8]uint32{
	0x8F02C1BA, 0xB7ED2F35, 0x61711891, 0xDE7A9041, 0x758EDD75, 0x47708A47, 0x07B0E7A5, 0x4B91FB25,
}

func init() {
	base := genSbox()

	// 生成预移位表 S0-S3，用临时 sboxes，init 完不再保留原始 S 盒
	var sboxes [4][256]byte
	for i := 0; i < 4; i++ {
		delta := sboxDeltas[i]
		for j := 0; j < 256; j++ {
			sboxes[i][j] = base[j] ^ delta
		}
	}

	for i := 0; i < 256; i++ {
		sbox0T0[i] = uint32(sboxes[0][i])
		sbox1T8[i] = uint32(sboxes[1][i]) << 8
		sbox2T16[i] = uint32(sboxes[2][i]) << 16
		sbox3T24[i] = uint32(sboxes[3][i]) << 24
	}

	// 本包依赖小端序，大端序上 unsafe.Pointer 转 *[8]uint32 会得到错误结果，启动时断言
	if binary.LittleEndian.Uint32([]byte{0x01, 0x00, 0x00, 0x00}) != 1 {
		panic("bastion: little-endian platform required")
	}
}

// rotr32 循环右移，正向 ARX 路径统一用右旋
func rotr32(x uint32, r int) uint32 { return bits.RotateLeft32(x, -r) }

// rotl32 循环左移，解密时用，是 rotr32 的逆运算
func rotl32(x uint32, r int) uint32 { return bits.RotateLeft32(x, r) }

// xmxFinalize MurmurHash3 的 finalizer，代数次数高且常数时间
func xmxFinalize(x uint32) uint32 {
	x ^= x >> 16
	x *= 0x45d9f3b
	x ^= x >> 16
	x *= 0x45d9f3b
	x ^= x >> 16
	return x
}

// S盒构造

// gf8Mul GF(2^8) 乘法
func gf8Mul(a, b byte) byte {
	var result byte = 0
	for i := 0; i < 8; i++ {
		if (b & 1) != 0 {
			result ^= a
		}
		carry := a & 0x80
		a <<= 1
		if carry != 0 {
			a ^= gf8Poly
		}
		b >>= 1
	}
	return result
}

// gf8Inv GF(2^8) 求逆元，Fermat 小定理，0→0
func gf8Inv(a byte) byte {
	if a == 0 {
		return 0
	}
	result := byte(1)
	base := a
	for exp := 254; exp > 0; exp >>= 1 {
		if exp&1 != 0 {
			result = gf8Mul(result, base)
		}
		base = gf8Mul(base, base)
	}
	return result
}

// affineBastion Bastion 仿射变换
func affineBastion(x byte) byte {
	b0 := int((x >> 0) & 1)
	b1 := int((x >> 1) & 1)
	b2 := int((x >> 2) & 1)
	b3 := int((x >> 3) & 1)
	b4 := int((x >> 4) & 1)
	b5 := int((x >> 5) & 1)
	b6 := int((x >> 6) & 1)
	b7 := int((x >> 7) & 1)

	y0 := b7 ^ b5 ^ b4 ^ b3 ^ b1 ^ 1
	y1 := b7 ^ b6 ^ b4 ^ b3 ^ b2
	y2 := b6 ^ b5 ^ b3 ^ b2 ^ b1 ^ 1
	y3 := b5 ^ b4 ^ b2 ^ b1 ^ b0 ^ 1
	y4 := b7 ^ b4 ^ b3 ^ b1 ^ b0
	y5 := b7 ^ b6 ^ b3 ^ b2 ^ b0 ^ 1
	y6 := b6 ^ b5 ^ b2 ^ b1 ^ b0
	y7 := b7 ^ b5 ^ b4 ^ b0 ^ 1

	return byte((y0 & 1) | ((y1 & 1) << 1) | ((y2 & 1) << 2) | ((y3 & 1) << 3) |
		((y4 & 1) << 4) | ((y5 & 1) << 5) | ((y6 & 1) << 6) | ((y7 & 1) << 7))
}

// genSbox 生成 Bastion 基础 S 盒：S(x) = affine(inv(x))
func genSbox() [256]byte {
	var result [256]byte
	for i := 0; i < 256; i++ {
		inv := gf8Inv(byte(i))
		result[i] = affineBastion(inv)
	}
	return result
}

// sboxWordApply32 对 uint32 做 S0-S3 S 盒替换，密钥扩展用
func sboxWordApply32(v uint32) uint32 {
	return sbox3T24[byte(v>>24)] | sbox2T16[byte(v>>16)] | sbox1T8[byte(v>>8)] | sbox0T0[byte(v)]
}

// 分组密码核心

// BlockCipher Bastion 分组密码实例
type BlockCipher struct {
	rounds    int
	roundKeys [(defaultRounds + 2) * 4]uint64 // (rounds+2)*4 个uint64：输入白化 + 16轮 + 输出白化
}

// NewCipher 创建 Bastion 分组密码实例，仅支持 32 字节密钥
func NewCipher(key []byte) (*BlockCipher, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("bastion: invalid key size %d (must be 32 bytes)", len(key))
	}
	c := &BlockCipher{
		rounds: defaultRounds,
	}
	c.expandKey(key)
	return c, nil
}

func (c *BlockCipher) BlockSize() int { return BlockSize }

// encryptBlockInline 在寄存器变量上跑完整加密，供 Encrypt 和 cbcEncryptBlock 复用
func encryptBlockInline(s0, s1, s2, s3, s4, s5, s6, s7 uint32, rounds int, keys *[maxRoundKeys]uint64) (uint32, uint32, uint32, uint32, uint32, uint32, uint32, uint32) {
	// 输入白化：解包 uint64 轮密钥为 uint32
	s0 ^= uint32(keys[0])
	s1 ^= uint32(keys[0] >> 32)
	s2 ^= uint32(keys[1])
	s3 ^= uint32(keys[1] >> 32)
	s4 ^= uint32(keys[2])
	s5 ^= uint32(keys[2] >> 32)
	s6 ^= uint32(keys[3])
	s7 ^= uint32(keys[3] >> 32)

	ki := 4
	for r := 0; r < rounds; r++ {
		// ColumnMix：奇偶轮交替 G/H Mix
			if r&1 == 0 {
					// G_Mix(0,2,4,6) + G_Mix(1,3,5,7)
					s0 += rotr32(s2, GM_R5)
					s1 += rotr32(s3, GM_R5)
					s6 ^= s0
					s7 ^= s1
					s6 = rotr32(s6, GM_R1)
					s7 = rotr32(s7, GM_R1)

					s4 += rotr32(s6, GM_R6)
					s5 += rotr32(s7, GM_R6)
					s2 ^= s4
					s3 ^= s5
					s2 = rotr32(s2, GM_R2)
					s3 = rotr32(s3, GM_R2)

					// 折回XOR
					s0 ^= rotr32(s4, GM_R9)
					s1 ^= rotr32(s5, GM_R9)

					s0 += rotr32(s2, GM_R7)
					s1 += rotr32(s3, GM_R7)
					s6 ^= s0
					s7 ^= s1
					s6 = rotr32(s6, GM_R3)
					s7 = rotr32(s7, GM_R3)

					s4 += rotr32(s6, GM_R8)
					s5 += rotr32(s7, GM_R8)
					s2 ^= s4
					s3 ^= s5
					s2 = rotr32(s2, GM_R4)
					s3 = rotr32(s3, GM_R4)
				} else {
					// H_Mix(0,2,4,6) + H_Mix(1,3,5,7)
					s0 += rotr32(s2, HM_R5)
					s1 += rotr32(s3, HM_R5)
					s6 ^= s0
					s7 ^= s1
					s6 = rotr32(s6, HM_R1)
					s7 = rotr32(s7, HM_R1)

					s4 += rotr32(s6, HM_R6)
					s5 += rotr32(s7, HM_R6)
					s2 ^= s4
					s3 ^= s5
					s2 = rotr32(s2, HM_R2)
					s3 = rotr32(s3, HM_R2)

					// 折回XOR
					s0 ^= rotr32(s4, HM_R9)
					s1 ^= rotr32(s5, HM_R9)

					s0 += rotr32(s2, HM_R7)
					s1 += rotr32(s3, HM_R7)
					s6 ^= s0
					s7 ^= s1
					s6 = rotr32(s6, HM_R3)
					s7 = rotr32(s7, HM_R3)

					s4 += rotr32(s6, HM_R8)
					s5 += rotr32(s7, HM_R8)
					s2 ^= s4
					s3 ^= s5
					s2 = rotr32(s2, HM_R4)
					s3 = rotr32(s3, HM_R4)
				}

		// ShiftRows：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
		s0, s1, s2, s3, s4, s5, s6, s7 = s2, s5, s7, s6, s3, s4, s0, s1

		// AddRoundKey
		s0 ^= uint32(keys[ki])
		s1 ^= uint32(keys[ki] >> 32)
		s2 ^= uint32(keys[ki+1])
		s3 ^= uint32(keys[ki+1] >> 32)
		s4 ^= uint32(keys[ki+2])
		s5 ^= uint32(keys[ki+2] >> 32)
		s6 ^= uint32(keys[ki+3])
		s7 ^= uint32(keys[ki+3] >> 32)
		ki += 4
	}

	// 输出白化
	s0 ^= uint32(keys[ki])
	s1 ^= uint32(keys[ki] >> 32)
	s2 ^= uint32(keys[ki+1])
	s3 ^= uint32(keys[ki+1] >> 32)
	s4 ^= uint32(keys[ki+2])
	s5 ^= uint32(keys[ki+2] >> 32)
	s6 ^= uint32(keys[ki+3])
	s7 ^= uint32(keys[ki+3] >> 32)

	return s0, s1, s2, s3, s4, s5, s6, s7
}

// Encrypt 单块加密，按小端读写，无对齐要求（调用方传非对齐子切片也安全）
// src/dst 不足 32 字节时后续按块读写会越过切片边界，入口先做长度校验
func (c *BlockCipher) Encrypt(dst, src []byte) {
	if len(src) < BlockSize || len(dst) < BlockSize {
		panic(fmt.Sprintf("bastion: src and dst must be at least %d bytes", BlockSize))
	}
	s0 := binary.LittleEndian.Uint32(src)
	s1 := binary.LittleEndian.Uint32(src[4:])
	s2 := binary.LittleEndian.Uint32(src[8:])
	s3 := binary.LittleEndian.Uint32(src[12:])
	s4 := binary.LittleEndian.Uint32(src[16:])
	s5 := binary.LittleEndian.Uint32(src[20:])
	s6 := binary.LittleEndian.Uint32(src[24:])
	s7 := binary.LittleEndian.Uint32(src[28:])

	s0, s1, s2, s3, s4, s5, s6, s7 = encryptBlockInline(
		s0, s1, s2, s3, s4, s5, s6, s7, c.rounds, &c.roundKeys)

	// 小端写回，无对齐要求
	binary.LittleEndian.PutUint32(dst, s0)
	binary.LittleEndian.PutUint32(dst[4:], s1)
	binary.LittleEndian.PutUint32(dst[8:], s2)
	binary.LittleEndian.PutUint32(dst[12:], s3)
	binary.LittleEndian.PutUint32(dst[16:], s4)
	binary.LittleEndian.PutUint32(dst[20:], s5)
	binary.LittleEndian.PutUint32(dst[24:], s6)
	binary.LittleEndian.PutUint32(dst[28:], s7)
}

// decryptBlockInline 在寄存器变量上跑完整解密，供 Decrypt 和 decryptBlocksBatch4 复用
func decryptBlockInline(s0, s1, s2, s3, s4, s5, s6, s7 uint32, rounds int, keys *[maxRoundKeys]uint64) (uint32, uint32, uint32, uint32, uint32, uint32, uint32, uint32) {
	// 输出白化解除
	ki := 4 + rounds*4
	s0 ^= uint32(keys[ki])
	s1 ^= uint32(keys[ki] >> 32)
	s2 ^= uint32(keys[ki+1])
	s3 ^= uint32(keys[ki+1] >> 32)
	s4 ^= uint32(keys[ki+2])
	s5 ^= uint32(keys[ki+2] >> 32)
	s6 ^= uint32(keys[ki+3])
	s7 ^= uint32(keys[ki+3] >> 32)

	for r := rounds - 1; r >= 0; r-- {
		ki = 4 + r*4
		// AddRoundKey解除
		s0 ^= uint32(keys[ki])
		s1 ^= uint32(keys[ki] >> 32)
		s2 ^= uint32(keys[ki+1])
		s3 ^= uint32(keys[ki+1] >> 32)
		s4 ^= uint32(keys[ki+2])
		s5 ^= uint32(keys[ki+2] >> 32)
		s6 ^= uint32(keys[ki+3])
		s7 ^= uint32(keys[ki+3] >> 32)

		// InvShiftRows：8-循环置换的逆
		s0, s1, s2, s3, s4, s5, s6, s7 = s6, s7, s0, s4, s5, s1, s3, s2

		// InverseColumnMix：奇偶轮交替InvG/InvH
			if r&1 == 0 {
					// InvG_Mix(0,2,4,6) + InvG_Mix(1,3,5,7)
					s2 = rotl32(s2, GM_R4) ^ s4
					s3 = rotl32(s3, GM_R4) ^ s5
					s4 -= rotr32(s6, GM_R8)
					s5 -= rotr32(s7, GM_R8)

					s6 = rotl32(s6, GM_R3) ^ s0
					s7 = rotl32(s7, GM_R3) ^ s1
					s0 -= rotr32(s2, GM_R7)
					s1 -= rotr32(s3, GM_R7)

					// 折回XOR逆运算（XOR自逆）
					s0 ^= rotr32(s4, GM_R9)
					s1 ^= rotr32(s5, GM_R9)

					s2 = rotl32(s2, GM_R2) ^ s4
					s3 = rotl32(s3, GM_R2) ^ s5
					s4 -= rotr32(s6, GM_R6)
					s5 -= rotr32(s7, GM_R6)

					s6 = rotl32(s6, GM_R1) ^ s0
					s7 = rotl32(s7, GM_R1) ^ s1
					s0 -= rotr32(s2, GM_R5)
					s1 -= rotr32(s3, GM_R5)
				} else {
					// InvH_Mix(0,2,4,6) + InvH_Mix(1,3,5,7)
					s2 = rotl32(s2, HM_R4) ^ s4
					s3 = rotl32(s3, HM_R4) ^ s5
					s4 -= rotr32(s6, HM_R8)
					s5 -= rotr32(s7, HM_R8)

					s6 = rotl32(s6, HM_R3) ^ s0
					s7 = rotl32(s7, HM_R3) ^ s1
					s0 -= rotr32(s2, HM_R7)
					s1 -= rotr32(s3, HM_R7)

					// 折回XOR逆运算（XOR自逆）
					s0 ^= rotr32(s4, HM_R9)
					s1 ^= rotr32(s5, HM_R9)

					s2 = rotl32(s2, HM_R2) ^ s4
					s3 = rotl32(s3, HM_R2) ^ s5
					s4 -= rotr32(s6, HM_R6)
					s5 -= rotr32(s7, HM_R6)

					s6 = rotl32(s6, HM_R1) ^ s0
					s7 = rotl32(s7, HM_R1) ^ s1
					s0 -= rotr32(s2, HM_R5)
					s1 -= rotr32(s3, HM_R5)
				}
	}

	// 输入白化解除
	s0 ^= uint32(keys[0])
	s1 ^= uint32(keys[0] >> 32)
	s2 ^= uint32(keys[1])
	s3 ^= uint32(keys[1] >> 32)
	s4 ^= uint32(keys[2])
	s5 ^= uint32(keys[2] >> 32)
	s6 ^= uint32(keys[3])
	s7 ^= uint32(keys[3] >> 32)

	return s0, s1, s2, s3, s4, s5, s6, s7
}

// Decrypt 单块解密，按小端读写，无对齐要求（调用方传非对齐子切片也安全）
// src/dst 不足 32 字节时后续按块读写会越过切片边界，入口先做长度校验
func (c *BlockCipher) Decrypt(dst, src []byte) {
	if len(src) < BlockSize || len(dst) < BlockSize {
		panic(fmt.Sprintf("bastion: src and dst must be at least %d bytes", BlockSize))
	}
	s0 := binary.LittleEndian.Uint32(src)
	s1 := binary.LittleEndian.Uint32(src[4:])
	s2 := binary.LittleEndian.Uint32(src[8:])
	s3 := binary.LittleEndian.Uint32(src[12:])
	s4 := binary.LittleEndian.Uint32(src[16:])
	s5 := binary.LittleEndian.Uint32(src[20:])
	s6 := binary.LittleEndian.Uint32(src[24:])
	s7 := binary.LittleEndian.Uint32(src[28:])

	s0, s1, s2, s3, s4, s5, s6, s7 = decryptBlockInline(
		s0, s1, s2, s3, s4, s5, s6, s7, c.rounds, &c.roundKeys)

	// 小端写回，无对齐要求
	binary.LittleEndian.PutUint32(dst, s0)
	binary.LittleEndian.PutUint32(dst[4:], s1)
	binary.LittleEndian.PutUint32(dst[8:], s2)
	binary.LittleEndian.PutUint32(dst[12:], s3)
	binary.LittleEndian.PutUint32(dst[16:], s4)
	binary.LittleEndian.PutUint32(dst[20:], s5)
	binary.LittleEndian.PutUint32(dst[24:], s6)
	binary.LittleEndian.PutUint32(dst[28:], s7)
}

// BlockEncrypt 单块加密
func BlockEncrypt(plaintext, key []byte) ([]byte, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("bastion: 密钥长度必须为 32 字节")
	}
	if len(plaintext) != BlockSize {
		return nil, fmt.Errorf("bastion: 明文长度必须为 %d 字节", BlockSize)
	}
	c, err := NewCipher(key)
	if err != nil {
		return nil, err
	}
	ciphertext := make([]byte, BlockSize)
	c.Encrypt(ciphertext, plaintext)
	return ciphertext, nil
}

// BlockDecrypt 单块解密
func BlockDecrypt(ciphertext, key []byte) ([]byte, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("bastion: 密钥长度必须为 32 字节")
	}
	if len(ciphertext) != BlockSize {
		return nil, fmt.Errorf("bastion: 密文长度必须为 %d 字节", BlockSize)
	}
	c, err := NewCipher(key)
	if err != nil {
		return nil, err
	}
	plaintext := make([]byte, BlockSize)
	c.Decrypt(plaintext, ciphertext)
	return plaintext, nil
}

func (c *BlockCipher) expandKey(key []byte) {
	totalWords := (c.rounds + 2) * 8 // 144个uint32字（16轮）
	// 直接在 roundKeys 上展开，把 uint64 数组 reinterpret 为 uint32 数组，依赖小端平台语义
	w := (*[maxRoundKeys * 2]uint32)(unsafe.Pointer(&c.roundKeys[0]))

	// 2次uint64替代8次uint32加载（LittleEndian等价）
	keyU64_0 := binary.LittleEndian.Uint64(key[0:8])
	keyU64_1 := binary.LittleEndian.Uint64(key[8:16])
	keyU64_2 := binary.LittleEndian.Uint64(key[16:24])
	keyU64_3 := binary.LittleEndian.Uint64(key[24:32])
	w[0] = uint32(keyU64_0)
	w[1] = uint32(keyU64_0 >> 32)
	w[2] = uint32(keyU64_1)
	w[3] = uint32(keyU64_1 >> 32)
	w[4] = uint32(keyU64_2)
	w[5] = uint32(keyU64_2 >> 32)
	w[6] = uint32(keyU64_3)
	w[7] = uint32(keyU64_3 >> 32)

	for i := 8; i < totalWords; i++ {
		temp := w[i-1]
		if i%8 == 0 {
			temp = rotl32(temp, 8)
			temp = sboxWordApply32(temp)
		} else if i%4 == 0 {
			temp = sboxWordApply32(temp)
			temp = rotl32(temp, 16)
			temp += rotl32(temp, 7)
		} else {
			temp = xmxFinalize(temp)
		}
		w[i] = w[i-8] ^ temp
	}

	// 轮常数预合并进轮密钥，运行时少一次异或
	for r := 0; r < c.rounds; r++ {
		base := (r + 1) * 8 // 跳过输入白化 w[0..7]
		sc := &roundConstants[r]
		for k := 0; k < 8; k++ {
			w[base+k] ^= sc[k]
		}
	}
}



// Zeroize 清零轮密钥，使用完立即调用防止内存残留
func (c *BlockCipher) Zeroize() {
	// 轮密钥按字节交给平台安全清零（Windows 走 ntdll，其他平台走循环兜底）
	zeroizeBytes(unsafe.Slice((*byte)(unsafe.Pointer(&c.roundKeys[0])), int(unsafe.Sizeof(c.roundKeys))))
	c.rounds = 0
}

// ARX压缩函数

// arxRounds ARX 压缩固定 16 轮，与分组密码轮数一致，AEAD/Hash 复用 G_Mix/H_Mix 组件，无 S 盒
const arxRounds = defaultRounds

// bastionARXRound 一轮 ARX 变换，偶数轮 G_Mix 奇数轮 H_Mix，与分组密码主循环一致
func bastionARXRound(s0, s1, s2, s3, s4, s5, s6, s7 uint32, round int) (uint32, uint32, uint32, uint32, uint32, uint32, uint32, uint32) {
	if round&1 == 0 {
			// G_Mix(0,2,4,6) + G_Mix(1,3,5,7)
			s0 += rotr32(s2, GM_R5)
			s1 += rotr32(s3, GM_R5)
			s6 ^= s0
			s7 ^= s1
			s6 = rotr32(s6, GM_R1)
			s7 = rotr32(s7, GM_R1)

			s4 += rotr32(s6, GM_R6)
			s5 += rotr32(s7, GM_R6)
			s2 ^= s4
			s3 ^= s5
			s2 = rotr32(s2, GM_R2)
			s3 = rotr32(s3, GM_R2)

			// 折回XOR：s4→s0短路反馈
			s0 ^= rotr32(s4, GM_R9)
			s1 ^= rotr32(s5, GM_R9)

			s0 += rotr32(s2, GM_R7)
			s1 += rotr32(s3, GM_R7)
			s6 ^= s0
			s7 ^= s1
			s6 = rotr32(s6, GM_R3)
			s7 = rotr32(s7, GM_R3)

			s4 += rotr32(s6, GM_R8)
			s5 += rotr32(s7, GM_R8)
			s2 ^= s4
			s3 ^= s5
			s2 = rotr32(s2, GM_R4)
			s3 = rotr32(s3, GM_R4)
		} else {
			// H_Mix(0,2,4,6) + H_Mix(1,3,5,7)
			s0 += rotr32(s2, HM_R5)
			s1 += rotr32(s3, HM_R5)
			s6 ^= s0
			s7 ^= s1
			s6 = rotr32(s6, HM_R1)
			s7 = rotr32(s7, HM_R1)

			s4 += rotr32(s6, HM_R6)
			s5 += rotr32(s7, HM_R6)
			s2 ^= s4
			s3 ^= s5
			s2 = rotr32(s2, HM_R2)
			s3 = rotr32(s3, HM_R2)

			// 折回XOR：s4→s0短路反馈
			s0 ^= rotr32(s4, HM_R9)
			s1 ^= rotr32(s5, HM_R9)

			s0 += rotr32(s2, HM_R7)
			s1 += rotr32(s3, HM_R7)
			s6 ^= s0
			s7 ^= s1
			s6 = rotr32(s6, HM_R3)
			s7 = rotr32(s7, HM_R3)

			s4 += rotr32(s6, HM_R8)
			s5 += rotr32(s7, HM_R8)
			s2 ^= s4
			s3 ^= s5
			s2 = rotr32(s2, HM_R4)
			s3 = rotr32(s3, HM_R4)
		}

	// ShiftRows：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
	s0, s1, s2, s3, s4, s5, s6, s7 = s2, s5, s7, s6, s3, s4, s0, s1

	return s0, s1, s2, s3, s4, s5, s6, s7
}

// bastionARXCompress 对 32 字节 block 做 16 轮 ARX 压缩，更新 state。
// 流程：消息模加注入 → 16 轮 ARX → Davies-Meyer 前馈
func bastionARXCompress(state *[8]uint32, block []byte, last bool) {
	// block 可能是用户数据的非对齐子切片，逐字小端加载无对齐要求
	var m [blockWords]uint32
	loadBlockWordsLE(&m, block)

	// 保存旧状态用于前馈
	o0, o1, o2, o3, o4, o5, o6, o7 := state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]

	// 末块封口常数：只加在吸收阶段，不参与末尾前馈
	seal := [8]uint32{}
	if last {
		seal = finalConst
	}

	// 模加注入消息（末块含封口常数）
	s0 := o0 + m[0] + seal[0]
	s1 := o1 + m[1] + seal[1]
	s2 := o2 + m[2] + seal[2]
	s3 := o3 + m[3] + seal[3]
	s4 := o4 + m[4] + seal[4]
	s5 := o5 + m[5] + seal[5]
	s6 := o6 + m[6] + seal[6]
	s7 := o7 + m[7] + seal[7]

	// 16 轮 ARX（偶数轮 G_Mix，奇数轮 H_Mix），每轮异或圆周率派生的轮常数做域分离
	for r := 0; r < arxRounds; r++ {
		s0, s1, s2, s3, s4, s5, s6, s7 = bastionARXRound(s0, s1, s2, s3, s4, s5, s6, s7, r)
		sc := &roundConstants[r]
		s0 ^= sc[0]
		s1 ^= sc[1]
		s2 ^= sc[2]
		s3 ^= sc[3]
		s4 ^= sc[4]
		s5 ^= sc[5]
		s6 ^= sc[6]
		s7 ^= sc[7]
	}

	// Davies-Meyer 前馈
	state[0] = (s0 ^ o0) + m[0]
	state[1] = (s1 ^ o1) + m[1]
	state[2] = (s2 ^ o2) + m[2]
	state[3] = (s3 ^ o3) + m[3]
	state[4] = (s4 ^ o4) + m[4]
	state[5] = (s5 ^ o5) + m[5]
	state[6] = (s6 ^ o6) + m[6]
	state[7] = (s7 ^ o7) + m[7]
}

// CTR模式

// CTRStream CTR 流加密
type CTRStream struct {
	block     *BlockCipher
	counter   [32]byte
	keystream [32]byte
	pos       int
}

// NewCTR 创建CTR流，nonce 24字节，计数器从0开始递增
func NewCTR(block *BlockCipher, nonce []byte) *CTRStream {
	if len(nonce) != NonceSize {
		panic(fmt.Sprintf("bastion CTR: nonce must be %d bytes", NonceSize))
	}
	var counter [BlockSize]byte
	copy(counter[:ctrCounterSize], nonce)
	return newCTRFromCounter(block, counter[:])
}

// newCTRFromCounter 用完整32字节counter块创建CTR流（内部用）
// counter 前24字节为 nonce，后8字节为计数器初值
func newCTRFromCounter(block *BlockCipher, counter []byte) *CTRStream {
	s := &CTRStream{
		block: block,
		pos:   BlockSize,
	}
	copy(s.counter[:], counter)
	return s
}

func (s *CTRStream) XORKeyStream(dst, src []byte) {
	for len(src) > 0 {
		if s.pos >= BlockSize {
			s.block.Encrypt(s.keystream[:], s.counter[:])
			s.pos = 0
			// 计数器 64 位小端一次性递增，避免字节循环进位的分支开销
			ctr := binary.LittleEndian.Uint64(s.counter[ctrCounterSize:]) + 1
			binary.LittleEndian.PutUint64(s.counter[ctrCounterSize:], ctr)
		}
		n := BlockSize - s.pos
		if n > len(src) {
			n = len(src)
		}
		// uint64批量异或加速
		xorWords(dst[:n], src[:n], s.keystream[s.pos:s.pos+n])
		s.pos += n
		src = src[n:]
		dst = dst[n:]
	}
}

// CTREncrypt 使用Bastion-CTR模式加密，nonce 24字节
func CTREncrypt(key, nonce, plaintext []byte) ([]byte, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("bastion CTR: key must be 32 bytes")
	}
	if len(nonce) != NonceSize {
		return nil, fmt.Errorf("bastion CTR: nonce must be %d bytes", NonceSize)
	}

	// 内部构造32字节counter块：前24字节nonce + 后8字节计数器初值0
	counter := make([]byte, BlockSize)
	copy(counter[:ctrCounterSize], nonce)
	return ctrEncryptWithIV(key, counter, plaintext)
}

// CTRDecrypt 使用Bastion-CTR模式解密（CTR加解密相同）
func CTRDecrypt(key, nonce, ciphertext []byte) ([]byte, error) {
	return CTREncrypt(key, nonce, ciphertext)
}

// CTRCryptBytes CTREncrypt/CTRDecrypt 的别名，语义上便于表达"加解密"
func CTRCryptBytes(key, nonce, data []byte) ([]byte, error) {
	return CTREncrypt(key, nonce, data)
}

func ctrEncryptWithIV(key, iv, plaintext []byte) ([]byte, error) {
	block, err := NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("bastion CTR: %w", err)
	}
	defer block.Zeroize()

	ct := make([]byte, len(plaintext))
	n := len(plaintext)
	if n == 0 {
		return ct, nil
	}
	blocks := n / BlockSize
	fullBytes := blocks * BlockSize
	// 完整块部分：大数据量走多核并行，小数据量走 4 块批量模式
	if fullBytes >= parallelThreshold {
		ctrCryptParallel(block, ct[:fullBytes], plaintext[:fullBytes], iv)
	} else if fullBytes > 0 {
		encryptBlocksBatch4(block, ct[:fullBytes], plaintext[:fullBytes], iv)
	}
	// 尾部不足一块：用流式 XORKeyStream 处理（计数器需推进到第 blocks 块）
	if fullBytes < n {
		var tailCtr [BlockSize]byte
		copy(tailCtr[:], iv)
		baseCtr := binary.LittleEndian.Uint64(iv[ctrCounterSize:]) + uint64(blocks)
		binary.LittleEndian.PutUint64(tailCtr[ctrCounterSize:], baseCtr)
		stream := newCTRFromCounter(block, tailCtr[:])
		stream.XORKeyStream(ct[fullBytes:], plaintext[fullBytes:])
	}
	return ct, nil
}

// ctrEncryptWithBlock 用已有 BlockCipher 做批量 CTR 加密，复用轮密钥
func ctrEncryptWithBlock(c *BlockCipher, nonce, dst, src []byte) {
	n := len(src)
	if n == 0 {
		return
	}
	var ctr [BlockSize]byte
	copy(ctr[:ctrCounterSize], nonce)
	// 后 8 字节默认为 0（Go 初始化），即起始计数器为 0

	blocks := n / BlockSize
	fullBytes := blocks * BlockSize
	if fullBytes >= parallelThreshold {
		ctrCryptParallel(c, dst[:fullBytes], src[:fullBytes], ctr[:])
	} else if fullBytes > 0 {
		encryptBlocksBatch4(c, dst[:fullBytes], src[:fullBytes], ctr[:])
	}
	if fullBytes < n {
		var tailCtr [BlockSize]byte
		copy(tailCtr[:], ctr[:])
		baseCtr := binary.LittleEndian.Uint64(ctr[ctrCounterSize:]) + uint64(blocks)
		binary.LittleEndian.PutUint64(tailCtr[ctrCounterSize:], baseCtr)
		stream := newCTRFromCounter(c, tailCtr[:])
		stream.XORKeyStream(dst[fullBytes:], src[fullBytes:])
	}
}

// getOptimalWorkerCount 根据数据量算合适的并行数，每 worker 至少 1024 块，上限 8 个
func getOptimalWorkerCount(dataSize int) int {
	cpuCount := runtime.NumCPU()
	if cpuCount <= 1 {
		return 1
	}
	blocks := dataSize / BlockSize
	if blocks < cpuCount*minBlocksPerWorker {
		w := blocks / minBlocksPerWorker
		if w < 1 {
			w = 1
		}
		if w > maxWorkers {
			w = maxWorkers
		}
		return w
	}
	if cpuCount > maxWorkers {
		return maxWorkers
	}
	return cpuCount
}

// encryptBlocksBatch4 软件 SIMD 4 块批量生成 CTR 密钥流并 XOR 写回 dst，4 组独立变量交错执行提升单核吞吐
// amd64 平台优先走汇编 SIMD（SSE2 4 块 / AVX2 8 块），尾部回退本软件实现
func encryptBlocksBatch4(c *BlockCipher, dst, src, counter []byte) {
	n := len(src) / BlockSize
	if n == 0 {
		return
	}
	if encryptBlocksDispatch(c, dst, src, counter) {
		return
	}
	// unsafe 一次性加载 nonce 前 24 字节为 6 个 uint32（小端平台语义）
	ctrPtr := (*[6]uint32)(unsafe.Pointer(&counter[0]))
	n0 := ctrPtr[0]
	n1 := ctrPtr[1]
	n2 := ctrPtr[2]
	n3 := ctrPtr[3]
	n4 := ctrPtr[4]
	n5 := ctrPtr[5]
	baseCtr := binary.LittleEndian.Uint64(counter[24:32])
	keys := &c.roundKeys
	rounds := c.rounds

	// 4 块独立状态字（消除数组索引与边界检查）
	var s0_0, s0_1, s0_2, s0_3 uint32
	var s1_0, s1_1, s1_2, s1_3 uint32
	var s2_0, s2_1, s2_2, s2_3 uint32
	var s3_0, s3_1, s3_2, s3_3 uint32
	var s4_0, s4_1, s4_2, s4_3 uint32
	var s5_0, s5_1, s5_2, s5_3 uint32
	var s6_0, s6_1, s6_2, s6_3 uint32
	var s7_0, s7_1, s7_2, s7_3 uint32

	i := 0
	// 4 块批量主循环
	for ; i+3 < n; i += 4 {
		// 初始化 4 组状态字
		c0 := baseCtr + uint64(i)
		c1 := baseCtr + uint64(i+1)
		c2 := baseCtr + uint64(i+2)
		c3 := baseCtr + uint64(i+3)

		s0_0, s1_0, s2_0, s3_0 = n0, n1, n2, n3
		s4_0, s5_0, s6_0, s7_0 = n4, n5, uint32(c0), uint32(c0>>32)
		s0_1, s1_1, s2_1, s3_1 = n0, n1, n2, n3
		s4_1, s5_1, s6_1, s7_1 = n4, n5, uint32(c1), uint32(c1>>32)
		s0_2, s1_2, s2_2, s3_2 = n0, n1, n2, n3
		s4_2, s5_2, s6_2, s7_2 = n4, n5, uint32(c2), uint32(c2>>32)
		s0_3, s1_3, s2_3, s3_3 = n0, n1, n2, n3
		s4_3, s5_3, s6_3, s7_3 = n4, n5, uint32(c3), uint32(c3>>32)

		// 输入白化（4 组共享轮密钥）
		k0 := uint32(keys[0])
		k1 := uint32(keys[0] >> 32)
		k2 := uint32(keys[1])
		k3 := uint32(keys[1] >> 32)
		k4 := uint32(keys[2])
		k5 := uint32(keys[2] >> 32)
		k6 := uint32(keys[3])
		k7 := uint32(keys[3] >> 32)
		s0_0 ^= k0
		s1_0 ^= k1
		s2_0 ^= k2
		s3_0 ^= k3
		s4_0 ^= k4
		s5_0 ^= k5
		s6_0 ^= k6
		s7_0 ^= k7
		s0_1 ^= k0
		s1_1 ^= k1
		s2_1 ^= k2
		s3_1 ^= k3
		s4_1 ^= k4
		s5_1 ^= k5
		s6_1 ^= k6
		s7_1 ^= k7
		s0_2 ^= k0
		s1_2 ^= k1
		s2_2 ^= k2
		s3_2 ^= k3
		s4_2 ^= k4
		s5_2 ^= k5
		s6_2 ^= k6
		s7_2 ^= k7
		s0_3 ^= k0
		s1_3 ^= k1
		s2_3 ^= k2
		s3_3 ^= k3
		s4_3 ^= k4
		s5_3 ^= k5
		s6_3 ^= k6
		s7_3 ^= k7

		ki := 4
		for r := 0; r < rounds; r++ {
			k0 = uint32(keys[ki])
			k1 = uint32(keys[ki] >> 32)
			k2 = uint32(keys[ki+1])
			k3 = uint32(keys[ki+1] >> 32)
			k4 = uint32(keys[ki+2])
			k5 = uint32(keys[ki+2] >> 32)
			k6 = uint32(keys[ki+3])
			k7 = uint32(keys[ki+3] >> 32)

			if r&1 == 0 {
				// G_Mix 4 组（完全展开，独立变量）
				// 组0
				s0_0 += rotr32(s2_0, GM_R5)
				s1_0 += rotr32(s3_0, GM_R5)
				s6_0 ^= s0_0
				s7_0 ^= s1_0
				s6_0 = rotr32(s6_0, GM_R1)
				s7_0 = rotr32(s7_0, GM_R1)
				s4_0 += rotr32(s6_0, GM_R6)
				s5_0 += rotr32(s7_0, GM_R6)
				s2_0 ^= s4_0
				s3_0 ^= s5_0
				s2_0 = rotr32(s2_0, GM_R2)
					s3_0 = rotr32(s3_0, GM_R2)
					s0_0 ^= rotr32(s4_0, GM_R9)
					s1_0 ^= rotr32(s5_0, GM_R9)
					s0_0 += rotr32(s2_0, GM_R7)
					s1_0 += rotr32(s3_0, GM_R7)
					s6_0 ^= s0_0
					s7_0 ^= s1_0
					s6_0 = rotr32(s6_0, GM_R3)
					s7_0 = rotr32(s7_0, GM_R3)
					s4_0 += rotr32(s6_0, GM_R8)
					s5_0 += rotr32(s7_0, GM_R8)
					s2_0 ^= s4_0
					s3_0 ^= s5_0
					s2_0 = rotr32(s2_0, GM_R4)
					s3_0 = rotr32(s3_0, GM_R4)
					// 组1
				s0_1 += rotr32(s2_1, GM_R5)
				s1_1 += rotr32(s3_1, GM_R5)
				s6_1 ^= s0_1
				s7_1 ^= s1_1
				s6_1 = rotr32(s6_1, GM_R1)
				s7_1 = rotr32(s7_1, GM_R1)
				s4_1 += rotr32(s6_1, GM_R6)
				s5_1 += rotr32(s7_1, GM_R6)
				s2_1 ^= s4_1
				s3_1 ^= s5_1
				s2_1 = rotr32(s2_1, GM_R2)
					s3_1 = rotr32(s3_1, GM_R2)
					s0_1 ^= rotr32(s4_1, GM_R9)
					s1_1 ^= rotr32(s5_1, GM_R9)
					s0_1 += rotr32(s2_1, GM_R7)
				s1_1 += rotr32(s3_1, GM_R7)
				s6_1 ^= s0_1
				s7_1 ^= s1_1
				s6_1 = rotr32(s6_1, GM_R3)
				s7_1 = rotr32(s7_1, GM_R3)
				s4_1 += rotr32(s6_1, GM_R8)
				s5_1 += rotr32(s7_1, GM_R8)
				s2_1 ^= s4_1
				s3_1 ^= s5_1
				s2_1 = rotr32(s2_1, GM_R4)
				s3_1 = rotr32(s3_1, GM_R4)
				// 组2
				s0_2 += rotr32(s2_2, GM_R5)
				s1_2 += rotr32(s3_2, GM_R5)
				s6_2 ^= s0_2
				s7_2 ^= s1_2
				s6_2 = rotr32(s6_2, GM_R1)
				s7_2 = rotr32(s7_2, GM_R1)
				s4_2 += rotr32(s6_2, GM_R6)
				s5_2 += rotr32(s7_2, GM_R6)
				s2_2 ^= s4_2
				s3_2 ^= s5_2
				s2_2 = rotr32(s2_2, GM_R2)
					s3_2 = rotr32(s3_2, GM_R2)
					s0_2 ^= rotr32(s4_2, GM_R9)
					s1_2 ^= rotr32(s5_2, GM_R9)
					s0_2 += rotr32(s2_2, GM_R7)
				s1_2 += rotr32(s3_2, GM_R7)
				s6_2 ^= s0_2
				s7_2 ^= s1_2
				s6_2 = rotr32(s6_2, GM_R3)
				s7_2 = rotr32(s7_2, GM_R3)
				s4_2 += rotr32(s6_2, GM_R8)
				s5_2 += rotr32(s7_2, GM_R8)
				s2_2 ^= s4_2
				s3_2 ^= s5_2
				s2_2 = rotr32(s2_2, GM_R4)
				s3_2 = rotr32(s3_2, GM_R4)
				// 组3
				s0_3 += rotr32(s2_3, GM_R5)
				s1_3 += rotr32(s3_3, GM_R5)
				s6_3 ^= s0_3
				s7_3 ^= s1_3
				s6_3 = rotr32(s6_3, GM_R1)
				s7_3 = rotr32(s7_3, GM_R1)
				s4_3 += rotr32(s6_3, GM_R6)
				s5_3 += rotr32(s7_3, GM_R6)
				s2_3 ^= s4_3
				s3_3 ^= s5_3
				s2_3 = rotr32(s2_3, GM_R2)
					s3_3 = rotr32(s3_3, GM_R2)
					s0_3 ^= rotr32(s4_3, GM_R9)
					s1_3 ^= rotr32(s5_3, GM_R9)
					s0_3 += rotr32(s2_3, GM_R7)
				s1_3 += rotr32(s3_3, GM_R7)
				s6_3 ^= s0_3
				s7_3 ^= s1_3
				s6_3 = rotr32(s6_3, GM_R3)
				s7_3 = rotr32(s7_3, GM_R3)
				s4_3 += rotr32(s6_3, GM_R8)
				s5_3 += rotr32(s7_3, GM_R8)
				s2_3 ^= s4_3
				s3_3 ^= s5_3
				s2_3 = rotr32(s2_3, GM_R4)
				s3_3 = rotr32(s3_3, GM_R4)
			} else {
				// H_Mix 4 组（完全展开，独立变量）
				// 组0
				s0_0 += rotr32(s2_0, HM_R5)
				s1_0 += rotr32(s3_0, HM_R5)
				s6_0 ^= s0_0
				s7_0 ^= s1_0
				s6_0 = rotr32(s6_0, HM_R1)
				s7_0 = rotr32(s7_0, HM_R1)
				s4_0 += rotr32(s6_0, HM_R6)
				s5_0 += rotr32(s7_0, HM_R6)
				s2_0 ^= s4_0
				s3_0 ^= s5_0
				s2_0 = rotr32(s2_0, HM_R2)
					s3_0 = rotr32(s3_0, HM_R2)
					s0_0 ^= rotr32(s4_0, HM_R9)
					s1_0 ^= rotr32(s5_0, HM_R9)
					s0_0 += rotr32(s2_0, HM_R7)
				s1_0 += rotr32(s3_0, HM_R7)
				s6_0 ^= s0_0
				s7_0 ^= s1_0
				s6_0 = rotr32(s6_0, HM_R3)
				s7_0 = rotr32(s7_0, HM_R3)
				s4_0 += rotr32(s6_0, HM_R8)
				s5_0 += rotr32(s7_0, HM_R8)
				s2_0 ^= s4_0
				s3_0 ^= s5_0
				s2_0 = rotr32(s2_0, HM_R4)
				s3_0 = rotr32(s3_0, HM_R4)
				// 组1
				s0_1 += rotr32(s2_1, HM_R5)
				s1_1 += rotr32(s3_1, HM_R5)
				s6_1 ^= s0_1
				s7_1 ^= s1_1
				s6_1 = rotr32(s6_1, HM_R1)
				s7_1 = rotr32(s7_1, HM_R1)
				s4_1 += rotr32(s6_1, HM_R6)
				s5_1 += rotr32(s7_1, HM_R6)
				s2_1 ^= s4_1
				s3_1 ^= s5_1
				s2_1 = rotr32(s2_1, HM_R2)
					s3_1 = rotr32(s3_1, HM_R2)
					s0_1 ^= rotr32(s4_1, HM_R9)
					s1_1 ^= rotr32(s5_1, HM_R9)
					s0_1 += rotr32(s2_1, HM_R7)
				s1_1 += rotr32(s3_1, HM_R7)
				s6_1 ^= s0_1
				s7_1 ^= s1_1
				s6_1 = rotr32(s6_1, HM_R3)
				s7_1 = rotr32(s7_1, HM_R3)
				s4_1 += rotr32(s6_1, HM_R8)
				s5_1 += rotr32(s7_1, HM_R8)
				s2_1 ^= s4_1
				s3_1 ^= s5_1
				s2_1 = rotr32(s2_1, HM_R4)
				s3_1 = rotr32(s3_1, HM_R4)
				// 组2
				s0_2 += rotr32(s2_2, HM_R5)
				s1_2 += rotr32(s3_2, HM_R5)
				s6_2 ^= s0_2
				s7_2 ^= s1_2
				s6_2 = rotr32(s6_2, HM_R1)
				s7_2 = rotr32(s7_2, HM_R1)
				s4_2 += rotr32(s6_2, HM_R6)
				s5_2 += rotr32(s7_2, HM_R6)
				s2_2 ^= s4_2
				s3_2 ^= s5_2
				s2_2 = rotr32(s2_2, HM_R2)
					s3_2 = rotr32(s3_2, HM_R2)
					s0_2 ^= rotr32(s4_2, HM_R9)
					s1_2 ^= rotr32(s5_2, HM_R9)
					s0_2 += rotr32(s2_2, HM_R7)
				s1_2 += rotr32(s3_2, HM_R7)
				s6_2 ^= s0_2
				s7_2 ^= s1_2
				s6_2 = rotr32(s6_2, HM_R3)
				s7_2 = rotr32(s7_2, HM_R3)
				s4_2 += rotr32(s6_2, HM_R8)
				s5_2 += rotr32(s7_2, HM_R8)
				s2_2 ^= s4_2
				s3_2 ^= s5_2
				s2_2 = rotr32(s2_2, HM_R4)
				s3_2 = rotr32(s3_2, HM_R4)
				// 组3
				s0_3 += rotr32(s2_3, HM_R5)
				s1_3 += rotr32(s3_3, HM_R5)
				s6_3 ^= s0_3
				s7_3 ^= s1_3
				s6_3 = rotr32(s6_3, HM_R1)
				s7_3 = rotr32(s7_3, HM_R1)
				s4_3 += rotr32(s6_3, HM_R6)
				s5_3 += rotr32(s7_3, HM_R6)
				s2_3 ^= s4_3
				s3_3 ^= s5_3
				s2_3 = rotr32(s2_3, HM_R2)
					s3_3 = rotr32(s3_3, HM_R2)
					s0_3 ^= rotr32(s4_3, HM_R9)
					s1_3 ^= rotr32(s5_3, HM_R9)
					s0_3 += rotr32(s2_3, HM_R7)
				s1_3 += rotr32(s3_3, HM_R7)
				s6_3 ^= s0_3
				s7_3 ^= s1_3
				s6_3 = rotr32(s6_3, HM_R3)
				s7_3 = rotr32(s7_3, HM_R3)
				s4_3 += rotr32(s6_3, HM_R8)
				s5_3 += rotr32(s7_3, HM_R8)
				s2_3 ^= s4_3
				s3_3 ^= s5_3
				s2_3 = rotr32(s2_3, HM_R4)
				s3_3 = rotr32(s3_3, HM_R4)
			}

			// ShiftRows 4 组：单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
			s0_0, s1_0, s2_0, s3_0, s4_0, s5_0, s6_0, s7_0 = s2_0, s5_0, s7_0, s6_0, s3_0, s4_0, s0_0, s1_0
			s0_1, s1_1, s2_1, s3_1, s4_1, s5_1, s6_1, s7_1 = s2_1, s5_1, s7_1, s6_1, s3_1, s4_1, s0_1, s1_1
			s0_2, s1_2, s2_2, s3_2, s4_2, s5_2, s6_2, s7_2 = s2_2, s5_2, s7_2, s6_2, s3_2, s4_2, s0_2, s1_2
			s0_3, s1_3, s2_3, s3_3, s4_3, s5_3, s6_3, s7_3 = s2_3, s5_3, s7_3, s6_3, s3_3, s4_3, s0_3, s1_3

			// AddRoundKey 4 组
			s0_0 ^= k0
			s1_0 ^= k1
			s2_0 ^= k2
			s3_0 ^= k3
			s4_0 ^= k4
			s5_0 ^= k5
			s6_0 ^= k6
			s7_0 ^= k7
			s0_1 ^= k0
			s1_1 ^= k1
			s2_1 ^= k2
			s3_1 ^= k3
			s4_1 ^= k4
			s5_1 ^= k5
			s6_1 ^= k6
			s7_1 ^= k7
			s0_2 ^= k0
			s1_2 ^= k1
			s2_2 ^= k2
			s3_2 ^= k3
			s4_2 ^= k4
			s5_2 ^= k5
			s6_2 ^= k6
			s7_2 ^= k7
			s0_3 ^= k0
			s1_3 ^= k1
			s2_3 ^= k2
			s3_3 ^= k3
			s4_3 ^= k4
			s5_3 ^= k5
			s6_3 ^= k6
			s7_3 ^= k7
			ki += 4
		}

		// 输出白化 + 写出 keystream 并 XOR src（unsafe指针批量XOR）
		k0 = uint32(keys[ki])
		k1 = uint32(keys[ki] >> 32)
		k2 = uint32(keys[ki+1])
		k3 = uint32(keys[ki+1] >> 32)
		k4 = uint32(keys[ki+2])
		k5 = uint32(keys[ki+2] >> 32)
		k6 = uint32(keys[ki+3])
		k7 = uint32(keys[ki+3] >> 32)
		s0_0 ^= k0
		s1_0 ^= k1
		s2_0 ^= k2
		s3_0 ^= k3
		s4_0 ^= k4
		s5_0 ^= k5
		s6_0 ^= k6
		s7_0 ^= k7
		s0_1 ^= k0
		s1_1 ^= k1
		s2_1 ^= k2
		s3_1 ^= k3
		s4_1 ^= k4
		s5_1 ^= k5
		s6_1 ^= k6
		s7_1 ^= k7
		s0_2 ^= k0
		s1_2 ^= k1
		s2_2 ^= k2
		s3_2 ^= k3
		s4_2 ^= k4
		s5_2 ^= k5
		s6_2 ^= k6
		s7_2 ^= k7
		s0_3 ^= k0
		s1_3 ^= k1
		s2_3 ^= k2
		s3_3 ^= k3
		s4_3 ^= k4
		s5_3 ^= k5
		s6_3 ^= k6
		s7_3 ^= k7

		// 用 unsafe 指针批量 XOR 写出 4 块
		off := i * BlockSize
		dstPtr := (*[8]uint32)(unsafe.Pointer(&dst[off]))
		srcPtr := (*[8]uint32)(unsafe.Pointer(&src[off]))
		dstPtr[0] = srcPtr[0] ^ s0_0
		dstPtr[1] = srcPtr[1] ^ s1_0
		dstPtr[2] = srcPtr[2] ^ s2_0
		dstPtr[3] = srcPtr[3] ^ s3_0
		dstPtr[4] = srcPtr[4] ^ s4_0
		dstPtr[5] = srcPtr[5] ^ s5_0
		dstPtr[6] = srcPtr[6] ^ s6_0
		dstPtr[7] = srcPtr[7] ^ s7_0

		off += BlockSize
		dstPtr = (*[8]uint32)(unsafe.Pointer(&dst[off]))
		srcPtr = (*[8]uint32)(unsafe.Pointer(&src[off]))
		dstPtr[0] = srcPtr[0] ^ s0_1
		dstPtr[1] = srcPtr[1] ^ s1_1
		dstPtr[2] = srcPtr[2] ^ s2_1
		dstPtr[3] = srcPtr[3] ^ s3_1
		dstPtr[4] = srcPtr[4] ^ s4_1
		dstPtr[5] = srcPtr[5] ^ s5_1
		dstPtr[6] = srcPtr[6] ^ s6_1
		dstPtr[7] = srcPtr[7] ^ s7_1

		off += BlockSize
		dstPtr = (*[8]uint32)(unsafe.Pointer(&dst[off]))
		srcPtr = (*[8]uint32)(unsafe.Pointer(&src[off]))
		dstPtr[0] = srcPtr[0] ^ s0_2
		dstPtr[1] = srcPtr[1] ^ s1_2
		dstPtr[2] = srcPtr[2] ^ s2_2
		dstPtr[3] = srcPtr[3] ^ s3_2
		dstPtr[4] = srcPtr[4] ^ s4_2
		dstPtr[5] = srcPtr[5] ^ s5_2
		dstPtr[6] = srcPtr[6] ^ s6_2
		dstPtr[7] = srcPtr[7] ^ s7_2

		off += BlockSize
		dstPtr = (*[8]uint32)(unsafe.Pointer(&dst[off]))
		srcPtr = (*[8]uint32)(unsafe.Pointer(&src[off]))
		dstPtr[0] = srcPtr[0] ^ s0_3
		dstPtr[1] = srcPtr[1] ^ s1_3
		dstPtr[2] = srcPtr[2] ^ s2_3
		dstPtr[3] = srcPtr[3] ^ s3_3
		dstPtr[4] = srcPtr[4] ^ s4_3
		dstPtr[5] = srcPtr[5] ^ s5_3
		dstPtr[6] = srcPtr[6] ^ s6_3
		dstPtr[7] = srcPtr[7] ^ s7_3
	}

	// 尾部不足 4 块：用单块 Encrypt 处理
	if i < n {
		var ks [BlockSize]byte
		var ctrBuf [BlockSize]byte
		copy(ctrBuf[:24], counter[:24])
		for ; i < n; i++ {
			binary.LittleEndian.PutUint64(ctrBuf[24:], baseCtr+uint64(i))
			c.Encrypt(ks[:], ctrBuf[:])
			off := i * BlockSize
			xorWords(dst[off:off+BlockSize], src[off:off+BlockSize], ks[:])
		}
	}
}

// ctrCryptParallel 多核并行 CTR 加密/解密，dst/src 长度须为 BlockSize 整数倍
func ctrCryptParallel(c *BlockCipher, dst, src, counter []byte) {
	n := len(src)
	workers := getOptimalWorkerCount(n)
	if workers <= 1 {
		encryptBlocksBatch4(c, dst, src, counter)
		return
	}

	blocks := n / BlockSize
	blocksPerWorker := (blocks + workers - 1) / workers
	var wg sync.WaitGroup
	wg.Add(workers)

	baseCtr := binary.LittleEndian.Uint64(counter[24:32])
	for w := 0; w < workers; w++ {
		startBlock := w * blocksPerWorker
		endBlock := startBlock + blocksPerWorker
		if endBlock > blocks {
			endBlock = blocks
		}
		if startBlock >= blocks {
			wg.Done()
			continue
		}

		// 每个 worker 独立的 counter 副本（前 24B nonce + 推进后的计数器）
		ctr := make([]byte, BlockSize)
		copy(ctr[:24], counter[:24])
		binary.LittleEndian.PutUint64(ctr[24:], baseCtr+uint64(startBlock))

		go func(startIdx, endIdx int, ctr []byte) {
			defer wg.Done()
			startOff := startIdx * BlockSize
			endOff := endIdx * BlockSize
			encryptBlocksBatch4(c, dst[startOff:endOff], src[startOff:endOff], ctr)
		}(startBlock, endBlock, ctr)
	}
	wg.Wait()
}

// CBC模式

// cbcEncryptBlock src XOR prev 后内联加密，写 dst 并更新 prev。
// 合并 xorBytes + block.Encrypt + copy(prev) 为一步：unsafe 批量 XOR + 寄存器内联加密
func cbcEncryptBlock(c *BlockCipher, dst, src, prev []byte) {
	s := *(*[blockWords]uint32)(unsafe.Pointer(&src[0]))
	s64 := (*[4]uint64)(unsafe.Pointer(&s))
	p64 := (*[4]uint64)(unsafe.Pointer(&prev[0]))
	s64[0] ^= p64[0]
	s64[1] ^= p64[1]
	s64[2] ^= p64[2]
	s64[3] ^= p64[3]

	s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7] = encryptBlockInline(
		s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], c.rounds, &c.roundKeys)

	*(*[blockWords]uint32)(unsafe.Pointer(&dst[0])) = s
	*(*[4]uint64)(unsafe.Pointer(&prev[0])) = *(*[4]uint64)(unsafe.Pointer(&dst[0]))
}

// CBCEncryptStream Bastion-CBC 流式加密，从 r 读取写入 w。
// 内部 4MB 缓冲池逐块读取，PKCS#7 填充在 EOF 时完成
func CBCEncryptStream(key, iv []byte, r io.Reader, w io.Writer) error {
	if len(key) != 32 {
		return fmt.Errorf("bastion CBC: key must be 32 bytes")
	}
	if len(iv) != BlockSize {
		return fmt.Errorf("bastion CBC: iv must be %d bytes", BlockSize)
	}

	block, err := NewCipher(key)
	if err != nil {
		return fmt.Errorf("bastion CBC: %w", err)
	}
	defer block.Zeroize()

	bufPtr := streamBufPool.Get().(*[]byte)
	defer streamBufPool.Put(bufPtr)
	buf := *bufPtr

	var prev [BlockSize]byte
	copy(prev[:], iv)
	carryLen := 0
	emptyReads := 0

	for {
		n, err := r.Read(buf[carryLen:])
		if n == 0 && err == nil {
			emptyReads++
			if emptyReads > maxEmptyReads {
				return fmt.Errorf("bastion CBC: %w", io.ErrNoProgress)
			}
		} else {
			emptyReads = 0
		}
		if n > 0 {
			end := carryLen + n
			data := buf[:end]
			fullEnd := (end / BlockSize) * BlockSize
			for i := 0; i < fullEnd; i += BlockSize {
				blk := data[i : i+BlockSize]
				cbcEncryptBlock(block, blk, blk, prev[:])
			}
			if _, werr := w.Write(data[:fullEnd]); werr != nil {
				return fmt.Errorf("bastion CBC: write error: %w", werr)
			}
			leftover := end - fullEnd
			copy(buf[:leftover], buf[fullEnd:end])
			carryLen = leftover
		}
		if err == io.EOF {
			// PKCS#7填充：至少1字节填充，空输入也会扩展为一个完整块。
			padLen := BlockSize - carryLen
			if padLen == 0 {
				padLen = BlockSize
			}
			var final [BlockSize]byte
			copy(final[:], buf[:carryLen])
			padByte := byte(padLen)
			for i := carryLen; i < BlockSize; i++ {
				final[i] = padByte
			}
			cbcEncryptBlock(block, final[:], final[:], prev[:])
			if _, werr := w.Write(final[:]); werr != nil {
				return fmt.Errorf("bastion CBC: write error: %w", werr)
			}
			return nil
		}
		if err != nil {
			return fmt.Errorf("bastion CBC: read error: %w", err)
		}
	}
}

// CBCDecryptStream Bastion-CBC 流式解密，保留末块 EOF 时做常量时间 PKCS#7 去填充
func CBCDecryptStream(key, iv []byte, r io.Reader, w io.Writer) error {
	if len(key) != 32 {
		return fmt.Errorf("bastion CBC: key must be 32 bytes")
	}
	if len(iv) != BlockSize {
		return fmt.Errorf("bastion CBC: iv must be %d bytes", BlockSize)
	}

	block, err := NewCipher(key)
	if err != nil {
		return fmt.Errorf("bastion CBC: %w", err)
	}
	defer block.Zeroize()

	bufPtr := streamBufPool.Get().(*[]byte)
	defer streamBufPool.Put(bufPtr)
	buf := *bufPtr

	// 批量解密缓冲：累积非末尾块，批量解密（≥64KB 自动多核）后统一 XOR 链并写出
	batchSrcPtr := streamBufPool.Get().(*[]byte)
	defer streamBufPool.Put(batchSrcPtr)
	batchSrc := (*batchSrcPtr)[:len(buf)]
	batchDstPtr := streamBufPool.Get().(*[]byte)
	defer streamBufPool.Put(batchDstPtr)
	batchDst := (*batchDstPtr)[:len(buf)]
	batchCap := len(batchSrc) / BlockSize

	var prev [BlockSize]byte
	copy(prev[:], iv)
	var pending [BlockSize]byte
	hasPending := false
	carryLen := 0
	batchCount := 0
	var batchErr error
	flushBatch := func() {
		if batchCount == 0 || batchErr != nil {
			return
		}
		batchBytes := batchCount * BlockSize
		cbcDecryptBlocks(block, batchDst[:batchBytes], batchSrc[:batchBytes])
		for i := 0; i < batchCount; i++ {
			off := i * BlockSize
			xorBytes(batchDst[off:off+BlockSize], prev[:])
			copy(prev[:], batchSrc[off:off+BlockSize])
		}
		if _, werr := w.Write(batchDst[:batchBytes]); werr != nil {
			batchErr = fmt.Errorf("bastion CBC: write error: %w", werr)
		}
		batchCount = 0
	}

	emptyReads := 0
	for {
		n, err := r.Read(buf[carryLen:])
		if n == 0 && err == nil {
			emptyReads++
			if emptyReads > maxEmptyReads {
				return fmt.Errorf("bastion CBC: %w", io.ErrNoProgress)
			}
		} else {
			emptyReads = 0
		}
		if n > 0 {
			end := carryLen + n
			data := buf[:end]
			fullEnd := (end / BlockSize) * BlockSize
			for i := 0; i < fullEnd; i += BlockSize {
				blk := data[i : i+BlockSize]
				if hasPending {
					copy(batchSrc[batchCount*BlockSize:], pending[:])
					batchCount++
					if batchCount == batchCap {
						flushBatch()
					}
				}
				copy(pending[:], blk)
				hasPending = true
			}
			if batchErr != nil {
				return batchErr
			}
			leftover := end - fullEnd
			copy(buf[:leftover], buf[fullEnd:end])
			carryLen = leftover
		}
		if err == io.EOF {
			flushBatch()
			if batchErr != nil {
				return batchErr
			}
			if carryLen != 0 {
				return fmt.Errorf("bastion CBC: invalid ciphertext length")
			}
			if !hasPending {
				return fmt.Errorf("bastion CBC: invalid ciphertext length")
			}
			var plain [BlockSize]byte
			block.Decrypt(plain[:], pending[:])
			xorBytes(plain[:], prev[:])

			// 常量时间验证 PKCS#7 填充。
			padLen := int(plain[BlockSize-1])
			pad := uint32(padLen)
			bad := byte(((pad-1)>>31)|((uint32(BlockSize)-pad)>>31)) & 1
			for i := 0; i < BlockSize; i++ {
				b := plain[BlockSize-1-i]
				idxDiff := uint32(int32(padLen - i - 1))
				mask := byte(0) - byte((idxDiff>>31)^1)
				bad |= (b ^ byte(padLen)) & mask
			}
			if subtle.ConstantTimeEq(int32(bad), 0) != 1 {
				return fmt.Errorf("bastion CBC: invalid padding")
			}
			writeLen := BlockSize - padLen
			if writeLen > 0 {
				if _, werr := w.Write(plain[:writeLen]); werr != nil {
					return fmt.Errorf("bastion CBC: write error: %w", werr)
				}
			}
			return nil
		}
		if err != nil {
			return fmt.Errorf("bastion CBC: read error: %w", err)
		}
	}
}

// CBCEncrypt CBC 加密（字节级，流式输出）
func CBCEncrypt(key, iv, plaintext []byte, w io.Writer) error {
	return CBCEncryptStream(key, iv, bytes.NewReader(plaintext), w)
}

// CBCDecrypt CBC 解密（字节级，流式输出）
func CBCDecrypt(key, iv, ciphertext []byte, w io.Writer) error {
	return CBCDecryptStream(key, iv, bytes.NewReader(ciphertext), w)
}

// CBCEncryptBytes CBC 加密返回密文字节
func CBCEncryptBytes(key, iv, plaintext []byte) ([]byte, error) {
	var buf bytes.Buffer
	buf.Grow(len(plaintext) + BlockSize)
	if err := CBCEncryptStream(key, iv, bytes.NewReader(plaintext), &buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// decryptBlocksBatch4 4 块粒度批量解密 CBC 密文块（仅解密，不含 XOR 链），4 块之间无数据依赖
// amd64 平台优先走汇编 SIMD（SSE2 一次 4 块，尾部回退单块），不可用时走下面的软件实现
func decryptBlocksBatch4(c *BlockCipher, dst, src []byte) {
	n := len(src) / BlockSize
	if n == 0 {
		return
	}
	if decryptFromInputDispatch(c, dst, src) {
		return
	}
	keys := &c.roundKeys
	rounds := c.rounds

	i := 0
	for ; i+3 < n; i += 4 {
		off0 := i * BlockSize
		off1 := off0 + BlockSize
		off2 := off1 + BlockSize
		off3 := off2 + BlockSize

		b0 := *(*[blockWords]uint32)(unsafe.Pointer(&src[off0]))
		b1 := *(*[blockWords]uint32)(unsafe.Pointer(&src[off1]))
		b2 := *(*[blockWords]uint32)(unsafe.Pointer(&src[off2]))
		b3 := *(*[blockWords]uint32)(unsafe.Pointer(&src[off3]))

		s0_0, s1_0, s2_0, s3_0, s4_0, s5_0, s6_0, s7_0 := decryptBlockInline(
			b0[0], b0[1], b0[2], b0[3], b0[4], b0[5], b0[6], b0[7], rounds, keys)
		s0_1, s1_1, s2_1, s3_1, s4_1, s5_1, s6_1, s7_1 := decryptBlockInline(
			b1[0], b1[1], b1[2], b1[3], b1[4], b1[5], b1[6], b1[7], rounds, keys)
		s0_2, s1_2, s2_2, s3_2, s4_2, s5_2, s6_2, s7_2 := decryptBlockInline(
			b2[0], b2[1], b2[2], b2[3], b2[4], b2[5], b2[6], b2[7], rounds, keys)
		s0_3, s1_3, s2_3, s3_3, s4_3, s5_3, s6_3, s7_3 := decryptBlockInline(
			b3[0], b3[1], b3[2], b3[3], b3[4], b3[5], b3[6], b3[7], rounds, keys)

		*(*[blockWords]uint32)(unsafe.Pointer(&dst[off0])) = [blockWords]uint32{s0_0, s1_0, s2_0, s3_0, s4_0, s5_0, s6_0, s7_0}
		*(*[blockWords]uint32)(unsafe.Pointer(&dst[off1])) = [blockWords]uint32{s0_1, s1_1, s2_1, s3_1, s4_1, s5_1, s6_1, s7_1}
		*(*[blockWords]uint32)(unsafe.Pointer(&dst[off2])) = [blockWords]uint32{s0_2, s1_2, s2_2, s3_2, s4_2, s5_2, s6_2, s7_2}
		*(*[blockWords]uint32)(unsafe.Pointer(&dst[off3])) = [blockWords]uint32{s0_3, s1_3, s2_3, s3_3, s4_3, s5_3, s6_3, s7_3}
	}

	// 尾部不足 4 块：用 decryptBlockInline 单块处理
	for ; i < n; i++ {
		off := i * BlockSize
		b := *(*[blockWords]uint32)(unsafe.Pointer(&src[off]))
		s0, s1, s2, s3, s4, s5, s6, s7 := decryptBlockInline(
			b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], rounds, keys)
		*(*[blockWords]uint32)(unsafe.Pointer(&dst[off])) = [blockWords]uint32{s0, s1, s2, s3, s4, s5, s6, s7}
	}
}

// cbcDecryptBlocks 批量解密 CBC 密文块，解密步骤互不依赖，可并行
func cbcDecryptBlocks(c *BlockCipher, dst, src []byte) {
	n := len(src)
	if n == 0 {
		return
	}
	// 大块数据启用并行，小块用串行 4 块批量 Decrypt
	if n >= parallelThreshold {
		cbcDecryptBlocksParallel(c, dst, src)
	} else {
		decryptBlocksBatch4(c, dst, src)
	}
}

// cbcDecryptBlocksParallel 多核并行解密 CBC 块，每个 worker 内部再用 4 块批量
func cbcDecryptBlocksParallel(c *BlockCipher, dst, src []byte) {
	n := len(src)
	workers := getOptimalWorkerCount(n)
	if workers <= 1 {
		decryptBlocksBatch4(c, dst, src)
		return
	}

	blocks := n / BlockSize
	blocksPerWorker := (blocks + workers - 1) / workers
	var wg sync.WaitGroup
	wg.Add(workers)

	for w := 0; w < workers; w++ {
		startBlock := w * blocksPerWorker
		endBlock := startBlock + blocksPerWorker
		if endBlock > blocks {
			endBlock = blocks
		}
		if startBlock >= blocks {
			wg.Done()
			continue
		}
		go func(startIdx, endIdx int) {
			defer wg.Done()
			startOff := startIdx * BlockSize
			endOff := endIdx * BlockSize
			decryptBlocksBatch4(c, dst[startOff:endOff], src[startOff:endOff])
		}(startBlock, endBlock)
	}
	wg.Wait()
}

// cbcDecryptBulk 批量 CBC 解密：并行解密 → 串行 XOR 链 → 去填充
func cbcDecryptBulk(c *BlockCipher, iv, dst, ciphertext []byte) (int, error) {
	n := len(ciphertext)
	// 块对齐与输出容量由调用方保证，这里再挡一次，避免下面的 n-BlockSize 取到负下标
	if n < BlockSize || n%BlockSize != 0 || len(dst) < n {
		return 0, fmt.Errorf("bastion CBC: invalid ciphertext length")
	}
	// 第一步：并行/批量解密所有块到 dst
	cbcDecryptBlocks(c, dst, ciphertext)

	// 第二步：串行 XOR prev 链
	var prev [BlockSize]byte
	copy(prev[:], iv)
	var cur [BlockSize]byte
	for i := 0; i < n; i += BlockSize {
		// 先保存本块原始密文再写 dst：dst 与 ciphertext 同一块内存时写出会覆盖待读的密文
		copy(cur[:], ciphertext[i:i+BlockSize])
		xorBytes(dst[i:i+BlockSize], prev[:])
		copy(prev[:], cur[:])
	}

	// 第三步：常量时间验证 PKCS#7 填充
	last := dst[n-BlockSize:]
	padLen := int(last[BlockSize-1])
	pad := uint32(padLen)
	bad := byte(((pad-1)>>31)|((uint32(BlockSize)-pad)>>31)) & 1
	for i := 0; i < BlockSize; i++ {
		b := last[BlockSize-1-i]
		idxDiff := uint32(int32(padLen - i - 1))
		mask := byte(0) - byte((idxDiff>>31)^1)
		bad |= (b ^ byte(padLen)) & mask
	}
	if subtle.ConstantTimeEq(int32(bad), 0) != 1 {
		return 0, fmt.Errorf("bastion CBC: invalid padding")
	}
	return n - padLen, nil
}

// CBCDecryptBytes CBC 解密返回明文字节，批量解密路径避免流式逐块开销
func CBCDecryptBytes(key, iv, ciphertext []byte) ([]byte, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("bastion CBC: key must be 32 bytes")
	}
	if len(iv) != BlockSize {
		return nil, fmt.Errorf("bastion CBC: iv must be %d bytes", BlockSize)
	}
	if len(ciphertext) == 0 || len(ciphertext)%BlockSize != 0 {
		return nil, fmt.Errorf("bastion CBC: invalid ciphertext length")
	}

	block, err := NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("bastion CBC: %w", err)
	}
	defer block.Zeroize()

	result := make([]byte, len(ciphertext))
	plainLen, err := cbcDecryptBulk(block, iv, result, ciphertext)
	if err != nil {
		return nil, err
	}
	return result[:plainLen], nil
}

// GCM模式

// deriveAEADKeys 从主密钥与 nonce 派生加密和认证子密钥，nonce 作为 HKDF salt
func deriveAEADKeys(master, nonce []byte) (encKey, macKey []byte, err error) {
	prk := HKDFExtract(master, nonce)
	encKey, err = HKDFExpand(prk, []byte("aead-key"), BlockSize)
	if err != nil {
		return nil, nil, fmt.Errorf("bastion AEAD: derive encKey: %w", err)
	}
	macKey, err = HKDFExpand(prk, []byte("tag-key"), BlockSize)
	if err != nil {
		return nil, nil, fmt.Errorf("bastion AEAD: derive macKey: %w", err)
	}
	return encKey, macKey, nil
}

// aeadWriteMACLen 写入 AEAD 尾部双长度域：le64(len_aad) || le64(len_ct)
func aeadWriteMACLen(mac *HMAC, aadLen, ctLen int) {
	var lenBuf [16]byte
	binary.LittleEndian.PutUint64(lenBuf[0:8], uint64(aadLen))
	binary.LittleEndian.PutUint64(lenBuf[8:16], uint64(ctLen))
	mac.Write(lenBuf[:])
}

// aeadTagFromHMAC 从 HMAC 输出中截取 16 字节认证标签。
func aeadTagFromHMAC(sum [hashSize]byte) [GCMTagSize]byte {
	var tag [GCMTagSize]byte
	copy(tag[:], sum[:])
	return tag
}

// GCMSeal 加密并认证，子密钥从 master key 与 nonce 派生
func GCMSeal(key, nonce, plaintext, additionalData []byte) ([]byte, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("bastion GCM: key must be 32 bytes")
	}
	if len(nonce) != gcmNonceSize {
		return nil, fmt.Errorf("bastion GCM: nonce must be %d bytes", gcmNonceSize)
	}

	encKey, macKey, err := deriveAEADKeys(key, nonce)
	if err != nil {
		return nil, fmt.Errorf("bastion GCM: %w", err)
	}
	// 子密钥用完清零
	defer zeroizeBytes(encKey)
	defer zeroizeBytes(macKey)

	// CTR 加密（使用 encKey）
	ctrBlock, err := NewCipher(encKey)
	if err != nil {
		return nil, fmt.Errorf("bastion GCM: %w", err)
	}
	defer ctrBlock.Zeroize()

	result := make([]byte, len(plaintext)+GCMTagSize)
	// 直接批量 CTR（复用 ctrBlock，避免流式逐块开销）
	ctrEncryptWithBlock(ctrBlock, nonce, result[:len(plaintext)], plaintext)

	// ARX HMAC 认证（使用 macKey），输出取前 16 字节作为标签
	// macKey 已通过 HKDF(nonce) 隐式绑定 nonce，无需在 HMAC 输入中写入 nonce
	mac := NewHMAC(macKey)
	mac.Write(additionalData)
	mac.Write(result[:len(plaintext)])
	aeadWriteMACLen(mac, len(additionalData), len(plaintext))
	tag := aeadTagFromHMAC(mac.Sum())
	copy(result[len(plaintext):], tag[:])

	return result, nil
}

// GCMOpen 解密并验证，先验证 tag 再解密，不输出未验证的明文
func GCMOpen(key, nonce, data, additionalData []byte) ([]byte, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("bastion GCM: key must be 32 bytes")
	}
	if len(nonce) != gcmNonceSize {
		return nil, fmt.Errorf("bastion GCM: nonce must be %d bytes", gcmNonceSize)
	}
	if len(data) < GCMTagSize {
		return nil, fmt.Errorf("bastion GCM: data too short")
	}

	cipherLen := len(data) - GCMTagSize
	ct := data[:cipherLen]
	expectedTag := data[cipherLen:]

	encKey, macKey, err := deriveAEADKeys(key, nonce)
	if err != nil {
		return nil, fmt.Errorf("bastion GCM: %w", err)
	}
	// 子密钥用完清零
	defer zeroizeBytes(encKey)
	defer zeroizeBytes(macKey)

	// 先验证 tag（使用 macKey），验证通过再解密。
	// macKey 已通过 HKDF(nonce) 隐式绑定 nonce，无需在 HMAC 输入中写入 nonce
	mac := NewHMAC(macKey)
	mac.Write(additionalData)
	mac.Write(ct)
	aeadWriteMACLen(mac, len(additionalData), cipherLen)
	actualTag := aeadTagFromHMAC(mac.Sum())

	if subtle.ConstantTimeCompare(actualTag[:], expectedTag) != 1 {
		return nil, fmt.Errorf("bastion GCM: authentication failed")
	}

	// tag 验证通过，用 encKey 解密。
	ctrIV := make([]byte, BlockSize)
	copy(ctrIV, nonce)

	return ctrEncryptWithIV(encKey, ctrIV, ct)
}

// AEAD Bastion-CTR + ARX HMAC 认证加密实例，提供 Encrypt/Decrypt 流式 API。
// 每次操作时从 masterKey 与 nonce 派生子密钥对，用后立即 Zeroize
type AEAD struct {
	masterKey []byte // 主密钥，每次操作时与 nonce 一起派生子密钥
}

// NewAEAD 用 32 字节主密钥构造 AEAD 实例，子密钥在每次操作时按 nonce 派生
func NewAEAD(key []byte) (*AEAD, error) {
	if len(key) != 32 {
		return nil, fmt.Errorf("bastion AEAD: key must be 32 bytes")
	}
	mk := make([]byte, 32)
	copy(mk, key)
	return &AEAD{masterKey: mk}, nil
}

// NonceSize 返回 nonce 字节长度
func (a *AEAD) NonceSize() int { return gcmNonceSize }

// TagSize 返回认证标签字节长度
func (a *AEAD) TagSize() int { return GCMTagSize }

// Encrypt 用 nonce 和 aad 加密 plaintext，流式输出 ciphertext 并返回认证标签。
// 双缓冲 + channel 把 CTR 加密和 HMAC 认证并行化
func (a *AEAD) Encrypt(nonce, aad []byte, plaintext io.Reader, ciphertext io.Writer) (tag []byte, err error) {
	if len(nonce) != gcmNonceSize {
		return nil, fmt.Errorf("bastion AEAD: nonce must be %d bytes", gcmNonceSize)
	}

	// 从 masterKey 与 nonce 派生本次操作的子密钥对
	encKey, macKey, err := deriveAEADKeys(a.masterKey, nonce)
	if err != nil {
		return nil, fmt.Errorf("bastion AEAD: %w", err)
	}
	defer zeroizeBytes(encKey)
	defer zeroizeBytes(macKey)

	ctrBlock, err := NewCipher(encKey)
	if err != nil {
		return nil, fmt.Errorf("bastion AEAD: %w", err)
	}
	defer ctrBlock.Zeroize()

	// 跟踪 CTR 计数器状态（批量处理需要手动推进）
	var ctrBuf [BlockSize]byte
	copy(ctrBuf[:], nonce)

	// 复用 4MB 双缓冲池：每块缓冲既是读入缓冲又是 CTR 加密输出缓冲，省一次 4MB 拷贝
	outBufPtr1 := streamBufPool.Get().(*[]byte)
	defer streamBufPool.Put(outBufPtr1)
	outBuf1 := *outBufPtr1
	outBufPtr2 := streamBufPool.Get().(*[]byte)
	defer streamBufPool.Put(outBufPtr2)
	outBuf2 := *outBufPtr2

	// HMAC goroutine：通过 channel 在后台消费密文并计算 HMAC。
	// macKey 已通过 HKDF(nonce) 隐式绑定 nonce，无需在 HMAC 输入中写入 nonce
	type macJob struct {
		ct   []byte
		done chan struct{}
	}
	macCh := make(chan macJob, 2)
	tagCh := make(chan [GCMTagSize]byte, 1)
	go func() {
		mac := NewHMAC(macKey)
		mac.Write(aad)
		ctLen := 0
		for job := range macCh {
			mac.Write(job.ct)
			ctLen += len(job.ct)
			close(job.done)
		}
		aeadWriteMACLen(mac, len(aad), ctLen)
		tagCh <- aeadTagFromHMAC(mac.Sum())
	}()

	// 双缓冲：bufs[0]、bufs[1] 交替用于 CTR 加密，
	// dones[i] 表示 bufs[i] 是否已被 HMAC 消费完毕（可复用）。
	bufs := [2][]byte{outBuf1, outBuf2}
	bufCap := len(bufs[0])
	dones := [2]chan struct{}{make(chan struct{}), make(chan struct{})}
	close(dones[0])
	close(dones[1])

	// pending 保留不足一个块的明文尾部，跨读取边界累积，直到凑成完整块或 EOF。
	pending := make([]byte, 0, BlockSize-1)

	cur := 0
	emptyReads := 0
	for {
		// 等待当前输出缓冲可用（HMAC 已处理完上一轮该缓冲的数据）
		<-dones[cur]

		// 先把上轮遗留的不足块尾部放到缓冲开头，与新读入的明文衔接。
		if len(pending) > 0 {
			copy(bufs[cur], pending)
		}
		n, readErr := plaintext.Read(bufs[cur][len(pending):bufCap])
		if n == 0 && readErr == nil {
			// 连续空读不是 EOF，一直读下去就是死循环，达到上限就按无进展退出
			emptyReads++
			if emptyReads > maxEmptyReads {
				close(macCh)
				<-tagCh
				return nil, fmt.Errorf("bastion AEAD: %w", io.ErrNoProgress)
			}
		} else {
			emptyReads = 0
		}
		if n > 0 {
			// 只处理完整块；剩余字节留到下一轮或 EOF 时处理。
			combinedLen := len(pending) + n
			fullEnd := (combinedLen / BlockSize) * BlockSize
			if fullEnd > 0 {
				// 大数据量按 64KB 阈值走多核并行 CTR（与解密路径一致），小块保持 4 块批量
				if fullEnd >= parallelThreshold {
					ctrCryptParallel(ctrBlock, bufs[cur][:fullEnd], bufs[cur][:fullEnd], ctrBuf[:])
				} else {
					encryptBlocksBatch4(ctrBlock, bufs[cur][:fullEnd], bufs[cur][:fullEnd], ctrBuf[:])
				}
				blocks := fullEnd / BlockSize
				baseCtr := binary.LittleEndian.Uint64(ctrBuf[ctrCounterSize:]) + uint64(blocks)
				binary.LittleEndian.PutUint64(ctrBuf[ctrCounterSize:], baseCtr)
			}

			// 保留未凑成完整块的尾部。
			pending = append(pending[:0], bufs[cur][fullEnd:combinedLen]...)

			// 为该缓冲准备新的 done 信号，供下一轮复用前等待
			dones[cur] = make(chan struct{})

			// 发送密文给 HMAC goroutine
			macCh <- macJob{ct: bufs[cur][:fullEnd], done: dones[cur]}

			// 写入 ciphertext Writer
			if _, werr := ciphertext.Write(bufs[cur][:fullEnd]); werr != nil {
				close(macCh)
				<-tagCh
				return nil, fmt.Errorf("bastion AEAD: %w", werr)
			}

			cur = 1 - cur
		}

		if readErr != nil {
			if readErr == io.EOF {
				break
			}
			close(macCh)
			<-tagCh
			return nil, fmt.Errorf("bastion AEAD: %w", readErr)
		}
	}

	// 处理最后不足一个块的明文尾部。
	if len(pending) > 0 {
		<-dones[cur]
		copy(bufs[cur], pending)
		var ks [BlockSize]byte
		ctrBlock.Encrypt(ks[:], ctrBuf[:])
		xorWords(bufs[cur][:len(pending)], bufs[cur][:len(pending)], ks[:len(pending)])

		dones[cur] = make(chan struct{})
		macCh <- macJob{ct: bufs[cur][:len(pending)], done: dones[cur]}

		if _, werr := ciphertext.Write(bufs[cur][:len(pending)]); werr != nil {
			close(macCh)
			<-tagCh
			return nil, fmt.Errorf("bastion AEAD: %w", werr)
		}
	}

	close(macCh)
	tagArr := <-tagCh
	return tagArr[:], nil
}

// Decrypt 流式验证并解密 ciphertext||tag，返回 io.ReadCloser 读取明文。
// io.Pipe + 双缓冲：后台 goroutine 边读边把密文发给 HMAC 和 CTR 解密，解密后写入 pipe。
// 注意：因为 io.Reader 无法 Seek，明文会先于 tag 验证流出，调用者需检查返回的错误
func (a *AEAD) Decrypt(nonce, aad []byte, ciphertextTag io.Reader) (io.ReadCloser, error) {
	if len(nonce) != gcmNonceSize {
		return nil, fmt.Errorf("bastion AEAD: nonce must be %d bytes", gcmNonceSize)
	}

	// 从 masterKey 与 nonce 派生本次操作的子密钥对
	encKey, macKey, err := deriveAEADKeys(a.masterKey, nonce)
	if err != nil {
		return nil, fmt.Errorf("bastion AEAD: %w", err)
	}
	// 注意：zeroize 必须在 goroutine 内部 defer，因为函数返回 pr 后
	// goroutine 仍在使用 encKey/macKey，外层 defer 会过早清零。

	pr, pw := io.Pipe()

	go func() {
		defer zeroizeBytes(encKey)
		defer zeroizeBytes(macKey)

		// HMAC goroutine：后台顺序计算 ARX-HMAC。
		// macKey 已通过 HKDF(nonce) 隐式绑定 nonce，无需在 HMAC 输入中写入 nonce
		type macJob struct {
			ct   []byte
			done chan struct{}
		}
		macCh := make(chan macJob, 2)
		tagCh := make(chan [GCMTagSize]byte, 1)
		go func() {
			mac := NewHMAC(macKey)
			mac.Write(aad)
			ctLen := 0
			for job := range macCh {
				mac.Write(job.ct)
				ctLen += len(job.ct)
				close(job.done)
			}
			aeadWriteMACLen(mac, len(aad), ctLen)
			tagCh <- aeadTagFromHMAC(mac.Sum())
		}()

		// 复用 4MB 缓冲池：两块密文/组合缓冲（供 HMAC 读取）与两块明文缓冲（解密输出）。
		// 主 goroutine 把上一轮遗留的 pending 字节放到组合缓冲开头后，直接向组合缓冲
		// 读取新密文；HMAC 读取组合缓冲中的原始密文，CTR 解密到独立的明文缓冲，从而
		// 避免 in-place 解密覆盖 HMAC 输入，也避免与下一轮读取复用同一块缓冲产生竞态。
		ctBufPtr1 := streamBufPool.Get().(*[]byte)
		defer streamBufPool.Put(ctBufPtr1)
		ctBuf1 := *ctBufPtr1
		ctBufPtr2 := streamBufPool.Get().(*[]byte)
		defer streamBufPool.Put(ctBufPtr2)
		ctBuf2 := *ctBufPtr2

		ptBufPtr1 := streamBufPool.Get().(*[]byte)
		defer streamBufPool.Put(ptBufPtr1)
		ptBuf1 := *ptBufPtr1
		ptBufPtr2 := streamBufPool.Get().(*[]byte)
		defer streamBufPool.Put(ptBufPtr2)
		ptBuf2 := *ptBufPtr2

		ctrBlock, err := NewCipher(encKey)
		if err != nil {
			close(macCh)
			<-tagCh
			pw.CloseWithError(fmt.Errorf("bastion AEAD: %w", err))
			return
		}
		defer ctrBlock.Zeroize()

		var ctrBuf [BlockSize]byte
		copy(ctrBuf[:], nonce)

		ctBufs := [2][]byte{ctBuf1, ctBuf2}
		ptBufs := [2][]byte{ptBuf1, ptBuf2}
		dones := [2]chan struct{}{make(chan struct{}), make(chan struct{})}
		close(dones[0])
		close(dones[1])

		// pending 保留末尾可能包含 tag 的字节（最多 GCMTagSize + BlockSize - 1）。
		pending := make([]byte, 0, GCMTagSize+BlockSize)

		cur := 0
		emptyReads := 0
		for {
			// 等待当前组合缓冲被 HMAC 消费完（也保证上一轮不会覆盖它）。
			<-dones[cur]

			// 把 pending 移到组合缓冲开头，再读取新密文到后续空间。
			if len(pending) > 0 {
				copy(ctBufs[cur], pending)
			}
			readCap := len(ctBufs[cur]) - len(pending)
			n, readErr := ciphertextTag.Read(ctBufs[cur][len(pending) : len(pending)+readCap])
			if readErr != nil && readErr != io.EOF {
				close(macCh)
				<-tagCh
				pw.CloseWithError(fmt.Errorf("bastion AEAD: %w", readErr))
				return
			}
			if n == 0 && readErr == nil {
				// 连续空读不是 EOF：pending 里还剩不足一块的密文时会一直空转，到上限就退出
				emptyReads++
				if emptyReads > maxEmptyReads {
					close(macCh)
					<-tagCh
					pw.CloseWithError(fmt.Errorf("bastion AEAD: %w", io.ErrNoProgress))
					return
				}
			} else {
				emptyReads = 0
			}
			if n == 0 && len(pending) == 0 {
				break
			}

			combined := ctBufs[cur][:len(pending)+n]
			pending = pending[:0]

			if len(combined) <= GCMTagSize {
				pending = append(pending[:0], combined...)
				cur = 1 - cur
				if readErr == io.EOF {
					break
				}
				continue
			}

			cipherLen := len(combined) - GCMTagSize
			remaining := cipherLen % BlockSize
			processableLen := cipherLen - remaining

			dones[cur] = make(chan struct{})
			macCh <- macJob{ct: combined[:processableLen], done: dones[cur]}

			// CTR 解密：从组合缓冲（密文）解密到独立的明文缓冲。
			// 大数据量启用多核并行，小块保持 4 块批量。
			if processableLen > 0 {
				if processableLen >= parallelThreshold {
					ctrCryptParallel(ctrBlock, ptBufs[cur][:processableLen], combined[:processableLen], ctrBuf[:])
				} else {
					encryptBlocksBatch4(ctrBlock, ptBufs[cur][:processableLen], combined[:processableLen], ctrBuf[:])
				}
				blocks := processableLen / BlockSize
				baseCtr := binary.LittleEndian.Uint64(ctrBuf[ctrCounterSize:]) + uint64(blocks)
				binary.LittleEndian.PutUint64(ctrBuf[ctrCounterSize:], baseCtr)
			}

			// 写入 pipe。
			if _, werr := pw.Write(ptBufs[cur][:processableLen]); werr != nil {
				close(macCh)
				<-tagCh
				return
			}

			// 保留剩余字节（不足一个块 + tag）。
			pending = append(pending[:0], combined[processableLen:]...)

			cur = 1 - cur
			if readErr == io.EOF {
				break
			}
		}

		if len(pending) < GCMTagSize {
			close(macCh)
			<-tagCh
			pw.CloseWithError(fmt.Errorf("bastion AEAD: data too short"))
			return
		}

		// 处理最后不足一个块的密文（finalRemaining），必须在关闭 macCh 前发给 HMAC。
		finalRemaining := len(pending) - GCMTagSize
		if finalRemaining > 0 {
			done := make(chan struct{})
			macCh <- macJob{ct: pending[:finalRemaining], done: done}
			<-done

			var ks [BlockSize]byte
			ctrBlock.Encrypt(ks[:], ctrBuf[:])
			xorWords(ptBufs[cur][:finalRemaining], pending[:finalRemaining], ks[:finalRemaining])
			if _, werr := pw.Write(ptBufs[cur][:finalRemaining]); werr != nil {
				close(macCh)
				<-tagCh
				return
			}
		}

		expectedTag := pending[finalRemaining:]
		close(macCh)
		actualTag := <-tagCh

		if subtle.ConstantTimeCompare(actualTag[:], expectedTag) != 1 {
			pw.CloseWithError(fmt.Errorf("bastion AEAD: authentication failed"))
			return
		}
		pw.Close()
	}()

	return pr, nil
}

// DecryptFromSeeker 与 Decrypt 相同但要求 ciphertextTag 可 Seek，可提前读尾部 tag 验证
func (a *AEAD) DecryptFromSeeker(nonce, aad []byte, ciphertextTag io.ReadSeeker) (io.ReadCloser, error) {
	if len(nonce) != gcmNonceSize {
		return nil, fmt.Errorf("bastion AEAD: nonce must be %d bytes", gcmNonceSize)
	}

	// 从 masterKey 与 nonce 派生本次操作的子密钥对
	encKey, macKey, err := deriveAEADKeys(a.masterKey, nonce)
	if err != nil {
		return nil, fmt.Errorf("bastion AEAD: %w", err)
	}
	// 注意：zeroize 必须在 goroutine 内部 defer，因为函数返回 pr 后
	// goroutine 仍在使用 encKey/macKey，外层 defer 会过早清零。

	// 记录当前读取位置，调用方可能已把指针定位到密文区起始处（如文件头之后）
	startPos, err := ciphertextTag.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, fmt.Errorf("bastion AEAD: seek current: %w", err)
	}
	// 定位到末尾，确定密文长度并读取尾部 tag。
	totalLen, err := ciphertextTag.Seek(0, io.SeekEnd)
	if err != nil {
		return nil, fmt.Errorf("bastion AEAD: seek end: %w", err)
	}
	if totalLen-startPos < int64(GCMTagSize) {
		return nil, fmt.Errorf("bastion AEAD: data too short")
	}
	// 32 位平台 int 只有 32 位，超过上限的长度转 int 会溢出成负数导致静默跳过解密，提前拒绝
	if remaining := totalLen - startPos; remaining > int64(^uint(0)>>1) {
		return nil, fmt.Errorf("bastion AEAD: data too large for this platform")
	}
	cipherLen := int(totalLen-startPos) - GCMTagSize

	if _, err := ciphertextTag.Seek(totalLen-int64(GCMTagSize), io.SeekStart); err != nil {
		return nil, fmt.Errorf("bastion AEAD: seek tag: %w", err)
	}
	var expectedTag [GCMTagSize]byte
	if _, err := io.ReadFull(ciphertextTag, expectedTag[:]); err != nil {
		return nil, fmt.Errorf("bastion AEAD: read tag: %w", err)
	}
	if _, err := ciphertextTag.Seek(startPos, io.SeekStart); err != nil {
		return nil, fmt.Errorf("bastion AEAD: seek start: %w", err)
	}

	pr, pw := io.Pipe()

	go func() {
		defer zeroizeBytes(encKey)
		defer zeroizeBytes(macKey)

		// HMAC goroutine。
		// macKey 已通过 HKDF(nonce) 隐式绑定 nonce，无需在 HMAC 输入中写入 nonce
		type macJob struct {
			ct   []byte
			done chan struct{}
		}
		macCh := make(chan macJob, 2)
		tagCh := make(chan [GCMTagSize]byte, 1)
		go func() {
			mac := NewHMAC(macKey)
			mac.Write(aad)
			ctLen := 0
			for job := range macCh {
				mac.Write(job.ct)
				ctLen += len(job.ct)
				close(job.done)
			}
			aeadWriteMACLen(mac, len(aad), ctLen)
			tagCh <- aeadTagFromHMAC(mac.Sum())
		}()

		// 双缓冲槽位：每个槽位独立的 readBuf（供 HMAC 读取原始密文）与 decryptBuf
		// （供 CTR 解密密文后写入 pipe）。通过 done 通道确保 HMAC 消费完 readBuf 后
		// 才复用该槽位，避免数据竞争。
		type slot struct {
			readBuf    []byte
			decryptBuf []byte
			done       chan struct{}
		}
		readBufPtr1 := streamBufPool.Get().(*[]byte)
		defer streamBufPool.Put(readBufPtr1)
		readBufPtr2 := streamBufPool.Get().(*[]byte)
		defer streamBufPool.Put(readBufPtr2)
		decryptBufPtr1 := streamBufPool.Get().(*[]byte)
		defer streamBufPool.Put(decryptBufPtr1)
		decryptBufPtr2 := streamBufPool.Get().(*[]byte)
		defer streamBufPool.Put(decryptBufPtr2)

		slots := [2]slot{
			{readBuf: *readBufPtr1, decryptBuf: *decryptBufPtr1, done: make(chan struct{})},
			{readBuf: *readBufPtr2, decryptBuf: *decryptBufPtr2, done: make(chan struct{})},
		}
		close(slots[0].done)
		close(slots[1].done)

		ctrBlock, err := NewCipher(encKey)
		if err != nil {
			close(macCh)
			<-tagCh
			pw.CloseWithError(fmt.Errorf("bastion AEAD: %w", err))
			return
		}
		defer ctrBlock.Zeroize()

		var ctrBuf [BlockSize]byte
		copy(ctrBuf[:], nonce)

		cur := 0
		remaining := cipherLen
		for remaining > 0 {
			<-slots[cur].done

			n := remaining
			if n > len(slots[cur].readBuf) {
				n = len(slots[cur].readBuf)
			}
			if _, err := io.ReadFull(ciphertextTag, slots[cur].readBuf[:n]); err != nil {
				close(macCh)
				<-tagCh
				pw.CloseWithError(fmt.Errorf("bastion AEAD: read ct: %w", err))
				return
			}

			slots[cur].done = make(chan struct{})
			macCh <- macJob{ct: slots[cur].readBuf[:n], done: slots[cur].done}

			fullEnd := (n / BlockSize) * BlockSize
			if fullEnd > 0 {
				// 大数据量启用多核并行 CTR 解密，小块保持 4 块批量；
				// 直接从 readBuf 解密到 decryptBuf，省掉一次拷贝。
				if fullEnd >= parallelThreshold {
					ctrCryptParallel(ctrBlock, slots[cur].decryptBuf[:fullEnd], slots[cur].readBuf[:fullEnd], ctrBuf[:])
				} else {
					encryptBlocksBatch4(ctrBlock, slots[cur].decryptBuf[:fullEnd], slots[cur].readBuf[:fullEnd], ctrBuf[:])
				}
				blocks := fullEnd / BlockSize
				baseCtr := binary.LittleEndian.Uint64(ctrBuf[ctrCounterSize:]) + uint64(blocks)
				binary.LittleEndian.PutUint64(ctrBuf[ctrCounterSize:], baseCtr)
			}
			if fullEnd < n {
				var ks [BlockSize]byte
				ctrBlock.Encrypt(ks[:], ctrBuf[:])
				tail := n - fullEnd
				xorWords(slots[cur].decryptBuf[fullEnd:n], slots[cur].readBuf[fullEnd:n], ks[:tail])
				baseCtr := binary.LittleEndian.Uint64(ctrBuf[ctrCounterSize:]) + 1
				binary.LittleEndian.PutUint64(ctrBuf[ctrCounterSize:], baseCtr)
			}

			if _, werr := pw.Write(slots[cur].decryptBuf[:n]); werr != nil {
				close(macCh)
				<-tagCh
				return
			}

			remaining -= n
			cur = 1 - cur
		}

		close(macCh)
		actualTag := <-tagCh

		if subtle.ConstantTimeCompare(actualTag[:], expectedTag[:]) != 1 {
			pw.CloseWithError(fmt.Errorf("bastion AEAD: authentication failed"))
			return
		}
		pw.Close()
	}()

	return pr, nil
}

// XTS模式

// XTSEncrypt Bastion-XTS 模式加密磁盘数据
func XTSEncrypt(key, data []byte, sector uint64) ([]byte, error) {
	key1, key2, err := splitXTSKey(key)
	if err != nil {
		return nil, err
	}

	block1, err := NewCipher(key1)
	if err != nil {
		return nil, fmt.Errorf("bastion XTS encrypt: %w", err)
	}
	defer block1.Zeroize()

	block2, err := NewCipher(key2)
	if err != nil {
		return nil, fmt.Errorf("bastion XTS encrypt: %w", err)
	}
	defer block2.Zeroize()

	if len(data) < BlockSize {
		return nil, fmt.Errorf("bastion XTS encrypt: data too short (need at least %d bytes)", BlockSize)
	}

	return xtsProcess(block1, block2, data, sector, true), nil
}

// XTSDecrypt Bastion-XTS 模式解密磁盘数据
func XTSDecrypt(key, data []byte, sector uint64) ([]byte, error) {
	key1, key2, err := splitXTSKey(key)
	if err != nil {
		return nil, err
	}

	block1, err := NewCipher(key1)
	if err != nil {
		return nil, fmt.Errorf("bastion XTS decrypt: %w", err)
	}
	defer block1.Zeroize()

	block2, err := NewCipher(key2)
	if err != nil {
		return nil, fmt.Errorf("bastion XTS decrypt: %w", err)
	}
	defer block2.Zeroize()

	if len(data) < BlockSize {
		return nil, fmt.Errorf("bastion XTS decrypt: data too short (need at least %d bytes)", BlockSize)
	}

	return xtsProcess(block1, block2, data, sector, false), nil
}

// splitXTSKey 把 64 字节密钥拆成前 32 和后 32
func splitXTSKey(key []byte) ([]byte, []byte, error) {
	if len(key) != 64 {
		return nil, nil, fmt.Errorf("bastion XTS: invalid key size %d (must be 64 bytes)", len(key))
	}
	return key[:32], key[32:], nil
}

// xtsProcessSegment 处理一段连续完整块：按批生成 tweak，先 XOR tweak → 批量加解密 → 再 XOR tweak。
// 批大小固定，tweak 数组在栈上复用，不再为整段数据一次性分配 tweak 数组，省掉大块内存搬运。
// 返回处理完后下一块的 tweak（密文窃取要用）
func xtsProcessSegment(c *BlockCipher, buf []byte, startTweak [BlockSize]byte, encrypt bool) [BlockSize]byte {
	const batch = 64
	n := len(buf) / BlockSize
	tweak := startTweak
	var tweaks [batch][BlockSize]byte

	for off := 0; off < n; {
		cnt := n - off
		if cnt > batch {
			cnt = batch
		}
		// 批内 tweak 从批首开始逐个乘 α
		tweaks[0] = tweak
		for i := 1; i < cnt; i++ {
			tweaks[i] = tweaks[i-1]
			gfMul2_256(tweaks[i][:])
		}

		chunk := buf[off*BlockSize : (off+cnt)*BlockSize]

		// 第一遍：block ^= tweak
		for i := 0; i < cnt; i++ {
			xorBytes(chunk[i*BlockSize:(i+1)*BlockSize], tweaks[i][:])
		}

		// 批量加解密（原地），amd64 平台走汇编 SIMD，不足 4 块回退单块
		if encrypt {
			if !encryptFromInputDispatch(c, chunk, chunk) {
				for i := 0; i < cnt; i++ {
					bo := i * BlockSize
					c.Encrypt(chunk[bo:bo+BlockSize], chunk[bo:bo+BlockSize])
				}
			}
		} else {
			if !decryptFromInputDispatch(c, chunk, chunk) {
				for i := 0; i < cnt; i++ {
					bo := i * BlockSize
					c.Decrypt(chunk[bo:bo+BlockSize], chunk[bo:bo+BlockSize])
				}
			}
		}

		// 第二遍：block ^= tweak
		for i := 0; i < cnt; i++ {
			xorBytes(chunk[i*BlockSize:(i+1)*BlockSize], tweaks[i][:])
		}

		// 推进到下一批的起始 tweak
		tweak = tweaks[cnt-1]
		gfMul2_256(tweak[:])
		off += cnt
	}
	return tweak
}

// xtsProcessFullBlocksParallel 多核并行处理 XTS 完整块。
// 主线程先把 tweak 顺序推进到各 worker 的分片起点（tweak 递推有顺序依赖，不能各 worker 各推一遍），
// worker 只在分片内按批就地推进；返回 buf 处理完后下一块的 tweak
func xtsProcessFullBlocksParallel(c *BlockCipher, buf []byte, startTweak [BlockSize]byte, encrypt bool) [BlockSize]byte {
	n := len(buf)
	workers := getOptimalWorkerCount(n)
	if workers <= 1 {
		return xtsProcessSegment(c, buf, startTweak, encrypt)
	}

	blocks := n / BlockSize
	blocksPerWorker := (blocks + workers - 1) / workers

	// 分片起点 tweak：主线程顺序推进一次拿到，顺带把尾部（下一块）tweak 推到 buf 之后
	starts := make([][BlockSize]byte, workers)
	next := startTweak
	k := 0
	for w := 0; w < workers; w++ {
		starts[w] = next
		for i := 0; i < blocksPerWorker && k < blocks; i++ {
			gfMul2_256(next[:])
			k++
		}
	}
	for ; k < blocks; k++ {
		gfMul2_256(next[:])
	}

	var wg sync.WaitGroup
	wg.Add(workers)

	for w := 0; w < workers; w++ {
		startBlock := w * blocksPerWorker
		endBlock := startBlock + blocksPerWorker
		if endBlock > blocks {
			endBlock = blocks
		}
		if startBlock >= blocks {
			wg.Done()
			continue
		}

		go func(startIdx, endIdx int, startTweak [BlockSize]byte) {
			defer wg.Done()
			startOff := startIdx * BlockSize
			endOff := endIdx * BlockSize
			xtsProcessSegment(c, buf[startOff:endOff], startTweak, encrypt)
		}(startBlock, endBlock, starts[w])
	}
	wg.Wait()

	return next
}

func xtsProcess(block1, block2 *BlockCipher, data []byte, sector uint64, encrypt bool) []byte {
	n := len(data)
	result := make([]byte, n)
	copy(result, data)

	var tweak [32]byte
	binary.LittleEndian.PutUint64(tweak[24:], sector)
	block2.Encrypt(tweak[:], tweak[:])

	fullBlocks := n / BlockSize
	partial := n % BlockSize

	processBlocks := fullBlocks
	if partial > 0 {
		processBlocks = fullBlocks - 1
	}

	if processBlocks > 0 {
		// 完整块部分：按批就地推进 tweak，大数据量多核并行；返回下一块的 tweak 供密文窃取使用
		fullBytes := processBlocks * BlockSize
		if fullBytes >= parallelThreshold {
			tweak = xtsProcessFullBlocksParallel(block1, result[:fullBytes], tweak, encrypt)
		} else {
			tweak = xtsProcessSegment(block1, result[:fullBytes], tweak, encrypt)
		}
	}

	if partial > 0 && fullBlocks >= 1 {
		prevOff := (fullBlocks - 1) * BlockSize
		lastOff := fullBlocks * BlockSize

		if encrypt {
			xtsCiphertextStealingEncrypt(block1, result, prevOff, lastOff, partial, tweak[:])
		} else {
			xtsCiphertextStealingDecrypt(block1, result, prevOff, lastOff, partial, tweak[:])
		}
	}

	return result
}

func xtsCiphertextStealingEncrypt(block *BlockCipher, data []byte, prevOff, lastOff int, m int, tweak []byte) {
	// 栈上定长数组，避免每次堆分配
	var Ppartial, Pprev, CC, PP [BlockSize]byte
	copy(Ppartial[:m], data[lastOff:lastOff+m])
	copy(Pprev[:], data[prevOff:prevOff+BlockSize])

	CC = Pprev
	xorBytes(CC[:], tweak)
	block.Encrypt(CC[:], CC[:])
	xorBytes(CC[:], tweak)

	copy(data[lastOff:lastOff+m], CC[:m])

	copy(PP[:BlockSize-m], CC[m:BlockSize])
	copy(PP[BlockSize-m:], Ppartial[:m])

	gfMul2_256(tweak)

	xorBytes(PP[:], tweak)
	block.Encrypt(PP[:], PP[:])
	xorBytes(PP[:], tweak)

	copy(data[prevOff:prevOff+BlockSize], PP[:])
}

func xtsCiphertextStealingDecrypt(block *BlockCipher, data []byte, prevOff, lastOff int, m int, tweak []byte) {
	// 栈上定长数组，避免每次堆分配
	var Cpartial, tweakN1, PP, CC [BlockSize]byte
	copy(Cpartial[:m], data[lastOff:lastOff+m])

	copy(tweakN1[:], tweak)
	gfMul2_256(tweakN1[:])

	copy(PP[:], data[prevOff:prevOff+BlockSize])
	xorBytes(PP[:], tweakN1[:])
	block.Decrypt(PP[:], PP[:])
	xorBytes(PP[:], tweakN1[:])

	copy(data[lastOff:lastOff+m], PP[BlockSize-m:BlockSize])

	copy(CC[:m], Cpartial[:m])
	copy(CC[m:], PP[:BlockSize-m])

	xorBytes(CC[:], tweak)
	block.Decrypt(CC[:], CC[:])
	xorBytes(CC[:], tweak)

	copy(data[prevOff:prevOff+BlockSize], CC[:])
}

// gfMul2_256 在 GF(2^256) 中乘以 α = 2
// 32 字节看作大端 256 位整数，乘 2 = 整体左移 1 位
func gfMul2_256(x []byte) {
	w0 := binary.BigEndian.Uint64(x[0:8])
	w1 := binary.BigEndian.Uint64(x[8:16])
	w2 := binary.BigEndian.Uint64(x[16:24])
	w3 := binary.BigEndian.Uint64(x[24:32])

	// 最高位（x[0] 的 bit7）决定是否做多项式约简
	carry := w0 >> 63

	// 整段左移 1 位，低一字节的高位进位补到高一字节的最低 bit
	w0 = (w0 << 1) | (w1 >> 63)
	w1 = (w1 << 1) | (w2 >> 63)
	w2 = (w2 << 1) | (w3 >> 63)
	w3 = w3 << 1

	// 无分支：carry=1 时异或多项式 0x0425（落在 x[31]/x[30]），否则掩码归零
	m := uint64(0) - uint64(carry)
	w3 ^= 0x0425 & m

	binary.BigEndian.PutUint64(x[0:8], w0)
	binary.BigEndian.PutUint64(x[8:16], w1)
	binary.BigEndian.PutUint64(x[16:24], w2)
	binary.BigEndian.PutUint64(x[24:32], w3)
}

// xorBytes dst 逐 64 位异或 src（对齐批量加速）
// 切片首地址非 8 字节对齐时退回逐字节：转 *uint64 在对齐要求严格的架构上会 panic
func xorBytes(dst, src []byte) {
	n := len(dst) & ^7 // 向下取整到8的倍数
	if n > 0 && (uintptr(unsafe.Pointer(&dst[0]))|uintptr(unsafe.Pointer(&src[0])))&7 != 0 {
		n = 0
	}
	if n > 0 {
		pd := unsafe.Slice((*uint64)(unsafe.Pointer(&dst[0])), n>>3)
		ps := unsafe.Slice((*uint64)(unsafe.Pointer(&src[0])), n>>3)
		for i := range pd {
			pd[i] ^= ps[i]
		}
	}
	// 处理剩余字节（非对齐回退时为整段）
	for i := n; i < len(dst); i++ {
		dst[i] ^= src[i]
	}
}

// xorWords 批量异或 dst ^= src ^ key，CTR 流用
// 任一切片首地址非 8 字节对齐时退回逐字节，理由同 xorBytes
func xorWords(dst, src, key []byte) {
	n := len(dst) & ^7
	if n > 0 && (uintptr(unsafe.Pointer(&dst[0]))|uintptr(unsafe.Pointer(&src[0]))|uintptr(unsafe.Pointer(&key[0])))&7 != 0 {
		n = 0
	}
	if n > 0 {
		pd := unsafe.Slice((*uint64)(unsafe.Pointer(&dst[0])), n>>3)
		ps := unsafe.Slice((*uint64)(unsafe.Pointer(&src[0])), n>>3)
		pk := unsafe.Slice((*uint64)(unsafe.Pointer(&key[0])), n>>3)
		for i := range pd {
			pd[i] = ps[i] ^ pk[i]
		}
	}
	for i := n; i < len(dst); i++ {
		dst[i] = src[i] ^ key[i]
	}
}

// ARX哈希

// arxInitialState ARX Hash 的固定初始 IV，SHA-256 风格素数平方根小数部分，小端序 8 个 uint32
var arxInitialState = [8]uint32{
	0x6a09e667, 0xbb67ae85,
	0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c,
	0x1f83d9ab, 0x5be0cd19,
}

// Hash Bastion ARX 哈希，纯 ARX 复用 G_Mix + ShiftRows，固定 16 轮，无 S 盒无密钥扩展
type Hash struct {
	state  [8]uint32
	buf    [hashBlockSize]byte
	bufLen int
	total  uint64
}

// NewHash 创建 ARX 哈希
func NewHash() *Hash {
	h := &Hash{}
	h.Reset()
	return h
}

// loadBlockWordsLE 从 src 按小端序加载 8 个 uint32 到 dst
func loadBlockWordsLE(dst *[8]uint32, src []byte) {
	for i := 0; i < 8; i++ {
		dst[i] = binary.LittleEndian.Uint32(src[i*4:])
	}
}

func (h *Hash) Reset() {
	h.state = arxInitialState
	h.bufLen = 0
	h.total = 0
}

func (h *Hash) Write(data []byte) (int, error) {
	n := len(data)
	h.total += uint64(n)

	if h.bufLen > 0 {
		copied := copy(h.buf[h.bufLen:], data)
		h.bufLen += copied
		data = data[copied:]

		if h.bufLen < hashBlockSize {
			return n, nil
		}
		h.compress(h.buf[:])
		h.bufLen = 0
	}

	for len(data) >= hashBlockSize {
		h.compress(data[:hashBlockSize])
		data = data[hashBlockSize:]
	}

	if len(data) > 0 {
		h.bufLen = copy(h.buf[:], data)
	}

	return n, nil
}

func (h *Hash) compress(block []byte) {
	bastionARXCompress(&h.state, block, false)
}

func (h *Hash) Sum() [hashSize]byte {
	state := h.state

	dataLen := h.bufLen
	padNeed := hashBlockSize - dataLen%hashBlockSize
	if padNeed < 9 {
		padNeed += hashBlockSize
	}

	var padded [hashBlockSize * 2]byte
	copy(padded[:], h.buf[:dataLen])
	padded[dataLen] = 0x80
	binary.LittleEndian.PutUint64(padded[dataLen+padNeed-8:], h.total*8)

	totalBlockBytes := dataLen + padNeed
	for i := 0; i < totalBlockBytes; i += hashBlockSize {
		// 最后一个填充块带封口标志，标记这是消息末尾
		last := i+hashBlockSize >= totalBlockBytes
		bastionARXCompress(&state, padded[i:i+hashBlockSize], last)
	}

	var result [hashSize]byte
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint32(result[i*4:], state[i])
	}
	return result
}

func (h *Hash) Size() int { return hashSize }

func (h *Hash) BlockSize() int { return hashBlockSize }

// XOF 用当前哈希状态为密钥，CTR 模式加密零块生成可扩展输出
func (h *Hash) XOF(output []byte) []byte {
	if len(output) == 0 {
		return output
	}

	// 使用当前哈希状态（完整计算带填充的摘要）作为密钥
	state := h.Sum()

	block, err := NewCipher(state[:])
	if err != nil {
		// 如果出错，返回零值（函数签名不允许返回error，安全降级）
		for i := range output {
			output[i] = 0
		}
		return output
	}
	defer block.Zeroize()

	// 分块生成密钥流，避免一次性分配与 output 等大的零切片
	stream := NewCTR(block, make([]byte, NonceSize))
	// 使用栈上固定零块作为 CTR 输入（CTR 对零加密 = 密钥流）
	var zeroBlock [BlockSize]byte // Go 自动初始化为零

	written := 0
	for written < len(output) {
		end := written + BlockSize
		if end > len(output) {
			end = len(output)
		}
		stream.XORKeyStream(output[written:end], zeroBlock[:end-written])
		written = end
	}
	return output
}

// Sum 一次性计算哈希
func Sum(data []byte) [hashSize]byte {
	h := NewHash()
	h.Write(data)
	return h.Sum()
}

// HMAC消息认证码

// HMAC Bastion HMAC，inner/outer 两把 Hash 初始状态分别由 (key xor ipad/opad) 异或得到
type HMAC struct {
	inner *Hash
	outer *Hash
	key   [hmacBlockSize]byte // 原始密钥（用于 Reset 时重新派生）
}

// NewHMAC 创建 HMAC
func NewHMAC(key []byte) *HMAC {
	var k [hmacBlockSize]byte
	if len(key) > hmacBlockSize {
		hashed := Sum(key)
		copy(k[:], hashed[:])
	} else {
		copy(k[:], key)
	}

	inner, outer := deriveHMACStates(k)

	h := &HMAC{
		inner: inner,
		outer: outer,
		key:   k,
	}
	return h
}

// deriveHMACStates 从密钥派生 inner/outer 两个 Hash，把 ipad/opad 异或到 ARX Hash 初始状态上
func deriveHMACStates(k [hmacBlockSize]byte) (*Hash, *Hash) {
	inner := NewHash()
	outer := NewHash()
	applyHMACKeyState(inner, outer, k)
	return inner, outer
}

// applyHMACKeyState 把 ipad/opad 异或到已有的 inner/outer 状态上：先复位计数再施加密钥状态，
// 复用现成的 Hash 结构，不再每次重新分配
func applyHMACKeyState(inner, outer *Hash, k [hmacBlockSize]byte) {
	inner.Reset()
	outer.Reset()

	var ipad, opad [hmacBlockSize]byte
	for i := 0; i < hmacBlockSize; i++ {
		ipad[i] = k[i] ^ 0x36
		opad[i] = k[i] ^ 0x5C
	}

	var ipadState, opadState [8]uint32
	loadBlockWordsLE(&ipadState, ipad[:])
	loadBlockWordsLE(&opadState, opad[:])
	for i := 0; i < 8; i++ {
		inner.state[i] ^= ipadState[i]
		outer.state[i] ^= opadState[i]
	}
}

// Reset 复位到密钥初始状态，复用已有的 inner/outer
func (h *HMAC) Reset() {
	if h.inner == nil || h.outer == nil {
		// 零值 HMAC 还没底层 Hash 时才分配
		h.inner, h.outer = deriveHMACStates(h.key)
		return
	}
	applyHMACKeyState(h.inner, h.outer, h.key)
}

func (h *HMAC) Write(data []byte) (int, error) {
	return h.inner.Write(data)
}

func (h *HMAC) Sum() [32]byte {
	innerCopy := *h.inner
	innerHash := innerCopy.Sum()

	outerCopy := *h.outer
	outerCopy.Write(innerHash[:])
	return outerCopy.Sum()
}

// Compute 一次性计算 HMAC
func Compute(key, message []byte) [32]byte {
	h := NewHMAC(key)
	h.Write(message)
	return h.Sum()
}

// 密钥派生函数

// HKDFExtract HKDF 提取，HMAC(salt, ikm)
func HKDFExtract(ikm, salt []byte) [kdfHashLen]byte {
	if salt == nil {
		salt = make([]byte, kdfHashLen)
	}
	return Compute(salt, ikm)
}

// HKDFExpand HKDF 扩展
func HKDFExpand(prk [kdfHashLen]byte, info []byte, length int) ([]byte, error) {
	if length <= 0 {
		return nil, fmt.Errorf("bastion HKDF expand: invalid length %d", length)
	}
	if length > 255*kdfHashLen {
		return nil, fmt.Errorf("bastion HKDF expand: length %d exceeds max %d", length, 255*kdfHashLen)
	}

	result := make([]byte, 0, length)
	var prev [kdfHashLen]byte
	hasPrev := false

	// HMAC 实例复用一个：每轮 Reset 后重新吸收，省掉每轮构造底层 Hash 的分配
	h := NewHMAC(prk[:])
	for counter := byte(1); len(result) < length; counter++ {
		h.Reset()
		if hasPrev {
			h.Write(prev[:])
		}
		h.Write(info)
		h.Write([]byte{counter})
		prev = h.Sum()
		hasPrev = true
		result = append(result, prev[:]...)
	}

	return result[:length], nil
}

// HKDF 完整的 HKDF 派生
func HKDF(ikm, salt, info []byte, length int) ([]byte, error) {
	prk := HKDFExtract(ikm, salt)
	return HKDFExpand(prk, info, length)
}

// PBKDF2 密码基密钥派生，按 RFC 2898
func PBKDF2(password, salt []byte, iterations, keyLen int) ([]byte, error) {
	if iterations < 1 {
		return nil, fmt.Errorf("bastion PBKDF2: iterations must be >= 1, got %d", iterations)
	}
	if keyLen < 1 {
		return nil, fmt.Errorf("bastion PBKDF2: keyLen must be >= 1, got %d", keyLen)
	}
	if len(password) == 0 {
		return nil, fmt.Errorf("bastion PBKDF2: password must not be empty")
	}

	// 输出长度上限为 int 最大值减一块取整余量，保证下面的块数乘法不会溢出；
	// 32 位平台上超大 keyLen 会让 numBlocks*kdfHashLen 溢出成负数导致 make 直接 panic
	if keyLen > (1<<31)-32 {
		return nil, fmt.Errorf("bastion PBKDF2: keyLen too large")
	}

	numBlocks := (keyLen + kdfHashLen - 1) / kdfHashLen

	result := make([]byte, numBlocks*kdfHashLen)

	// 块数不足 4 时并行只有开销没有收益，直接串行
	if numBlocks < 4 {
		for blockIdx := 1; blockIdx <= numBlocks; blockIdx++ {
			acc := pbkdf2Block(password, salt, iterations, blockIdx)
			copy(result[(blockIdx-1)*kdfHashLen:], acc[:])
		}
		return result[:keyLen], nil
	}

	// RFC 2898 各输出块链完全独立，多核并行每个 worker 处理一段块区间
	workers := getOptimalWorkerCount(numBlocks * kdfHashLen)
	if workers > numBlocks {
		workers = numBlocks
	}
	var wg sync.WaitGroup
	blocksPerWorker := (numBlocks + workers - 1) / workers
	for w := 0; w < workers; w++ {
		start := w*blocksPerWorker + 1
		end := (w+1)*blocksPerWorker + 1
		if end > numBlocks+1 {
			end = numBlocks + 1
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for idx := s; idx < e; idx++ {
				acc := pbkdf2Block(password, salt, iterations, idx)
				copy(result[(idx-1)*kdfHashLen:], acc[:])
			}
		}(start, end)
	}
	wg.Wait()

	return result[:keyLen], nil
}

// pbkdf2Block 计算 PBKDF2 的第 blockIdx 个输出块（各块独立，可并行）
func pbkdf2Block(password, salt []byte, iterations, blockIdx int) [kdfHashLen]byte {
	h := NewHMAC(password)
	h.Write(salt)
	// PBKDF2 块索引按 RFC 2898 使用大端序编码
	var beBlock [4]byte
	binary.BigEndian.PutUint32(beBlock[:], uint32(blockIdx))
	h.Write(beBlock[:])
	uPrev := h.Sum()

	var accumulator [kdfHashLen]byte
	copy(accumulator[:], uPrev[:])

	for j := 2; j <= iterations; j++ {
		h.Reset()
		h.Write(uPrev[:])
		uCurr := h.Sum()

		for k := 0; k < kdfHashLen; k++ {
			accumulator[k] ^= uCurr[k]
		}
		uPrev = uCurr
	}
	return accumulator
}

// DeriveKey（BLAKE3风格密钥派生）

// DeriveKey BLAKE3 风格密钥派生，用 context+"\x00"+keyMaterial 的哈希作为密钥，CTR 扩展
func DeriveKey(context string, keyMaterial []byte, length int) []byte {
	if length <= 0 {
		return nil
	}

	// 构建 context + 0x00 + keyMaterial
	input := make([]byte, len(context)+1+len(keyMaterial))
	copy(input, context)
	input[len(context)] = 0x00
	copy(input[len(context)+1:], keyMaterial)

	// 计算基础哈希作为派生密钥的根密钥
	baseKey := Sum(input)

	if length <= hashSize {
		// 如果请求长度 <= 32字节，直接截取哈希
		return baseKey[:length]
	}

	// 需要扩展：用CTR模式对零块加密生成更多密钥材料
	// 使用零nonce（确定性扩展）
	nonce := make([]byte, NonceSize)
	result, err := CTREncrypt(baseKey[:], nonce, make([]byte, length))
	if err != nil {
		// 安全降级：返回零值
		return make([]byte, length)
	}
	return result
}
