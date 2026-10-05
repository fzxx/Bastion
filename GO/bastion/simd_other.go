//go:build !amd64

// 非 amd64 平台没有汇编 SIMD 实现，批量加密全部走软件路径
package bastion

// encryptBlocksDispatch 非 amd64 恒返回 false，由调用者走软件批量加密
func encryptBlocksDispatch(c *BlockCipher, dst, src, counter []byte) bool {
	return false
}

// encryptFromInputDispatch 非 amd64 恒返回 false
func encryptFromInputDispatch(c *BlockCipher, dst, src []byte) bool {
	return false
}

// decryptFromInputDispatch 非 amd64 恒返回 false
func decryptFromInputDispatch(c *BlockCipher, dst, src []byte) bool {
	return false
}
