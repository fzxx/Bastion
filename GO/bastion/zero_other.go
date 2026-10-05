//go:build !windows

// 非 Windows 平台的安全清零实现：Go 没有对应的平台清零 API，循环写 0 作为兜底
package bastion

import "runtime"

// zeroizeBytes 清零字节切片，清除临时密钥材料，编译器不会优化掉；
// runtime.KeepAlive 保证函数返回前底层数组不被提前回收
func zeroizeBytes(b []byte) {
	for i := range b {
		b[i] = 0
	}
	runtime.KeepAlive(b)
}
