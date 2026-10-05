//go:build windows

// Windows 平台的安全清零实现：调用 ntdll 导出的 RtlZeroMemory，外部调用不会被编译器优化掉
// SecureZeroMemory / RtlSecureZeroMemory 只是 SDK 头文件里的内联实现，不是 DLL 导出符号，无法 syscall
package bastion

import (
	"runtime"
	"syscall"
	"unsafe"
)

// procRtlZeroMemory 对应 ntdll.dll 的 RtlZeroMemory
var procRtlZeroMemory = syscall.NewLazyDLL("ntdll.dll").NewProc("RtlZeroMemory")

// zeroizeBytes 清零字节切片，清除临时密钥材料：走系统例程，加载失败时退回循环写 0
func zeroizeBytes(b []byte) {
	if len(b) == 0 {
		return
	}
	if err := procRtlZeroMemory.Find(); err == nil {
		procRtlZeroMemory.Call(uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)))
		runtime.KeepAlive(b) // 地址以 uintptr 传入，调用期间保活底层数组
		return
	}
	for i := range b {
		b[i] = 0
	}
}
