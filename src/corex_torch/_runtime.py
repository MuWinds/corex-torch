"""CoreX 运行时（libcudart.so.10.2）的 ctypes 绑定：设备、显存、拷贝、流、事件。

为什么直连 runtime API 而不是 driver API：runtime 会自己延迟建 primary context，
省掉 ``cuInit`` / ``cuCtxCreate`` 这一步——后者在本平台坏卡上挂死过 15 分钟
（``corex-triton/docs/00-FACTS.md`` §5），而且 ``probes/002_cudart_binding.sh`` 已经实测：
只用 ctypes 打开 ``libcudart.so.10.2``、不 import 任何 torch，分配 / H2D-D2H 往返 / 流 / 事件
全部 rc=0 且往返逐字节相等。设备层没有必要绕道 driver API。

Module: ``corex_torch``。这是本仓库最底层：``device`` / ``stream`` / 分配器都从这里取句柄，
它自己不 import torch——设备层要能在一台只有 CoreX 运行时、没有 torch 的机器上被单独验证。

Constraints:
    * 所有函数在首次真正调用时才 dlopen libcudart：本模块被 import 时不允许碰设备，
      否则在登录节点（没有 /usr/local/corex）import 就炸。
    * 每个返回值都过 ``_check``：CoreX 的 runtime 会用 0 表示成功，但个别接口在未实现时
      返回 0 却不改出参，所以关键结果（指针、句柄、计数器）另外断言一次。
    * 句柄（stream / event / 显存指针）一律用 ``c_void_p`` 承载，不用 int 转换：
      设备指针在本平台是 0x304000002000000 这种 60 位值，转 int 再转回来容易丢位。
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional

#: CoreX 安装根目录。与 ``corex-triton/src/triton_corex/driver.py`` 保持同一个环境变量名。
COREX_HOME = os.environ.get("COREX_HOME", "/usr/local/corex")

#: 运行时库文件名。10.2 是兼容层自报的 CUDA 版本（``CUDART_VERSION 10020``），
#: 直接写带版本号的名字，避免某天出现 ``libcudart.so`` 指向别处时静默换库。
CUDART_SONAME = "libcudart.so.10.2"

#: ``cudaMemcpyKind`` 的取值（CUDA 10.2 的定义）。
MEMCPY_HOST_TO_DEVICE = 1
MEMCPY_DEVICE_TO_HOST = 2
MEMCPY_DEVICE_TO_DEVICE = 3

#: 设备属性号。只收我们真正用到的那些，全部是 CUDA 里长期稳定的编号。
ATTR_MAX_THREADS_PER_BLOCK = 1
ATTR_SHARED_MEM_PER_BLOCK = 8
ATTR_WARP_SIZE = 10
ATTR_MULTIPROCESSOR_COUNT = 16


class CorexRuntimeError(RuntimeError):
    """runtime API 返回非 0 时抛出。

    Attributes:
        name: 出错的函数名，报错信息里要能一眼看出是哪一步断的。
        code: CUDA 错误码（``cudaGetErrorString`` 认得它）。
    """

    def __init__(self, name: str, code: int, detail: str) -> None:
        super().__init__(f"{name} 失败 rc={code} {detail}".rstrip())
        self.name = name
        self.code = code


_cudart: Optional[ctypes.CDLL] = None


def _lib() -> ctypes.CDLL:
    """惰性打开并配好 argtypes 的 libcudart。首次调用可能触发 context 创建。"""
    global _cudart
    if _cudart is not None:
        return _cudart

    candidates = [os.path.join(COREX_HOME, "lib64", CUDART_SONAME),
                  os.path.join(COREX_HOME, "lib", CUDART_SONAME)]
    path = next((p for p in candidates if os.path.exists(p)), CUDART_SONAME)
    lib = ctypes.CDLL(path)

    # argtypes/restype 写全：ctypes 默认把指针当 int 处理，
    # 在 60 位设备地址上会静默截断（设备指针是 0x304000002000000）。
    lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.cudaSetDevice.argtypes = [ctypes.c_int]
    lib.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.cudaDeviceSynchronize.argtypes = []
    lib.cudaDeviceReset.argtypes = []
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    lib.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t),
                                   ctypes.POINTER(ctypes.c_size_t)]
    lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.cudaStreamQuery.argtypes = [ctypes.c_void_p]
    lib.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
    lib.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float),
                                         ctypes.c_void_p, ctypes.c_void_p]
    lib.cudaDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int),
                                           ctypes.c_int, ctypes.c_int]
    lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                                  ctypes.c_uint]
    lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
    lib.cudaGetErrorString.argtypes = [ctypes.c_int]
    lib.cudaGetErrorString.restype = ctypes.c_char_p
    lib.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]

    # cudaGetDeviceProperties 只读前 256 字节的设备名，所以不声明完整结构体，
    # 用一个 raw buffer 接住——厂商的 cudaDeviceProp 布局没有文档，按 CUDA 的字段表去对齐
    # 迟早错位，而我们真正需要的属性走 cudaDeviceGetAttribute（按编号取，不受布局影响）。
    lib.cudaGetDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.c_int]

    for fn in ("cudaGetDeviceCount", "cudaSetDevice", "cudaGetDevice",
               "cudaDeviceSynchronize", "cudaDeviceReset", "cudaMalloc", "cudaFree",
               "cudaMemcpy", "cudaMemcpyAsync", "cudaMemGetInfo",
               "cudaStreamCreate", "cudaStreamDestroy", "cudaStreamSynchronize",
               "cudaStreamQuery", "cudaEventCreate", "cudaEventDestroy",
               "cudaEventRecord", "cudaEventSynchronize", "cudaEventElapsedTime",
               "cudaDeviceGetAttribute", "cudaGetErrorString", "cudaHostAlloc",
               "cudaFreeHost", "cudaRuntimeGetVersion"):
        getattr(lib, fn).restype = ctypes.c_int
    _cudart = lib
    return lib


def _check(name: str, code: int) -> int:
    """把非 0 返回码翻成带错误串的异常。"""
    if code != 0:
        raise CorexRuntimeError(name, code, _error_string(code))
    return code


def _error_string(code: int) -> str:
    try:
        raw = _lib().cudaGetErrorString(code)
        return raw.decode(errors="replace") if raw else ""
    except Exception:                                   # noqa: BLE001 - 报错路径不能再炸一次
        return ""


def is_available() -> bool:
    """CoreX 运行时在不在、设备数是否大于 0。不抛异常。"""
    try:
        return device_count() > 0
    except Exception:                                   # noqa: BLE001
        return False


def runtime_version() -> int:
    """runtime 自报的 CUDA 版本号（本平台实测 10020）。"""
    ver = ctypes.c_int(0)
    _check("cudaRuntimeGetVersion", _lib().cudaRuntimeGetVersion(ctypes.byref(ver)))
    return ver.value


def device_count() -> int:
    """可用设备数。"""
    n = ctypes.c_int(0)
    _check("cudaGetDeviceCount", _lib().cudaGetDeviceCount(ctypes.byref(n)))
    return n.value


def current_device() -> int:
    """当前线程绑定的设备号。"""
    dev = ctypes.c_int(-1)
    _check("cudaGetDevice", _lib().cudaGetDevice(ctypes.byref(dev)))
    return dev.value


def set_device(index: int) -> None:
    """把当前线程绑到 ``index`` 号设备（顺带保证它的 primary context 就位）。"""
    _check("cudaSetDevice", _lib().cudaSetDevice(int(index)))


def synchronize() -> None:
    """等当前设备上所有流的工作完成。"""
    _check("cudaDeviceSynchronize", _lib().cudaDeviceSynchronize())


def reset() -> None:
    """销毁当前设备的 primary context（仅用于进程退出/测试收尾）。"""
    _check("cudaDeviceReset", _lib().cudaDeviceReset())


def device_attribute(attr: int, index: int = 0) -> int:
    """读一条设备属性。

    Args:
        attr: ``cudaDeviceAttr`` 编号，见本模块 ``ATTR_*`` 常量。
        index: 设备号。

    Returns:
        属性值。个别属性在本平台恒为 0（未实现），调用方不要把它当「关闭」的证据。
    """
    val = ctypes.c_int(0)
    _check("cudaDeviceGetAttribute",
           _lib().cudaDeviceGetAttribute(ctypes.byref(val), int(attr), int(index)))
    return val.value


def device_name(index: int = 0) -> str:
    """设备名。走 ``cudaGetDeviceProperties`` 的前 256 字节，避免结构体布局猜错。"""
    class _Props(ctypes.Structure):
        _fields_ = [("name", ctypes.c_char * 256), ("_rest", ctypes.c_byte * 4096)]

    props = _Props()
    _check("cudaGetDeviceProperties",
           _lib().cudaGetDeviceProperties(ctypes.cast(ctypes.byref(props), ctypes.c_void_p),
                                          int(index)))
    return props.name.decode(errors="replace")


def memory_info(index: int = 0) -> tuple[int, int]:
    """``(free, total)`` 字节数。"""
    if index != current_device():
        set_device(index)
    free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
    _check("cudaMemGetInfo", _lib().cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)))
    return free.value, total.value


def malloc(nbytes: int) -> ctypes.c_void_p:
    """在设备上分配 ``nbytes`` 字节，返回句柄。

    分配失败会抛 ``CorexRuntimeError``（最常见是 out of memory）。调用方负责 ``free``。
    """
    ptr = ctypes.c_void_p()
    _check("cudaMalloc", _lib().cudaMalloc(ctypes.byref(ptr), int(nbytes)))
    if not ptr.value:
        raise CorexRuntimeError("cudaMalloc", 0, "返回成功但指针为空")
    return ptr


def free(ptr: ctypes.c_void_p) -> None:
    """释放 ``malloc`` 得到的设备内存。"""
    if ptr and ptr.value:
        _check("cudaFree", _lib().cudaFree(ptr))


def _ptr(value) -> "ctypes.c_void_p":
    """把「设备地址」统一成 ``c_void_p``。

    为什么要这个 Helper：``ctypes.c_void_p(existing_c_void_p)`` 在 Python 3.10 与 3.12 都抛
    ``TypeError: cannot be converted to pointer``——它只接受整数或可 cast 的对象。而我们的
    指针有的来自 ``cudaMalloc``（已是 ``c_void_p``），有的是调用方 ``ctypes.cast`` 出来的，
    还有的是 ``data_ptr()`` 给的 int，三种都会出现。

    Args:
        value: ``c_void_p`` / int / 任何 ctypes 对象（buffer、array、结构体）。

    Returns:
        指向同一地址的 ``c_void_p``。
    """
    if isinstance(value, ctypes.c_void_p):
        return value
    if isinstance(value, int):
        return ctypes.c_void_p(value)
    return ctypes.cast(value, ctypes.c_void_p)


def memcpy(void_ptr, other_ptr, nbytes: int, kind: int,
           stream: Optional["ctypes.c_void_p"] = None) -> None:
    """同步/异步拷贝。``stream`` 为 None 时走同步版本。

    Args:
        void_ptr: 目的地址（设备或主机指针）。
        other_ptr: 源地址。
        nbytes: 字节数。
        kind: ``MEMCPY_*`` 之一。
        stream: 设备流；给 None 走同步 ``cudaMemcpy``。

    Raises:
        CorexRuntimeError: 拷贝返回非 0。
    """
    dst, src = _ptr(void_ptr), _ptr(other_ptr)
    if stream is None:
        _check("cudaMemcpy", _lib().cudaMemcpy(dst, src, int(nbytes), int(kind)))
        return
    _check("cudaMemcpyAsync", _lib().cudaMemcpyAsync(dst, src, int(nbytes), int(kind),
                                                     stream))


def malloc_host(nbytes: int) -> "ctypes.c_void_p":
    """分配页锁定（pinned）主机内存。

    为什么值得单独有：驱动审计（``docs/qwen4/COREX-DRIVER-FEATURES.md`` §B2）证实
    host 路径的 ``cudaMemcpy`` **每次都要 ``vmalloc`` + ``get_user_pages_fast``**，
    那是毫秒级操作。小张量的 H2D/D2H 必须走固定的 pinned 暂存区，否则一次拷贝就可能
    罚掉 1 ms 量级。D2D 不受影响（走 ``mem_handle``，不 pin）。
    """
    ptr = ctypes.c_void_p()
    _check("cudaHostAlloc", _lib().cudaHostAlloc(ctypes.byref(ptr), int(nbytes), 0))
    if not ptr.value:
        raise CorexRuntimeError("cudaHostAlloc", 0, "返回成功但指针为空")
    return ptr


def free_host(ptr) -> None:
    """释放 ``malloc_host`` 的 pinned 内存。"""
    handle = _ptr(ptr)
    if handle and handle.value:
        _check("cudaFreeHost", _lib().cudaFreeHost(handle))


class Stream:
    """一条设备流。``handle`` 直接喂给 ``cuLaunchKernel`` / ``cudaMemcpyAsync``。

    Constraints:
        * 默认流（``cudaStreamDefault``）用 ``Stream.default()`` 表示，不要 new 一条再当默认流用——
          两者的同步语义不同，混用会得到「看起来同步了其实没有」的竞态。
    """

    def __init__(self, handle: ctypes.c_void_p) -> None:
        self.handle = handle
        self._owned = True

    @classmethod
    def create(cls) -> "Stream":
        """建一条新流。"""
        handle = ctypes.c_void_p()
        _check("cudaStreamCreate", _lib().cudaStreamCreate(ctypes.byref(handle)))
        return cls(handle)

    @classmethod
    def default(cls) -> "Stream":
        """默认流（句柄为 0，不持所有权）。"""
        stream = cls(ctypes.c_void_p(None))
        stream._owned = False
        return stream

    def synchronize(self) -> None:
        """等这条流上的工作完成。"""
        _check("cudaStreamSynchronize", _lib().cudaStreamSynchronize(self.handle))

    def query(self) -> bool:
        """流是否已经空。返回 False 表示还有工作没做完（对应 ``cudaErrorNotReady``）。"""
        code = _lib().cudaStreamQuery(self.handle)
        if code == 0:
            return True
        if code == 600:                                 # cudaErrorNotReady
            return False
        _check("cudaStreamQuery", code)
        return False

    def close(self) -> None:
        """销毁这条流（默认流不销毁）。"""
        if self._owned and self.handle and self.handle.value:
            _check("cudaStreamDestroy", _lib().cudaStreamDestroy(self.handle))
            self._owned = False

    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Event:
    """一条计时事件，用来给算子/整段前向计时，也是「流间同步」的唯一手段。

    本平台禁用 CUDA graph，所以计时与同步全部落在「记录事件 + 等事件」上。
    """

    def __init__(self) -> None:
        self.handle = ctypes.c_void_p()
        _check("cudaEventCreate", _lib().cudaEventCreate(ctypes.byref(self.handle)))

    def record(self, stream: Optional[Stream] = None) -> None:
        """在 ``stream`` 上打一个点（``None`` = 默认流）。"""
        _check("cudaEventRecord",
               _lib().cudaEventRecord(self.handle, (stream or Stream.default()).handle))

    def synchronize(self) -> None:
        """等这个点被设备执行到。"""
        _check("cudaEventSynchronize", _lib().cudaEventSynchronize(self.handle))

    def elapsed_ms(self, other: "Event") -> float:
        """``self`` 与 ``other`` 之间经过的毫秒数。两个事件都要已经 ``synchronize``。"""
        ms = ctypes.c_float(0.0)
        _check("cudaEventElapsedTime",
               _lib().cudaEventElapsedTime(ctypes.byref(ms), self.handle, other.handle))
        return ms.value

    def close(self) -> None:
        """销毁事件。"""
        if self.handle and self.handle.value:
            _check("cudaEventDestroy", _lib().cudaEventDestroy(self.handle))
            self.handle = ctypes.c_void_p()

    def __enter__(self) -> "Event":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def properties(index: int = 0) -> dict:
    """把设备后端需要的属性收成一个字典。

    Returns:
        含 ``name`` / ``warp_size`` / ``max_threads_per_block`` / ``shared_mem_per_block``
        / ``multiprocessor_count`` / ``total_memory``。``shared_mem_per_block`` 实测 131072，
        但能不能真申请到还要看启动时的返回码（见 ``docs/00-FACTS.md`` §8 的 UNKNOWN）。
    """
    if index != current_device():
        set_device(index)
    free, total = memory_info(index)
    return {
        "name": device_name(index),
        "warp_size": device_attribute(ATTR_WARP_SIZE, index),
        "max_threads_per_block": device_attribute(ATTR_MAX_THREADS_PER_BLOCK, index),
        "shared_mem_per_block": device_attribute(ATTR_SHARED_MEM_PER_BLOCK, index),
        "multiprocessor_count": device_attribute(ATTR_MULTIPROCESSOR_COUNT, index),
        "total_memory": total,
        "free_memory": free,
    }
