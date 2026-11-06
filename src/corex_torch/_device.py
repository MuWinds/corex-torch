"""把 corex 注册成 torch 的设备，并把第一批 aten 算子接到设备层上。

分两步，缺一步都不行（`docs/00-FACTS.md` §12 有三级失败梯度的实测）：

1. **C++ shim**（`_corex_shim.so`，由 `tools/build_shim.py` 编出）：注册 ``c10::Allocator``
   与 ``DeviceGuardImplInterface``。**分配器只能从 C++ 注册**——Python 侧没有任何入口，
   而且没注册时 ``c10::GetAllocator(PrivateUse1)`` 只是返回 nullptr 不报错，后面才解引用空指针。
2. **Python 侧**（本模块）：``rename_privateuse1_backend``、device module、以及逐个 aten 算子的
   ``PrivateUse1`` 实现。算子用 ``torch.library.Library("aten","IMPL").impl(...)`` 注册，
   不需要 ``TORCH_LIBRARY_IMPL``，改一行就能重跑。

Module: ``corex_torch``。被 ``corex_torch.register()`` 调用；用户侧只写
``import corex_torch; corex_torch.register()``，之后才有 ``torch.device("corex")``。

Constraints:
    * 注册是**进程级一次性**动作：``rename_privateuse1_backend`` 改的是全局后端名，
      重复调用同名虽然无害，但 ``_register_ops`` 会重复注册同名实现并报错，所以用标志位挡住。
    * 没实现的算子一律显式抛 ``NotImplementedError`` 并带上算子名——静默走 CPU 会让人以为
      算在卡上，排查成本远高于报错。
    * 不提供任何 graph 能力（平台禁令，``docs/05-DRIVER-CONSTRAINTS.md`` R2）。
"""

from __future__ import annotations

import importlib
import types
from typing import Any, Optional, Sequence

from . import _runtime

#: 设备名。写到 ``torch.device(...)`` 里就是这个名字。
DEVICE_NAME = "corex"

#: C++ shim 的模块名（``tools/build_shim.py`` 的产物，落在包目录里）。
SHIM_MODULE = "corex_torch._corex_shim"

_state: dict[str, Any] = {"registered": False, "device": 0, "library": None, "shim": None}


def shim():
    """取已编译的 C++ shim 模块。

    Raises:
        RuntimeError: 还没编。消息里给出确切的构建命令——这个错误会在最早期出现，
            不写清楚的话，使用者会以为是 torch 的问题。
    """
    if _state["shim"] is None:
        # 必须先 import torch：shim 链接了 libc10/libtorch_cpu，torch 还没加载时
        # dlopen 会报 `libc10.so: cannot open shared object file`（2026-11-07 实测）。
        import torch  # noqa: F401
        try:
            _state["shim"] = importlib.import_module(SHIM_MODULE)
        except ImportError as exc:                      # noqa: F841
            raise RuntimeError(
                "corex 设备需要 C++ shim 先编出来：分配器只能从 C++ 注册（见 docs/00-FACTS.md §12）。\n"
                "  在仓库根目录执行： python tools/build_shim.py\n"
                f"  期望产物： src/corex_torch/_corex_shim.so（{SHIM_MODULE}）"
            ) from exc
    return _state["shim"]


def _strides_of(size: Sequence[int], memory_format: Optional[Any]) -> list[int]:
    """按行主序算 strides。

    Args:
        size: 张量形状。
        memory_format: torch 的 memory_format；``None`` 或 ``contiguous_format`` 走行主序。

    Returns:
        行主序 strides。

    Raises:
        NotImplementedError: 非连续布局（channels_last 等）本阶段不支持——宁可报错，
            也不要按行主序糊一个出来，那会让后面的逐元素内核算错位置。
    """
    import torch

    allowed = {None, torch.contiguous_format, torch.preserve_format}
    if memory_format not in allowed:
        raise NotImplementedError(
            f"corex 设备暂不支持 memory_format={memory_format}（只做行主序）"
        )
    strides: list[int] = []
    acc = 1
    for dim in reversed(list(size)):
        strides.append(acc)
        acc *= int(dim)
    return list(reversed(strides))


def _empty_impl(size, *args, dtype=None, layout=None, device=None, pin_memory=None,
                memory_format=None):
    """``aten::empty.memory_format`` 的 PrivateUse1 实现。"""
    import torch

    if layout not in (None, torch.strided):
        raise NotImplementedError(f"corex 设备只支持 strided 布局，收到 layout={layout}")
    dtype = dtype or torch.get_default_dtype()
    index = _resolve_device(device)
    return shim().empty_strided([int(x) for x in size], _strides_of(size, memory_format),
                                _scalar_type(dtype), index)


def _empty_strided_impl(size, stride, *, dtype=None, layout=None, device=None,
                        pin_memory=None):
    """``aten::empty_strided`` 的 PrivateUse1 实现。"""
    import torch

    dtype = dtype or torch.get_default_dtype()
    return shim().empty_strided([int(x) for x in size], [int(x) for x in stride],
                                _scalar_type(dtype), _resolve_device(device))


def _resolve_device(device: Optional[Any]) -> int:
    """从 torch.device 里取设备号；``None`` 用当前设备。"""
    if device is None:
        return _state["device"]
    index = getattr(device, "index", None)
    return _state["device"] if index is None else int(index)


#: ``c10::ScalarType`` 的枚举编号。为什么需要这张表：C++ shim 要的是 ``at::ScalarType``，
#: 而 ``torch.dtype`` **没有 ``__int__``**（实测 2.8 抛
#: ``TypeError: int() argument must be ... not 'torch.dtype'``），Python 侧也没有别的公开
#: 转换入口。c10 的枚举编号是 ABI 稳定的，但下面用 ``element_size`` 交叉校验一次，
#: 万一哪天对不上会当场报错，而不是把 fp32 当 fp16 解释。
_SCALAR_TYPE = {}

#: 期望的元素字节数，用于校验上面那张表。
_SCALAR_SIZE = {}


def _scalar_type(dtype) -> int:
    """把 ``torch.dtype`` 翻成 ``c10::ScalarType`` 的整数编号。

    Args:
        dtype: torch.dtype。

    Returns:
        ScalarType 编号（如 fp32 = 6）。

    Raises:
        NotImplementedError: 还没支持的 dtype。
        RuntimeError: 枚举编号与元素宽度对不上（说明映射表过期了）。
    """
    import torch

    if not _SCALAR_TYPE:
        _SCALAR_TYPE.update({
            torch.uint8: 0, torch.int8: 1, torch.int16: 2, torch.int32: 3,
            torch.int64: 4, torch.float16: 5, torch.float32: 6, torch.float64: 7,
            torch.bool: 11, torch.bfloat16: 15,
        })
        _SCALAR_SIZE.update({0: 1, 1: 1, 2: 2, 3: 4, 4: 8, 5: 2, 6: 4, 7: 8, 11: 1, 15: 2})
    code = _SCALAR_TYPE.get(dtype)
    if code is None:
        raise NotImplementedError(f"corex 设备还没覆盖 dtype={dtype}")
    actual = torch.empty(0, dtype=dtype).element_size()
    if actual != _SCALAR_SIZE[code]:
        raise RuntimeError(
            f"ScalarType 映射过期：{dtype} 期望 {_SCALAR_SIZE[code]} 字节，实际 {actual}"
        )
    return code


def _copy_from_impl(src, dst, non_blocking: bool = False) -> Any:
    """``aten::_copy_from``：H2D / D2H / D2D 三种方向。

    **参数顺序是 trap**：schema 是 ``_copy_from(Tensor self, Tensor dst, bool non_blocking)``，
    也就是第一个实参是**源**、第二个才是**目的**。一开始按「(dst, src)」命名写，结果
    ``t.cpu()`` 走进 H2D 分支、把主机缓冲当地址写，读回来是一堆垃圾且不报错
    （2026-11-07 在 probes/008 里抓到）。注意返回值是目的张量。

    为什么拷贝不写成 Triton 内核：D2D 走驱动 DMA 实测 595–649 GB/s（`probes/007`），
    而 Triton 逐元素内核只有 295 GB/s 量级，还白占 SM。主机侧的两条路必须经 pinned
    暂存区——pageable 只有 4.6 GB/s，pinned 是 27.7 GB/s（`docs/00-FACTS.md` §11）。
    """
    if dst.shape != src.shape:
        raise NotImplementedError("corex 的 _copy_from 暂不支持形状变化（走 resize 的那条路）")
    nbytes = int(dst.numel()) * int(dst.element_size())
    if nbytes == 0:
        return dst

    dst_dev = dst.device.type == DEVICE_NAME
    src_dev = src.device.type == DEVICE_NAME
    # **每条原始拷贝前都必须把当前设备绑到「指针所属的那张卡」**：这张卡的 cudaMemcpy
    # 在当前 runtime 设备与指针设备不一致时**不报错、rc=0、静默别名**——两张卡低地址相同的
    # 偏移会被当成同一块内存，数据悄悄写错地方，事后无从排查。反面实验见 probes/112 §1.4b。
    if dst_dev:
        _runtime.set_device(int(dst.device.index or 0))
    elif src_dev:
        _runtime.set_device(int(src.device.index or 0))
    if dst_dev and src_dev:
        _runtime.memcpy(dst.data_ptr(), src.data_ptr(), nbytes,
                        _runtime.MEMCPY_DEVICE_TO_DEVICE)
        return dst
    if dst_dev and not src_dev:
        _host_to_device(dst.data_ptr(), src.data_ptr(), nbytes, int(src.is_pinned()))
        return dst
    if src_dev and not dst_dev:
        _device_to_host(dst.data_ptr(), src.data_ptr(), nbytes, int(dst.is_pinned()))
        return dst
    raise NotImplementedError(
        f"corex 的 _copy_from 不处理 {src.device.type} -> {dst.device.type}"
    )


def _host_to_device(dev_ptr: int, host_ptr: int, nbytes: int, pinned: int) -> None:
    """H2D：非页锁定的主机内存统一经 pinned 暂存区中转。"""
    if pinned:
        _runtime.memcpy(dev_ptr, host_ptr, nbytes, _runtime.MEMCPY_HOST_TO_DEVICE)
        return
    import ctypes

    stage = _runtime.staging(nbytes)
    ctypes.memmove(stage, host_ptr, nbytes)             # 主机到主机，普通 memmove
    _runtime.memcpy(dev_ptr, stage, nbytes, _runtime.MEMCPY_HOST_TO_DEVICE)


def _device_to_host(host_ptr: int, dev_ptr: int, nbytes: int, pinned: int) -> None:
    """D2H：非页锁定的目标内存经 pinned 暂存区中转。"""
    if pinned:
        _runtime.memcpy(host_ptr, dev_ptr, nbytes, _runtime.MEMCPY_DEVICE_TO_HOST)
        return
    import ctypes

    stage = _runtime.staging(nbytes)
    _runtime.memcpy(stage, dev_ptr, nbytes, _runtime.MEMCPY_DEVICE_TO_HOST)
    ctypes.memmove(host_ptr, stage, nbytes)


def _zero_impl(tensor) -> Any:
    """``aten::zero_``：整块清零。走 ``cudaMemset`` 而不是内核。"""
    nbytes = int(tensor.numel()) * int(tensor.element_size())
    if nbytes:
        _runtime.set_device(int(tensor.device.index or 0))
        _runtime.memset(tensor.data_ptr(), 0, nbytes)
    return tensor


def _fill_scalar_impl(tensor, value, non_blocking: bool = False) -> Any:
    """``aten::fill_.Scalar``：填一个标量。0 走 ``cudaMemset``，其余走算子层内核。"""
    nbytes = int(tensor.numel()) * int(tensor.element_size())
    if not nbytes:
        return tensor
    if float(value) == 0.0:
        _runtime.memset(tensor.data_ptr(), 0, nbytes)
        return tensor
    from . import _ops_elementwise as ops

    ops.fill(tensor.data_ptr(), int(tensor.numel()), _dtype_name(tensor.dtype), float(value))
    return tensor


def _local_scalar_dense_impl(tensor) -> Any:
    """``aten::_local_scalar_dense``：``.item()`` 用，把一个标量搬回主机。"""
    nbytes = int(tensor.element_size())
    _runtime.set_device(int(tensor.device.index or 0))
    stage = _runtime.staging(nbytes)
    _runtime.memcpy(stage, tensor.data_ptr(), nbytes, _runtime.MEMCPY_DEVICE_TO_HOST)
    return _scalar_from_stage(stage, tensor.dtype)


def _scalar_from_stage(stage_ptr: int, dtype) -> Any:
    """从 pinned 暂存区按 dtype 解出一个 Python 标量。"""
    import ctypes
    import struct

    import torch

    raw = ctypes.string_at(stage_ptr, torch.empty((), dtype=dtype).element_size())
    fmt = {torch.float32: "<f", torch.float16: "<e", torch.float64: "<d",
           torch.int8: "<b", torch.uint8: "<B", torch.int16: "<h",
           torch.int32: "<i", torch.int64: "<q", torch.bool: "<?",
           torch.bfloat16: None}.get(dtype)
    if fmt is None:
        raise NotImplementedError(f"corex 的 .item() 还没覆盖 dtype={dtype}")
    return struct.unpack(fmt, raw)[0]


def _dtype_name(dtype) -> str:
    """把 torch.dtype 翻成 Triton 签名用的类型串。"""
    import torch

    table = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16",
             torch.int8: "i8", torch.int32: "i32", torch.int64: "i64"}
    if dtype not in table:
        raise NotImplementedError(f"corex 的算子层还没覆盖 dtype={dtype}")
    return table[dtype]


def _out_binary(name: str):
    """生成一个二元逐元素算子的 ``*.out`` 实现。

    为什么挂 ``.out`` 而不是 ``.Tensor``/``.Scalar``：两个 functional 重载在本版 torch 里是
    **composite**（会自己拆到 out 重载上），所以 ``a + b`` 与 ``t + 1`` 最终都落到
    ``aten::add.out``。实测依据来自 ``probes/111`` 的缺口清单：只挂 empty 时 ``t + 1`` 报的是
    ``aten::add.out`` 未实现，而不是 ``add.Tensor``。

    Args:
        name: 算子名（``add`` / ``sub`` / ``mul`` / ``div``）。

    Returns:
        ``(self, other, *, alpha=..., out=...)`` 形式的实现。形状不同或非连续一律显式
        报 ``NotImplementedError``——本阶段不做广播与 stride 处理，猜一个会静默算错。
    """

    def impl(self, other, *, alpha=1, out=None):
        import torch

        if out is None:
            raise NotImplementedError(f"corex 的 {name}.out 要求显式 out")
        if not isinstance(other, torch.Tensor):
            raise NotImplementedError(f"corex 的 {name}.out 在标量路径上还没接（composite 应已包装成 0 维张量）")
        shape = tuple(torch.broadcast_shapes(self.shape, other.shape))
        if tuple(out.shape) != shape:
            raise NotImplementedError(
                f"corex 的 {name}.out 暂不支持 resize：out.shape={tuple(out.shape)} 期望 {shape}"
            )
        for label, t in (("self", self), ("other", other), ("out", out)):
            if not t.is_contiguous():
                raise NotImplementedError(f"corex 的 {name}.out 暂不支持非连续 {label}")
            if t.device.type != DEVICE_NAME:
                raise NotImplementedError(f"corex 的 {name}.out 要求 {label} 在 corex 上，收到 {t.device}")
        from . import _ops_elementwise as ops

        getattr(ops, name)(self.data_ptr(), other.data_ptr(), out.data_ptr(),
                           int(out.numel()), _dtype_name(out.dtype), alpha=float(alpha))
        return out

    return impl


def _register_ops() -> None:
    """把第一批 aten 算子挂到 PrivateUse1 上。

    Returns:
        无。库对象存在 ``_state["library"]`` 里，保活——``torch.library.Library`` 一旦被回收，
        注册的实现会一起失效，症状是「刚注册完还能用，过一会儿又报未实现」。
    """
    import torch

    lib = torch.library.Library("aten", "IMPL")
    _state["library"] = lib
    lib.impl("empty.memory_format", _empty_impl, "PrivateUse1")
    lib.impl("empty_strided", _empty_strided_impl, "PrivateUse1")
    lib.impl("_copy_from", _copy_from_impl, "PrivateUse1")
    lib.impl("_copy_from_and_resize", _copy_from_impl, "PrivateUse1")
    lib.impl("zero_", _zero_impl, "PrivateUse1")
    lib.impl("fill_.Scalar", _fill_scalar_impl, "PrivateUse1")
    lib.impl("_local_scalar_dense", _local_scalar_dense_impl, "PrivateUse1")
    for op in ("add", "sub", "mul", "div"):
        lib.impl(f"{op}.out", _out_binary(op), "PrivateUse1")


def _make_device_module() -> types.ModuleType:
    """造出 ``torch.corex`` 这个模块。

    为什么必须有：torch 的 C++ lazy init 会 ``import torch.corex`` 取设备层函数，
    ``_register_device_module`` 只是把模块挂到 ``torch.<name>`` 并塞进 ``sys.modules``。

    Returns:
        一个 ``types.ModuleType``，属性与 ``torch.cuda`` 的常用子集对齐。
    """
    module = types.ModuleType(f"torch.{DEVICE_NAME}")

    def _lazy_init() -> None:
        """torch 在访问设备前会调它（``torch.accelerator.current_device_index`` 也会）。

        我们在这里保证 CoreX 运行时可用 + shim 已初始化。幂等。
        """
        register()

    def is_available() -> bool:
        return _runtime.is_available()

    def device_count() -> int:
        return shim().device_count() if _state["registered"] else 0

    def current_device() -> int:
        return _state["device"]

    def set_device(index: int) -> None:
        _state["device"] = int(index)
        _runtime.set_device(int(index))

    def synchronize(device: Optional[int] = None) -> None:
        _runtime.synchronize()

    def empty_cache() -> None:
        shim().empty_cache()

    def memory_allocated(device: Optional[int] = None) -> int:
        return int(shim().allocated_bytes())

    def max_memory_allocated(device: Optional[int] = None) -> int:
        return int(shim().reserved_bytes())

    def memory_reserved(device: Optional[int] = None) -> int:
        return int(shim().reserved_bytes())

    def current_stream(device: Optional[int] = None):
        return _runtime.Stream.default()

    for fn in (_lazy_init, is_available, device_count, current_device, set_device,
               synchronize, empty_cache, memory_allocated, max_memory_allocated,
               memory_reserved, current_stream):
        setattr(module, fn.__name__, fn)
    return module


def register(device_index: int = 0) -> None:
    """把 corex 一次性注册进 torch。幂等。

    Args:
        device_index: 初始设备号。

    Raises:
        RuntimeError: torch 不是 2.8（本仓库只按 2.8 的 PrivateUse1 接口写），
            或 C++ shim 还没编出来。
    """
    if _state["registered"]:
        return
    import torch

    major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if (major, minor) != (2, 8):
        raise RuntimeError(f"corex 后端只针对 torch 2.8 写，当前是 {torch.__version__}")

    _state["device"] = int(device_index)
    # 顺序不能反：rename 必须早于任何 torch.device("corex") 的构造，
    # 而 device module 必须早于 C++ 的 lazy init（它会 import torch.corex）。
    torch.utils.rename_privateuse1_backend(DEVICE_NAME)
    module = _make_device_module()
    torch._register_device_module(DEVICE_NAME, module)
    shim().init(int(device_index))
    _register_ops()
    _state["registered"] = True
