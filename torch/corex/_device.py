"""把 corex 注册成 torch 的设备，并把第一批 aten 算子接到设备层上。

分两步，缺一步都不行（`docs/00-FACTS.md` §12 有三级失败梯度的实测）：

1. **C++ shim**（`_corex_shim.so`，由 `tools/build_shim.py` 编出）：注册 ``c10::Allocator``
   与 ``DeviceGuardImplInterface``。**分配器只能从 C++ 注册**——Python 侧没有任何入口，
   而且没注册时 ``c10::GetAllocator(PrivateUse1)`` 只是返回 nullptr 不报错，后面才解引用空指针。
2. **Python 侧**（本模块）：``rename_privateuse1_backend``、device module、以及逐个 aten 算子的
   ``PrivateUse1`` 实现。算子用 ``torch.library.Library("aten","IMPL").impl(...)`` 注册，
   不需要 ``TORCH_LIBRARY_IMPL``，改一行就能重跑。

Module: ``torch.corex``。被 ``torch.corex.register()`` 调用；用户侧只写
``import torch; torch.corex.register()``，之后才有 ``torch.device("corex")``。

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
SHIM_MODULE = "torch.corex._corex_shim"

_state: dict[str, Any] = {"registered": False, "device": 0, "library": None, "shim": None}

#: ``manual_seed`` 的警告只打一次（否则每个调用点都会刷一行，日志被淹）。
_SEED_WARNED = False


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
                f"  期望产物： torch/corex/_corex_shim.so（{SHIM_MODULE}）"
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

#: 已经校验过的 dtype（校验只做一次，见 :func:`_scalar_type`）。
_SCALAR_VALIDATED: set = set()


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
    # 校验只做一次：``torch.empty(0, dtype=...)`` 是一次真实的 CPU 分配 + 分发，
    # 每次算子调用都做一遍会白吃掉几微秒，而 dtype→字节数是常量。
    if dtype not in _SCALAR_VALIDATED:
        actual = torch.empty(0, dtype=dtype).element_size()
        if actual != _SCALAR_SIZE[code]:
            raise RuntimeError(
                f"ScalarType 映射过期：{dtype} 期望 {_SCALAR_SIZE[code]} 字节，实际 {actual}"
            )
        _SCALAR_VALIDATED.add(dtype)
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
    dst_dev = dst.device.type == DEVICE_NAME
    src_dev = src.device.type == DEVICE_NAME

    # 非连续怎么处理（线性 memcpy 只对「两侧都连续」成立；按字节扫转置视图不报错但数值错）：
    #   1) 两侧都在卡上 → 交给 `_aten_extra._copy_impl` 的按 stride 分段搬（它已经实现过一遍，
    #      再写一份只会两边不一致）。`clone()` 在本版本走 `at::_copy_from` 而不是 `copy_`，
    #      所以这条路必须在这里接住，否则 `clone`/`contiguous` 直接报错；
    #   2) 卡 → 主机且卡侧非连续 → 先搬进一块连续暂存，再线性 D2H（同样用 1 的分段搬）；
    #   3) 主机 → 卡且**主机侧非连续** → 主机上 `.contiguous()` 是原生实现，直接物化；
    #   4) 主机 → 卡且**目的非连续** → 先线性 H2D 到连续暂存，再按 stride 写进目的。
    #      （这条原来是直接报错，`probes/008` 因此一直 PARTIAL 9/10——而 `from_blob`
    #      造出来的转置视图正是这种形态，属于设备层该支持的正常用法。）
    if not (src.is_contiguous() and dst.is_contiguous()):
        if dst_dev and src_dev:
            from . import _aten_extra

            return _aten_extra._copy_impl(dst, src)
        if src_dev and not dst_dev and not src.is_contiguous():
            # 卡 → 主机且卡侧非连续：`.contiguous()` 会走「设备→设备 + 非连续源」那条路
            # （`_aten_extra._copy_impl`，`probes/020` 验过逐字节正确），比我们在这里
            # 再造一份分段搬稳。**不要**自己分配暂存再直接调 `_copy_impl`——
            # 那条路上 `_copy_impl` 会把主机张量当设备张量（`probes/027` 的报错）。
            src = src.contiguous()
        elif not src_dev and not src.is_contiguous():
            src = src.contiguous()
        if not dst.is_contiguous():
            from . import _aten_extra

            if not src_dev:
                # H2D + 非连续目的：先线性搬到连续暂存，再按 stride 写进去。
                stage = _empty_impl(list(dst.shape), dtype=dst.dtype, device=dst.device)
                _runtime.set_device(int(dst.device.index or 0))
                nbytes_stage = int(stage.numel()) * int(stage.element_size())
                _host_to_device(stage.data_ptr(), src.data_ptr(), nbytes_stage,
                                int(src.is_pinned()))
                _aten_extra._copy_impl(dst, stage)
                return dst
            if not dst_dev:
                # 非连续**主机**目的：`torch.Tensor.cpu()` 会按源的 stride 分配 CPU 张量
                # （`probes/027` 实测 dst.stride()=(1,2)），所以这条必须支持，否则
                # 「非连续的卡上视图读回主机」直接失败（`probes/008` 的非连续 strides 用例）。
                # 做法：线性 D2H 到连续暂存，再用 CPU 原生的 copy_ 写进带 stride 的目的。
                import torch as _torch

                src_c = src.contiguous()
                stage = _torch.empty(list(dst.shape), dtype=dst.dtype)
                _runtime.set_device(int(src.device.index or 0))
                _device_to_host(stage.data_ptr(), src_c.data_ptr(),
                                int(stage.numel()) * int(stage.element_size()),
                                int(stage.is_pinned()))
                dst.copy_(stage)                 # CPU→CPU，原生实现，尊重 stride
                return dst
            return _aten_extra._copy_impl(dst, src)

    nbytes = int(dst.numel()) * int(dst.element_size())
    if nbytes == 0:
        return dst

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

    nbytes = torch.empty((), dtype=dtype).element_size()
    raw = ctypes.string_at(stage_ptr, nbytes)
    if dtype == torch.bfloat16:
        # bf16 就是 fp32 的高 16 位：补 16 个 0 低位再按 f32 解释，不需要任何 bf16 转换指令。
        return struct.unpack("<f", raw + b"\x00\x00")[0]
    fmt = {torch.float32: "<f", torch.float16: "<e", torch.float64: "<d",
           torch.int8: "<b", torch.uint8: "<B", torch.int16: "<h",
           torch.int32: "<i", torch.int64: "<q", torch.bool: "<?",
           torch.float8_e4m3fn: None}.get(dtype)
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


def _scalar_out(name: str, self, other, alpha, out) -> Any:
    """``self <op> 标量`` 那条分支（``t / 2``、``t + 1`` 这类）。

    为什么要单独处理：``div.out`` 的 ``other`` 可以是 Python 数值，而 composite 不会替我们
    包成 0 维张量——实测 ``scores / math.sqrt(H)`` 直接带着 float 进来（`probes/009` 的
    注意力小块就是撞在这里）。标量走内核的 ``*_scalar`` 入口，比造一个 0 维张量再走广播省一次分配。

    ``alpha`` 的语义：``add``/``sub`` 的 schema 是 ``self ± alpha*other``，把 ``alpha*other``
    先算成一个标量再传下去，语义与 aten 一致。
    """
    if tuple(out.shape) != tuple(self.shape):
        raise NotImplementedError(
            f"corex 的 {name}.out(标量) 暂不支持 resize：out.shape={tuple(out.shape)} "
            f"期望 {tuple(self.shape)}")
    for label, t in (("self", self), ("out", out)):
        if t.device.type != DEVICE_NAME:
            raise NotImplementedError(f"corex 的 {name}.out 要求 {label} 在 corex 上，收到 {t.device}")
        if not t.is_contiguous():
            raise NotImplementedError(f"corex 的 {name}.out 暂不支持非连续 {label}")
    from . import _ops_elementwise as ops

    value = float(other) * float(alpha) if name in ("add", "sub") else float(other)
    getattr(ops, f"{name}_scalar")(self.data_ptr(), out.data_ptr(), int(out.numel()),
                                   _dtype_name(out.dtype), value)
    return out


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
            return _scalar_out(name, self, other, alpha, out)
        # **快路径**：三者同形、都连续、同设备——decode/训练里绝大多数调用都在这里。
        # 为什么值得单独写：`torch.broadcast_shapes` 在 `probes/024` 的 profile 里每步被调 28 次、
        # 自身耗时在 Python 侧排进前五；同形时它是恒等函数，白算一次元组还建了个 dict。
        if (self.shape == other.shape and self.shape == out.shape
                and self.is_contiguous() and other.is_contiguous() and out.is_contiguous()
                and other.device == self.device and out.device == self.device):
            return _out_binary_into(name, self, other, out, alpha)
        shape = tuple(torch.broadcast_shapes(self.shape, other.shape))
        if tuple(out.shape) != shape:
            raise NotImplementedError(
                f"corex 的 {name}.out 暂不支持 resize：out.shape={tuple(out.shape)} 期望 {shape}"
            )
        for label, t in (("self", self), ("other", other), ("out", out)):
            if t.device.type != DEVICE_NAME:
                raise NotImplementedError(f"corex 的 {name}.out 要求 {label} 在 corex 上，收到 {t.device}")
        # 非连续/需广播：内核只认「三个同形连续指针」，所以**先物化成 out 的形状**再算。
        # 语义与 torch 一致（结果相同），代价是一次拷贝；之前直接报错，但真实前向里到处是
        # 切片视图（`x[..., 0::2]`）与广播乘子（`cos[:, None, :]`），报错等于把「能算但慢」
        # 变成「不能算」（`probes/018` 的 mini decoder 撞在 `mul.out` 上）。
        #
        # **广播必须显式 expand**：只做 `.contiguous()` 会保留 [T,1,HD] 这种形状，而内核按
        # out.numel() 线性遍历，于是把更小的 other 当同形读——**不报错、数值全错**
        # （`probes/019` 的 5a 阶段：RoPE 的 `x1 * cos[:, None, :]` 误差 6.4）。
        if not self.is_contiguous() or tuple(self.shape) != shape:
            self = self.expand(shape).contiguous()
        if not other.is_contiguous() or tuple(other.shape) != shape:
            other = other.expand(shape).contiguous()
        if not out.is_contiguous():
            stage = _empty_impl(list(out.shape), dtype=out.dtype, device=out.device)
            _out_binary_into(name, self, other, stage, alpha)
            out.copy_(stage)                 # copy_ 走按 stride 的分段搬，见 _aten_extra
            return out
        return _out_binary_into(name, self, other, out, alpha)

    return impl


def _out_binary_into(name: str, self, other, out, alpha) -> Any:
    """把 ``self <op> other`` 写进连续的 ``out``（三个张量都必须已连续、同形）。"""
    from . import _ops_elementwise as ops

    # ``alpha`` 只有 add/sub 有（``out = self ± alpha*other``）；mul/div 的 schema 里没有，
    # 无条件传会得到 ``TypeError: mul() got an unexpected keyword argument 'alpha'``。
    fn = getattr(ops, name)
    if name in ("add", "sub"):
        fn(self.data_ptr(), other.data_ptr(), out.data_ptr(),
           int(out.numel()), _dtype_name(out.dtype), alpha=float(alpha))
    else:
        fn(self.data_ptr(), other.data_ptr(), out.data_ptr(),
           int(out.numel()), _dtype_name(out.dtype))
    return out


def _cpp_owned_schemas() -> set:
    """查 C++ 侧已经接管的 aten schema（目前是逐元素算子那一批）。

    为什么要问：**torch 的注册是「后注册者赢」**，我们在 Python 侧再注册一次就会盖掉 C++ 的
    快路径。而症状是「一切都对，只是慢」——`probes/125` 里 `.out` 一直停在 42–78 µs 就是这个原因，
    查了一轮才定位到。

    Returns:
        schema 名字集合；C++ 侧没有这个接口（旧 shim）时返回空集合。
    """
    try:
        shim = _state.get("shim")
        if shim is None or not hasattr(shim, "elementwise_op_names"):
            return set()
        return {str(name) for name in shim.elementwise_op_names()}
    except Exception:                                   # noqa: BLE001
        return set()


def _cpp_supports_broadcast() -> bool:
    """C++ 的逐元素快路径是否支持广播/0 维标量实参。

    为什么要单独问：C++ 只要「同形、连续、同 dtype」，而模型里大量是广播
    （`x1 * cos[:, None, :]`）与 0 维标量（`view.add_(2.5)`）。能力不够就让位，
    否则会把「能算但慢」变成「不能算」——`probes/023` 里 `x1 * 2.0` 抛「只支持同形」就是这么来的。

    Returns:
        真表示 C++ 已能处理广播与 0 维实参，Python 侧可以完全不注册这些 `.out`。
    """
    try:
        shim = _state.get("shim")
        if shim is None or not hasattr(shim, "elementwise_supports_broadcast"):
            return False
        return bool(shim.elementwise_supports_broadcast())
    except Exception:                                   # noqa: BLE001
        return False


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
    # 逐元素二元算子的 `.out`：**C++ 已经接管的就不要重复注册**。
    # 重复注册的后果不是「覆盖成我们的」，而是「我们的盖住了 C++ 的」——`probes/125` 里那条
    # `Overriding ... aten::add.out ... previous kernel: RegisterCPU_0.cpp` 就是它，
    # 于是 `.out` 一直停在 42–78 µs，而 C++ 的 inplace 路径只要 **1.96 µs**（差 20–40 倍）。
    # 这条和 mm 那条是同一个坑：**后注册者赢**，所以「谁快谁注册」必须显式判断。
    #
    # 但只有在 C++ **能力足够**时才让位：C++ 的快路径目前只做「同形、连续、同 dtype」，
    # 而真实前向里到处是广播（`x1 * cos[:, None, :]`）。让位过早的后果是
    # `probes/023` 里 `x1 * 2.0` 直接抛「只支持同形」——把「能算但慢」变成「不能算」。
    # 所以用 `elementwise_supports_broadcast()` 显式询问，C++ 侧补齐后自动切换。
    owned = _cpp_owned_schemas() if _cpp_supports_broadcast() else set()
    for op in ("add", "sub", "mul", "div"):
        if f"{op}.out" in owned:
            continue
        lib.impl(f"{op}.out", _out_binary(op), "PrivateUse1")
    # 第二批 aten（元数据类、规约、拼装）单独一个模块：它们要么靠 composite 自动成立、
    # 要么需要逐个判断「为什么不交给 composite」，混在这个文件里会让注册表变成一锅粥。
    # 模块还没写时不能阻塞设备，所以 import 失败只是少一批算子。
    try:
        from . import _aten_extra
    except ImportError:
        pass
    else:
        _aten_extra.register(lib)
    # 需要真算的算子（矩阵乘/规约/softmax/索引）单独一处：它们的内核在 _ops_* 里，
    # 这里只做形状与布局检查。缺了它设备仍然能用，只是这些算子报未实现。
    try:
        from . import _aten_kernels
    except ImportError:
        pass
    else:
        _aten_kernels.register(lib)


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
        """当前流。本后端只有一个 legacy null stream，所以永远返回它。"""
        return _runtime.Stream.default()

    class Stream:
        """``torch.corex.Stream``：把运行时的流对象暴露给 torch 的惯例 API。

        为什么要包一层而不是直接给 ``_runtime.Stream``：torch 生态里的代码会按惯例用
        ``torch.<device>.Stream()``、``with torch.<device>.stream(s):``、``s.wait_event(e)``、
        ``s.query()``，这些名字得在 device module 上找得到，否则框架（coffin）接进来时
        还得自己造适配层。

        这里用**持有 + 属性转发**而不是继承：``_runtime.Stream.__init__`` 要一个已创建的句柄
        （构造走 ``Stream.create()``），继承的话子类 ``__init__`` 得复刻这套创建逻辑，
        而转发只有一行。
        """

        def __init__(self, device: Optional[int] = None, priority: int = 0) -> None:
            self._inner = _runtime.Stream.create()

        def __getattr__(self, name: str):
            # 只在自身找不到时触发，所以 `self._inner` 本身不会递归（它已在 __dict__ 里）。
            return getattr(self._inner, name)

        def wait_event(self, event) -> None:
            """等一个事件（本平台只有 legacy null stream，等价于全同步）。"""
            event.synchronize()

        # `with` 语句查的是**类型**上的 `__enter__`/`__exit__`，`__getattr__` 转发对它们无效
        # （2026-11-07 在 probes/028 撞到 `AttributeError: __enter__`）。所以这两条必须显式写。
        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

        def __repr__(self) -> str:
            return f"<corex.Stream handle=0x{(self._inner.handle.value or 0):x}>"

    class Event(_runtime.Event):
        """``torch.corex.Event``：记时/同步用。

        ``elapsed_time`` 是 torch 的命名惯例，运行时里的同名能力叫 ``elapsed_ms``——
        两边都留着，谁也不必改。
        """

        def __init__(self, enable_timing: bool = False) -> None:
            _runtime.Event.__init__(self)
            self.enable_timing = bool(enable_timing)

        def elapsed_time(self, other) -> float:
            """两个事件之间的毫秒数（``other`` 在前，与 torch 的语义一致）。"""
            return float(self.elapsed_ms(other))

    def stream(stream_obj=None):
        """``with torch.corex.stream(s):`` —— 上下文管理器形式切流。

        直接返回 ``Stream`` 本身：它的 ``__enter__``/``__exit__`` 已经在运行时里实现了
        （见 ``_runtime.Stream``）。本平台只有 legacy null stream，所以这里的作用是
        **语义占位**——保证框架代码能按习惯写，不会被 ``AttributeError`` 拦住。

        Args:
            stream_obj: ``None`` 表示默认流。

        Returns:
            可作上下文管理器用的 ``Stream``。
        """
        return stream_obj if stream_obj is not None else _runtime.Stream.default()

    def manual_seed(seed: int) -> None:
        """卡上没有 RNG，这里只提醒一次，不抛异常。

        为什么不抛：``torch.manual_seed`` 是**全局**入口（同时 seed CPU），在我们的 device
        module 里抛会让下游任何一句 ``torch.manual_seed(0)`` 直接炸掉，惩罚过重且指向不明。
        但也不能完全静默——「静默成功」会让人以为随机序列可复现。所以第一次调用时警告一次；
        真正想用随机算子的人会撞上 ``aten::normal_`` 的显式 ``NotImplementedError``。
        """
        global _SEED_WARNED
        if not _SEED_WARNED:
            _SEED_WARNED = True
            import warnings

            warnings.warn(
                "corex 设备没有 RNG 通道：种子不生效，卡上的随机算子会显式报 NotImplementedError。"
                "需要随机数请先在 CPU 生成再 .to('corex')。",
                stacklevel=2,
            )

    def manual_seed_all(seed: int) -> None:
        """同 :func:`manual_seed`，多卡版本。"""
        manual_seed(seed)

    def _is_in_bad_fork() -> bool:
        """torch 的 fork 安全检查用；本后端没有 fork 语义，恒 False。"""
        return False

    def initial_seed() -> int:
        """没有 RNG，返回固定值。"""
        return 0

    for fn in (_lazy_init, is_available, device_count, current_device, set_device,
               synchronize, empty_cache, memory_allocated, max_memory_allocated,
               memory_reserved, current_stream, stream, manual_seed, manual_seed_all,
               _is_in_bad_fork, initial_seed):
        setattr(module, fn.__name__, fn)
    # 类要单独设：`fn.__name__` 对类来说是 `Stream`/`Event`，用同一个循环也行，但分开更清楚。
    module.Stream = Stream
    module.Event = Event
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
