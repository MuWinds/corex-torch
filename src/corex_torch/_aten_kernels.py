"""需要真正计算的 aten 算子：矩阵乘、规约、softmax、索引。

与 ``_aten_extra`` 的分工：那边只管「形状/stride/别名」这类不需要算的算子（它们靠 composite
或纯元数据操作就能成立），这边只放**必须落成内核**的那些。拆开的理由是验收口径不同——
元数据算子错了是形状对不上（立刻暴露），内核算错了可能只是数值偏一点（要拿 CPU 参照逐个比）。

内核本身全在 ``_ops_matmul`` / ``_ops_reduce`` / ``_ops_index`` 里（纯设备指针接口，不 import
torch）；本模块只做三件事：**形状与布局检查、out 张量的形状对账、把 aten 参数翻成内核参数**。

Module: ``corex_torch``。由 ``_device._register_ops`` 传入 ``torch.library.Library`` 后注册。

Constraints:
    * 一律要求**连续内存**。非连续输入直接 ``NotImplementedError``——线性内核按 stride=行主序
      假设写，喂给它一个转置视图不会报错，只会算出另一组数。
    * ``*.out`` 变体不负责 resize：``out`` 的形状与期望不符就报错。resize 要动 storage，
      在没有 caching allocator 配合的情况下容易留下悬垂指针，本阶段不做。
    * bf16 的规约/softmax/矩阵乘在厂商 llc 上编不出来（``fp_extend``/``fp_round`` 选不出指令），
      内核层会抛 ``NotImplementedError``，这里原样透传，不捕获也不降精度。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

DEVICE_NAME = "corex"


def _mod(name: str):
    """按需 import 内核模块（torch / triton 都可能在纯编译机上缺席）。"""
    import importlib

    return importlib.import_module(f".{name}", __package__)


def _empty(shape: Sequence[int], dtype, device) -> Any:
    """按行主序造一个张量（复用设备层的 empty 实现，不另写一份）。"""
    from . import _device

    return _device._empty_impl(list(shape), dtype=dtype, device=device)


def _dtype_name(dtype) -> str:
    from . import _device

    return _device._dtype_name(dtype)


def _need(tensor, op: str, label: str) -> None:
    """检查「在 corex 上 + 连续」。

    Raises:
        NotImplementedError: 不满足条件。消息里带算子名与哪一侧，便于直接定位调用点。
    """
    if tensor.device.type != DEVICE_NAME:
        raise NotImplementedError(f"corex 的 {op} 要求 {label} 在 corex 上，收到 {tensor.device}")
    if not tensor.is_contiguous():
        raise NotImplementedError(f"corex 的 {op} 暂不支持非连续 {label}（先 contiguous）")


def _mm_impl(mat1, mat2):
    """``aten::mm``。"""
    _need(mat1, "mm", "self")
    _need(mat2, "mm", "mat2")
    if mat1.dim() != 2 or mat2.dim() != 2:
        raise NotImplementedError(f"corex 的 mm 只处理二维，收到 {tuple(mat1.shape)} 与 {tuple(mat2.shape)}")
    m, k = int(mat1.shape[0]), int(mat1.shape[1])
    k2, n = int(mat2.shape[0]), int(mat2.shape[1])
    if k != k2:
        raise RuntimeError(f"mm 形状不匹配：{tuple(mat1.shape)} @ {tuple(mat2.shape)}")
    out = _empty([m, n], mat1.dtype, mat1.device)
    _mod("_ops_matmul").mm(mat1.data_ptr(), mat2.data_ptr(), out.data_ptr(),
                           m, n, k, _dtype_name(mat1.dtype))
    return out


def _bmm_impl(self, mat2):
    """``aten::bmm``。"""
    _need(self, "bmm", "self")
    _need(mat2, "bmm", "mat2")
    if self.dim() != 3 or mat2.dim() != 3 or self.shape[0] != mat2.shape[0]:
        raise NotImplementedError(
            f"corex 的 bmm 只处理同 batch 的三维，收到 {tuple(self.shape)} 与 {tuple(mat2.shape)}")
    b, m, k = (int(x) for x in self.shape)
    n = int(mat2.shape[2])
    out = _empty([b, m, n], self.dtype, self.device)
    _mod("_ops_matmul").bmm(self.data_ptr(), mat2.data_ptr(), out.data_ptr(),
                            b, m, n, k, _dtype_name(self.dtype))
    return out


def _softmax_out_impl(self, dim: int, half_to_float: bool, *, out=None):
    """``aten::_softmax.out``：只做最后一维。

    为什么只做最后一维：内核是「一行一段连续内存」的行内 online softmax。中间维需要跨步走，
    本阶段没有 strided 内核，给出明确报错比悄悄算错强。
    """
    if out is None:
        raise NotImplementedError("corex 的 _softmax.out 要求显式 out")
    _need(self, "_softmax", "self")
    ndim = self.dim()
    if ndim == 0 or (int(dim) % ndim) != ndim - 1:
        raise NotImplementedError(f"corex 的 softmax 只支持最后一维，收到 dim={dim}（ndim={ndim}）")
    if half_to_float and self.dtype.is_floating_point and self.dtype.itemsize == 2:
        raise NotImplementedError("corex 的 _softmax.out 暂不支持 half_to_float（fp16 计算留在 fp16）")
    if tuple(out.shape) != tuple(self.shape) or out.dtype != self.dtype:
        raise NotImplementedError(
            f"corex 的 _softmax.out 不做 resize/换 dtype：out={tuple(out.shape)}/{out.dtype} "
            f"self={tuple(self.shape)}/{self.dtype}")
    _need(out, "_softmax", "out")
    cols = int(self.shape[-1])
    rows = int(self.numel()) // cols if cols else 0
    _mod("_ops_reduce").softmax_last(self.data_ptr(), out.data_ptr(), rows, cols,
                                     _dtype_name(self.dtype))
    return out


def _sum_intlist_out_impl(self, dim, keepdim: bool = False, *, dtype=None, out=None):
    """``aten::sum.IntList_out``：支持「全归约」与「最后一维归约」两种。

    ``sum`` / ``sum.dim_IntList`` 在本版 torch 里是 composite，最终都落到这个 out 变体上，
    所以它是规约链路的公共收口。
    """
    if out is None:
        raise NotImplementedError("corex 的 sum.IntList_out 要求显式 out")
    _need(self, "sum", "self")
    if dtype is not None and dtype != self.dtype:
        # 整数求和的 dtype 提升（i32->i64）由 composite 传下来，我们的内核不做类型转换。
        raise NotImplementedError(
            f"corex 的 sum 暂不支持 dtype 提升：self={self.dtype} dtype={dtype}")
    dims = None if dim is None else [int(d) for d in dim]
    ndim = self.dim()
    reduce_all = dims is None or len(dims) == 0 or len(set(dims)) == ndim
    if reduce_all:
        if int(out.numel()) != 1:
            raise NotImplementedError(f"corex 的 sum 全归约要求 out 只有 1 个元素，收到 {tuple(out.shape)}")
        _mod("_ops_reduce").sum_all(self.data_ptr(), out.data_ptr(), int(self.numel()),
                                    _dtype_name(self.dtype))
        return out
    if len(dims) != 1 or (dims[0] % ndim) != ndim - 1:
        raise NotImplementedError(f"corex 的 sum 只支持「全归约」或「最后一维」，收到 dim={dim}")
    cols = int(self.shape[-1])
    rows = int(self.numel()) // cols if cols else 0
    if int(out.numel()) != rows:
        raise NotImplementedError(f"corex 的 sum 期望 out.numel()=={rows}，收到 {tuple(out.shape)}")
    _mod("_ops_reduce").sum_last(self.data_ptr(), out.data_ptr(), rows, cols,
                                 _dtype_name(self.dtype))
    return out


def _mean_out_impl(self, dim, keepdim: bool = False, *, dtype=None, out=None):
    """``aten::mean.out``：与 ``sum`` 同样的两种形状。"""
    if out is None:
        raise NotImplementedError("corex 的 mean.out 要求显式 out")
    _need(self, "mean", "self")
    dims = None if dim is None else [int(d) for d in dim]
    ndim = self.dim()
    reduce_all = dims is None or len(dims) == 0 or len(set(dims)) == ndim
    if reduce_all:
        if int(out.numel()) != 1:
            raise NotImplementedError(f"corex 的 mean 全归约要求 out 只有 1 个元素")
        _mod("_ops_reduce").sum_all(self.data_ptr(), out.data_ptr(), int(self.numel()),
                                    _dtype_name(self.dtype))
        _scale_inplace(out, 1.0 / max(1, int(self.numel())))
        return out
    if len(dims) != 1 or (dims[0] % ndim) != ndim - 1:
        raise NotImplementedError(f"corex 的 mean 只支持「全归约」或「最后一维」，收到 dim={dim}")
    cols = int(self.shape[-1])
    rows = int(self.numel()) // cols if cols else 0
    _mod("_ops_reduce").mean_last(self.data_ptr(), out.data_ptr(), rows, cols,
                                  _dtype_name(self.dtype))
    return out


def _scale_inplace(tensor, factor: float) -> None:
    """把张量整体乘一个标量（走元素级内核，避免为了一个常量再写一个规约变体）。"""
    ops = _mod("_ops_elementwise")
    ops.mul_scalar(tensor.data_ptr(), tensor.data_ptr(), int(tensor.numel()),
                   _dtype_name(tensor.dtype), float(factor))


def _index_select_impl(self, dim: int, index) -> Any:
    """``aten::index_select``：只做第 0 维（embedding / 取行的形状）。"""
    _need(self, "index_select", "self")
    if int(dim) != 0:
        raise NotImplementedError(f"corex 的 index_select 只支持 dim=0，收到 {dim}")
    if index.device.type != DEVICE_NAME:
        raise NotImplementedError(f"corex 的 index_select 要求 index 也在 corex 上，收到 {index.device}")
    if not index.is_contiguous():
        raise NotImplementedError("corex 的 index_select 暂不支持非连续 index")
    import torch

    # torch 的 index_select 在 CUDA 上约定用 int64 下标，内核按宽度分别有实体；
    # 传错宽度会按一半的元素数读，且不报错（`_ops_index.gather_rows` 的 idx_dtype）。
    if index.dtype == torch.int64:
        idx_dtype = "i64"
    elif index.dtype == torch.int32:
        idx_dtype = "i32"
    else:
        raise NotImplementedError(f"corex 的 index_select 只支持 i32/i64 下标，收到 {index.dtype}")
    rows = int(index.numel())
    cols = int(self.numel()) // int(self.shape[0]) if self.dim() else 0
    out = _empty([rows] + list(self.shape[1:]), self.dtype, self.device)
    _mod("_ops_index").gather_rows(self.data_ptr(), index.data_ptr(), out.data_ptr(),
                                   rows, cols, _dtype_name(self.dtype), idx_dtype=idx_dtype)
    return out


def register(lib) -> None:
    """把需要内核的 aten 算子挂到 PrivateUse1 上。

    Args:
        lib: ``torch.library.Library("aten", "IMPL")`` 实例，由 ``_device._register_ops`` 传入。

    Returns:
        无。注册失败会直接抛——这类算子缺失时的报错是 ``NotImplementedError`` 指向 aten 内部，
        很难看出是我们的分发层没挂上，所以宁可在注册期就炸。
    """
    lib.impl("mm", _mm_impl, "PrivateUse1")
    lib.impl("bmm", _bmm_impl, "PrivateUse1")
    lib.impl("_softmax.out", _softmax_out_impl, "PrivateUse1")
    lib.impl("sum.IntList_out", _sum_intlist_out_impl, "PrivateUse1")
    lib.impl("mean.out", _mean_out_impl, "PrivateUse1")
    lib.impl("index_select", _index_select_impl, "PrivateUse1")
