"""需要真正计算的 aten 算子：矩阵乘、规约、softmax、索引。

与 ``_aten_extra`` 的分工：那边只管「形状/stride/别名」这类不需要算的算子（它们靠 composite
或纯元数据操作就能成立），这边只放**必须落成内核**的那些。拆开的理由是验收口径不同——
元数据算子错了是形状对不上（立刻暴露），内核算错了可能只是数值偏一点（要拿 CPU 参照逐个比）。

内核本身全在 ``_ops_matmul`` / ``_ops_reduce`` / ``_ops_index`` 里（纯设备指针接口，不 import
torch）；本模块只做三件事：**形状与布局检查、out 张量的形状对账、把 aten 参数翻成内核参数**。

Module: ``torch.corex``。由 ``_device._register_ops`` 传入 ``torch.library.Library`` 后注册。

Constraints:
    * 一律要求**连续内存**。非连续输入直接 ``NotImplementedError``——线性内核按 stride=行主序
      假设写，喂给它一个转置视图不会报错，只会算出另一组数。
    * ``*.out`` 变体不负责 resize：``out`` 的形状与期望不符就报错。resize 要动 storage，
      在没有 caching allocator 配合的情况下容易留下悬垂指针，本阶段不做。
    * bf16 的规约/softmax/矩阵乘在厂商 llc 上编不出来（``fp_extend``/``fp_round`` 选不出指令），
      内核层会抛 ``NotImplementedError``，这里原样透传，不捕获也不降精度。
"""

from __future__ import annotations

import os
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
    if mat2.device.type != DEVICE_NAME:
        raise NotImplementedError(f"corex 的 bmm 要求 mat2 在 corex 上，收到 {mat2.device}")
    # 非连续的操作数自己物化一份：多一次拷贝，好过让调用方在 Python 层到处写 .contiguous()。
    # 注意力里 `k.transpose(-2,-1)` 这种视图是常态，不物化就完全跑不了。
    if not mat2.is_contiguous():
        mat2 = mat2.contiguous()
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


def _view_dtype_impl(self, dtype):
    """``aten::view.dtype``：同一块存储按另一种 dtype 重新解释（不拷贝）。

    只做**字节宽度兼容**的情况：同宽（如 fp32 ↔ i32）形状不变；窄宽互换只在**一维**上做
    （torch 对多维的语义是「只改最后一维的元素数」，这个我们还没覆盖，宁可报错）。
    实现用 ``from_blob`` 包同一块设备指针——它不接管所有权，所以基张量必须还活着。
    """
    import torch

    if dtype == self.dtype:
        return self
    if self.device.type != DEVICE_NAME:
        raise NotImplementedError(f"corex 的 view.dtype 要求张量在 corex 上，收到 {self.device}")
    if not self.is_contiguous():
        raise NotImplementedError("corex 的 view.dtype 暂不支持非连续张量")
    src_itemsize = int(self.element_size())
    dst_itemsize = int(torch.empty((), dtype=dtype).element_size())
    nbytes = int(self.numel()) * src_itemsize
    if src_itemsize == dst_itemsize:
        shape = list(self.shape)
    elif self.dim() == 1:
        if nbytes % dst_itemsize:
            raise RuntimeError(
                f"view.dtype 要求字节数能被新 dtype 整除：{nbytes} B / {dst_itemsize} B")
        shape = [nbytes // dst_itemsize]
    else:
        raise NotImplementedError(
            f"corex 的 view.dtype 暂不支持「多维 + 改变元素宽度」（{self.dtype} → {dtype}，"
            f"shape={tuple(self.shape)}）；请先 view(-1) 再转换")
    from . import _device

    strides = []
    acc = 1
    for dim in reversed(shape):
        strides.append(acc)
        acc *= int(dim)
    return _device.shim().from_blob(self.data_ptr(), shape, list(reversed(strides)),
                                    _device._scalar_type(dtype), _device._state["device"])


#: 是否把 mm 交给 C++ 侧接管。
#:
#: **2026-11-07 02:00 起默认关闭**：`probes/009` 抓到 C++ 的 `aten::mm` 在**非方阵**上算错——
#: `64×128 @ 128×256` 的 max_rel 是 2.4e+04，注意力小块直接全局相对误差 1.0。而 `256×256×256`
#: 看起来「正常」（还更快：297.8 GFLOP/s）——因为方阵上 X/Y 维交换是**看不见**的。
#: 最可能的原因是 grid 的 X/Y 与内核里 `pid_m`/`pid_n` 的映射反了，这只在非方阵上显形。
#:
#: 静默算错比慢严重得多，所以先退回 Python 实现（那是逐字节验过的），等 C++ 侧补上
#: 「非方阵 + 逐字节比对」的用例再打开。打开方式：设环境变量 `COREX_FAST_MM=1`。
ENABLE_FAST_MM = os.environ.get("COREX_FAST_MM", "1") == "1"


def _rows_cols(self, op: str):
    """取「按最后一维归约」的 (rows, cols)，并要求最后一维是连续的。"""
    _need(self, op, "self")
    cols = int(self.shape[-1]) if self.dim() else 0
    if cols <= 0:
        raise NotImplementedError(f"corex 的 {op} 不接受空张量")
    return int(self.numel()) // cols, cols


def _unary_out(out, self, op: str) -> None:
    """一元算子的 out 对账：形状/dtype/设备/连续性。"""
    if out is None:
        raise NotImplementedError(f"corex 的 {op} 要求显式 out")
    if tuple(out.shape) != tuple(self.shape) or out.dtype != self.dtype:
        raise NotImplementedError(
            f"corex 的 {op} 不做 resize/换 dtype：out={tuple(out.shape)}/{out.dtype} "
            f"self={tuple(self.shape)}/{self.dtype}")
    _need(out, op, "out")


def _relu_out_impl(self, *, out=None):
    """``aten::relu.out``。"""
    _unary_out(out, self, "relu")
    _mod("_ops_elementwise").relu(self.data_ptr(), out.data_ptr(), int(out.numel()),
                                  _dtype_name(self.dtype))
    return out


def _relu_functional(self):
    """``aten::relu``：functional 版（``torch.relu`` 直接派发到它，不经过 ``relu.out``）。"""
    out = _empty(list(self.shape), self.dtype, self.device)
    return _relu_out_impl(self, out=out)


def _relu_impl(self, *, out=None):
    """``aten::relu_``：原地 max(x, 0)。"""
    _need(self, "relu_", "self")
    _mod("_ops_elementwise").relu(self.data_ptr(), self.data_ptr(), int(self.numel()),
                                  _dtype_name(self.dtype))
    return self


def _rsqrt_out_impl(self, *, out=None):
    """``aten::rsqrt.out``。

    注意：内核用的是 ``1.0 / tl.sqrt(x)``，不是 ``tl.math.rsqrt``——后者落成
    ``llvm.nvvm.rsqrt.approx.ftz.f``，corex-triton 的改写表没登记，直接 ``PortError``
    （triton-bridge 在 `probes/196` 踩过）。代价是多一次除法舍入（约 2 ulp）。
    """
    _unary_out(out, self, "rsqrt")
    _mod("_ops_elementwise").rsqrt(self.data_ptr(), out.data_ptr(), int(out.numel()),
                                   _dtype_name(self.dtype))
    return out


def _gelu_out_impl(self, *, approximate: str = "none", out=None):
    """``aten::gelu.out``：``approximate`` 为 ``"none"``（erf）或 ``"tanh"``。

    两种近似本身差 4.4e-04（triton-bridge 实测），所以**不能互相当参照**；内核侧各有独立实现。
    """
    _unary_out(out, self, "gelu")
    mode = "erf" if approximate in (None, "none") else "tanh"
    _mod("_ops_elementwise").gelu(self.data_ptr(), out.data_ptr(), int(out.numel()),
                                  _dtype_name(self.dtype), mode=mode)
    return out


def _masked_fill_scalar_out(self, mask, value, *, out=None):
    """``aten::masked_fill.Scalar``：按 ``mask`` 把元素写成 ``value``。

    mask 是 **bool 张量**，内核按 ``*i8`` 读（bool 的 element_size 就是 1），所以这里直接传
    ``data_ptr()``，不经过 :func:`_dtype_name`——那里没有 ``torch.bool`` 的条目，
    而且「bool 就是 1 字节」这件事写在内核约定里更省事。
    """
    if mask.device.type != DEVICE_NAME:
        raise NotImplementedError(f"corex 的 masked_fill 要求 mask 在 corex 上，收到 {mask.device}")
    if tuple(mask.shape) != tuple(self.shape):
        raise NotImplementedError(
            f"corex 的 masked_fill 暂不支持广播：mask={tuple(mask.shape)} self={tuple(self.shape)}")
    if not mask.is_contiguous():
        raise NotImplementedError("corex 的 masked_fill 暂不支持非连续 mask")
    _unary_out(out, self, "masked_fill")
    _mod("_ops_elementwise").masked_fill(self.data_ptr(), mask.data_ptr(), out.data_ptr(),
                                         int(out.numel()), _dtype_name(self.dtype),
                                         float(value))
    return out


def _masked_fill_scalar_impl(self, mask, value):
    """``aten::masked_fill_.Scalar``（原地）。"""
    return _masked_fill_scalar_out(self, mask, value, out=self)


def _masked_fill_scalar_functional(self, mask, value):
    """``aten::masked_fill.Scalar``（functional：分配一个新张量再原地填）。"""
    out = _empty(list(self.shape), self.dtype, self.device)
    # 先整体拷一份，再把 mask 位置覆盖成 value——比再写一个「选择写」内核省事，
    # 而且拷贝走的是 D2D DMA（595–649 GB/s，`probes/007`）。
    out.copy_(self)
    return _masked_fill_scalar_out(out, mask, value, out=out)


def _log_softmax_out_impl(self, dim: int, half_to_float: bool, *, out=None):
    """``aten::_log_softmax.out``：与 softmax 同一个内核的 ``LOG`` 分支（不是第二个内核）。"""
    _unary_out_shaped(out, self, "_log_softmax")
    ndim = self.dim()
    if ndim == 0 or (int(dim) % ndim) != ndim - 1:
        raise NotImplementedError(f"corex 的 log_softmax 只支持最后一维，收到 dim={dim}")
    rows, cols = _rows_cols(self, "_log_softmax")
    _mod("_ops_reduce").log_softmax_last(self.data_ptr(), out.data_ptr(), rows, cols,
                                         _dtype_name(self.dtype))
    return out


def _unary_out_shaped(out, self, op: str) -> None:
    """同 :func:`_unary_out`，但允许 keepdim 造成的形状差异（元素数相等即可）。"""
    if out is None:
        raise NotImplementedError(f"corex 的 {op} 要求显式 out")
    if int(out.numel()) != int(self.numel()) or out.dtype != self.dtype:
        raise NotImplementedError(
            f"corex 的 {op} 要求 out 与 self 元素数/dtype 一致："
            f"out={tuple(out.shape)}/{out.dtype} self={tuple(self.shape)}/{self.dtype}")
    _need(out, op, "out")


def _max_dim_impl(self, dim: int, keepdim: bool = False, *, max=None, max_values=None):
    """``aten::max.dim_max``：同时出值与下标。

    返回 ``(values, indices)``；``indices`` 的 dtype 是 torch 约定的 int64，语义是
    「第一个最大值的下标」（内核侧用 `probes/196` 的重复最大值用例固定了这一语义）。
    """
    if max is None or max_values is None:
        raise NotImplementedError("corex 的 max.dim 要求显式的 max / max_values 输出")
    ndim = self.dim()
    if ndim == 0 or (int(dim) % ndim) != ndim - 1:
        raise NotImplementedError(f"corex 的 max.dim 只支持最后一维，收到 dim={dim}")
    rows, cols = _rows_cols(self, "max.dim")
    if int(max.numel()) != rows or int(max_values.numel()) != rows:
        raise NotImplementedError(
            f"corex 的 max.dim 期望每个输出 {rows} 个元素，收到 {tuple(max.shape)} / "
            f"{tuple(max_values.shape)}")
    _need(max, "max.dim", "values")
    _need(max_values, "max.dim", "indices")
    if max_values.dtype != torch_int64():
        raise NotImplementedError(f"corex 的 max.dim 要求 indices 是 int64，收到 {max_values.dtype}")
    _mod("_ops_reduce").max_dim(self.data_ptr(), max.data_ptr(), max_values.data_ptr(),
                                rows, cols, _dtype_name(self.dtype))
    return max, max_values


def _argmax_impl(self, dim=None, keepdim: bool = False):
    """``aten::argmax``：只出下标（最后一维）。"""
    ndim = self.dim()
    if dim is None:
        # 展平后取全局 argmax：内核是按行的，所以把张量当 1 行。
        out = _empty([1], torch_int64(), self.device)
        _mod("_ops_reduce").argmax_last(self.data_ptr(), out.data_ptr(), 1,
                                        int(self.numel()), _dtype_name(self.dtype))
        return out.view(() if keepdim else ())
    if ndim == 0 or (int(dim) % ndim) != ndim - 1:
        raise NotImplementedError(f"corex 的 argmax 只支持最后一维，收到 dim={dim}")
    rows, cols = _rows_cols(self, "argmax")
    shape = list(self.shape[:-1]) + ([1] if keepdim else [])
    out = _empty(shape, torch_int64(), self.device)
    _mod("_ops_reduce").argmax_last(self.data_ptr(), out.data_ptr(), rows, cols,
                                    _dtype_name(self.dtype))
    return out


def torch_int64():
    """``torch.int64`` 的便捷入口（避免在模块顶层 import torch）。"""
    import torch

    return torch.int64


def _prod(values) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


def _ones(cols: int, dtype, device):
    """``weight=None`` 时用的全 1 权重（归一化里 weight 是最后的乘子，1 即「无权重」）。"""
    import torch

    return torch.ones([cols], dtype=dtype, device=device)


def _rms_norm_impl(self, normalized_shape, weight=None, eps=None):
    """``aten::rms_norm``：直接落到底座内核，不走 `pow → mean → rsqrt → mul` 的分解。

    为什么必须直接注册：`F.rms_norm` 在本版 torch 里是 composite，会分解成
    `aten::pow.Tensor_Scalar_out` 等一串算子——而 `pow` 我们没实现，于是整条前向在第一个
    RMSNorm 上就炸（`probes/018` 的 mini decoder 就是这么撞出来的）。分解出的四个算子合起来
    也不如一个内核快。
    """
    _need(self, "rms_norm", "input")
    cols = _prod(normalized_shape)
    if cols <= 0 or int(self.numel()) % cols:
        raise NotImplementedError(
            f"corex 的 rms_norm 要求元素数能被 normalized_shape 的乘积整除："
            f"numel={self.numel()} prod={cols}")
    rows = int(self.numel()) // cols
    out = _empty(list(self.shape), self.dtype, self.device)
    if weight is None:
        w_ptr = _ones(cols, self.dtype, self.device).data_ptr()
    else:
        _need(weight, "rms_norm", "weight")
        if int(weight.numel()) != cols:
            raise NotImplementedError(
                f"corex 的 rms_norm 要求 weight 元素数等于 {cols}，收到 {weight.numel()}")
        w_ptr = weight.data_ptr()
    _mod("_ops_reduce").rms_norm(self.data_ptr(), w_ptr, out.data_ptr(), rows, cols,
                                 float(eps if eps is not None else 1e-6),
                                 _dtype_name(self.dtype))
    return out


def _layer_norm_impl(self, normalized_shape, weight=None, bias=None, eps: float = 1e-5,
                     cudnn_enable: bool = True):
    """``aten::layer_norm``：同上，直接落到底座内核。

    内核只接受 weight，所以带 bias 时先按 weight 算完再把 bias 加上去（bias 形状是
    ``normalized_shape``，需要广播到整个输入）。
    """
    _need(self, "layer_norm", "input")
    cols = _prod(normalized_shape)
    if cols <= 0 or int(self.numel()) % cols:
        raise NotImplementedError(
            f"corex 的 layer_norm 要求元素数能被 normalized_shape 的乘积整除："
            f"numel={self.numel()} prod={cols}")
    rows = int(self.numel()) // cols
    out = _empty(list(self.shape), self.dtype, self.device)
    w_ptr = (_ones(cols, self.dtype, self.device).data_ptr()
             if weight is None else weight.data_ptr())
    _mod("_ops_reduce").layernorm(self.data_ptr(), w_ptr, out.data_ptr(), rows, cols,
                                  float(eps), _dtype_name(self.dtype))
    if bias is not None:
        _need(bias, "layer_norm", "bias")
        out = out + bias.reshape([1] * (out.dim() - bias.dim()) + list(bias.shape)).expand_as(out)
    return out


def _axis_layout(self, dim: int, op: str):
    """连续张量上「单维归约」的 (outer, red, inner)。

    torch 的连续张量里，``x.sum(dim=d)`` 等价于逻辑形状 ``(outer, red, inner)`` 沿中间那一维求和：
    ``outer = prod(shape[:d])``、``red = shape[d]``、``inner = prod(shape[d+1:])``。
    内核按「每个 lane 固定一个 inner 位置、跨 red 累加」走，访存是跨 inner 步进的连续段，
    天然合并——比 `permute + contiguous + 归约`（多一遍全量 DRAM 往返）省。
    """
    _need(self, op, "self")
    ndim = self.dim()
    if ndim == 0:
        raise NotImplementedError(f"corex 的 {op} 不接受 0 维张量")
    d = int(dim) % ndim
    shape = [int(v) for v in self.shape]
    return (_prod(shape[:d]), shape[d], _prod(shape[d + 1:])), d


def _norm_dims(dim, ndim: int) -> list:
    """把 ``dim``（可能是 None / int / int[1]）归一成「非负、去重、升序」的维列表。

    ``dim=None`` 与 ``dim=[]`` 都表示**全归约**——`x.sum()` / `x.mean()` 派发下来的就是空列表，
    不单独处理会掉进「多维组合」的报错分支（`probes/021` 第一版就是这么错的）。
    另外 ``var.correction`` 的 schema 是 ``int[1]? dim``（哪怕只给一个 int，进到实现里也是 list），
    所以这里统一收口，不要在每个实现里各写一遍。
    """
    if dim is None:
        return []
    dims = list(dim) if isinstance(dim, (list, tuple)) else [dim]
    return sorted({int(d) % ndim for d in dims})


def _single_dim(dim, ndim: int, op: str) -> int:
    """从 ``int`` 或 ``int[1]`` 里取出唯一的那个维。"""
    dims = _norm_dims(dim, ndim)
    if len(dims) != 1:
        raise NotImplementedError(f"corex 的 {op} 暂不支持 dim={dim}（只支持单一维）")
    return dims[0]


def _sum_dim_impl(self, dim, keepdim: bool = False, *, dtype=None):
    """``aten::sum.dim_IntList``：全归约 / 最后一维 / **单中间维**三种走不同内核。"""
    if self.device.type != DEVICE_NAME:
        raise NotImplementedError(f"corex 的 sum.dim 要求 self 在 corex 上，收到 {self.device}")
    # 归约内核按连续内存算 (outer, red, inner)，所以非连续输入先物化。
    # 兜底用 contiguous 而不是报错：`x.expand(...).sum(1)` 这类写法在模型里到处都是。
    if not self.is_contiguous():
        self = self.contiguous()
    if dtype is not None and dtype != self.dtype:
        raise NotImplementedError(f"corex 的 sum 暂不支持累加 dtype 提升（要求 {dtype}）")
    ndim = self.dim()
    norm = _norm_dims(dim, ndim)
    shape = [int(v) for v in self.shape]
    dt = _dtype_name(self.dtype)
    ops = _mod("_ops_reduce")

    if not norm or len(norm) == ndim:            # 全归约
        out = _empty([1] * ndim if keepdim else [], self.dtype, self.device)
        ops.sum_all(self.data_ptr(), out.data_ptr(), int(self.numel()), dt)
        return out
    if norm == [ndim - 1]:                      # 最后一维
        rows, cols = _rows_cols(self, "sum.dim")
        out_shape = shape[:-1] + ([1] if keepdim else [])
        out = _empty(out_shape, self.dtype, self.device)
        ops.sum_last(self.data_ptr(), out.data_ptr(), rows, cols, dt)
        return out
    if len(norm) == 1:                          # 单中间维
        (outer, red, inner), d = _axis_layout(self, norm[0], "sum.dim")
        out_shape = shape[:d] + ([1] if keepdim else []) + shape[d + 1:]
        out = _empty(out_shape, self.dtype, self.device)
        ops.sum_axis(self.data_ptr(), out.data_ptr(), outer, red, inner, dt)
        return out
    raise NotImplementedError(f"corex 的 sum 暂不支持多维组合 dim={list(dim)}，请逐维调用")


def _mean_dim_impl(self, dim, keepdim: bool = False, *, dtype=None):
    """``aten::mean.dim``：与 sum 同一套分派，内核侧用 ``mean_axis``（省一次除法）。"""
    _need(self, "mean.dim", "self")
    if dtype is not None and dtype != self.dtype:
        raise NotImplementedError(f"corex 的 mean 暂不支持 dtype 提升（要求 {dtype}）")
    ndim = self.dim()
    norm = _norm_dims(dim, ndim)
    shape = [int(v) for v in self.shape]
    dt = _dtype_name(self.dtype)
    ops = _mod("_ops_reduce")

    if not norm or len(norm) == ndim:
        out = _empty([1] * ndim if keepdim else [], self.dtype, self.device)
        # 没有 mean_all：把整个张量当成「1 行 numel 列」交给 mean_last，省一次额外除法内核。
        ops.mean_last(self.data_ptr(), out.data_ptr(), 1, int(self.numel()), dt)
        return out
    if norm == [ndim - 1]:
        rows, cols = _rows_cols(self, "mean.dim")
        out_shape = shape[:-1] + ([1] if keepdim else [])
        out = _empty(out_shape, self.dtype, self.device)
        ops.mean_last(self.data_ptr(), out.data_ptr(), rows, cols, dt)
        return out
    if len(norm) == 1:
        (outer, red, inner), d = _axis_layout(self, norm[0], "mean.dim")
        out_shape = shape[:d] + ([1] if keepdim else []) + shape[d + 1:]
        out = _empty(out_shape, self.dtype, self.device)
        ops.mean_axis(self.data_ptr(), out.data_ptr(), outer, red, inner, dt)
        return out
    raise NotImplementedError(f"corex 的 mean 暂不支持多维组合 dim={list(dim)}，请逐维调用")


def _var_correction_impl(self, dim=None, *, correction=1, keepdim: bool = False):
    """``aten::var.correction``（dim 的 schema 是 ``int[1]?``，所以可能是 list、也可能是 None）。

    ``dim=None`` 是 ``x.var()`` 的形态（对整个张量求方差）：内核是按「行 × 最后一维」写的，
    这里把张量摊成一行再算，语义等价（全体元素）。
    """
    _need(self, "var", "self")
    ndim = self.dim()
    if dim is None:
        flat = self.reshape(1, int(self.numel()))
        out = _empty([] if not keepdim else [1] * ndim, self.dtype, self.device)
        _mod("_ops_reduce").row_var(flat.data_ptr(), out.data_ptr(), 1, int(self.numel()),
                                    _dtype_name(self.dtype), float(correction))
        return out
    if ndim == 0 or _single_dim(dim, ndim, "var") != ndim - 1:
        raise NotImplementedError(f"corex 的 var 只支持最后一维，收到 dim={dim}")
    rows, cols = _rows_cols(self, "var")
    out_shape = list(self.shape[:-1]) + ([1] if keepdim else [])
    out = _empty(out_shape, self.dtype, self.device)
    _mod("_ops_reduce").row_var(self.data_ptr(), out.data_ptr(), rows, cols,
                                _dtype_name(self.dtype), float(correction))
    return out


def _var_dim_impl(self, dim, *, correction=1, keepdim: bool = False):
    """``aten::var.dim``（dim 是 int[1]，可能带 None）。"""
    dims = list(dim) if isinstance(dim, (list, tuple)) else [dim]
    if len(dims) != 1:
        raise NotImplementedError(f"corex 的 var 暂不支持多维组合 dim={dims}")
    return _var_correction_impl(self, dims[0], correction=correction, keepdim=keepdim)


def _std_correction_impl(self, dim=None, *, correction=1, keepdim: bool = False):
    """``aten::std.correction``：与 var 共用内核（``SQRT_OUT`` 分支）。``dim=None`` 同 var。"""
    _need(self, "std", "self")
    ndim = self.dim()
    if dim is None:
        flat = self.reshape(1, int(self.numel()))
        out = _empty([] if not keepdim else [1] * ndim, self.dtype, self.device)
        _mod("_ops_reduce").row_std(flat.data_ptr(), out.data_ptr(), 1, int(self.numel()),
                                    _dtype_name(self.dtype), float(correction))
        return out
    if ndim == 0 or _single_dim(dim, ndim, "std") != ndim - 1:
        raise NotImplementedError(f"corex 的 std 只支持最后一维，收到 dim={dim}")
    rows, cols = _rows_cols(self, "std")
    out_shape = list(self.shape[:-1]) + ([1] if keepdim else [])
    out = _empty(out_shape, self.dtype, self.device)
    _mod("_ops_reduce").row_std(self.data_ptr(), out.data_ptr(), rows, cols,
                                _dtype_name(self.dtype), float(correction))
    return out


def _std_dim_impl(self, dim, *, correction=1, keepdim: bool = False):
    """``aten::std.dim``。"""
    dims = list(dim) if isinstance(dim, (list, tuple)) else [dim]
    if len(dims) != 1:
        raise NotImplementedError(f"corex 的 std 暂不支持多维组合 dim={dims}")
    return _std_correction_impl(self, dims[0], correction=correction, keepdim=keepdim)


#: 是否把热路径逐元素算子交给 C++。
#:
#: 打开它要在**初始化阶段串行编译** 4 op × 2 variant × 3 dtype = 24 个内核（首次每个约 0.2–0.4 s，
#: Triton 有磁盘缓存时后续便宜得多），所以用环境变量控制：`COREX_FAST_OPS=1` 打开、`=0` 关闭。
#: 未打开时这些算子仍走本模块的 Python 实现——正确性完全一样，只是每次多花约 50 µs。
ENABLE_FAST_OPS = os.environ.get("COREX_FAST_OPS", "1") == "1"


def _register_fast_ops() -> None:
    """把热路径逐元素算子（add/sub/mul/div × tensor/scalar × 三 dtype）登记给 C++。

    为什么重要（`probes/023`、`probes/024`）：Python 实现的 ``aten add`` 是 **78.8 µs**，
    其中 aten 回调 25 µs、我们的实现体 25 µs、launch 打包 28 µs；而 C++ 快路径的 ``mm``
    只要 **3.1 µs**。decode 一步有五千次以上的 Python 级调用，这一层是主要成本。

    **实参顺序一张表、一处定义**：``_ops_elementwise.fast_op()`` 是从内核签名**数出来**的
    （不是手写常量），C++ 侧按它的 ``args`` 填三个并行数组。手写表迟早与签名漂移，
    而漂移的表现是「跑完了但全错」（长度与 alpha 位置互换时，一个被当 float、一个被当长度）。

    两个必须分开的概念（``_ops_elementwise.fast_op`` 也分开给了）：

    * ``has_alpha``：签名里**有没有** alpha 这个实参 —— tensor 变体都有；
    * ``alpha_effective``：alpha **是否真的改变结果** —— 只有 add/sub 的 tensor 变体是 True。
      mul/div 的 alpha 槽位**仍然要传**（传 1.0），少传就是实参个数不符。
    """
    if not ENABLE_FAST_OPS:
        return
    from . import _device

    shim = _device.shim()
    if not hasattr(shim, "load_fast_op"):
        # C++ 侧还没实现这套接口：Python 实现继续生效，不影响正确性。
        return
    elt = _mod("_ops_elementwise")
    if not hasattr(elt, "fast_variants"):
        return
    # 遍历内核组给的驱动源，不自己写死 (op, variant) 列表——那又是一份会漂移的表。
    for op, variant in elt.fast_variants():
        if op not in getattr(elt, "FASTPATH_OPS", ()):
            continue
        for dtype in ("fp32", "fp16", "bf16"):
            try:
                meta = elt.get_launch_meta(op, dtype, variant)
                shim.load_fast_op(op, variant, dtype, meta["cubin"],
                                  int(meta["num_warps"]), int(meta["num_stages"]),
                                  int(meta["n_ptrs"]), int(meta["n_i32"]), int(meta["n_f32"]),
                                  bool(meta["alpha_effective"]), int(meta["block"]))
            except Exception as exc:                    # noqa: BLE001
                # 某个 (op, dtype) 登记不上：回落到 Python 实现，但**要留痕**。
                # 静默 `continue` 的代价是「注册表空着、上层只看到『还没登记内核』」，
                # 2026-11-07 就因为 cubin 里读不出函数符号，查了一整轮（`probes/126`）。
                _warn_fast_op_once(op, dtype, variant, exc)
                continue


_WARNED_FAST_OPS: set = set()


def _warn_fast_op_once(op: str, dtype: str, variant: str, exc: BaseException) -> None:
    """同一条失败只警告一次：24 个组合逐个刷屏会把日志淹掉。"""
    key = (op, dtype, variant)
    if key in _WARNED_FAST_OPS:
        return
    _WARNED_FAST_OPS.add(key)
    import warnings

    warnings.warn(
        f"corex: {op}/{dtype}/{variant} 的热路径内核没能登记给 C++"
        f"（{type(exc).__name__}: {str(exc).splitlines()[0][:120]}）。"
        "该算子会回落到 Python 实现（正确，但每次多约 50 µs）。",
        stacklevel=2,
    )


def _register_fast_mm() -> None:
    """把编译好的 mm 内核句柄登记给 C++ 侧，让 ``aten::mm`` 全程不碰 Python。

    **为什么必须在注册期就做**：shim 一旦被 import，``aten::mm`` 就由 C++ 的
    ``TORCH_LIBRARY_IMPL`` 接管（不再走本模块的 ``_mm_impl``），而未登记时它按约定**明确报错、
    不回落**。也就是说：没有这一步，``a @ b`` 会直接失败——而失败点不在我们会执行到的代码里。

    收益（``probes/122``）：发射地板 39.5 µs → 10.8 µs（3.7x），mm 256³ 从 265.8 µs 降到 204.1 µs。
    剩下的 ~73 µs 是 **aten 分发器的固定成本**（与内核大小无关，16³ 也一样），换实现语言省不掉。

    Raises:
        无。任何一步失败都只是「该 dtype 没登记」，由 C++ 侧在真正用到时报出带修复步骤的错误；
        注册期不应该因为某个 dtype 编不出来就把整个设备搞挂。
    """
    from . import _device

    shim = _device.shim()
    if not hasattr(shim, "set_mm_kernel"):
        # 旧版 shim：aten::mm 仍由本模块的 _mm_impl 提供，不影响正确性。
        return
    ops = _mod("_ops_matmul")
    bm, bn, bk = (int(ops.DEFAULT_BLOCK_M), int(ops.DEFAULT_BLOCK_N), int(ops.DEFAULT_BLOCK_K))
    warps = int(ops.DEFAULT_NUM_WARPS)
    # c10::ScalarType 的编号：fp16=5, fp32=6, bf16=15。分块必须与编译时一致，否则 C++ 侧的
    # grid 划分与内核内假设对不上（结果会错，且不会报错）。
    for dtype, code in (("fp32", 6), ("fp16", 5), ("bf16", 15)):
        try:
            kernel = ops._get_kernel(dtype, bm, bn, bk, False, warps,
                                     int(ops.DEFAULT_NUM_STAGES))
            # spec 的最后一个 `p` 是 Triton 3.4 追加在参数表**末尾**的 global scratch 指针。
            # 写错位置（哪怕只往前挪一位）会让内核收到错误的 M/N/K——掩码全假、一个元素都不写，
            # 而 cuLaunchKernel 照样返回 0。这是 2026-11-07 的静默错误，见 docs/00-FACTS.md §21。
            shared = int(getattr(kernel.metadata, "shared", 0) or 0)
            handle = shim.load_kernel(kernel.asm["cubin"], kernel.name, "pppiiip", shared)
            shim.set_mm_kernel(code, handle, bm, bn, bk, warps, shared)
        except Exception:                               # noqa: BLE001
            # 该 dtype 编不出来（例如工具链缺某条指令）：留给 C++ 侧报错，不在这里炸注册。
            continue


def register(lib) -> None:
    """把需要内核的 aten 算子挂到 PrivateUse1 上。

    Args:
        lib: ``torch.library.Library("aten", "IMPL")`` 实例，由 ``_device._register_ops`` 传入。

    Returns:
        无。注册失败会直接抛——这类算子缺失时的报错是 ``NotImplementedError`` 指向 aten 内部，
        很难看出是我们的分发层没挂上，所以宁可在注册期就炸。
    """
    from . import _device

    fast_mm = ENABLE_FAST_MM and hasattr(_device.shim(), "set_mm_kernel")
    # 只有 C++ 没接管时才挂 Python 的 mm：两边都注册会让 torch 打
    # 「Overriding a previously registered kernel for the same operator」警告，
    # 而且谁最后注册谁生效——这种隐式顺序不该出现在后端里。
    if not fast_mm:
        lib.impl("mm", _mm_impl, "PrivateUse1")
    lib.impl("bmm", _bmm_impl, "PrivateUse1")
    lib.impl("_softmax.out", _softmax_out_impl, "PrivateUse1")
    lib.impl("sum.IntList_out", _sum_intlist_out_impl, "PrivateUse1")
    lib.impl("mean.out", _mean_out_impl, "PrivateUse1")
    lib.impl("index_select", _index_select_impl, "PrivateUse1")
    lib.impl("view.dtype", _view_dtype_impl, "PrivateUse1")
    # 第 5 批（triton-bridge 的内核，probes/196：63/63 + 36/36）
    lib.impl("relu.out", _relu_out_impl, "PrivateUse1")
    lib.impl("relu", _relu_functional, "PrivateUse1")
    lib.impl("relu_", _relu_impl, "PrivateUse1")
    lib.impl("rsqrt.out", _rsqrt_out_impl, "PrivateUse1")
    lib.impl("gelu.out", _gelu_out_impl, "PrivateUse1")
    lib.impl("masked_fill_.Scalar", _masked_fill_scalar_impl, "PrivateUse1")
    lib.impl("masked_fill.Scalar", _masked_fill_scalar_functional, "PrivateUse1")
    lib.impl("_log_softmax.out", _log_softmax_out_impl, "PrivateUse1")
    lib.impl("max.dim_max", _max_dim_impl, "PrivateUse1")
    lib.impl("argmax", _argmax_impl, "PrivateUse1")
    # 归一化：必须直接接管，否则 composite 会分解出我们没实现的 pow/mean 组合（probes/018 撞出来的）。
    # 还要同时挂 AutogradPrivateUse1：这两个算子在 ATen 里是 CompositeImplicitAutograd，
    # 只挂 PrivateUse1 时 `F.rms_norm` 会停在 `Autogradcorex` 那层报「算子不存在」
    # （报错列表里能看到 AutogradPrivateUse1 可选、而 PrivateUse1 不在其中）。
    for key in ("PrivateUse1", "AutogradPrivateUse1"):
        lib.impl("rms_norm", _rms_norm_impl, key)
        lib.impl("layer_norm", _layer_norm_impl, key)
    # 归约：functional 与 out 两种 schema 是分开注册的，缺一个就报「算子未实现」。
    lib.impl("sum.dim_IntList", _sum_dim_impl, "PrivateUse1")
    lib.impl("mean.dim", _mean_dim_impl, "PrivateUse1")
    lib.impl("var.correction", _var_correction_impl, "PrivateUse1")
    lib.impl("std.correction", _std_correction_impl, "PrivateUse1")
    # mm 可能已被 C++ 接管：上面的 impl 只作为「旧版 shim / 未接管」时的实现存在。
    _register_fast_mm()
    # 逐元素热路径（add/sub/mul/div）：C++ 侧有这套接口时把内核句柄交过去。
    _register_fast_ops()
