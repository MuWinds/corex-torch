"""aten 分发层第二批：**只做分发与形状/stride 校验，不做数值内核**。

分工（``docs/03-TORCH28-API.md`` §11）：数值计算属于内核组（``_ops_elementwise`` /
``_ops_reduce`` / ``_ops_matmul``），本模块负责「哪些算子 PrivateUse1 上真的缺、缺了以后
怎么用已有能力拼出来」。所以这里出现的内核调用只有三种：直接复用
``_ops_elementwise`` 里已有的、走 ``_runtime`` 的 D2D 拷贝、以及 C++ shim 的建视图。

**为什么每条注册都要能回答「为什么不交给 composite」**：上游给很多算子挂了
``CompositeImplicitAutograd`` / ``CompositeExplicitAutograd``（dispatch 表里表现为
``Undefined: registered ... [default backend kernel]`` 被复制到每个后端键上）。这类算子
本来就能在 PrivateUse1 上跑，我们再 impl 一遍等于把上游语义盖住——上游一升级行为就分叉。
所以本文件只注册满足下面**任一**条件的算子，并逐条写清理由：

1. dispatch 表里**没有** PrivateUse1 键，也没有 composite 条目（真缺）；
2. 有 composite，但 composite 的下一跳正好落在缺的那个算子上（链式缺口，实测出来的）；
3. 有 composite，但那条路在本版本里**是错的**（例如把 ``alpha`` 传给没有 alpha 形参的内核）。

实测底稿：``probes/120_aten_coverage.sh`` 的基线表（42 个算子，OK 5 / FAIL 37）——
其中 19 个失败同源于一个 ``aten::as_strided``。

Module: ``torch.corex``。由 ``_device._register_ops()`` 调用（一行桥接）。

Constraints:
    * **不 import torch 于模块顶层**：torch 由调用方（``_device``）带进来，避免在
      「只想 import 包看看版本」的场景里触发重量级初始化。
    * 注册失败一条不影响其它条：整批算子有先后依赖关系，一条异常把整批带走不划算。
    * 不实现任何数值内核，也不碰 graph / cp.async（``docs/05-DRIVER-CONSTRAINTS.md`` R2）。
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from . import _meta

#: 本模块实际注册（成功）的算子名，供探针与文档对账「这个算子走的是我们还是 composite」。
REGISTERED: list[str] = []

#: 注册过程中发现的问题（可能算错 / 依赖别人修），``warnings()`` 原样返回。
_WARNINGS: list[str] = []

#: 本轮实测确认、但根因不在本模块（在 ``_device.py`` / 设备层）的条目，登记在这里让上层
#: 一眼看到。每条都带实测出处；对应问题修掉后应该删掉这条。
_STATIC_WARNINGS: list[str] = [
    "`_device._out_binary` 的标量路径未接：`t + 2.0` / `t - 2.0` 走 add.out / sub.out 时抛"
    " NotImplementedError（probes/120、121 实测）。本模块只能覆盖 aten 级 add_.Scalar，"
    "Python 的 `t.add_(2.0)` 会直接落到 add.out 上，所以这条仍会咬到人",
    "`_device._copy_from_impl` 是线性 memcpy（只校验 shape 不校验 stride）：对非连续源直接"
    " `.cpu()` 会拷错。本模块的 copy_.Tensor 已兜住 clone/contiguous 这类「先 empty_like 再"
    " copy_」的路径（probes/120 的 contiguous/clone_of_view 因此转绿），但 `_copy_from` 本身"
    "仍需按 stride 实现",
    "device module 没提供 manual_seed_all / _is_in_bad_fork：torch.manual_seed() 会**静默成功**"
    "而不改任何状态，别人会误以为 corex 上的随机序列可复现。本阶段不做设备端 RNG（见 RNG_NOTE），"
    "建议这两个函数显式抛 NotImplementedError",
]

#: 已确认**在这个版本里必然缺**、但本模块没有实现的算子，以及为什么。
#:
#: 每一条都是实测出来的「下一跳」（报错原文点名了缺的是哪个重载），不是照
#: native_functions.yaml 猜的。**已被别人覆盖的条目要删掉**，否则这张表会变成过期清单：
#:   * ``sum.IntList_out`` / ``mean.out`` / ``_softmax.out`` / ``index_select`` 已由
#:     ``_aten_kernels.py`` 接到 ``_ops_reduce`` / ``_ops_index`` 上（2026-11-07 01:09）；
#:   * ``mul.out`` / ``div.out`` 的 alpha bug 已由 ``_device._out_binary`` 修掉。
KNOWN_GAPS: list[str] = [
    # --- 规约/扫描类：还缺内核 ---
    "aten::max.dim_max",          # max.dim 的下一跳：要同时出 values 与 indices（_ops_reduce 只有 max_last 的 values）
    "aten::argmax.out",           # 需要「比较 + 记下标」的扫描内核
    "aten::_log_softmax.out",     # log_softmax.int 的下一跳；_ops_reduce 只有 softmax_last，没有 log 版
    "aten::var.correction",       # 自身缺失；两趟（mean → 平方差）或一趟 Welford
    # --- 逐元素/激活：需要内核（_ops_elementwise 目前只有 add/sub/mul/div/标量/fill/silu/copy）---
    "aten::rsqrt.out",            # 1/sqrt(x)
    "aten::relu",                 # max(x, 0)
    "aten::relu_",                # 同上（in-place）
    "aten::gelu.out",             # erf / tanh 版 gelu
    "aten::masked_fill_.Scalar",  # 按 mask 选择写
    # --- 随机数：卡上没有 RNG（本阶段不做）---
    "aten::normal_",              # torch.randn(..., device='corex') 的下一跳
    "aten::uniform_",             # torch.rand / rand_like 一类
    "aten::bernoulli_",           # dropout 一类
    # --- 语义上要单独处理，不在本批 ---
    "aten::view.dtype",           # dtype 重解释：要换 TypeMeta 建视图，shim 只按源 dtype 建
]

#: 随机数一类的处置口径（写进文档，避免有人以为「报错」是没实现好）。
RNG_NOTE = (
    "corex 上不提供设备端 RNG（KMD/兼容层没有可用的随机数通道，本阶段不做）。"
    "需要随机数时先在 CPU 上生成再 .to('corex')；卡上调用 torch.randn / rand / bernoulli "
    "会以 NotImplementedError 失败，这是预期行为，不是缺陷。"
)

#: 已经 import 过的模块引用（避免函数内反复 import 的开销与循环依赖问题）。
_MODULES: dict[str, Any] = {}


def _mod(name: str):
    """惰性取本包内的子模块（``_runtime`` / ``_device`` / ``_ops_elementwise``）。"""
    if name not in _MODULES:
        import importlib

        _MODULES[name] = importlib.import_module(f".{name}", __package__)
    return _MODULES[name]


def _shim():
    """取 C++ shim（as_strided 只在这里有）。"""
    return _mod("_device").shim()


def _dtype_name(dtype) -> str:
    """torch.dtype → 内核层的类型串。"""
    return _mod("_device")._dtype_name(dtype)


def _check_corex(tensor, op: str) -> None:
    """要求张量在 corex 上。

    Raises:
        NotImplementedError: 不在 corex 上（混设备会静默算错，宁可报错）。
    """
    device_name = _mod("_device").DEVICE_NAME
    if tensor.device.type != device_name:
        raise NotImplementedError(f"corex 的 {op} 只接 corex 张量，收到 {tensor.device}")


def _check_contiguous(tensor, op: str, label: str) -> None:
    """要求张量连续。

    为什么这批不容忍非连续：内核层是按「线性 memcpy / 线性下标」写的，非连续张量直接喂进去
    会算到别人的位置上，且不会报错。调用方先 ``.contiguous()`` 是安全且语义等价的。
    """
    if not tensor.is_contiguous():
        raise NotImplementedError(
            f"corex 的 {op} 暂不支持非连续 {label}（先 .contiguous() 再调）"
        )


# ---------------------------------------------------------------------------
# 1) 元数据 / 视图：这一组是「19 个算子的同一个根因」
# ---------------------------------------------------------------------------


def _as_strided_impl(self, size, stride, storage_offset: Optional[int] = None):
    """``aten::as_strided``：共享同一存储的视图。

    为什么不交给 composite：dispatch 表里 ``aten::as_strided`` **没有** PrivateUse1 键、
    也没有 composite 条目（``probes/120`` 基线：它就是 19 个算子的公共下一跳）。
    为什么必须落 C++：见 ``corex_shim.cpp`` 里 ``as_strided`` 的注释（共享 StorageImpl、
    独立 sizes/strides、视图持有存储所有权，这三件事只有 c10 的 ABI 能表达）。
    """
    _check_corex(self, "as_strided")
    sizes = [int(s) for s in size]
    strides = [int(s) for s in stride]
    offset = int(self.storage_offset()) if storage_offset is None else int(storage_offset)
    return _shim().as_strided(self, sizes, strides, offset)


def _view_impl(self, size):
    """``aten::view``：改形状的视图。

    为什么不交给 composite：``aten::view`` 在 2.8 的 dispatch 表里只有 CPU/Meta/... 的
    实体实现，没有 PrivateUse1、也没有 composite（``view_as`` 等反而有 composite，
    它们的下一跳就是它）。
    """
    _check_corex(self, "view")
    try:
        # 先把 -1 推出来（embedding 的 composite 第一步就是 view(-1)），再算 strides。
        sizes = _meta.infer_view_size(size, int(self.numel()))
        strides = _meta.view_strides(tuple(self.shape), tuple(self.stride()), sizes)
    except ValueError as exc:
        raise NotImplementedError(
            f"corex 的 view 只支持连续源张量：{exc}"
        ) from exc
    return _shim().as_strided(self, sizes, strides, int(self.storage_offset()))


def _reshape_alias_impl(self, size, stride):
    """``aten::_reshape_alias``：reshape 认定「可视图」时走的那条。

    为什么不交给 composite：它本身就是 composite 的终点（``aten::reshape`` 有 composite，
    内部按能不能视图二选一：能视图就走 ``_reshape_alias``，否则 copy+view）。实测基线里
    ``reshape`` 报的就是它。
    """
    _check_corex(self, "_reshape_alias")
    return _shim().as_strided(self, [int(s) for s in size], [int(s) for s in stride],
                              int(self.storage_offset()))


# ---------------------------------------------------------------------------
# 2) in-place 标量/张量：复用 _ops_elementwise 已有内核
# ---------------------------------------------------------------------------


def _inplace_tensor(name: str) -> Callable[..., Any]:
    """生成一个 in-place 二元张量算子的实现（``mul_`` / ``div_``）。

    为什么不交给 composite：它们的 composite 落在 ``{name}.out``，而 ``.out`` 那条路在本版本
    **报 TypeError**（``_device._out_binary`` 无条件传 ``alpha=``，而 ``ops.mul``/``ops.div``
    没有 alpha 形参，见 ``warnings()``）。另外 in-place 不需要额外分配 out 张量。

    Args:
        name: ``mul`` 或 ``div``。

    Returns:
        ``(self, other)`` 形式的实现。
    """

    def impl(self, other):
        _check_corex(self, f"{name}_")
        _check_corex(other, f"{name}_")
        if tuple(self.shape) != tuple(other.shape):
            raise NotImplementedError(
                f"corex 的 {name}_ 暂不支持广播：{tuple(self.shape)} vs {tuple(other.shape)}"
            )
        _check_contiguous(self, f"{name}_", "self")
        _check_contiguous(other, f"{name}_", "other")
        if self.dtype != other.dtype:
            raise NotImplementedError(f"corex 的 {name}_ 要求两边 dtype 相同")
        if self.numel() == 0:
            return self
        ops = _mod("_ops_elementwise")
        getattr(ops, name)(self.data_ptr(), other.data_ptr(), self.data_ptr(),
                           int(self.numel()), _dtype_name(self.dtype))
        return self

    return impl


def _inplace_scalar(name: str, has_alpha: bool) -> Callable[..., Any]:
    """生成一个 in-place 标量算子的实现（``add_``/``sub_``/``mul_``/``div_`` 的标量版）。

    为什么不交给 composite：两个原因各自独立成立——
      1) ``add_.Scalar`` 的 composite 落到 ``add.out``，而 ``_device._out_binary`` 在
         「other 不是 Tensor」时直接抛 NotImplementedError（实测基线里就是这么失败的）；
      2) ``mul_.Scalar`` / ``div_.Scalar`` 落到 ``mul.out``/``div.out``，那是带 alpha 的老问题。

    Args:
        name: 内核名前缀（``add`` / ``sub`` / ``mul`` / ``div``）。
        has_alpha: schema 里是否有 ``alpha`` 形参（只有 add/sub 的标量重载有）。

    Returns:
        ``(self, other, alpha=1)`` 形式的实现。
    """

    def impl(self, other, alpha: Any = 1):
        op = f"{name}_.Scalar"
        _check_corex(self, op)
        _check_contiguous(self, op, "self")
        if self.numel() == 0:
            return self
        ops = _mod("_ops_elementwise")
        # alpha 的语义是 other*alpha（add/sub 才有），这里折成一个标量传给内核，
        # 免得内核层为 alpha 多开一路特化。
        scale = float(alpha) if has_alpha else 1.0
        getattr(ops, f"{name}_scalar")(self.data_ptr(), self.data_ptr(), int(self.numel()),
                                       _dtype_name(self.dtype), float(other) * scale)
        return self

    return impl


# ---------------------------------------------------------------------------
# 3) 激活：silu 的 composite 终点；内核已存在，只缺分发
# ---------------------------------------------------------------------------


def _silu_out(self, *, out=None):
    """``aten::silu.out``：silu 的 composite 终点。

    为什么不交给 composite：``aten::silu`` 有 composite，但它的实现就是调 ``silu.out``，
    而后者没有 PrivateUse1 键（``probes/120``：``silu`` 报的正是 ``aten::silu.out``）。
    内核层已经有 ``_ops_elementwise.silu``，所以这里只做分发、不写新内核。
    """
    if out is None:
        raise NotImplementedError("corex 的 silu.out 要求显式 out")
    _check_corex(self, "silu.out")
    _check_corex(out, "silu.out")
    _check_contiguous(self, "silu.out", "self")
    _check_contiguous(out, "silu.out", "out")
    if tuple(out.shape) != tuple(self.shape):
        raise NotImplementedError("corex 的 silu.out 暂不支持 resize")
    if out.dtype != self.dtype:
        raise NotImplementedError("corex 的 silu.out 要求 out 与 self 同 dtype")
    if self.numel() == 0:
        return out
    ops = _mod("_ops_elementwise")
    ops.silu(self.data_ptr(), out.data_ptr(), int(self.numel()), _dtype_name(self.dtype))
    return out


# ---------------------------------------------------------------------------
# 4) 拼装：cat / stack（stack 有 composite，会自己落到 cat）
# ---------------------------------------------------------------------------


def _cat_out(tensors, dim: int = 0, *, out=None):
    """``aten::cat.out``：沿一维拼接。

    为什么不交给 composite：``aten::cat`` / ``stack`` 在 2.8 里都**有** composite，但它们的
    终点是 ``aten::cat.out``，而后者没有 PrivateUse1 键（基线里 cat 与 stack 报的都是它）。
    stack 不用单独实现——它的 composite 是 unsqueeze + cat，cat 通了它就通。

    实现方式是纯 D2D 拷贝拼装（不写内核）：按「外层块 × 一串行」切分，每段一次
    ``cudaMemcpy``。D2D 实测 595–649 GB/s（``docs/00-FACTS.md`` §11），比逐元素内核还快，
    所以拼接这一类不该进 Triton。
    """
    if out is None:
        raise NotImplementedError("corex 的 cat.out 要求显式 out")
    if len(tensors) == 0:
        raise NotImplementedError("corex 的 cat 至少需要一个输入")
    dtype_name = _dtype_name(out.dtype)
    del dtype_name  # 只用来提前校验 dtype 受支持
    shape0 = tuple(int(x) for x in tensors[0].shape)
    rank = len(shape0)
    axis = int(dim) % rank if rank else 0
    try:
        expected = _meta.check_cat_inputs([tuple(int(x) for x in t.shape) for t in tensors], axis)
    except ValueError as exc:
        raise NotImplementedError(f"corex 的 cat 输入不合法：{exc}") from exc
    if tuple(int(x) for x in out.shape) != tuple(expected):
        raise NotImplementedError(
            f"corex 的 cat.out 暂不支持 resize：out.shape={tuple(out.shape)} 期望 {tuple(expected)}"
        )
    inner = _meta.prod(tuple(int(x) for x in expected[axis + 1:]))
    for index, tensor in enumerate(tensors):
        _check_corex(tensor, "cat.out")
        if tensor.dtype != out.dtype:
            raise NotImplementedError("corex 的 cat 要求所有输入与 out 同 dtype")
        _check_contiguous(tensor, "cat.out", f"inputs[{index}]")
        # 只要求「拼接维之后的各维连续」，**并且 size(axis)==1 时不约束 stride**。
        # 为什么：torch 的连续性判定忽略长度为 1 的维——`[4, 1, 16]`、stride `(16, 64, 1)`
        # 的 `is_contiguous()` 是 True，因为 size-1 维上取什么 stride 都等价；而这一维在
        # 拼接里只贡献 1 个槽位，每行仍是连续的 16 个元素，能整段 memcpy。
        # 之前这里写「每维 stride 必须等于尾积」，把一个完全合法的张量拒了（probes/022 撞到）。
        if int(tensor.shape[axis]) > 1 and tensor.stride(axis) != inner:
            raise NotImplementedError(
                f"corex 的 cat 要求 inputs[{index}] 在 dim={axis} 之后连续"
                f"（stride={tuple(tensor.stride())}，期望 stride({axis})={inner}）"
            )
    if out.numel() == 0:
        return out

    runtime = _mod("_runtime")
    device_index = int(out.device.index or 0)
    # 硬规则：每次 raw memcpy 之前把当前设备绑到 out 所在的卡（跨设备指针会静默别名）。
    runtime.set_device(device_index)
    item = int(out.element_size())
    outer = _meta.prod(tuple(int(x) for x in expected[:axis]))
    out_row_bytes = int(expected[axis]) * inner * item
    memcpy2d = _shim().memcpy2d
    out_base = out.data_ptr()
    dim_offset = 0                      # 沿拼接维已经积累的元素数
    for tensor in tensors:
        length = int(tensor.shape[axis])
        if length == 0:
            continue
        row_bytes = length * inner * item
        if outer > 1:
            # 一条 memcpy2D 搬完这个输入的 outer 行：同步 cudaMemcpy 在本平台每次 5–13 µs
            # （probes/124 实测），所以**调用次数**比字节数贵。把 outer 条合成 1 条是这批
            # 性能问题的主解法：KV cache 的 [4,L,16] 拼接从 8 条降到 2 条。
            memcpy2d(out_base + dim_offset * inner * item, out_row_bytes,
                     tensor.data_ptr(), row_bytes, row_bytes, outer, device_index)
        else:
            runtime.memcpy(out_base + dim_offset * inner * item, tensor.data_ptr(), row_bytes,
                           runtime.MEMCPY_DEVICE_TO_DEVICE)
        dim_offset += length
    return out


# ---------------------------------------------------------------------------
# 5) copy_：非连续张量的搬运用它兜住
# ---------------------------------------------------------------------------


def _inner_run(dst, src) -> tuple[int, int]:
    """算出两侧都能当作「末尾若干维连续」的元素数，以及覆盖到的维数。

    为什么需要这个：非连续拷贝不能一次线性 memcpy（那样会按物理内存顺序而不是逻辑顺序搬），
    但也不能逐元素搬（几百万元素会被 memcpy 调用开销吃掉）。折中做法是把「末尾连续的一段」
    合成一次 memcpy，外层再按 stride 走。

    Args:
        dst: 目的张量。
        src: 源张量（形状与 dst 相同）。

    Returns:
        (run_elems, run_dims)：内层每次搬的元素数与覆盖的维数。
    """
    run = 1
    dims = 0
    dst_stride, src_stride = dst.stride(), src.stride()
    for dim in range(dst.dim() - 1, -1, -1):
        length = int(dst.shape[dim])
        if length == 1:                     # size-1 维不约束 stride，跳过但不计入 run
            dims += 1
            continue
        if dst_stride[dim] != run or src_stride[dim] != run:
            break
        run *= length
        dims += 1
    return run, dims


def _copy_impl(self, src, non_blocking: bool = False):
    """``aten::copy_.Tensor``：按**逻辑布局**把 src 搬进 self。

    为什么不交给 composite：composite 的下一跳是 ``_copy_from``，而 ``_device._copy_from_impl``
    做的是线性 ``cudaMemcpy``（只校验 shape、不校验 stride）。源张量一旦是非连续视图
    （``.t()`` / ``slice`` / ``expand`` 的结果），线性 memcpy 会把**物理顺序**的数据搬过来，
    于是 ``.t().contiguous()`` 得到的是转置前的数据——**不报错，只是算错**。
    ``probes/120`` 里 ``contiguous`` 这一行的 ``max_abs_diff=2.000e+00`` 就是它。

    这条也顺带覆盖 ``clone`` / ``.cpu()`` 这类「先 empty_like 再 copy_」的路径，
    是 ``as_strided`` 之后必须紧接着有的一块，否则「视图能用但一物化就错」。
    """
    device_name = _mod("_device").DEVICE_NAME
    _check_corex(self, "copy_")
    if src.device.type != device_name:
        # 主机 ↔ 设备这一段交给设备层：它已经处理了 pinned 暂存区与「拷贝前绑卡」的硬规则。
        from ._device import _copy_from_impl

        return _copy_from_impl(src, self)
    _check_corex(src, "copy_")
    if self.dtype != src.dtype:
        raise NotImplementedError(
            f"corex 的 copy_ 要求同 dtype：{self.dtype} <- {src.dtype}（类型转换走 to()）"
        )
    if tuple(self.shape) != tuple(src.shape):
        raise NotImplementedError(
            f"corex 的 copy_ 暂不支持广播：{tuple(self.shape)} <- {tuple(src.shape)}"
        )
    total = int(self.numel())
    if total == 0:
        return self

    runtime = _mod("_runtime")
    # 硬规则：raw memcpy 之前把当前设备绑到 self 所在的卡。
    runtime.set_device(int(self.device.index or 0))
    item = int(self.element_size())
    # 快路径 1：两边都连续 → 一次 memcpy。这是最常见的情形（clone / .cpu() / 逐元素
    # 算子的输出），必须只有一次 C 调用。
    if self.is_contiguous() and src.is_contiguous():
        runtime.memcpy(self.data_ptr(), src.data_ptr(), total * item,
                       runtime.MEMCPY_DEVICE_TO_DEVICE)
        return self

    # 把 stride / base / memcpy 都提到循环外：`self.stride()` 每次调用都要新建一个 tuple，
    # 放在逐元素循环里就是「每个元素付一次 Python 对象分配」——这是之前慢的主因之一。
    dst_stride = tuple(int(x) for x in self.stride())
    src_stride = tuple(int(x) for x in src.stride())
    sizes = tuple(int(x) for x in self.shape)
    memcpy = runtime.memcpy
    memcpy2d = _shim().memcpy2d
    kind = runtime.MEMCPY_DEVICE_TO_DEVICE
    dst_base = self.data_ptr()
    src_base = src.data_ptr()
    import itertools

    # 快路径 2：源是**广播视图**（存在 size>1 且 stride==0 的维），且这些维都在前导位置。
    # 为什么必须单独处理：`_inner_run` 一遇到 stride==0 就 break，run 退化成 1，于是变成
    # 「逐元素一次 memcpy」——`x.expand(...)` 这种每步出现好几次的形状，几千次 4 字节
    # memcpy 就是毫秒级开销。广播的前导维彼此等价，只要对每个前导下标把同一块源搬一次。
    zero_dims = [dim for dim in range(self.dim()) if src_stride[dim] == 0 and sizes[dim] > 1]
    if zero_dims and self.is_contiguous() and zero_dims == list(range(len(zero_dims))):
        block_elems = 1
        for dim in range(len(zero_dims), self.dim()):
            block_elems *= sizes[dim]
        block_bytes = block_elems * item
        for index in itertools.product(*[range(sizes[dim]) for dim in zero_dims]):
            dst_off = sum(i * dst_stride[dim] for i, dim in zip(index, zero_dims)) * item
            memcpy(dst_base + dst_off, src_base, block_bytes, kind)
        return self

    device_index = int(self.device.index or 0)

    # 路径 A：目的后缀连续（dst 最内层若干维紧挨）——最优情形：一圈外层下标 + 一条 memcpy2D。
    # `slice+contiguous`、`cat` 的相邻块都走这条。
    run, run_dims = _inner_run(self, src)
    if run_dims > 0:
        outer_dims = list(range(self.dim() - run_dims))
        if not outer_dims:                       # 两侧都连续（理论上被路径 1 拦掉了）
            memcpy(dst_base, src_base, total * item, kind)
            return self
        packed = True
        for left, right in zip(outer_dims, outer_dims[1:]):
            if (dst_stride[left] != sizes[right] * dst_stride[right]
                    or src_stride[left] != sizes[right] * src_stride[right]):
                packed = False
                break
        if packed:
            height = 1
            for dim in outer_dims:
                height *= sizes[dim]
            memcpy2d(dst_base, dst_stride[outer_dims[0]] * item,
                     src_base, src_stride[outer_dims[0]] * item,
                     run * item, height, device_index)
            return self

    # 路径 B0：源是「均匀步长视图」（每个非 size-1 维的 stride 都等于 k × 该维之后的元素数），
    # 而目的连续。这时整个拷贝等价于「带源步长 k 的 1-D 拷贝」：一条 memcpy2D 就够
    # （width=1 个元素、height=numel、src_pitch=k×item）。`x[..., 0::2]`、`x[::2]`
    # 这类步长切片走这条：32 个元素从 160 µs（逐元素）降到 ~10 µs。
    if self.is_contiguous() and total > 1:
        uniform_step = None
        ok_uniform = True
        tail = 1
        for dim in range(self.dim() - 1, -1, -1):
            if sizes[dim] > 1:
                if src_stride[dim] % tail != 0:
                    ok_uniform = False
                    break
                step = src_stride[dim] // tail
                if uniform_step is None:
                    uniform_step = step
                elif uniform_step != step:
                    ok_uniform = False
                    break
            tail *= sizes[dim]
        if ok_uniform and uniform_step is not None:
            memcpy2d(dst_base, item, src_base, uniform_step * item, item, total, device_index)
            return self

    # 路径 B：目的后缀在源上不连续（转置 / 步长切片），但**源自己有一个最内层连续维**。
    # 这条是 probes/022 那 2 ms 的正解：`[4,6,16].transpose(1,2)` 的源沿 h 维 stride=1，
    # 于是一整段 h（16 个元素）能用一条 memcpy2D 搬到目的（目的里这 16 个元素步长是 6），
    # 调用数从 numel(384) 降到 numel/16(24)。此前 `_inner_run` 只在「目的后缀」上找连续段，
    # 一转置就退化成逐元素，每元素付 ~5 µs 的 Python + 同步 memcpy 固定成本。
    src_inner = None
    for dim in range(self.dim()):
        if src_stride[dim] == 1 and sizes[dim] > 1:
            src_inner = dim
            break
    if src_inner is not None:
        run_len = sizes[src_inner]
        outer_dims = [dim for dim in range(self.dim()) if dim != src_inner]
        outer_shape = [sizes[dim] for dim in outer_dims]
        dst_steps = [dst_stride[dim] for dim in outer_dims]
        src_steps = [src_stride[dim] for dim in outer_dims]
        step = dst_stride[src_inner]
        for index in itertools.product(*[range(length) for length in outer_shape]):
            dst_offset = 0
            src_offset = 0
            for i, dst_step, src_step in zip(index, dst_steps, src_steps):
                dst_offset += i * dst_step
                src_offset += i * src_step
            memcpy2d(dst_base + dst_offset * item, step * item,
                     src_base + src_offset * item, item, item, run_len, device_index)
        return self

    # 路径 C：最坏情形——两侧都没有连续段（例如二维步长切片）。只能逐元素；保留它是为了
    # 「不静默算错」，量级很难看（probes/124 有记录），真要快需要一条 gather 内核。
    outer_dims = list(range(self.dim() - run_dims))
    outer_shape = [sizes[dim] for dim in outer_dims]
    move_bytes = run * item
    dst_steps = [dst_stride[dim] for dim in outer_dims]
    src_steps = [src_stride[dim] for dim in outer_dims]
    for index in itertools.product(*[range(length) for length in outer_shape]):
        dst_offset = 0
        src_offset = 0
        for i, dst_step, src_step in zip(index, dst_steps, src_steps):
            dst_offset += i * dst_step
            src_offset += i * src_step
        memcpy(dst_base + dst_offset * item, src_base + src_offset * item, move_bytes, kind)
    return self


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------


#: C++ 已经接管的 schema 名。注册时必须**跳过**它们——Python 注册会覆盖 C++ 注册
#: （`torch.library.Library.impl` 是后注册者生效），于是静默退回 Python 实现，
#: 性能数据全废。这个坑在 `aten::mm` 上踩过一次（那条 `Overriding ... previous kernel`
#: 警告就是它）。能力探测：老 .so 没有 `elementwise_op_names` 时这个集合为空。
_CPP_OWNED: set[str] = set()


def _safe_impl(lib, op: str, fn: Callable[..., Any]) -> None:
    """注册一条实现；失败只记一条 warning，不打断整批。

    为什么要兜住：这批算子之间有依赖顺序（比如说 ``_reshape_alias`` 必须早于某些 composite
    的下一跳），但「某一条重复注册」这种错误不该把后面的全废掉。重复注册的典型来源是
    ``_device.py`` 里已经挂过的算子，这里如实报告而不是抛出去。
    """
    if op in _CPP_OWNED:
        return                                          # C++ 已接管，不能覆盖
    try:
        lib.impl(op, fn, "PrivateUse1")
    except Exception as exc:                            # noqa: BLE001 - 注册失败要能让整批继续
        _WARNINGS.append(f"{op} 注册失败（可能上游或 _device.py 已注册）：{type(exc).__name__}: {exc}")
        return
    REGISTERED.append(op)


def register(lib) -> None:
    """把这批算子挂到 PrivateUse1 上。幂等。

    Args:
        lib: ``_device._register_ops()`` 建好的 ``torch.library.Library("aten", "IMPL")``。
            要求是**同一个** Library 对象——同一 op+key 在第二个 Library 上会撞「重复注册」。
    """
    if REGISTERED:
        return

    # C++ 接管的 schema：问 shim 要清单（能力探测，老 .so 上拿不到就照旧全注册）。
    try:
        _CPP_OWNED.update(_shim().elementwise_op_names())
    except Exception:                                   # noqa: BLE001 - 老 .so 没有这个入口
        pass

    # 1) 视图类：一个 as_strided 解锁 13 个算子，view/_reshape_alias 是它们各自的终点。
    _safe_impl(lib, "as_strided", _as_strided_impl)
    _safe_impl(lib, "view", _view_impl)
    _safe_impl(lib, "_reshape_alias", _reshape_alias_impl)

    # 2) in-place（composite 落在坏掉/缺失的 .out 上）。
    _safe_impl(lib, "mul_.Tensor", _inplace_tensor("mul"))
    _safe_impl(lib, "div_.Tensor", _inplace_tensor("div"))
    _safe_impl(lib, "add_.Scalar", _inplace_scalar("add", has_alpha=True))
    _safe_impl(lib, "sub_.Scalar", _inplace_scalar("sub", has_alpha=True))
    _safe_impl(lib, "mul_.Scalar", _inplace_scalar("mul", has_alpha=False))
    _safe_impl(lib, "div_.Scalar", _inplace_scalar("div", has_alpha=False))

    # 3) 激活：composite 的终点，内核已有。
    _safe_impl(lib, "silu.out", _silu_out)

    # 4) 拼装：cat 通了，stack 的 composite 就通了。
    _safe_impl(lib, "cat.out", _cat_out)

    # 5) 非连续搬运：composite 的 copy_ 会落到线性 memcpy 上，物化视图时静默拷错。
    _safe_impl(lib, "copy_.Tensor", _copy_impl)


def warnings() -> list[str]:
    """注册过程中发现的「可能算错 / 依赖别人修」的点，一句话一条。

    Returns:
        列表副本（调用方改它不会影响本模块的状态）。
    """
    return _STATIC_WARNINGS + list(_WARNINGS)
