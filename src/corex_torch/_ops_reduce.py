"""规约类算子：行内 sum/mean/max、全量 sum、softmax、rms_norm / layernorm（无 torch 依赖）。

接口风格沿用 `_ops_elementwise`：入参是设备指针 + 维度 + dtype 字符串，结果写进调用方给的
指针；本层不 import torch、不管 stride（只处理连续、行主序的 2 维展平视图）。

为什么规约类要单独一个文件：开销结构和元素级完全不同。元素级是一个线程搬几个元素，
瓶颈在带宽；规约要在块内做 `tl.sum`/`tl.max`（warp shuffle + shared），行内还要**多遍扫描**
（softmax 至少两遍：先 max 再 exp/sum），所以口径、BLOCK 选择、误差来源都不一样，混在一个
文件里会让人误以为它们同一套成本模型。

**误差口径（和逐字节相等是两件事，别混）**：规约的累加顺序和 CPU 的「顺序累加」不同，
同一份 fp32 数据算出来的和**不可能**逐位相等。本层保证的是「与 CPU 参照的最大绝对误差
落在 fp32 累加顺序差异的量级内」，实测值由 `probes/180_reduce_softmax.sh` 打印并与
`corex-triton/docs/00-FACTS.md` §8/§11 里的量级对照。`softmax` 的 exp 走 `ex2.approx.f32`，
额外多一层近似，单独列。

Module: ``corex_torch``。被 ``_device`` 的 aten 实现调用；也可被用户直接调用。

Constraints:
    * **不 import torch**；只吃设备指针（``int`` / ``ctypes.c_void_p`` / 有 ``data_ptr()`` 的对象）。
    * 维度参数一律 i32：单次调用面向 ≤ 2^31 元素（8 GiB fp32）。更大要上层分块。
    * ``sum_all`` 是**两段式规约**（先让 G 个程序各出一份部分和，再归约这 G 份），
      因此模块内会缓存一块 ≤ 4 KiB 的 scratch；这是本层唯一自己持有的显存，理由见
      ``_scratch`` 的注释。其余算子都是「写进调用方指针」，不分配。
    * 不支持的 dtype → ``NotImplementedError``（``bf16`` 在厂商 llc 上困在
      ``Cannot select: bf16 = fp_round``，与 ``_ops_elementwise`` 同一个根因）。
"""

from __future__ import annotations

import ctypes
from typing import Any, Dict, Optional, Tuple

from . import _runtime, _triton
from ._ops_elementwise import dtype_info, itemsize, traffic_bytes

__all__ = [
    "sum_last", "mean_last", "max_last", "sum_all", "softmax_last",
    "log_softmax_last", "max_dim", "argmax_last",
    "row_var", "row_std", "sum_axis", "mean_axis",
    "rms_norm", "layernorm", "cache_info", "clear_cache",
]

#: 行内规约/逐行写入的块大小。1000 列的 softmax 一行的 fp32 数据是 4 KB，L1 装得下，
#: 所以「多遍扫描」实际只打一次 DRAM（见探针的 GB/s 口径说明）。
DEFAULT_BLOCK = 1024

#: ``sum_all`` 每个程序处理多少元素（决定部分和的个数 G）。1024 元素 ≈ 一个程序 1 轮，
#: G 太大地二段归约的启动开销会盖过收益。
DEFAULT_CHUNK = 1024

DEFAULT_NUM_WARPS = 4

#: 行内规约的三种算子编号（进 constexpr，运行期零分支）。
#: **内核里只能用字面量**：Triton 不允许 @jit 函数访问普通模块级全局（会抛
#: `NameError: Cannot access global variable _OP_MAX from within @jit'ed function`），
#: 要么写成 `tl.constexpr(2)`，要么用字面量。这里选字面量并在这个注释里点明对应关系，
#: 免得有人日后把 `OP == 2` 当成魔数改掉。
_OP_SUM, _OP_MEAN, _OP_MAX = 0, 1, 2        # 内核里对应 `OP == 0 / 1 / 2`

try:                                      # pragma: no cover - 依赖 triton 是否安装
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


# --------------------------------------------------------------------------- #
# 内核
# --------------------------------------------------------------------------- #

if triton is not None:

    @triton.jit
    def _row_reduce_kernel(a_ptr, out_ptr, rows, cols,
                           BLOCK: tl.constexpr, OP: tl.constexpr, OUT_DTYPE: tl.constexpr):
        """一个程序处理一行：按 BLOCK 分段扫描并向量化累加，最后做块内规约。

        为什么用「BLOCK 长向量累加」而不是把整行读进寄存器：cols 是运行期值（可能 4097），
        寄存器装不下；向量累加让每个线程持有多个部分和，段内没有依赖，最后 `tl.sum` 一次。
        这也决定了累加顺序：同一线程内是顺序累加，跨线程是规约树 —— 与 CPU 的顺序累加不同，
        所以本内核**不承诺逐位相等**。
        """
        row = tl.program_id(0)
        base = row * cols
        offs = tl.arange(0, BLOCK)
        # 2 = max，1 = mean，0 = sum（见模块顶部的 _OP_* 注释；内核里必须用字面量）
        if OP == 2:
            # 求 max 不用 `tl.full((BLOCK,), -inf)`：上游 Triton（LLVM 20）会把这种**非零向量常量**
            # 打成 `splat (float 0xFFF0000000000000)`，而厂商 llc 是 LLVM 16、解析不了这个语法
            # （`error: expected value token`，probes/180 实测）。零常量会被打成 zeroinitializer
            # 所以 `other=0.0` 没事，非零才有事。
            # 绕法：**钳位下标**（越界处重复读最后一个元素，对 max 无害）+ 初值取真实加载值。
            idx0 = tl.minimum(offs, cols - 1)
            acc = tl.load(a_ptr + base + idx0).to(tl.float32)
            for c0 in range(BLOCK, cols, BLOCK):
                idx = tl.minimum(c0 + offs, cols - 1)
                acc = tl.maximum(acc, tl.load(a_ptr + base + idx).to(tl.float32))
            r = tl.max(acc, axis=0)
        else:
            acc = tl.zeros((BLOCK,), tl.float32)
            for c0 in range(0, cols, BLOCK):
                idx = c0 + offs
                mask = idx < cols
                acc += tl.load(a_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
            r = tl.sum(acc, axis=0)
            if OP == 1:
                r = r / cols.to(tl.float32)
        tl.store(out_ptr + row, r.to(OUT_DTYPE))

    @triton.jit
    def _partial_sum_kernel(a_ptr, part_ptr, n, chunk, BLOCK: tl.constexpr):
        """第一段：第 pid 个程序把自己那段 [pid*chunk, (pid+1)*chunk) 压成一个部分和。

        ``chunk`` 是**运行期**参数而不是 constexpr，这一点是刻意的：做成 constexpr 时
        LLVM 会把 4 次迭代的循环整个展开、再把归纳变量加宽成 i64，然后吐出
        ``trunc nsw i64 … to i32`` —— ``nsw`` 是 LLVM 17 才有的语法，厂商 llc 是 16，
        直接报 ``expected type``（``probes/180`` 实测，n=4M 那档才触发，因为只有它让
        chunk 涨到 4096 > BLOCK）。运行期上界不会展开，也就没有这个变换。
        """
        pid = tl.program_id(0)
        start = pid * chunk
        offs = tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), tl.float32)
        for j in range(0, chunk, BLOCK):
            idx = start + j + offs
            mask = idx < n
            acc += tl.load(a_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        tl.store(part_ptr + pid, tl.sum(acc, axis=0))

    @triton.jit
    def _final_sum_kernel(part_ptr, out_ptr, g, BLOCK: tl.constexpr, OUT_DTYPE: tl.constexpr):
        """第二段：把 g 个部分和归约成一个。g ≤ BLOCK，所以单程序一遍扫完。"""
        offs = tl.arange(0, BLOCK)
        v = tl.load(part_ptr + offs, mask=offs < g, other=0.0)
        tl.store(out_ptr, tl.sum(v, axis=0).to(OUT_DTYPE))

    @triton.jit
    def _softmax_kernel(x_ptr, out_ptr, rows, cols,
                        BLOCK: tl.constexpr, OUT_DTYPE: tl.constexpr,
                        LOG: tl.constexpr):
        """逐行 softmax / log-softmax：三遍扫描（max → sum(exp) → 写回）。

        为什么是通用的三遍而不是「把整行留在寄存器里」：cols 运行期才知道、可能到 4097，
        寄存器放不下。行内三遍在 L1/L2 里命中（一行 4 KB 量级），DRAM 上仍只读一次，但
        **源码层面确实读了三次**，所以探针的 GB/s 口径按「读 1 写 1」算有效吞吐，同时把
        「实际发出 3× 读」写在注释里，不糊。

        ``LOG`` 是 constexpr：``aten::_log_softmax`` 与 ``aten::_softmax`` 的差别只在
        **最后一遍写什么**（``x - m - log(denom)`` vs ``exp(x-m)/denom``），前三遍
        （求行最大值、求 exp 之和）完全相同。所以这里用一个分支而不是第二个内核——
        两份内核会让「改一处忘一处」变成必然（比如以后换 online 算法就得改两次）。

        ``out_ptr`` 可以与 ``x_ptr`` 相同：第三遍对每个 lane 是「先读自己那个元素、再写回
        自己那个位置」，跨 lane 没有依赖，原地安全。
        """
        row = tl.program_id(0)
        base = row * cols
        offs = tl.arange(0, BLOCK)
        # 第一遍：行最大值。用钳位下标 + 真实初值，避开「非零向量常量」的 splat 语法问题
        # （与 `_row_reduce_kernel` 的 max 分支同因同治，见那边的注释）。
        idx0 = tl.minimum(offs, cols - 1)
        mx = tl.load(x_ptr + base + idx0).to(tl.float32)
        for c0 in range(BLOCK, cols, BLOCK):
            idx = tl.minimum(c0 + offs, cols - 1)
            mx = tl.maximum(mx, tl.load(x_ptr + base + idx).to(tl.float32))
        m = tl.max(mx, axis=0)
        # 第二遍：exp 之和。掩码处不能用 `other=-inf`（还是非零常量），改成先按 0 加载、
        # 再用 `tl.where` 把无效 lane 的贡献置 0 —— 零常量打成 zeroinitializer，llc 认。
        # 注意：无效 lane 上仍会真的算一次 exp(v-m)，v=0 时结果可能是 inf，但被 where 丢掉，
        # 不会进累加（select 不是乘法，不存在 inf*0 的 NaN）。
        ps = tl.zeros((BLOCK,), tl.float32)
        for c0 in range(0, cols, BLOCK):
            idx = c0 + offs
            mask = idx < cols
            v = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
            ps += tl.where(mask, tl.exp(v - m), 0.0)
        denom = tl.sum(ps, axis=0)
        # 第三遍：写回。log-softmax 省掉最后那次除法，改成减 log(denom)（与 aten 的
        # `x - max - log(sum(exp(x-max)))` 逐项对应）。
        log_denom = tl.log(denom)
        for c0 in range(0, cols, BLOCK):
            idx = c0 + offs
            mask = idx < cols
            v = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
            if LOG:
                y = v - m - log_denom
            else:
                y = tl.exp(v - m) / denom
            tl.store(out_ptr + base + idx, y.to(OUT_DTYPE), mask=mask)

    @triton.jit
    def _max_dim_kernel(a_ptr, val_ptr, idx_ptr, rows, cols,
                        BLOCK: tl.constexpr, OUT_DTYPE: tl.constexpr,
                        WRITE_VAL: tl.constexpr):
        """逐行 ``max(dim=1)`` / ``argmax(dim=1)``：返回最大值与**第一个**最大值的下标。

        一次扫描同时求值和小标，靠两个事实：
        1. 行最大值只需要「当前最大值 + 它出现的**首个**下标」；
        2. 块内「首个等于 cand 的下标」= ``tl.min(where(valid & (v == cand), idx, cols))``
           —— 用一个**运行期**的哨兵 ``cols``（大于任何合法下标），而不是 ``-1`` 常量，
           避免非零向量常量。
        初始化取第 0 个**真实元素**（不是 ``-inf``）：非零向量常量在 bi 后端会打成
        ``splat (float 0xFFF…)``，LLVM 16 解析不了（§7.2）。

        ``WRITE_VAL=0`` 时只写下标（argmax 用），此时 ``val_ptr`` 可以是 0（不解引用）。
        下标按 aten 语义写 **int64**、且取**第一个**最大值（不是最后一个）。
        NaN 语义未覆盖：torch 的 max 传播 NaN、argmax 返回首个 NaN 下标，这里没做。
        """
        row = tl.program_id(0)
        base = row * cols
        offs = tl.arange(0, BLOCK)
        # 初值用真实元素，best 初始 0（第 0 个元素就是「目前的第一个最大」）。
        # 三个量都必须保持**标量**语义：`cand` 容易顺手写成向量（它由 `tl.where(cond, v, m)`
        # 得到），那样 `tl.where(cand > m, ...)` 会把 `best` 也撑成 BLOCK 维向量，Triton 立刻
        # 报 "Loop-carried variable best has initial type int32 but is re-assigned to <[1024]>"。
        m = tl.load(a_ptr + base).to(tl.float32)
        best = 0
        for c0 in range(0, cols, BLOCK):
            idx = c0 + offs
            valid = idx < cols
            v = tl.load(a_ptr + base + tl.minimum(idx, cols - 1)).to(tl.float32)
            # 本块（连同已有的 m）里的最大值：标量
            cand = tl.max(tl.where(valid, v, m), axis=0)
            # 块内第一个等于 cand 的下标；用运行期哨兵 cols（大于任何合法下标）
            i_first = tl.min(tl.where(valid & (v == cand), idx, cols), axis=0)
            best = tl.where(cand > m, i_first, best)
            m = cand
        if WRITE_VAL:
            tl.store(val_ptr + row, m.to(OUT_DTYPE))
        tl.store(idx_ptr + row, best.to(tl.int64))

    @triton.jit
    def _row_var_kernel(a_ptr, out_ptr, rows, cols, correction,
                        BLOCK: tl.constexpr, OUT_DTYPE: tl.constexpr,
                        SQRT_OUT: tl.constexpr):
        """逐行方差/标准差：``var = sum((x-mean)^2) / (cols - correction)``。

        **两趟，不是一趟 ``sum(x^2) - n*mean^2``**。那个恒等式代数上没错，但 ``sum(x^2)`` 与
        ``n*mean^2`` 是两个**量级相近的大数**，相减时有效位被整段抵消：``x ~ 1e6`` 时
        ``sum(x^2) ~ 4e15``，fp32 在这个量级的 ulp 是 5e8，而真正的方差项只有 1e3 量级
        ——一趟公式的误差比待求量还大几个数量级（``probes/197`` §1 有实测对照）。
        两趟的代价只是把行**再读一遍**，而行在 L1/L2 里（一行 4 KB 量级），DRAM 上仍只读一次。

        为什么不是 Welford：Welford 要把 ``(mean, M2, count)`` 三元组做**跨 lane 的树形合并**
        （Triton 里得给 ``tl.reduce`` 塞自定义 combine），而两趟已经把误差压到「输出舍入」这一档；
        等真出现「行大到装不进 L1/L2、第二趟要走 DRAM」时再换 Welford 才有收益。

        ``correction``（贝塞尔校正）是 fp32 标量：1.0 = 样本方差（torch 默认），0.0 = 总体方差。
        ``SQRT_OUT=1`` 直接出标准差（同一条内核，省一条内核、省一次往返）。
        ``cols - correction <= 0`` 时结果为 inf/NaN，与 torch 相同（不特判）。
        """
        row = tl.program_id(0)
        base = row * cols
        offs = tl.arange(0, BLOCK)
        n_f = cols.to(tl.float32)
        # 第一趟：和 → 均值
        s = tl.zeros((BLOCK,), tl.float32)
        for c0 in range(0, cols, BLOCK):
            idx = c0 + offs
            v = tl.load(a_ptr + base + idx, mask=idx < cols, other=0.0).to(tl.float32)
            s += v
        mean = tl.sum(s, axis=0) / n_f
        # 第二趟：平方差。**尾部 lane 必须用 where 掩掉**——它们读的是 other=0.0，
        # 若直接累加 (0-mean)^2，等于给方差白加一堆 mean^2 的赝品。
        d = tl.zeros((BLOCK,), tl.float32)
        for c0 in range(0, cols, BLOCK):
            idx = c0 + offs
            mask = idx < cols
            v = tl.load(a_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
            dv = v - mean
            d += tl.where(mask, dv * dv, 0.0)
        var = tl.sum(d, axis=0) / (n_f - correction)
        if SQRT_OUT:
            var = tl.sqrt(var)
        tl.store(out_ptr + row, var.to(OUT_DTYPE))

    @triton.jit
    def _sum_axis_kernel(a_ptr, out_ptr, outer, red, inner,
                         BLOCK: tl.constexpr, OUT_DTYPE: tl.constexpr,
                         MEAN: tl.constexpr):
        """对**单个中间维**做 sum/mean：逻辑形状 ``(outer, red, inner)``，归约 ``red``。

        布局假设（与 torch 连续张量一致）：``inner`` 最内层且**连续**，``red`` 的步长是
        ``inner``，``outer`` 的步长是 ``red*inner``。于是每个 lane 固定一个 inner 位置、跨 red
        累加即可——不需要 stride/gather 语义，访存是「跨 inner 长度步进的连续段」，天然合并。

        为什么不做「先 permute 到最后一维再归约」：那次 permute + contiguous 会多出**一次全量
        拷贝**（对激活就是多一遍 DRAM 往返）。这条内核原地读一遍就出结果。

        多维组合（同时归约两个以上非相邻维）仍由上层显式报错——那需要通用步长向量，
        是另一个量级的工作量，而模型里几乎用不到。
        """
        o = tl.program_id(0)
        blk = tl.program_id(1)
        i = (blk * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        mask = i < inner.to(tl.int64)
        acc = tl.zeros((BLOCK,), tl.float32)
        # **指针行走，不是 `r * inner`**：步长 1 的运行期循环会让 LLVM 的 indvars 把归纳变量
        # 加宽到 i64，再在每个用到它的地方 `trunc nuw nsw i64 %iv to i32` —— 那两个标记是
        # LLVM 17 才有的，厂商 llc（16）直接 `error: expected type`
        # （实测原文：`%44 = trunc nuw nsw i64 %indvars.iv to i32`）。把「第 r 行的地址」做成
        # 一个**循环携带的 i64 向量**自己往前挪，就没有归纳变量可加宽了。
        # 第三处同一族的坑（前两处见 §7.2：`splat` 非零向量常量、`trunc nsw`）。
        off = o.to(tl.int64) * red.to(tl.int64) * inner.to(tl.int64) + i
        stride = inner.to(tl.int64)
        for _ in range(0, red):
            v = tl.load(a_ptr + off, mask=mask, other=0.0).to(tl.float32)
            acc += v
            off += stride
        if MEAN:
            acc = acc / red.to(tl.float32)
        tl.store(out_ptr + o.to(tl.int64) * inner.to(tl.int64) + i, acc.to(OUT_DTYPE),
                 mask=mask)

    @triton.jit
    def _rms_norm_kernel(x_ptr, w_ptr, out_ptr, rows, cols, eps,
                         BLOCK: tl.constexpr, OUT_DTYPE: tl.constexpr):
        """RMSNorm：``out = x / sqrt(mean(x^2) + eps) * w``，两遍扫描（求平方和 → 写回）。

        权重 ``w`` 按**整行共享**（长度 cols 的第 0 维广播），也就是 torch 里
        ``F.rms_norm(x, (cols,), w, eps)`` 的形状；不是逐行的二维权重。
        """
        row = tl.program_id(0)
        base = row * cols
        offs = tl.arange(0, BLOCK)
        ss = tl.zeros((BLOCK,), tl.float32)
        for c0 in range(0, cols, BLOCK):
            idx = c0 + offs
            v = tl.load(x_ptr + base + idx, mask=idx < cols, other=0.0).to(tl.float32)
            ss += v * v
        ms = tl.sum(ss, axis=0) / cols.to(tl.float32)
        rstd = 1.0 / tl.sqrt(ms + eps)
        for c0 in range(0, cols, BLOCK):
            idx = c0 + offs
            mask = idx < cols
            v = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + idx, mask=mask, other=0.0).to(tl.float32)
            tl.store(out_ptr + base + idx, (v * rstd * w).to(OUT_DTYPE), mask=mask)

    @triton.jit
    def _layernorm_kernel(x_ptr, w_ptr, out_ptr, rows, cols, eps,
                          BLOCK: tl.constexpr, OUT_DTYPE: tl.constexpr):
        """LayerNorm：``out = (x - mean) / sqrt(var + eps) * w``。

        一遍里同时累加 ``sum`` 与 ``sumsq``（而不是两遍分别求），代价是「大均值、小方差」时
        有相消误差（``var = E[x^2] - E[x]^2`` 的形式最坏）。本仓库的测试值均值接近 0，
        实测误差见探针；如果将来要喂「均值远大于标准差」的激活，应改成两遍算法。
        """
        row = tl.program_id(0)
        base = row * cols
        offs = tl.arange(0, BLOCK)
        s1 = tl.zeros((BLOCK,), tl.float32)
        s2 = tl.zeros((BLOCK,), tl.float32)
        for c0 in range(0, cols, BLOCK):
            idx = c0 + offs
            v = tl.load(x_ptr + base + idx, mask=idx < cols, other=0.0).to(tl.float32)
            s1 += v
            s2 += v * v
        n_f = cols.to(tl.float32)
        mean = tl.sum(s1, axis=0) / n_f
        var = tl.sum(s2, axis=0) / n_f - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)
        for c0 in range(0, cols, BLOCK):
            idx = c0 + offs
            mask = idx < cols
            v = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + idx, mask=mask, other=0.0).to(tl.float32)
            tl.store(out_ptr + base + idx, ((v - mean) * rstd * w).to(OUT_DTYPE), mask=mask)


# --------------------------------------------------------------------------- #
# 缓存与 scratch
# --------------------------------------------------------------------------- #

#: ``(内核名, dtype, op, BLOCK, num_warps, CHUNK) → CompiledKernel``。
_KERNEL_CACHE: Dict[Tuple[Any, ...], Any] = {}

#: 编译失败缓存（同一配置不必每次重试 0.4 s）。
_FAILED_CACHE: Dict[Tuple[Any, ...], BaseException] = {}

#: 两段式规约的部分和缓冲。**本层唯一自己持有的显存**：``sum_all`` 的接口没有 scratch 参数，
#: 而单程序扫全量会把带宽压到 1/16（一个 SM）。所以按「需要的最大 G」缓存一块，
#: 上限 ``_MAX_PARTS`` 个 fp32（4 KiB），进程内只增不减。
_SCRATCH: Dict[str, Any] = {"ptr": None, "parts": 0}

#: 部分和个数的上限：再多就说明 n 很大，也不该靠单程序归约了。
_MAX_PARTS = 1024


def _scratch(parts: int) -> int:
    """保证 scratch 至少能放 ``parts`` 个 fp32，返回设备地址。"""
    want = min(int(parts), _MAX_PARTS)
    if _SCRATCH["ptr"] is None or _SCRATCH["parts"] < want:
        if _SCRATCH["ptr"] is not None:
            _runtime.free(_SCRATCH["ptr"])
        _SCRATCH["ptr"] = _runtime.malloc(want * 4)
        _SCRATCH["parts"] = want
    return int(_SCRATCH["ptr"].value or 0)


def _compile(kernel_fn, signature: Dict[str, str], constexprs: Dict[str, Any],
             dtype: Optional[str], num_warps: int, what: str):
    """编译并对错误分类（与 ``_ops_elementwise`` 同一套规则）。

    Raises:
        NotImplementedError: ``corex-triton`` 的改写表缺该 dtype 的 PTX 模板（16 位/bf16
            的典型症状），或该 dtype 已知编不过。
        RuntimeError: 其它编译失败——那是真 bug，别伪装成「不支持」。
    """
    try:
        return _triton.compile_kernel(kernel_fn, signature, constexprs, num_warps=num_warps)
    except Exception as exc:                            # noqa: BLE001
        detail = f"{dtype} 的 {what} 内核编译失败：{exc}"
        is_port_error = type(exc).__name__.endswith("PortError") or "未登记" in str(exc)
        # bf16 现在是正常支持的 dtype：``_bf16.rewrite_bf16_conversions`` 覆盖标量与向量两种形式的
        # fpext/fptrunc（``probes/194`` 验出来的 `@llvm.bi.bfloat2float.v<N>bf16`）。所以 bf16 编不动
        # 就是真 bug，原样抛，别伪装成「不支持」。
        if is_port_error:
            raise NotImplementedError(detail) from exc
        raise RuntimeError(detail) from exc


def _get_kernel(name: str, dtype: Optional[str], op: Optional[int], block: int,
                num_warps: int, chunk: Optional[int] = None):
    """取（必要时编译）一个内核。失败结果同样缓存。"""
    key = (name, dtype, op, int(block), int(num_warps), chunk)
    got = _KERNEL_CACHE.get(key)
    if got is not None:
        return got
    err = _FAILED_CACHE.get(key)
    if err is not None:
        raise err
    try:
        kernel = _build_kernel(name, dtype, op, block, num_warps, chunk)
    except BaseException as exc:                        # noqa: BLE001
        _FAILED_CACHE[key] = exc
        raise
    _KERNEL_CACHE[key] = kernel
    return kernel


def _build_kernel(name: str, dtype: Optional[str], op: Optional[int], block: int,
                  num_warps: int, chunk: Optional[int]):
    """真正编译的那一半。"""
    out_dt = getattr(tl, dtype_info(dtype)[2])
    if name == "row_reduce":
        ptr_ty = dtype_info(dtype)[0]
        sig = {"a_ptr": ptr_ty, "out_ptr": ptr_ty, "rows": "i32", "cols": "i32",
               "BLOCK": "constexpr", "OP": "constexpr", "OUT_DTYPE": "constexpr"}
        cexpr = {"BLOCK": int(block), "OP": int(op), "OUT_DTYPE": out_dt}
        return _compile(_row_reduce_kernel, sig, cexpr, dtype, num_warps, f"row_reduce(op={op})")
    if name == "partial_sum":
        ptr_ty = dtype_info(dtype)[0]
        sig = {"a_ptr": ptr_ty, "part_ptr": "*fp32", "n": "i32", "chunk": "i32",
               "BLOCK": "constexpr"}
        cexpr = {"BLOCK": int(block)}
        return _compile(_partial_sum_kernel, sig, cexpr, dtype, num_warps, "partial_sum")
    if name == "final_sum":
        sig = {"part_ptr": "*fp32", "out_ptr": "*fp32", "g": "i32",
               "BLOCK": "constexpr", "OUT_DTYPE": "constexpr"}
        cexpr = {"BLOCK": int(block), "OUT_DTYPE": out_dt}
        return _compile(_final_sum_kernel, sig, cexpr, dtype, num_warps, "final_sum")
    if name == "softmax":
        ptr_ty = dtype_info(dtype)[0]
        sig = {"x_ptr": ptr_ty, "out_ptr": ptr_ty, "rows": "i32", "cols": "i32",
               "BLOCK": "constexpr", "OUT_DTYPE": "constexpr", "LOG": "constexpr"}
        # ``op`` 槽位借来区分 softmax(0) 与 log-softmax(1)：两者共用同一条内核，只有最后
        # 一遍写回不同（见 `_softmax_kernel` 的 docstring）。
        cexpr = {"BLOCK": int(block), "OUT_DTYPE": out_dt, "LOG": bool(op)}
        return _compile(_softmax_kernel, sig, cexpr, dtype, num_warps,
                        f"softmax(log={bool(op)})")
    if name == "row_var":
        ptr_ty = dtype_info(dtype)[0]
        sig = {"a_ptr": ptr_ty, "out_ptr": ptr_ty, "rows": "i32", "cols": "i32",
               "correction": "fp32",
               "BLOCK": "constexpr", "OUT_DTYPE": "constexpr", "SQRT_OUT": "constexpr"}
        # ``op`` 槽位：0 = 方差，1 = 标准差（同一条内核，只有最后一步 sqrt 不同）
        cexpr = {"BLOCK": int(block), "OUT_DTYPE": out_dt, "SQRT_OUT": bool(op)}
        return _compile(_row_var_kernel, sig, cexpr, dtype, num_warps,
                        f"row_var(sqrt={bool(op)})")
    if name == "sum_axis":
        ptr_ty = dtype_info(dtype)[0]
        sig = {"a_ptr": ptr_ty, "out_ptr": ptr_ty, "outer": "i32", "red": "i32",
               "inner": "i32", "BLOCK": "constexpr", "OUT_DTYPE": "constexpr",
               "MEAN": "constexpr"}
        # ``op`` 槽位：0 = sum，1 = mean
        cexpr = {"BLOCK": int(block), "OUT_DTYPE": out_dt, "MEAN": bool(op)}
        return _compile(_sum_axis_kernel, sig, cexpr, dtype, num_warps,
                        f"sum_axis(mean={bool(op)})")
    if name == "max_dim":
        ptr_ty = dtype_info(dtype)[0]
        sig = {"a_ptr": ptr_ty, "val_ptr": ptr_ty, "idx_ptr": "*i64",
               "rows": "i32", "cols": "i32",
               "BLOCK": "constexpr", "OUT_DTYPE": "constexpr", "WRITE_VAL": "constexpr"}
        cexpr = {"BLOCK": int(block), "OUT_DTYPE": out_dt, "WRITE_VAL": bool(op)}
        return _compile(_max_dim_kernel, sig, cexpr, dtype, num_warps,
                        f"max_dim(write_val={bool(op)})")
    if name in ("rms_norm", "layernorm"):
        ptr_ty = dtype_info(dtype)[0]
        sig = {"x_ptr": ptr_ty, "w_ptr": ptr_ty, "out_ptr": ptr_ty,
               "rows": "i32", "cols": "i32", "eps": "fp32",
               "BLOCK": "constexpr", "OUT_DTYPE": "constexpr"}
        cexpr = {"BLOCK": int(block), "OUT_DTYPE": out_dt}
        fn = _rms_norm_kernel if name == "rms_norm" else _layernorm_kernel
        return _compile(fn, sig, cexpr, dtype, num_warps, name)
    raise ValueError(f"未知内核 {name!r}")


# --------------------------------------------------------------------------- #
# 对外算子
# --------------------------------------------------------------------------- #

def _row_op(op: int, a_ptr, out_ptr, rows: int, cols: int, dtype: str,
            block: Optional[int], num_warps: Optional[int], stream: Optional[int]) -> None:
    dtype_info(dtype)
    if rows < 0 or cols < 0:
        raise ValueError(f"rows/cols 不能为负：rows={rows} cols={cols}")
    if rows == 0 or cols == 0:
        return                                  # 没有元素就没有输出，避免 0 块网格
    block = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel("row_reduce", dtype, op, block,
                         int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, (rows,), a_ptr, out_ptr, int(rows), int(cols), stream=stream)


def sum_last(a_ptr, out_ptr, rows: int, cols: int, dtype: str, *,
             block: Optional[int] = None, num_warps: Optional[int] = None,
             stream: Optional[int] = None) -> None:
    """``out[r] = sum(a[r, :])``（行主序 rows×cols，按行规约）。

    误差口径：与 CPU 顺序累加的**最大绝对误差**在 fp32 累加顺序差异的量级内，不承诺逐位相等。
    """
    _row_op(_OP_SUM, a_ptr, out_ptr, rows, cols, dtype, block, num_warps, stream)


def mean_last(a_ptr, out_ptr, rows: int, cols: int, dtype: str, *,
              block: Optional[int] = None, num_warps: Optional[int] = None,
              stream: Optional[int] = None) -> None:
    """``out[r] = mean(a[r, :])``。"""
    _row_op(_OP_MEAN, a_ptr, out_ptr, rows, cols, dtype, block, num_warps, stream)


def max_last(a_ptr, out_ptr, rows: int, cols: int, dtype: str, *,
             block: Optional[int] = None, num_warps: Optional[int] = None,
             stream: Optional[int] = None) -> None:
    """``out[r] = max(a[r, :])``。``max`` 与累加无关，**结果是逐位精确的**。"""
    _row_op(_OP_MAX, a_ptr, out_ptr, rows, cols, dtype, block, num_warps, stream)


def sum_all(a_ptr, out_ptr, n: int, dtype: str, *,
            block: Optional[int] = None, num_warps: Optional[int] = None,
            chunk: Optional[int] = None, stream: Optional[int] = None) -> None:
    """``out[0] = sum(a[0:n])``，两段式规约（部分和 → 归约）。

    为什么两段：单程序扫全量只能吃满一个 SM（16 MP 里的 1 个），1M 元素会掉到 ~1/10 带宽。
    第一段用 ``G = cdiv(n, CHUNK)`` 个程序并行出部分和，第二段单程序归约这 G 个（G ≤ 1024）。
    """
    dtype_info(dtype)
    if n < 0:
        raise ValueError(f"n 不能为负：{n}")
    if n == 0:
        return
    block = int(block or DEFAULT_BLOCK)
    chunk = int(chunk or DEFAULT_CHUNK)
    warps = int(num_warps or DEFAULT_NUM_WARPS)
    # G 的三条约束，任何一条破了都会**静默算错**（不是崩，比崩难查）：
    #   1) G ≤ _MAX_PARTS：scratch 的大小上限；
    #   2) G ≤ BLOCK：第二段是单程序按 BLOCK 装载部分和再 tl.sum，G 超过 BLOCK 就会漏掉后面的
    #      部分和（`offs < g` 的掩码根本到不了那么大）；
    #   3) **chunk 必须是 BLOCK 的整数倍**：内核按 `for j in range(0, chunk, BLOCK)` 扫，
    #      最后一个窗口会向上取整到 BLOCK —— chunk 不是整数倍时相邻程序会读重叠的一段，
    #      等于重复累加（probes/180 实测：n=100000 时 max_abs 17.5，n=1M block=4096 时 187）。
    g = min(_MAX_PARTS, block, max(1, (n + chunk - 1) // chunk))
    chunk = (n + g - 1) // g                      # 先保证 G 个程序能覆盖 n
    chunk = ((chunk + block - 1) // block) * block  # 再向上取到 BLOCK 的整数倍（消掉重叠）
    g = max(1, (n + chunk - 1) // chunk)          # 取整后实际需要的程序数（只会更小）
    part = _scratch(g)
    k1 = _get_kernel("partial_sum", dtype, None, block, warps, None)
    _triton.launch(k1, (g,), a_ptr, part, int(n), int(chunk), stream=stream)
    k2 = _get_kernel("final_sum", dtype, None, block, warps)
    _triton.launch(k2, (1,), part, out_ptr, int(g), stream=stream)


def softmax_last(x_ptr, out_ptr, rows: int, cols: int, dtype: str, *,
                 block: Optional[int] = None, num_warps: Optional[int] = None,
                 stream: Optional[int] = None) -> None:
    """逐行 softmax：``out[r, c] = exp(x[r,c] - max_r) / sum_c exp(x[r,c] - max_r)``。

    ``out_ptr`` 可以与 ``x_ptr`` 相同（原地安全，见 ``_softmax_kernel`` 的说明）。

    精度：``tl.exp`` 落成 ``ex2.approx.f32``，所以本算子**多一层近似**（与 silu 同源）。
    与 CPU 参照的误差由探针单列，不与「逐字节相等」混在一起。
    """
    _softmax(0, x_ptr, out_ptr, rows, cols, dtype, block, num_warps, stream)


def log_softmax_last(x_ptr, out_ptr, rows: int, cols: int, dtype: str, *,
                     block: Optional[int] = None, num_warps: Optional[int] = None,
                     stream: Optional[int] = None) -> None:
    """逐行 log-softmax：``out = x - max_r - log(sum_c exp(x - max_r))``（aten 语义）。

    与 ``softmax_last`` 共用同一条内核，只有最后一遍写回不同（``LOG`` constexpr）。
    精度：``tl.log`` 走 libdevice ``__nv_logf``，``exp`` 仍是 ``ex2.approx.f32``；
    误差由探针单列。**不是** ``log(softmax(x))``——那个会先下溢到 0 再取 log，
    aten 的 ``_log_softmax`` 用的是「减最大值」的稳定形式，这里与之一致。
    """
    _softmax(1, x_ptr, out_ptr, rows, cols, dtype, block, num_warps, stream)


def _softmax(log: int, x_ptr, out_ptr, rows: int, cols: int, dtype: str,
             block: Optional[int], num_warps: Optional[int],
             stream: Optional[int]) -> None:
    dtype_info(dtype)
    if rows < 0 or cols < 0:
        raise ValueError(f"rows/cols 不能为负：rows={rows} cols={cols}")
    if rows == 0 or cols == 0:
        return
    block = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel("softmax", dtype, int(log), block,
                         int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, (rows,), x_ptr, out_ptr, int(rows), int(cols), stream=stream)


def max_dim(a_ptr, val_ptr, idx_ptr, rows: int, cols: int, dtype: str, *,
            block: Optional[int] = None, num_warps: Optional[int] = None,
            stream: Optional[int] = None) -> None:
    """``torch.max(x, dim=1)``：同时出**值**与**下标**（下标是 int64、取第一个最大值）。

    Args:
        a_ptr: 输入设备指针（rows×cols，行主序）。
        val_ptr: 行最大值的输出指针（``rows`` 个元素，dtype 同输入）。
        idx_ptr: 下标输出指针（``rows`` 个 **int64**，与 aten 的 ``max.dim`` 一致）。
        rows/cols: 形状。
        dtype: 输入/值的 dtype 字符串。
    """
    _max_dim(1, a_ptr, val_ptr, idx_ptr, rows, cols, dtype, block, num_warps, stream)


def argmax_last(a_ptr, idx_ptr, rows: int, cols: int, dtype: str, *,
                block: Optional[int] = None, num_warps: Optional[int] = None,
                stream: Optional[int] = None) -> None:
    """``torch.argmax(x, dim=1)``：只出下标（int64，第一个最大值）。

    与 ``max_dim`` 共用内核（``WRITE_VAL=0`` 时不解引用 ``val_ptr``，传 0 即可）。
    """
    _max_dim(0, a_ptr, 0, idx_ptr, rows, cols, dtype, block, num_warps, stream)


def _max_dim(write_val: int, a_ptr, val_ptr, idx_ptr, rows: int, cols: int, dtype: str,
             block: Optional[int], num_warps: Optional[int],
             stream: Optional[int]) -> None:
    dtype_info(dtype)
    if rows < 0 or cols < 0:
        raise ValueError(f"rows/cols 不能为负：rows={rows} cols={cols}")
    if rows == 0 or cols == 0:
        return
    block = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel("max_dim", dtype, int(write_val), block,
                         int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, (rows,), a_ptr, val_ptr, idx_ptr, int(rows), int(cols),
                   stream=stream)


def _norm(name: str, x_ptr, w_ptr, out_ptr, rows: int, cols: int, eps: float,
          dtype: str, block: Optional[int], num_warps: Optional[int],
          stream: Optional[int]) -> None:
    dtype_info(dtype)
    if rows < 0 or cols < 0:
        raise ValueError(f"rows/cols 不能为负：rows={rows} cols={cols}")
    if rows == 0 or cols == 0:
        return
    block = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel(name, dtype, None, block, int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, (rows,), x_ptr, w_ptr, out_ptr, int(rows), int(cols),
                   float(eps), stream=stream)


def rms_norm(x_ptr, w_ptr, out_ptr, rows: int, cols: int, eps: float, dtype: str, *,
             block: Optional[int] = None, num_warps: Optional[int] = None,
             stream: Optional[int] = None) -> None:
    """RMSNorm：``out = x / sqrt(mean(x^2) + eps) * w``，``w`` 长度 ``cols``。

    Args:
        eps: fp32 标量。传 0 会得到 inf（不对输入做特殊处理，与 aten 一致）。
    """
    _norm("rms_norm", x_ptr, w_ptr, out_ptr, rows, cols, eps, dtype, block, num_warps, stream)


def layernorm(x_ptr, w_ptr, out_ptr, rows: int, cols: int, eps: float, dtype: str, *,
              block: Optional[int] = None, num_warps: Optional[int] = None,
              stream: Optional[int] = None) -> None:
    """LayerNorm（无 bias）：``out = (x - mean) / sqrt(var + eps) * w``，``w`` 长度 ``cols``。

    方差用 ``E[x^2] - E[x]^2`` 的单遍形式；「均值远大于标准差」的输入会有相消误差，
    见 ``_layernorm_kernel`` 的说明与探针实测。
    """
    _norm("layernorm", x_ptr, w_ptr, out_ptr, rows, cols, eps, dtype, block, num_warps, stream)


# --------------------------------------------------------------------------- #
# 方差 / 中间维归约
# --------------------------------------------------------------------------- #

def row_var(a_ptr, out_ptr, rows: int, cols: int, dtype: str, correction: float, *,
            block: Optional[int] = None, num_warps: Optional[int] = None,
            stream: Optional[int] = None) -> None:
    """逐行方差（``torch.var(x, dim=-1, correction=...)``）。

    两趟算法（先均值、再平方差），**不用** ``E[x^2]-E[x]^2`` 那个一趟恒等式——理由与实测见
    ``_row_var_kernel`` 的注释和 ``probes/197`` §1。

    Args:
        rows/cols: 形状（``cols`` 是被归约的最后一维长度）。
        correction: 贝塞尔校正，**fp32 标量**。1.0 = 样本方差（torch 默认），0.0 = 总体方差。
        out_ptr: ``rows`` 个元素的输出（dtype 同输入）。
    """
    _var(0, a_ptr, out_ptr, rows, cols, dtype, correction, block, num_warps, stream)


def row_std(a_ptr, out_ptr, rows: int, cols: int, dtype: str, correction: float, *,
            block: Optional[int] = None, num_warps: Optional[int] = None,
            stream: Optional[int] = None) -> None:
    """逐行标准差 = ``sqrt(逐行方差)``。

    与 :func:`row_var` **共用同一条内核**（``SQRT_OUT`` constexpr）：多写一条内核只会多一份
    要同步维护的代码，而 ``sqrt`` 是最后一步的标量运算，分支成本为零。
    """
    _var(1, a_ptr, out_ptr, rows, cols, dtype, correction, block, num_warps, stream)


def _var(sqrt_out: int, a_ptr, out_ptr, rows: int, cols: int, dtype: str,
         correction: float, block: Optional[int], num_warps: Optional[int],
         stream: Optional[int]) -> None:
    dtype_info(dtype)
    if rows < 0 or cols < 0:
        raise ValueError(f"rows/cols 不能为负：rows={rows} cols={cols}")
    if rows == 0 or cols == 0:
        return
    blk = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel("row_var", dtype, int(sqrt_out), blk,
                         int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, (rows,), a_ptr, out_ptr, int(rows), int(cols),
                   float(correction), stream=stream)


def sum_axis(a_ptr, out_ptr, outer: int, red: int, inner: int, dtype: str, *,
             block: Optional[int] = None, num_warps: Optional[int] = None,
             stream: Optional[int] = None) -> None:
    """对**单个中间维**求和：逻辑形状 ``(outer, red, inner)`` → 输出 ``(outer, inner)``。

    ``inner`` 必须是最内层连续维、``red`` 是紧邻其左的那一维（torch 连续张量里
    ``x.sum(dim=-2)`` 就是这种形状）。上层负责把「归约某一维」翻译成这三个长度：
    被归约维左侧的所有维折叠成 ``outer``，右侧折叠成 ``inner``。

    多维组合（同时归约两个非相邻维）**不支持**，请由上层显式报错——见
    ``_sum_axis_kernel`` 的说明。
    """
    _axis_reduce(0, a_ptr, out_ptr, outer, red, inner, dtype, block, num_warps, stream)


def mean_axis(a_ptr, out_ptr, outer: int, red: int, inner: int, dtype: str, *,
              block: Optional[int] = None, num_warps: Optional[int] = None,
              stream: Optional[int] = None) -> None:
    """对单个中间维求均值（``sum_axis / red``，与 :func:`sum_axis` 共用内核）。"""
    _axis_reduce(1, a_ptr, out_ptr, outer, red, inner, dtype, block, num_warps, stream)


def _axis_reduce(mean: int, a_ptr, out_ptr, outer: int, red: int, inner: int,
                 dtype: str, block: Optional[int], num_warps: Optional[int],
                 stream: Optional[int]) -> None:
    dtype_info(dtype)
    if min(outer, red, inner) < 0:
        raise ValueError(f"outer/red/inner 不能为负：{outer}/{red}/{inner}")
    if outer == 0 or red == 0 or inner == 0:
        return                                  # red==0 时 torch 给 0（sum）/NaN（mean），不模拟
    blk = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel("sum_axis", dtype, int(mean), blk,
                         int(num_warps or DEFAULT_NUM_WARPS))
    grid = (int(outer), (int(inner) + blk - 1) // blk)
    _triton.launch(kernel, grid, a_ptr, out_ptr, int(outer), int(red), int(inner),
                   stream=stream)


# --------------------------------------------------------------------------- #
# 自省（探针用）
# --------------------------------------------------------------------------- #

def cache_info() -> Dict[str, Any]:
    """已编译内核与 scratch 的规模，用来在探针里确认「没有每次重编」。"""
    return {"kernels": len(_KERNEL_CACHE), "keys": sorted(map(str, _KERNEL_CACHE)),
            "scratch_f32": _SCRATCH["parts"]}


def clear_cache() -> None:
    """丢掉已编译内核与失败记录（不动 scratch：它还会被复用）。"""
    _KERNEL_CACHE.clear()
    _FAILED_CACHE.clear()
