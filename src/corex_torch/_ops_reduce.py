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
                        BLOCK: tl.constexpr, OUT_DTYPE: tl.constexpr):
        """逐行 softmax：三遍扫描（max → sum(exp) → 写 exp/denom）。

        为什么是通用的三遍而不是「把整行留在寄存器里」：cols 运行期才知道、可能到 4097，
        寄存器放不下。行内三遍在 L1/L2 里命中（一行 4 KB 量级），DRAM 上仍只读一次，但
        **源码层面确实读了三次**，所以探针的 GB/s 口径按「读 1 写 1」算有效吞吐，同时把
        「实际发出 3× 读」写在注释里，不糊。

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
        # 第三遍：写回
        for c0 in range(0, cols, BLOCK):
            idx = c0 + offs
            mask = idx < cols
            v = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
            tl.store(out_ptr + base + idx, (tl.exp(v - m) / denom).to(OUT_DTYPE), mask=mask)

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
        if is_port_error or dtype == "bf16":
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
               "BLOCK": "constexpr", "OUT_DTYPE": "constexpr"}
        cexpr = {"BLOCK": int(block), "OUT_DTYPE": out_dt}
        return _compile(_softmax_kernel, sig, cexpr, dtype, num_warps, "softmax")
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
    dtype_info(dtype)
    if rows < 0 or cols < 0:
        raise ValueError(f"rows/cols 不能为负：rows={rows} cols={cols}")
    if rows == 0 or cols == 0:
        return
    block = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel("softmax", dtype, None, block, int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, (rows,), x_ptr, out_ptr, int(rows), int(cols), stream=stream)


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
