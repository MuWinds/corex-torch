"""矩阵乘：``mm``（M×K · K×N）与 ``bmm``（批量），走 ``tl.dot`` 的 **FMA 展开**路径。

**这里不碰矩阵单元（TCU）**。原因写在 `corex-triton/docs/00-FACTS.md` §13：bi 后端的矩阵
运算是 `llvm.bi.matrix.mad.*` 这一族 intrinsic，要接必须落在 `TritonGPU → LLVM` 那一层
（重编 `libtriton`），靠改写 llir 文本做不到。所以本模块的 `tl.dot` 只会被降成 FMA 循环，
性能按 FMA 的天花板衡量；厂商 TCU GEMM 那 1577 GFLOP/s **只当天花板参照，不是目标**
（`docs/00-FACTS.md` §9 已经把这条口径定下来了）。

`input_precision="ieee"` 是刻意的：`corex-triton` 的 `CorexOptions.allowed_dot_input_precisions`
只有 `("ieee",)`，走 TF32 之类的近似路径在这张卡上没有对应指令。

精度口径：累加器固定 fp32（``tl.zeros(..., tl.float32)``），所以：
    * fp32 输入：与 CPU 双精度参照的差异来自「K 段累加顺序」，给 max_abs 与相对误差；
    * fp16/bf16 输入：先把元素转 fp32 再 dot（**不是** fp16 的 mma），累加也是 fp32，
      精度比 aten 的 fp16 矩阵乘更高，也因此不能拿 aten 的 fp16 结果做逐位对照。

Module: ``corex_torch``。被 ``_device`` 的 aten 实现调用。

Constraints:
    * **不 import torch**；入参是设备指针 + 维度 + dtype 字符串。
    * 维度一律 i32（单次 ≤ 2^31 元素）；批偏移内部用 i64，避免 ``B*M*K`` 溢出。
    * 只支持连续行主序（a: M×K，b: K×N，c: M×N）；stride 归 `_device.py`。
    * ``bf16`` → ``NotImplementedError``（厂商 llc：``bf16 = fp_round`` 选不出指令）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from . import _triton
from ._ops_elementwise import dtype_info

__all__ = ["mm", "bmm", "flops", "cache_info", "clear_cache"]

#: 默认分块。这三档是**扫出来的**（``probes/181_mm.sh`` §3，256^3）：
#: (16,16,16) → 189.2、(16,16,32) → 153.1、(32,32,32,w1) → 139.9、(32,32,32,w4) → 100.5、
#: (64,64,16,w4) → 141.9、(16,32,32,w2) → 114.3 GFLOP/s。
#: 结论：这张卡的 FMA 路径吃「更多并发块 + 更少每块工作」，与矩阵单元正好相反
#: （``corex-triton/docs/00-FACTS.md`` §15.2 的结论一致）。
DEFAULT_BLOCK_M = 16
DEFAULT_BLOCK_N = 16
DEFAULT_BLOCK_K = 16

#: 默认 warp 数。1 warp = 32 线程/块，配合 16×16 的 tile 正好。
DEFAULT_NUM_WARPS = 1

#: 流水级数。本平台没有 cp.async，多级流水没有硬件通路（``_triton.compile_kernel`` 的默认）。
DEFAULT_NUM_STAGES = 1

#: 内核缓存：``(dtype, M/N/K 分块, 是否批量, warps, stages) → CompiledKernel``。
_KERNEL_CACHE: Dict[Tuple[Any, ...], Any] = {}
_FAILED_CACHE: Dict[Tuple[Any, ...], BaseException] = {}

try:                                      # pragma: no cover
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   BATCH: tl.constexpr, OUT_DTYPE: tl.constexpr):
        """分块矩阵乘：``c = a @ b``（行主序），``BATCH`` 为真时按第 3 维取批。

        形参说明：
            a_ptr: M×K，b_ptr: K×N，c_ptr: M×N。批量时三者的批偏移分别是 M*K / K*N / M*N。
            M/N/K: 运行期维度。K 不是 BLOCK_K 的整数倍时，尾部用掩码补 0（对累加无影响）；
                   M/N 不是分块整数倍时，用掩码挡住越界的输出。

        为什么累加器固定 fp32：fp16 输入下 mma 累加器本来是 fp32 的，但这里走 FMA，
        累加精度取决于累加器类型；用 fp32 才能让「fp16 输入」的结果与 CPU 双精度参照对得上。
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        if BATCH:
            bid = tl.program_id(2).to(tl.int64)
            a_ptr += bid * M.to(tl.int64) * K.to(tl.int64)
            b_ptr += bid * K.to(tl.int64) * N.to(tl.int64)
            c_ptr += bid * M.to(tl.int64) * N.to(tl.int64)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * K + offs_k[None, :])
        b_ptrs = b_ptr + (offs_k[:, None] * N + offs_n[None, :])
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + offs_k
            amask = (offs_m[:, None] < M) & (kk[None, :] < K)
            bmask = (kk[:, None] < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=amask, other=0.0)
            b = tl.load(b_ptrs, mask=bmask, other=0.0)
            # 转 fp32 再 dot：fp32 上是 no-op，fp16 上则保证走 FMA 而不是 mma（mma 这条链路还没接）。
            acc += tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="ieee")
            a_ptrs += BLOCK_K
            # b 是 K×N：沿 k 走 BLOCK_K 行 = 沿内存走 BLOCK_K*N 个元素。
            # 写成 `b_ptrs += BLOCK_K` 只在 N == 1 时才对；这个 bug 在 K ≤ BLOCK_K 的用例上
            # 完全不显形（循环只跑一轮、根本不递增），是「dyadic 输入 + K=64」那一档把它抓出来的。
            b_ptrs += BLOCK_K * N
        cmask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptr + (offs_m[:, None] * N + offs_n[None, :]), acc.to(OUT_DTYPE), mask=cmask)


def flops(M: int, N: int, K: int, batch: int = 1) -> int:
    """矩阵乘的浮点运算次数（口径：乘加算 2 次）。``bmm`` 传 ``batch=B``。"""
    return 2 * int(M) * int(N) * int(K) * int(batch)


def _compile_checked(sig: Dict[str, str], cexpr: Dict[str, Any], dtype: str, num_warps: int,
                     num_stages: int):
    """编译并对错误分类（与 ``_ops_elementwise`` / ``_ops_reduce`` 同一套规则）。"""
    try:
        return _triton.compile_kernel(_mm_kernel, sig, cexpr,
                                      num_warps=num_warps, num_stages=num_stages)
    except Exception as exc:                            # noqa: BLE001
        detail = f"{dtype} 的 mm 内核编译失败：{exc}"
        is_port_error = type(exc).__name__.endswith("PortError") or "未登记" in str(exc)
        # bf16 是正常支持的 dtype（``_bf16`` 改写层已覆盖标量与向量的 fpext/fptrunc），
        # 所以 bf16 编不动就是真 bug，原样抛。
        if is_port_error:
            raise NotImplementedError(detail) from exc
        raise RuntimeError(detail) from exc


def _get_kernel(dtype: str, block_m: int, block_n: int, block_k: int, batch: bool,
                num_warps: int, num_stages: int):
    """取（必要时编译）矩阵乘内核。失败结果同样缓存，避免每次重试 0.4 s。"""
    key = (dtype, int(block_m), int(block_n), int(block_k), bool(batch),
           int(num_warps), int(num_stages))
    got = _KERNEL_CACHE.get(key)
    if got is not None:
        return got
    err = _FAILED_CACHE.get(key)
    if err is not None:
        raise err
    try:
        ptr_ty = dtype_info(dtype)[0]
        sig = {"a_ptr": ptr_ty, "b_ptr": ptr_ty, "c_ptr": ptr_ty,
               "M": "i32", "N": "i32", "K": "i32",
               "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr",
               "BATCH": "constexpr", "OUT_DTYPE": "constexpr"}
        cexpr = {"BLOCK_M": int(block_m), "BLOCK_N": int(block_n), "BLOCK_K": int(block_k),
                 "BATCH": bool(batch), "OUT_DTYPE": getattr(tl, dtype_info(dtype)[2])}
        kernel = _compile_checked(sig, cexpr, dtype, num_warps, num_stages)
    except BaseException as exc:                        # noqa: BLE001
        _FAILED_CACHE[key] = exc
        raise
    _KERNEL_CACHE[key] = kernel
    return kernel


def mm(a_ptr, b_ptr, c_ptr, M: int, N: int, K: int, dtype: str, *,
       block_m: Optional[int] = None, block_n: Optional[int] = None,
       block_k: Optional[int] = None, num_warps: Optional[int] = None,
       num_stages: Optional[int] = None, stream: Optional[int] = None) -> None:
    """``c = a @ b``，行主序：a 是 M×K，b 是 K×N，c 是 M×N。

    Args:
        a_ptr: 左矩阵设备指针（M×K）。
        b_ptr: 右矩阵设备指针（K×N）。
        c_ptr: 输出设备指针（M×N）。
        M/N/K: 维度。三者都可以不是分块大小的整数倍（尾部掩码处理）。
        dtype: ``"fp32"`` / ``"fp16"``；``"bf16"`` 会抛 ``NotImplementedError``。
        block_m/block_n/block_k: 分块大小，默认见 ``DEFAULT_BLOCK_*``。
        num_warps/num_stages: 编译参数；默认 1 warp、单级（没有 cp.async）。
        stream: 流句柄，``None`` 用默认流。

    Raises:
        NotImplementedError: dtype 不受支持（bf16），或该 dtype 的内核编不出来。
    """
    dtype_info(dtype)
    if min(M, N, K) <= 0:
        if M < 0 or N < 0 or K < 0:
            raise ValueError(f"维度不能为负：M={M} N={N} K={K}")
        return                               # 任一维为 0：没有输出（与 aten 的语义一致：空矩阵）
    bm = int(block_m or DEFAULT_BLOCK_M)
    bn = int(block_n or DEFAULT_BLOCK_N)
    bk = int(block_k or DEFAULT_BLOCK_K)
    kernel = _get_kernel(dtype, bm, bn, bk, False,
                         int(num_warps or DEFAULT_NUM_WARPS),
                         int(num_stages or DEFAULT_NUM_STAGES))
    grid = ((M + bm - 1) // bm, (N + bn - 1) // bn)
    _triton.launch(kernel, grid, a_ptr, b_ptr, c_ptr, int(M), int(N), int(K), stream=stream)


def bmm(a_ptr, b_ptr, c_ptr, B: int, M: int, N: int, K: int, dtype: str, *,
        block_m: Optional[int] = None, block_n: Optional[int] = None,
        block_k: Optional[int] = None, num_warps: Optional[int] = None,
        num_stages: Optional[int] = None, stream: Optional[int] = None) -> None:
    """批量矩阵乘：``c[i] = a[i] @ b[i]``，a 是 B×M×K，b 是 B×K×N，c 是 B×M×N（行主序连续）。

    与 :func:`mm` 用**同一个内核**（``BATCH`` 是 constexpr），只是网格多一维、指针加批偏移。
    """
    dtype_info(dtype)
    if B < 0 or M < 0 or N < 0 or K < 0:
        raise ValueError(f"维度不能为负：B={B} M={M} N={N} K={K}")
    if B == 0 or M == 0 or N == 0 or K == 0:
        return
    bm = int(block_m or DEFAULT_BLOCK_M)
    bn = int(block_n or DEFAULT_BLOCK_N)
    bk = int(block_k or DEFAULT_BLOCK_K)
    kernel = _get_kernel(dtype, bm, bn, bk, True,
                         int(num_warps or DEFAULT_NUM_WARPS),
                         int(num_stages or DEFAULT_NUM_STAGES))
    grid = ((M + bm - 1) // bm, (N + bn - 1) // bn, int(B))
    _triton.launch(kernel, grid, a_ptr, b_ptr, c_ptr, int(M), int(N), int(K), stream=stream)


def cache_info() -> Dict[str, Any]:
    """已编译内核规模（探针用来确认没有每次重编）。"""
    return {"kernels": len(_KERNEL_CACHE), "keys": sorted(map(str, _KERNEL_CACHE))}


def clear_cache() -> None:
    """丢掉已编译内核与失败记录。"""
    _KERNEL_CACHE.clear()
    _FAILED_CACHE.clear()
