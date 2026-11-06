"""索引类算子：``gather_rows``（embedding / index_select 用），无 torch 依赖。

形状约定：``src`` 是 ``S×cols`` 的行主序表，``idx`` 是长度 ``rows`` 的整型数组，
``out`` 是 ``rows×cols``；语义就是 ``out[r, :] = src[idx[r], :]``。

为什么单独一个文件而不是塞进 `_ops_reduce`：它是**数据搬运**，没有规约也没有算术，
唯一的坑是地址计算（``idx[r] * cols`` 的位宽与负索引），成本模型和规约类不同。

**不做边界检查**：``idx`` 越界就是越界读（和 aten 的 `index_select` 之外的裸指针操作一样）。
要安全索引由调用方先 clamp——这一层没有 second pass 去验证下标，那是性能陷阱。

Module: ``corex_torch``。被 ``_device`` 的 aten 实现调用。

Constraints:
    * **不 import torch**；入参是设备指针 + 维度 + dtype 字符串。
    * 只支持连续行主序；行内 ``cols`` 可以是任意值（尾部掩码）。
    * 下标宽度：默认 ``i32``（torch 的 ``index_select`` 在 CUDA 上用 int64，但 aten 内部会按
      需要窄化；这里把选择权交给调用方，``idx_dtype="i64"`` 也支持）。
    * ``bf16`` → ``NotImplementedError``（厂商 llc：``bf16 = fp_round`` 选不出指令）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from . import _triton
from ._ops_elementwise import dtype_info

__all__ = ["gather_rows", "cache_info", "clear_cache"]

#: 行内块大小（一行 cols 个元素分几次搬）。1024 元素 ≈ 4 KB，一次访存事务能盖住。
DEFAULT_BLOCK = 1024

DEFAULT_NUM_WARPS = 4

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
    def _gather_rows_kernel(src_ptr, idx_ptr, out_ptr, rows, cols, BLOCK: tl.constexpr):
        """``out[r, :] = src[idx[r], :]``。

        网格是二维：``program_id(0)`` 选输出行，``program_id(1)`` 选行内块。这样一行很宽时
        也能并行（一行 4097 个元素只用一个程序会浪费）。

        ``if`` 上的 ``row < rows`` 判断是为了二维网格的尾块：grid 的 y 维按 ``cols/BLOCK``
        取整不会越界，但 x 维是 ``rows``，`rows` 本身不会越界，所以这里只需要 cols 的掩码。
        地址用 i64 乘：``idx[r] * cols`` 在 ``S*cols > 2^31`` 时 i32 会溢出成负偏移。
        """
        row = tl.program_id(0)
        blk = tl.program_id(1)
        src_row = tl.load(idx_ptr + row).to(tl.int64)
        off = blk * BLOCK + tl.arange(0, BLOCK)
        mask = off < cols
        v = tl.load(src_ptr + src_row * cols.to(tl.int64) + off.to(tl.int64), mask=mask)
        tl.store(out_ptr + row.to(tl.int64) * cols.to(tl.int64) + off.to(tl.int64),
                 v, mask=mask)


def _compile_checked(sig: Dict[str, str], cexpr: Dict[str, Any], dtype: str, num_warps: int):
    """编译并对错误分类（与其它算子层同一套规则）。"""
    try:
        return _triton.compile_kernel(_gather_rows_kernel, sig, cexpr, num_warps=num_warps)
    except Exception as exc:                            # noqa: BLE001
        detail = f"{dtype} 的 gather_rows 内核编译失败：{exc}"
        is_port_error = type(exc).__name__.endswith("PortError") or "未登记" in str(exc)
        if is_port_error or dtype == "bf16":
            raise NotImplementedError(detail) from exc
        raise RuntimeError(detail) from exc


def _get_kernel(dtype: str, idx_dtype: str, block: int, num_warps: int):
    """取（必要时编译）内核。失败结果同样缓存。"""
    key = (dtype, idx_dtype, int(block), int(num_warps))
    got = _KERNEL_CACHE.get(key)
    if got is not None:
        return got
    err = _FAILED_CACHE.get(key)
    if err is not None:
        raise err
    try:
        sig = {"src_ptr": dtype_info(dtype)[0], "idx_ptr": f"*{idx_dtype}",
               "out_ptr": dtype_info(dtype)[0], "rows": "i32", "cols": "i32",
               "BLOCK": "constexpr"}
        kernel = _compile_checked(sig, {"BLOCK": int(block)}, dtype, num_warps)
    except BaseException as exc:                        # noqa: BLE001
        _FAILED_CACHE[key] = exc
        raise
    _KERNEL_CACHE[key] = kernel
    return kernel


def gather_rows(src_ptr, idx_ptr, out_ptr, rows: int, cols: int, dtype: str, *,
                idx_dtype: str = "i32", block: Optional[int] = None,
                num_warps: Optional[int] = None, stream: Optional[int] = None) -> None:
    """``out[r, :] = src[idx[r], :]``（embedding / index_select 用）。

    Args:
        src_ptr: 源表设备指针（S×cols，行主序连续）。
        idx_ptr: 下标设备指针（rows 个整数；宽度由 ``idx_dtype`` 定）。
        out_ptr: 输出设备指针（rows×cols）。
        rows/cols: 输出形状；``cols`` 任意（行内尾块用掩码）。
        dtype: 元素类型 ``"fp32"`` / ``"fp16"``；``"bf16"`` 抛 ``NotImplementedError``。
        idx_dtype: ``"i32"``（默认）/ ``"i64"``。
        block/num_warps: 行内分块与 warp 数。
        stream: 流句柄。

    Raises:
        NotImplementedError: dtype 不受支持，或内核编不出来。
    """
    dtype_info(dtype)
    if rows < 0 or cols < 0:
        raise ValueError(f"rows/cols 不能为负：rows={rows} cols={cols}")
    if rows == 0 or cols == 0:
        return
    if idx_dtype not in ("i32", "i64"):
        raise ValueError(f"idx_dtype 只支持 i32/i64，收到 {idx_dtype!r}")
    block = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel(dtype, idx_dtype, block, int(num_warps or DEFAULT_NUM_WARPS))
    grid = (int(rows), (int(cols) + block - 1) // block)
    _triton.launch(kernel, grid, src_ptr, idx_ptr, out_ptr, int(rows), int(cols), stream=stream)


def cache_info() -> Dict[str, Any]:
    """已编译内核规模（探针用来确认没有每次重编）。"""
    return {"kernels": len(_KERNEL_CACHE), "keys": sorted(map(str, _KERNEL_CACHE))}


def clear_cache() -> None:
    """丢掉已编译内核与失败记录。"""
    _KERNEL_CACHE.clear()
    _FAILED_CACHE.clear()
