"""元素级算子的第一批：add / sub / mul / div / *_scalar / fill / silu / copy（无 torch 依赖）。

为什么要单独一层、且**不 import torch**：`_device.py` 那层要把算子挂到 aten 上，但 aten 的实现
依赖 C++ shim（`docs/00-FACTS.md` §12：PrivateUse1 的分配器只能从 C++ 注册，纯 Python 拿不到
Tensor 的存储）。如果算子只写成「吃 torch 张量」的形式，那么在 shim 就位之前一行都验证不了。
所以这一层的约定是：

    入参 = 设备指针（``int`` / ``ctypes.c_void_p`` / 任何有 ``data_ptr()`` 的对象）
         + 元素数 ``n``
         + dtype 字符串（``"fp32"`` / ``"fp16"`` / ``"bf16"``）
    出参 = 调用方给的输出指针（本层不分配、不持有、不释放任何内存）

这样它现在就能在真机上单独验证（``probes/170_eltwise.sh`` / ``171_eltwise_dtype.sh``），
等 shim 好了再由 ``_device.py`` 把张量的 ``data_ptr()`` 转进来即可，一行都不用重写。

Module: ``corex_torch``。被 ``_device`` 的 aten 实现调用；也可被用户直接调用。

接口契约（``_device.py`` 按这套名字与位置参数调用，**位置不能改**）::

    add(a_ptr, b_ptr, out_ptr, n, dtype, alpha=1.0)      # out = a + alpha*b
    sub(a_ptr, b_ptr, out_ptr, n, dtype, alpha=1.0)      # out = a - alpha*b
    mul(a_ptr, b_ptr, out_ptr, n, dtype)                 # out = a*b
    div(a_ptr, b_ptr, out_ptr, n, dtype)                 # out = a/b
    add_scalar(a_ptr, out_ptr, n, dtype, scalar)         # 同理 sub_scalar/mul_scalar/div_scalar
    fill(out_ptr, n, dtype, value)
    silu(a_ptr, out_ptr, n, dtype)
    copy(src_ptr, dst_ptr, n, dtype)

Constraints:
    * **不 import torch**：整层的可验证性建立在「只要有 CoreX 运行时 + triton 就能跑」。
    * 只处理**连续内存、1 维展平**：非连续的 stride 归 `_device.py`（它只在连续时调本层）。
      所有内核都用 ``off = program_id * BLOCK + arange(BLOCK)``，不做任何 stride 计算。
    * 内核按 ``(算子, dtype, op, BLOCK, num_warps)`` 缓存：首编约 400 ms，之后每次 6–8 ms
      （``corex-triton/docs/00-FACTS.md`` §14.3），每次调用都重编会把延迟抬到不可接受。
    * dtype 不在支持表里 → ``NotImplementedError``（绝不静默降精度：降精度会让数值口径失效，
      而且这种错在推理里表现为「结果慢慢变差」，最难查）。
    * ``fp16`` 已实测全通（``probes/171``：33/33 逐字节相等），但它的前提是 corex-triton 的
      改写表认得 16 位访存模板（``b16`` / ``mov.u16``）；那一版之前 fp16 会抛 ``NotImplementedError``
      并把 ``PortError`` 的原文带出来。
    * ``bf16`` 已实测**部分不可用**：``mul`` / ``copy`` 能跑，其余算子在厂商 llc 的
      ``Cannot select: bf16 = fp_round`` 上失败（fp32 → bf16 的收窄没有可选指令）。
      按契约报 ``NotImplementedError``；要真支持得写一套「全程 bf16、不出现 fp32 中间值」的内核。
      详见 ``docs/04-TRITON-BRIDGE.md`` §6.7。

数值口径（写在这里免得每次重问）：
    * ``add/sub/mul`` 在 **fp32 里算**（``alpha`` 与 ``scalar`` 都是 fp32），存回张量 dtype 时
      显式收窄。对本仓库测试用的「精确可表示」值，它与「在 fp16 里算」逐位相同（两个 fp16
      数的和/积在 fp32 里精确，只有一次舍入）；``div`` 只在除数是 2 的幂时保证逐位相同
      （否则 fp32 再舍到 fp16 会引入第二次舍入）。
    * ``silu`` **不与 CPU 参照逐位相等**（``tl.exp`` 落成 ``ex2.approx.f32``，见
      ``corex-triton/docs/00-FACTS.md`` §11），验收用最大相对误差。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from . import _triton

# --------------------------------------------------------------------------- #
# 类型与常量
# --------------------------------------------------------------------------- #

#: dtype 字符串 → (Triton 签名里的指针类型, 每元素字节数, Triton 的 dtype 名)。
#: 键就是对外 API 的 ``dtype`` 参数取值；新增类型只在这里加一行。
DTYPES: Dict[str, Tuple[str, int, str]] = {
    "fp32": ("*fp32", 4, "float32"),
    "fp16": ("*fp16", 2, "float16"),
    "bf16": ("*bf16", 2, "bfloat16"),
}

#: 「已知编不过」的 dtype 集合。命中时按 ``NotImplementedError`` 报（契约要求），而不是当成
#: bug 抛 ``RuntimeError``。
#:
#: **bf16 已经从这里移除了**：它之前编不过是因为 Triton 发 ``fpext``/``fptrunc`` 而 bi 后端只认
#: ``llvm.bi.bfloat2float``/``llvm.bi.float2bfloat``（``Cannot select: f32 = fp_extend`` /
#: ``bf16 = fp_round``）。现在由 ``_bf16.rewrite_bf16_conversions`` 在 biir 进 llc 之前把这两条
#: 指令换掉（挂在 ``_triton.CorexTorchBackend.make_cubin`` 上），所以自然写法直接可用。
#: 见 ``probes/192_bf16_ops.sh`` 与 ``docs/04-TRITON-BRIDGE.md`` §8。
_TENTATIVE_DTYPES: frozenset = frozenset()

#: 每块的元素数。选 1024 而不是 128：这张卡上 Triton 一条都不会向量化
#: （``corex-triton/docs/00-FACTS.md`` §15.1），带宽已经贴着平台天花板，而更大的 BLOCK
#: 让每个线程拿多条独立访存指令，能把延迟藏得更深。实测扫描见 ``probes/170_eltwise.sh`` §3。
DEFAULT_BLOCK = 1024

#: 每块 warp 数。4 warp × 32 lane = 128 线程/块（本平台 warp 按 32 布局，见 ``_triton``）。
DEFAULT_NUM_WARPS = 4

#: 二元算子的编号，进内核的 constexpr（同一个内核用 constexpr 分支，运行期零分支）。
_OP_ADD, _OP_SUB, _OP_MUL, _OP_DIV = 0, 1, 2, 3


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #

def dtype_info(dtype: str) -> Tuple[str, int, str]:
    """返回 ``dtype`` 的 ``(指针类型, 字节数, Triton dtype 名)``。

    Raises:
        NotImplementedError: dtype 不在支持表里。
    """
    try:
        return DTYPES[dtype]
    except KeyError:
        raise NotImplementedError(
            f"dtype={dtype!r} 还没实现；本层支持 {sorted(DTYPES)}"
        ) from None


def itemsize(dtype: str) -> int:
    """``dtype`` 每元素字节数。"""
    return dtype_info(dtype)[1]


def traffic_bytes(n: int, dtype: str, buffers: int) -> int:
    """``buffers`` 个 ``n`` 元素缓冲的字节总数——性能口径的分母，写成显式函数避免算错。"""
    return int(n) * int(buffers) * itemsize(dtype)


def _grid(n: int, block: int) -> Tuple[int]:
    """一维网格。``_triton.launch`` 负责补齐成 3 元组（Triton 自带的 runner 不补）。"""
    return ((int(n) + int(block) - 1) // int(block),)


def _check_n(n: int) -> None:
    if n < 0:
        raise ValueError(f"n 不能为负：{n}")


# --------------------------------------------------------------------------- #
# 内核
#
# @triton.jit 要用 inspect.getsourcelines 读源码，所以内核必须是**真实文件**里的函数，
# 不能在 eval/exec 里拼出来（``probes/006`` 注释里踩过这条）。
# --------------------------------------------------------------------------- #

try:                                      # pragma: no cover - 依赖 triton 是否安装
    import triton
    import triton.language as tl
except ImportError:                       # 纯 CPU 机器上 import 本模块也应该成功
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _binary_kernel(a_ptr, b_ptr, out_ptr, n, alpha,
                       BLOCK: tl.constexpr, OP: tl.constexpr, OUT_DTYPE: tl.constexpr):
        """张量-张量二元逐元素。

        ``alpha`` 只对 add/sub 有意义（torch 的 ``out = a ± alpha*b``）；mul/div 传 1.0 忽略它。
        ``alpha`` 用 fp32 传：与 aten 的 ``Scalar`` 提升规则一致，且避免为 fp16/bf16 各写一套
        标量打包（``_triton._pack_arg`` 只认 fp32/fp64 标量）。

        ``n`` 用 i32：单次调用面向 ≤ 2^31 元素（8 GiB fp32）；更大的张量由上层分块。
        掩码语义：``tl.load`` 不给 ``other``，被掩掉的 lane 读数无定义，但它们不会被存出去
        （``tl.store`` 用同一个 mask），所以不会污染结果。
        """
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        a = tl.load(a_ptr + off, mask=mask)
        b = tl.load(b_ptr + off, mask=mask)
        if OP == 0:
            v = a + alpha * b
        elif OP == 1:
            v = a - alpha * b
        elif OP == 2:
            v = a * b
        else:
            v = a / b
        tl.store(out_ptr + off, v.to(OUT_DTYPE), mask=mask)

    @triton.jit
    def _scalar_kernel(a_ptr, out_ptr, n, scalar,
                       BLOCK: tl.constexpr, OP: tl.constexpr, OUT_DTYPE: tl.constexpr):
        """张量-标量二元逐元素：``out = a <op> scalar``。"""
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        a = tl.load(a_ptr + off, mask=mask)
        if OP == 0:
            v = a + scalar
        elif OP == 1:
            v = a - scalar
        elif OP == 2:
            v = a * scalar
        else:
            v = a / scalar
        tl.store(out_ptr + off, v.to(OUT_DTYPE), mask=mask)

    @triton.jit
    def _fill_kernel(out_ptr, n, value, BLOCK: tl.constexpr, OUT_DTYPE: tl.constexpr):
        """``out[0:n] = value``。value 是 fp32 标量，存回时收窄到张量 dtype。

        ``_device.py`` 的 ``fill_.Scalar`` 只在 ``value == 0`` 时走 ``cudaMemset``，
        非 0（尤其 fp16 的 0.5 这类精确值）全靠这条内核，所以 ``probes/171`` 单独覆盖它。
        """
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        tl.store(out_ptr + off, value.to(OUT_DTYPE), mask=mask)

    @triton.jit
    def _silu_kernel(a_ptr, out_ptr, n, BLOCK: tl.constexpr, OUT_DTYPE: tl.constexpr):
        """``out = x * sigmoid(x) = x / (1 + exp(-x))``。

        为什么按 ``x / (1 + exp(-x))`` 而不是 ``x * tl.sigmoid(x)``：后者多一次除法，
        多一次舍入；这一版只有一次除法。
        """
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        x = tl.load(a_ptr + off, mask=mask)
        xf = x.to(tl.float32)
        y = xf / (1.0 + tl.exp(-xf))
        tl.store(out_ptr + off, y.to(OUT_DTYPE), mask=mask)

    @triton.jit
    def _copy_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
        """逐元素拷贝。**只作为 stride 非连续时的兜底**。

        连续内存下必须走 ``cudaMemcpyAsync``（D2D 实测 595–649 GB/s）：Triton 逐元素内核
        只有 ~295 GB/s，而且 memcpy 走 DMA 引擎不占 SM（``docs/00-FACTS.md`` §9/§11、
        ``probes/007``）。这条内核存在的唯一理由是「布局不连续，memcpy 不能直接表达」。
        """
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        v = tl.load(src_ptr + off, mask=mask)
        tl.store(dst_ptr + off, v, mask=mask)


#: 已编译内核算缓存：``(内核名, dtype, op, BLOCK, num_warps) → CompiledKernel``。
_KERNEL_CACHE: Dict[Tuple[Any, ...], Any] = {}

#: 编译失败的缓存：``同 key → 异常``。编译失败是确定性的（同一份 IR、同一个 llc），
#: 不做这个缓存的话，fp16 这种「编译链不支持」的 dtype 每次调用都要白等一次 0.4 s 的编译。
_FAILED_CACHE: Dict[Tuple[Any, ...], BaseException] = {}


def _tl_dtype(dtype: str):
    """dtype 字符串 → Triton 的 dtype 对象（进 constexpr，参与 Triton 的缓存键）。"""
    _ptr, _size, name = dtype_info(dtype)
    return getattr(tl, name)


def _compile(kernel_fn, signature: Dict[str, str], constexprs: Dict[str, Any],
             dtype: Optional[str], num_warps: int, what: str):
    """编译并对错误分类。

    分类规则（契约要求「不支持的 dtype 显式抛 NotImplementedError」）：
        * ``corex-triton`` 的 ``rewrite.PortError``（「未登记的 PTX 操作数模板」）说明**编译器**
          还没覆盖这个 dtype 需要的指令模板（16 位访存就是这种情况，见 §6.7），属于
          「不支持」而不是「我们写错了」→ ``NotImplementedError``。
        * ``bf16`` 这类未确证类型编不过也按 ``NotImplementedError``。
        * 其余（fp32 编不过）是真 bug → 原样抛，别把真故障伪装成「不支持」。
    """
    try:
        return _triton.compile_kernel(kernel_fn, signature, constexprs, num_warps=num_warps)
    except Exception as exc:                            # noqa: BLE001 - 要按原因分类重抛
        detail = f"{dtype} 的 {what} 内核编译失败：{exc}"
        is_port_error = type(exc).__name__.endswith("PortError") or "未登记" in str(exc)
        if is_port_error:
            raise NotImplementedError(
                detail + "（这是 corex-triton 改写表缺该 dtype 的 PTX 模板，不是本层的 bug，"
                "见 docs/04-TRITON-BRIDGE.md §6.7）"
            ) from exc
        if dtype in _TENTATIVE_DTYPES:
            raise NotImplementedError(
                detail + "（该 dtype 在这条 llc 链路上未确证，见 probes/171_eltwise_dtype.sh）"
            ) from exc
        raise RuntimeError(detail) from exc


def _get_kernel(kind: str, dtype: str, op: Optional[int], block: int, num_warps: int):
    """取（必要时编译）一个内核。``kind`` 是 ``binary``/``scalar``/``fill``/``silu``/``copy``。

    Raises:
        NotImplementedError: dtype 或该 dtype 需要的 PTX 模板不在支持范围内。
        RuntimeError: 其它编译失败（真 bug）。
    """
    key = (kind, dtype, op, int(block), int(num_warps))
    got = _KERNEL_CACHE.get(key)
    if got is not None:
        return got
    cached_err = _FAILED_CACHE.get(key)
    if cached_err is not None:              # 同一配置已经失败过，别再花 0.4 s 重试
        raise cached_err
    try:
        kernel = _build_kernel(kind, dtype, op, block, num_warps)
    except BaseException as exc:            # noqa: BLE001 - 缓存后原样重抛
        _FAILED_CACHE[key] = exc
        raise
    _KERNEL_CACHE[key] = kernel
    return kernel


def _build_kernel(kind: str, dtype: str, op: Optional[int], block: int, num_warps: int):
    """真正做编译的那一半（不带缓存）。"""
    ptr_ty = dtype_info(dtype)[0]
    out_dt = _tl_dtype(dtype)
    if kind == "binary":
        sig = {"a_ptr": ptr_ty, "b_ptr": ptr_ty, "out_ptr": ptr_ty, "n": "i32",
               "alpha": "fp32", "BLOCK": "constexpr", "OP": "constexpr",
               "OUT_DTYPE": "constexpr"}
        cexpr = {"BLOCK": int(block), "OP": int(op), "OUT_DTYPE": out_dt}
        kernel = _compile(_binary_kernel, sig, cexpr, dtype, num_warps, "binary")
    elif kind == "scalar":
        sig = {"a_ptr": ptr_ty, "out_ptr": ptr_ty, "n": "i32", "scalar": "fp32",
               "BLOCK": "constexpr", "OP": "constexpr", "OUT_DTYPE": "constexpr"}
        cexpr = {"BLOCK": int(block), "OP": int(op), "OUT_DTYPE": out_dt}
        kernel = _compile(_scalar_kernel, sig, cexpr, dtype, num_warps, "scalar")
    elif kind == "fill":
        sig = {"out_ptr": ptr_ty, "n": "i32", "value": "fp32",
               "BLOCK": "constexpr", "OUT_DTYPE": "constexpr"}
        cexpr = {"BLOCK": int(block), "OUT_DTYPE": out_dt}
        kernel = _compile(_fill_kernel, sig, cexpr, dtype, num_warps, "fill")
    elif kind == "silu":
        sig = {"a_ptr": ptr_ty, "out_ptr": ptr_ty, "n": "i32",
               "BLOCK": "constexpr", "OUT_DTYPE": "constexpr"}
        cexpr = {"BLOCK": int(block), "OUT_DTYPE": out_dt}
        kernel = _compile(_silu_kernel, sig, cexpr, dtype, num_warps, "silu")
    elif kind == "copy":
        sig = {"src_ptr": ptr_ty, "dst_ptr": ptr_ty, "n": "i32", "BLOCK": "constexpr"}
        cexpr = {"BLOCK": int(block)}
        kernel = _compile(_copy_kernel, sig, cexpr, None, num_warps, "copy")
    else:
        raise ValueError(f"未知内核种类 {kind!r}")
    return kernel


# --------------------------------------------------------------------------- #
# 对外算子（位置参数顺序见文件头的接口契约）
# --------------------------------------------------------------------------- #

def _run_binary(op: int, a_ptr, b_ptr, out_ptr, n: int, dtype: str, alpha: float,
                block: Optional[int], num_warps: Optional[int],
                stream: Optional[int]) -> None:
    """二元张量-张量算子的公共实现。"""
    _check_n(n)
    dtype_info(dtype)                        # 先校验 dtype，坏参数不该走到编译
    if n == 0:
        return                               # 空张量：不启动内核（0 块网格的边界行为不值得碰）
    block = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel("binary", dtype, op, block, int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, _grid(n, block), a_ptr, b_ptr, out_ptr, int(n), float(alpha),
                   stream=stream)


def _run_scalar(op: int, a_ptr, out_ptr, n: int, dtype: str, scalar: float,
                block: Optional[int], num_warps: Optional[int],
                stream: Optional[int]) -> None:
    """二元张量-标量算子的公共实现。"""
    _check_n(n)
    dtype_info(dtype)
    if n == 0:
        return
    block = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel("scalar", dtype, op, block, int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, _grid(n, block), a_ptr, out_ptr, int(n), float(scalar),
                   stream=stream)


def add(a_ptr, b_ptr, out_ptr, n: int, dtype: str, alpha: float = 1.0, *,
        block: Optional[int] = None, num_warps: Optional[int] = None,
        stream: Optional[int] = None) -> None:
    """``out = a + alpha * b``（torch 的 ``add`` 语义）。

    Args:
        a_ptr: 被加数设备指针。
        b_ptr: 加数设备指针（张量，不是标量；标量走 :func:`add_scalar`）。
        out_ptr: 输出设备指针，可与 ``a_ptr`` 相同（原地安全）。
        n: 元素数。
        dtype: ``"fp32"`` / ``"fp16"`` / ``"bf16"``。
        alpha: 乘在 ``b`` 上的标量，默认 1.0。
        block: 每块元素数，默认 ``DEFAULT_BLOCK``。
        num_warps: 每块 warp 数，默认 ``DEFAULT_NUM_WARPS``。
        stream: 流句柄，``None`` 用默认流。
    """
    _run_binary(_OP_ADD, a_ptr, b_ptr, out_ptr, n, dtype, alpha, block, num_warps, stream)


def sub(a_ptr, b_ptr, out_ptr, n: int, dtype: str, alpha: float = 1.0, *,
        block: Optional[int] = None, num_warps: Optional[int] = None,
        stream: Optional[int] = None) -> None:
    """``out = a - alpha * b``（torch 的 ``sub`` 语义）。参数含义同 :func:`add`。"""
    _run_binary(_OP_SUB, a_ptr, b_ptr, out_ptr, n, dtype, alpha, block, num_warps, stream)


def mul(a_ptr, b_ptr, out_ptr, n: int, dtype: str, *,
        block: Optional[int] = None, num_warps: Optional[int] = None,
        stream: Optional[int] = None) -> None:
    """``out = a * b``。参数含义同 :func:`add`（没有 ``alpha``）。"""
    _run_binary(_OP_MUL, a_ptr, b_ptr, out_ptr, n, dtype, 1.0, block, num_warps, stream)


def div(a_ptr, b_ptr, out_ptr, n: int, dtype: str, *,
        block: Optional[int] = None, num_warps: Optional[int] = None,
        stream: Optional[int] = None) -> None:
    """``out = a / b``。参数含义同 :func:`add`（没有 ``alpha``）。

    不检查除零：与 aten 一致（IEEE 语义下 ``x/0`` 得 inf、``0/0`` 得 nan）。
    """
    _run_binary(_OP_DIV, a_ptr, b_ptr, out_ptr, n, dtype, 1.0, block, num_warps, stream)


def add_scalar(a_ptr, out_ptr, n: int, dtype: str, scalar: float, *,
               block: Optional[int] = None, num_warps: Optional[int] = None,
               stream: Optional[int] = None) -> None:
    """``out = a + scalar``。

    Args:
        scalar: 标量值。按 fp32 传给内核，运算在 fp32 里做，存回时收窄。
    """
    _run_scalar(_OP_ADD, a_ptr, out_ptr, n, dtype, scalar, block, num_warps, stream)


def sub_scalar(a_ptr, out_ptr, n: int, dtype: str, scalar: float, *,
               block: Optional[int] = None, num_warps: Optional[int] = None,
               stream: Optional[int] = None) -> None:
    """``out = a - scalar``。"""
    _run_scalar(_OP_SUB, a_ptr, out_ptr, n, dtype, scalar, block, num_warps, stream)


def mul_scalar(a_ptr, out_ptr, n: int, dtype: str, scalar: float, *,
               block: Optional[int] = None, num_warps: Optional[int] = None,
               stream: Optional[int] = None) -> None:
    """``out = a * scalar``。"""
    _run_scalar(_OP_MUL, a_ptr, out_ptr, n, dtype, scalar, block, num_warps, stream)


def div_scalar(a_ptr, out_ptr, n: int, dtype: str, scalar: float, *,
               block: Optional[int] = None, num_warps: Optional[int] = None,
               stream: Optional[int] = None) -> None:
    """``out = a / scalar``。"""
    _run_scalar(_OP_DIV, a_ptr, out_ptr, n, dtype, scalar, block, num_warps, stream)


def fill(out_ptr, n: int, dtype: str, value: float, *,
         block: Optional[int] = None, num_warps: Optional[int] = None,
         stream: Optional[int] = None) -> None:
    """``out[0:n] = value``。

    注意 ``_device.py`` 只在 ``value == 0`` 时走 ``cudaMemset``，非 0 值（含 fp16 的 0.5
    这类精确值）全部走这条内核，所以 fp16 的正确性由 ``probes/171`` 单独覆盖。
    """
    _check_n(n)
    dtype_info(dtype)
    if n == 0:
        return
    block = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel("fill", dtype, None, block, int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, _grid(n, block), out_ptr, int(n), float(value), stream=stream)


def silu(a_ptr, out_ptr, n: int, dtype: str, *,
         block: Optional[int] = None, num_warps: Optional[int] = None,
         stream: Optional[int] = None) -> None:
    """``out = x * sigmoid(x)``。

    精度口径：**不与 CPU 参照逐位相等**（``tl.exp`` 是 ``ex2.approx.f32``），
    验收用最大相对误差，见 ``probes/170_eltwise.sh`` §2。
    """
    _check_n(n)
    dtype_info(dtype)
    if n == 0:
        return
    block = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel("silu", dtype, None, block, int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, _grid(n, block), a_ptr, out_ptr, int(n), stream=stream)


def copy(src_ptr, dst_ptr, n: int, dtype: str, *,
         block: Optional[int] = None, num_warps: Optional[int] = None,
         stream: Optional[int] = None) -> None:
    """逐元素拷贝 ``dst = src``。**连续内存不要用这条**。

    连续情形应由 ``_device.py`` 走 ``_runtime.memcpy(..., MEMCPY_DEVICE_TO_DEVICE)``：
    D2D 实测 595–649 GB/s，本内核约 295 GB/s，且 memcpy 不占 SM
    （``docs/00-FACTS.md`` §9/§11）。这里只兜底布局不连续、memcpy 表达不了的情况。
    """
    _check_n(n)
    dtype_info(dtype)
    if n == 0:
        return
    block = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel("copy", dtype, None, block, int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, _grid(n, block), src_ptr, dst_ptr, int(n), stream=stream)


# --------------------------------------------------------------------------- #
# 自省（探针用；不属于 aten 契约）
# --------------------------------------------------------------------------- #

def cache_info() -> Dict[str, Any]:
    """已编译内核的规模，用来在探针里确认「没有每次重编」。"""
    return {"kernels": len(_KERNEL_CACHE), "keys": sorted(map(str, _KERNEL_CACHE))}


def clear_cache() -> None:
    """丢掉已编译内核与失败记录（换配置重新计时、或换过 llc 之后用）。"""
    _KERNEL_CACHE.clear()
    _FAILED_CACHE.clear()


__all__ = [
    "DTYPES", "DEFAULT_BLOCK", "DEFAULT_NUM_WARPS",
    "add", "sub", "mul", "div",
    "add_scalar", "sub_scalar", "mul_scalar", "div_scalar",
    "fill", "silu", "copy",
    "dtype_info", "itemsize", "traffic_bytes", "cache_info", "clear_cache",
]
