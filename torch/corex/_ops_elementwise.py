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

Module: ``torch.corex``。被 ``_device`` 的 aten 实现调用；也可被用户直接调用。

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
    # i8 只用于**比较结果**（torch 的 bool 在内存里就是 1 字节 0/1），
    # 不当成普通数值 dtype 用，见 _cmp_kernel。
    "i8": ("*i8", 1, "int8"),
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
        elif OP == 3:
            v = a / scalar
        elif OP == 4:
            v = tl.maximum(a, scalar)          # clamp 的下界（OP=4）/ 上界（OP=5）就靠这两条
        else:
            v = tl.minimum(a, scalar)
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


    @triton.jit
    def _cast_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr,
                     IN_DTYPE: tl.constexpr, OUT_DTYPE: tl.constexpr):
        """逐元素 dtype 转换：``dst = src.to(OUT_DTYPE)``。

        为什么必须有这条：`_copy_from` 的快速路径是**按字节 memcpy**，只在两侧同 dtype 时成立。
        之前没有 dtype 分支，`x.float()`（fp16→fp32）就直接按目的的元素宽度读了源缓冲，
        越界读、数值全错（2026-11-07 用 probes/167 抓到：max_abs_diff 2.1）。
        """
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        v = tl.load(src_ptr + off, mask=mask)
        tl.store(dst_ptr + off, v.to(OUT_DTYPE), mask=mask)

    @triton.jit
    def _cmp_kernel(a_ptr, out_ptr, n, scalar, BLOCK: tl.constexpr, OP: tl.constexpr):
        """标量比较：``out = a <op> scalar``，结果是 i8 存 0/1（torch 的 bool 张量布局）。

        ``OP``：0 gt / 1 lt / 2 ge / 3 le / 4 eq / 5 ne。
        """
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        a = tl.load(a_ptr + off, mask=mask).to(tl.float32)
        if OP == 0:
            v = a > scalar
        elif OP == 1:
            v = a < scalar
        elif OP == 2:
            v = a >= scalar
        elif OP == 3:
            v = a <= scalar
        elif OP == 4:
            v = a == scalar
        else:
            v = a != scalar
        tl.store(out_ptr + off, v.to(tl.int8), mask=mask)

    @triton.jit
    def _where_kernel(c_ptr, a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr,
                      OUT_DTYPE: tl.constexpr):
        """``out = where(c, a, b)``：条件按 i8 读（torch 的 bool 布局）。"""
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        c = tl.load(c_ptr + off, mask=mask)
        a = tl.load(a_ptr + off, mask=mask)
        b = tl.load(b_ptr + off, mask=mask)
        v = tl.where(c != 0, a, b)
        tl.store(out_ptr + off, v.to(OUT_DTYPE), mask=mask)

    @triton.jit
    def _math_kernel(a_ptr, out_ptr, n, BLOCK: tl.constexpr, OP: tl.constexpr,
                     OUT_DTYPE: tl.constexpr):
        """一元数学算子：``out = f(x)``，``OP`` 选具体是哪个 f。

        为什么用一条内核 + constexpr 分支，而不是每个算子一条内核：这些算子的访存形状完全一样，
        差别只在中间那几行表达式；合成一条可以让 `_KERNEL_CACHE` 的键只多一个 ``OP``，
        编译产物按 constexpr 特化，性能与分开写一致。

        ``tanh`` 用 ``exp`` 手写而不是 ``tl.math.tanh``：后者落成 libdevice 调用，
        corex-triton 的改写表不一定登记（``rsqrt`` 就踩过同样的坑，见本文件 ``_rsqrt_kernel``
        的注释）。手写版只用 ``tl.exp`` 与四则运算，已在本平台验证过。
        """
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        a = tl.load(a_ptr + off, mask=mask)
        x = a.to(tl.float32)
        if OP == 0:                                  # neg
            v = -x
        elif OP == 1:                                # abs
            v = tl.abs(x)
        elif OP == 2:                                # exp
            v = tl.exp(x)
        elif OP == 3:                                # log
            v = tl.log(x)
        elif OP == 4:                                # sqrt
            v = tl.sqrt(x)
        elif OP == 5:                                # tanh
            e = tl.exp(-2.0 * tl.abs(x))
            v = (1.0 - e) / (1.0 + e)
            v = tl.where(x < 0.0, -v, v)
        elif OP == 6:                                # sigmoid
            v = 1.0 / (1.0 + tl.exp(-x))
        elif OP == 7:                                # reciprocal
            v = 1.0 / x
        elif OP == 8:                                # sign
            v = tl.where(x > 0.0, 1.0, tl.where(x < 0.0, -1.0, 0.0))
        elif OP == 9:                                # floor
            v = tl.math.floor(x)
        else:                                        # ceil
            v = tl.math.ceil(x)
        tl.store(out_ptr + off, v.to(OUT_DTYPE), mask=mask)

    @triton.jit
    def _relu_kernel(a_ptr, out_ptr, n, BLOCK: tl.constexpr, OUT_DTYPE: tl.constexpr):
        """``out = max(x, 0)``。

        ``0.0`` 是**零**常量（LLVM 打成 ``zeroinitializer``），不会撞 ``splat (float 0.0)``
        的语法问题（§7.2）。原地（``out_ptr == a_ptr``）安全：每个 lane 只读自己那个元素。
        """
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        v = tl.load(a_ptr + off, mask=mask)
        tl.store(out_ptr + off, tl.maximum(v, 0.0).to(OUT_DTYPE), mask=mask)

    @triton.jit
    def _rsqrt_kernel(a_ptr, out_ptr, n, BLOCK: tl.constexpr, OUT_DTYPE: tl.constexpr):
        """``out = 1/sqrt(x)``。

        **为什么是 ``1.0 / tl.sqrt(x)`` 而不是 ``tl.math.rsqrt(x)``**：后者在 Triton 3.4 里落成
        ``%59 = tail call float @llvm.nvvm.rsqrt.approx.ftz.f(float %43)``，而 corex-triton 的
        改写表里**没有登记**这个 intrinsic，直接抛 PortError（实测原文：
        ``未登记的 NVVM intrinsic 调用：%59 = tail call float @llvm.nvvm.rsqrt.approx.ftz.f``）。
        要么改 corex-triton 的模板，要么换一条已经验证过的路——``tl.sqrt`` 是本仓库
        ``rms_norm``/``layernorm`` 一直在用的（``probes/180``/``182``/``192`` 都跑通了），
        所以这里用 ``1/sqrt``：多一次 ``div.rn``、多一次舍入，换来零新增 intrinsic。

        数值口径：``sqrt.approx`` + ``div.rn`` 合计约 2 ulp fp32，**不承诺逐字节相等**，
        探针按 max_rel 判（``probes/196`` §1）。
        """
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        f = tl.load(a_ptr + off, mask=mask).to(tl.float32)
        tl.store(out_ptr + off, (1.0 / tl.sqrt(f)).to(OUT_DTYPE), mask=mask)

    @triton.jit
    def _gelu_kernel(a_ptr, out_ptr, n, BLOCK: tl.constexpr, OUT_DTYPE: tl.constexpr,
                     MODE: tl.constexpr):
        """GELU 两种形式（``MODE=0`` erf 精确版，``MODE=1`` tanh 近似版）。

        * ``MODE=0``（aten 的 ``approximate="none"``）：
          ``0.5 * x * (1 + erf(x / sqrt(2)))``；``erf`` 走 ``tl.math.erf``
          （libdevice ``__nv_erff``，``corex-triton/docs/00-FACTS.md`` §11 验过 erf 6/6）。
        * ``MODE=1``（``approximate="tanh"``）：
          ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))``。

        **为什么 tanh 用 exp 现推而不是调 libdevice**：Triton 3.4 的 ``tl.math`` 里**没有**
        ``tanh``（实测 ``hasattr(tl.math, "tanh") == False``，而 ``erf``/``rsqrt``/``log`` 都有）。
        用 ``tanh(z) = 2/(1+exp(-2z)) - 1`` 只需要已经验证过的 ``tl.exp``，不必再赌
        ``libdevice`` 模块在 jit 作用域里能否解析。代价是多两次舍入，而 tanh 近似本身与精确
        GELU 就差 ~1e-3 量级，这两次舍入被它完全盖住。
        """
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        f = tl.load(a_ptr + off, mask=mask).to(tl.float32)
        if MODE == 0:
            y = 0.5 * f * (1.0 + tl.math.erf(f * 0.7071067811865476))
        else:
            z = 0.7978845608028654 * (f + 0.044715 * f * f * f)
            t = 2.0 / (1.0 + tl.exp(-2.0 * z)) - 1.0
            y = 0.5 * f * (1.0 + t)
        tl.store(out_ptr + off, y.to(OUT_DTYPE), mask=mask)

    @triton.jit
    def _masked_fill_kernel(a_ptr, mask_ptr, out_ptr, n, value,
                            BLOCK: tl.constexpr, OUT_DTYPE: tl.constexpr):
        """``out = where(mask, value, x)``（``aten::masked_fill_.Scalar``）。

        ``mask`` 是 **bool 张量**（每元素 1 字节、值只有 0/1），所以按 ``*i8`` 读、用
        ``m != 0`` 判真——0 是零常量，安全。``value`` 与 ``fill`` 一样是 fp32 标量，
        存回时收窄到张量 dtype（bf16/fp16 的收窄由 ``_bf16`` 改写层处理）。
        原地（``out_ptr == a_ptr``）安全。
        """
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < n
        m = tl.load(mask_ptr + off, mask=mask)
        v = tl.load(a_ptr + off, mask=mask)
        tl.store(out_ptr + off, tl.where(m != 0, value.to(OUT_DTYPE), v), mask=mask)


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


def _signature(kind: str, dtype: str, op: Optional[int], block: int):
    """内核签名的**单一真源**。

    返回 ``(sig, cexpr, kernel_fn, what, check_dtype)``。``_build_kernel()`` 用它编译，
    ``fast_op()`` 用它**派生**给 C++ 的实参顺序表——两边绝不各写一份。

    为什么这点很重要：实参顺序错了不会报错，只会算错（``n`` 和 ``alpha`` 位置互换时，
    前者当成 float 解释、后者当成长度，结果是错得离谱但"跑完了"）。所以顺序表必须是
    **从签名数出来的**，而不是手写常量。
    """
    ptr_ty = dtype_info(dtype)[0]
    out_dt = _tl_dtype(dtype)
    if kind == "binary":
        sig = {"a_ptr": ptr_ty, "b_ptr": ptr_ty, "out_ptr": ptr_ty, "n": "i32",
               "alpha": "fp32", "BLOCK": "constexpr", "OP": "constexpr",
               "OUT_DTYPE": "constexpr"}
        return sig, {"BLOCK": int(block), "OP": int(op), "OUT_DTYPE": out_dt}, \
            _binary_kernel, "binary", dtype
    if kind == "scalar":
        sig = {"a_ptr": ptr_ty, "out_ptr": ptr_ty, "n": "i32", "scalar": "fp32",
               "BLOCK": "constexpr", "OP": "constexpr", "OUT_DTYPE": "constexpr"}
        return sig, {"BLOCK": int(block), "OP": int(op), "OUT_DTYPE": out_dt}, \
            _scalar_kernel, "scalar", dtype
    if kind == "fill":
        sig = {"out_ptr": ptr_ty, "n": "i32", "value": "fp32",
               "BLOCK": "constexpr", "OUT_DTYPE": "constexpr"}
        return sig, {"BLOCK": int(block), "OUT_DTYPE": out_dt}, \
            _fill_kernel, "fill", dtype
    if kind == "silu":
        sig = {"a_ptr": ptr_ty, "out_ptr": ptr_ty, "n": "i32",
               "BLOCK": "constexpr", "OUT_DTYPE": "constexpr"}
        return sig, {"BLOCK": int(block), "OUT_DTYPE": out_dt}, \
            _silu_kernel, "silu", dtype
    if kind == "copy":
        sig = {"src_ptr": ptr_ty, "dst_ptr": ptr_ty, "n": "i32", "BLOCK": "constexpr"}
        return sig, {"BLOCK": int(block)}, _copy_kernel, "copy", None
    if kind == "cast":
        # ``op`` 槽位装的是输入 dtype 的下标；输出 dtype 由 kind 之后的 OUT_DTYPE 决定。
        in_ptr, _, in_tl = dtype_info(_CAST_DTYPES[int(op)])
        sig = {"src_ptr": in_ptr, "dst_ptr": ptr_ty, "n": "i32",
               "BLOCK": "constexpr", "IN_DTYPE": "constexpr", "OUT_DTYPE": "constexpr"}
        return sig, {"BLOCK": int(block), "IN_DTYPE": in_tl, "OUT_DTYPE": out_dt}, \
            _cast_kernel, f"cast({_CAST_DTYPES[int(op)]}->{dtype})", dtype
    if kind == "cmp":
        # 输出固定 i8：torch 的 bool 张量就是 1 字节 0/1。
        sig = {"a_ptr": ptr_ty, "out_ptr": "*i8", "n": "i32", "scalar": "fp32",
               "BLOCK": "constexpr", "OP": "constexpr"}
        return sig, {"BLOCK": int(block), "OP": int(op)}, _cmp_kernel, f"cmp(op={int(op)})", dtype
    if kind == "where":
        sig = {"c_ptr": "*i8", "a_ptr": ptr_ty, "b_ptr": ptr_ty, "out_ptr": ptr_ty,
               "n": "i32", "BLOCK": "constexpr", "OUT_DTYPE": "constexpr"}
        return sig, {"BLOCK": int(block), "OUT_DTYPE": out_dt}, _where_kernel, "where", dtype
    if kind == "math":
        # ``op`` 槽位装的是 OP 码（0 neg / 1 abs / 2 exp / 3 log / 4 sqrt / 5 tanh /
        # 6 sigmoid / 7 reciprocal / 8 sign / 9 floor / 10 ceil）。
        sig = {"a_ptr": ptr_ty, "out_ptr": ptr_ty, "n": "i32",
               "BLOCK": "constexpr", "OP": "constexpr", "OUT_DTYPE": "constexpr"}
        return sig, {"BLOCK": int(block), "OP": int(op), "OUT_DTYPE": out_dt}, \
            _math_kernel, f"math(op={int(op)})", dtype
    if kind == "relu":
        sig = {"a_ptr": ptr_ty, "out_ptr": ptr_ty, "n": "i32",
               "BLOCK": "constexpr", "OUT_DTYPE": "constexpr"}
        return sig, {"BLOCK": int(block), "OUT_DTYPE": out_dt}, \
            _relu_kernel, "relu", dtype
    if kind == "rsqrt":
        sig = {"a_ptr": ptr_ty, "out_ptr": ptr_ty, "n": "i32",
               "BLOCK": "constexpr", "OUT_DTYPE": "constexpr"}
        return sig, {"BLOCK": int(block), "OUT_DTYPE": out_dt}, \
            _rsqrt_kernel, "rsqrt", dtype
    if kind == "gelu":
        # ``op`` 槽位装 MODE（0=erf 精确版 / 1=tanh 近似版），两种形式共用一条内核。
        sig = {"a_ptr": ptr_ty, "out_ptr": ptr_ty, "n": "i32",
               "BLOCK": "constexpr", "OUT_DTYPE": "constexpr", "MODE": "constexpr"}
        return sig, {"BLOCK": int(block), "OUT_DTYPE": out_dt, "MODE": int(op)}, \
            _gelu_kernel, f"gelu(mode={int(op)})", dtype
    if kind == "masked_fill":
        sig = {"a_ptr": ptr_ty, "mask_ptr": "*i8", "out_ptr": ptr_ty, "n": "i32",
               "value": "fp32", "BLOCK": "constexpr", "OUT_DTYPE": "constexpr"}
        return sig, {"BLOCK": int(block), "OUT_DTYPE": out_dt}, \
            _masked_fill_kernel, "masked_fill", dtype
    raise ValueError(f"未知内核种类 {kind!r}")


def _build_kernel(kind: str, dtype: str, op: Optional[int], block: int, num_warps: int):
    """真正做编译的那一半（不带缓存）。"""
    sig, cexpr, fn, what, check_dtype = _signature(kind, dtype, op, block)
    return _compile(fn, sig, cexpr, check_dtype, num_warps, what)


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


#: 转换内核能接受的输入 dtype（下标进 constexpr，避免在签名里拼字符串）。
_CAST_DTYPES = ("fp32", "fp16", "bf16", "i8")


def cast(src_ptr, dst_ptr, n: int, src_dtype: str, dst_dtype: str, *,
         block: Optional[int] = None, num_warps: Optional[int] = None,
         stream: Optional[int] = None) -> None:
    """逐元素 dtype 转换：``dst = src.to(dst_dtype)``（两侧都是连续设备缓冲）。"""
    _check_n(n)
    dtype_info(src_dtype)
    dtype_info(dst_dtype)
    if n == 0:
        return
    if src_dtype not in _CAST_DTYPES:
        raise NotImplementedError(f"corex 的 cast 还不支持输入 dtype={src_dtype}")
    block = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel("cast", dst_dtype, _CAST_DTYPES.index(src_dtype), block,
                         int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, _grid(n, block), src_ptr, dst_ptr, int(n), stream=stream)


def cmp_scalar(op: int, a_ptr, out_ptr, n: int, dtype: str, scalar: float, *,
               block: Optional[int] = None, num_warps: Optional[int] = None,
               stream: Optional[int] = None) -> None:
    """标量比较：``out = a <op> scalar``（``out`` 是 bool 张量，按 i8 写）。"""
    _check_n(n)
    dtype_info(dtype)
    if n == 0:
        return
    block = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel("cmp", dtype, int(op), block, int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, _grid(n, block), a_ptr, out_ptr, int(n), float(scalar), stream=stream)


def where_self(c_ptr, a_ptr, b_ptr, out_ptr, n: int, dtype: str, *,
               block: Optional[int] = None, num_warps: Optional[int] = None,
               stream: Optional[int] = None) -> None:
    """``out = where(c, a, b)``；``c`` 是 bool 张量（i8），``a``/``b`` 与 ``out`` 同 dtype。"""
    _check_n(n)
    dtype_info(dtype)
    if n == 0:
        return
    block = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel("where", dtype, None, block, int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, _grid(n, block), c_ptr, a_ptr, b_ptr, out_ptr, int(n), stream=stream)


def scalar_op(op: int, a_ptr, out_ptr, n: int, dtype: str, scalar: float, *,
              block: Optional[int] = None, num_warps: Optional[int] = None,
              stream: Optional[int] = None) -> None:
    """通用「张量-标量」入口：``out = f(x, scalar)``。

    ``op`` 见 ``_scalar_kernel``：0 add / 1 sub / 2 mul / 3 div / 4 max / 5 min。
    单独开这个口子是为了让 clamp（``min(max(x, lo), hi)``）这类「组合算子」复用同一内核，
    不必为它再写一条只差一个比较的内核。
    """
    _run_scalar(int(op), a_ptr, out_ptr, n, dtype, scalar, block, num_warps, stream)


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


def math_unary(op: int, a_ptr, out_ptr, n: int, dtype: str, *,
               block: Optional[int] = None, num_warps: Optional[int] = None,
               stream: Optional[int] = None) -> None:
    """一元数学算子：``out = f(x)``，``op`` 见 ``_math_kernel`` 的 OP 码。

    精度口径：``exp``/``log``/``tanh``/``sigmoid`` 走的是近似指令（``tl.exp`` 落成
    ``ex2.approx.f32``），与 CPU 不逐位相等，验收用相对误差；``neg``/``abs``/``sign``/
    ``reciprocal``（除法本身）是精确的。
    """
    _check_n(n)
    dtype_info(dtype)
    if n == 0:
        return
    block = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel("math", dtype, int(op), block, int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, _grid(n, block), a_ptr, out_ptr, int(n), stream=stream)


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


def _run_unary(kind: str, op: Optional[int], a_ptr, out_ptr, n: int, dtype: str,
               block: Optional[int], num_warps: Optional[int],
               stream: Optional[int]) -> None:
    """一元算子的公共实现（relu / rsqrt / gelu 共用）。"""
    _check_n(n)
    dtype_info(dtype)
    if n == 0:
        return
    blk = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel(kind, dtype, op, blk, int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, _grid(n, blk), a_ptr, out_ptr, int(n), stream=stream)


def relu(a_ptr, out_ptr, n: int, dtype: str, *,
         block: Optional[int] = None, num_warps: Optional[int] = None,
         stream: Optional[int] = None) -> None:
    """``out = max(x, 0)``（``aten::relu`` / ``aten::relu_``）。

    原地就是传 ``out_ptr == a_ptr``（``relu_`` 直接这么调），每个 lane 只读自己那个元素，
    跨 lane 无依赖，安全。判据是**逐字节相等**：``max`` 没有舍入。
    """
    _run_unary("relu", None, a_ptr, out_ptr, n, dtype, block, num_warps, stream)


def rsqrt(a_ptr, out_ptr, n: int, dtype: str, *,
          block: Optional[int] = None, num_warps: Optional[int] = None,
          stream: Optional[int] = None) -> None:
    """``out = 1/sqrt(x)``（``aten::rsqrt`` / ``aten::rsqrt.out``）。

    **不承诺逐字节相等**：实现是 ``1.0 / tl.sqrt(x)``（``sqrt.approx`` + ``div.rn``，约 2 ulp），
    与 CPU 的 ``1/sqrt`` 有 1e-7 量级的相对差。探针按 max_rel 判。
    （``tl.math.rsqrt`` 在这张卡上走不通——落在未登记的 ``nvvm.rsqrt.approx.ftz`` 上，
    见 ``_rsqrt_kernel`` 的注释。）
    """
    _run_unary("rsqrt", None, a_ptr, out_ptr, n, dtype, block, num_warps, stream)


#: ``gelu`` 的两种近似方式 → 内核里的 ``MODE``。
GELU_MODES = {"erf": 0, "tanh": 1}


def gelu(a_ptr, out_ptr, n: int, dtype: str, mode: str = "erf", *,
         block: Optional[int] = None, num_warps: Optional[int] = None,
         stream: Optional[int] = None) -> None:
    """``aten::gelu`` / ``aten::gelu.out``。

    Args:
        mode: ``"erf"``（默认，aten 的 ``approximate="none"``，用 ``tl.math.erf``）或
            ``"tanh"``（``approximate="tanh"``）。两种形式共用一条内核，靠 constexpr 分叉。

    **不承诺逐字节相等**：两种形式都要算超越函数（``erf`` 经 libdevice、tanh 经
    ``ex2.approx`` 现推），探针按 max_abs/max_rel 判，并**分别对各自的数学形式**取参照
    （tanh 版不与精确 GELU 比，否则测的是近似误差而不是实现误差）。
    """
    if mode not in GELU_MODES:
        raise ValueError(f"gelu 的 mode 只支持 {sorted(GELU_MODES)}，收到 {mode!r}")
    _run_unary("gelu", GELU_MODES[mode], a_ptr, out_ptr, n, dtype,
               block, num_warps, stream)


def masked_fill(a_ptr, mask_ptr, out_ptr, n: int, dtype: str, value: float, *,
                block: Optional[int] = None, num_warps: Optional[int] = None,
                stream: Optional[int] = None) -> None:
    """``out = where(mask, value, x)``（``aten::masked_fill_.Scalar`` / ``masked_fill.Scalar``）。

    Args:
        mask_ptr: **bool 张量**的指针（每元素 1 字节，torch 里 bool 的 element_size 就是 1）。
            这里固定按 ``*i8`` 读、``m != 0`` 判真，所以上层不需要传 mask 的 dtype。
        value: 标量，fp32 传入（与 aten 的 ``Scalar`` 提升一致）。
        out_ptr: 可与 ``a_ptr`` 相同（``masked_fill_`` 就地改，原地安全）。

    判据：**逐字节相等**（选值不是算术，没有舍入；bf16/fp16 的 ``value`` 收窄是唯一的
    转换，且与 torch 的 ``static_cast<bf16>(value)`` 同一次 RNE）。
    """
    _check_n(n)
    dtype_info(dtype)
    if n == 0:
        return
    blk = int(block or DEFAULT_BLOCK)
    kernel = _get_kernel("masked_fill", dtype, None, blk,
                         int(num_warps or DEFAULT_NUM_WARPS))
    _triton.launch(kernel, _grid(n, blk), a_ptr, mask_ptr, out_ptr, int(n),
                   float(value), stream=stream)


# --------------------------------------------------------------------------- #
# 给 C++ 快路径的稳定入口（语义化，不暴露内部参数）
# --------------------------------------------------------------------------- #

#: 语义名字 → (内核种类, constexpr 槽位)。
#:
#: **``variant`` 只区分 Tensor / Scalar 两种变体**（C++ 侧按 (op, dtype, variant) 登记）。
#: ``gelu`` 的 erf/tanh 不是 variant，而是单独的关键字 ``mode``——两者是**同一个** (op, variant)
#: 下的两种数学形式，混进 variant 会让「变体」这个词失去意义。
_SEMANTIC: Dict[Tuple[str, str], Tuple[str, Optional[int]]] = {
    ("add", "tensor"): ("binary", _OP_ADD),
    ("sub", "tensor"): ("binary", _OP_SUB),
    ("mul", "tensor"): ("binary", _OP_MUL),
    ("div", "tensor"): ("binary", _OP_DIV),
    ("add", "scalar"): ("scalar", _OP_ADD),
    ("sub", "scalar"): ("scalar", _OP_SUB),
    ("mul", "scalar"): ("scalar", _OP_MUL),
    ("div", "scalar"): ("scalar", _OP_DIV),
    ("fill", "tensor"): ("fill", None),
    ("silu", "tensor"): ("silu", None),
    ("copy", "tensor"): ("copy", None),
    ("relu", "tensor"): ("relu", None),
    ("rsqrt", "tensor"): ("rsqrt", None),
    ("gelu", "tensor"): ("gelu", None),      # opcode 由 mode 决定，见 fast_op()
    ("masked_fill", "tensor"): ("masked_fill", None),
}

#: 真正进 C++ 快路径的那批（其余也登记在 _SEMANTIC 里，方便以后一起挪）。
FASTPATH_OPS: Tuple[str, ...] = ("add", "sub", "mul", "div")


def _resolve(op: str, variant: str, mode: str) -> Tuple[str, Optional[int]]:
    """把 (语义名, 变体, mode) 解析成 (内核种类, constexpr 槽位)。"""
    key = (op, variant)
    if key not in _SEMANTIC:
        raise NotImplementedError(
            f"没有 (op={op!r}, variant={variant!r}) 这个组合；"
            f"可用的是 {sorted(_SEMANTIC)}")
    kind, slot = _SEMANTIC[key]
    if kind == "gelu":
        if mode not in GELU_MODES:
            raise ValueError(f"gelu 的 mode 只支持 {sorted(GELU_MODES)}，收到 {mode!r}")
        slot = GELU_MODES[mode]
    return kind, slot


def fast_variants() -> Tuple[Tuple[str, str], ...]:
    """所有已登记的 ``(op, variant)`` 组合，按名字排序。

    C++ 侧生成注册表时应当**遍历这个**而不是自己写死列表——写死的那份一定会漂移。
    """
    return tuple(sorted(_SEMANTIC))


def fast_op(op: str, dtype: str = "fp32", variant: str = "tensor", *, mode: str = "erf",
            block: Optional[int] = None,
            num_warps: Optional[int] = None) -> Dict[str, Any]:
    """该 ``(op, variant, dtype)`` 的**实参顺序表**（不编译、不碰卡）。

    这是给 C++ 侧填「三个并行数组」用的机器可读描述。所有字段都从
    :func:`_signature` 的签名**数出来**，不是手写常量（原因见 :func:`_signature`）。

    Returns:
        字典，关键字段：

        * ``args``：**有序**的 ``((角色, 类别), …)``，类别是 ``"ptr"`` / ``"i32"`` / ``"f32"``。
          按这个顺序填参就不会错位。
        * ``n_ptrs`` / ``n_i32`` / ``n_f32``：三类实参的个数（C++ 的三个并行数组长度）。
        * ``has_alpha``：**内核签名里有没有 alpha 这个实参**。
        * ``alpha_effective``：alpha **是否真的改变数学结果**（只有 add/sub 是 True；
          mul/div 的内核签名里仍有 alpha 槽位，但传任何值都被忽略——照签名传 1.0 即可）。
          两者分开报是刻意的：只看 ``has_alpha`` 会让人以为 mul/div 可以少传一个参数。
        * ``ptr_roles`` / ``f32_roles`` / ``ptr_types``：指针的角色（``a``/``b``/``out``…）
          与 Triton 类型串（``mask`` 是 ``*i8``），用于 C++ 侧的断言与日志。
        * ``block`` / ``num_warps`` / ``grid_formula`` / ``grid_fn(n)``：启动参数。
        * ``kernel_kind`` / ``opcode``：内部槽位（排查用，C++ 不需要它）。
    """
    blk = int(block or DEFAULT_BLOCK)
    warps = int(num_warps or DEFAULT_NUM_WARPS)
    kind, slot = _resolve(op, variant, mode)
    sig, _cexpr, _fn, _what, _check = _signature(kind, dtype, slot, blk)

    runtime = [(name, ty) for name, ty in sig.items() if ty != "constexpr"]
    ptrs = [(name, ty) for name, ty in runtime if ty.startswith("*")]
    i32s = [name for name, ty in runtime if ty in ("i32", "i64")]
    f32s = [name for name, ty in runtime if ty == "fp32"]

    def role(name: str) -> str:
        """``a_ptr`` → ``a``，``mask_ptr`` → ``mask``（去掉 ``_ptr`` 后缀，便于阅读）。"""
        return name[:-4] if name.endswith("_ptr") else name

    args: Tuple[Tuple[str, str], ...] = tuple(
        (role(n), "ptr" if ty.startswith("*") else ("i32" if ty in ("i32", "i64") else "f32"))
        for n, ty in runtime)

    return {
        "op": op, "variant": variant, "dtype": dtype, "mode": mode,
        "kernel_kind": kind, "opcode": slot,
        "args": args,
        "n_ptrs": len(ptrs), "n_i32": len(i32s), "n_f32": len(f32s),
        "has_alpha": "alpha" in [role(n) for n, _ in ptrs]
                     or "alpha" in [role(n) for n in i32s + f32s],
        # 只有 **tensor 变体的 add/sub** 里 alpha 才真的改变数学结果。scalar 变体没有 alpha
        # 这个实参（它有 `scalar`），所以这里必须是 False——否则会和 has_alpha 自相矛盾
        # （`has_alpha=False` 却 `alpha_effective=True`），正是容易看错的那一类。
        "alpha_effective": (variant == "tensor" and op in ("add", "sub")),
        "ptr_roles": tuple(role(n) for n, _ in ptrs),
        "f32_roles": tuple(role(n) for n in f32s),
        "i32_roles": tuple(role(n) for n in i32s),
        "ptr_types": tuple(ty for _, ty in ptrs),
        "block": blk, "num_warps": warps,
        "grid_formula": "cdiv(n, block)",
        "grid_fn": lambda n, _b=blk: ((int(n) + _b - 1) // _b,),
    }


def get_kernel(op: str, dtype: str, variant: str = "tensor", *, mode: str = "erf",
               block: Optional[int] = None, num_warps: Optional[int] = None):
    """按**语义名字**取编译好的内核（带缓存），返回 `CompiledKernel`。

    这是给 C++ 侧 ``shim.load_kernel`` 用的入口：``get_kernel("add","fp16","scalar")``。
    ``kernel.asm["cubin"]`` 就是要喂给 ``load_kernel`` 的字节串。

    Raises:
        NotImplementedError: ``(op, variant)`` 组合不存在，或该 dtype 的内核编不过且属于
            「工具链不支持」类。
        RuntimeError: 其它编译失败（真 bug）。
    """
    blk = int(block or DEFAULT_BLOCK)
    warps = int(num_warps or DEFAULT_NUM_WARPS)
    kind, slot = _resolve(op, variant, mode)
    return _get_kernel(kind, dtype, slot, blk, warps)


def get_cubin(op: str, dtype: str, variant: str = "tensor", *, mode: str = "erf") -> bytes:
    """``get_kernel(...).asm["cubin"]`` 的便捷写法。"""
    return get_kernel(op, dtype, variant, mode=mode).asm["cubin"]


def get_launch_meta(op: str, dtype: str, variant: str = "tensor", *, mode: str = "erf",
                    block: Optional[int] = None,
                    num_warps: Optional[int] = None) -> Dict[str, Any]:
    """一次性拿到 C++ 侧 ``load_kernel`` 需要的全部东西：cubin + 启动参数。

    比让 C++ 侧自己拼 ``num_warps`` 安全：那些值得从已编译内核的 metadata 里读，
    而不是再抄一遍默认值。
    """
    kernel = get_kernel(op, dtype, variant, mode=mode, block=block, num_warps=num_warps)
    md = kernel.metadata
    recipe = fast_op(op, dtype, variant, mode=mode, block=block, num_warps=num_warps)
    return {
        "cubin": kernel.asm["cubin"],
        "name": getattr(kernel, "name", ""),
        "num_warps": int(getattr(md, "num_warps", recipe["num_warps"])),
        "num_stages": int(getattr(md, "num_stages", 1)),
        "shared": int(getattr(md, "shared", 0)),
        "block": recipe["block"],
        "args": recipe["args"],
        "n_ptrs": recipe["n_ptrs"], "n_i32": recipe["n_i32"], "n_f32": recipe["n_f32"],
        "has_alpha": recipe["has_alpha"], "alpha_effective": recipe["alpha_effective"],
        "ptr_roles": recipe["ptr_roles"], "ptr_types": recipe["ptr_types"],
    }


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
    "DTYPES", "DEFAULT_BLOCK", "DEFAULT_NUM_WARPS", "GELU_MODES",
    "add", "sub", "mul", "div",
    "add_scalar", "sub_scalar", "mul_scalar", "div_scalar",
    "fill", "silu", "copy", "relu", "rsqrt", "gelu", "masked_fill",
    # ---- C++ 快路径用的稳定入口（语义化）----
    "FASTPATH_OPS", "fast_variants", "fast_op", "get_kernel", "get_cubin",
    "get_launch_meta",
    "dtype_info", "itemsize", "traffic_bytes", "cache_info", "clear_cache",
]
