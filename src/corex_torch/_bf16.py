"""把 Triton 的 bf16 转换改写成 ``bi`` 后端能选的形态。

**为什么需要这一层**：厂商 torch 2.1 在这张卡上能跑 bf16，但它的转换走的是 intrinsic——
`__CUDA_ARCH__ = 300`、`__ILUVATAR__ = 1`，于是 `BFloat16-inl.h` 里
`(__CUDA_ARCH__ >= 800 || defined(__ILUVATAR__))` 为真，`__float2bfloat16` /
`__bfloat162float` 落到

    call float  @llvm.bi.bfloat2float(bfloat %x)      ; 展宽
    call bfloat @llvm.bi.float2bfloat(float %x)       ; 收窄

而 Triton 的 lowering 发的是标准 LLVM 的 ``fpext`` / ``fptrunc``，bi 后端只认前者：
`fpext` 报 `Cannot select: f32 = fp_extend`、`fptrunc` 报 `Cannot select: bf16 = fp_round`。
所以缺的不是「bf16 能力」，是**指令的名字**。

**两种形态分开处理（`probes/015` 逐个喂 llc 实测）**：

| 形态 | 做法 | 依据 |
| --- | --- | --- |
| 标量 `fpext bfloat` / `fptrunc … to bfloat` | 换成 `llvm.bi.*` intrinsic | `probes/012`：厂商 IR 就是这么写的，llc OK |
| **向量** `<N x bfloat>` | **整数位运算**（展宽＝`zext`+`shl`+`bitcast`；收窄＝`bitcast`+RNE+`lshr`+`trunc`+`bitcast`） | `probes/015`：向量 intrinsic **不被支持**（`input module cannot be verified`），位运算两条都 OK |

为什么矩阵乘/规约一定要走向量分支：它们是整块 tile 的 `.to(tl.bfloat16)`，Triton 会发
`fptrunc <8 x float> %x to <8 x bfloat>`（`probes/014` 在真编译的 biir 里抓到的就是这一条）。

Module: ``corex_torch``。被 ``_triton.activate()`` 注册的后端子类调用，作用在 biir 文本上、
进 ``llc`` 之前。

Constraints:
    * 只改 **bfloat** 的转换；fp16↔fp32 后端自己支持，别碰。
    * 向量收窄用「+0x7FFF+最低位」的 RNE。**NaN 的边界与厂商 intrinsic 不完全一致**：
      极端 payload 的 NaN 有可能被加成 inf。推理路径上 NaN 只会在数学已经出错时出现，
      发现它就是该查上游，所以这里接受这个取舍（要精确等价就得把向量拆成标量走 intrinsic，
      那会让 store 路径的指令数翻三倍）。
    * 纯文本改写、幂等：改写后不再有 bf16 的 fpext/fptrunc 行。
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

#: 标量 intrinsic 的声明。改写后如果 IR 里还没有就补在末尾。
SCALAR_DECLARATIONS = (
    "declare float @llvm.bi.bfloat2float(bfloat)",
    "declare bfloat @llvm.bi.float2bfloat(float)",
)

#: 向量形态该走哪条路（进程级，可被编译失败自动切换）。
#:
#: ``"intrinsic"``：``llvm.bi.bfloat2float.v<N>bf16`` 这类带 mangled 后缀的向量重载。
#:   **注意它只是「半实现」**：`llc` 会照样接受（把未解析的调用原样留在目标文件里、rc=0），
#:   真正拦住它的是下一步 `lld --no-undefined` 报 ``undefined symbol``（`probes/192` 实测，
#:   规约与矩阵乘两种内核都栽在这里）。所以「只看 llc 编得过」会把「parse 进去但没 lowering」
#:   误判成「支持」——这个教训值得记一辈子。
#:   部分元素数确实有实现（`<4 x …>` 那档在 mm 上真跑通过），所以不能一刀切禁掉。
#: ``"bits"``：整数位运算。**已验过逐字节相等**（`probes/013` 的 bf16 矩阵乘 4096/4096）。
#:
#: 策略：先试 intrinsic，链接报 ``undefined symbol`` 就自动切到 bits 并记住（见
#: ``_triton.CorexTorchBackend.make_cubin``）。这样既有机会吃「有实现的那几档」，
#: 又不会因为某档没实现就把整条 bf16 路堵死。
_STRATEGY = {"vector": "scalar"}

#: 两种取值，只给调用方判断用。
_VECTOR_STRATEGIES = ("scalar", "intrinsic", "bits")

_NAME = r"%[-A-Za-z0-9_.$]+"

#: 标量两种方向。
_SCALAR_WIDEN = re.compile(
    rf"(?P<lhs>{_NAME})\s*=\s*fpext\s+bfloat\s+(?P<arg>{_NAME})\s+to\s+float")
_SCALAR_NARROW = re.compile(
    rf"(?P<lhs>{_NAME})\s*=\s*fptrunc\s+float\s+(?P<arg>{_NAME})\s+to\s+bfloat")

#: 向量两种方向（``<8 x bfloat>`` ↔ ``<8 x float>``）。
_VEC_WIDEN = re.compile(
    rf"(?P<lhs>{_NAME})\s*=\s*fpext\s+<(?P<n>\d+) x bfloat>\s+(?P<arg>{_NAME})\s+to\s+<(?P=n) x float>")
_VEC_NARROW = re.compile(
    rf"(?P<lhs>{_NAME})\s*=\s*fptrunc\s+<(?P<n>\d+) x float>\s+(?P<arg>{_NAME})\s+to\s+<(?P=n) x bfloat>")


def _vec_intrinsic(base: str, n: int, src_ty: str, dst_ty: str, arg: str, lhs: str) -> str:
    """生成一条向量 intrinsic 调用 + 它需要的声明。

    ``base`` 是 ``bfloat2float`` / ``float2bfloat``；mangled 后缀的形状是
    ``.v<N>bf16``（元素数 + 源/目的共同的 bf16 家族），例如 ``<8 x float> ← <8 x bfloat>``
    用 ``llvm.bi.bfloat2float.v8bf16``。
    """
    name = f"llvm.bi.{base}.v{n}bf16"
    return f"{lhs} = call {dst_ty} @{name}({src_ty} {arg})"


def _vec_const(n: int, value: int) -> str:
    """造一个显式向量常量（``bits`` 策略用）。

    不能用 ``splat (i32 16)``：那是 LLVM 20 的写法，厂商 llc 是 16，会报
    ``expected value token``（triton-bridge 在 `probes/190` 里踩过）。
    """
    return "<" + ", ".join([f"i32 {value}"] * n) + ">"


class _NameGen:
    """给改写出来的新 SSA 值起名。

    为什么不能直接拿原值名加后缀：Triton 的 llir 里值是**编号值**（``%257``），而
    ``%257.i`` 不是合法标识符——llc 会报 ``expected '=' after instruction id``
    （2026-11-07 在 probes/013 里踩过）。编号值的名字只能是 ``%<数字>``；具名值必须以
    字母/下划线/``$``/``.`` 开头。所以这里另起一个不冲突的具名前缀。
    """

    def __init__(self, ir: str) -> None:
        self._prefix = "ctbf"
        while self._prefix in ir:                        # 极端情况下的避让
            self._prefix += "x"
        self._n = 0

    def __call__(self) -> str:
        self._n += 1
        return f"%{self._prefix}{self._n}"


def get_vector_strategy() -> str:
    """当前向量转换策略（``"intrinsic"`` 或 ``"bits"``）。"""
    return _STRATEGY["vector"]


def set_vector_strategy(name: str) -> None:
    """切换向量转换策略（进程级）。

    Args:
        name: ``"intrinsic"`` 或 ``"bits"``。

    Raises:
        ValueError: 名字不认识——静默忽略会让「以为切了其实没切」变成排查噩梦。
    """
    if name not in _VECTOR_STRATEGIES:
        raise ValueError(f"未知的向量转换策略 {name!r}，只支持 {_VECTOR_STRATEGIES}")
    _STRATEGY["vector"] = name


def rewrite_bf16_conversions(ir: str, vector_strategy: Optional[str] = None) -> Tuple[str, int]:
    """把 biir 里的 bf16 转换换成 llc 认的形态。

    Args:
        ir: ``CorexBackend.make_biir`` 产出的 LLVM IR 文本。
        vector_strategy: 向量形态的处理方式；``None`` 用当前进程策略（见 :func:`get_vector_strategy`）。

    Returns:
        ``(改写后的文本, 改动条数)``。条数为 0 表示这个内核不做 bf16 转换。

    Notes:
        改写只在「一条指令独立成行」的形态上做。漏掉的形态会让 llc 报 ``Cannot select``，
        属于「立刻暴露」而不是静默算错——这是这套文本改写可以接受的风险边界。
    """
    changed = 0
    strategy = vector_strategy or _STRATEGY["vector"]
    if strategy not in _VECTOR_STRATEGIES:
        raise ValueError(f"未知的向量转换策略 {strategy!r}，只支持 {_VECTOR_STRATEGIES}")
    fresh = _NameGen(ir)
    #: 向量 intrinsic 用到的元素数集合，决定要补哪些 declare。
    vec_widths: set = set()

    def _widen_scalar(match: "re.Match[str]") -> str:
        nonlocal changed
        changed += 1
        return (f"{match.group('lhs')} = call float @llvm.bi.bfloat2float("
                f"bfloat {match.group('arg')})")

    def _narrow_scalar(match: "re.Match[str]") -> str:
        nonlocal changed
        changed += 1
        return (f"{match.group('lhs')} = call bfloat @llvm.bi.float2bfloat("
                f"float {match.group('arg')})")

    def _widen_vec(match: "re.Match[str]") -> str:
        nonlocal changed
        changed += 1
        lhs, arg, n = match.group("lhs"), match.group("arg"), int(match.group("n"))
        if strategy == "intrinsic":
            vec_widths.add(n)
            return _vec_intrinsic("bfloat2float", n, f"<{n} x bfloat>", f"<{n} x float>",
                                  arg, lhs)
        if strategy == "scalar":
            return _scalarize_wide(n, arg, lhs)
        h, z, s = fresh(), fresh(), fresh()
        # bf16 -> f32 就是「把 16 位搬到高半」：zext 到 i32 再左移 16，按位重解释。
        return (f"{h} = bitcast <{n} x bfloat> {arg} to <{n} x i16>\n"
                f"{z} = zext <{n} x i16> {h} to <{n} x i32>\n"
                f"{s} = shl <{n} x i32> {z}, {_vec_const(n, 16)}\n"
                f"{lhs} = bitcast <{n} x i32> {s} to <{n} x float>")

    def _narrow_vec(match: "re.Match[str]") -> str:
        nonlocal changed
        changed += 1
        lhs, arg, n = match.group("lhs"), match.group("arg"), int(match.group("n"))
        if strategy == "intrinsic":
            vec_widths.add(n)
            return _vec_intrinsic("float2bfloat", n, f"<{n} x float>", f"<{n} x bfloat>",
                                  arg, lhs)
        if strategy == "scalar":
            return _scalarize_narrow(n, arg, lhs)
        i, lo, lsb, c1, c2, hi, t = (fresh() for _ in range(7))
        # f32 -> bf16：取高 16 位，但先按 RNE 补偿（+0x7FFF+最低位），否则与厂商 intrinsic
        # 在「正好半程」的值上差 1 ulp。
        return (f"{i} = bitcast <{n} x float> {arg} to <{n} x i32>\n"
                f"{lo} = lshr <{n} x i32> {i}, {_vec_const(n, 16)}\n"
                f"{lsb} = and <{n} x i32> {lo}, {_vec_const(n, 1)}\n"
                f"{c1} = add <{n} x i32> {i}, {_vec_const(n, 0x7FFF)}\n"
                f"{c2} = add <{n} x i32> {c1}, {lsb}\n"
                f"{hi} = lshr <{n} x i32> {c2}, {_vec_const(n, 16)}\n"
                f"{t} = trunc <{n} x i32> {hi} to <{n} x i16>\n"
                f"{lhs} = bitcast <{n} x i16> {t} to <{n} x bfloat>")

    def _scalarize_wide(n: int, arg: str, lhs: str) -> str:
        """把一个向量展宽拆成逐元素「extract → 标量 intrinsic → insert」链。

        为什么逐个拆而不是发向量 intrinsic：``llvm.bi.bfloat2float.v8bf16`` 这类 mangled 名字
        **llc 收下但不实现**——目标文件里留一个外部符号，`lld --no-undefined` 才报
        ``undefined symbol``（`probes/195` 三级判定）。标量版是真实现的（元素级 33/33 已证明），
        所以拆开是唯一稳妥的向量通路。
        """
        lines, prev = [], "poison"
        for i in range(n):
            e, c = fresh(), fresh()
            lines.append(f"{e} = extractelement <{n} x bfloat> {arg}, i32 {i}")
            lines.append(f"{c} = call float @llvm.bi.bfloat2float(bfloat {e})")
            dst = lhs if i == n - 1 else fresh()
            lines.append(f"{dst} = insertelement <{n} x float> {prev}, float {c}, i32 {i}")
            prev = dst
        return "\n".join(lines)

    def _scalarize_narrow(n: int, arg: str, lhs: str) -> str:
        """同 :func:`_scalarize_wide`，收窄方向（f32 → bf16）。"""
        lines, prev = [], "poison"
        for i in range(n):
            e, c = fresh(), fresh()
            lines.append(f"{e} = extractelement <{n} x float> {arg}, i32 {i}")
            lines.append(f"{c} = call bfloat @llvm.bi.float2bfloat(float {e})")
            dst = lhs if i == n - 1 else fresh()
            lines.append(f"{dst} = insertelement <{n} x bfloat> {prev}, bfloat {c}, i32 {i}")
            prev = dst
        return "\n".join(lines)

    out = _SCALAR_WIDEN.sub(_widen_scalar, ir)
    out = _SCALAR_NARROW.sub(_narrow_scalar, out)
    out = _VEC_WIDEN.sub(_widen_vec, out)
    out = _VEC_NARROW.sub(_narrow_vec, out)

    if os.environ.get("COREX_BF16_DUMP"):
        # 诊断用：把进 llc 之前的最终文本落盘，便于看清「漏改的那条长什么样」。
        with open(os.environ["COREX_BF16_DUMP"], "w", encoding="utf-8") as fh:
            fh.write(out)
    if not changed:
        return ir, 0

    declares = list(SCALAR_DECLARATIONS)
    for n in sorted(vec_widths):
        declares.append(f"declare <{n} x float> @llvm.bi.bfloat2float.v{n}bf16(<{n} x bfloat>)")
        declares.append(f"declare <{n} x bfloat> @llvm.bi.float2bfloat.v{n}bf16(<{n} x float>)")
    missing = [decl for decl in declares if decl not in out]
    if missing:
        out = out.rstrip() + "\n" + "\n".join(missing) + "\n"
    return out, changed
