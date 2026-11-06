"""Triton 的编译与启动入口：把 corex-triton 的能力接到 corex-torch 上。

**这一层只做三件事**：定位 corex-triton 的源码、把它的后端激活、提供一个「网格+实参
→ 真的跑起来」的薄封装。编译（`ttir → ttgir → bi LLVM IR → llc → lld → cubin`）与运行期
（context / stream / 显存 / 计时）全部由 `triton_corex` 提供——它已经不依赖 torch 了
（`corex-triton` 提交 `cabf39a`：主 context 用 `cuDevicePrimaryCtxRetain`，stream 用 legacy
默认流），所以本仓库没必要再复制一份。

为什么要留这一层而不是直接 `import triton_corex`：
``triton_corex`` 是个没打包安装的源码目录，而且它注册的是**全局**后端表（进程级副作用）。
把「找路径 + 注册 + 激活」收在一处，调用方（算子层、设备层）就不用关心这些顺序问题。

Module: ``corex_torch``。被 ``_ops_elementwise`` 与 ``_device`` 调用；本身不 import torch。

Constraints:
    * **不 import torch**：整条编译/启动路径必须能在只有 CoreX 运行时 + triton 的进程里跑通。
    * 激活是幂等的，但**不要**在同一个进程里反复切换后端——`driver.active` 是全局的。
    * 不提供 graph、不用 cp.async（平台禁令，见 ``docs/05-DRIVER-CONSTRAINTS.md``）。
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Any, Dict, Optional, Tuple

#: corex-triton 的源码目录（里面有 ``triton_corex`` 包）。默认按仓库同级目录推断。
COREX_TRITON_SRC_ENV = "COREX_TRITON_SRC"

#: 编译目标。arch 固定 70：不是 NVIDIA 的算力，而是借这个数字告诉上游 lowering
#: 「没有 cp.async、没有 bf16 mma」——这两样在这张卡上会挂卡或编不出来。
TARGET_ARCH = 70
TARGET_WARP_SIZE = 32

_ACTIVATED = False


def corex_triton_src() -> Optional[str]:
    """定位 corex-triton 的 ``src`` 目录。

    Returns:
        目录路径；既没设环境变量、同级目录也不存在时返回 ``None``。
    """
    env = os.environ.get(COREX_TRITON_SRC_ENV)
    if env:
        return env
    # corex-torch/src/corex_torch/_triton.py → 上四层是仓库父目录
    guess = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), "corex-triton", "src")
    return guess if os.path.isdir(os.path.join(guess, "triton_corex")) else None


def _ensure_importable() -> None:
    """把 corex-triton 的源码目录挂进 ``sys.path``。"""
    path = corex_triton_src()
    if path and path not in sys.path:
        sys.path.insert(0, path)


def target():
    """当前编译目标（`GPUTarget("corex", 70, 32)`）。"""
    from triton.backends.compiler import GPUTarget

    return GPUTarget("corex", TARGET_ARCH, TARGET_WARP_SIZE)


def activate():
    """把 corex 后端注册进 Triton 并设为活动后端（幂等）。

    为什么必须显式 ``set_active``：``CompiledKernel._init_handles`` 走的是 ``driver.active``，
    它按后端表顺序挑第一个 ``is_active()`` 为真的后端，而 NVIDIA 那个在这张卡上同样为真
    （libcuda 存在），于是会拿到 ptxas 那套 launcher 并失败。

    Returns:
        被激活的 driver 实例。
    """
    global _ACTIVATED
    _ensure_importable()
    from triton.runtime.driver import driver as _driver_config

    import triton_corex  # noqa: F401  - 导入即把 backend 注册进 triton.backends.backends
    from triton_corex.driver import CorexDriver

    if not _ACTIVATED:
        _driver_config.set_active(CorexDriver())
        _ACTIVATED = True
    return _driver_config.active


def _normalize_grid(grid) -> Tuple[int, int, int]:
    """把一维/二维网格补成 Triton 启动器要求的 3 元组。

    ``CompiledKernel.__getitem__`` 的 runner 直接读 ``grid[0]/grid[1]/grid[2]``，
    它**不做**补齐（补齐是 ``JITFunction.__getitem__`` 才有的）。只传 ``(16,)`` 会在
    Triton 自己的 compiler.py 里抛 ``IndexError: tuple index out of range``。
    """
    dims = tuple(int(x) for x in (grid if isinstance(grid, (tuple, list)) else (grid,)))
    if len(dims) > 3:
        raise ValueError(f"网格最多 3 维，收到 {dims}")
    return dims + (1,) * (3 - len(dims))


def _plain(value: Any) -> Any:
    """把设备指针统一成 int。

    为什么要转：``triton_corex.driver._pack_arg`` 对指针走 ``int(pyval)``，而
    ``int(ctypes.c_void_p(...))`` 在 CPython 上是「把 8 字节地址当十进制字符串解析」，
    地址字节不全是 ASCII 数字时会抛 ``ValueError``（``docs/00-FACTS.md`` §7.1）。
    所以在进入对方代码之前就把 ``c_void_p`` 拆成 int。
    """
    if isinstance(value, ctypes.c_void_p):
        return int(value.value or 0)
    if isinstance(value, (bytes, bytearray)) and len(value) == 8:
        return int.from_bytes(bytes(value), "little")
    return value


def compile_kernel(fn, signature: Dict[str, str], constexprs: Optional[Dict[str, Any]] = None,
                   num_warps: int = 4, num_stages: int = 1):
    """编译一个 ``@triton.jit`` 内核，返回 ``CompiledKernel``（含 ``cubin``）。

    Args:
        fn: ``triton.jit`` 装饰过的函数。
        signature: 参数名到 Triton 类型串的映射（``*fp32`` / ``i32`` / ``constexpr``）。
        constexprs: 编译期常量。
        num_warps: 每块的 warp 数（布局按 32 算，见 ``TARGET_WARP_SIZE``）。
        num_stages: 流水级数。固定 1：本平台没有 ``cp.async``，多级流水没有硬件通路。

    Returns:
        编译产物，``kernel[grid](*args)`` 即可启动。

    Raises:
        RuntimeError: 编译失败（多半是 IR 里混进了厂商 llc 不认的语法）。
    """
    from triton.compiler import ASTSource, compile as triton_compile

    activate()
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs or {}, attrs=None)
    return triton_compile(src, target=target(),
                          options={"num_warps": num_warps, "num_stages": num_stages})


def launch(kernel, grid, *args, stream: Optional[int] = None) -> None:
    """启动一个已编译的内核。

    Args:
        kernel: ``compile_kernel`` 的返回值。
        grid: 网格，``int`` 或 1–3 元组。
        *args: 内核实参；设备指针可以是 ``torch.Tensor`` / int / ``c_void_p``。
        stream: 流句柄；``None`` 用 corex 后端的默认（legacy null stream）。

    Raises:
        RuntimeError: 实参个数与签名不符，或启动返回非 0。
    """
    gx, gy, gz = _normalize_grid(grid)
    packed = tuple(_plain(a) for a in args)
    if stream is None:
        kernel[(gx, gy, gz)](*packed)
    else:
        kernel[(gx, gy, gz)](*packed, stream=int(stream))


def synchronize() -> None:
    """等设备上所有工作完成（走的是 corex 后端自己的 ``cuCtxSynchronize``）。"""
    activate()
    from triton_corex.driver import CorexDeviceInterface

    CorexDeviceInterface.synchronize()
