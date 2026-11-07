"""Triton 的编译与启动入口：把 corex-triton 的能力接到 corex-torch 上。

**这一层只做三件事**：定位 corex-triton 的源码、把它的后端激活、提供一个「网格+实参
→ 真的跑起来」的薄封装。编译（`ttir → ttgir → bi LLVM IR → llc → lld → cubin`）与运行期
（context / stream / 显存 / 计时）全部由 `triton_corex` 提供——它已经不依赖 torch 了
（`corex-triton` 提交 `cabf39a`：主 context 用 `cuDevicePrimaryCtxRetain`，stream 用 legacy
默认流），所以本仓库没必要再复制一份。

为什么要留这一层而不是直接 `import triton_corex`：
``triton_corex`` 是个没打包安装的源码目录，而且它注册的是**全局**后端表（进程级副作用）。
把「找路径 + 注册 + 激活」收在一处，调用方（算子层、设备层）就不用关心这些顺序问题。

Module: ``torch.corex``。被 ``_ops_elementwise`` 与 ``_device`` 调用；本身不 import torch。

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
    # corex-torch/torch/corex/_triton.py → 上四层是 coffin-repo，corex-triton 与 corex-torch 平级；
    # 它的后端源码在 corex-triton/corex/src（fork 之后插件部分沉到 corex/ 下）。
    guess = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), "corex-triton", "corex", "src")
    return guess if os.path.isdir(os.path.join(guess, "triton_corex")) else None



def _is_fork_backend() -> bool:
    """当前 ``triton.backends.corex`` 是不是 fork 自带的那个后端。

    判别方式：后端 compiler 模块的 ``__file__`` 在 triton 包目录之内。插件形态下
    ``triton_corex`` 在 triton 包外，路径里不会出现 ``/triton/backends/corex/``。
    """
    try:
        import triton.backends.corex.compiler as _compiler
    except Exception:
        return False
    path = getattr(_compiler, "__file__", "") or ""
    return "triton/backends/corex" in path.replace("\\", "/")


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

    另外这里**换了编译器类**：``CorexTorchBackend`` 在 biir 进 ``llc`` 之前把 Triton 发的
    ``fpext``/``fptrunc``（bfloat）改写成厂商的 ``llvm.bi.bfloat2float`` / ``llvm.bi.float2bfloat``。
    不做这步，任何 bf16 内核都会死在 ``Cannot select: fp_extend`` / ``fp_round``（见
    ``_bf16`` 与 ``probes/011``、``probes/012``）。

    Returns:
        被激活的 driver 实例。
    """
    global _ACTIVATED
    _ensure_importable()
    from triton.backends import Backend, backends
    from triton.runtime.driver import driver as _driver_config

    # fork 形态优先：corex 后端就在 triton 树里（``triton.backends.corex``），bf16 改写也已经
    # 搬进它自己的 ``backend/rewrite.py``，torch 这边不需要再包一层子类。
    #
    # 注意：Triton 3.4 的后端表来自 **entry point**（setup.py 里为每个 in-tree 后端生成
    # ``triton.backends.<name>``），所以「把 fork 的 python 目录挂进 PYTHONPATH」只能让
    # ``import triton.backends.corex`` 成功，``backends`` 字典里仍然是空的。这里补登记一次，
    # 否则会掉回插件分支并报 `No module named 'triton_corex'`。
    if "corex" not in backends and _is_fork_backend():
        from triton.backends.corex.compiler import CorexBackend as _ForkCompiler
        from triton.backends.corex.driver import CorexDriver as _ForkDriver

        from . import _bf16

        class _TorchCorexCompiler(_ForkCompiler):
            """fork 的编译器 + 一层 bf16 转换改写。

            为什么 fork 里还要包这一层：``corex-triton`` 的 ``backend/rewrite.py`` 目前只做
            llir 的通用改写，**没有** bf16 转换（实测 ``llc -march=bi`` 直接
            ``LLVM ERROR: Cannot select: bf16 = fp_round``）。这段改写应该搬进 fork 的
            ``rewrite.py``，搬完这里就退化成一层空壳、可以直接删。
            """

            @staticmethod
            def make_cubin(src, metadata, options) -> bytes:
                rewritten, hits = _bf16.rewrite_bf16_conversions(src)
                if hits:
                    counts = dict(metadata.get("corex_rewrite_counts") or {})
                    counts["bf16_conversions"] = hits
                    metadata["corex_rewrite_counts"] = counts
                return _ForkCompiler.make_cubin(rewritten, metadata, options)

        backends["corex"] = Backend(_TorchCorexCompiler, _ForkDriver)

    if "corex" in backends and _is_fork_backend():
        if not _ACTIVATED:
            _driver_config.set_active(backends["corex"].driver())
            _ACTIVATED = True
        return _driver_config.active

    import triton_corex  # noqa: F401  - 插件形态：导入即注册 backend
    from triton_corex.compiler import CorexBackend
    from triton_corex.driver import CorexDriver

    from . import _bf16

    class CorexTorchBackend(CorexBackend):
        """corex-triton 的编译器 + 一层 bf16 转换改写（不改上游仓库，只在这里包一层）。"""

        @staticmethod
        def make_cubin(src, metadata, options) -> bytes:
            """biir → cubin。进 ``llc`` 之前先做 bf16 转换的 intrinsic 替换。"""
            rewritten, hits = _bf16.rewrite_bf16_conversions(src)
            if hits:
                # 命中数进 metadata：编译脚本据此确认改写真的发生了（为 0 说明上游换 lowering 了）。
                counts = dict(metadata.get("corex_rewrite_counts") or {})
                counts["bf16_conversions"] = hits
                metadata["corex_rewrite_counts"] = counts
            return CorexBackend.make_cubin(rewritten, metadata, options)

    if not _ACTIVATED:
        backends["corex"] = Backend(CorexTorchBackend, CorexDriver)
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
    if _is_fork_backend():
        from triton.backends import backends

        backends["corex"].driver().get_device_interface().synchronize()
        return
    from triton_corex.driver import CorexDeviceInterface

    CorexDeviceInterface.synchronize()
