"""把 corex-triton 的编译链接到 corex-torch 的运行时的适配层。

分工：**编译全归 corex-triton**（`TritonGPU → bi LLVM IR → llc → lld → cubin`，那 4 万行
改写规则没理由复制一份），**运行期全归本仓库**（context / 流 / 显存都由 ``_runtime`` 提供）。
``corex-triton`` 的 driver 是给厂商 torch 2.1 写的：它借 ``torch.cuda`` 建 context、拿流、
分配 global scratch。我们这套跑在上游 CPU 版 torch 2.8 上，没有 ``torch.cuda``，
于是把那 100 行运行期自己重写一遍，比去改已交付的仓库更省事、也更不容易踩坏它。

Module: ``corex_torch``。被 ``_device`` 的算子层调用（``launch(kernel, grid, args...)``），
也可能被用户直接调用（绕过 torch 直接跑 Triton kernel）。

Constraints:
    * **不 import torch**。整条 kernel 启动路径必须能在只有 CoreX 运行时 + triton 的环境里跑通
      （``probes/151`` 已在无 torch 进程里验证过这条路可行）。
    * 不调用 ``triton.backends.driver.GPUDriver.__init__``：它内部 ``import torch`` 并去读
      ``torch.cuda``。我们从 ``DriverBase`` 起，自己装那 4 个钩子
      （triton 3.4 的 ``_init_handles`` 一共只用 6 个东西，见 ``docs/04-TRITON-BRIDGE.md``）。
    * 一次只允许一个 driver 实例被激活（Triton 的 ``driver.active`` 是进程级全局）。
    * 不提供 graph、不用 cp.async（平台禁令，见 ``docs/05-DRIVER-CONSTRAINTS.md``）。
"""

from __future__ import annotations

import ctypes
import os
import struct
import sys
from typing import Any, Dict, Optional, Tuple

from . import _runtime

#: corex-triton 的源码目录。默认按仓库同级目录推断（corex-torch 与 corex-triton 并列）。
COREX_TRITON_SRC_ENV = "COREX_TRITON_SRC"

#: cubin 缓存：同一份内核反复启动不该反复装载（装载是把 cubin 拷进显存）。
_MODULE_CACHE: Dict[str, Tuple[ctypes.c_void_p, ctypes.c_void_p]] = {}

#: 本后端的编译目标。arch 固定 70：不是 NVIDIA 的算力，而是借这个数字告诉上游 lowering
#: 「没有 cp.async、没有 bf16 mma」——这两样在这张卡上会挂卡或编不出来
#: （``corex-triton/docs/00-FACTS.md`` §2、§4）。warp 32 同理：布局按 32 自洽，实测成立（§8）。
TARGET_ARCH = 70
TARGET_WARP_SIZE = 32


def _corex_triton_src() -> Optional[str]:
    """定位 corex-triton 的 ``src`` 目录（里面有 ``triton_corex`` 包）。"""
    env = os.environ.get(COREX_TRITON_SRC_ENV)
    if env:
        return env
    # corex-torch/src/corex_torch/_triton.py → 上四层是仓库父目录
    guess = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), "corex-triton", "src")
    return guess if os.path.isdir(os.path.join(guess, "triton_corex")) else None


def _ensure_triton_corex_importable() -> None:
    """把 corex-triton 的源码目录挂进 ``sys.path``（它没有打包安装，只能这样引用）。"""
    path = _corex_triton_src()
    if path and path not in sys.path:
        sys.path.insert(0, path)


class _LibCuda:
    """libcuda 的惰性单例：只绑我们真正用到的 5 个入口。

    为什么用 driver API 启动而不是 runtime API：kernel 句柄与流句柄两套 API 在这张卡上
    指向**同一个 primary context**（``probes/152`` 实测句柄相同），而 ``cuLaunchKernel``
    是唯一能吃「函数句柄 + 参数指针数组」的入口——runtime 的 ``cudaLaunchKernel`` 需要
    先 ``cudaGetFuncBySymbol``，多一层且没有收益。
    """

    _lib: Optional[ctypes.CDLL] = None

    @classmethod
    def get(cls) -> ctypes.CDLL:
        if cls._lib is None:
            # 保证 context 已经存在（本平台 cudaSetDevice 就会建 primary context，实测见 probes/152）
            _runtime.set_device(_runtime.current_device())
            path = os.path.join(_runtime.COREX_HOME, "lib64", "libcuda.so.1")
            lib = ctypes.CDLL(path if os.path.exists(path) else "libcuda.so.1")
            lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
            lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                                ctypes.c_void_p, ctypes.c_char_p]
            lib.cuLaunchKernel.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                ctypes.c_uint, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
            lib.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            for fn in ("cuModuleLoadData", "cuModuleGetFunction", "cuLaunchKernel",
                       "cuFuncSetAttribute", "cuGetErrorString"):
                getattr(lib, fn).restype = ctypes.c_int
            cls._lib = lib
        return cls._lib


def _errstr(rc: int) -> str:
    """把 cu 错误码翻成人话。失败时返回空串，不让报错路径再炸一次。"""
    try:
        lib = _LibCuda.get()
        text = ctypes.c_char_p()
        lib.cuGetErrorString(int(rc), ctypes.byref(text))
        return text.value.decode(errors="replace") if text.value else ""
    except Exception:                                   # noqa: BLE001
        return ""


def _check(name: str, rc: int) -> None:
    if rc != 0:
        raise RuntimeError(f"{name} 失败 rc={rc} {_errstr(rc)}".rstrip())


class CorexTorchUtils:
    """``driver.active.utils``：设备属性 + 模块装载。

    与 ``triton_corex.driver.CorexUtils`` 的区别只有一个：这里的一切都来自 ``_runtime``，
    不经过 ``torch.cuda``。
    """

    @staticmethod
    def get_device_properties(device: int = 0) -> Dict[str, Any]:
        """后端需要的设备属性。

        Returns:
            含 ``max_shared_mem``（上游 ``_init_handles`` 用它做越界检查）与 ``name``。
            ``max_shared_mem`` 取设备属性报的真实值 128 KiB；实际启动时如果超过 48 KiB，
            ``load_binary`` 会对函数单独放开上限。
        """
        props = _runtime.properties(device)
        return {"max_shared_mem": props["shared_mem_per_block"], "name": props["name"]}

    @staticmethod
    def load_binary(name: str, kernel: bytes, shared: int, device: int = 0):
        """装载 cubin 并取出 kernel 函数。

        Args:
            name: kernel 符号名（``metadata.name``）。
            kernel: ``cubin`` 阶段的产物（已由 lld 链成 ET_EXEC）。
            shared: 动态 shared 字节数；超过 48 KiB 时给函数单独放开上限。
            device: 设备号（context 是进程级的，这里只用来做一次设备绑定）。

        Returns:
            ``(module, function, n_regs, n_spills, n_max_threads)``。寄存器统计厂商工具链
            没有等价入口，给 0；``n_max_threads`` 给设备属性里的真实值 4096。

        Raises:
            RuntimeError: 装载失败，或装载成功但取不到符号（通常是 IR 少了
                ``iluvatar_kernel`` 标记，此时 rc=500）。
        """
        key = str(hash(kernel))
        cached = _MODULE_CACHE.get(key)
        if cached is not None:
            return cached[0], cached[1], 0, 0, 4096

        _runtime.set_device(device)
        lib = _LibCuda.get()
        mod = ctypes.c_void_p()
        _check("cuModuleLoadData", lib.cuModuleLoadData(ctypes.byref(mod),
                                                       ctypes.c_char_p(kernel)))
        fn = ctypes.c_void_p()
        rc = lib.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode())
        if rc != 0:
            raise RuntimeError(
                f"cuModuleGetFunction({name}) 失败 rc={rc} {_errstr(rc)}；"
                "若为 500，先确认改写阶段给 kernel 加了 iluvatar_kernel 标记"
            )
        # 动态 shared 不需要 opt-in：这张卡默认上限就是 128 KiB，132 KiB 才报 invalid argument，
        # 而 cuFuncSetAttribute(8, N) 只能**收紧**（probes/153 实测）。所以这里不调它，
        # 免得把本来能跑的内核设成跑不了。
        _MODULE_CACHE[key] = (mod, fn)
        return mod, fn, 0, 0, 4096


def _device_addr(value: Any) -> int:
    """把「设备指针的三种写法」统一成整数地址。

    为什么要单独一个函数：``int(ctypes.c_void_p(...))`` 在 CPython 上**不是**取地址——
    ctypes 给 ``c_void_p`` 实现了 ``__int__``，走的是 ``int(bytes(c_void_p))``，也就是把 8 字节
    地址当成十进制字符串去解析。地址字节碰巧全是 ASCII 数字时它「能用」，否则抛
    ``ValueError: invalid literal for int() with base 10: b'\\x00\\x00\\x00\\x02...'``
    ——报错信息里连指针都不会出现，极难往「取地址」上想（2026-11-07 在本机复现过）。

    Args:
        value: ``torch.Tensor``（取 ``data_ptr()``）/ ``ctypes.c_void_p`` / 8 字节 ``bytes`` / int。

    Returns:
        设备地址整数。

    Raises:
        TypeError: 既不是 Tensor 也不是指针也不是整数。
    """
    if hasattr(value, "data_ptr"):
        return int(value.data_ptr())
    if isinstance(value, ctypes.c_void_p):
        return int(value.value or 0)
    if isinstance(value, (bytes, bytearray)) and len(value) == 8:
        return int.from_bytes(bytes(value), "little")
    if isinstance(value, int):
        return value
    raise TypeError(f"无法当作设备指针：{type(value).__name__}")


def _pack_arg(value: Any, ty: str) -> ctypes.c_void_p:
    """把一个 Python 实参打包成 ``cuLaunchKernel`` 要的「指向参数值的指针」。

    Args:
        value: 实参。``*`` 开头的类型是设备指针（``torch.Tensor`` / int / c_void_p）。
        ty: Triton 签名里的类型字符串。

    Returns:
        指向该参数值的 void 指针（ctypes 会保活到调用结束，调用方不要留引用）。

    Raises:
        RuntimeError: 遇到还没覆盖的类型——宁可显式报错，也不要猜一个宽度传下去。
    """
    if ty.startswith("*"):
        addr = _device_addr(value)
        return ctypes.cast(ctypes.pointer(ctypes.c_void_p(addr)), ctypes.c_void_p)
    if ty in ("i8", "u8"):
        return ctypes.cast(ctypes.pointer(ctypes.c_int8(int(value))), ctypes.c_void_p)
    if ty in ("i16", "u16"):
        return ctypes.cast(ctypes.pointer(ctypes.c_int16(int(value))), ctypes.c_void_p)
    if ty in ("i32", "u32"):
        return ctypes.cast(ctypes.pointer(ctypes.c_int32(int(value))), ctypes.c_void_p)
    if ty in ("i64", "u64"):
        return ctypes.cast(ctypes.pointer(ctypes.c_int64(int(value))), ctypes.c_void_p)
    if ty == "fp32":
        return ctypes.cast(ctypes.pointer(ctypes.c_float(float(value))), ctypes.c_void_p)
    if ty == "fp64":
        return ctypes.cast(ctypes.pointer(ctypes.c_double(float(value))), ctypes.c_void_p)
    if ty == "fp16":
        # 半精度标量按 IEEE 754 binary16 打包：kernel 侧读到的是 2 字节，宽度错一位就是读垃圾。
        raw = struct.pack("<e", float(value))
        return ctypes.cast(ctypes.pointer(ctypes.c_uint16(
            int.from_bytes(raw, "little"))), ctypes.c_void_p)
    raise RuntimeError(f"参数类型 {ty} 还没有打包规则（见 corex_torch._triton._pack_arg）")


class CorexTorchLauncher:
    """把 ``kernel[grid](args...)`` 翻成 ``cuLaunchKernel``。

    与 ``triton_corex.driver.CorexLauncher`` 的差异：stream 与 global scratch 都由本仓库的
    运行时提供（scratch 走 ``_runtime.malloc``，一次性、按大小缓存）。
    """

    #: global scratch 缓冲（Triton 3.4 起 kernel 末尾多一个指针参数，profile 用）。
    _scratch_ptr = 0
    _scratch_size = 0

    def __init__(self, src, metadata) -> None:
        self.src = src
        self.metadata = metadata
        #: 真正的内核参数：``constexpr`` 是编译期常量，不出现在 cubin 参数表里。
        self.arg_types = [ty for _, ty in src.signature.items() if ty != "constexpr"]
        size = int(getattr(metadata, "global_scratch_size", 0) or 0)
        if size > 0 and size > CorexTorchLauncher._scratch_size:
            CorexTorchLauncher._scratch_ptr = int(_runtime.malloc(size).value or 0)
            CorexTorchLauncher._scratch_size = size

    def __call__(
        self,
        gridX: int, gridY: int, gridZ: int,
        stream: int,
        function,
        packed_metadata,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        *args,
    ) -> None:
        """启动内核。

        Args:
            gridX/gridY/gridZ: 网格。
            stream: 流句柄（0 = 默认流）。
            function: ``load_binary`` 给出的函数句柄。
            packed_metadata: Triton 打包好的元数据，这里不用（参数已在 ``arg_types`` 里）。
            launch_metadata: 同上，仅用于 profiler 钩子。
            launch_enter_hook: Triton 的进入钩子；本后端不接 profiler，忽略。
            launch_exit_hook: 同上。
            *args: 内核实参，顺序与 ``src.signature`` 一致。

        Raises:
            RuntimeError: 实参个数与签名不符，或 ``cuLaunchKernel`` 返回非 0。
        """
        if len(args) != len(self.arg_types):
            raise RuntimeError(
                f"参数个数不符：签名要求 {len(self.arg_types)} 个 {self.arg_types}，收到 {len(args)} 个"
            )
        packed = [_pack_arg(v, ty) for v, ty in zip(args, self.arg_types)]
        # 末尾补 global scratch 指针：没有需求就传 0。
        packed.append(ctypes.cast(ctypes.pointer(ctypes.c_void_p(
            CorexTorchLauncher._scratch_ptr)), ctypes.c_void_p))
        params = (ctypes.c_void_p * len(packed))(*packed)
        warp = getattr(self.metadata.target, "warp_size", TARGET_WARP_SIZE)
        block = int(self.metadata.num_warps) * int(warp)
        shared = int(self.metadata.shared)
        _check("cuLaunchKernel", _LibCuda.get().cuLaunchKernel(
            function,
            int(gridX), int(gridY), int(gridZ),
            block, 1, 1,
            shared,
            ctypes.c_void_p(stream) if stream else None,
            params,
            None,
        ))


class CorexTorchDriver:
    """``triton.runtime.driver.active`` 的实现。

    从 ``DriverBase`` 起（不是 ``GPUDriver``）：后者的 ``__init__`` 里有 ``import torch``
    并去读 ``torch.cuda``，而本仓库跑在 CPU 版 torch 2.8 上。``DriverBase.__init__`` 是空实现，
    所以我们自己装 Triton 3.4 期待的那 4 个钩子。
    """

    def __init__(self) -> None:
        # 这 4 个属性是 triton 内部按「属性」而不是「方法」来调的（见 GPUDriver.__init__），
        # 所以必须是绑定好的可调用对象，不能只留同名方法。
        self.get_current_device = _runtime.current_device
        self.set_current_device = _runtime.set_device
        self.get_current_stream = lambda device=0: int(_runtime.Stream.default().handle.value or 0)
        self.get_device_capability = lambda device=0: (TARGET_ARCH, 0)
        self.utils = CorexTorchUtils()
        self.launcher_cls = CorexTorchLauncher

    @classmethod
    def is_active(cls) -> bool:
        """有 CoreX 用户态库就算激活。"""
        return os.path.exists(os.path.join(_runtime.COREX_HOME, "lib64", "libcuda.so.1"))

    def get_current_target(self):
        """当前编译目标。"""
        from triton.backends.compiler import GPUTarget

        return GPUTarget("corex", TARGET_ARCH, TARGET_WARP_SIZE)

    def get_active_torch_device(self):
        """Triton 3.4 的抽象方法：当前活动设备对应的 torch device。

        这里返回字符串而不是 ``torch.device``：本模块不 import torch（见文件头约束）。
        Triton 只在 profiler / 工具路径上用它，返回等价的字符串不影响编译与启动。
        """
        return f"corex:{_runtime.current_device()}"

    def get_benchmarker(self):
        """默认用 Triton 自带的 ``do_bench``：它只依赖 stream 与事件。"""
        from triton.testing import do_bench

        return do_bench

    def get_empty_cache_for_benchmark(self, size: int = 256 * 1024 * 1024):
        """``do_bench`` 用它刷 L2；返回一块设备缓冲的句柄（不是 torch 张量）。"""
        return _runtime.malloc(int(size))

    def clear_cache(self, cache) -> None:
        """``do_bench`` 计时前刷 L2：整块清零。"""
        size = 256 * 1024 * 1024
        _runtime.memcpy(cache, cache, size, _runtime.MEMCPY_DEVICE_TO_DEVICE)

    def get_device_interface(self):
        """给 ``do_bench`` 之类工具用的设备接口。"""
        if getattr(self, "_device_interface", None) is None:
            self._device_interface = CorexTorchDeviceInterface()
        return self._device_interface


class CorexTorchDeviceInterface:
    """``triton.testing.do_bench`` 需要的设备能力：同步、计时、属性。

    少了这层，``do_bench`` 会在 ``di.synchronize()`` 上抛 ``AttributeError``，
    报错位置在 Triton 自己的 testing 模块里，看不出是驱动缺接口。
    """

    Event = _runtime.Event

    @staticmethod
    def synchronize() -> None:
        _runtime.synchronize()

    @staticmethod
    def get_current_device() -> int:
        return _runtime.current_device()

    @staticmethod
    def get_current_stream(device: int = 0) -> int:
        return int(_runtime.Stream.default().handle.value or 0)

    @staticmethod
    def get_device_properties(device: int = 0) -> Dict[str, Any]:
        return CorexTorchUtils.get_device_properties(device)

    @staticmethod
    def empty_cache() -> None:
        _runtime.synchronize()


def activate() -> CorexTorchDriver:
    """注册并激活 corex 后端，返回 driver 实例（幂等）。

    为什么要显式覆盖 ``driver.active``：``CompiledKernel._init_handles`` 走的是
    ``driver.active``，它按后端表顺序挑第一个 ``is_active()`` 为真的后端，而 NVIDIA 那个
    在这张卡上同样为真（libcuda 存在），会拿到 ptxas 那套 launcher 并失败。

    Returns:
        已激活的 driver 实例。
    """
    _ensure_triton_corex_importable()
    from triton.backends import Backend, backends
    from triton.runtime.driver import driver as _driver_config

    from triton_corex.compiler import CorexBackend

    backends["corex"] = Backend(CorexBackend, CorexTorchDriver)
    inst = CorexTorchDriver()
    _driver_config.set_active(inst)
    return inst


def _normalize_grid(grid) -> Tuple[int, int, int]:
    """把一维/二维网格补成 Triton 启动器要求的 3 元组。

    ``CompiledKernel.__getitem__`` 的 runner 直接读 ``grid[0]/grid[1]/grid[2]``，
    它**不做**补齐（补齐是 ``JITFunction.__getitem__`` 才有的）。所以只传 ``(16,)``
    会在 Triton 自己的 compiler.py 里抛 ``IndexError: tuple index out of range``，
    报错位置完全看不出是我们少给了一维。
    """
    dims = tuple(int(x) for x in (grid if isinstance(grid, (tuple, list)) else (grid,)))
    if len(dims) > 3:
        raise ValueError(f"网格最多 3 维，收到 {dims}")
    return dims + (1,) * (3 - len(dims))


def launch(kernel, grid, *args, stream: Optional[int] = None) -> None:
    """启动一个已编译的内核。

    Args:
        kernel: ``compile_kernel`` 的返回值。
        grid: 网格，``int`` 或 1–3 元组。
        *args: 内核实参；设备指针可以是 ``torch.Tensor`` / int / ``c_void_p``。
        stream: 流句柄；``None`` 用默认流。

    Raises:
        RuntimeError: 实参个数与签名不符，或启动返回非 0。
    """
    gx, gy, gz = _normalize_grid(grid)
    if stream is None:
        kernel[(gx, gy, gz)](*args)
    else:
        kernel[(gx, gy, gz)](*args, stream=stream)


def compile_kernel(fn, signature: Dict[str, str], constexprs: Optional[Dict[str, Any]] = None,
                   num_warps: int = 4, num_stages: int = 1):
    """编译一个 ``@triton.jit`` 内核，返回 ``CompiledKernel``（含 ``cubin``）。

    Args:
        fn: ``triton.jit`` 装饰过的函数。
        signature: 参数名到 Triton 类型串的映射（``*fp32`` / ``i32`` / ``constexpr``）。
        constexprs: 编译期常量。
        num_warps: 每块的 warp 数（本平台 warp 布局按 32 算，见 ``TARGET_WARP_SIZE``）。
        num_stages: 流水级数。固定 1：本平台没有 ``cp.async``，多级流水没有硬件通路。

    Returns:
        编译产物。``kernel[grid](*args)`` 即可启动。

    Raises:
        RuntimeError: 编译失败（多半是 IR 里混进了厂商 llc 不认的语法，见
            ``corex-triton/docs/00-FACTS.md`` §9、§14.2）。
    """
    from triton.compiler import ASTSource, compile as triton_compile

    activate()
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs or {},
                    attrs=None)
    return triton_compile(src, target=CorexTorchDriver().get_current_target(),
                          options={"num_warps": num_warps, "num_stages": num_stages})
