"""探针 101：只用 Python 能让 ``torch.empty(4, device="corex")`` 成功吗？

为什么做这个探测：docs/03-TORCH28-API.md 的结论是「分配器必须落在 C++」还是
「Python 侧就够了」，直接决定第一阶段要不要写 C++ 扩展。凭印象写不算数，必须逐层
试到失败，并把**失败在哪一句**记下来。

为什么每个阶段单独起进程：``torch.utils.rename_privateuse1_backend`` 每个进程只能调
一次（见 torch/utils/backend_registration.py 的 docstring），同进程里连做几个阶段会
互相污染。所以本脚本用 ``--stage`` 自我重入，父进程只负责收集。

跑法（在 corex28 venv 里）::

    ~/venvs/corex28/bin/python probes/101_torch28_privateuse1.py           # 全部阶段
    ~/venvs/corex28/bin/python probes/101_torch28_privateuse1.py --stage 2  # 单跑某段

Module: ``corex-torch/probes``。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import traceback

BACKEND = "corex"


def _try(label: str, fn) -> None:
    """执行一段试验并打印结果；异常只记类型与首行，不中断后续试验。

    Args:
        label: 打印用的试验名。
        fn: 无参可调用对象。
    """
    try:
        value = fn()
    except BaseException as exc:  # noqa: BLE001 - 这里就是要抓到所有失败形态
        first = str(exc).strip().splitlines()
        head = first[0] if first else ""
        print(f"  [{label}] EXC {type(exc).__name__}: {head[:300]}")
        if len(first) > 1:
            print(f"  [{label}] EXC 续行: {first[1].strip()[:300]}")
    else:
        print(f"  [{label}] OK -> {value!r}"[:400])


def stage1_baseline() -> None:
    """不改任何东西：未 rename 时 PrivateUse1 的表现。"""
    import torch

    print("默认 backend 名 =", torch._C._get_privateuse1_backend_name())
    print("torch.device('privateuseone') =", torch.device("privateuseone"))
    _try("empty(privateuseone)", lambda: torch.empty(4, device="privateuseone"))
    _try("zeros(privateuseone)", lambda: torch.zeros(4, device="privateuseone"))
    _try("accelerator.is_available", lambda: torch.accelerator.is_available())
    _try("accelerator.current_accelerator", lambda: torch.accelerator.current_accelerator())


def stage2_rename_only() -> None:
    """只 rename，不装任何 module、不注册任何 C++ 实现。"""
    import torch

    torch.utils.rename_privateuse1_backend(BACKEND)
    print("rename 后 backend 名 =", torch._C._get_privateuse1_backend_name())
    print("torch.device('corex') =", torch.device(BACKEND))
    print("torch.device('corex', 1) =", torch.device(BACKEND, 1))
    print("_dispatch_key_for_device('corex') =", torch._C._dispatch_key_for_device(BACKEND))
    # 下面四条是「Python 侧能不能自己立起设备」的核心：预期全部失败。
    _try("empty(4, corex)", lambda: torch.empty(4, device=BACKEND))
    _try("zeros(4, corex)", lambda: torch.zeros(4, device=BACKEND))
    _try("tensor([1,2], corex)", lambda: torch.tensor([1, 2], device=BACKEND))
    _try("randn(4, corex)", lambda: torch.randn(4, device=BACKEND))
    _try("accelerator.is_available", lambda: torch.accelerator.is_available())
    _try("accelerator.current_accelerator", lambda: torch.accelerator.current_accelerator())


def _make_device_module():
    """造一个最小 device module。

    这些函数名不是猜的：torch/utils/backend_registration.py 的
    ``rename_privateuse1_backend`` docstring 逐条列了 AMP/random/common 三组契约。
    这里故意全部实现成「能用但没有卡」的桩，用来观察 torch 会主动调用哪几个。

    Returns:
        一个带静态方法的类（torch._register_device_module 只要求属性可访问）。
    """

    class FakeCorexModule:
        """最小的 corex device module 桩。"""

        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def _is_in_bad_fork() -> bool:
            return False

        @staticmethod
        def manual_seed_all(seed: int) -> None:
            return None

        @staticmethod
        def get_rng_state(device="corex"):
            import torch

            return torch.zeros(1, dtype=torch.uint8)

        @staticmethod
        def set_rng_state(new_state, device="corex") -> None:
            return None

        @staticmethod
        def get_amp_supported_dtype():
            import torch

            return [torch.float32]

        @staticmethod
        def _get_device_properties(device=None):
            return {"name": "fake", "major": 0, "minor": 0, "total_memory": 0}

        @staticmethod
        def _lazy_init() -> None:
            return None

    return FakeCorexModule


def stage3_device_module() -> None:
    """rename + 注册 device module：看 torch.corex.* 能不能用、能用到什么程度。"""
    import torch

    torch.utils.rename_privateuse1_backend(BACKEND)
    mod = _make_device_module()
    _try("_register_device_module", lambda: torch._register_device_module(BACKEND, mod))
    print("hasattr(torch,'corex') =", hasattr(torch, "corex"))
    _try("torch.corex.is_available()", lambda: torch.corex.is_available())
    _try("torch.corex.device_count()", lambda: torch.corex.device_count())
    _try("accelerator.is_available", lambda: torch.accelerator.is_available())
    _try("accelerator.device_count", lambda: torch.accelerator.device_count())
    _try("accelerator.current_accelerator", lambda: torch.accelerator.current_accelerator())
    _try("empty(4, corex) 仍然失败？", lambda: torch.empty(4, device=BACKEND))
    # generate_methods 打的是 monkey patch，不需要 kernel，先看它挂上了什么。
    _try(
        "generate_methods_for_privateuse1_backend",
        lambda: torch.utils.generate_methods_for_privateuse1_backend(),
    )
    t = torch.tensor([1.0, 2.0])
    print("hasattr(Tensor,'is_corex') =", hasattr(torch.Tensor, "is_corex"))
    print("hasattr(Tensor,'corex')   =", hasattr(torch.Tensor, "corex"))
    _try("cpu_tensor.is_corex", lambda: t.is_corex)
    _try("cpu_tensor.corex()", lambda: t.corex())
    _try("cpu_tensor.to('corex')", lambda: t.to(BACKEND))


def stage4_fallback_redispatch() -> None:
    """给 PrivateUse1 挂 Python fallback，看第一跳到底进哪个 aten op。

    这条路是「能不能一个 kernel 都不写、靠 fallback 把 op 转回 CPU」的关键：
    ``torch.library.Library("_", "IMPL").fallback(fn, "PrivateUse1")`` 只在全局
    namespace ``_`` 上允许（torch/library.py 的 fallback 源码里有断言）。
    """
    import torch

    torch.utils.rename_privateuse1_backend(BACKEND)
    calls: list[str] = []

    def fb(op, *args, **kwargs):
        calls.append(str(op))
        raise RuntimeError(f"PYFALLBACK_HIT:{op}")

    lib = torch.library.Library("_", "IMPL")
    try:
        lib.fallback(fb, "PrivateUse1")
        print("  [fallback 注册] OK")
    except BaseException as exc:  # noqa: BLE001
        print(f"  [fallback 注册] EXC {type(exc).__name__}: {exc}")
        return
    _try("empty(4, corex) 走 fallback？", lambda: torch.empty(4, device=BACKEND))
    _try("zeros(4, corex) 走 fallback？", lambda: torch.zeros(4, device=BACKEND))
    print("  fallback 收到的 op 顺序 =", calls)
    # fallthrough 版本：把 PrivateUse1 的 fallthrough 交给 dispatcher 继续走别名键，
    # 看会不会落到 CPU（如果会，那『复用 CPU kernel』就有现成通路）。
    lib2 = torch.library.Library("_", "IMPL")
    try:
        lib2.fallback(torch.library.fallthrough_kernel, "PrivateUse1")
        print("  [fallthrough 注册] OK")
    except BaseException as exc:  # noqa: BLE001
        print(f"  [fallthrough 注册] EXC {type(exc).__name__}: {exc}")
        return
    _try("fallthrough 后 empty(4, corex)", lambda: torch.empty(4, device=BACKEND))
    _try("fallthrough 后 zeros(4, corex)", lambda: torch.zeros(4, device=BACKEND))


def stage5_python_op_impl() -> None:
    """rename + 用 Python 注册单个 aten 算子的 PrivateUse1 实现，验证注册通路本身。"""
    import torch

    torch.utils.rename_privateuse1_backend(BACKEND)
    lib = torch.library.Library("aten", "IMPL")
    try:
        lib.impl("empty.memory_format", lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("PYIMPL_EMPTY_HIT")), "PrivateUse1")
        print("  [impl empty.memory_format] OK")
    except BaseException as exc:  # noqa: BLE001
        print(f"  [impl empty.memory_format] EXC {type(exc).__name__}: {exc}")
        return
    _try("empty(4, corex)", lambda: torch.empty(4, device=BACKEND))
    _try("zeros(4, corex)（zeros 先走 empty 还是 zeros?）",
         lambda: torch.zeros(4, device=BACKEND))


STAGES = {
    "1": ("基线（未 rename）", stage1_baseline),
    "2": ("只 rename", stage2_rename_only),
    "3": ("rename + device module", stage3_device_module),
    "4": ("PrivateUse1 fallback / fallthrough", stage4_fallback_redispatch),
    "5": ("Python 注册单个 aten 算子", stage5_python_op_impl),
}


def _run_one(stage: str) -> int:
    print(f"--- stage {stage}: {STAGES[stage][0]} ---")
    try:
        STAGES[stage][1]()
    except BaseException:  # noqa: BLE001
        traceback.print_exc(limit=6)
        return 1
    return 0


def main(argv: list[str]) -> int:
    if "--stage" in argv:
        return _run_one(argv[argv.index("--stage") + 1])
    import torch

    print("=" * 72)
    print("torch", torch.__version__, "cuda", torch.version.cuda,
          "cuda.is_available", torch.cuda.is_available())
    print("=" * 72)
    rc = 0
    for stage in STAGES:
        # 每个阶段一个全新解释器：rename 只能调一次，进程内串跑会互相污染。
        proc = subprocess.run(
            [sys.executable, __file__, "--stage", stage],
            capture_output=True, text=True, timeout=300,
        )
        print(proc.stdout.rstrip())
        if proc.returncode != 0:
            rc = 1
            print(textwrap.indent(proc.stderr.rstrip(), "  [stderr] "))
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
