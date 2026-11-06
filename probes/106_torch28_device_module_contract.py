"""探针 106：torch 2.8 到底会调用自定义 device module 的哪些函数？

为什么做这个探测：``rename_privateuse1_backend`` 的 docstring 列了三组契约（AMP / random /
common），但那是「建议实现」还是「必需实现」看不出来。这里用一个会记录每次属性访问的
替身模块（TracingModule）挂进 ``torch._register_device_module``，然后逐条跑常见 torch API，
把「哪个 API 摸到了 device module 的哪个函数」全部抓出来——这份表就是文档里
「device module 必需函数」一节的证据。

跑法（在 corex28 venv 里）::

    ~/venvs/corex28/bin/python probes/106_torch28_device_module_contract.py

Module: ``corex-torch/probes``。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback

BACKEND = "corex"
_PROBE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_shim_builder():
    """把 102 号探针当模块载入，复用它的 C++ shim（编译缓存命中，0 秒）。

    文件名以数字开头，不能用普通 import，只能按路径载入。

    Returns:
        102 号探针模块；失败时返回 None（没有 shim 也能观察一部分 Python 侧行为）。
    """
    path = os.path.join(_PROBE_DIR, "102_torch28_cpp_ext.py")
    if not os.path.exists(path):
        print(f"警告：找不到 {path}，跳过 shim 相关试验")
        return None
    spec = importlib.util.spec_from_file_location("probe102", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TracingModule:
    """记录每一次属性访问的 device module 替身。

    属性访问发生在 torch 内部，所以这张 ``hits`` 表就是「torch 主动向设备后端要了什么」
    的直接证据。所有函数都返回无害值，让被观测的 API 能继续往下走。
    """

    def __init__(self) -> None:
        object.__setattr__(self, "hits", [])
        object.__setattr__(self, "ctx", "-")

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        self.hits.append((self.ctx, name))

        def _stub(*args, **kwargs):
            if name == "is_available":
                return True
            if name == "device_count":
                return 1
            if name == "current_device":
                return 0
            if name == "_is_in_bad_fork":
                return False
            if name == "get_amp_supported_dtype":
                import torch

                return [torch.float32]
            if name == "get_rng_state":
                import torch

                return torch.zeros(1, dtype=torch.uint8)
            return None

        return _stub


def main() -> int:
    import torch

    shim_mod = _load_shim_builder()
    if shim_mod is not None:
        try:
            ext = shim_mod._build_shim()
            ext.register_shim()
            print("C++ shim 已注册（分配器 + guard impl）")
            lib = torch.library.Library("aten", "IMPL")
            lib.impl(
                "empty.memory_format",
                lambda size, *, dtype=None, layout=None, device=None,
                pin_memory=None, memory_format=None: ext.empty_pu1(
                    list(size), torch.float32 if dtype is None else dtype),
                "PrivateUse1",
            )
            print("aten::empty.memory_format 的 PrivateUse1 实现已从 Python 注册")
        except BaseException:  # noqa: BLE001
            traceback.print_exc(limit=3)
            ext = None
    else:
        ext = None

    torch.utils.rename_privateuse1_backend(BACKEND)
    tracer = TracingModule()
    torch._register_device_module(BACKEND, tracer)

    def run(label: str, fn) -> None:
        """跑一条 API，并把这一段里触发的 device module 访问单独列出来。"""
        tracer.ctx = label
        before = len(tracer.hits)
        try:
            value = fn()
            note = "OK"
        except BaseException as exc:  # noqa: BLE001
            first = str(exc).strip().splitlines()
            note = f"EXC {type(exc).__name__}: {(first[0] if first else '')[:80]}"
            value = None
        accessed = [n for _, n in tracer.hits[before:]]
        print(f"\n[{label}] {note}")
        print(f"    摸到的 device module 函数: {accessed if accessed else '（一个都没摸）'}")
        if value is not None and not isinstance(value, torch.Tensor):
            print(f"    返回值: {value!r}"[:160])

    print("\n=== A. 属性/发现类 API ===")
    run("torch.corex.is_available()", lambda: torch.corex.is_available())
    run("torch.accelerator.is_available()", lambda: torch.accelerator.is_available())
    run("torch.accelerator.device_count()", lambda: torch.accelerator.device_count())
    run("torch.accelerator.current_device_index()", lambda: torch.accelerator.current_device_index())
    run("torch.accelerator.current_accelerator()", lambda: torch.accelerator.current_accelerator())

    print("\n=== B. 随机数（rename docstring 的 random 组）===")
    run("torch.manual_seed(1234)", lambda: torch.manual_seed(1234))
    run("torch.random.manual_seed(1234)", lambda: torch.random.manual_seed(1234))
    run("torch.randn(4, device='corex')", lambda: torch.randn(4, device=BACKEND))

    print("\n=== C. 张量创建与搬运 ===")
    run("torch.empty(2, device='corex')", lambda: torch.empty(2, device=BACKEND))
    run("torch.zeros(2, device='corex')", lambda: torch.zeros(2, device=BACKEND))
    run("torch.tensor([1.], device='corex')", lambda: torch.tensor([1.0], device=BACKEND))
    run("torch.tensor([1.]).to('corex')", lambda: torch.tensor([1.0]).to(BACKEND))
    run("torch.device('corex').type", lambda: torch.device(BACKEND).type)

    print("\n=== D. generate_methods_for_privateuse1_backend 产出的方法 ===")
    run("generate_methods(...)", lambda: torch.utils.generate_methods_for_privateuse1_backend())
    t = torch.tensor([1.0, 2.0])
    run("cpu_t.is_corex", lambda: t.is_corex)
    run("cpu_t.corex()", lambda: t.corex())

    print("\n=== E. 完整访问表（按调用顺序去重）===")
    seq: list[str] = []
    for ctx, name in tracer.hits:
        item = f"{ctx} -> {name}"
        if item not in seq:
            seq.append(item)
    for item in seq:
        print("   ", item)
    print("PROBE106_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
