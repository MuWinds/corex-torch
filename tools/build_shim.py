"""把 ``src/csrc/corex_shim.cpp`` 编成一个能被 ``corex_torch`` 直接 import 的扩展模块。

为什么要有这个脚本，而不是手敲 ``torch.utils.cpp_extension.load``：

* **编译必须发生在登录节点**（没有 ``/usr/local/corex``、没有 GPU 驱动），所以命令行必须能在
  一台「只有 torch + g++」的机器上复现；把 include/lib 路径、C++17、``-ldl`` 这些一次性钉死，
  比每次回忆参数可靠。
* 产物要落在 ``src/corex_torch/_corex_shim.so``：``_device.py`` 用 ``from . import _corex_shim``
  取它，包内相对导入只看文件位置，不看构建目录。
* ``*.so`` 已在 ``.gitignore`` 里——它是构建产物，不进仓库。

编译期**不依赖 CUDA 工具链**：shim 里所有 ``cuda*`` 都是运行时 ``dlopen``/``dlsym``
（原因见 ``corex_shim.cpp`` 的头注释），所以这里既不链 ``libcudart``、也不需要 ``nvcc``。

用法（一台有 torch 2.8 + g++ 的机器）::

    ~/venvs/corex28/bin/python tools/build_shim.py            # 默认产物进 src/corex_torch/
    ~/venvs/corex28/bin/python tools/build_shim.py --verbose  # 打印 ninja 的完整编译命令
    ~/venvs/corex28/bin/python tools/build_shim.py --out /tmp/_corex_shim.so --clean

Module: ``corex-torch/tools``。
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import time

#: 扩展模块的 **import 名**。必须与产物文件名一致：CPython 按文件名找
#: `PyInit_<文件名>` 符号（`_corex_shim.so` → `PyInit__corex_shim`），对不上会报
#: `ImportError: dynamic module does not define module export function`（实测踩过）。
#: C++ 侧的命名空间仍叫 `corex_shim`，只有**模块/文件名**带下划线前缀（表示私有）。
MODULE_NAME = "_corex_shim"

#: 仓库根目录：本文件在 ``<repo>/tools/`` 下。
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _default_src() -> pathlib.Path:
    """shim 源码默认路径。"""
    return REPO_ROOT / "src" / "csrc" / "corex_shim.cpp"


def _default_out() -> pathlib.Path:
    """产物默认路径：包内，供 ``from . import _corex_shim`` 使用。"""
    return REPO_ROOT / "src" / "corex_torch" / "_corex_shim.so"


def _default_build_dir() -> pathlib.Path:
    """ninja 的工作目录（``build/`` 已在 .gitignore 里）。"""
    return REPO_ROOT / "build" / "shim"


def build(src: pathlib.Path, out: pathlib.Path, build_dir: pathlib.Path,
          clean: bool, verbose: bool) -> pathlib.Path:
    """编译 shim 并把产物拷到 ``out``。

    Args:
        src: ``corex_shim.cpp`` 路径。
        out: 最终产物路径（会被父目录自动创建）。
        build_dir: ninja 工作目录。
        clean: 先删掉 build_dir 强制全量重编。
        verbose: 打印完整编译命令。

    Returns:
        实际产物的路径。

    Raises:
        FileNotFoundError: 源码不存在。
        RuntimeError: 编译失败，或编出来的东西不是 .so。
    """
    if not src.is_file():
        raise FileNotFoundError(f"找不到 shim 源码：{src}")

    # torch 的导入放在函数里：本脚本的 --help 不该因为没装 torch 就跑不起来。
    import torch
    from torch.utils import cpp_extension

    if clean and build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[build_shim] torch {torch.__version__} cuda={torch.version.cuda}")
    print(f"[build_shim] 源码 {src}")
    print(f"[build_shim] 工作目录 {build_dir}")
    # -std=c++17 是 torch 2.8 扩展的下限；-O2 与探针 102 实测口径一致。
    # -ldl 为了 dlopen/dlsym：glibc 2.34 之后它们并进 libc，但老一点的发行版仍需显式链接。
    extra_cflags = ["-O2", "-std=c++17"]
    extra_ldflags = ["-ldl"]

    started = time.perf_counter()
    module = cpp_extension.load(
        name=MODULE_NAME,
        sources=[str(src)],
        build_directory=str(build_dir),
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        is_python_module=True,
        with_cuda=False,          # 显式关掉：本机没有 CUDA，也不许它去找 nvcc
        verbose=verbose,
    )
    elapsed = time.perf_counter() - started

    built = build_dir / f"{MODULE_NAME}.so"
    if not built.is_file():
        raise RuntimeError(f"编译返回成功但没找到产物：{built}")
    shutil.copy2(built, out)

    # 自检：模块已经在本进程里 import 过了，这里只碰「不需要设备」的接口。
    # 特别注意**不要**在这里调 device_count()/init()——那会 dlopen libcudart，
    # 而登录节点没有 /usr/local/corex，会直接抛错（这正是 A 段验收的分界线）。
    exported = sorted(name for name in dir(module) if not name.startswith("_"))
    initialized = module.is_initialized()
    if initialized:
        raise RuntimeError("刚 import 的 shim 报告 is_initialized() == True，不应该")

    print(f"[build_shim] 导出接口 {exported}")
    print(f"[build_shim] is_initialized() = {initialized}（未 dlopen 运行时，符合预期）")
    print(f"[build_shim] 编译耗时 {elapsed:.1f} s")
    print(f"[build_shim] 产物 {out} ({out.stat().st_size} B)")
    return out


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="编译 corex_shim 扩展模块")
    parser.add_argument("--src", type=pathlib.Path, default=None,
                        help=f"shim 源码（默认 {_default_src()}）")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help=f"产物路径（默认 {_default_out()}）")
    parser.add_argument("--build-dir", type=pathlib.Path, default=None,
                        help=f"ninja 工作目录（默认 {_default_build_dir()}）")
    parser.add_argument("--clean", action="store_true", help="先清空构建目录再编")
    parser.add_argument("--verbose", action="store_true", help="打印完整编译命令")
    args = parser.parse_args(argv)

    src = args.src or _default_src()
    out = args.out or _default_out()
    build_dir = args.build_dir or _default_build_dir()
    try:
        build(src.resolve(), out.resolve(), build_dir.resolve(), args.clean, args.verbose)
    except BaseException as exc:  # noqa: BLE001 - 统一在入口打一条能看懂的失败
        print(f"[build_shim] 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("BUILD_SHIM_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
