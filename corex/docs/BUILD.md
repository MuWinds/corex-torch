# 构建：fork 的 torch wheel、设备名补丁、shim 打包

这份文档回答：**怎么从零编出一个「装完就能用卡」的 torch wheel，以及只改设备后端时怎么快速迭代。**

事实与实测输出在 [`00-FACTS.md`](00-FACTS.md) §19–§21；接口契约在 [`INTERFACE.md`](INTERFACE.md)。

---

## 1. 两种构建形态

| 形态 | 什么时候用 | 怎么做 |
| --- | --- | --- |
| **交付态**：自编 wheel | 给别人用、CI、换机器 | 在 fork 树里 `setup.py bdist_wheel`（纯 CPU 构建），shim 在同一趟里打进包 |
| **开发态**：源码树 + 单独编 shim | 只改 `torch/corex/*.py` 或算子时 | 装上游 CPU wheel，改 `torch/corex/*` 直接生效；shim 用 `tools/build_shim.py` 编 |

交付态的判据是**全新 venv、只 `pip install`、不跑任何编译也能用卡**（`probes/224`）；
开发态不能这么验，因为 shim 是我们自己编的那个 `.so`。

## 2. 环境准备

- **py310**：目标运行时是 python 3.10（厂商 torch 2.1、`corex28` 环境、以及 py310 ABI 的 shim
  都在 py310 上）。登录节点默认 `python3` 是 3.12，别拿它当构建解释器。
- **CMake ≥ 3.27**：torch 2.8 硬要求。装到 `~/venvs/torchbuild` 之后**必须把它的 bin 放进 `PATH`**
  ——`tools/setup_helpers/cmake.py` 是用 `which cmake` 找的，`pip install` 完不 export 等于没装。
- **ninja** 在 PATH 里（同理，`shutil.which("ninja")` 拿不到会把 `None` 拼进命令行）。
- **MAX_JOBS=12**：48 并发会 `cc1plus: fatal error: Killed signal terminated`。
  内存不是瓶颈（1 TB 空闲、ulimit unlimited），是瞬时竞争。
- 下载依赖在**登录节点**做：计算节点没有外网（pip 会一直重试然后失败）。

## 3. 上游 torch 2.8.0+cpu 进隔离 venv（开发态）

```bash
VENV=~/venvs/corex28
python -m venv "$VENV"                       # 不带 --system-site-packages
$VENV/bin/python -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
$VENV/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    filelock jinja2 networkx sympy fsspec typing-extensions numpy
curl -L -C - --retry 30 --retry-all-errors -o ~/wheels/torch-2.8.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl \
    "https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl"
$VENV/bin/python -m pip install --no-deps --no-index ~/wheels/torch-2.8.0+cpu-*.whl
```

为什么这么绕：venv 自带 pip 23.0.1 有名字规范化 bug（`typing-extensions` vs `typing_extensions`），
会把 wheel 丢掉退化成源码构建；而 pip 自带的下载器在这台登录节点上只有约 52 kB/s，
同一 URL 用 `curl` 是 7.1 MB/s。完整证据见 [`00-FACTS.md`](00-FACTS.md) §4。

> 算子层的 `tl` 需要 **triton 3.4**（torch 2.8 的官方配对），本仓库的 venv 里要单独装；
> 否则报 `AttributeError: 'NoneType' object has no attribute 'float32'`。

## 4. 编 fork 的 torch 本体（交付态）

前置：`git clone --depth 1 --branch v2.8.0 --recurse-submodules`（tar 包没有 `.git`，
`setup.py` 初始化不了子模块）。

```bash
export PATH=~/venvs/torchbuild/bin:$PATH     # cmake >= 3.27 在这里
export MAX_JOBS=12
export LC_ALL=C.UTF-8                        # 见下面的 locale 卡点
export USE_CUDA=0 USE_NCCL=0 USE_DISTRIBUTED=0 USE_NNPACK=0
python setup.py bdist_wheel                  # 产物 dist/torch-2.8.0a0+git<sha>-cp310-*.whl
```

六个卡点（每个都浪费过半天）：

| 卡点 | 现象 | 做法 |
| --- | --- | --- |
| CMake 版本 | `no cmake >= 3.27.0 found: [3.22.0, 3.22.0]` | 装新版并把 `bin` 放进 `PATH` |
| 子模块 | `Submodule initialization failed` | 用带 `.git` 的 clone，不要 tar |
| NCCL | `git clone https://github.com/NVIDIA/nccl` 失败 | `USE_NCCL=0 USE_DISTRIBUTED=0` |
| NNPACK | `ModuleNotFoundError: No module named 'six'` | `USE_NNPACK=0` |
| **locale** | `UnicodeDecodeError: 0xe0 in position …`（读 `CMakeCache.txt`） | **删旧缓存 + `LC_ALL=C.UTF-8` 重配**，让缓存本身就是 UTF-8；给 `cmake.py` 的 `open` 加 `errors="replace"` 只兜住一处，不彻底 |
| 并发 | `cc1plus: fatal error: Killed signal terminated` | `MAX_JOBS=12` |

第一次编要清缓存（换解释器/换 `build/` 目录时），之后是增量的。产物验证：
`VERSION 2.8.0a0+git<sha>`、`HAS_COREX_SUBMODULE True`、CPU 矩阵乘正常。

## 5. 设备名补丁（9 个文件，约 150 行）

清单与逐文件作用见 [`00-FACTS.md`](00-FACTS.md) §20。构建相关的两条纪律：

- 补丁必须**同时**改 C++ 与 Python 两侧：只改 `torch/cuda/__init__.py` 会漏掉
  `at::Tensor::cuda()`（它按枚举判断，不看字符串）。
- `torch/corex/*` 里的 `DEVICE_NAME = "cuda"` 与补丁是一套的，改一个不改另一个会出现
  「`device="cuda"` 能用、`torch.corex.xxx` 报设备名不符」这类自相矛盾的现象。

## 6. shim 编进 wheel

wheel 是预编译产物，pip 安装期不编译，所以 shim 必须在**编 torch 的同一趟**里产出并打进包：

| 位置 | 改动 |
| --- | --- |
| `torch/CMakeLists.txt` | 末尾加 `corex_shim` 目标（MODULE，`PREFIX ""`、`OUTPUT_NAME _corex_shim`），链 `torch_python` |
| 同上 | `target_compile_definitions(... TORCH_EXTENSION_NAME=_corex_shim)` |
| `setup.py` 的 `build_extensions()` | 把 `<build>/lib/_corex_shim.so` 拷到 `<build_lib>/torch/corex/_corex_shim.so` |

两个坑：

1. **导出符号名**：`torch/extension.h` 把 `PYBIND11_MODULE` 重定义成依赖 `TORCH_EXTENSION_NAME`
   的版本，不定义它就导出字面量 `PyInit_TORCH_EXTENSION_NAME`，import 报
   `dynamic module does not define module export function (PyInit__corex_shim)`。
   查法：`nm -D --defined-only _corex_shim.so | grep PyInit`。
2. **不要用 CMake 的 `install()`**：bdist_wheel 下安装前缀是 staging 根，
   `install(TARGETS … DESTINATION torch/corex)` 会落到 `<prefix>/torch/torch/corex`（多一层 torch），
   而 wheel 只打 `<build_lib>/torch/**`。在 `setup.py` 里直接拷，路径唯一确定。

**Windows 打包上传的坑**：tar 保留的 mtime 可能比节点的 `.o` 还旧，ninja 认为不用重编，
wheel 里还是旧二进制——装完必须 `touch` 改动过的 C++ 文件。这条已经浪费过一轮。

## 7. 开发态：编 shim 与快速迭代

```bash
$VENV/bin/python corex/tools/build_shim.py          # 登录节点即可，编译期不需要 CUDA
# → corex/src/csrc/corex_shim.cpp → torch/corex/_corex_shim.so
```

- 编译期**不 include 任何 CUDA 头、不链 `libcudart`**（全部 `dlopen`/`dlsym`），所以登录节点能编；
  `ldd` 结果只有 `libc10/libtorch_cpu/libtorch/libtorch_python/libdl`。
- 编译器用系统 **g++ 10.3.1** 就够（首次 26.7 s，之后 ninja 缓存命中 0 s）。

改 Python 侧（`torch/corex/*.py`）时的迭代纪律：

```bash
python tools/remote.py pushdir ./torch/corex <远端 fork 树>/torch/corex
python -m scow put probes/xxx.sh <远端队列>/queue/xxx.sh     # 换了名字再投
```

- `pushdir` **推完自动清 `__pycache__`**（tar 保留 Windows mtime，比远端 `.pyc` 更旧，
  CPython 会继续用旧字节码——这条单独造成过两轮「改了没生效」）。
- 推完 `src` **再**排队探针，不要在同一条命令里连着做：队列可能在解包中途启动。
- 算子层要 `COREX_TRITON_SRC` 指向 `corex-triton/src`（它没打包安装）。

## 8. 验证（交付态）

全新 venv，只 `pip install` 编出来的 wheel，不跑任何编译：

```bash
bash corex/probes/224_final_check.sh        # 脚本里用 $PY 指定解释器
```

期待输出（原样）：

```
SHIM_FROM_WHEEL=427064      SYMBOL=PyInit__corex_shim
CUDA_AVAILABLE True   CUDA_COUNT 2   X_CUDA cuda:0   IS_CUDA True   ADD_CUDA True
STREAM Stream   VERSION_CUDA 10.2   IS_BUILT True
算子矩阵 64/64 OK     mini decoder PASS（6.858e-06、argmax 1.000）
```

`SHIM_FROM_WHEEL` 是 `.so` 的字节数——它证明 shim 来自 wheel 而不是现场的第二次编译。

## 9. 时间与体积（实测）

| 项目 | 数值 |
| --- | --- |
| venv（只有 torch CPU + numpy 等 8 个包） | 872 MB |
| py310 fork 全量构建目标数 | 5881 |
| wheel | 86.6 MB |
| shim `.so` | 248 384 B（开发态）/ 427 064 B（打包后） |
| 首次 shim 编译 | 26.4–26.7 s |
