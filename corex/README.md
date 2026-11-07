# corex-torch

这份文档回答：**这是什么、怎么装、怎么跑第一个张量、目录里各是什么、有哪些硬限制。**

把 **PyTorch 2.8** 接到天数智芯 CoreX（天垓100 / BI-V100）上：fork 的 torch 树 + 自编 wheel +
9 个文件约 150 行补丁，把 `corex` 注册成一个 PrivateUse1 设备，并把设备名别名成 `cuda`——
所以 `model.cuda()`、`device="cuda"`、`torch.cuda.synchronize()` 都不用改。
算子下沉成 Triton 内核（由姊妹仓库 [`corex-triton`](../../corex-triton/corex/README.md) 编译）。

## 为什么不能直接装 CUDA 版 wheel

| 环节 | 上游 torch 2.8 | 这台机器实际 |
| --- | --- | --- |
| 需要的 CUDA | 11.8 / 12.6 / 12.8 | CoreX 兼容层只声明 **CUDA 10.2**（`CUDART_VERSION 10020`） |
| 内核形态 | wheel 里是 `sm_80/90` 的 cubin + PTX，靠驱动 JIT | 目标 ISA 不是 PTX/SASS，是厂商自定义的 `bi`；**没有 `ptxas`，也没有 PTX JIT** |
| 运行时库 | `libcudart.so.12`、`libcublas.so.12`… | 只有 `.so.10.2` 一族 |
| 结论 | 装上去 `torch.cuda` 起不来 | 换 wheel 的版本号解决不了，必须换设备层 |

## 三层分工

```
  torch 2.8（fork 自编 wheel：上游 CPU 构建 + torch/corex + 设备名别名补丁）
        │  aten 算子 / 张量 / 自动微分 / torch.cuda.*
        ▼  第一层：corex 设备后端（PrivateUse1，DEVICE_NAME="cuda"）
  设备层：context、stream、event、显存分配（2 的幂缓存池）、H2D/D2H/D2D
        │
        ▼  第二层：算子下沉
  Triton 内核（corex-triton 3.4 后端：ttir→ttgir→llir→biir→cubin）
        │
        ▼  CoreX 运行时（libcudart.so.10.2 / libcuda.so.1 / iluvatar.ko）
  天垓100（BI-V100，32 GB 标称 / 31.72 GiB 可用 / 16 SM / warp 64 lane）
```

分层的理由是**验收口径清晰**：设备层只要能分配、能拷贝、能同步，就能用指针往返的字节比较单独验收；
算子层只认「与 CPU 参照值对齐」。详见 [`corex/docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 安装

两条路，按需要选：

**A. 交付态（推荐给使用者）**——装自编 wheel，装完就能用卡，不需要任何编译：

```bash
python -m venv ~/venvs/corex28
~/venvs/corex28/bin/pip install dist/torch-2.8.0a0+git<sha>-cp310-cp310-linux_x86_64.whl
~/venvs/corex28/bin/pip install triton==3.4.0        # 算子层需要
```

**B. 开发态（改设备后端时）**——装上游 CPU wheel，`torch/corex/*` 改完直接生效，
shim 用 `tools/build_shim.py` 单独编（登录节点即可，编译期不需要 CUDA）：

```bash
~/venvs/corex28/bin/python corex/tools/build_shim.py
export COREX_TRITON_SRC=/path/to/coffin-repo/corex-triton/src   # 它没打包安装
```

wheel 怎么编、设备名补丁清单、shim 怎么打进包、六个构建卡点：见
[`corex/docs/BUILD.md`](docs/BUILD.md)。**照那份做能一次过。**

## 一分钟跑起来（计算节点上）

```python
import torch                 # fork 的 torch/__init__.py 会 import torch.corex 并 register()（幂等）
# 需要显式时：import torch.corex; torch.corex.register()

x = torch.zeros(1024, device="cuda")      # 与 device="corex" 等价
y = x + 1.0
print(y.cpu())
print(torch.cuda.is_available(), torch.cuda.device_count())   # True 2（会话里有几张就几张）
```

验收脚本是 [`corex/probes/008_device_end_to_end.sh`](probes/008_device_end_to_end.sh)
（`DEVICE_VERDICT PASS`，10/10），用
`python corex/tools/remote.py sbx-run corex/probes/008_device_end_to_end.sh --name 008dev` 投到沙箱跑。

## 运行环境的三个前提

1. 卡只在计算节点上，碰卡的事必须通过沙箱队列跑：**唯一运行器**
   `corex_drive_re/sandbox/sandbox.sh`，队列根 `~/coffin-work/base/sandbox`，
   **单脚本上限 1800 s**。登录节点没有 `/usr/local/corex`，也没有 GPU 驱动。
2. **四条硬禁令**：禁 CUDA graph、禁 `cp.async`、禁自造 cmdbuf/ioctl、禁按 NVIDIA 经验估多流
   ——前两条踩过，用一次挂一张卡。
3. 厂商 torch 2.1（`torch-2.1.0+corex.3.2.1`，在共享 py310 里）**保持原样**，
   它是数值基线与算子行为的参照，不参与本次适配。

## 能力与限制

| 项目 | 状态 |
| --- | --- |
| 设备层（分配/拷贝/stream/事件） | 可用，`probes/008` 10/10、`probes/028` 6/6 |
| 算子全集 | **64/64**（`probes/161`）；推理最小集 39/43（`probes/120`，差额是「非末维 `dim=0`」的归约/索引） |
| bf16 | 可用（元素级 33/33 逐字节相等）；算术位置按算子分别定，见 `corex/docs/ARCHITECTURE.md` §5 |
| 矩阵乘 | fp16 / int8 走 Triton 后端；fp32 走 FMA |
| `torch.cuda.*` | `is_available/device_count/Stream/Event/synchronize` 等委托给 `torch.corex`；`torch.cuda.amp`、`torch.cuda.CUDAGraph` **没有**实现（后者本来就被禁） |
| decode 步时 | **5.1 ms/步**（fork wheel，`probes/222`，`VOCAB=32 DIM=64 HEADS=4 LAYERS=2`） |
| `torch.compile`/inductor | **不在范围**：inductor 假设有 graph capture，本平台禁用 |

## 目录

| 路径 | 作用 |
| --- | --- |
| `corex/docs/00-FACTS.md` | 事实表：每条都有命令与观察到的输出（节号连续不复用） |
| `corex/docs/ARCHITECTURE.md` | 接缝为什么是 PrivateUse1、三层分工、不做什么 |
| `corex/docs/INTERFACE.md` | 分配器/guard/device module/`DEVICE_NAME` 别名/aten 接管边界 |
| `corex/docs/BUILD.md` | fork wheel 构建、设备名补丁、shim 打包 |
| `corex/docs/05-DRIVER-CONSTRAINTS.md` | 驱动侧的硬禁用清单与 R1–R10 规矩 |
| `corex/probes/` | 编号探针脚本，全部可重放；结论回写 `00-FACTS.md` |
| `corex/tools/remote.py` | 远端通道：登录节点执行 + 沙箱队列投递/收结果 |
| `corex/tools/build_shim.py` | 编 C++ shim（`corex/src/csrc/corex_shim.cpp` → `torch/corex/_corex_shim.so`） |
| `corex/src/csrc/` | 分配器 + `DeviceGuardImplInterface` + Tensor factory + 快速启动路径（编译期零 CUDA 依赖） |
| `torch/corex/` | 设备后端：`_runtime.py`（ctypes 直连 CoreX）、`_triton.py`（接 corex-triton）、`_device.py`（注册 + aten 实现）、`_ops_*.py`、`_aten_*.py`、`_bf16.py` |

## 与其他目录的关系

- `corex-triton/`：Triton 3.4 → `bi` 后端，本仓库复用它「拿到 cubin 之后」的一切；
- `corex_drive_re/`：驱动侧逆向与约束，也是沙箱运行器的所在地；
- `coffin/`：推理引擎本体，是这份设备后端的第一批真实消费者；
- `platform/scow/`：作业提交与文件通道。
