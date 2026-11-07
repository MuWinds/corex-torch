# AGENTS.md —— corex-torch 的规矩

这份文档回答：**在这个仓库（fork 的 PyTorch 2.8 + corex 设备后端）里动手之前，必须先知道的硬约束与验收口径是什么。**

适用范围：本目录下所有 `.py`、`.sh`、`.c`、`.cpp`、`.h`、`.md`。`corex-torch/` 是独立 git 仓库，
所以本文件是它的顶层规则。

形态：仓库根是 **PyTorch 2.8 的 fork 树**。我们的东西分三处：

| 路径 | 内容 |
| --- | --- |
| `corex/`（本目录） | 文档、`probes/`、`tools/`、`src/csrc/`（shim 源码）。**不放上游算子代码** |
| `torch/corex/` | 设备后端本体：`_device.py`（注册 + aten 实现）、`_runtime.py`、`_triton.py`、`_ops_*.py`、`_aten_*.py`、`_bf16.py` |
| `torch/cuda/__init__.py` 等 9 个文件 | 设备名别名补丁（`"cuda"` → PrivateUse1），见 `corex/docs/00-FACTS.md` §20 |

通用注释与提交规范继承仓库根 `AGENTS.md`（一律中文、写「为什么」、Google/Doxygen 两套 docstring、
`<类型>: <中文一句话>` 提交格式）。本文件只写**本仓库独有**的硬约束。

---

## 1. 事实优先：先探测，再写代码

这个仓库要把 torch 2.8 接上一张没有任何公开文档的卡（天数智芯 天垓100 / BI-V100，
CoreX 3.2.1 兼容层只声明 CUDA 10.2）。**任何「torch 里应该是这样」的写法都当作错误。**

- 新结论必须回写 [`docs/00-FACTS.md`](docs/00-FACTS.md)：一条事实 + 产生它的命令 + 观察到的输出。
  节号连续、不复用（改结论就改那一节，不要新增同号节）。
- 拿不到证据的写 `UNKNOWN`（[`docs/00-FACTS.md`](docs/00-FACTS.md) §23），并说明下一次探测要什么手段。
  不许用回忆补空。
- 探针一律进 `probes/NNN_名字.sh`，脚本自带「为什么做这个探测」的头注释。**结论变了就新写一条，
  不要改旧的**——旧的还要能重放出旧结论。编号取当前最大值 +1：

  ```sh
  ls probes/*.sh | grep -o 'probes/[0-9]*' | sort -n | tail -1
  ```

## 2. 四条硬禁令

1. **禁用 CUDA graph**（`cudaGraph*`、stream capture、`torch.cuda.CUDAGraph`）。驱动对 graph 的适配
   是坏的，用一次可能挂掉一张卡。
2. **禁用 `cp.async`**（`cp.async.ca/cg`、`__pipeline_memcpy_async`）。能过编译，但会挂卡。
3. **禁用自造 cmdbuf / 自造 ioctl**：只走官方 UMD 入口。自拼命令缓冲已经跑死过一张卡。
4. **禁用「按 NVIDIA 经验估多流」**：本平台的队列与 fence 语义与 CUDA 不同，
   任何「靠 stream 并行提速」的设想必须先测再说。

前两条的后果是**静默挂卡**：过编译、跑起来没输出、退出码拿不到。它们落在「我们自己写的那部分代码」上
（上游的算子内核一行不动），所以任何看起来像异步拷贝的写法先停下来。

## 3. 沙箱

- 卡只在计算节点上，所有碰卡的脚本走沙箱队列。**唯一运行器**是
  `corex_drive_re/sandbox/sandbox.sh`，队列根 `~/coffin-work/base/sandbox`
  （`queue/ logs/ done/ work/`），**单脚本上限 1800 s**（runner 默认 900，作业传入 1800）。
- 投递：`python tools/remote.py sbx-run probes/00X_x.sh --name 00X_x`（投递并等待），
  或 `sbx-push` / `sbx-wait` / `sbx-log` / `sbx-reset` 分步做。
- **登录节点没有 `/usr/local/corex`**，也没有 GPU 驱动；共享 py310 环境在登录节点 `import torch`
  会报 `libcufft.so.10` 缺失。纯 CPU 的 pip/venv/读源码/编 shim 放登录节点，
  任何 `torch.cuda`、CoreX 头文件、碰卡的事放沙箱。
- 队列文件**不能**含字面量 `#!/bin/bash`（网关会把整个文件过滤成 0 字节并返回成功）。
  脚本一律以 `set -u` 开头。
- 沙箱脚本的 CWD 是 `~/coffin-work/base/sandbox/work`，输出统一写
  `~/coffin-work/base/sandbox/logs/<name>.out`。
- 队列是**串行**的：一个脚本跑完才轮到下一个。别投长脚本占住队列，把一段探测拆成几个短脚本。
- 三条会**静默出错**的操作细节（详细现象见 [`docs/00-FACTS.md`](docs/00-FACTS.md) §22）：
  不要覆盖 runner 已设好的 `CUDA_VISIBLE_DEVICES`（逻辑卡号 ≠ 物理卡号）；
  重跑要换脚本名（`done/<name>.running.sh` 会挡住同名脚本）；
  探针要 `exit` 出被测命令的退出码（最后一句是 `echo` 时 rc 恒为 0）。

## 4. 硬件与作业的硬限制

- **单作业时限上限 4320 分钟（3 天）**：QoS 的 `MAXTIMELIMITPERTASK` 三档都是 `3-00:00:00`，
  超过不排队、直接 500。
- **8 卡作业必须显式写 `coreCount`**：门户按 14 核·62.5 GB/卡换算，8 卡默认 112 核，
  超过单节点的 110 核；`96` 是能站住的值（单节点 110 核 / 480 GB / 8 卡）。
- **卡的好坏是会话级事实**：同一机箱每次坏的那张可能不同，一律运行期逐卡测
  （8×8 bf16 matmul + 5 s 硬超时）。不要写「某节点卡全坏」这类判断。

## 5. 验证口径

- 每个阶段只认「真机跑出数、和参照值对得上」。参照值优先用 CPU 上的 numpy/torch 单线程算；
  厂商 torch 2.1 的算子只作**第二参照**，不作唯一基线。
- **正确性先于性能**，且判据要能抓静默算错：非方阵/非整除形状 + 与 CPU 逐元素比对；
  能量化就「逐字节相等」，不能就写误差量级与门槛。性能数字必须带口径
  （卡型号、卡号、dtype、迭代次数、是否首轮编译、单流还是批），口径不同的两个数不能相除。
- 统计一律 **p50 + 迭代 ≥500 + 两条基线**（空循环、最小真内核）：这张卡约 4% 迭代会被
  驱动抢占停 ≥100 µs，均值会把这个噪声算进结果。
- 数值比较只比同一份改动前后，不写 A/B 双实现对着跑两遍的测试。
- 面向卡与算子的临时测试写在 `~/coffin-work/base/sandbox/work` 下，不进仓库；
  仓库只放「后端自身」的要素（设备注册协议、运行时绑定、算子到 Triton 的映射、构建脚本）。

## 6. 提交

格式与类型标记沿用仓库根：`feat/fix/perf/refactor/docs/test/chore/repo`，
首行中文、不超过 72 字符、不加句号；正文写清「为什么」「放弃了什么」「实测数字」。

范围标记用 `torch`，例如：

```
torch: 设备名别名到 cuda，客户代码不用改

9 个文件约 150 行：c10 的字符串表与 is_cuda 认 PrivateUse1，torch/cuda 委托给
torch.corex。自编 wheel 上 CUDA_AVAILABLE True、算子矩阵 64/64。
```
