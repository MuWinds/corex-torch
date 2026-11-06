# AGENTS.md —— corex-torch 的规矩

适用范围：本目录下所有 `.py`、`.sh`、`.c`、`.cpp`、`.h`、`.md`。
`corex-torch/` 是独立 git 仓库，所以本文件是它的顶层规则。

通用注释与提交规范继承仓库根 `AGENTS.md` 与 [`../coffin/AGENTS.md`](../coffin/AGENTS.md)
（一律中文、写「为什么」、「Google/Doxygen」两套 docstring、`<类型>: <中文一句话>` 提交格式）。
本文件只写**本仓库独有**的硬约束。

---

## 1. 事实优先：先探测，再写代码

这个仓库要把上游 PyTorch 2.8 接上一张没有任何公开文档的卡（天数智芯 天垓100，
CoreX 3.2.1 兼容层只声明 CUDA 10.2）。**任何「torch 里应该是这样」的写法都当作错误。**

- 新结论必须回写 `docs/00-FACTS.md`，格式是一条事实 + 产生它的命令 + 观察到的输出。
- 拿不到证据的写 `UNKNOWN`，并说明下一次探测要什么手段。不许用回忆补空。
- 探针一律进 `probes/NNN_名字.sh`，编号不复用，脚本自带「为什么做这个探测」的头注释。
  探测结论变了就新写一条探针，不要改旧的（旧的还要能重放出旧结论）。

## 2. 两条禁令（与 corex-triton 同源，这里同样适用）

1. **不要用 CUDA graph**（`cudaGraph*`、stream capture、`torch.cuda.CUDAGraph`）。
   驱动对 graph 的适配是坏的，用一次可能挂掉一张卡。
2. **不要用 `cp.async`**（`cp.async.ca/cg`、`__pipeline_memcpy_async`）。同上，能过编译但挂卡。

torch 2.8 自己的 C++ 代码里这两样到处都是（`c10/cuda`, `aten/src/ATen/cuda`）。本仓库
**不重编 torch 内核**，只在上游 CPU 版 wheel 之上新增设备后端，所以这两条约束落在
「我们自己写的那部分代码」上：任何看起来像异步拷贝的写法先停下来。

## 3. 沙箱与登录节点

- 卡只在计算节点上，所有碰卡的脚本都通过 `cx-sandbox` 队列跑：
  `python tools/remote.py sbx run probes/00X_x.sh --name 00X_x`。
- **登录节点没有 `/usr/local/corex`**（`ls -d /usr/local/corex*` 为空），也没有 GPU 驱动；
  共享 py310 环境在登录节点 `import torch` 会报 `libcufft.so.10` 缺失。纯 CPU 的
  pip/venv/读源码工作放登录节点，任何 `torch.cuda` / CoreX 头文件的事放沙箱。
- 队列文件**不能**含字面量 `#!/bin/bash`（网关会把它过滤成 0 字节并报成功）。
  脚本一律以 `set -u` 开头。
- 沙箱脚本的 CWD 是 `~/coffin-work/base/sandbox/work`，输出统一写
  `~/coffin-work/base/sandbox/logs/<name>.out`。
- 队列是**串行**的：一个脚本跑完才轮到下一个，单脚本上限 1800 s。别投长脚本占住队列，
  把一段探测拆成几个短脚本。

## 4. 端口与开关（不重编 torch 的前提）

本仓库的第一条纪律是 **不修改、不替换上游 torch 的二进制**：

- 设备通过 `PrivateUse1` 注册成 `torch.device("corex")`，torch 本体保持上游 CPU wheel 原样；
- 厂商 torch 2.1（`torch-2.1.0+corex.3.2.1`）**保留在共享 py310 里不动**，它是数值基线
  （算子行为、误差量级的参照），不是被替换对象；
- 一旦某一步必须改 torch 源码才能继续，先停下来把它写进 `docs/01-APPROACH.md` 的
  「升级为改源码」条目，说清楚改什么、为什么绕不过去，不要顺手改。

## 5. 验证口径

- 每个阶段只认「真机跑出数、和参照值对得上」。参照值优先用 CPU 上的 numpy/torch
  单线程算；厂商 torch 2.1 的算子只作为**第二参照**，不作为唯一基线。
- 数值比较只比同一份改动前后，不写 A/B 双实现对着跑两遍的测试。
- 性能数字必须带口径：卡号、迭代次数、是否首轮编译。
- 面向卡与算子的临时测试写在 `~/coffin-work/base/sandbox/work` 下，不进仓库；
  仓库只放「后端自身」的要素（设备注册协议、运行时绑定、算子到 Triton 的映射、构建脚本）。

## 6. 提交

格式与类型标记沿用根仓库：`feat/fix/perf/refactor/docs/test/chore/repo`，
首行中文、不超过 72 字符、不加句号；正文写清「为什么」「放弃了什么」「实测数字」。

范围标记用 `torch`，例如：

```
torch: torch 2.8 CPU 上立起 corex 设备

PrivateUse1 注册成功，torch.zeros(8, device="corex") 走自建 context 的
cudaMalloc，H2D/D2H 往返逐字节相等。
```
