# corex-torch —— 让 PyTorch 2.8 跑在天垓100（天数智芯 BI-V100）上

本仓库把**上游 PyTorch 2.8**（CPU 版 wheel，未经任何厂商修补）接到天数智芯 CoreX 3.2.1 上。
它**不重编 torch**，也不碰厂商 torch 2.1：新设备以 out-of-tree 的 `PrivateUse1` 形式注册成
`torch.device("corex")`，内存与流由我们自己通过 CoreX 运行时管理，算子在需要时下沉成
Triton 内核——那部分已经由姊妹仓库 [`corex-triton`](../corex-triton/README.md) 做好。

## 为什么不能直接装 CUDA 版 wheel

| 环节 | 上游 torch 2.8 | 这台机器实际 |
| --- | --- | --- |
| 需要的 CUDA | 11.8 / 12.6 / 12.8 | CoreX 兼容层只声明 **CUDA 10.2**（`CUDART_VERSION 10020`） |
| 内核形态 | wheel 里是 `sm_80/90` 的 cubin + PTX，靠驱动 JIT | 目标 ISA 不是 PTX/SASS，是厂商自定义的 `bi`；**没有 `ptxas`，也没有 PTX JIT** |
| 运行时库 | `libcudart.so.12`、`libcublas.so.12`… | 只有 `.so.10.2` 一族 |
| 结论 | 装上去 `torch.cuda` 起不来 | 换 wheel 的版本号解决不了，必须换设备层 |

所以适配的落点是：**让 torch 2.8 承认一个叫 `corex` 的设备**，把这张卡的能力通过
我们自己写的设备层暴露给 aten——算子先覆盖推理真正用到的那些，其余靠 CPU 兜底。

## 三层分工

```
  torch 2.8（上游 CPU wheel，原样不动）
        │  aten 算子 / 张量 / 自动微分
        ▼  本仓库第一层：corex 设备后端（PrivateUse1）
  设备层：context、stream、event、显存分配、H2D/D2H
        │
        ▼  本仓库第二层：算子下沉
  Triton 内核（复用 corex-triton 3.4 后端：bi IR → llc → lld → cubin）
        │
        ▼  CoreX 运行时（libcudart.so.10.2 / libcuda.so.1）
  天垓100（BI-V100，warp 64 lane）
```

分层的理由是**验收口径清晰**：设备层只要能分配、能拷贝、能同步，就能用指针往返的字节比较
单独验收；算子层只认「与 CPU 参照值对齐」；两层互不阻塞。

## 目录

| 路径 | 作用 |
| --- | --- |
| `docs/00-FACTS.md` | 已确证的环境事实，每条都带产生它的命令 |
| `docs/01-APPROACH.md` | 路线、阶段划分、每阶段的验收口径、不做什么 |
| `docs/02-RUNTIME-API.md` | CoreX 运行时可当「设备层」用的接口清单（ctypes 绑定面） |
| `docs/03-TORCH28-API.md` | torch 2.8 的 PrivateUse1 扩展点盘点（必需清单 + 证据） |
| `docs/04-TRITON-BRIDGE.md` | corex-triton 与外部 runtime 的解耦接缝 |
| `probes/` | 编号探针脚本，全部可重放；结论回写进 `00-FACTS.md` |
| `tools/remote.py` | 远端通道：登录节点执行 + 沙箱队列投递/收结果 |
| `src/` | 后端实现（随阶段推进） |

## 运行环境的三个前提

1. 卡只在计算节点上，通过 `cx-sandbox` 这个长期作业（SCOW 作业 4935，节点 `gpu-node-07`，
   健康卡 6、7）跑；`tools/remote.py` 就是它的客户端。**队列是串行**的。
2. **严禁** CUDA graph 与 `cp.async`：这两条是踩出来的，用一次挂一张卡
   （`../platform/scow/AGENTS.md` 第 1、5 条）。
3. 厂商 torch 2.1（`torch-2.1.0+corex.3.2.1`，装在共享 py310 环境）**保持原样**，
   它是数值基线与算子行为的参照，不参与本次适配。

## 与其他目录的关系

- `corex-triton/`：Triton 3.4 → `bi` 后端的既有成果，本仓库复用它「拿到 cubin 之后」的一切；
- `coffin/`：推理引擎本体，是这份设备后端的第一批真实消费者；
- `platform/scow/`：作业提交与文件通道。
