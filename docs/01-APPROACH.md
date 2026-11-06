# 路线与阶段

## 结论先说

对接点是 **PrivateUse1 设备后端**，不是重编 torch，也不是 CUDA 兼容层：

```
 torch 2.8（上游 CPU wheel）
      │
      │  torch.device("corex")  ← 本仓库注册的设备
      ▼
 ┌─────────────────────────────────────────┐
 │ 设备层（本仓库）：context / stream /     │
 │ event / 显存分配 / H2D-D2H  ← libcudart  │
 └─────────────────────────────────────────┘
      │  张量在卡上的内存
      ▼
 aten 算子 → 我们的实现 → Triton kernel（corex-triton 3.4 后端）
      │                       │
      │                       ▼  llc -march=bi → lld → cubin
      │                  ciLaunchKernel（ctypes）
      ▼
 天垓100
```

## 为什么是这个接缝（而不是别的三条路）

1. **换 wheel 版本号**：不行。CoreX 只声明 CUDA 10.2，torch 2.8 的 CUDA wheel 需要 `.so.12`
   一族，且内核是 PTX/SASS——这张卡的 ISA 是 `bi`，物理上没有 PTX 通路
   （`../corex-triton/docs/00-FACTS.md` §1–§3）。
2. **给 CoreX 做一个 CUDA 12 兼容层再装官方 wheel**：那是另一个工程量级（cuBLASLt、cuDNN 9、
   CUDA 12 的 mempool/launchEx 全都要补），而且 torch 2.8 的 `c10/cuda` 大量用 graph——
   在这张卡上是**禁用项**。方向反了：应该让上层绕开 CUDA 语义，不是让驱动假装成 CUDA 12。
3. **重编 torch 2.8 的 CUDA 内核到 `bi`**：`aten/src/ATen/native/cuda` 里有上千个 `.cu`，
   全都假设 PTX/`__shfl`/`cp.async`。规模与风险都不可控，且 `cp.async` 在本平台是挂卡项。
4. **PrivateUse1（选它）**：torch 官方给 out-of-tree 加速卡留的接口（torch_npu/torch_mlu
   走的就是这条）。它的好处是**逐层可验收**：设备层只要分配/拷贝/同步对，就能拿指针往返的
   字节相等单独验收；算子是另一个维度，可以一个一个加，没实现的先用 CPU 兜底。

## 阶段划分与验收口径

| 阶段 | 做什么 | 验收（只认这些） |
| --- | --- | --- |
| 0 | 摸清两端：CoreX 运行时能给什么、torch 2.8 要求什么 | **已完成**，见 `00-FACTS.md` §3、`03-TORCH28-API.md`、`04-TRITON-BRIDGE.md` |
| 1 | torch 2.8 CPU 在 `~/venvs/corex28` 可用 | **已完成**：`2.8.0+cpu`，`torch.cuda.is_available() == False`（`probes/100`） |
| 2 | **立起设备**：`rename_privateuse1_backend("corex")` + 分配器 + 设备属性 + 流/事件 | **已完成**：`probes/008` `DEVICE_VERDICT PASS`——`torch.zeros(4, device="corex")`、`.cpu()` 往返逐字节相等、`.item()`、非连续 strides、4 MiB 填读回、`empty_cache` 全过 |
| 3 | **元素级算子**（`add`/`sub`/`mul`/`div`/`fill`/`silu`/`copy`） | **已完成**：fp32 45/45、fp16 33/33 逐字节相等（`probes/170`、`171`）；add 295.5 GB/s = Triton 基线的 1.00x。**bf16 不支持**，根因是厂商 llc 选不出 `bf16 = fp_round(f32)`（`04-TRITON-BRIDGE.md` §6） |
| 4 | **推理最小算子集**：规约（`sum`/`mean`）、`softmax`、`mm`/`bmm`、`rms_norm`、`embedding`、`view`/`transpose` | 进行中。每个算子与 CPU 参照的误差量级写进表；`tl.dot` 走 FMA（矩阵单元未接，约 1/15 算力） |
| 5 | **集成**：coffin 的一条最小前向在 torch 2.8 + corex 上跑通 | 逐 token 与厂商 torch 2.1 的 logits 对齐（误差量级单独记） |

阶段 2 之前不写任何算子代码：**设备层不通，算子层的一切数字都不可信**。

## 算子该走什么机制（决定性能天花板）

厂商 torch 2.1 的算子成绩是**四种不同机制**混在一起的：DMA 引擎、ATen 手写逐元素 kernel、
厂商 TCU GEMM、ATen warp 级规约 kernel（`00-FACTS.md` §9）。所以「我们比厂商 torch 慢多少」
这个比值没有单独的解释力，必须按机制分别对照：

| 类别 | 不接矩阵单元时的现实目标 | 走什么 |
| --- | --- | --- |
| 拷贝（`copy_`、`_to_copy`、`.contiguous()`、H2D/D2H） | 与 DMA 持平（471 GB/s） | **`cudaMemcpyAsync`**，绝不写 Triton 内核 |
| 逐元素（`add`/`mul`/`silu`/…） | ATen 402 GB/s 是天花板，Triton 当前 295 GB/s | Triton 内核（先做对，再谈向量化宽度） |
| 规约/softmax 类 | 与 Triton 自身历史数据比，不与 ATen 的 warp kernel 混比 | Triton 内核 |
| 矩阵乘（`mm`/`bmm`/`addmm`） | FMA 展开约 105 GFLOP/s，是 TCU 的 1/15 | Triton 内核；**追平要等 TCU 接进 `TritonGPU→LLVM`** |
| 其余未实现算子 | 不假装支持 | 显式 CPU 兜底并记账，不许静默算错 |

**这条约束的实用含义**：coffin 的 decode 路径是带宽型（每步搬 KV 与权重），
拷贝与逐元素占大头，所以「矩阵乘慢 15 倍」不至于毁掉 decode；但 prefill 是算力型，
不接 TCU 就是 15 倍差距。阶段 4/5 的优先级按这个来排，不要反过来先优矩阵乘。

## 已知难点（按会被咬到的顺序）

1. **分配器只能从 C++ 注册**（待 `03-TORCH28-API.md` 证实）。若成立，就要一个很小的
   C++ 扩展：`at::SetAllocator` + `DeviceGuardImplInterface` + 一个 aten 分发键。
   它只做「转发到 ctypes 打开的那几个 C 函数」，不碰任何 CUDA 语义。
2. **context 谁建**。厂商 torch 2.1 替我们建 primary context 的日子结束了；现在要自己
   `cuInit` + `cuCtxCreate`，而 `corex-triton/docs/00-FACTS.md` §5 记录过裸 `cuCtxCreate`
   在坏卡上挂死过。沙箱里只能逐次实测，且探针必须带硬超时。
3. **没实现的算子怎么办**。PrivateUse1 不会自动 fallback 到 CPU，必须有意识地注册
   `_to_copy` 与一批 `cpu_fallback`，否则报错信息会指向 aten 内部、看不出是「没实现」。
4. **`torch.compile` 不在范围内**。inductor 的默认路径就有 CUDA graph 级别的假设，
   本平台禁用 graph；把 inductor 接进来等于给一条禁用的路修桥。这条写进「不做什么」。

## 不做什么

- **不重编、不 patch torch 2.8 的二进制**。设备后端全部走 out-of-tree 注册。
- **不动共享 py310 里的厂商 torch 2.1**：它是数值基线。
- **不实现全部 aten**。算子按 coffin 推理路径的实际需要增量加；其余走 CPU 兜底并记账。
- **不做 CUDA graph、不用 cp.async**（平台禁令，见 `AGENTS.md` §2）。
- **不为了「跑通」放宽数值口径**：每个算子都要与 CPU 参照对齐，误差量级要写出来。

## 升级为「必须改 torch 源码」的条目

（空。真到了那一步再往这里加：改什么、为什么绕不过去、有没有更小的替代。）
