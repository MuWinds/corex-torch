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
| 3 | **元素级算子**（`add`/`sub`/`mul`/`div`/`fill`/`silu`/`copy`） | **已完成**：fp32 45/45、fp16 33/33、**bf16 33/33** 逐字节相等（`probes/170`、`171`、`192`）；add 295.5 GB/s = Triton 基线的 1.00x |
| 4 | **推理最小算子集**：规约、`softmax`/`log_softmax`、`mm`/`bmm`、`rms_norm`/`layernorm`、`index_select`、激活、`max`/`argmax`、视图族 | **已完成**：内核层 `probes/180`–`197`、torch 层 `009`/`017`/`021` 全绿；`probes/120` 覆盖 39/43，剩 4 项全是「非末维 `dim=0`」的归约/索引，按需再补 |
| 5 | **集成**：coffin 的一条最小前向在 torch 2.8 + corex 上跑通 | **已验证到「逐步生成」这一层**：`probes/018` 两层 decoder 整段前向（全局相对 6.63e-06、argmax 1.000）；`probes/022` 增量 decode + KV cache（逐 token 最差 6.62e-06、argmax 6/6、cache 两次运行逐字节相同）。**尚未接 coffin 本体**（要真权重与 tokenizer） |
| 6 | **性能** | **进行中**：decode `17.1 → 9.2 ms/步`（`probes/124` 的搬运路径重构）。当前天花板是「算子层是 Python」：C++ 快路径的 `mm` 只要 **3.1 µs**，而 Python 实现的算子 **67–127 µs**。下一步是把热路径算子也挪进 C++（照 `mm` 的 `TORCH_LIBRARY_IMPL` + `set_*_kernel` 模式） |

阶段 2 之前不写任何算子代码：**设备层不通，算子层的一切数字都不可信**。

## 当前缺口（截至 2026-11-07 02:10，逐条可复现）

| 缺口 | 归属 | 说明 |
| --- | --- | --- |
| `var` / `std`（`var.correction`） | 需要新内核 `row_var` | 两趟（mean→平方差的均值）在 fp32 下会掉精度；等一趟 Welford 内核 |
| `sum`/`mean` 的**中间维**与多维组合 | `_ops_reduce` 扩参数 | 现在只支持「全归约」与「最后一维」；中间维需要跨步读，没有 strided 内核 |
| `_softmax` 的**非最后维** | 同上 | 同上 |
| `max.dim`/`argmax` 的**非最后维**、`keepdim=True` | `_aten_kernels` 补形状 | 内核已支持任意维的展平版本，缺的是形状换算 |
| `masked_fill` 的**广播 mask** | `_ops_elementwise` 扩参数 | 现在要求 mask 与 self 同形 |
| `embedding`（`index_select` 之外的 gather 语义） | `_aten_extra` | `index_select` 已通，`embedding` 是 composite，未逐项验 |
| `matmul`（三维以上批量） | `_aten_kernels` | `mm`/`bmm` 已通，`matmul` 需要 reshape + 逐批循环 |
| 矩阵单元（TCU） | `corex-triton` 阶段 5 | `tl.dot` 走 FMA，算力约为 TCU 的 1/15 |
| `_softmax.out` 的 C++ 接管 | 可选性能项 | 受 aten 分发器 73 µs 的成本上限约束，收益有限 |

## 下一步优先级

1. `var`/`std` 的 `row_var` 内核（模型做归一化时常用），以及 `sum`/`mean` 的中间维支持；
2. 接 coffin 的一条最小前向（阶段 5）：先用 `rms_norm + mm + softmax` 拼一个 block，逐 token 与
   厂商 torch 2.1 的 logits 对齐；这一步会立刻暴露「缺哪个算子」，比按清单猜准；
3. 若那时还剩时间，再考虑 `_softmax.out` 的 C++ 接管与「多 kernel 连续发射」入口。

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

## bf16 通路（2026-11-07 决策：走内核侧 bf16，不做「解包成 fp16」）

厂商 llc 缺的是**两个转换指令**（bf16→f32 的 `fp_extend`、f32→bf16 的 `fp_round`），
不是 bf16 算术本身。所以「全程 bf16」的正确实现是**在 IR 里彻底不出现这两个转换**：
把缓冲区按 `u16` 重新解释，用整数位运算自己造转换（展宽＝左移 16 位后 `bitcast`；
收窄＝`bitcast` 成 uint32 后右移 16 位，需要 RNE 时加 `0x7FFF` 与最低位）。

由此**算术位置按算子分别定，不一刀切**：

| 算子 | 计算精度 | 理由 |
| --- | --- | --- |
| `add`/`sub`/`mul`/`div`、`fill`、`copy`、`gather` | bf16 直接算 | 逐元素，bf16 算术本身就是 IEEE round-to-nearest |
| `silu` 等激活 | fp32 算完转回 bf16 | `ex2.approx.f32` 没有 bf16 版本，算不了纯 bf16 的 exp |
| `sum`/`mean`/`max` 规约 | **fp32 累加**，输入位运算展宽 | 纯 bf16 累加 1000 项相对误差到 1e-2 量级，模型不可用 |
| `softmax` | fp32 算 | 同上，且行和偏离 1 要保持在 1e-7 量级 |
| `mm`/`bmm` | **fp32 累加、bf16 输入输出** | 工业界标准（TCU/GPU 同款）；纯 bf16 累加在 K=1024 时相对误差约 1e-1 |

**退路**（需要证据才切）：bf16 只在搬运与逐元素上支持，矩阵乘与规约继续显式报错。
判据是内核侧拿不到 `bitcast` 这条位运算通道——那时纯 bf16 硬算等于把数值不可用的内核
伪装成「支持 bf16」，不做。

## 目标达成的证据索引（逐条对应）

| 目标要素 | 证据 |
| --- | --- |
| 独立仓库 `corex-torch` | 本仓库，`git log` 全部提交 |
| 以 `corex-triton`（Triton 3.4）为基座 | `src/corex_torch/_triton.py` 只做适配不改上游；编译链 `ttir→ttgir→llir→biir→cubin` 全用它的后端 |
| 设备分配 | `probes/008`（`DEVICE_VERDICT PASS` 10/10），分配器在 `src/csrc/corex_shim.cpp` |
| 拷贝（H2D/D2H/D2D，含非连续） | `probes/008`、`020`（四方向逐字节）、`026`（p50 1 KiB~1 MiB 与大小无关） |
| stream / 事件 | `probes/028`（6/6 PASS），`torch.corex.{Stream,Event,current_stream,stream}` |
| 关键算子走 Triton 内核 | `probes/180`–`197`（内核层）、`009`(9/9)、`017`(12/12)、`021`(13/13)、`120`(39/43)、`192`(bf16 33/33+15/15+5/5) |
| 真机验证（作业 4935 / gpu-node-07 / 卡 6、7） | 所有探针日志头部 `HOST=gpu-node-07`、`INHERITED_CVD=6,7`、收尾 `CARD_STILL_HEALTHY` |
| 禁止 CUDA graph 与 cp.async | `grep -r "cudaGraph\|cp\.async" src/ tools/` 只命中注释里的禁令说明，无任何调用 |
| 端到端（超出最小要求） | `probes/018` 两层 decoder 整段前向（6.631e-06、argmax 1.000）；`probes/022` 增量 decode + KV cache（每 token 6.62e-06、argmax 6/6、cache 逐字节稳定，4.9 ms/步） |
