# 架构：接缝为什么是 PrivateUse1，三层怎么分工，以及不做什么

这份文档回答：**为什么这台卡的适配落在「torch 设备后端」这一层、三层各自负责什么、哪些事明确不做。**

接口契约在 [`INTERFACE.md`](INTERFACE.md)，构建在 [`BUILD.md`](BUILD.md)，
驱动侧约束在 [`05-DRIVER-CONSTRAINTS.md`](05-DRIVER-CONSTRAINTS.md)，事实在 [`00-FACTS.md`](00-FACTS.md)。

---

## 1. 现状一句话

**fork 的 PyTorch 2.8 树 + 自编 wheel + 9 个文件约 150 行补丁**，把 `corex` 注册成一个
PrivateUse1 设备；设备名别名成 `cuda`，所以客户代码里 `model.cuda()`、`device="cuda"` 照旧可用。
算子按推理路径增量下沉到 Triton 内核（由 [`corex-triton`](../../../corex-triton/corex/README.md) 编译），
未实现的显式 CPU 兜底并记账。

> 历史形态变过一次：早期是「在上游 CPU wheel 之上挂 out-of-tree 插件、不碰 torch 二进制」，
> 但那条路在「客户代码要 `torch.cuda`」和「算子要接管 aten」两处都撞墙（[`00-FACTS.md`](00-FACTS.md) §20），
> 于是改成 fork 树 + 自编 wheel。早期那套「在已装 wheel 上挂插件、不碰 torch 本体」的说法随之作废，不要再引用。

## 2. 为什么是这个接缝（而不是另外三条路）

1. **换 wheel 版本号**：不行。CoreX 只声明 CUDA 10.2，torch 2.8 的 CUDA wheel 需要 `.so.12` 一族，
   且内核是 PTX/SASS——这张卡的 ISA 是 `bi`，物理上没有 PTX 通路。
2. **给 CoreX 做一个 CUDA 12 兼容层再装官方 wheel**：那是另一个工程量级（cuBLASLt、cuDNN 9、
   CUDA 12 的 mempool/launchEx 全都要补），而且 `c10/cuda` 大量用 graph——在这张卡上是禁用项。
   方向反了：应该让上层绕开 CUDA 语义，不是让驱动假装成 CUDA 12。
3. **重编 torch 的 CUDA 内核到 `bi`**：`aten/src/ATen/native/cuda` 里上千个 `.cu` 全都假设
   PTX/`__shfl`/`cp.async`，规模与风险都不可控，且 `cp.async` 在本平台是挂卡项。
4. **PrivateUse1（选它）**：torch 官方给 out-of-tree 加速卡留的接口（torch_npu/torch_mlu 走的就是这条）。
   好处是**逐层可验收**：设备层只要分配/拷贝/同步对，就能用指针往返的字节相等单独验收；
   算子是另一个维度，可以一个一个加，没实现的先 CPU 兜底。

## 3. 三层分工

```
  torch 2.8（fork 自编 wheel：上游 CPU 构建 + torch/corex + 设备名别名补丁）
        │  aten 算子 / 张量 / 自动微分 / torch.cuda.*
        ▼  第一层：corex 设备后端（PrivateUse1，DEVICE_NAME="cuda"）
  设备层：context、stream、event、显存分配（缓存池）、H2D/D2H/D2D
        │
        ▼  第二层：算子下沉
  Triton 内核（corex-triton 3.4 后端：ttir→ttgir→llir→biir→cubin）
        │
        ▼  CoreX 运行时（libcudart.so.10.2 / libcuda.so.1 / iluvatar.ko）
  天垓100（BI-V100，32 GB 标称 / 31.72 GiB 可用 / 16 SM / warp 64 lane）
```

分层的理由是**验收口径清晰、互不阻塞**：

| 层 | 验收（只认这些） |
| --- | --- |
| 设备层 | `probes/008`：`torch.zeros(4, device="corex")`、`.cpu()` 往返逐字节相等、非连续 strides、4 MiB 填读回、`empty_cache`（10/10） |
| 算子层 | 与 CPU 参照逐元素比（能量化就「逐字节相等」，不能就写误差量级与门槛），见 `probes/009`/`017`/`021`/`161` |
| 集成层 | mini decoder 整段前向（相对误差 6.6e-06、argmax 1.000）与增量 decode（每 token 对齐） |

## 4. 已知难点（按会被咬到的顺序）

1. **分配器只能从 C++ 注册**，而且忘记注册时拿到的是 **nullptr、不报错**
   （[`00-FACTS.md`](00-FACTS.md) §9）。所以必须有一个很小的 C++ shim：`at::SetAllocator` +
   `DeviceGuardImplInterface` + 一个 aten 分发键，它只转发到 ctypes 打开的几个 C 函数。
2. **context 谁建**。不能指望厂商 torch 2.1 替我们建 primary context；自己走
   `cuInit` → `cuDeviceGet` → `cuDevicePrimaryCtxRetain`（与 torch 同一个主 context）。
   裸 `cuCtxCreate` 当年「挂死」的真因是坏卡，不是 API 缺实现（[`00-FACTS.md`](00-FACTS.md) §6）。
3. **没实现的算子不会自动 fallback**。PrivateUse1 不会替你退回 CPU，必须有意识地注册
   `_to_copy` 与一批 `cpu_fallback`，否则报错信息会指向 aten 内部、看不出是「没实现」。
4. **`.out` 变体与 Python/C++ 注册的先后**。Python 后注册会盖住 C++（`torch.library` 的
   `Overriding … previous kernel` 警告就是它），所以快路径让位要靠
   `elementwise_op_names()` 这类「C++ 已接管」清单显式跳过。
5. **`torch.compile`/inductor 不在范围内**：inductor 的第一条优化假设就是有 graph 级 capture 能力，
   本平台禁用 graph；接进来等于给一条禁用的路修桥。

## 5. bf16 的算术位置（设计决定，不是实现细节）

厂商 llc 缺的是**两个转换指令**（`fp_extend`、`fp_round`），不是 bf16 算术本身。所以「支持 bf16」的
正确做法是**在 IR 里彻底不出现这两个转换**（换成 `llvm.bi.bfloat2float` / `llvm.bi.float2bfloat`，
向量形态标量化，见 [`00-FACTS.md`](00-FACTS.md) §15）。由此算术位置**按算子分别定**：

| 算子 | 计算精度 | 理由 |
| --- | --- | --- |
| `add`/`sub`/`mul`/`div`、`fill`、`copy`、`gather` | bf16 直接算 | 逐元素，bf16 算术本身就是 round-to-nearest |
| `silu` 等激活 | fp32 算完转回 bf16 | `ex2.approx.f32` 没有 bf16 版本 |
| `sum`/`mean`/`max` 规约 | **fp32 累加**，输入位运算展宽 | 纯 bf16 累加 1000 项相对误差到 1e-2 量级，模型不可用 |
| `softmax` | fp32 算 | 同上，行和偏离 1 要保持在 1e-7 量级 |
| `mm`/`bmm` | **fp32 累加、bf16 输入输出** | 工业界标准；纯 bf16 累加在 K=1024 时相对误差约 1e-1 |

## 6. 不做什么

- **不改上游的 CUDA 算子内核**：`aten/src/ATen/native/cuda` 一行不动（构建是 `USE_CUDA=0`，那些内核根本不参与）；我们只加设备后端，
  算子以 out-of-tree 的方式注册。
- **不动共享 py310 里的厂商 torch 2.1**：它是数值基线（算子行为、误差量级的参照），不是被替换对象。
- **不实现全部 aten**：算子按推理路径的实际需要增量加，其余显式 CPU 兜底并记账，不许静默算错。
- **不做 CUDA graph、不用 `cp.async`、不自造 cmdbuf、不按 NVIDIA 经验估多流**（平台四条硬禁令，
  见 [`../AGENTS.md`](../AGENTS.md) §2）。
- **不为了「跑通」放宽数值口径**：每个算子都要与 CPU 参照对齐，误差量级要写出来。
- **不做「多卡 TP 加速」这类跨卡协议**：本仓库只提供单进程内的设备后端；集合通信走厂商 `libnccl`，
  那是集成阶段的事。

## 7. 目标达成的证据索引

| 目标要素 | 证据 |
| --- | --- |
| 独立仓库 `corex-torch` | 本仓库，`git log` 全部提交 |
| 以 `corex-triton`（Triton 3.4）为基座 | `torch/corex/_triton.py` 只做适配不改上游；编译链全用它的后端 |
| 设备分配 | `probes/008`（`DEVICE_VERDICT PASS` 10/10），分配器在 `corex/src/csrc/corex_shim.cpp` |
| 拷贝（H2D/D2H/D2D，含非连续） | `probes/008`、`020`（四方向逐字节）、`026`（p50 与大小无关） |
| stream / 事件 | `probes/028`（6/6 PASS），`torch.corex.{Stream,Event,current_stream,stream}` |
| 关键算子走 Triton 内核 | `probes/170`/`171`/`192`（内核层）、`009`(9/9)、`017`(12/12)、`021`(13/13)、`120`(39/43)、`161`(64/64) |
| 禁止 CUDA graph 与 `cp.async` | `grep -r "cudaGraph\|cp\.async" torch/corex corex/src` 只命中注释里的禁令说明 |
| 端到端 | `probes/018`（整段前向 6.631e-06、argmax 1.000）、`022`（增量 decode + KV cache，逐 token 对齐、cache 逐字节稳定） |
| `torch.cuda` 兼容 | `probes/204`/`218`/`224`（`CUDA_AVAILABLE True`、`IS_CUDA True`、算子矩阵 64/64） |
