# 驱动侧已经确证的东西，以及它对 torch 设备后端意味着什么

这份文档回答：**这台卡的「CUDA 兼容层」有哪些是真的、哪些是壳，由此推出设备层必须遵守哪些规则（R1–R10）。**

本文件不重复驱动逆向的过程，只把**已经定案、且会约束本仓库代码**的事实摘出来，并逐条推出规则。
原始证据在 [`../../../docs/corex_drive/GRAPH-CHANNEL.md`](../../../docs/corex_drive/GRAPH-CHANNEL.md)
（ioctl 协议、CP 包格式、一次提交 N 个 launch）与
[`../../../docs/corex_drive/README.md`](../../../docs/corex_drive/README.md)（驱动侧总览）。
下文引证一律指向文件名而不是节号——那几份文档的节号会随重写变动。

**为什么先看驱动再写代码**：这张卡的「CUDA 兼容层」不是 NVIDIA 那种实现——提交协议、
队列与 fence 的粒度、地址空间语义都和 CUDA 惯例有出入。照着 CUDA 的直觉写设备层，
会在「多流」「事件」「批提交」三个地方产生看起来正常、实际丢同步的代码。

## 0. 先读这份：驱动的「不能用」清单

判据是「内核侧有没有下发通道」，不是「头文件里有没有」——用户态导出了完整符号但 KMD 没通道的
API，用下去是走进未定义路径，而驱动 `recover=0`（没有恢复能力），所以结果就是整节点废掉。
CoreX 3.2.1 的静态分析（`libixthunk.so` 的 40 个 `Thunk_*` 入口 + `iluvatar.ko` 字符串表 +
反汇编）逐条点出了「有壳无内核」的东西。

对 corex-torch 直接生效的硬禁用（A 节）：

| 禁用项 | 依据 | 我们的处置 |
| --- | --- | --- |
| CUDA graph / stream capture（`cudaStreamBeginCapture`、`cudaGraph*`、`cudaGraphExecUpdate`） | `libixthunk` 40 入口里 `Graph=0`，`.ko` 里 `graph=0 capture=0` | 不提供、不暴露、不在文档里暗示可 capture |
| `cudaGraphLaunch_ptsz` | 56 条 801 桩之一，**静默返回 801** | 连带规则：**不用 `--default-stream per-thread`** 编译，否则拿到的就是桩 |
| `cp.async`（任何形式） | `.ko` 里 `cp.async=0`，实测挂卡 | corex-triton 侧已避开；我们生成内核时不吐 |
| cooperative launch | 无 thunk 入口 | 不调 |
| cluster / `cudaLaunchKernelEx` | **连声明都没有** | 启动一律用 `cuLaunchKernel` |
| **memory pool / `cudaMallocAsync`** | **连声明都没有** | **必须自己实现分配器**（缓存池），见 R7 |
| 动态并行 `cudaLaunchDevice` | 头文件声明、库不导出 | 不调 |
| texture / surface / array / graphics interop | 显式 `mov $0x321`（801） | 不调；好在会明确报错不挂卡 |
| `cudaOccupancyMaxPotentialBlockSize`（及 3 个变体）、`cudaMemAttachHost` | 头文件有、**库不导出** → 链接期就崩 | 不调；写 C++ 扩展时如遇 undefined reference 先查这份名单 |

另外三条审计结论会影响设备层的实现方式：

- **`cudaMemcpyAsync` 是真的**，且设备到设备走 `mem_handle`、不每次 pin；
- **host 路径的 `cudaMemcpy` 每次都 `vmalloc` + `get_user_pages_fast` 重新 pin 页**
  （`itr_dma.c`，没有缓存容器）——那是毫秒级操作。所以小张量的 H2D/D2H 必须走固定的
  pinned 暂存区，不能每次拷贝都现 pin；
- 页大小编译期固定 **2 MiB**（`PG_MODE=2M`），分配粒度按这个算。

## 1. 驱动链路已经确证的事实

| 事实 | 对设备层的影响 |
| --- | --- |
| 用户态是两层：`libcudart`/`libcuda` → `libixthunk.so`（**纯转发**）→ `/dev/iluvatarN` 的 ioctl | 我们调 `cuda*` 等于调 ioctl，没有第三层优化空间 |
| 提交语义是「**一次 ioctl 带一整块命令缓冲**」：`AIP_QUEUE_SUBMIT{hEngine, cmd句柄, cmdsize, patch列表}` | UMD 自己就会把多个 launch 打进一条 cmdbuf |
| **生产路径已经在批提交**：单卡每步 decode 的 kernel 数在数百量级、ioctl 在数十量级，单条 cmdbuf 最多 46 个 `NDR_KICKOFF`；包链是固定四元组 `POLL_MEM_REG + 0x1a + NDR_KICKOFF + EVENT_WRITE` | 发射侧不需要我们再优化；但**也不能假设「一个 kernel 一次 ioctl」** |
| 一条 cmdbuf 里的多个 `NDR_KICKOFF` **每个后面都要跟自己的 `EVENT_WRITE`**，只发一次会让后续 launch 被丢掉 | 串行性由这些包保证；自己拼 cmdbuf 极危险 |
| **KMD 没有图通道**：CUDA Graph API 会挂卡 | 本仓库**不提供**任何 graph 能力 |
| 队列（engine）是 KMD 内部对象，`hEngine = (context_id << 32) \| queue_id`；`AIP_QUEUE_CREATE` 不读用户给的 ring 描述 | 流=队列，数量与所有权都在内核侧；不要假设能自造流 |
| 每个队列只有 **4 个 user fence 槽**（`usr_fences[4]`） | 同时挂起的事件数必须克制 |
| 进程里**没有任何 `/dev/iluvatar*` 的 mmap**：提交靠 ioctl + 设备内存里的 cmdbuf | 别去找「用户态 ring + 门铃」这条路 |
| `UNIFIED_ADDRESSING = 0`、`CONCURRENT_KERNELS = 1`（`probes/150_noctx_runtime.sh`） | 主机指针不能当设备指针；不要指望多流真并发 |
| `cuCtxGetApiVersion` 报 `10000`（与 `CUDART_VERSION 10020` 不一致） | 不要用 api version 做分支判断 |
| 自造 ioctl/自造 cmdbuf 已把一张卡跑死过一次（`pkt_bd_len` 少算 4 字节） | **只走官方入口** |

## 2. 由此推出的规则（写代码前先读这一节）

**R1 只调官方 UMD 入口。** 设备层用 `cuda*`（runtime）与 `cu*`（driver）；不碰 `/dev/iluvatar*`、
不拼 cmdbuf、不复制任何手工构造器。那一层已经把一张卡跑死过，收益（发射次数）与风险不成比例。

**R2 不提供 graph。** 不实现 `torch.corex.CUDAGraph`，不在文档里暗示可以「像 CUDA 那样 capture」。
要减少发射次数，走的是驱动侧已验证的「一次提交 N 条命令」那条路，并且**必须**连带实现
「私有参数池 + 包内地址改写 + fence 序号重写」——不做这三件事，重放会读到被回收的参数缓冲，
结果是设备页错误、卡死。

**R3 事件要少而短。** 每队列 4 个 user fence 槽意味着「同时挂起的事件」有硬上限。
`Event` 用完立刻 `cudaEventDestroy`；计时用一对事件，不要每步新建一堆。

**R4 性能模型按单流算。** `CONCURRENT_KERNELS = 1`、`UNIFIED_ADDRESSING = 0`，加上
「同一 cmdbuf 里的独立 kernel 会互相覆盖」这一观测，说明多流的收益在本平台不能按 NVIDIA 的经验估。
任何「靠 stream 并行提速」的设想都要先测。

**R5 设备指针与主机指针严格分开。** 非统一寻址：`cudaMalloc` 出来的指针只能给设备用，
Triton 的 `*fp32` 实参必须是设备内存。分配器返回的指针不许被当成可解引用的 CPU 地址。

**R6 串行性来自 UMD 的包，不来自「顺序下发」。** 所以**只能**通过官方入口提交；
任何「我自己按顺序放两条命令，硬件会按序执行」的假设都是错的。

**R7 分配器自己写，而且是缓存池。** 平台没有 memory pool（`cudaMallocAsync` 连声明都没有），
`cudaMalloc` 是**真分配**（3–12 µs/次）：页大小 2 MiB、走 KMD 的 `CreateAllocation`。
aten 会频繁申请/释放小块张量，直接把 `cudaMalloc` 挂到分配器上会被分配开销吃掉。
所以按 size-bucket 缓存（2 的幂，最小 512 B），大块一次拿、小块从池里切；`empty_cache()` 才还回去。

**R8 pinned 暂存区是必需的，不是优化。** host↔device 的拷贝每次都会重新 pin 页（毫秒级），
`.cpu()` / `.to(...)` / 权重上传都吃这条。做法：进程级保留一小块（可增长）pinned 缓冲，
拷贝先走它，再 `cudaMemcpyAsync` 设备段（走 `mem_handle`，不 pin）。

**R9 不用 `--default-stream per-thread`。** 该模式下 `cudaGraphLaunch_ptsz` 那类桩会静默返回 801；
我们本来也不用 graph，但编译选项要钉住，避免哪天被别人的默认值带偏。

**R10 权限/恢复的现实**：`iluvatar.ko` 的 `recover=0`、`itr_cp_hung_timeout_ms=0`（默认永不判定），
意思是**代码一旦把 CP 卡住，这张卡就废到管理员复位为止**。所以探针一律带硬超时、先验卡、
只在队列沙箱里跑；新写法先在 CPU/纯编译路径上验证，再上卡。

## 3. 仍然 UNKNOWN（会影响后面阶段，不会被上面堵住）

| 未知 | 怎么测 |
| --- | --- |
| 多条流在 wall-clock 上到底有没有并发收益 | 两条流各跑一段带宽 kernel，比较总时间 |
| `cudaMalloc` 的代价分布（不只是 p50） | 量 1 KB / 4 MB / 256 MB 的单次分配耗时分布；决定 size-bucket 的档位 |
| `cudaEventRecord` 的固定开销 | 空 kernel 下 record+sync 的往返耗时分布；决定计时口径 |
| 显存标称 32 GB、`cudaMemGetInfo` 报 31.72 GiB 可用，且 free/total 在空载时相同 | 分配几个大块后再读一次，确认这项统计是否真在更新 |
