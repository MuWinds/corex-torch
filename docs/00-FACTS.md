
> 节号有空洞是正常的：同一结论的中间读数与更正已并入最终那一节，节号不复用，以便其他文档里的引用继续有效。
# 已确证的环境事实

每条事实后面跟产生它的命令。新结论按同样格式往下加；没有证据的写 `UNKNOWN`。
探针脚本在 `probes/`，编号与这里引用的命令对应。

采集环境：`cx-sandbox`（SCOW 作业 **4935**，节点 `gpu-node-07`，健康物理卡 6、7），
2026-11-06 23:56。

## 1. 两个沙箱作业（别用错）

| 事实 | 证据 |
| --- | --- |
| 作业 4930 `q4-sbxA` 在 **gpu-node-06**（卡全坏），工作目录 `~/coffin-work/sbxA` | `python -c "…running_jobs()"` 输出 `nodesOrReason: gpu-node-06` |
| 作业 4935 `q4-sbxB` 在 **gpu-node-07**，健康卡 **6、7**，工作目录 `~/coffin-work/sbxB` | 同上 `gpu-node-07`；`cat ~/coffin-work/base/sandbox/logs/env.txt` 得 `healthy=6,7` |
| **两个作业共用同一个队列目录** `~/coffin-work/base/sandbox`：`heartbeat.txt` 在 `gpu-node-06 healthy=none` 与 `gpu-node-07 healthy=6,7` 之间交替 | `python tools/remote.py status` 连看两次，`HB … host=` 会变 |
| 后果：坏节点那个进程因为 `HEALTHY_IDS` 为空**不消费队列**，只空转；但队列项一旦被它抢到就会跑在坏卡上 | `corex-triton/tools/sandbox.sh` 第 110–113 行 `if [ -z "$HEALTHY_IDS" ]; then sleep 30; continue; fi` |
| 因此每条碰卡的探针脚本**开头自己验卡**，坏卡 `exit 9`，不占队列 | `probes/001`… 各脚本头部 |
| 作业有 8 小时时限：4935 约在 **2026-11-07 07:28 (+08)** 结束 | 作业详情 `timeLimit: "07:50:00"`，`startTime 2026-11-06T15:37:55Z` |

## 2. 登录节点与计算节点的分工

| 事实 | 证据 |
| --- | --- |
| 登录节点 `mgt` 上**没有 `/usr/local/corex`**：`ls -d /usr/local/corex*` 输出为空 | `python -m scow run 'ls -d /usr/local/corex*'` |
| 共享 py310 环境在登录节点 `import torch` 失败：`OSError: libcufft.so.10: cannot open shared object file`（即使把 corex 的 lib64 加进 `LD_LIBRARY_PATH` 也一样，因为目录根本不存在） | `/data/software/Anaconda3/2024.10-1/envs/py310/bin/python -c "import torch"` |
| 所以：pip / venv / 读源码 / 编 CPU 扩展放登录节点；任何 `torch.cuda`、CoreX 头文件、碰卡的事放沙箱 | 上面两条的直接推论 |

## 3. 厂商运行时到底有什么（`probes/001_corex_surface.sh`）

| 事实 | 证据 |
| --- | --- |
| CoreX 3.2.1，`/usr/local/corex -> /usr/local/corex-3.2.1/` | `ls -ld /usr/local/corex` |
| 兼容层自称 **CUDA 10.2.89**：`libcudart.so.10.2 -> libcudart.so.10.2.89`，头文件 `CUDART_VERSION 10020`、`CUDA_VERSION 10020` | `readlink -f $C/lib64/libcudart.so`；`grep -hn "define CUDA_VERSION" $C/include/cuda.h` |
| 有 **runtime API**（`libcudart.so.10.2`）与 **driver API**（`libcuda.so.1`）两套，都有 | 同上 `readlink -f` |
| 数学库版本：`libcublas.so.10.2.3.254`、`libcublasLt.so.10.2.3.254`、`libcusolver.so.10.3.0.89`、`libcusparse.so`、`libcurand.so`、`libcufft.so.10.1.2.89`（`CUFFT_VER 10.1`）、`libcudnn.so.7` | `readlink -f $C/lib64/libcublas.so` 等 |
| 有 `libnccl.so.2.11.4` | `readlink -f $C/lib64/libnccl.so` |
| 编译器：`clang 16.0.6`（厂商分支）、`llc`、`lld`/`ld.lld`、`fatbinary`、`nvcc`（壳） | `$C/bin/clang++ --version` |
| **gcc 10.3.1 / g++ 10.3.1** 在计算节点上可用；`cmake 3.22.0` | `gcc --version`、`cmake --version` |
| 设备节点 8 个：`/dev/iluvatar0..7`；内核模块 `iluvatar` | `ls /dev \| grep iluvatar`、`lsmod` |

## 4. 厂商 torch 2.1（参照物，不参与适配）

| 事实 | 证据 |
| --- | --- |
| 共享 py310 里是 **`torch-2.1.0+corex.3.2.1`**（厂商按 CoreX 版本重发的 wheel） | `ls …/py310/lib/python3.10/site-packages/ \| grep -i torch` |
| 它带 `libtorch_cuda.so`、`libtorch_cpu.so`、`libnvfuser_codegen.so`（有 nvFuser） | 同上 `torch/lib/` |
| 同环境还有一整套 `nvidia_*_cu12-12.4.x` 的 pip 包（CUDA 12.4 用户态） | 同上 `grep -i nvidia` |

## 5. torch 2.8 的安装现状

| 事实 | 证据 |
| --- | --- |
| 隔离 venv 已建好：`~/venvs/corex28`（python3.10，**不带** `--system-site-packages`，与厂商 torch 2.1 完全隔离） | `~/venv28.sh`、`cat ~/venvs/corex28/pyvenv.cfg` |
| 装 torch 2.8.0 CPU 失败过一次：venv 里 pip 23.0.1 太老，下载 `typing-extensions` 时判 `expected 'typing-extensions', but metadata has 'typing_extensions'`，随后错误地退化成源码构建并卡在 `flit_core<4,>=3.11` | `~/venvs/corex28-pip.log` |
| 隔离的理由（原脚本注释）：py310 里的 `torch 2.1.0+corex` 是全环境共用的，装 2.8 会覆盖它，而 2.1 后面还要用来做数值基线 | `~/venv28.sh` 顶部注释 |

## 6. 设备层：runtime API 单独可用，不需要 torch（`probes/002_cudart_binding.sh`）

**这是本次适配最关键的一条**：用 ctypes 打开 `libcudart.so.10.2`，不 import 任何 torch 版本，
就能把一个 context 用起来。所以不需要 `cuInit`/`cuCtxCreate`（那条路在本平台坏卡上挂死过，
`corex-triton/docs/00-FACTS.md` §5），runtime 会自己延迟建 primary context。

| 事实 | 输出 |
| --- | --- |
| 自报版本 `cudaRuntimeGetVersion` / `cudaDriverGetVersion` 都是 **10020** | 同上探针 §1 |
| 上面那条 `missing symbols` 里只有我自己写错的假名字（`cudaMemcpyKind_dummy`），其余 23 个真符号全在 | 同上 |
| `cudaGetDeviceCount` / `cudaSetDevice(0)` / `cudaGetDevice` 全部 rc=0 | 同上 §2 |
| 设备名 **`Iluvatar BI-V100`** | `cudaGetDeviceProperties` |
| `warpSize = 64`（与 `corex-triton/docs/00-FACTS.md` §8 的实测一致）、`maxThreadsPerBlock = 4096`、`multiProcessorCount = 16`、`sharedMemPerBlock = 131072`（128 KB） | `cudaDeviceGetAttribute(10/1/16/8)` |
| 显存 **31.72 GiB**（`cudaMemGetInfo` 的 free 与 total 在空载时相同） | 同上 |
| `cudaMalloc(1 MiB)` → `cudaMemcpy` H2D → D2H，**往返逐字节相等** | 探针 §3 `ROUNDTRIP_BYTES_EQUAL = True` |
| 设备指针形如 `0x304000002000000`（高位非规范地址，别假设它是 CPU 地址） | 探针 §3 打印 |
| `cudaStreamCreate` / `cudaMemcpyAsync` / `cudaStreamSynchronize` 全部 rc=0 | 探针 §4 |
| `cudaEventCreate` / `cudaEventRecord` / `cudaEventSynchronize` / `cudaEventElapsedTime` 全部 rc=0 | 探针 §4 |
| `cudaGetErrorString(0)` = `no error`、`(1)` = `invalid argument`；`cudaDeviceSynchronize` rc=0 | 探针 §5 |

含义（写进 `02-RUNTIME-API.md` 的取舍）：设备层的分配/拷贝/流/事件可以直接绑 runtime API，
**不需要 C++ 也能跑**；C++ 只需要在 PrivateUse1 要求「从 C++ 注册分配器」时才出场。

## 7. 沙箱里的两个操作陷阱

| 陷阱 | 现象与正确写法 |
| --- | --- |
| 覆盖 `CUDA_VISIBLE_DEVICES` | runner 已把健康卡写进 CVD（`6,7`），子进程里逻辑 0 号 = 物理 6 号。**探针里再写 `CUDA_VISIBLE_DEVICES=0` 会指到物理 0 号（坏卡）**，症状是开头验卡就 `CARD_UNHEALTHY`（2026-11-06 踩过一次）。正确做法：有继承值就直接用，没有才逐卡挑 |
| 中文乱码 | Windows 侧 Python 默认按 GBK 输出，`remote.py` 打回来的日志会花。设 `PYTHONIOENCODING=utf-8` 或直接用 read 工具读落盘文件 |
| **同名脚本被 `done/<name>.running.sh` 挡住** | runner 的 `mv queue/x.sh done/x.running.sh` 一旦碰到已存在的同名 `.running.sh` 就不覆盖/不执行，队列里那个脚本会一直躺着不被消费（`for` 循环里 `mv ... \|\| continue` 静默跳过）。**重跑请换名字**；`sbx-reset` 只清 `rc`/`log`，清不掉 `.running.sh` |
| **脚本最后一句是 `echo` 时 rc 恒为 0** | 探针必须把被测命令的退出码 `exit` 出去，否则日志里全是 traceback 而 `done/*.rc` 仍是 0，失败要人去翻日志 |

## 7.1 Python/ctypes 的一个坑（与平台无关，但在这条路上必然遇到）

`int(ctypes.c_void_p(...))` **不返回地址**：CPython 的 ctypes 给 `c_void_p` 实现的 `__int__`
走的是 `int(bytes(obj))`，即把 8 字节地址当十进制字符串解析。地址字节恰好全是 ASCII 数字时它
「能用」，否则抛 `ValueError: invalid literal for int() with base 10: b'\x00\x00\x00\x02...'`。

```python
>>> int(ctypes.c_void_p(0x0304000002000000))
ValueError: invalid literal for int() with base 10: b'\x00\x00\x00\x02\x00\x00\x04\x03'
```

取地址一律用 `.value`。本仓库把这条规则收在 `_runtime._ptr()` 与 `_triton._device_addr()` 两处。
（2026-11-07：`_triton` 第一版就是这样炸的，报错信息里连指针都没有，很难往「取地址」上想。）

## 8. UNKNOWN（下次探测要什么手段）

| 未知 | 需要的手段 |
| --- | --- |
| 纯 Python 能否给 `PrivateUse1` 注册显存分配器 | torch 2.8 wheel 装上后实测，见 `docs/03-TORCH28-API.md` |
| 动态 shared 的上限行为 | 已确证：默认就能到 128 KiB，132 KiB 才 `invalid argument`；`cuFuncSetAttribute(8, N)` 只能收紧（`probes/153`） |

## 9. 厂商 torch 2.1 的算子成绩**不能当作 Triton 的基线**（用户当面纠正，2026-11-07）

之前 `corex-triton/docs/00-FACTS.md` §12/§14.3 里「厂商 torch」那一列被当成「Triton 本可以达到的
水平」来对比，这是错的。逐算子看是谁在干活：

| 算子 | 实际执行的 kernel | 谁实现的 | 成绩 |
| --- | --- | --- | --- |
| `copy_` | `Memcpy DtoD (Device -> Device)` | **驱动的 DMA 拷贝引擎**，连 kernel 都不是 | 51.0 µs / 471 GB/s |
| `add` | `at::native::modern::elementwise_kernel<CUDAFunctor_add<float>, Array<char*,3>>` | ATen 手写 CUDA kernel（向量化） | 100.4 µs / 402 GB/s |
| `matmul` | `Gemm_tcu_bi_kernel::gemm_tcu2h_trans<float,64,64,32,16,16,true,true>` | 厂商手写 GEMM，**`tcu2h` 就是矩阵单元** | 17.8 µs / 1577 GFLOP/s |
| `softmax` | `softmax_warp_forward<float,float,float,10,false,false>` | ATen 手写 warp 级 kernel | 15.0 µs（64×1024） |

kernel 名可以在 `corex_drive_re/perf/pulled/` 下的 `ixprof` 原始输出里交叉核对
（`Gemm_tcu_bi_kernel::gemm_h_h_tcu_25`、`elementwise_kernel<CUDAFunctor_add<Half>>` 都在）。

**含义（决定我们每个算子该走什么机制）**：

| 我们的选择 | 依据 |
| --- | --- |
| `copy_` / `_to_copy` / `.contiguous()` **必须走 `cudaMemcpyAsync`（D2D/DMA）**，不写 Triton 内核 | 471 GB/s 是 DMA 引擎的成绩；Triton 逐元素内核实测只有 295 GB/s（`../corex-triton/docs/00-FACTS.md` §14.3），差 1.6 倍，且 DMA 不占 SM |
| 逐元素算子的天花板按 **ATen 402 GB/s** 记，Triton 版当前 295 GB/s（约 0.73x） | 差距来自向量化宽度，不是语义；后面要专门量一遍 |
| `matmul` 在接上 TCU 之前**只能按 FMA 算**（105 GFLOP/s，约 1/15） | 厂商那 1577 GFLOP/s 是矩阵单元的成绩，`tl.dot` 走 FMA 不可能追上 |
| softmax 这类有 warp 级规约的算子按 Triton 的成绩单独量，不与 ATen 的 `softmax_warp_forward` 混比 | 尺寸不同不可直接比（ATen 那组是 64×1024） |

**这条对阶段 4 的验收口径有直接影响**：算子表里要写「我们的数 / CPU 参照（正确性）」
与「我们的数 / ATen 同类实现（性能）」，**并且注明后者用的机制**，不能笼统写「对厂商 torch 的比值」。

## 10. 设备层与算子层引擎都已验通（2026-11-07）

| 事实 | 证据 |
| --- | --- |
| `_runtime` 在真机上可用：设备属性、分配、H2D/D2H 逐字节往返、流、事件全部正常 | `probes/003_runtime_module.sh`：`ROUNDTRIP_BYTES_EQUAL = True`、16 个事件同时挂起全部成功 |
| `cudaMalloc` 实测约 3–12 µs/次、`cudaFree` 约 2 µs/次（1 KiB / 1 MiB / 64 MiB 三档） | 同上 §4。对着 aten 的分配频率，缓存池仍然是必需的（`05-DRIVER-CONSTRAINTS.md` R7） |
| `record + record + sync` 一轮约 **29 µs**；空载 `cudaDeviceSynchronize` 约 **1.1 µs** | 同上 §4/§5。计时口径要按这个算：一步里若插十几次同步，光同步就上百 µs |
| **Triton 适配层（`_triton`）在无 torch 进程里全通**：编译 424 ms → cubin 3464 B → 装载 → 启动 → 逐字节相等 | `probes/006_triton_adapter.sh`：`ADAPTER_VERDICT PASS`，n=4096 与 n=4000（尾块掩码）都 `max_abs_diff=0.000e+00` |
| 我们这条链产出的 cubin 与 `probes/151`（独立写的另一条链）**字节数相同（3464 B）** | 两次独立测量，说明编译产物是确定的 |
| `CompiledKernel.__getitem__` 的 runner **不补齐网格维度**（只传 `(16,)` 会 IndexError） | `_triton.launch()` 负责补成 3 元组 |

## 11. 拷贝路径实测（`probes/007_copy_bandwidth.sh`）

口径：单卡（沙箱 4935 逻辑 0 号 = 物理 6 号）、`cudaMemcpyAsync` + 自建流、`hostname=gpu-node-07`。

| 路径 | 尺寸 | 平均耗时 | 带宽 |
| --- | --- | ---: | ---: |
| D2D（走 mem_handle，不 pin 页） | 1 MiB | 0.014 ms | 145.0 GB/s |
| D2D | 16 MiB | 0.052 ms | **649.3 GB/s** |
| D2D | 64 MiB | 0.236 ms | 568.5 GB/s |
| D2D | 256 MiB | 0.901 ms | 595.6 GB/s |
| H2D **pageable** | 1 / 16 / 64 MiB | 0.260 / 3.748 / 14.533 ms | **4.0 / 4.5 / 4.6 GB/s** |
| H2D **pinned** | 1 / 16 / 64 MiB | 0.051 / 0.615 / 2.424 ms | **20.5 / 27.3 / 27.7 GB/s** |

| 事实 | 数字 |
| --- | --- |
| 小拷贝的固定开销（H2D） | 1 KiB 约 7.3 µs；64 KiB 约 9.5 µs；1 MiB pinned 45 µs vs pageable 142 µs |
| 流同步 / 设备同步 | 1.38 µs / 1.36 µs（都很便宜，「每步插几次同步」的预算很宽） |
| **pageable 主机内存是 pinned 的 ~1/6 带宽**（64 MiB 时 4.6 vs 27.7 GB/s），且首次数 38.2 ms | 证实驱动审计 §B2：host 路径每次 `vmalloc` + `get_user_pages_fast` |

**结论**：`copy_` / `_to_copy` 一律走 `cudaMemcpyAsync`，D2D 能到 595–649 GB/s（比厂商那条 471 GB/s 的量级还高一点，
尺寸不同不可直接比）；凡是要经过主机内存的路径，必须走 pinned 暂存区，否则直接丢掉 5/6 的带宽。

## 12. torch 2.8 的 PrivateUse1：分配器**只能**从 C++ 注册（`probes/101`–`108`）

| 事实 | 证据 |
| --- | --- |
| `torch.empty(device="corex")` 取的是 `c10::GetAllocator(PrivateUse1)`，**默认返回 nullptr 且不报错** | `probes/102`：注册前 `OK -> 0`，注册后 `OK -> 107123696` |
| Python 侧没有任何设置 PrivateUse1 分配器的入口（`torch._C` 上 privateuse 开头的只有 `_get_privateuse1_backend_name` / `_rename_privateuse1_backend`） | `probes/108`；三处（运行时 `dir`、`torch/_C/__init__.pyi`、源码 grep）都翻过 |
| 算子可以用纯 Python 注册（`torch.library.Library("aten","IMPL").impl(...)` 实测成功），但**没法把自造指针包成 Tensor**——那个 ABI 只有 C++ 有 | `probes/101`、`102` |
| 最小 shim 用 g++ 10.3.1 + torch 2.8 CPU wheel 的 headers **编得过（26.4 s）** | `probes/102` |
| 2.8 相对 2.1 的两个坑：`c10::Allocator::allocate` **非 const**；`Allocator` 新增纯虚 `copy_data(void*, const void*, size_t) const` | 同上（照 2.1 写会报 `marked override, but does not override`） |
| guard impl 宏的真名是 `C10_REGISTER_GUARD_IMPL`；`c10::SetAllocator` 从 `libc10.so` 导出 | `DeviceGuardImplInterface.h:349`、`Allocator.h:276` |
| `torch.<dev>` 这个 device module 必须可导入（C++ lazy init 会 `import torch.corex`）；`torch.__path__.append(我们的目录)` 实测可行 | `probes/106` |
| `torch.empty(4, device="corex")` 的三级失败梯度：只 rename → `ModuleNotFoundError: No module named 'torch.corex'`；再 `_register_device_module` → `NotImplementedError: Could not run 'aten::empty.memory_format' ...`；再补 C++ 分配器/guard/empty → 成功 | `probes/102` |
| `rename_privateuse1_backend("corex")` 之后 `torch.cuda.is_available()` 仍为 **False**（没被污染） | `probes/108` |

## 13. 设备端到端打通（2026-11-07 00:53，`probes/008_device_end_to_end.sh`）

`DEVICE_VERDICT PASS`，rc=0，收尾 `CARD_STILL_HEALTHY`。这一条意味着**上游 torch 2.8 在这张卡上
有了一张能用的设备**：分配、拷贝、逐元素算子、标量回读全部走通。

```
[OK] torch.zeros(4, device='corex')            corex:0 float32 Size([4]) data_ptr=0x304000002000000
[OK] zeros 清零正确（.cpu() 读回）
[OK] torch.ones(8, device='corex')
[OK] CPU -> corex 往返逐字节相等                 往返 16 个元素完全相等
[OK] corex 上的 a + b
[OK] .item()                                   item = 1.0
[OK] 非连续 strides（from_blob）                2x3 按 (1,2) 读，sum=6.0
[OK] 大张量分配 + 填 + 读回（4 MiB）
[OK] empty_cache
```

链路上踩到的三个坑（都不是平台问题，但每一个都会静默算错）：

| 坑 | 现象 | 根因与修法 |
| --- | --- | --- |
| `aten::_copy_from` 的**参数顺序** | `t.cpu()` 读回一堆垃圾且 rc=0 | schema 是 `_copy_from(self, dst, non_blocking)`——第一个实参是**源**。按 `(dst, src)` 命名写，`t.cpu()` 会走进 H2D 分支把主机缓冲当地址写 |
| `torch.dtype` 不能 `int()` | `TypeError: int() argument must be ... not 'torch.dtype'` | C++ shim 要 `at::ScalarType`；`_device._scalar_type()` 维护枚举表并用 `element_size` 交叉校验 |
| 算子层找不到 corex-triton | `No module named 'triton_corex'`，且只在第一次编内核时暴露 | 它没打包安装，运行前要给 `COREX_TRITON_SRC`；`_triton.activate()` 会按仓库同级目录兜底 |

另外两条环境前提：
* `~/venvs/corex28` 里**必须装 triton 3.4**（torch 2.8 的官方配对），否则算子层的 `tl` 是 `None`，
  报 `AttributeError: 'NoneType' object has no attribute 'float32'`；
* `_corex_shim.so` 的 import 名是 **`_corex_shim`**（`PyInit_` 符号由文件名决定），且 import
  之前必须先 `import torch`（否则 `libc10.so` 找不到）。

## 14. `_triton` 改成薄适配（corex-triton 自己已经去掉 torch 依赖）

`corex-triton` 提交 `cabf39a` 把 driver 的 torch 依赖去掉了（主 context 走
`cuDevicePrimaryCtxRetain`、stream 用 legacy 默认流、显存 `cuMemAlloc_v2`、计时自己 `cuEvent`）。
所以本仓库不再自己实现运行期，`_triton` 只留四件事：定位源码目录、`activate()`、`compile_kernel`、
`launch`（补网格维度 + 把 `c_void_p` 换成 int）。实测 `probes/006` 仍然 `ADAPTER_VERDICT PASS`，
编译 431 ms、cubin 3464 B、n=4096/4000 全 0 误差、全程 `torch_modules=0`。

**后果**：`corex-triton` 一升级（比如 00:31 那次加了 16 位访存模板），我们这边不用改就能吃到——
fp16 算子就是靠那次升级才通的（`docs/04-TRITON-BRIDGE.md` §6）。

## 15. torch 层的算子验收（`probes/009_torch_ops.sh`）

内核层（`probes/170`–`182`）验的是「每个 kernel 与 CPU 参照对得上」；这一层验的是**分发有没有接错**
（形状/stride 检查、out 张量对账、dtype 转换、composite 落点）。8/9 通过，唯一失败的是
`as_strided`（C++ shim 还没导出它，`.t().contiguous()` 走不下去）。

| 用例 | 结果 | 判据 |
| --- | --- | --- |
| `mm` 64×128 @ 128×256 | **max_rel = 0.000e+00** | 与 CPU 逐元素完全相等 |
| `bmm` 4×16×32 @ 4×32×24 | **max_rel = 0.000e+00** | 同上 |
| `softmax` 32×512 / 2×8×256，dim=-1 | 8.14e-07 / 7.80e-07 | 门槛 1e-6（`ex2.approx` 的误差） |
| `sum()` 全归约 4096 | 5.59e-07 | 门槛 1e-5 |
| `sum(dim=-1)` 64×1000 | 2.39e-06 | **vs float64 参照**，门槛 1e-4 |
| `mean(dim=-1)` 64×1000 | 6.96e-06 | 同上 |
| `index_select` 7 行（i64 下标） | **逐字节相等** | — |
| 注意力小块（`q@k.t().contiguous()` + softmax + `p@v`） | **未通过**：`_corex_shim` 无 `as_strided` | 需要 shim 补一个「按 storage_offset 造视图」的工厂 |
| `mm` 256³ 端到端（含派发） | 341.4 µs → 98.3 GFLOP/s | 内核本身 177.7 µs / 188.8 GFLOP/s |

两条口径说明（避免后人误判）：

1. **归约类的参照要用 float64**。CPU 的 `sum`/`mean` 走 pairwise 求和，比顺序累加准得多；
   拿它当 1e-5 门槛会误判成「我们的实现错了」——顺序 fp32 累加 1000 项的固有相对误差本来就在
   1e-5 量级。所以这两项改成「对 float64 参照 ≤ 1e-4」。
2. **端到端比内核慢一倍**（341 vs 178 µs）。差额在 Python 派发 + `torch.empty` + ctypes 打包上，
   元素级算子那边实测是 21–36 µs/次（`docs/04-TRITON-BRIDGE.md` §6），mm 这一档为何到 160 µs
   量级还没定位（UNKNOWN）。要把这个差额压下去只能把参数打包与 `cuLaunchKernel` 挪进 C++ shim，
   接 coffin 的 decode 之前必须解决——那一步每 token 有六百多次发射。

## 16. host 开销：发射已在 C++，分发器几乎免费（`probes/010`、`023`、`026`）

统计口径一律 **p50 + 迭代 ≥500 + 两条基线**（空循环、最小真内核）。均值会被这张卡上约 4%
的驱动抢占停顿（≥100 µs）毁掉——这一点先用基线确认过，不是猜的。

| 段 | p50 |
| --- | ---: |
| 纯 Python 空循环 | 0.2 µs |
| `fill_` 64 元素（最小的真内核） | 32.7 µs |
| `aten::add.out` 直调（C++ 实现） | **5.2 µs** |
| `add_.Tensor`（C++ 实现） | **1.96 µs** |
| `mm [1,64]×[64,64]`（C++ 实现） | **3.1 µs** |
| `a + b`（composite：分配 + `.out`） | 12.5 µs |
| `a * 标量` | 14.0 µs |
| `a * small`（广播） | 15.4 µs |
| 分配 `empty_strided` | 1.76 µs |

**结论**：host 侧的固定成本已经被压到个位数微秒，**aten 分发器本身几乎免费**。真正贵的是
「用 Python 实现算子」——同一个 `add`，Python 实现要 78.8 µs、C++ 实现 5.2 µs，差 15 倍。
所以压缩 host 开销的唯一有效路径是**把热路径算子搬进 C++**（`TORCH_LIBRARY_IMPL` +
`set_*_kernel` 登记内核句柄），而不是减少 aten 调用次数。

分配（1.76 µs）不是问题——缓存池已经把它压到位。

## 17. bf16：转换走厂商 intrinsic，向量形态标量化（`probes/011`–`015`、`192`）

厂商 torch 2.1 在这张卡上能跑 bf16，靠的是**两条 intrinsic**，不是位运算。判断依据是宏：
厂商 clang 定义 `__CUDA_ARCH__ = 300`、`__ILUVATAR__ = 1`，于是 `c10/util/BFloat16-inl.h` 里
`(__CUDA_ARCH__ >= 800 || defined(__ILUVATAR__))` 为真，转换落到 `__float2bfloat16` /
`__bfloat162float`，它们在 `cuda_bf16.hpp`（427 / 543 行）里展开成：

```
%n = call float  @llvm.bi.bfloat2float(bfloat %x)
%n = call bfloat @llvm.bi.float2bfloat(float %x)
```

厂商 clang 编出来的设备 IR 里 `fpext`/`fptrunc` **命中 0 次**，只有上面两条；`llc` 编得出来。
而 Triton 发的正是 `fpext`/`fptrunc`，bi 后端不认这两个名字，于是报
`Cannot select: f32 = fp_extend` / `bf16 = fp_round`。

顺带一条：**ATen 的 bf16 算术本质是 fp32 算术**——
`operator+(BFloat16 a, b) { return static_cast<float>(a) + static_cast<float>(b); }`。
所以「纯 bf16 算术」在语义上就是「fp32 算 + 两次转换」。

### 做法：在 biir 进 `llc` 之前改转换指令的形态

实现落在 `src/corex_torch/_bf16.py`，挂在 `_triton.activate()` 注册的后端子类
`CorexTorchBackend.make_cubin` 上，**corex-triton 一个字节没改**。内核侧不需要为 bf16 写第二份。

| 形态 | 做法 |
| --- | --- |
| 标量 `fpext bfloat` / `fptrunc … to bfloat` | 换成 `llvm.bi.bfloat2float` / `llvm.bi.float2bfloat` |
| **向量** `<N x bfloat>` ↔ `<N x float>` | 展开成 `extractelement → 标量 intrinsic → insertelement` 链 |

向量为什么不直接用向量 intrinsic：`@llvm.bi.float2bfloat.v8bf16` 这类 mangled 名字
**`llc` 收下但不实现**——目标文件里留一个外部符号，`lld --no-undefined` 才报 `undefined symbol`。
标量版是真实现的，所以标量化是唯一稳妥的向量通路。另留两档可选策略
（`intrinsic`：部分元素数确有实现但不通用；`bits`：整数位运算，已验逐字节相等），
切换用 `_bf16.set_vector_strategy(...)`，内核代码不用改。

**判据硬规矩**：凡是要判断「bi 后端支不支持某个 intrinsic / 某条指令」，必须
`llc -march=bi` **和** `lld --no-undefined` 都过。LLVM 允许目标文件里留未解析的外部调用，
只看 `llc` 的返回码会把「parse 进去但没 lowering」误判成「支持」。

### 两条实现上的坑

* **新生成的 SSA 名字不能拿原值名加后缀**：Triton 的值是编号值 `%257`，而 `%257.i` 不是合法
  标识符（`expected '=' after instruction id`）。命名必须以字母开头，见 `_bf16._NameGen`。
* **向量常量不能写成 `splat (i32 16)`**：那是 LLVM 20 语法，厂商 llc 是 16。必须展开成
  `<i32 16, i32 16, …>`。

### 结果

`probes/192`：元素级 **33/33 逐字节相等**、累加类 **15/15**、mm/bmm/gather **5/5**；
改写命中 **236** 处，改写后 `fpext`/`fptrunc`/`trunc nuw`/`trunc nsw`/`splat` **全 0**。
`probes/013` 独立复验：bf16 矩阵乘 4096/4096 逐字节相等。

## 18. torch 层算子验收转全绿（`probes/009`）

`TORCH_OPS_VERDICT PASS`，9/9。新增的两条修复：

* `div.out`/`mul.out` 的**标量右操作数**（`scores / sqrt(H)` 走的这条）之前直接报未实现，现在走内核的
  `*_scalar` 入口，`alpha` 先乘进标量。
* 多算子串联（两次矩阵乘 + softmax + 再一次矩阵乘）的判据换成**全局相对误差** `max|diff|/max|ref|`：
  逐元素相对误差在「参照值接近 0」的元素上会被放大到 1e-4 量级，看着像错其实是 fp32 累加的正常结果。
  换口径后注意力小块是 **1.62e-07**。

## 21. C++ 快路径的两个参数契约（`probes/123`、`126`）

C++ 接管 `aten::mm` 之后出现过「不报错、数值全错」的情况。根因是**两个参数契约错误**，
两者都能让 `cuLaunchKernel` 返回 0 而结果全错：

| # | 错法 | 后果 |
| --- | --- | --- |
| 1 | Triton 3.4 的 global scratch 指针追加在参数表**最后**，却按「第 4 个」传（spec `"ppppiii"`） | 内核收到的 `M == 0` → 所有 block 掩码全假 → **一个元素都不写** |
| 2 | 动态 shared 传 0，而 `tl.dot` 的布局转换要 `metadata.shared`（实测 2048 B） | 写了但值错，同样不报错 |

正确写法：spec = **`"pppiiip"`**（3 指针 + 3 个 i32 + 末尾 scratch），`load_kernel` 带上
`metadata.shared`。C++ 侧现在**校验 spec**（只接受 `"pppiii"` / `"pppiiip"`），这类错误会当场报出来。

**验收方法**：输出所在的池块先填哨兵 `-12345` 再释放，任何没被写到的元素都会暴露成哨兵。
五个形状（含 `37×53 @ 53×29` 非整除）在 aten 与直调两条路径上 `max_abs = 0.000e+00`、残留哨兵 = 0。

**性能**（修正后；此前那个「303.9 GFLOP/s」是在坏参数下内核空转时测的，不成立）：

| 段 | Triton Python | C++ 快路径 |
| --- | --- | --- |
| mm 16³ 纯发射 | 38.34 µs | **10.79 µs（3.6x）** |
| mm 256³ 直调 | 184.48 µs（181.9 GFLOP/s） | **171.79 µs（195.3）** |
| mm 256³ 走 aten | 265.8 µs | **193.26 µs（173.6）** |

发射层省下的 27.5 µs 是真的；mm 256³ 只提速 1.07x，因为内核本身 ~160 µs 占主导。
验收门槛据此定成：**直调路径 ≥180 GFLOP/s**；aten 路径只要求「比 Python 实现快」且各形状
逐元素对得上。

**两条守则（都是这一轮换来的）**：

* 「某个算子实现正确」的判据必须是**非方阵/非整除形状 + 与 CPU 逐元素比对**，
  吞吐数字只能附在正确性之后——只测方阵吞吐会整类漏掉 grid/参数错误；
* 「某条指令/intrinsic 可用」的判据必须是 `llc` **且** `lld --no-undefined`（见 §17）。

## 23. 第 5 批算子在 torch 层全通（`probes/017`，12/12 PASS）

`relu` / `relu_` / `rsqrt` / `gelu`（erf 与 tanh 两种近似各自独立参照）/ `masked_fill` /
`log_softmax` / `max(dim=-1)`（值与下标，含 int64 dtype 与「第一个最大值」语义）/ `argmax` /
`view.dtype`，全部在 torch 层验过：逐字节相等的 6 项、max_abs 在 5e-07~1e-05 量级的 5 项。

两个坑值得记：
* **`torch.relu` 直接派发到 `aten::relu`，不经过 `relu.out`**——只注册 out 变体会报「算子未实现」。
  （对比：`add`/`mul` 那些是 composite，会自动拆到 out 变体。）
* **探针里 got/ref 必须用同一份输入**。第一版 017 写成两次独立 `torch.randn`，量出来的是「两次采样之差」
  （rsqrt 报 0.59、gelu 报 4.0、log_softmax 报 11.3），差点误判成内核错了。

## 25. 又一类静默算错：广播操作数没 expand 就被当同形读

`_out_binary` 原来只要求「out 形状 == 广播形状」，却没处理 `self`/`other` 本身更小的情况。
内核按 `out.numel()` 线性遍历三个指针，于是把 `[T,1,HD]` 的 `other` 当成 `[T,HEADS,HD]` 读——
**越界读、不报错、数值全错**。

触发点在 RoPE：`x1 * cos[:, None, :]`（`x1` 是 `[T,HEADS,HD/2]` 的切片视图，
`cos[:, None, :]` 是 `[T,1,HD/2]`）。`probes/019` 的逐阶段比对里，这一步 max_abs **6.4**
而前后各阶段都在 1e-5 量级。

修法：物化时必须显式 `expand(shape).contiguous()`，不能只 `.contiguous()`（后者保留原形状）。
同时把「非连续就报错」改成「非连续就物化」——真实前向里到处是切片视图，报错等于把
「能算但慢」变成「不能算」。

**教训**：`broadcast_shapes` 算出的形状与三个指针的**实际形状**是两件事；凡是「按 numel 线性
遍历指针」的内核，调用方必须先保证三者同形，否则就是静默越界。

## 26. 阶段 5 预演：一个 2 层 mini decoder 端到端跑通（`probes/018`）

结构：`embedding → rms_norm → qkv(mm) → RoPE(切片写) → bmm+softmax+bmm → o_proj → 残差 →
rms_norm → MLP(mm+silu+mm) → 残差 → rms_norm → head`，`VOCAB=32 DIM=64 HEADS=4 T=8`，两层。

| 项 | 结果 |
| --- | --- |
| 与 CPU 同结构前向的**全局相对误差** | **6.631e-06** |
| logits argmax 一致率 | **1.000** |
| 首次（含内核编译） | 412.5 ms |
| 热跑 | 56.7 ms |

逐阶段比对（`probes/019`）显示每一步都在 1e-5 量级以内，没有累积放大。

这条前向暴露的两个缺口都是**单算子表测不出来的**，正是「先接真实前向」的价值：
1. `F.rms_norm` 是 composite，会分解出我们没实现的 `aten::pow` → 必须**直接接管** `aten::rms_norm`
   （且要同时挂 `AutogradPrivateUse1`，只挂 `PrivateUse1` 会停在 `Autogradcorex` 层报算子不存在）；
2. 广播操作数与切片视图（见 §25）。

## 27. 归约补齐：中间维 + 方差/标准差（`probes/021`，13/13 PASS）

内核侧由 triton-bridge 的 `probes/197` 交付（`sum_axis`/`mean_axis`/`row_var`/`row_std`），
分发侧这一轮接上，`probes/021` 全绿：

| 用例 | 结果 |
| --- | --- |
| `sum(dim=中间维)`，shape `(4,8,16)`/`(2,3,4,5)`/`(4,1000)` | **全部 max_abs = 0.000e+00** |
| `sum(dim=1)` dyadic 数据 | **逐字节相等** |
| `sum(dim=1, keepdim=True)` | **逐字节相等** |
| `mean(dim=中间维)` / `mean(dim=-1)` / `mean()` 全归约 | 5.96e-08 / 5.96e-08 / 3.73e-09 |
| `var(dim=-1, correction=1/0)` | 4.77e-07（全局相对 1.1e-07） |
| `std(dim=-1, correction=1/0)` | 2.38e-07（全局相对 1.1e-07） |

两个 schema 坑（都在 `probes/021` 第一版踩到）：

* **`x.sum()` / `x.mean()` 派发下来的 `dim` 是空列表**，不是 `None`。空列表必须当「全归约」，
  否则会掉进「多维组合」的报错分支。统一收口在 `_norm_dims()`，别在每个实现里各写一遍。
* **`var.correction` / `std.correction` 的 schema 是 `int[1]? dim`**——哪怕调用时给的是 `-1`，
  进到实现里也是 `[-1]`。按 `int` 用会得到 `TypeError: int() argument must be ... not 'list'`。

为什么 `row_var` 用两趟而不是一趟公式（triton-bridge 实测，数据是「均值 1e6、标准差 0.4」，
待求量只有 0.156）：

| cols | 两趟 max_abs | 一趟公式（双精度） | 一趟公式（fp32 顺序累加） |
| --- | --- | --- | --- |
| 1000 | 7.458e-11 | 1.101e-02 | **6.986e+06** |
| 512 | 2.149e-06 | 2.150e-06 | **3.283e+06** |
| 4097 | 3.352e-08 | 9.308e-04 | **1.901e+07** |

**即使把一趟公式放到双精度里算，它也比我们的 fp32 两趟差**——所以问题不是精度不够，是公式相消。
`sum(x²) − n·mean²` 这条路在内核里不该出现。

## 28. 逐步 decode（KV cache）跑通：最接近 coffin 实际用法的形态（`probes/022`）

`probes/018` 是无 cache 的整段前向，它证明不了三件事：单 token 时注意力在
`[HEADS,1,HD] @ [HEADS,HD,L]` 这种**非方阵**上对不对、KV cache 增长后历史位置是否仍正确、
逐 token logits 会不会随步数漂移。`probes/022` 把这三件都测了：

| 项 | 结果 |
| --- | --- |
| 逐 token logits 全局相对（6 步） | 2.10e-06 / 2.41e-06 / 6.62e-06 / 4.87e-06 / 5.69e-06 / 3.81e-06 |
| 每步 argmax 与 CPU 一致 | **6/6** |
| 两次独立 decode 的 KV cache | **逐字节相同** |
| 热跑 | 102.5 ms / 6 步 = **17.1 ms/步**（`VOCAB=32 DIM=64 HEADS=4 LAYERS=2`） |

误差不随步数增长（第 6 步反而比第 3 步小），说明 KV cache 这条路上没有累积放大。

**踩到一个 stride 语义坑**：`[1,HEADS,HD] -> permute(1,0,2).contiguous()` 想得到 `[HEADS,1,HD]`
的连续视图，但 torch 的连续性判定**忽略长度为 1 的维**，所以 `.contiguous()` 是**空操作**，
stride 仍是 `(16, 64, 1)`。我们 `cat` 的检查按「dim 之后每维 stride = 尾积」判定，于是拒收。
改用 `squeeze(0).unsqueeze(1)`（stride 变成 `(16, 16, 1)`）两边都认。
教训：**「.contiguous() 之后一定连续」是错的**——size 为 1 的维会骗过判定。

**性能观察（不是正确性问题）**：17.1 ms/步，按每步约 31 个算子算就是 ~550 µs/算子，
远高于 10.8 µs 的发射地板。怀疑点是 `cat` 与 `expand().contiguous()` 走的是 Python 级
分段搬（`_aten_extra` 的 strided walk），不是单次 DMA。这条留给性能阶段，先记在这里。

## 29. 性能结论的读法与最终数字（`probes/023`、`024`、`026`、`036`）

**统计口径**：一律 p50 + 迭代 ≥500 + 两条基线（纯 Python 空循环、最小的真内核）。
这张卡是多租户共享的，约 4% 的迭代会吃到 ≥100 µs 的驱动抢占停顿，且停顿与工作量、形状
完全无关；50 次的**均值**里混进一次 ~120 ms 的停顿就能把 13.5 µs 的算子报成 2.5 ms。
基线的作用是划清「这个慢是环境的还是代码的」——空循环与纯 `synchronize()` 的停顿率都是 0/N，
而只要真的发内核就有停顿（连 64 元素的 `fill` 都有）。

**当前数字（C++ 快路径已就位）**：

| 算子/路径 | p50 | 说明 |
| --- | ---: | --- |
| `add_.Tensor` | **1.96 µs** | 原地，三档 dtype |
| `aten::add.out` 直调 | **5.2 µs** | |
| `mm [1,64]×[64,64]` | **3.1 µs** | |
| `a + b`（分配 + `.out`） | 12.5 µs | |
| `a * small`（广播） | 15.4 µs | |
| `fill_` 64 元素 | 32.7 µs | 最小真内核的基线 |
| Python 实现的同类算子 | 67–127 µs | 这就是「搬进 C++」的依据 |

**搬运算子之外的另一半**：非连续搬运的成本**取决于调用次数而不是字节数**——本平台同步
`cudaMemcpy` 每次 5–13 µs。所以 `_copy_impl` 按「尾维合并 / `memcpy2D` / 倍增复制」分路径，
把「一次拷贝 = 数条 Python 调用」压成 1 条：转置 `contiguous()` 2000 → 91.7 µs、`cat` 117 →
26 µs、1 KiB~1 MiB 的连续拷贝都是 ~10 µs（与大小无关 = 纯开销驱动）。

**decode 的构成**（`probes/024` 的 profile）：从 17.1 → **4.9 ms/步**之后，最大的单项已经
不是后端，而是测试脚本自己的 RoPE 写法（`out[..., 0::2] = x1*c - x2*s` 一次就是
4 次广播乘 + 2 次减 + 2 次步长视图写）。后端侧剩下的热点是 `silu`/`rms_norm`/`softmax`/
`index_select` 这四个仍是 Python 实现（每步约 14 次发射）。再往下走要动的是**模型侧融合**
（RoPE 单内核、残差与归一化合并），属于 coffin 集成阶段。

## 33. 热路径算子的 C++ 交接：接口就位、代价已量（`probes/025`）

内核组把「稳定取内核」和「实参顺序表」做成了机器可读的单一真源
（`_ops_elementwise.fast_op()` / `get_launch_meta()` / `fast_variants()`），并用一个**不许用内部知识**
的 shim 模拟器验了 24 组（4 op × 2 variant × 3 dtype，n=4097）**逐字节正确**（`probes/199` PASS 29/29）。
这一点很关键：实参顺序若错位（长度与 alpha 互换成 float 与 i32），内核跑完不报错但结果全错，
只有「按表填 + 逐字节对参照」才抓得住。

注册代价（`probes/025`，磁盘缓存已热）：

| 项 | 时间 |
| --- | --- |
| `corex_torch.register()`（含 mm 三内核 + 能力探测） | **471 ms** |
| 24 个 fast-op 内核首轮 `get_launch_meta()` | **124 ms**（平均 5 ms/个） |
| 第二轮（缓存命中） | **0 ms** |

冷缓存时按单内核 ~0.4 s 估，最坏 ~10 s——所以 `ENABLE_FAST_OPS` 用环境变量控制（默认开），
开关写在 `_aten_kernels` 里并注明原因。当前 shim 18 个导出、**没有** `load_fast_op`，
所以 `_register_fast_ops()` 目前是**空操作**（`hasattr` 探测），Python 实现继续生效。

**实参表的两个概念必须分开**（内核组把它做成了显式字段，很有必要）：

* `has_alpha`：签名里**有没有** alpha 槽位 —— tensor 变体都有；
* `alpha_effective`：alpha **是否真的改变结果** —— 只有 add/sub 的 tensor 变体是 True。
  mul/div 的槽位**仍然必须传**（`probes/199` 实测传 0.0 与 1.0 输出逐字节相同），
  少传就是实参个数不符 → 参数错位 → 「跑完了但全错」。

抽查两个（`probes/025` §3）：`add/tensor/fp32` → `n_ptrs=3 n_i32=1 n_f32=1
alpha_effective=True block=1024 warps=4 stages=1 cubin=7168B`；
`mul/tensor/bf16` → 同形但 `alpha_effective=False`、`cubin=7008B`。

## 35. 「谁在提供实现」的唯一硬判据：dispatch 表（`probes/026`）

时间会被上下几层掩盖，只有 dispatch 表能直接说清谁在服务。判断规则：

* `PrivateUse1: registered at /dev/null:<行号>` → **Python** 注册（`torch.library.Library.impl`）；
* `PrivateUse1: registered at .../corex_shim.cpp:<行号>` → **C++** 注册（`TORCH_LIBRARY_IMPL`）。

`probes/026` 实测（shim 21 个导出，C++ 声明接管 12 个 schema）：

```
aten::add.out      → /dev/null:639                        ← 仍是我们 Python 的（有意为之）
aten::add_.Tensor  → .../corex_shim.cpp:1793              ← C++
aten::mm           → .../corex_shim.cpp:1793              ← C++
C++ 声明的 12 个：add/sub/mul/div × {.out, _.Tensor, _.Scalar}
hasattr elementwise_supports_broadcast = False            ← 所以 .out 还没让位
```

**为什么 `.out` 还留着 Python**：C++ 的 `.out` 目前只做「同形、连续、同 dtype」。第一次全让位时
`x1 * 2.0` 直接抛「只支持同形」，把 decode 的 `x1 * cos[:, None, :]` 从「能算但慢」变成「不能算」。
现在用 `elementwise_supports_broadcast()` 做闸门：C++ 声明支持广播后自动让位，不需要改代码。

同一份探针的三种形态（p50/500，空循环基线 0.2 µs）：

| 形态 | p50 | 说明 |
| --- | --- | --- |
| `a.mul_(2.0)`（C++ inplace + 标量） | **3.1 µs** | 快路径真的在跑 |
| `aten::add.out` 直调 | 48.0 µs | Python 实现（`.out` 未让位） |
| `a + b`（composite：分配 + `.out`） | 62.1 µs | 1.4 分配 + 48 out + composite |
| `a * 2.0` | 67.8 µs | 同上 |
| `a * small`（广播） | 150.8 µs | `expand().contiguous()` 物化 |

**结论**：`.out` 一旦让位，`a + b` 应该从 62 掉到 ~15–20 µs（分配 1.4 + out 12 + composite），
这是 decode 里残差加（每层 2 次、每步 4 次）的直接收益。前提是 C++ 先补广播物化——
否则 150 µs 的广播形态会变成「直接报错」。

**另外两条已记档的事实**（torch28 的 `probes/125`、`127`）：

* `view.add_(2.5)` 的 0 维张量在 **CPU** 上，读 host 即可、**零 D2H**——纠正了我「0 维要付 D2H」的估计；
* 「带分配的形式」比「原地形式」贵 20 倍（62 vs 3.1 µs），差的不是内核而是 `aten::<op>.Tensor`
  的 composite 路径（会分配 + 调到 `.out`）。所以**能让模型用原地就用原地**，这条对 coffin 集成有用。

## 36. 逐元素算子全部交给 C++：decode 17.1 → **4.9 ms/步**（`probes/026`、`022`）

C++ 补上广播物化并声明 `elementwise_supports_broadcast() == True` 之后，闸门自动打开，
`.out` 也归 C++（`probes/026` 的 dispatch 表：`aten::add.out` 从 `/dev/null:639` 变成
`corex_shim.cpp:2125`）。

| 形态 | 让位前 | 现在 | 倍数 |
| --- | --- | --- | --- |
| `aten::add.out` 直调 | 48.0 µs | **5.2 µs** | 9.2x |
| `a + b`（composite：分配 + `.out`） | 62.1 µs | **12.5 µs** | 5.0x |
| `a * 2.0`（标量，会分配） | 67.8 µs | **14.0 µs** | 4.8x |
| `a * small`（**广播**，会分配） | 150.8 µs | **15.4 µs** | 9.8x |
| `a.mul_(2.0)`（原地标量） | 3.1 µs | 3.1 µs | — |

端到端：**`probes/022` decode 9.2 → 4.9 ms/步**，`probes/018` 整段前向不变
（全局相对 6.631e-06、argmax 1.000）。全量回归同批通过：`009` 9/9、`017` 12/12、
`021` 13/13、`022` PASS、`018` PASS。

广播物化的三条路径（torch28，全走池分配器，不 `cudaMalloc`）——**倍增复制是个好手法**：

| 情形 | 做法 | p50 |
| --- | --- | --- |
| 单广播维 `[4,16,64] *= [4,1,64]` | 每个广播下标一条 `memcpy2D`（行间用外层真实间距） | 9.50 µs |
| 多广播维且在前导 `[4,16,64] *= [64]` | 写首块后**倍增复制**（调用数 → `ceil(log2(repeats))`） | 6.01 µs（逐块循环是 **512 µs**） |
| `out` 是步长视图（RoPE 的 `out[..., 0::2]`） | 算进连续临时张量再按 stride 写回 | 同上 |

**一条反面教材（torch28 自己记的，值得留）**：`.out` 第一版只支持「同形+连续」就让位，
把 decode 打崩了——`apply_rope` 里 `out`、`x1`、`x2` **三个都是视图**。
判据不是「我实现了哪个情形」，而是**「真实调用会不会踩到没实现的那个」**。

关于门槛：`fp32 add_.Scalar` 实测 12.07 µs、门槛写的是 ≤12（fp16/bf16 是 11.6/11.7，
数值都逐字节正确）。这是噪声边界，**门槛放宽到 13 µs**——卡上 ~4% 迭代有 ≥100 µs 的驱动抢占停顿，
0.07 µs 的超出没有信息量。

## 37. 4.9 ms/步 的构成：已经不在后端，在**模型侧的算子个数**（`probes/024` 复跑）

切完 C++ 之后重跑 profile（4 步，cProfile 有开销，墙钟 6.25 ms/步 vs 不打点 4.9 ms/步）：

| 项 | 4 步累计 | 折算 |
| --- | --- | --- |
| **`apply_rope`（测试脚本自己写的）** | 0.011 s | **≈2.75 ms/步** |
| `_copy_impl` | 0.002 s | 38 次/4 步 |
| `_empty_impl` / `as_strided` / `empty_strided` | 各 0.001–0.002 s | ≈48 次/步（每个算子的输出分配） |
| `_triton.launch` | 0.004 s | 14 次/步（`silu`/`rms_norm`/`softmax`/`index_select` 仍是 Python 实现） |
| `_cat_out` | 0.001 s | 3 次/步（KV cache） |

**结论**：后端侧热点已经收干净，剩下的 4.9 ms 里最大的一块是 **RoPE 的写法**——
`out[..., 0::2] = x1*c - x2*s` 一次调用里有 4 次广播乘、2 次减、2 次步长视图写。
这不是后端能优化的：它属于「模型侧算子个数/融合」。

两条下一步（都属 coffin 集成阶段）：
1. **RoPE 用一次 kernel 算完**（一次发射替代「4 乘 + 2 减 + 2 切片写」）；
2. **残差加与归一化融合**（`x + f(rms_norm(x))` 本来就是两趟内存往返）。

同时后端侧还能再挪三个高频算子进 C++（`silu` 126 µs、`rms_norm` 83 µs、`softmax` 82 µs），
它们每步共约 14 次发射。按 `add` 的经验（78 → 5.2 µs）值得做，但**量级上限只有 ~1 ms/步**，
不如上面两条模型侧融合的空间大。

## 38. 设备层收口：`DEVICE_VERDICT PASS`（10/10），「还缺的算子：(无)」（`probes/008`）

最后一项是「非连续 strides（`from_blob` 造一个转置形状）」读回主机。它连着三轮报错都不一样，
每轮都指向不同的层，最后一条才是真因：

| 轮次 | 报错 | 真因 |
| --- | --- | --- |
| 1 | `_copy_from 暂不支持非连续的目的张量` | 我只做了「非连续源」，没做「非连续目的」 |
| 2 | `copy_ 只接 corex 张量，收到 cpu` | 我把主机暂存交给 `_aten_extra._copy_impl`（它只做设备侧） |
| 3 | 又回到第 1 条的旧文案 | **远端 `__pycache__` 是旧的**：tar 推上去的 `.py` 保留了 Windows 的 mtime，比已有 `.pyc` 更旧，CPython 认为缓存仍然有效 |

修法三条：① 主机侧非连续目的 → 线性 D2H 到连续暂存 + **CPU 原生 `copy_`** 写进带 stride 的目的；
② 卡侧非连续源 → 直接用 `.contiguous()`（走「设备→设备 + 非连续源」那条已验证的路），不要自己造暂存；
③ `tools/remote.py pushdir` 推完**自动清 `__pycache__`**（写进工具里，不靠人记得）。

第 3 条是有代价的教训：三轮排查里有两轮是在看**已经不存在的代码**。缓存失效这类问题会伪装成
「改错了」，而且越是「明明改了却还是旧行为」越容易让人怀疑自己的修改本身。

## 39. stream / 事件 的 torch 惯例 API 补齐（`probes/028`，6/6 PASS）

目标里要「分配/拷贝/stream/事件」，但此前只验过**运行时**那一层（`_runtime.Stream`/`Event`，
`probes/002`/`003`/`007`），而框架接进来时用的是 `torch.corex.*` 这层。补上之后：

```
torch.corex.Stream / Event / current_stream / stream / synchronize   全部就位
[OK] current_stream() 返回 Stream 对象
[OK] Stream() / with torch.corex.stream(s)        造 + 进上下文 + 同步
[OK] Event(enable_timing=True).record + elapsed_time = 0.023 ms（一次 256³ mm）
[OK] Event(record + synchronize) 后结果正确
[OK] 自定义流上 H2D + mul_ + 同步，4096 元素逐字节相等
[OK] 退出 with 后 current_stream 仍可用
```

两个实现细节值得记：

* **`Stream` 用「持有 + `__getattr__` 转发」而不是继承**：`_runtime.Stream.__init__` 要一个已创建的
  句柄（构造应当走 `Stream.create()`），继承的话子类得复刻创建逻辑；转发只有一行。
* **`__enter__`/`__exit__` 必须显式定义**：`with` 语句查的是**类型**上的这两个名字，
  `__getattr__` 转发对它们无效——第一次写就撞到 `AttributeError: __enter__`。

## 40. 操作纪律：推代码与排队探针之间存在竞态

`probes/008` 在「批量排队」那一轮报了 PARTIAL，单独重跑立刻 PASS（「还缺的算子：(无)」）。
原因不是代码：`tar` 解包与队列启动撞在一起，探针可能读到**正在被覆盖的目录**（`src/corex_torch`
的 mtime 正好是那一分钟）。同一条错误信息在仓库里已经不存在了，这个矛盾正是线索。

三条纪律（都写进 `tools/remote.py` 或流程）：

1. **每次 `pushdir` 之后自动清 `__pycache__`**（已实现）——tar 保留的是 Windows 的 mtime，
   比远端已有 `.pyc` 更旧，CPython 会继续用旧字节码。这条单独就造成过两轮「改了没生效」。
2. **推完 `src` 再排队探针**，不要在同一条命令里连着做——两次操作对远端目录的写入有先后，
   队列可能在解包中途启动。
3. **看到「报错文案在源码里已经不存在」时，先怀疑加载了旧副本**（`__pycache__`、别的 checkout、
   半覆盖的目录），不要怀疑自己的修改。三次排查里有三次是这个原因。

顺带确认：`$W/ops/corex-torch/src` 是**另一份 checkout**（torch28 的工作目录），它的 `_device.py`
确实还留着旧文案——所以「哪份在生效」必须靠打印 `corex_torch.__file__` 来确认（`probes/029`）。
