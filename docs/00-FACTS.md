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

## 16. 每次发射的 host 开销归因（`probes/010_overhead.sh`）

口径：mm 256³、指针现成、内核已编译、每段循环外只同步一次、单卡（沙箱 4935 逻辑 0 号）。

| 段 | 单次耗时 |
| --- | ---: |
| 直接调 `_ops_matmul.mm`（内核 + 启动） | **206.1 µs**（首次 40.0 µs） |
| mm 16³ 直接调（内核时间≈0，即**纯启动地板**） | **39.6 µs** |
| `shim.empty_strided`（分配） | **1.76 µs** |
| aten `a @ b`（分发 + 分配 + 内核） | **265.8 µs** |
| 归因 | 内核 ≈166.5 µs；Python 启动 ≈39.6 µs；aten 再加 ≈59.7 µs |

**含义**：每次算子调用的纯 host 开销约 **100 µs**（40 启动 + 60 aten），与内核本身同量级。
coffin 的 decode 每 token 六百多次发射（`../../docs/corex_drive/GRAPH-REPLAY.md` §17.1），
按这个数就是每步几十毫秒的白费。压缩的唯一现实路径是**把参数打包与 `cuLaunchKernel` 挪进 C++ shim**，
并用 `TORCH_LIBRARY_IMPL` 把最热的算子（`aten::mm`、`aten::_softmax.out`）从 Python 分发里挪出去。
分配（1.76 µs）不是问题——缓存池已经把它压到位了。

## 17. bf16：厂商用的是 intrinsic，不是位运算（`probes/011`–`015`）

这一条推翻了我最初的设计（「用整数位运算自己造转换」）。**先看厂商怎么做，比自己发明省事且更准。**

| 事实 | 证据 |
| --- | --- |
| 厂商 clang 定义的宏：**`__CUDA_ARCH__ = 300`、`__ILUVATAR__ = 1`** | `probes/011` §2：`clang++ --cuda-device-only -dM -E` |
| 于是 `c10/util/BFloat16-inl.h` 里 `(__CUDA_ARCH__ >= 800 || defined(__ILUVATAR__))` 为真，bf16 转换走 `__float2bfloat16` / `__bfloat162float`（不是 `round_to_nearest_even` 那条纯位运算 fallback） | 同上，读 `torch/include/c10/util/BFloat16-inl.h:26-56` |
| 这两个函数在 `cuda_bf16.hpp`（427 / 543 行）里落到 **两条 LLVM intrinsic**：`llvm.bi.bfloat2float(bfloat) -> float`、`llvm.bi.float2bfloat(float) -> bfloat` | `probes/012` §1 |
| 厂商 clang 编出来的设备 IR 里 **`fpext`/`fptrunc` 命中 0 次**，只有上面两条 intrinsic，llc 编得出来 | `probes/012` §2 |
| **ATen 的 bf16 算术本质是 fp32 算术**：`operator+(BFloat16 a, b) { return static_cast<float>(a) + static_cast<float>(b); }` | `BFloat16-inl.h:110` |
| `llvm.bi.*` **不支持向量重载**：`call <8 x bfloat> @llvm.bi.float2bfloat(<8 x float>)` 报 `input module cannot be verified` | `probes/015` V2 / W1 |
| 向量转换的正确形态是**整数位运算**：展宽 `bitcast→zext→shl 16→bitcast`、收窄 `bitcast→+0x7FFF+lsb→lshr 16→trunc→bitcast`，两者 llc 都过 | `probes/015` V3/V4/W2 全部 `llc OK` |
| 真实内核里矩阵乘的失败点就是一条向量指令：`%n = fptrunc <8 x float> %x to <8 x bfloat>` | `probes/014`（从真编译的 biir 里抓出来的唯一一条） |
| 改写实现后 **bf16 矩阵乘编得过且结果逐字节相等（4096/4096）**，改写命中 17 处 | `probes/013`：`BF16_INTRINSIC_VERDICT PASS` |

实现落在 `src/corex_torch/_bf16.py`：标量 → intrinsic（与厂商语义完全一致）、向量 → 位运算
（intrinsic 没有向量重载）。挂在 `_triton.activate()` 注册的后端子类 `CorexTorchBackend.make_cubin`
上，**corex-triton 一个字节没改**。

两条必须记住的坑：

* **新生成的 SSA 名字不能拿原值名加后缀**：Triton 的值是编号值 `%257`，而 `%257.i` 不是合法标识符，llc 报
  `expected '=' after instruction id`。命名必须以字母开头（见 `_bf16._NameGen`）。
* **向量常量不能写成 `splat (i32 16)`**：那是 LLVM 20 语法，厂商 llc 是 16（triton-bridge 在 `probes/190` 踩过）。
  必须展开成 `<i32 16, i32 16, …>`。

## 18. torch 层算子验收转全绿（`probes/009`）

`TORCH_OPS_VERDICT PASS`，9/9。新增的两条修复：

* `div.out`/`mul.out` 的**标量右操作数**（`scores / sqrt(H)` 走的这条）之前直接报未实现，现在走内核的
  `*_scalar` 入口，`alpha` 先乘进标量。
* 多算子串联（两次矩阵乘 + softmax + 再一次矩阵乘）的判据换成**全局相对误差** `max|diff|/max|ref|`：
  逐元素相对误差在「参照值接近 0」的元素上会被放大到 1e-4 量级，看着像错其实是 fp32 累加的正常结果。
  换口径后注意力小块是 **1.62e-07**。
