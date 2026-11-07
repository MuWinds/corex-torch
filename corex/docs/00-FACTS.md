# 已确证的环境事实（corex-torch）

这份文档回答：**torch 2.8 要在这张卡上跑起来，哪些事已经被证据钉死了——每条事实、产生它的命令、以及观察到的输出是什么。**

节号**连续、不重号、不复用**：改结论就改那一节并保留证据，不要新增同号节。
探针在 `probes/`，编号取当前最大值 +1；`UNKNOWN` 一律写进 §23。

采集环境是 **SCOW 的 `cx-sandbox` 队列**，节点与卡号都是**会话级**的，不写进结论：
卡的好坏一律运行期逐卡测（8×8 bf16 matmul + 5 s 硬超时，见 [`../AGENTS.md`](../AGENTS.md) §3）。
性能数字一律 **p50 + 迭代 ≥500 + 两条基线**（纯 Python 空循环、最小的真内核）——
这张卡是多租户共享的，约 4% 迭代会吃到 ≥100 µs 的驱动抢占停顿，均值会被它毁掉。

---

## 1. 软件栈：厂商运行时有什么

| 事实 | 命令 | 输出 |
| --- | --- | --- |
| CoreX 3.2.1，`/usr/local/corex` 指向 `/usr/local/corex-3.2.1` | `ls -ld /usr/local/corex` | 软链一条 |
| 兼容层自称 **CUDA 10.2.89**：`libcudart.so.10.2 -> .10.2.89`，`CUDART_VERSION 10020`、`CUDA_VERSION 10020` | `readlink -f $C/lib64/libcudart.so`；`grep -hn "define CUDA_VERSION" $C/include/cuda.h` | 与括号内一致 |
| 有 **runtime API**（`libcudart.so.10.2`）与 **driver API**（`libcuda.so.1`）两套，都有 | 同上 | 两个文件都在 |
| 数学库：`libcublas.so.10.2.3.254`、`libcublasLt.so.10.2.3.254`、`libcusolver.so.10.3.0.89`、`libcusparse.so`、`libcurand.so`、`libcufft.so.10.1.2.89`、`libcudnn.so.7`；另有 `libnccl.so.2.11.4` | `readlink -f $C/lib64/libcublas.so` 等 | 版本如左 |
| 编译器：`clang 16.0.6`（厂商分支）、`llc`、`lld`/`ld.lld`、`fatbinary`、`nvcc`（壳） | `$C/bin/clang++ --version` | `16.0.6` |
| 设备节点 8 个 `/dev/iluvatar0..7`，内核模块 `iluvatar` | `ls /dev \| grep iluvatar`、`lsmod` | 8 个设备节点 |
| 主机工具链：gcc/g++ **10.3.1**、cmake 3.22.0、ninja | `gcc --version`、`cmake --version` | 版本符合 |

## 2. 登录节点与计算节点的分工

| 事实 | 命令 | 输出 |
| --- | --- | --- |
| 登录节点 `mgt` 上**没有 `/usr/local/corex`** | `python -m scow run 'ls -d /usr/local/corex*'` | 空 |
| 共享 py310 环境在登录节点 `import torch` 失败（把 corex 的 lib64 加进 `LD_LIBRARY_PATH` 也一样，目录不存在） | `py310/bin/python -c "import torch"` | `OSError: libcufft.so.10: cannot open shared object file` |
| 所以：pip / venv / 读源码 / 编 CPU 扩展放登录节点；任何 `torch.cuda`、CoreX 头文件、碰卡的事放沙箱 | 上面两条的直接推论 | — |
| 计算节点**没有外网**（pip 在 gpu-node 上一直重试然后失败） | 在计算节点 `pip install` | 超时 |
| 登录节点被别人压满时编译极慢（实测 10 分钟只编 4 个文件） | 计时 | 编译放计算节点，`$HOME` 共享不用搬数据 |

## 3. 厂商 torch 2.1（数值基线，不参与适配）

| 事实 | 命令 | 输出 |
| --- | --- | --- |
| 共享 py310 里是 **`torch-2.1.0+corex.3.2.1`**（厂商按 CoreX 版本重发的 wheel） | `ls $PY310/lib/python3.10/site-packages/ \| grep -i torch` | 如左 |
| 它带 `libtorch_cuda.so`、`libtorch_cpu.so`、`libnvfuser_codegen.so`（有 nvFuser） | `ls torch/lib/` | 三个 .so |
| 同环境还有一整套 `nvidia_*_cu12-12.4.x` 的 pip 包 | `grep -i nvidia` | 一组包 |
| 它是 **`USE_CUDA=1` 编出来的真 CUDA 后端**，底层 CUDA 兼容层来自 CoreX；`torch/version.py` 里 `cuda='10.2'` | 静态读 wheel | 如左 |
| 厂商对 `torch/cuda/__init__.py` 的唯一补丁是 `is_bf16_supported()` 里一句 `if torch.corex: return True` | 静态 diff | 一行 |

## 4. torch 2.8 的安装现状

| 事实 | 命令 | 输出 |
| --- | --- | --- |
| 隔离 venv：`~/venvs/corex28`（python3.10，**不带** `--system-site-packages`，与厂商 torch 2.1 完全隔离） | `cat ~/venvs/corex28/pyvenv.cfg` | `include-system-site-packages = false` |
| 装 `2.8.0+cpu` 失败过一次：venv 自带 pip 23.0.1 把 `typing-extensions` / `typing_extensions` 判成两个名字，退化成源码构建后卡在 `flit_core` | `cat ~/venvs/corex28-pip.log` | `expected 'typing-extensions', but metadata has 'typing_extensions'` |
| pip 自带的下载器在登录节点下 184 MB 的 wheel 只有约 52 kB/s；同一 URL 用 `curl` 是 **7.1 MB/s** | 计时 | 差两个数量级 |
| 最终装法：升 pip → 先装纯 Python 依赖 → `curl` 下 wheel → `pip install --no-deps --no-index` | `probes/100_torch28_install.sh` | `TORCH28_VERDICT PASS` |
| wheel sha256 与 index 页一致 | `sha256sum` vs index | `16d75fa4…e2077e` |
| 版本证据：`2.8.0+cpu`、`torch.version.cuda = None`、`cuda.is_available() = False`、backend 名 `privateuseone` | `probes/100` §版本段 | 如左；venv 占 872 MB，`pip list` 里没有 triton |

## 5. 设备：名字、容量、属性

| 事实 | 命令 | 输出 |
| --- | --- | --- |
| 自报版本 `cudaRuntimeGetVersion` / `cudaDriverGetVersion` 都是 **10020** | `probes/002_cudart_binding.sh` §1 | `10020` |
| 设备名 **`Iluvatar BI-V100`** | `cudaGetDeviceProperties` | 如左 |
| `warpSize = 64`、`maxThreadsPerBlock = 4096`、**`multiProcessorCount = 16`**、`sharedMemPerBlock = 131072`（128 KB） | `cudaDeviceGetAttribute(10/1/16/8)` | 如左 |
| 显存**标称 32 GB**，`cudaMemGetInfo` 报 **31.72 GiB 可用**（空载时 free 与 total 相同） | 同上 | `31.72 GiB` |
| `cudaMalloc(1 MiB)` → H2D → D2H **往返逐字节相等** | `probes/002` §3 | `ROUNDTRIP_BYTES_EQUAL = True` |
| 设备指针形如 `0x304000002000000`（高位非规范地址，**不要**假设它能当 CPU 地址解引用） | 同上 §3 | 如左 |
| `cudaStreamCreate` / `cudaMemcpyAsync` / `cudaStreamSynchronize` / `cudaEventCreate` / `cudaEventRecord` / `cudaEventSynchronize` / `cudaEventElapsedTime` 全部 rc=0 | 同上 §4 | 全 0 |
| `cudaGetErrorString(0)` = `no error`、`(1)` = `invalid argument` | 同上 §5 | 如左 |

## 6. 设备层可以只用 ctypes，不需要 torch

**这是整条路线最关键的一条**：用 ctypes 打开 `libcudart.so.10.2`，不 import 任何 torch 版本，
就能把一个 context 用起来。所以不需要 `cuInit`/`cuCtxCreate`——runtime 会自己延迟建 primary context。

| 事实 | 命令 | 输出 |
| --- | --- | --- |
| 23 个真符号全部可解析（唯一 `missing` 是探针自己写错的假名字） | `probes/002` §符号段 | `missing symbols` 只有假名字 |
| `cudaGetDeviceCount` / `cudaSetDevice(0)` / `cudaGetDevice` 全部 rc=0 | 同上 §2 | 全 0 |
| 16 个事件同时挂起全部成功 | `probes/003_runtime_module.sh` §4 | 全 0 |
| `cudaMalloc` 约 **3–12 µs/次**、`cudaFree` 约 **2 µs/次**（1 KiB / 1 MiB / 64 MiB 三档） | 同上 §4 | 如左 |
| `record + record + sync` 一轮约 **29 µs**；空载 `cudaDeviceSynchronize` 约 **1.1 µs** | 同上 §4/§5 | 如左 |

含义：设备层的分配/拷贝/流/事件可以直接绑 runtime API，**不需要 C++ 也能跑**；
C++ 只在「PrivateUse1 要求从 C++ 注册分配器」时出场（§10）。

## 7. 拷贝路径与带宽（三档口径）

口径：单卡、`cudaMemcpyAsync` + 自建流、尺寸见下表（`probes/007_copy_bandwidth.sh`）。

| 路径 | 尺寸 | 平均耗时 | 带宽 |
| --- | --- | ---: | ---: |
| D2D（走 mem_handle，不 pin 页） | 1 MiB | 0.014 ms | 145.0 GB/s |
| D2D | 16 MiB | 0.052 ms | **649.3 GB/s** |
| D2D | 64 MiB | 0.236 ms | 568.5 GB/s |
| D2D | 256 MiB | 0.901 ms | 595.6 GB/s |
| H2D **pageable** | 1 / 16 / 64 MiB | 0.260 / 3.748 / 14.533 ms | **4.0 / 4.5 / 4.6 GB/s** |
| H2D **pinned** | 1 / 16 / 64 MiB | 0.051 / 0.615 / 2.424 ms | **20.5 / 27.3 / 27.7 GB/s** |

**三档带宽口径（引用时只准用这三档，别自己造第四个数）**：

| 档 | 数值 | 出处 |
| --- | --- | --- |
| D2D（驱动 DMA 引擎，`mem_handle`，不 pin 页） | **596–649 GB/s** | 上表 16/64/256 MiB |
| SM 拷贝（kernel 里宽访存，≥4 元素/线程） | **553–607 GB/s** | corex-triton `00-FACTS` §22 |
| 厂商口径（平台峰值） | **575 GB/s** | `probes/035_mem_bw.sh` |

| 事实 | 命令 | 输出 |
| --- | --- | --- |
| 小拷贝的固定开销（H2D）：1 KiB 约 7.3 µs、64 KiB 约 9.5 µs；1 MiB pinned 45 µs vs pageable 142 µs | `probes/007` | 如左 |
| 流同步 / 设备同步各约 1.38 / 1.36 µs（都很便宜，「每步插几次同步」的预算很宽） | `probes/007` | 如左 |
| **pageable 主机内存只有 pinned 的约 1/6 带宽**（64 MiB 时 4.6 vs 27.7 GB/s），首次数 38.2 ms | `probes/007` | 印证驱动审计：host 路径每次 `vmalloc` + `get_user_pages_fast` |

**结论**：`copy_` / `_to_copy` 一律走 `cudaMemcpyAsync`；凡是要过主机内存的路径必须走 pinned 暂存区，
否则直接丢掉 5/6 的带宽。

### 7.1 Python/ctypes 的一个坑：取地址一律用 `.value`

`int(ctypes.c_void_p(...))` **不返回地址**：CPython 给 `c_void_p` 实现的 `__int__` 走
`int(bytes(obj))`，即把 8 字节地址当十进制字符串解析。地址字节恰好全是 ASCII 数字时它「能用」，
否则抛：

```python
>>> int(ctypes.c_void_p(0x0304000002000000))
ValueError: invalid literal for int() with base 10: b'\x00\x00\x00\x02\x00\x00\x04\x03'
```

本仓库把这条规则收在 `_runtime._ptr()` 与 `_triton._device_addr()` 两处。
（`_triton` 第一版就是这样炸的，报错信息里连指针都没有，很难往「取地址」上想。）

## 8. 厂商 torch 的算子成绩来自四种机制，不能当 Triton 的基线

| 算子 | 实际执行的 kernel | 谁实现的 | 成绩 |
| --- | --- | --- | --- |
| `copy_` | `Memcpy DtoD (Device -> Device)` | **驱动的 DMA 引擎**，连 kernel 都不是 | 51.0 µs / 471 GB/s |
| `add` | `at::native::modern::elementwise_kernel<CUDAFunctor_add<float>, …>` | ATen 手写 CUDA kernel（向量化） | 100.4 µs / 402 GB/s |
| `matmul` | `Gemm_tcu_bi_kernel::gemm_tcu2h_trans<…>` | 厂商手写 GEMM，**`tcu2h` 就是矩阵单元** | 17.8 µs / 1577 GFLOP/s |
| `softmax` | `softmax_warp_forward<float, float, float, 10, false, false>` | ATen 手写 warp 级 kernel | 15.0 µs（64×1024） |

kernel 名可在 `corex_drive_re/perf/pulled/` 的 `ixprof` 原始输出里交叉核对。
**含义（决定每个算子走什么机制）**：

| 我们的选择 | 依据 |
| --- | --- |
| 拷贝必须走 `cudaMemcpyAsync`（DMA），不写 Triton 内核 | DMA 471 GB/s 不占 SM；Triton 逐元素内核只有 295 GB/s |
| 逐元素按 ATen 的 402 GB/s 记天花板，Triton 版当前 295.9 GB/s（约 0.73x） | 差距来自向量化宽度，不是语义 |
| 矩阵乘在对上矩阵单元之前只能按 FMA 算（104.6–190.9 GFLOP/s） | 1577 GFLOP/s 是矩阵单元的成绩 |
| softmax 这类 warp 级规约按 Triton 自己的数字量，不与 ATen 混比 | 尺寸不同不可直接比 |

所以算子表必须写两条：**「我们的数 / CPU 参照（正确性）」**与**「我们的数 / ATen 同类实现（性能）」**，
并注明后者用的机制；不能笼统写「对厂商 torch 的比值」。

## 9. PrivateUse1 的最小必需面

| 事实 | 命令 | 输出 |
| --- | --- | --- |
| `torch.empty(device="corex")` 取 `c10::GetAllocator(PrivateUse1)`，**默认返回 nullptr 且不报错** | `probes/102 --stage alloc` | 注册前 `OK -> 0`，注册后 `OK -> 107123696` |
| Python 侧**没有任何**设置 PrivateUse1 分配器的入口（`dir(torch._C)` 里 privateuse 开头的只有两个名字） | `probes/108` | 两个名字 |
| 算子可以用纯 Python 注册（`torch.library.Library("aten","IMPL").impl` 成功），但**没法把自造指针包成 Tensor**——那个 ABI 只有 C++ 有 | `probes/101`、`102` | 如左 |
| `DeviceGuardImplInterface` 必须实现 8 个纯虚：`type/exchangeDevice/getDevice/setDevice/uncheckedSetDevice/getStream/exchangeStream/deviceCount` | `grep -n "= 0;" c10/core/impl/DeviceGuardImplInterface.h` | 8 行 |
| 缺 guard impl 的报错 | `probes/101` stage 3 | `PyTorch is not linked with support for corex devices` |
| `c10::Allocator` 在 2.8 有两个签名坑：`allocate` **非 const**、且新增纯虚 `copy_data` | 照 2.1 写会编不过 | `marked 'override', but does not override` / `invalid new-expression of abstract class type` |
| `guard impl` 宏真名 `C10_REGISTER_GUARD_IMPL`；`c10::SetAllocator` 从 `libc10.so` 导出 | `nm -DC` | 两个符号都在 |
| `torch.<dev>` 这个 device module 必须**可导入**（C++ lazy init 会 `import torch.corex`）；`torch.__path__.append(我们的目录)` 实测可行 | `probes/106`、`107` | `import torch.corex OK` |
| device module 实际会被摸的函数：`is_available`、`device_count`、`_lazy_init`（**文档没列但实测会调**）、`_is_in_bad_fork`/`manual_seed_all`、`current_device` | `probes/106`（记录属性访问的替身模块） | 如左；`torch.empty/zeros/tensor/randn(device=…)` 一个都不摸 |
| `rename("corex")` 之后 `torch.cuda.is_available()` 仍为 **False**、`device_count()` 仍为 0；但 `torch.accelerator.current_accelerator()` 立刻变成 `corex` | `probes/108` | 如左 |
| 三级失败梯度：只 rename → `ModuleNotFoundError: No module named 'torch.corex'`；加 device module → `NotImplementedError: Could not run 'aten::empty.memory_format' …`；再加 C++ 分配器/guard/empty → 成功 | `probes/102` | 三段原文 |

## 10. 最小可运行链路

| 事实 | 命令 | 输出 |
| --- | --- | --- |
| 用 g++ 10.3.1 + 上游 CPU wheel 的 headers 编一个最小扩展，**能编过（26.4 s）** | `probes/102 --stage build` | `BUILD_MIN_OK triple(14)=42` |
| 真 shim（分配器 + guard impl + 手搓张量 + Python 注册 `empty.memory_format`）编得过（26.7 s），产物 248 384 B | `probes/110_shim_build.sh` | `BUILD_SHIM_OK` |
| 编译期**不需要** CUDA 头、不链 `libcudart`（全部 `dlopen`/`dlsym`） | `ldd _corex_shim.so` | 只有 `libc10/libtorch_cpu/libtorch/libtorch_python/libdl` |
| 手搓张量三行（上游没发 `ATen/native/EmptyTensor.h`，所以不依赖 `empty_generic`） | `probes/102` §4.3 | `make_tensor<c10::TensorImpl>` + `set_sizes_contiguous` |
| `at::detail::empty_generic` 有符号但**头文件没发** | `nm -DC libtorch_cpu.so \| grep empty_generic` | 一个 `T` 符号；`ls include/ATen/native/EmptyTensor.h` 不存在 |
| 配好之后 `torch.empty(2,3,device="corex")` 成功，`device=corex:0`、`data_ptr` 是设备地址 | `probes/102 --stage shim` | 如左 |

## 11. shim 的接口、缓存池与多设备

统一导出的接口（**个数以源码 `m.def(` 计数为准**，当前 22 处；
`grep -c 'm.def(' corex/src/csrc/corex_shim.cpp`）：

```
init  is_initialized  device_count  empty_strided  from_blob  as_strided
reserved_bytes  allocated_bytes  empty_cache  slab_alignment
load_kernel  launch  last_error  kernel_count
set_mm_kernel  mm_kernel_dtypes  mm_kernel_info  load_fast_op
elementwise_kernel_info  elementwise_supports_broadcast  elementwise_op_names  memcpy2d
```

| 事实 | 命令 | 输出 |
| --- | --- | --- |
| Python 侧拿到的就是 `from corex_torch import _corex_shim as shim`（**必须先 `import torch`**，否则 `libc10.so` 找不到） | `probes/111_shim_smoke.sh` | `SHIM_SMOKE_PASS` |
| `empty_strided` 的 strides **原样**写进 TensorImpl；`(1,2)` 这种非连续布局读回正确 | `probes/111` | `STRIDE_PASSTHROUGH_OK` |
| `from_blob` 不接管所有权（包装张量析构后仍可 `rt.free()`） | `probes/111` | `FROM_BLOB_NO_OWNERSHIP_OK` |
| 缓存池收敛：200 个张量活着 → `reserved=18874368 / allocated=4440960`；全释放后 `allocated=0` 而 `reserved` 留着；`empty_cache` 后两者归零 | `probes/111` | `SHIM_ALLOC_PASS` |
| `slab_alignment()` 观测到 **16 MiB / 32 MiB**（两次运行不同） | `probes/111` | 远高于 2 MiB 页粒度 |
| 边界：空张量 `(0,3)` 账本不变；0 维标量记账 4 B；4 MiB 走直通 | `probes/112_shim_edges.sh` | `SHIM_EDGES_PASS` |
| **多设备硬规则**：`cudaMemcpy` 在「当前 runtime 设备」与指针所属设备不一致时**不报错、rc=0，静默别名**（1 号卡的指针在 0 号卡上下文里被当成同一偏移） | `probes/112` | 如左——每次拷贝前必须 `setDevice` |
| `init` 幂等：重复 `init(0)` 不重复注册、账本不变 | `probes/112` | 如左 |

## 12. 设备端到端打通

`probes/008_device_end_to_end.sh`：`DEVICE_VERDICT PASS`（10/10），收尾 `CARD_STILL_HEALTHY`。
**上游 torch 2.8 在这张卡上有了一张能用的设备**：分配、拷贝、逐元素算子、标量回读全部走通。

```
[OK] torch.zeros(4, device='corex')            corex:0 float32 Size([4]) data_ptr=0x304000002000000
[OK] zeros 清零正确（.cpu() 读回）              [OK] torch.ones(8, device='corex')
[OK] CPU -> corex 往返逐字节相等                [OK] corex 上的 a + b
[OK] .item()                                   [OK] 非连续 strides（from_blob）
[OK] 大张量分配 + 填 + 读回（4 MiB）            [OK] empty_cache
```

三个静默算错型的坑（都不是平台问题，但每一个都会让结果全错而 rc=0）：

| 坑 | 现象 | 根因与修法 |
| --- | --- | --- |
| `aten::_copy_from` 的**参数顺序** | `t.cpu()` 读回一堆垃圾且 rc=0 | schema 是 `_copy_from(self, dst, non_blocking)`——第一个实参是**源** |
| `torch.dtype` 不能 `int()` | `TypeError: int() argument must be … not 'torch.dtype'` | C++ 要 `at::ScalarType`；`_device._scalar_type()` 维护枚举表并用 `element_size` 交叉校验 |
| 算子层找不到 `triton_corex` | `No module named 'triton_corex'`，只在第一次编内核时暴露 | 运行前给 `COREX_TRITON_SRC`；`_triton.activate()` 按仓库同级目录兜底 |

另外两条环境前提：`~/venvs/corex28` 里**必须装 triton 3.4**（否则算子层的 `tl` 是 `None`）；
`_corex_shim.so` 的 import 名是 **`_corex_shim`**（`PyInit_` 符号由文件名决定）。

## 13. `_triton` 只是薄适配

`corex-triton` 已经把驱动层的 torch 依赖去掉（主 context 走 `cuDevicePrimaryCtxRetain`、
stream 用 legacy 默认流、显存 `cuMemAlloc_v2`、计时自己 `cuEvent`），所以本仓库不再实现运行期。
`_triton` 只留四件事：定位源码目录、`activate()`、`compile_kernel`、`launch`
（补网格维度 + 把 `c_void_p` 换成 int）。

`probes/006_triton_adapter.sh`：`ADAPTER_VERDICT PASS`，编译 424 ms、cubin 3464 B、
n=4096 与 n=4000（尾块掩码）都 `max_abs_diff=0.000e+00`、全程 `torch_modules=0`。
`CompiledKernel.__getitem__` 的 runner **不补齐网格维度**（只传 `(16,)` 会 IndexError），
所以 `launch()` 负责补成 3 元组。

## 14. host 开销：发射已在 C++，分发器几乎免费

统计口径一律 p50 + 迭代 ≥500 + 两条基线（空循环 0.2 µs、最小真内核 `fill_` 32.7 µs）。

| 段 | p50 |
| --- | ---: |
| `aten::add.out` 直调（C++ 实现） | **5.2 µs** |
| `add_.Tensor`（C++ 实现） | **1.96 µs** |
| `mm [1,64]×[64,64]`（C++ 实现） | **3.1 µs** |
| `a + b`（composite：分配 + `.out`） | 12.5 µs |
| `a * 标量` / `a * small`（广播） | 14.0 / 15.4 µs |
| 分配 `empty_strided` | 1.76 µs |
| Python 实现的同类算子 | 67–127 µs |

**结论**：host 侧固定成本已压到个位数微秒，aten 分发器本身几乎免费；真正贵的是「用 Python 实现算子」
（同一个 `add`：Python 78.8 µs vs C++ 5.2 µs）。压 host 开销的唯一有效路径是**把热路径算子搬进 C++**
（`TORCH_LIBRARY_IMPL` + `set_*_kernel`），不是减少 aten 调用次数。

## 15. bf16：转换走厂商 intrinsic，向量形态标量化

厂商 clang 定义 `__CUDA_ARCH__ = 300`、`__ILUVATAR__ = 1`，于是 `c10/util/BFloat16-inl.h` 里
`(__CUDA_ARCH__ >= 800 || defined(__ILUVATAR__))` 为真，转换落到两条 intrinsic：

```
%n = call float  @llvm.bi.bfloat2float(bfloat %x)
%n = call bfloat @llvm.bi.float2bfloat(float %x)
```

厂商 IR 里 `fpext`/`fptrunc` **命中 0 次**；而 Triton 发的正是这两个名字，bi 后端不认，
报 `Cannot select: f32 = fp_extend` / `bf16 = fp_round`。另外 ATen 的 bf16 算术本质是 fp32 算术
（`operator+(BFloat16 a, b) { return (float)a + (float)b; }`）。

做法（`torch/corex/_bf16.py`，挂在 `_triton.activate()` 注册的后端子类 `make_cubin` 上，
**corex-triton 一个字节没改**）：

| 形态 | 做法 |
| --- | --- |
| 标量 `fpext bfloat` / `fptrunc … to bfloat` | 换成 `llvm.bi.bfloat2float` / `llvm.bi.float2bfloat` |
| **向量** `<N x bfloat>` ↔ `<N x float>` | 展开成 `extractelement → 标量 intrinsic → insertelement` 链 |

向量为什么不能直接用向量 intrinsic：`@llvm.bi.float2bfloat.v8bf16` 这类 mangled 名字
**`llc` 收下但不实现**，目标文件里留一个外部符号，`lld --no-undefined` 才报 undefined。
所以标量化是唯一稳妥的向量通路；另留两档策略（`intrinsic`、`bits`），用 `_bf16.set_vector_strategy()` 切。

**判据硬规矩**：判断「bi 后端支不支持某个 intrinsic / 某条指令」，必须 `llc -march=bi`
**和** `lld --no-undefined` 都过——只看 `llc` 的返回码会把「parse 进去但没 lowering」误判成「支持」。

两条实现坑：新生成的 SSA 名字**不能**拿原值名加后缀（Triton 的值是编号值 `%257`，
`%257.i` 不是合法标识符）；向量常量**不能**写成 `splat (i32 16)`（LLVM 20 语法，厂商 llc 是 16）。

结果：`probes/192` 元素级 **33/33 逐字节相等**、累加类 15/15、mm/bmm/gather 5/5；
改写命中 236 处，改写后 `fpext`/`fptrunc`/`trunc nuw`/`trunc nsw`/`splat` 全 0。

## 16. 算子覆盖

| 事实 | 命令 | 输出 |
| --- | --- | --- |
| **算子全集 64/64**（用例集是 torch 语义算子表，口径以该探针输出为准） | `probes/161_ops_matrix.sh` | `64/64 OK`；`sdpa max_abs_diff 7.15e-07` |
| 更窄的**推理最小集** 39/43（剩 4 项是「非末维 `dim=0`」的归约/索引） | `probes/120_aten_coverage.sh` | `39/43`；这 43 条是推理路径的子集，不是全集降级 |
| 第 5 批（relu/rsqrt/gelu/masked_fill/log_softmax/max.dim/argmax/view.dtype）12/12 | `probes/017_torch_ops_batch5.sh` | `12/12 PASS` |
| 归约补齐：中间维 + `var`/`std`，**13/13 PASS** | `probes/021_axis_reduce.sh` | `13/13`；`var` 4.77e-07、`std` 2.38e-07 |
| 元素级：fp32 45/45、fp16 33/33、**bf16 33/33** 逐字节相等 | `probes/170`、`171`、`192` | 如左 |
| 功能矩阵 12/12；mini decoder 相对误差 6.631e-06、argmax 1.000 | `probes/177_final_verify.sh` | 如左 |
| 矩阵单元（TCU）接上后 fp16/int8 矩阵乘可用 | corex-triton `00-FACTS` §19/§23 | 见那边的 `TCU-SPEC.md` |

三条验收口径（避免后人误判）：

1. **归约类的参照要用 float64**：CPU 的 `sum`/`mean` 走 pairwise 求和，拿它当 1e-5 门槛会把
   顺序 fp32 累加的固有误差判成「实现错了」；这两项改成「对 float64 参照 ≤1e-4」。
2. **多算子串联用全局相对误差** `max|diff|/max|ref|`：逐元素相对误差在参照接近 0 的元素上会被放大。
3. **「实现正确」的判据必须是非方阵/非整除形状 + 与 CPU 逐元素比对**；吞吐只能附在正确性之后。

## 17. 静默算错类的 bug（每一类都留了判据）

| 类 | 现象 | 判据与修法 |
| --- | --- | --- |
| 广播操作数没 expand 就被当同形读 | `[T,1,HD]` 的 `other` 按 `[T,HEADS,HD]` 读，越界读、不报错、数值全错（RoPE 处 max_abs **6.4**） | `probes/019` 逐阶段比对；物化必须 `expand(shape).contiguous()` |
| 卡上 dtype 转换按字节 memcpy | `x.float()`（fp16→fp32）按目的宽度读源缓冲，越界（`probes/167` max_abs **2.1**，而 `.to("cpu").float()` 是对的） | 新增 `_cast_kernel`；只看 CPU 侧永远发现不了 |
| `sdpa` 交给 torch 自己分解 | 走到 `Autogradcorex` 层报「算子不存在」；补齐依赖后结果仍然错（max_abs **3.38**，而三步手算只差 3.6e-07） | 显式实现 `softmax(q kᵀ scale + bias) v`，掩码在 CPU 造好再搬；必须同时挂 `PrivateUse1` 与 `AutogradPrivateUse1` |
| C++ 快路径的两个参数契约 | spec 写成 `"ppppiii"`（把 global scratch 当第 4 个参数）→ 内核收到 `M == 0`，一个元素都不写；动态 shared 传 0 → 写了但值错 | spec 必须是 `"pppiiip"`（3 指针 + 3 个 i32 + 末尾 scratch）；C++ 侧校验 spec |
| `.out` 第一版只支持「同形 + 连续」就让位 | `apply_rope` 里 `out`/`x1`/`x2` 三个都是视图，decode 直接崩 | 补广播/步长视图物化；判据是「真实调用会不会踩到」，不是「我实现了哪个」 |
| 归约 / `cat` 的 stride 语义 | `[1,H,HD]` 经 `permute(1,0,2).contiguous()` 仍是 `(16,64,1)`——**size 为 1 的维会骗过连续性判定** | 改用 `squeeze(0).unsqueeze(1)`；别信「`.contiguous()` 之后一定连续」 |

**验收手法**：输出所在的池块先填哨兵 `-12345` 再释放，任何没被写到的元素都会暴露成哨兵
（五个形状含 `37×53 @ 53×29` 非整除，`max_abs = 0.000e+00`、残留哨兵 = 0）。

## 18. 性能与 decode 步时

| 项 | 数字（p50，口径见节首） |
| --- | --- |
| `add_.Tensor`（C++ 快路径，三档 dtype） | 1.82–1.96 µs |
| `aten::add.out` 让位后 | 2.23 µs |
| `a + b`（含分配） | 11.9–12.5 µs |
| `a * small`（广播，含分配） | 15.4 µs |
| `mm 64³` | 2.64 µs |
| 非连续搬运（转置 `contiguous()`） | 2000 → 91.7 µs；`cat` 117 → 26 µs；1 KiB–1 MiB 连续拷贝都约 10 µs（**调用次数驱动，不是字节**） |
| `exp` / `softmax`（只有 Python 实现） | 70.2 / 81.7 µs |
| **decode 步时（fork 自编 wheel）** | **5.1 ms/步**（`probes/222`，`VOCAB=32 DIM=64 HEADS=4 LAYERS=2`） |

**5.1 ms/步 的构成**（`probes/024` 的 profile，仍成立的部分）：后端侧热点已经收干净，
最大的一块是**测试脚本自己的 RoPE 写法**（`out[..., 0::2] = x1*c - x2*s` 一次调用就是 4 次广播乘
+ 2 次减 + 2 次步长视图写）；后端侧剩下的热点是 `silu`/`rms_norm`/`softmax`/`index_select`
四个仍是 Python 实现。再往下要动的是**模型侧融合**（RoPE 单内核、残差与归一化合并），属集成阶段，
不是后端能优化的。

绝对毫秒数按当前口径不逐条换算——那份 profile 出自更早的栈；**占比结论**（RoPE 写法 > 未搬进
C++ 的四个算子 > 分配/拷贝）在这条链上没有变。

## 19. fork 树：编 torch 本体的六个卡点

目标是「fork 自己编出来的 wheel」（纯 CPU 构建 `USE_CUDA=0`，不需要卡）：

| 卡点 | 现象 | 做法 |
| --- | --- | --- |
| CMake 版本 | `no cmake >= 3.27.0 found: [3.22.0, 3.22.0]` | `pip install "cmake>=3.27"` 到 `~/venvs/torchbuild`，**并把它的 bin 放进 PATH**（`tools/setup_helpers/cmake.py` 用 `which cmake`） |
| 子模块 | `Submodule initialization failed` | tar 包没有 `.git` → 在登录节点 `git clone --depth 1 --branch v2.8.0 --recurse-submodules`（2.3 GB） |
| NCCL | `git clone https://github.com/NVIDIA/nccl` 失败 | `USE_NCCL=0 USE_DISTRIBUTED=0` |
| NNPACK | `ModuleNotFoundError: No module named 'six'` | `USE_NNPACK=0`（顺带 `pip install six`） |
| **locale / 编码** | `UnicodeDecodeError: 0xe0 in position …`（读 `CMakeCache.txt`） | 删旧缓存 + `LC_ALL=C.UTF-8` 重配，让缓存本身就是 UTF-8 |
| 并发 | `cc1plus: fatal error: Killed signal terminated` | 内存不是瓶颈（1 TB 空闲、ulimit unlimited），是瞬时竞争；`MAX_JOBS` 从 48 降到 12 才稳 |

进展与产物：上游 v2.8.0 + 我们的 `torch/corex` + `torch/__init__.py` 接入，清缓存后重编
1568 个目标，`setup.py bdist_wheel` 产出 `torch-2.8.0a0+gitba56102-cp312-*.whl`（86 MB），
装进干净 venv 验证 `HAS_COREX_SUBMODULE True`。目标运行时是 py310（厂商 torch 2.1 与 shim 都在
py310 上），于是又用 py310 环境建 `~/venvs/torchbuild310` 重编（5881 个目标），产物
`torch-2.8.0a0+gitba56102-cp310-cp310-linux_x86_64.whl`（86.6 MB）。

## 20. `torch.cuda` 这个别名：为什么改、改在哪

厂商 torch 2.1 是 `USE_CUDA=1` 编出来的真 CUDA 后端；**这条路 2.8 走不通**：
`cmake/public/cuda.cmake:72` 有 `if(CUDA_VERSION VERSION_LESS 11.0)` 直接报错退出，
而 CoreX 兼容层声明 10.2；就算绕过，2.8 的 CUDA 后端还用到 11.x 才有的 API
（cudaMemPool、cuMemMap 多播、新版 cuBLAS 接口），会变成成片的 `#if CUDA_VERSION` 补丁。

**做法：设备名别名**。打补丁最终形态是 **9 个文件、约 150 行**：

| 文件 | 作用 |
| --- | --- |
| `c10/core/Device.cpp` | 字符串表里 `"cuda"` 与 `"corex"` **两个**都解析到 PrivateUse1（老写法继续可用） |
| `c10/core/DeviceType.cpp` | 保留名单去掉 `"cuda"`（上游会以「in-tree device name」拒绝 rename） |
| `c10/core/Device.h` | `Device::is_cuda()` 对 PrivateUse1 为真 |
| `c10/core/TensorImpl.h` | `TensorImpl::is_cuda()` 直接比枚举，**也必须改**（只改 Device.h 不够） |
| `torch/cuda/__init__.py` | 委托 `is_available/device_count/current_device/set_device/synchronize/Stream/Event/current_stream`，并把 `Tensor.cuda()` 接到 `.to("cuda")`（C++ 的 `at::Tensor::cuda()` 用枚举，绕过字符串别名） |
| `torch/backends/cuda/__init__.py` | `is_built()` 在 corex 可用时为真 |
| `torch/__init__.py` | `torch.version.cuda = "10.2"`（与厂商同口径；`is None` 会让不少库走错分支） |
| `torch/corex/*` | `DEVICE_NAME = "cuda"`；`current_stream()` 走 `_runtime.Stream.default()` |

回报：客户代码一行不改（`model.cuda()`、`device="cuda"`、`torch.cuda.synchronize()`），
底层仍是已验证的 PrivateUse1 后端。代价：`torch.cuda.amp`、`torch.cuda.CUDAGraph` 这类
「必须真有 CUDA」的功能没有实现（后者本来就在平台禁用清单里）。

验证（自编 wheel，干净 py310 venv）：

```
CUDA_AVAILABLE True   CUDA_COUNT 2   VERSION_CUDA 10.2   IS_BUILT True
X_CUDA cuda:0         IS_CUDA True   ZEROS_CUDA cuda:0   ADD_CUDA True
torch.device("corex").type == cuda   算子矩阵 64/64 OK
```

两个踩过的坑：`torch.cuda.Stream()` 曾被后面一行 `Stream = _corex_backend.Stream` 覆盖；
`torch.corex._device.current_stream` 是设备模块工厂内部的方法，模块级没有这个名字。

## 21. shim 编进 wheel：装完不用再编

wheel 是预编译产物，pip 安装期不会编译，所以 shim 必须在编 torch 的同一趟里产出并打进包：

| 位置 | 改动 |
| --- | --- |
| `torch/CMakeLists.txt` | 末尾加 `corex_shim` 目标（MODULE，`PREFIX ""`、`OUTPUT_NAME _corex_shim`），链 `torch_python` |
| 同上 | `target_compile_definitions(... TORCH_EXTENSION_NAME=_corex_shim)` ← 导出符号名靠它 |
| `setup.py` 的 `build_extensions()` | 把 `<build>/lib/_corex_shim.so` 拷到 `<build_lib>/torch/corex/_corex_shim.so` |

两个坑：**`torch/extension.h` 把 `PYBIND11_MODULE` 重定义成依赖 `TORCH_EXTENSION_NAME` 的版本**，
不定义它就导出字面量 `PyInit_TORCH_EXTENSION_NAME`，import 报
`dynamic module does not define module export function (PyInit__corex_shim)`
（查法：`nm -D --defined-only _corex_shim.so | grep PyInit`）；
**不要用 CMake 的 `install()`**——bdist_wheel 下安装前缀是 staging 根，
`install(TARGETS … DESTINATION torch/corex)` 会落到 `<prefix>/torch/torch/corex`（多一层 torch）。
另外从 Windows 打包上传时 tar 保留的 mtime 可能比节点的 `.o` 还旧，ninja 认为不用重编，
装完必须 `touch` 改动过的 C++ 文件。

验证（**全新 venv，只 pip install，不跑任何编译**，`probes/224_final_check.sh`）：

```
SHIM_FROM_WHEEL=427064      SYMBOL=PyInit__corex_shim
CUDA_AVAILABLE True   CUDA_COUNT 2   X_CUDA cuda:0   IS_CUDA True   ADD_CUDA True
STREAM Stream   VERSION_CUDA 10.2   IS_BUILT True
算子矩阵 64/64 OK     mini decoder PASS（6.858e-06、argmax 1.000）
```

## 22. 沙箱里的操作纪律（都是踩出来的）

| 纪律 | 现象 |
| --- | --- |
| **覆盖 `CUDA_VISIBLE_DEVICES` 会把逻辑卡号指到物理 0 号** | runner 已把健康卡写进 CVD，子进程里逻辑 0 号 ≠ 物理 0 号；探针里再写 `CUDA_VISIBLE_DEVICES=0` 会指到物理 0 号（可能是坏卡），症状是开头验卡就 `CARD_UNHEALTHY` |
| **同名脚本会被 `done/<name>.running.sh` 挡住** | runner 的 `mv queue/x.sh done/x.running.sh` 碰到已存在的 `.running.sh` 就不覆盖，队列里那个脚本一直躺着不被消费（`for` 循环里 `\|\| continue` 静默跳过）。**重跑请换名字** |
| **脚本最后一句是 `echo` 时 rc 恒为 0** | 探针必须把被测命令的退出码 `exit` 出去，否则日志里全是 traceback 而 `done/*.rc` 仍是 0 |
| **`__pycache__` 会让改动不生效** | tar 推上去的 `.py` 保留 Windows 的 mtime，比已有 `.pyc` 更旧，CPython 认为缓存有效；`tools/remote.py pushdir` 推完自动清（已实现） |
| **推完 `src` 再排队探针**，不要在同一条命令里连着做 | 队列可能在解包中途启动，探针读到正在被覆盖的目录（曾让 `probes/008` 报 PARTIAL） |
| **看到「报错文案在源码里已经不存在」时先怀疑加载了旧副本** | 三次排查里有三次是这个原因（`__pycache__`、别的 checkout、半覆盖的目录） |
| 中文日志乱码 | Windows 侧 Python 默认 GBK；设 `PYTHONIOENCODING=utf-8` 或直接读落盘文件 |

## 23. UNKNOWN

| 未知 | 需要的手段 |
| --- | --- |
| 不调 `SetAllocator` 时具体在哪一行崩 | 只知道 `GetAllocator(PrivateUse1)` 返回 0；在沙箱里故意不注册分配器、跑建张量用例，抓 core dump 栈 |
| 多流在 wall-clock 上到底有没有并发收益 | 两条流各跑一段带宽 kernel 比较总时间——本平台不能按 NVIDIA 经验估 |
| `cudaEventRecord` 的固定开销上限（多租户停顿的分布） | 空 kernel 下 record+sync 的分布统计，而不只是 p50 |
| 算子矩阵里「非末维 `dim=0`」的 4 条（39/43 的差额） | 用 permute→算→permute 回来的通用外壳兜住，再复跑 `probes/120` |
