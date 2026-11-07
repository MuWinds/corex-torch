# 与 corex-triton 的接缝：把运行期从厂商 torch 挪到 corex-torch 自己

本文回答一个问题：**corex-triton 已经跑通了这张卡，但它的运行期完全架在厂商 torch 2.1 的
`torch.cuda` 上；corex-torch 这边是上游 CPU 版 torch 2.8，没有可用的 `torch.cuda`。
这条接缝最薄在哪里？**

结论先说（每条都有本目录 `probes/150`–`161` 的实测支撑）：

| # | 结论 | 证据 |
| --- | --- | --- |
| 1 | **不需要改 corex-triton 的任何文件**。子类化 `CorexDriver` / `CorexUtils` / `CorexLauncher` / `CorexDeviceInterface` 四个类 + 覆盖 `GPUDriver.__init__` 带来的 4 个属性即可；全部 10 个 torch 触点都落在可覆盖的方法里 | §1、§4 |
| 2 | **编译侧 0 个 torch 触点**。`triton_corex/compiler.py` 与 `__init__.py` 里 `grep -n torch` 无命中；实测无 torch 进程里 `triton.compile` 直接拿到 `asm["cubin"]` | `probes/161`（登录节点）、`probes/151`（沙箱） |
| 3 | **运行期也不需要 torch**。ctypes 直连 `libcuda.so.1` 建 context → `cuModuleLoadData` → `cuLaunchKernel` 全部 rc=0，4096 元素逐字节相等 | `probes/151` |
| 4 | **设备层（libcudart）与算子层（libcuda）共享同一个 primary context**，与谁先初始化无关。所以两层之间**不需要传 context 句柄**，corex-torch 也不需要自己 `cuCtxCreate` | `probes/152` |
| 5 | 唯一「非覆盖即不可」的一步是 `_ensure_context()`：它在 `_LibCuda.get()` 里、被 `load_binary` 调用。绕法两行（覆盖 `load_binary`，或替换模块属性），不需要动上游 | §4.3 |
| 6 | `UNIFIED_ADDRESSING = 0`：host 指针不能当 device 指针用，算子层 H2D 策略必须「先拷到设备缓冲再传指针」 | `probes/150` §(a) |
| 7 | 动态 shared **默认上限就是 128 KiB**（不是 CUDA 那套「48 KiB + opt-in」），adapter 不需要补 `cuFuncSetAttribute` | `probes/153` |

---

## 1. 第 1 步：torch 触点全表

口径：逐行读 `corex-triton/src/triton_corex/{driver,compiler,__init__,rewrite}.py`，
把每一处「要 torch 提供什么」列出来，再回答「外部 provider 应该提供什么接口」。
行号以本文写作时的文件为准（`corex-triton` 是已交付仓库，本文不改它一个字）。

### 1.1 表 A：`triton_corex` 自己的触点

`compiler.py` 与 `__init__.py`：**0 处**（`grep -n torch` 无命中）。
`rewrite.py`：**0 处**（纯文本改写，不看 runtime）。

`driver.py` 一共 **10 个代码块 + 1 处继承来的基类行为**：

| # | 行号 | 位置 | 它要 torch 提供什么 | 外部 provider 必须提供的接口 |
| --- | --- | --- | --- | --- |
| 1 | 49 | `_LibCuda.get()` | 在 dlopen 之前调 `_ensure_context()`，保证「有 current context」才装载 | 一个 `ensure_context()`：保证当前线程有 context。**实测只要能 `cudaSetDevice(0)`**（见 §3.4），或什么都不做（primary context 会由 cudart 自动建） |
| 2 | 70–77 | `_ensure_context()` | `torch.cuda.is_available()` + `torch.cuda.init()` + `torch.zeros(1, device="cuda")` + `torch.cuda.synchronize()`，四件事合起来只为了「**建 primary context 并等它就绪**」 | `device_count() > 0`、`set_device(0)`、`synchronize()`。`_runtime` 里已有这三个，**不需要分配那块 1 元素张量** |
| 3 | 148–151 | `CorexUtils.get_current_stream()` | `torch.cuda.current_stream(dev).cuda_stream` → **一个 `CUstream` 句柄（整数）** | `current_stream_int(dev)`。返回我们自己的 `Stream.handle`（`c_void_p`）转 int，或直接 `0`（默认流；`probes/151` 实测 NULL stream 能启动） |
| 4 | 186–192 | `CorexLauncher.__init__` | `metadata.global_scratch_size > 0` 时 `torch.empty(size, int8, device="cuda")` → **一块设备暂存区**，把 `data_ptr()` 当 kernel 最后一个参数 | `alloc_scratch(nbytes) -> int`。`_runtime.malloc(nbytes)` 就够；vecadd 的 `global_scratch_size = 0`，这块平时不触发 |
| 5 | 258–261 | `CorexDriver.get_active_torch_device()` | `torch.device("cuda", dev)` | 返回 `torch.device("cuda", idx)`（若 torch 可用）或 `None`。**3.4 里没有调用点**：`grep -rn get_active_torch_device` 只命中抽象声明与 nvidia/amd 各自的实现，没有任何 caller，所以给 `None` 不影响任何现有路径 |
| 6 | 268–274 | `CorexDriver.get_empty_cache_for_benchmark()` | `torch.empty(size // 4, dtype=torch.int, device="cuda")`：`do_bench` 用来刷 L2 的缓冲 | `malloc(size)` + 返回一个「能 `fill_zero` 的东西」。见 #7 |
| 7 | 276–281 | `CorexDriver.clear_cache(cache)` | `cache.zero_()`（torch 张量方法） | 上一条返回值上的 `fill_zero()`（用 `cudaMemset`/`cuMemsetD8`，或用一次 H2D 零缓冲覆盖） |
| 8 | 299–301 | `CorexDeviceInterface.__init__()` | `self.Event = torch.cuda.Event` | 一个 `Event(enable_timing=True)` 构造器（`cudaEventCreate` + `elapsed_time`）。**注意 `do_bench` 会带 `enable_timing=` 关键字**，`_runtime.Event.__init__()` 收不到这个参数，需要一层适配 |
| 9 | 303–316 | `CorexDeviceInterface.synchronize / get_current_device / get_current_stream` | `torch.cuda.synchronize()` / `.current_device()` / `.current_stream(dev).cuda_stream` | `_runtime.synchronize()` / `_runtime.current_device()` / `current_stream_int(dev)` |
| 10 | 322–325 | `CorexDeviceInterface.empty_cache()` | `torch.cuda.empty_cache()` | 空操作即可：corex-torch 的分配器没有缓存池（`_runtime.malloc` 直连 `cudaMalloc`） |
| 11 | 231（`class CorexDriver(GPUDriver)`） | 继承来的基类行为 | **`GPUDriver.__init__` 里就有 `import torch`**，并装 4 个属性（见 §1.2 第 1 条） | 子类 `__init__` **不调 `super().__init__()`**，自己装那 4 个属性（§4.2 给了骨架） |
| 12 | 154–173 | `_pack_arg()`（不是 torch 调用） | 用 `pyval.data_ptr()` 取设备指针 | 不变。torch 2.8 CPU 上的 PrivateUse1 张量照样有 `data_ptr()`，所以这一层不用改 |

三处数量级的修正（不是接缝，但顺手记下，避免照抄旧常量）：

| 位置 | 现在写的 | 实测 | 证据 |
| --- | --- | --- | --- |
| `CorexUtils.get_device_properties()["max_shared_mem"]`（driver.py:105） | `48 * 1024` | **131072**（128 KiB，`MAX_SHARED_MEMORY_PER_BLOCK` 与 `_OPTIN` 都是 131072） | `probes/150` §(a) |
| `load_binary` 的第 5 个返回值 `n_max_threads`（driver.py:146） | `1024` | **4096**（`MAX_THREADS_PER_BLOCK`）；`_init_handles` 用它做 `num_warps * warp_size > n_max_threads` 的越界判断，写 1024 会让 `num_warps > 32` 的 kernel 被误判 | `probes/150` §(a) + `triton/compiler/compiler.py:492-494` |
| `do_bench` 的带宽换算要 `mem_clock_rate` / `mem_bus_width` / `multiprocessor_count` | 未提供 | `1200000` kHz / `4096` bit / `16` | `probes/150` §(a) + `triton/testing.py:428-441` |

### 1.2 表 B：上游 Triton 3.4 自己碰 torch 的地方（不在我们的文件里，但绕不过）

行号来自 `~/venvs/triton34/lib/python3.10/site-packages/triton/`：

| # | 位置 | 内容 | 对本设计的影响 |
| --- | --- | --- | --- |
| 1 | `backends/driver.py:39-49` | `GPUDriver.__init__`：`import torch`；然后装 `get_device_capability`、`get_current_stream`（优先 `torch._C._cuda_getCurrentRawStream`，否则 `lambda idx: torch.cuda.current_stream(idx).cuda_stream`）、`get_current_device`、`set_current_device` | **这是整个接缝唯一的硬点**：只要实例化一个继承自 `GPUDriver` 的类并调 `super().__init__()`，就必然 `import torch`。子类不调它即可（`DriverBase.__init__` 是空函数，Python 不强制） |
| 2 | `compiler/compiler.py:474-494` | `CompiledKernel._init_handles`：`get_current_device()` → `launcher_cls(src, metadata)` → `utils.get_device_properties(dev)["max_shared_mem"]` → `utils.load_binary(name, kernel, shared, dev)`（**5 元组**）→ `get_current_target().warp_size` | 这就是 adapter 的「必填面」：**6 个东西**，不多不少 |
| 3 | `compiler/compiler.py:496-499` | `__getattribute__` 拦 `run`，第一次启动才 `_init_handles()` | 所以 **`triton.compile` 全程不碰 driver**——这正是 §2 里编译侧 0 torch 的机制原因 |
| 4 | `compiler/compiler.py:519-521` | `CompiledKernel.__getitem__.runner`：`get_current_device()` + `get_current_stream(device)` | 启动时若显式传 `stream=` 就不查；否则要这两个 |
| 5 | `runtime/driver.py:8-12` | `_create_driver()`：`[x.driver for x in backends.values() if x.driver.is_active()]`，**必须恰好一个**，否则 `RuntimeError("N active drivers")` | 我们直接 `driver.set_active(<我们的实例>)`，绕开这套自动挑选（也就避开了 amd/nvidia 的 `is_active` 里 `import torch`） |
| 6 | `backends/amd/driver.py:614-618`、`backends/nvidia/driver.py:735-739` | 两个 `is_active()` 都 `import torch`，且只 `except ImportError` | 实测在这套环境里 `import torch` 抛的是 **`OSError: libcufft.so.10`**（动态链接错误，不是 `ImportError`），**不会被吞掉**：`probes/160` 在登录节点上 `driver.active.get_current_target()` 直接抛 OSError。所以无 torch 进程里绝不能走 `driver.active` 的懒解析 |
| 7 | `runtime/jit.py:551,713` | `@triton.jit` 直接调用（不预先 `triton.compile`）的路径同样取 `get_current_device` / `get_current_stream` | 同一个必填面 |
| 8 | `runtime/autotuner.py:121` | `driver.active.get_benchmarker()` | 保留父类的 `get_benchmarker`（返回 `triton.testing.do_bench`，与目标无关） |
| 9 | `testing.py:147,152,428,429,441,526` | `get_device_interface()`、`get_empty_cache_for_benchmark()`、`get_device_properties(...)["mem_clock_rate"/"mem_bus_width"/"multiprocessor_count"]` | `do_bench` 与 `do_bench_cudagraph` 的入口；我们不用 cudagraph（硬禁），只需前两个可用 |

### 1.3 判定：子类化就够，不需要改源码

判据是「有没有一处 torch 触点落在**无法覆盖**的位置」。逐条查：

- 10 个代码块**全部**在 `CorexUtils` / `CorexLauncher` / `CorexDriver` / `CorexDeviceInterface`
  的**实例方法或静态方法**里 → 子类覆盖即生效。
- 唯一的「非方法」是第 11 条：`GPUDriver.__init__`。它由 `CorexDriver.__init__` 里的
  `super().__init__()` 触发 → 子类不调即可。
- `triton_corex/__init__.py` 的 `register()` 只往 `triton.backends.backends` 塞一个
  `Backend(CorexBackend, CorexDriver)` **类**（不实例化），`set_active()` 才实例化 →
  我们换成自己的 `Backend(CorexBackend, CorexTritonDriver)` 并自己 `set_active`。
- `compiler.py` 一个 torch 都没有 → 编译链原样复用，连子类都不用做。

**结论：继承覆盖 4 个类 + 1 个 `__init__` 策略 = 接缝闭合，`corex-triton` 零改动。**

### 1.4 需要改动上游的项

**0 条。**（这一节保留结构，将来真出现时往这里加。）

两条「看起来非改不可、实际不用改」的，记下来免得后人重走：

| 看起来必须改的 | 为什么不用改 | 绕法 |
| --- | --- | --- |
| `triton/backends/driver.py:41` 的 `GPUDriver.__init__` 里 `import torch` | 父类没被强制调用：`DriverBase.__init__` 是 `pass`。子类自己装那 4 个属性即可 | 覆盖 `__init__`，不写 `super().__init__()`（§4.2） |
| `triton_corex/driver.py:70` 的 `_ensure_context()` 里 `import torch`，而它在 `_LibCuda.get()` 里被 `load_binary` 调用 | `CorexUtils.load_binary` 是可覆盖方法；另外那行是 `from triton_corex.driver import _ensure_context` 写在**函数体内**，是运行时查模块属性，因此 `triton_corex.driver._ensure_context = 我们的实现` 也能生效 | 首选覆盖 `load_binary`（显式），monkeypatch 作为「一行版本」备选 |

---

## 2. 第 2 步：编译侧真的不需要 torch（实测）

### 2.1 无 torch 进程里拿到 `asm["cubin"]`（沙箱真机）

`probes/151_torchfree_launch.sh`，用 `~/venvs/triton34/bin/python`（`--system-site-packages`，
所以**厂商 torch 2.1 在 sys.path 上可见**——这正是要显式检查 `sys.modules` 的原因）：

```
[start]                     torch_modules=0 []
[after_import_triton]       torch_modules=0 []
[after_import_triton_corex] torch_modules=0 []
[after_compile]             torch_modules=0 []
COMPILE_OK sec=0.519 asm_keys=['biir','cubin','llir','source','ttgir','ttir']
cubin_bytes=3464 sha256=22eaaf72b37c21e2e0a29aa34d71cf3488b0208170c934585e11d67f352fa9fd
metadata name='vecadd' num_warps=4 shared=0 num_ctas=1 global_scratch_size=0 warp_size=32
```

`sys.modules` 里 torch 相关模块**从头到尾是 0 个**。机制原因有两条源码证据：
`triton_corex/driver.py` 的所有 `import torch` 都在函数体内（模块顶层只有 `ctypes/hashlib/os`）；
`compiler.py:496-499` 的 `__getattribute__` 让 `_init_handles()`（唯一会碰 driver 的地方）
推迟到第一次启动。

命令：`python tools/remote.py sbx-run probes/151_torchfree_launch.sh --name 151 --tail 0`

### 2.2 登录节点能编到 `biir`，只差厂商 `llc`

`probes/161_login_biir.sh`（登录节点，不走队列，不碰卡）：

```
CORE_HOME_EXISTS=no          # 登录节点没有 /usr/local/corex
[start] torch_modules=0 []
triton=3.4.0 backends=['amd','corex','nvidia'] corex_registered=True
[after_import_triton_corex] torch_modules=0 []
COMPILE_RAISED RuntimeError: 找不到厂商 llc：/usr/local/corex/bin/llc（用 COREX_HOME 指定 CoreX 安装根）
[after_compile] torch_modules=0 []
dump_glob=.../biir161_*.ll hits=['biir161_vecadd.ll']
define_line=define iluvatar_kernel void @vecadd(ptr addrspace(1) %0, ptr addrspace(1) %1,
            ptr addrspace(1) %2, i32 %3, ptr addrspace(1) readnone %4) local_unnamed_addr #0 !dbg !4 {
biir bytes=3646 triple_bi=True iluvatar_kernel=1 llvm_bi_hits=0 nvvm_sreg_hits=4 params=5
no_torch=True dump_ok=True failed_at_llc=True
PROBE161_VERDICT PASS
```

三条可抄的结论：

1. `import triton_corex` **只在注册后端表，不 import torch**（`torch_modules=0`）。
2. 编译的 5 个阶段（`ttir → ttgir → llir → biir → cubin`）里**前 4 个在登录节点就能跑完**，
   产物是 `bi-iluvatar-ilurt` 三元组 + `define iluvatar_kernel` 的合法 bi IR；
   只有最后一步 `cubin` 需要厂商 `llc`（登录节点没有）。
3. `vecadd` 里 `llvm.bi.*` 计数是 **0** 属于正常（它不用任何 warp 级原语），
   `llvm.nvvm.read.ptx.sreg.*` 命中 **4** 次才是「线程号已换成 NVVM intrinsic」的证据。
   **拿 `llvm.bi.` 计数当改写成功的判据是错的**——这一条是 `probes/160` 第一版的教训（见下）。

### 2.3 `probes/160` 第一版为什么 FAIL（留作证据，不改它）

`probes/160_login_compile_no_torch.sh` 判定 FAIL，不是链路的锅，是探针自己两个假设错了：

| 探针的错 | 实际 | 出处 |
| --- | --- | --- |
| 以为 `COREX_DUMP_IR=<路径>` 就落盘到 `<路径>` | 落盘名会**插入 kernel 名**：`<stem>_<kernel><suffix>`，实测 `login_biir_vecadd.ll` | `triton_corex/compiler.py:259-265` |
| 以为改写成功 = `llvm.bi.` 计数 > 0 | `vecadd` 不含 warp 级原语，计数必然 0；正确判据是 triple + `iluvatar_kernel` + `nvvm sreg` | `probes/161` 重跑 |

按仓库纪律，旧的探针保留原样（它还要能重放出「当时那个错判」），修正版另开 161。

### 2.4 这个进程里第一次 `import torch` 到底发生在哪一步

`probes/160` 的 160b 段（登录节点）：

```
[before] torch_modules=0
SET_ACTIVE_RAISED OSError: libcufft.so.10: cannot open shared object file
[after]  torch_modules=5 ['torch._utils','torch._utils_internal','torch.corex','torch.torch_version','torch.version']
```

也就是说：`triton_corex.set_active()` → `CorexDriver()` → `GPUDriver.__init__` → `import torch`
**是第一个、也是唯一一个**隐式 torch 触点。登录节点上它连动态链接都过不去（缺 `libcufft.so.10`），
所以这条在登录节点永远是红的；在沙箱里它能过（厂商 torch 完整）。

配套的 `driver.active` 解析实验（同一登录节点）：

```
[1] is_active per backend:
    amd     -> RAISED OSError: libcufft.so.10: cannot open shared object file
    nvidia  -> RAISED OSError: libcufft.so.10: cannot open shared object file
    corex   -> False                      # 登录节点没有 /usr/local/corex，is_active 正确地返回假
[3] RESOLVE_RAISED OSError: libcufft.so.10 ...
```

四条含义：
1. `driver.active` 本身是 `LazyProxy`，但**任何属性访问**都会触发 `_create_driver()`。
2. `_create_driver()` 先问每个后端的 `is_active()`，而 amd/nvidia 的实现里 `import torch`，
   且只 `except ImportError` —— 动态链接错误 `OSError` **不会被吞**，直接冒泡。
3. 所以**无 torch 进程里不能碰 `driver.active`**；编译链恰好一次都没碰（§2.1）。
4. `corex.is_active()` 只看 `/usr/local/corex/lib64/libcuda.so.1` 在不在，登录节点返回 False —— 行为正确。

### 2.5 UNKNOWN

| 未知 | 需要的手段 |
| --- | --- |
| torch 2.8 CPU wheel 装进 `~/venvs/corex28` 之后，`driver.active` 的懒解析在**沙箱里**是否自动选中 corex（CPU 版 `torch.cuda.is_available()` 应返回 False → amd/nvidia 的 `is_active` 都假 → 只剩 corex 一个） | torch 装好后跑 `probes/151` 的等价脚本但用 `corex28` 的 python，打印 `driver.active` 解析结果与 `torch_modules`。**在这条确认之前，adapter 一律显式 `driver.set_active()`**，不要依赖自动挑选 |

---

## 3. 第 3 步：启动侧（a)(b)(c) 与跨 API 共享 context

### 3.1 (a) 设备属性：只用 ctypes 打 `libcuda.so.1`，21 条属性全 rc=0

`probes/150_noctx_runtime.sh`（`PROBE150_VERDICT PASS`，用 `~/venvs/corex28/bin/python`，
那个 venv 里**连 torch 都没装**）：

```
python 3.10.16 …/venvs/corex28/bin/python
libcuda /usr/local/corex/lib64/libcuda.so.1 exists= True
torch_in_sys_modules(before) False
cuInit(0) rc=0(SUCCESS)
cuDeviceGetCount rc=0 count=2                  # 继承 CVD=6,7，子进程里重编号 0/1
cuDriverGetVersion rc=0 version=10020          # 10.2
cuDeviceGetName rc=0 name='Iluvatar BI-V100'
cuDeviceTotalMem rc=0 bytes=34359738368        # 32 GiB
attrs_ok=21 attrs_failed=0
```

| 属性（CUDA 10.2 编号） | 值 | 备注 |
| --- | --- | --- |
| `COMPUTE_CAPABILITY_MAJOR/MINOR` (75/76) | 7 / 0 | 与 `GPUTarget("corex", 70, 32)` 里的 70 一致 |
| `WARP_SIZE` (10) | **64** | 与 `corex-triton/docs/00-FACTS.md` §8 的硬件实测一致；`GPUTarget(...,32)` 的 32 是**布局自洽值**，不是硬件值 |
| `MAX_THREADS_PER_BLOCK` (1) / `MAX_BLOCK_DIM_X` (2) | 4096 | `load_binary` 的 `n_max_threads` 应该报 4096，不是现在的 1024 |
| `MAX_BLOCK_DIM_Y/Z` (3/4) | 4096 / 256 | |
| `MAX_GRID_DIM_X` (5) | 2147483647 | |
| `MAX_SHARED_MEMORY_PER_BLOCK` (8) | **131072** | 128 KiB；`_OPTIN` (97) 也是 131072 |
| `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR` (81) | 131072 | |
| `MULTIPROCESSOR_COUNT` (16) | 16 | |
| `MAX_THREADS_PER_MULTIPROCESSOR` (39) | 8192 | |
| `MAX_REGISTERS_PER_BLOCK` (12) / `_PER_MULTIPROCESSOR` (82) | 262144 / 262144 | |
| `CLOCK_RATE` (13) / `MEMORY_CLOCK_RATE` (36) | 1500000 / 1200000 kHz | `do_bench` 的带宽换算要后一个 |
| `GLOBAL_MEMORY_BUS_WIDTH` (37) | 4096 bit | 同上 |
| `L2_CACHE_SIZE` (38) | 33554432 | |
| `UNIFIED_ADDRESSING` (41) | **0** | 见 §3.5 |
| `CONCURRENT_KERNELS` (31) | 1 | |

一条 shim 的破绽：`cuCtxGetApiVersion` 报 **api=10000**（10.0），而 `cuDriverGetVersion`
报 10020（10.2）。与 `corex-triton/docs/00-FACTS.md` §5「裸 driver API 有壳子成分」呼应 ——
**适配层不要拿 api version 做分支判断**。

### 3.2 (b) 显存往返：三档尺寸逐字节一致

同一探针，`cuMemAlloc_v2` + `cuMemcpyHtoD_v2`/`DtoH_v2` + `cuCtxSynchronize`：

| 字节数 | 设备指针 | 256 对齐 | 结果 | D2H 耗时 |
| --- | --- | --- | --- | --- |
| 4096 | `0x304000002000000` | 是 | `equal=True first_diff=none` | 0.027 ms |
| 1048576 | `0x304000002000000` | 是 | `equal=True first_diff=none` | 0.134 ms |
| 1237（非 4 倍数） | `0x304000002000000` | 是 | `equal=True first_diff=none` | 0.023 ms |

`cuMemGetInfo` free/total = 32480 MiB；`re-alloc 1MiB` rc=0（`cuMemFree_v2` 有效，只是受
CUDA 缓存行为影响复用了同一地址段）；`torch_in_sys_modules(after) False`；事后 `CARD_STILL_HEALTHY`。

### 3.3 (c) 无 torch 进程里装载 + 启动 kernel（**判决点**）

`probes/151_torchfree_launch.sh`：

```
ctx_symbols {'primary_ctx':'cuDevicePrimaryCtxRetain','set_cur':'cuCtxSetCurrent','alloc':'cuMemAlloc_v2',…}
cuDevicePrimaryCtxRetain rc=0 ctx=0x21e7a40   cuCtxSetCurrent rc=0
device_ptrs a=0x304000002000000 b=0x304000002004000 c=0x304000002008000
launch grid=(32,1,1) block=(128,1,1) shared=0
cuModuleLoadData rc=0 mod=0x2950dd0
cuModuleGetFunction('vecadd') rc=0 fn=0x220a5d0
cuStreamCreate(cuStreamCreate) rc=0 stream=0x22f7330
cuLaunchKernel(stream=null) rc=0 sync_rc=0
cuLaunchKernel(stream=own)  rc=0 sync_rc=0
RESULT equal_bytes=True n=4096 c[0]=1.0 c[7]=3.625 c[-1]=1536.625
PROBE151_VERDICT PASS
```

四个能直接用的实现事实：

1. **kernel 参数表 = 5**：从 `biir` 的 `define iluvatar_kernel ...` 行数出来（`@vecadd(ptr,ptr,ptr,i32,ptr)`），
   前 4 个是用户参数，第 5 个是 Triton 3.4 末尾的 global scratch 指针（`driver.py:186-192`
   的注释与此处一致）。猜错一位就是静默读到垃圾，所以**参数表必须从 IR 数，不要硬编码**。
2. **`stream=NULL` 就能启动**：适配层可以先不给流。给了自建 `cuStreamCreate` 的流同样 rc=0。
3. 判定用 **逐字节相等**：输入取 fp32 精确可表示值（`1.5+0.25i` / `-0.5+0.125i`），
   参考值在 Python 侧算，不是「误差够小」。
4. `blockDim = num_warps * target.warp_size = 4*32 = 128`，`shared = metadata.shared = 0`。

### 3.4 跨 API：libcudart 与 libcuda 是**同一个** primary context

`probes/152_cross_api_context.sh`（`PROBE152_VERDICT PASS`，两种顺序各一遍，rc_a=0 rc_b=0）：

| 顺序 | 量 | 结果 |
| --- | --- | --- |
| 共同 | `cuInit(0)` 之后、任何 cudart 调用之前 | `cuCtxGetCurrent ctx=NULL`、`cuCtxGetDevice rc=201(invalid device context)` —— 负对照，context 确实还没建 |
| **A：runtime 先** | `cudaSetDevice(0)` 之后 | `cuCtxGetCurrent ctx=0x1393780` **已经非 NULL**（这张卡上 SetDevice 就把 primary context 建好并设 current 了，与 NVIDIA 行为不同） |
| | `cuDevicePrimaryCtxRetain(0)` | `ctx=0x1393780` → **`SAME_PRIMARY_CONTEXT=True`** |
| **B：driver 先** | `cuDevicePrimaryCtxRetain`+`cuCtxSetCurrent`，再 `cudaSetDevice`+`cudaMalloc` | 全部 rc=0，句柄仍是 `0x2beeec0` → **`SAME_PRIMARY_CONTEXT=True`** |

在 runtime 建出来的 context 里用 driver API 启动（全程 `torch_modules=0`）：

```
cuModuleLoadData rc=0  cuModuleGetFunction('vecadd') rc=0
cudaStreamCreate rc=0 stream=0x13b96c0
cuLaunchKernel(stream=null)       rc=0(SUCCESS) cuCtxSync rc=0 cudaDeviceSync rc=0(no error)
cuLaunchKernel(stream=cudaStream) rc=0(SUCCESS) cuCtxSync rc=0 cudaDeviceSync rc=0(no error)
```

指针双向互通，全部逐字节相等：

| 方向 | 结果 |
| --- | --- |
| `cudaMalloc` 的指针 → `cuMemcpyDtoH` | rc=0 `equal_bytes=True` |
| 同一指针 → `cudaMemcpy(D2H)` | rc=0 `equal_bytes=True` |
| `cuMemAlloc` 的指针 → `cudaMemcpy(D2H)` | rc=0 `equal=True` |

**结论**：设备层（cudart）与算子层（libcuda）在**同一个进程**里天然共享一个 primary context，
与谁先初始化**无关**。因此：

- corex-torch **不需要** `cuCtxCreate`，也**不需要**往 Triton 的 launcher 传 context 句柄；
- `CorexUtils.get_current_stream()` 直接把 `_runtime.Stream.handle` 转 int 返出去即可
  （`cudaStreamCreate` 的流当 `CUstream` 用，实测 rc=0）；连流都可以先不给（NULL 也行）；
- 唯一要守的是「同一进程内所有调用看到同一个 `CUDA_VISIBLE_DEVICES` 语义」，因为
  `cudaSetDevice(i)` 与 `cuDevicePrimaryCtxRetain(i)` 的 `i` 都是**重编号后的逻辑号**。

### 3.5 `UNIFIED_ADDRESSING = 0` 的影响（Lead 要的那一句）

`probes/150` §(a)：`UNIFIED_ADDRESSING id=41 = 0`。

> UVA 关闭 ⇒ host 指针与 device 指针是两个地址空间。Triton kernel 里所有 `*fp32` 实参**必须**是
> `cudaMalloc`/`cuMemAlloc` 出来的设备内存；任何 `torch.Tensor` 若落在 host 内存上，`data_ptr()`
> 传过去会被当成设备地址读 → 静默垃圾或 ECC 报错。所以算子层的 H2D 策略必须是「先拷到设备缓冲
> 再传指针」，不能省；`cudaHostAlloc`/pinned 也救不了这一点（它只影响拷贝带宽，不改变地址空间）。

### 3.6 动态 shared 的上限：**默认就是 128 KiB，不需要 opt-in**

`CorexLauncher.__call__`（driver.py:217）把 `metadata.shared` 原样当 `cuLaunchKernel` 的
`sharedMemBytes` 传下去，所以这张尺子是 launcher 自己的事。`probes/153_shared_limit.sh`
（`PROBE153_VERDICT PASS`）用一个 `shared = 0` 的 vecadd kernel 当尺子（驱动只校验「申请量 ≤
该函数的动态 shared 上限」，不需要 kernel 真的用 shared）：

| 申请量 | 未做 opt-in | `cuFuncSetAttribute(8, 128 KiB)` 之后 | `cuFuncSetAttribute(8, 100 KiB)` 之后 |
| --- | --- | --- | --- |
| 0 / 48 KiB / 48 KiB+4 / 64 KiB / 96 KiB | rc=0 | rc=0 | rc=0 |
| 100 KiB | rc=0 | rc=0 | rc=0 |
| **128 KiB（131072）** | **rc=0** | rc=0 | **rc=1(invalid argument)** |
| 132 KiB（135168） | rc=1(invalid argument) | rc=1(invalid argument) | rc=1 |

三条结论（直接关掉 `docs/00-FACTS.md` §8 里那条 UNKNOWN）：

1. **这张卡的 shim 不实现 CUDA 的「默认动态 shared 上限 48 KiB」**：不做
   `cuFuncSetAttribute` 就能申请到 131072（与 `MAX_SHARED_MEMORY_PER_BLOCK` 相等），
   132 KiB 才报 `rc=1(invalid argument)`。
2. `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES=8, N)` 这个符号存在且
   真的生效（设成 100 KiB 后 128 KiB 立刻变成 rc=1）—— 所以它只能用来**收紧**，不是放宽。
3. **adapter 里不需要在 `load_binary` 之后补 `cuFuncSetAttribute`**；只要 `metadata.shared`
   不超过 128 KiB，`cuLaunchKernel` 就收。顺带把 corex-triton 里 48 KiB 那个硬编码上限
   （driver.py:105）的保守量也解释了：那是**它自己给自己设的检查阈值**，不是硬件上限。

---

## 4. minimal adapter 设计草图

下面是**设计**，不落到 `src/`（`src/` 归 Lead）。目标：一个文件、约 200 行、零改动上游。

### 4.1 文件与依赖

建议 `src/corex_torch/triton_bridge.py`：

- 依赖 `triton_corex`（复用它的编译链与 `_pack_arg`/`cu*` 绑定）与 `corex_torch._runtime`（设备层）；
- **不 import torch**（`get_active_torch_device` 里按需 `import torch` 并 `try/except`）；
- `install()` 一个幂等入口：注册 backend 组并 `driver.set_active()`。

### 4.2 五个类 + 一个函数

```python
# 1) 驱动：整个接缝的核心
class CorexTritonDriver(triton_corex.driver.CorexDriver):
    def __init__(self) -> None:
        # 故意不调 super().__init__()：GPUDriver.__init__（triton/backends/driver.py:41）
        # 里就是 import torch，并把 4 个属性指向 torch.cuda.*。
        self.get_device_capability = self._device_capability          # 属性 1
        self.get_current_stream = lambda idx=0: _stream_int(idx)      # 属性 2
        self.get_current_device = _runtime.current_device             # 属性 3
        self.set_current_device = _runtime.set_device                 # 属性 4
        self.utils = CorexTritonUtils()
        self.launcher_cls = CorexTritonLauncher

    def _device_capability(self, device=None) -> tuple[int, int]:
        # 3.4 里没有任何 driver.active.get_device_capability 的调用点（只有 testing.py 直连
        # torch.cuda），但基类把它当属性装着，给个一致的值免得将来报 AttributeError。
        return (70, 0)          # 与 GPUTarget("corex", 70, 32) 对齐；实测属性号 75/76 = 7/0

    def get_active_torch_device(self):
        # 3.4 里没有调用点；给了是为了 torch 侧将来（inductor / _internal_testing）能用。
        try:
            import torch
            return torch.device(_DEVICE_STR, self.get_current_device())   # _DEVICE_STR = "cuda"
        except Exception:
            return None

    def get_empty_cache_for_benchmark(self, size: int = 256 * 1024 * 1024):
        return _ScratchBuffer(size)     # 见 4.3 #5

    def clear_cache(self, cache) -> None:
        cache.fill_zero()

# 2) 工具层：属性 + 装载
class CorexTritonUtils(triton_corex.driver.CorexUtils):
    @staticmethod
    def get_device_properties(device: int = 0) -> dict:
        p = _runtime.properties(device)
        return {"max_shared_mem": p["shared_mem_per_block"],   # 131072，不是 48 KiB
                "name": p["name"],
                "multiprocessor_count": p["multiprocessor_count"],   # do_bench 要
                "mem_clock_rate": 1200000, "mem_bus_width": 4096}   # do_bench 要（实测值）

    @staticmethod
    def load_binary(name, kernel, shared, device=0):
        # 覆盖理由：父类里 _LibCuda.get() -> _ensure_context() -> import torch。
        # 这里保留父类全部行为，只是把「建 context」换成 _runtime.set_device(device)。
        _runtime.set_device(device)
        return _load_module_cached(name, kernel)   # 复用父类的 cuModuleLoadData/缓存逻辑
        # 返回 5 元组 (module, function, 0, 0, 4096)：n_max_threads 用实测的 4096

    @staticmethod
    def get_current_stream(device: int = 0) -> int:
        return _stream_int(device)

# 3) 启动器：只多了 global scratch 的来源
class CorexTritonLauncher(triton_corex.driver.CorexLauncher):
    def __init__(self, src, metadata) -> None:
        self.src = src
        self.metadata = metadata
        self.arg_types = [ty for _, ty in src.signature.items() if ty != "constexpr"]
        size = int(getattr(metadata, "global_scratch_size", 0) or 0)
        self._scratch = _runtime.malloc(size) if size > 0 else None   # ← 唯一改动点
        self._scratch_ptr = (self._scratch.value or 0) if self._scratch else 0

# 4) 设备接口：给 do_bench 用
class CorexTritonDeviceInterface(triton_corex.driver.CorexDeviceInterface):
    def __init__(self) -> None:
        self.Event = CorexEvent                  # 不 import torch.cuda.Event（父类会）

    @staticmethod
    def synchronize() -> None: _runtime.synchronize()
    @staticmethod
    def get_current_device() -> int: return _runtime.current_device()
    @staticmethod
    def get_current_stream(device: int = 0) -> int: return _stream_int(device)
    @staticmethod
    def empty_cache() -> None: ...               # 空操作：_runtime 没有缓存池

# 5) 事件适配器：do_bench 会调 Event(enable_timing=True)
class CorexEvent:
    def __init__(self, enable_timing: bool = False) -> None:
        self._ev = _runtime.Event()              # cudaEventCreate（CUDA 语义下默认开计时）
    def record(self, stream=None) -> None:
        self._ev.record(stream or _runtime.Stream.default())
    def synchronize(self) -> None: self._ev.synchronize()
    def elapsed_time(self, other: "CorexEvent") -> float:
        return self._ev.elapsed_ms(other._ev)    # 注意 Triton 用 elapsed_time(ms)

# 6) 安装入口
def install() -> None:
    from triton.backends import Backend, backends
    from triton.runtime.driver import driver as _drv
    backends["corex"] = Backend(triton_corex.CorexBackend, CorexTritonDriver)
    _drv.set_active(CorexTritonDriver())          # 显式注入，绕开 _create_driver 的
                                                  # 「恰好一个 is_active」+ amd/nvidia 里 import torch
```

### 4.3 四个必须记住的点（都是踩过的）

| # | 点 | 为什么 |
| --- | --- | --- |
| 1 | `CorexTritonDriver.__init__` **不调 `super().__init__()`** | `GPUDriver.__init__`（`triton/backends/driver.py:41`）里 `import torch` + 4 个 torch.cuda 属性。基类 `DriverBase.__init__` 是空函数，不调完全合法 |
| 2 | 必须自己装 **4 个属性**：`get_device_capability`、`get_current_stream`、`get_current_device`、`set_current_device` | 它们不是方法而是**实例属性**（基类 `__init__` 里赋的）；不装就会 `AttributeError`。`get_current_stream` 的签名是 `(device: int) -> int` |
| 3 | `install()` 里用 `driver.set_active(...)`，**不要**用 `triton_corex.set_active()` | 后者实例化父类 `CorexDriver` → 必碰 torch；且 `_create_driver()` 在厂商 torch 也在同一环境时会报 `2 active drivers` |
| 4 | 覆盖 `load_binary` 而不是覆盖 `_LibCuda` | `_LibCuda.get()` 是模块级单例、`_ensure_context` 在它内部按**模块属性**查；覆盖 `load_binary`（父类唯一调用它的地方）最省事，也最容易被后人看懂 |
| 5 | `get_empty_cache_for_benchmark` 的返回物要有 `fill_zero()`，且 `clear_cache(cache)` 只调这一个 | `do_bench` 的口径是「每次计时前把 L2 刷掉」，内容无所谓；`_runtime` 没有 `Memset` 绑定，用一次 H2D 零缓冲覆盖最简单 |
| 6 | `CorexEvent` 的构造器签名必须是 `(enable_timing: bool = False)` | `triton/testing.py:156,168,169` 都是 `di.Event(enable_timing=True)`；`_runtime.Event()` 不收参数 |
| 7 | **不要**在 `load_binary` 之后补 `cuFuncSetAttribute` | 这张卡的动态 shared 默认上限就是 128 KiB（§3.6），不 opt-in 也能申请到 |

### 4.4 验收顺序（跑一遍就知道接缝闭没闭上）

| 步骤 | 命令 | 期望 |
| --- | --- | --- |
| 编译链 | `python tools/remote.py sbx-run probes/151_torchfree_launch.sh --name 151 --tail 0` | `PROBE151_VERDICT PASS`（已通过） |
| 跨 API context | `python tools/remote.py sbx-run probes/152_cross_api_context.sh --name 152 --tail 0` | `PROBE152_VERDICT PASS`（已通过） |
| adapter 装上后 | 同一脚本但改成 `install(); kernel[(grid,1,1)](...)`（Triton 自己的启动路径），对照 151 的逐字节相等 | `equal_bytes=True` + `sys.modules` 里出现 `corex_torch` 而**没有** `torch.cuda` 相关调用 |
| 首次真正用 torch | 上游 torch 2.8 CPU 装好后，`torch.zeros(8, device="corex")` | 不在本文范围，见 `docs/03-*` |

---

## 5. 探针索引（全部可重放）

| 探针 | 位置 | 问的问题 | 结论 |
| --- | --- | --- | --- |
| `150_noctx_runtime.sh` | 沙箱（卡） | 无 torch 进程里能否 `cuInit`+primary ctx+属性+显存往返 | PASS，21 属性全 rc=0，三档往返逐字节相等 |
| `151_torchfree_launch.sh` | 沙箱（卡） | 无 torch 进程里能否编译出 cubin 并自己启动 | PASS，cubin 3464 B，逐字节相等 |
| `152_cross_api_context.sh` | 沙箱（卡） | cudart 与 libcuda 是否同一个 primary context（两种顺序） | PASS，句柄相同，指针双向互通 |
| `153_shared_limit.sh` | 沙箱（卡） | 动态 shared 是 48 KiB 默认还是 128 KiB？要不要 opt-in | PASS，**默认 128 KiB**，132 KiB 才拒；`cuFuncSetAttribute` 只能收紧 |
| `160_login_compile_no_torch.sh` | 登录节点 | import 链 + 编译到 biir + 第一次 import torch 在哪 | 链路 OK；探针对 dump 文件名/判据的假设错（见 §2.3），保留作证据 |
| `161_login_biir.sh` | 登录节点 | 修正版：无 torch 编到 biir 并核对 bi IR | PASS |
| `170_eltwise.sh` | 沙箱（卡） | 元素级算子 fp32：正确性 + 尾块 + 带宽（含 L2 口径对比） | PASS，45/45 逐字节相等，add 内核 295.5 GB/s（见 §6） |
| `171_eltwise_dtype.sh` | 沙箱（卡） | 同上但 fp16 / bf16 | fp16 **33/33 逐字节相等**；bf16 6/33（`fp_round` 选不出来，见 §6.7） |
| `180_reduce_softmax.sh` | 沙箱（卡） | 规约（sum/mean/max/sum_all）+ softmax：正确性/尾块/带宽/bf16 契约 | PASS，22/22；softmax max_abs ≤ 1.15e-08（见 §7） |
| `181_mm.sh` | 沙箱（卡） | mm/bmm：dyadic 逐字节 + 非 dyadic 误差 + GFLOP/s + 分块扫描 | PASS，11/11；256³ **188.8 GFLOP/s**（见 §7） |
| `182_norm_index.sh` | 沙箱（卡） | rms_norm / layernorm / gather_rows：正确性 + 带宽 | PASS，12/12（见 §7） |
| `190_bf16_path.sh` | 沙箱（卡） | bf16 位运算通路可行性：展宽/收窄/纯 bf16 算术各编不编得过 | 展宽通、**收窄撞 `trunc nuw`**（见 §8.4） |
| `191_bf16_narrow.sh` | 沙箱（卡） | 收窄三条候选路（`.to(u16)` / `tl.view` / 成对 u32 打包）+ 纯 bf16 算术逐字节 | A 挂、B 挂、C 通；成对 add 撞 `splat (i32 16)`（见 §8.4） |
| `192_bf16_ops.sh` | 沙箱（卡） | **bf16 全算子验收**：元素级逐字节 / 累加类对照 / mm K=256,1024 / 性能 / 改写体检 | **PASS**，33/33 + 15/15 + 5/5，改写命中 236，可疑语法 0（见 §8） |
| `193_bf16_reduce_diag.sh` | 沙箱（卡） | 定位改写层漏了什么：把真实 biir 里 `fpext/fptrunc` 的原文打出来 | 找到 `fpext <8 x bfloat>` 向量形式漏改（见 §8.2） |
| `194_bf16_vector_ir.sh` | 沙箱（卡） | 向量 intrinsic 有哪些拼写可用（**只看 llc，结论已被 195 更正**） | ⚠ 误判「`.v8bf16` 可用」，保留作证据 |
| `195_bf16_link_verdict.sh` | 沙箱（卡） | **链接判据**（llc + `lld --no-undefined`）三级对照 | V1 `llc=FAIL` / V2 `llc=OK lld=FAIL` / V3 标量化 `llc=OK lld=OK`（见 §8.2） |
| `196_new_ops.sh` | 沙箱（卡） | **第 5 批算子**：relu/rsqrt/gelu/masked_fill/max.dim+argmax/log_softmax × 3 dtype × 尾块 | **PASS**，63/63 + 36/36，bf16 命中 130（见 §9） |
| `197_var_axis.sh` | 沙箱（卡） | **最后一批**：`row_var`/`row_std`（两趟 vs 一趟公式对照）、中间维 `sum_axis`/`mean_axis` | **PASS**，27/27 + 18/18，bf16 命中 17（见 §10） |
| `198_bmm_repro.sh` | 沙箱（卡） | 复现「`bmm` 2505 µs」并做分布+对照组归因 | **假阳性**：p50 两形状都是 13.5 µs，均值被 4% 的 ~500 µs 环境停顿污染（见 §11） |
| `199_fastop_api.sh` | 沙箱（卡） | C++ 快路径的两个入口：`get_kernel` / `fast_op`，并用「只按表 launch」的 shim 模拟器验收 | **PASS**，29/29；24 组逐字节相等（见 §12） |

载荷约定（两个沙箱探针都要）：

```bash
tar czf tb_src.tgz -C corex-triton/src triton_corex
python tools/remote.py put tb_src.tgz /data/home/2024240384035/coffin-work/base/sandbox/work/tb/tb_src.tgz
```

沙箱探针一律：`set -u` 开头、**继承 runner 的 `CUDA_VISIBLE_DEVICES`**（不覆盖）、
开头先验卡（8×8 bf16 matmul）、坏卡 `exit 9`、每个 python 段套 `timeout --foreground -k 5`。

---

## 6. 元素级算子第一批（`src/corex_torch/_ops_elementwise.py`）

阶段 4 的第一个交付：8 个元素级算子跑在卡上，与 CPU 参照**逐字节相等**，带宽追平文档基线。
两个探针：`probes/170_eltwise.sh`（fp32 主表）、`probes/171_eltwise_dtype.sh`（fp16 / bf16）。

### 6.1 接口契约（`_device.py` 按这套调用，位置参数不能改）

```python
add(a_ptr, b_ptr, out_ptr, n, dtype, alpha=1.0)      # out = a + alpha*b
sub(a_ptr, b_ptr, out_ptr, n, dtype, alpha=1.0)      # out = a - alpha*b
mul(a_ptr, b_ptr, out_ptr, n, dtype)                 # out = a*b
div(a_ptr, b_ptr, out_ptr, n, dtype)                 # out = a/b
add_scalar(a_ptr, out_ptr, n, dtype, scalar)         # 同理 sub_scalar/mul_scalar/div_scalar
fill(out_ptr, n, dtype, value)
silu(a_ptr, out_ptr, n, dtype)
copy(src_ptr, dst_ptr, n, dtype)                     # 逐元素拷贝，只兜底 stride 非连续
```

- `ptr` 可以是 `int`（`tensor.data_ptr()`）、`ctypes.c_void_p`，或任何有 `data_ptr()` 的对象；
- `dtype ∈ {fp32, fp16, bf16}`；不支持的取值或编不出来的内核 → **`NotImplementedError`**，
  绝不静默降精度（fp32 编不出来才是真 bug，那种会抛 `RuntimeError`）；
- 只有连续内存、1 维展平（`off = pid*BLOCK + arange(BLOCK)`，不算任何 stride）；
- 每个函数内部按 `(内核种类, dtype, 算子, BLOCK, num_warps)` 缓存内核，**失败也缓存**
  （否则 fp16/bf16 每次调用都要白等一次 0.4 s 的编译重试）；
- 额外的 `block= / num_warps= / stream=` 都是 keyword-only，不影响上面的位置参数。

### 6.2 为什么这一层刻意不 import torch

`_device.py` 要把算子挂到 aten 上，但那要等 C++ shim（`docs/00-FACTS.md` §12：PrivateUse1
的分配器只能从 C++ 注册，纯 Python 拿不到 Tensor 的存储）。如果算子只写成「吃 torch 张量」，
shim 就位之前**一行都验证不了**。所以这一层的入参是设备指针 + 元素数 + dtype 字符串，
出参写进调用方给的指针：现在就能在真机上跑出数，将来 `_device.py` 只需要把 `data_ptr()`
转进来。`probes/170` 的每一段都打印 `torch_modules = 0`，证据在这。

### 6.3 内核与缓存

5 个内核覆盖 11 个 API：`binary`（constexpr `OP` 合并 add/sub/mul/div）、`scalar`、
`fill`、`silu`、`copy`。用 constexpr 合并而不是写 4 个内核：运行期零分支，cubin 各自独立
（`OP` 进缓存键），调试时 `COREX_DUMP_IR=...` 会把 kernel 名带进文件名。

实测首次编 11 个内核之后，一次 `add` 调用 **0.081 ms**（`170` §「缓存」），说明命中缓存时
不再编译。

### 6.4 正确性（fp32）：45/45 逐字节相等

口径：数据全部取 fp32/fp16/bf16 都能精确表示的值 —— `a ∈ [-12.0, 12.0]` 步长 0.25，
`b ∈ {0.5, 1.0, 2.0, 4.0}`（**2 的幂**：这样 `div` 的商不会出现「fp32 再舍到 fp16/bf16」
的二次舍入，除法也能按逐字节判定）。判定用 `struct.pack` 后的**字节串相等**，不是容差。

n 取三档：`4096`（BLOCK 整数倍）、**`4097`**（尾块只多 1 个有效元素）、**`4000`**（尾块少 96 个）。

| 用例 | n=4096 | n=4097 | n=4000 |
| --- | --- | --- | --- |
| `add` / `add(alpha=0.5)` / `add(alpha=2.5)` | 逐字节相等 | 逐字节相等 | 逐字节相等 |
| `sub` / `sub(alpha=2)` | 逐字节相等 | 逐字节相等 | 逐字节相等 |
| `mul` / `div` | 逐字节相等 | 逐字节相等 | 逐字节相等 |
| `add_scalar(0.5)` / `sub_scalar(2.0)` / `mul_scalar(0.25)` / `div_scalar(4.0)` | 逐字节相等 | 逐字节相等 | 逐字节相等 |
| `fill(0.5)` / `fill(-2.0)` | 逐字节相等 | 逐字节相等 | 逐字节相等 |
| `copy` | 逐字节相等 | 逐字节相等 | 逐字节相等 |
| `silu` | max_abs 9.537e-07 / max_rel 7.956e-07 | 同左 | 同左 |

`silu` 是唯一不按逐字节判的算子：`tl.exp` 落成 `ex2.approx.f32`
（`corex-triton/docs/00-FACTS.md` §11），近似值本身就与 CPU 的 `math.exp` 不同。
**这是硬件的近似指令，不是我们的改写错**，所以给它单独一个 1e-6 的相对误差门槛，日志里
两个数都打出来，不用「误差够小」糊过去。

### 6.5 性能（1M fp32）：先纠正一个口径错误

第一版探针直接在 1M fp32 上循环定时，报出 `add` **594 GB/s** —— 比同尺寸 D2D memcpy 还高，
不可能是显存带宽。原因：3 个缓冲共 12 MiB，**全在 32 MiB 的 L2 里**，量到的是 L2 带宽。

改成 `do_bench` 的口径之后才对：**每轮先用 256 MiB 的 fill 顶掉 L2，再用一对事件把内核包住**
（事件在 GPU 时间轴上，Python 派发不算进去），取 20 轮的中位数。三个数一起看才有意义：

| 算子 | 流量 | 内核本身（L2 刷） | 端到端墙钟（含派发） | L2 热（不刷） |
| --- | --- | --- | --- | --- |
| `add` | 12.00 MiB | **42.6 µs / 295.5 GB/s** | 58.0 µs / 216.9 GB/s（派发 35.8 µs） | 565.2 GB/s |
| `sub` | 12.00 MiB | 42.4 µs / 296.6 GB/s | 49.9 µs（派发 27.7 µs） | 565.7 GB/s |
| `mul` | 12.00 MiB | 42.6 µs / 295.2 GB/s | 50.3 µs（派发 28.2 µs） | 569.6 GB/s |
| `div` | 12.00 MiB | 43.0 µs / 292.3 GB/s | 50.2 µs（派发 27.5 µs） | 553.9 GB/s |
| `add_scalar` | 8.00 MiB | 30.5 µs / 275.3 GB/s | 46.0 µs（派发 25.5 µs） | 409.4 GB/s |
| `fill` | 4.00 MiB | 21.1 µs / 199.1 GB/s | 40.4 µs（派发 21.1 µs） | 217.7 GB/s |
| `silu` | 8.00 MiB | 30.8 µs / 272.2 GB/s | 42.6 µs（派发 22.0 µs） | 407.4 GB/s |
| `copy` | 8.00 MiB | 30.3 µs / 276.9 GB/s | 42.1 µs（派发 21.9 µs） | 414.6 GB/s |
| `memcpy D2D`（对照） | 8.00 MiB | 24.9 µs / 336.6 GB/s | — | 同左 |

三个参照并列（同一口径：1M fp32、3 份流量、L2 冷）：

| 实现 | GB/s | 与本表的关系 |
| --- | --- | --- |
| Triton 逐元素基线（`corex-triton/docs/00-FACTS.md` §14.3） | 295.0 | 我们 **295.5**，**1.00x**——这条基线被复现了 |
| ATen `elementwise_kernel`（同尺寸本探针实测） | 375.0 | 我们 0.79x（文档里记的 402 是另一个尺寸，只当天花板） |
| 厂商 torch `copy_`（DMA，`docs/00-FACTS.md` §9） | 471 | 与本表无关，但说明 `copy` 该走 memcpy |

三条可抄的结论：

1. **295 GB/s 就是这个尺寸的天花板**（同尺寸 DMA memcpy 只有 336.6，而我们比它慢 12%）。
   42.6 µs 里有约 10 µs 是这张卡的启动地板（空 kernel 9.9 µs，`corex-triton/docs/00-FACTS.md`
   §12），所以「295 GB/s」是**尺寸相关的**数，不是卡的上限。
2. **Python 派发开销 21–36 µs/次**，比内核本身还长。`_device.py` 每个 aten 算子调一次这层，
   小张量上派发会成为唯一瓶颈。要进一步压，只能把参数打包与 `cuLaunchKernel` 挪到 C++（shim
   顺手就能做）。
3. **BLOCK=1024/4 warps 是最优**（`170` §3.4 扫描：128→259.5、256→272.0、512→284.4、
   **1024→297.8**、2048→243.7、4096/8→263.9 GB/s）。2048 掉得厉害，说明这块 SRAM/tile
   的取整在 8 元素/线程之后反而变差。`DEFAULT_BLOCK=1024` 不是拍的。

附带一条离群值记录：第一版曾在 `div` 上报 581 µs / 21.6 GB/s，同配置复测 **43.1 µs /
292.0 GB/s**（本探针 `div 复测` 行），确认是测量噪声，不是「除法慢 26 倍」。

### 6.6 fp16：全通，33/33 逐字节相等（依赖 corex-triton 的 16 位模板）

| 项 | 结果 |
| --- | --- |
| fp16 全算子 × n=4096/4097/4000 | **33/33 逐字节相等** |
| `fill(fp16, 0.5)` 的字节 | `0038003800380038` = 每个元素 `0x3800`（fp16 的 0.5，小端）✔ |
| fp16 `silu` | max_rel **0.000e+00**（fp16 粒度把 `ex2.approx` 的误差吸收掉了） |
| fp16 带宽（1M 元素，L2 刷） | `add` 28.4 µs / **221.3 GB/s**（复测 27.9 µs / 225.2）；`fill` 137.4；`silu` 189.5；`copy` 199.9；memcpy 232.8 |

fp16 的 GB/s 比 fp32 低（221 vs 295）**不是变慢了**：流量减半（6.29 MB vs 12.58 MB），
而 10 µs 的启动地板是不变的，所以同样的时间内搬的字节少了一半。看「耗时」就明白：
fp32 42.6 µs vs fp16 28.4 µs。

**一个必须记下来的依赖关系**：本层的 fp16 能力**绑定在 corex-triton 的改写表版本上**。
第一次跑（payload 抓的是 00:30 的源码）fp16 编不过：

```
triton_corex.rewrite.PortError: 未登记的 PTX 操作数模板：
    'mov.u16 $0, 0x0; @$2 ld.global.b16 { $0 }, [ $1 + 0 ];'
```

第二次跑（payload 抓的是 00:31:45 的 `rewrite.py`，44343 B，把 `b8|b16|b32|b64` 与 `mov.u16`
前缀都认下来了）就全通了。**本层一行都没改**。所以：升级 corex-triton 时要重跑 `171`
确认 16 位模板还在；本层不该自己维护一份访存改写表。

### 6.7 bf16：6/33，根因是 `fp_round` 选不出来（按契约显式报不支持）

| 项 | 结果 |
| --- | --- |
| bf16 能跑通的 | `mul`、`copy`（两个算子 × 三档 n = 6 项，全部逐字节相等） |
| bf16 编不过的 | `add` / `sub` / `div` / `add_scalar` / `mul_scalar` / `fill` / `silu` |
| 报错原文 | `llc -march=bi 失败：LLVM ERROR: Cannot select: 0x…: bf16 = fp_round 0x…, TargetConstant:i64<0>…` |

根因很具体：这些内核把运算放在 **fp32** 里（`alpha` / `scalar` 都是 fp32 标量），最后
`.to(tl.bfloat16)` 收窄 —— 这个 `fp32 → bf16` 的舍入在 bi 后端**没有可选的指令**。
`mul` / `copy` 能过，是因为它们不需要那个转换（bf16 进、bf16 出）。

旁证（`171` §B）：**无掩码、且不做显式转换**的 bf16 内核能编能跑（cubin 4056 B，n=4096
逐字节相等）。所以 bf16 不是完全没路，但要走通得写一套「**全程 bf16、不出现 fp32 中间值**」
的内核，标量也得换成 bf16 传（`_triton._pack_arg` 目前没有 bf16 标量打包）。

按契约的处理：**显式 `NotImplementedError`**，消息里带 llc 的原文。绝不静默降精度 ——
降精度的错在推理里表现为「结果慢慢变差」，比编不过难查得多。

### 6.8 结论与下一步

| # | 结论 |
| --- | --- |
| 1 | 8 个元素级算子（11 个 API）在真机上全通，fp32 **45/45 逐字节相等**（含 4097/4000 尾块） |
| 2 | 带宽 **295.5 GB/s**，与 Triton 基线 1.00x；ATen 同尺寸 375.0 → 0.79x |
| 3 | fp16 全通（33/33），前提是 corex-triton 的 16 位访存模板在位 |
| 4 | bf16 明确不支持（`fp_round` 选不出来），按契约报 `NotImplementedError` |
| 5 | **派发开销 21–36 µs/次** 是目前最大的非内核成本，要压只能进 C++ |

下一步（按性价比）：① 把参数打包 + `cuLaunchKernel` 挪进 shim（顺手，收益直接）；
② bf16 若要支持，写 bf16-native 内核（避免 `fp_round`）；③ 大 n 下的真实上限要单独量
（本表是 1M 元素、L2 冷，启动地板占了 1/4）。

---

## 7. 规约 / softmax / 矩阵乘 / norm / index（阶段 4 第二批）

三个新模块：`_ops_reduce.py`（行内 sum/mean/max、全量 sum、softmax、rms_norm/layernorm）、
`_ops_matmul.py`（mm/bmm，`tl.dot` 走 FMA）、`_ops_index.py`（gather_rows）。
三个探针：`probes/180`（规约+softmax）、`probes/181`（矩阵乘）、`probes/182`（norm+index），
全部 `PASS`，全程 `torch_modules = 0`。

### 7.1 接口（与 `_device.py` 的契约，位置参数不能改）

```python
sum_last(a_ptr, out_ptr, rows, cols, dtype)          # out[r] = sum(a[r, :])
mean_last(a_ptr, out_ptr, rows, cols, dtype)
max_last(a_ptr, out_ptr, rows, cols, dtype)
sum_all(a_ptr, out_ptr, n, dtype)                    # out[0] = sum(a)
softmax_last(x_ptr, out_ptr, rows, cols, dtype)      # 逐行；out 可与 x 同指针
mm(a_ptr, b_ptr, c_ptr, M, N, K, dtype)              # 行主序 M×K · K×N
bmm(a_ptr, b_ptr, c_ptr, B, M, N, K, dtype)
rms_norm(x_ptr, w_ptr, out_ptr, rows, cols, eps, dtype)
layernorm(x_ptr, w_ptr, out_ptr, rows, cols, eps, dtype)
gather_rows(src_ptr, idx_ptr, out_ptr, rows, cols, dtype)   # + idx_dtype="i32"/"i64"
```

额外 keyword-only：`block / num_warps / stream`，矩阵乘另有 `block_m / block_n / block_k /
num_stages`。不支持的 dtype → `NotImplementedError`（带 llc 原文）。

### 7.2 三个「厂商 llc 解析不了」的坑（本批踩出来的，都建议上游补）

这一批的迭代几乎全花在这三条上，它们都不是我的算法问题，而是**上游 IR 里比 LLVM 16 新的东西**：

| # | 现象 | 根因 | 我的绕法 |
| --- | --- | --- | --- |
| 1 | `llc: error: expected value token`，指到 `%14 = phi <8 x float> [ … ], [ splat (float 0xFFF0000000000000), … ]` | `tl.full((BLOCK,), -inf, …)` / `other=-inf` 会生成**非零向量常量**，上游 LLVM 20 把它打成 `splat (float …)`，LLVM 16 没有这个语法。零常量打成 `zeroinitializer`，所以 `other=0.0` 一直没事 | max/softmax 的第一遍改成**钳位下标**（`tl.minimum(idx, cols-1)`，越界处重复读最后一个元素，对 max 无害）+ 初值取真实加载值；softmax 第二遍用 `tl.where(mask, exp(v-m), 0.0)` |
| 2 | `llc: error: expected type`，指到 `%62 = trunc nsw i64 %54 to i32` | LLVM 自己的循环归纳变量加宽（把 32 位 IV 提到 64 位再截回）在 LLVM 17 起了 `nsw` 标记。`corex-triton/rewrite.py` 已经会剥 `zext nneg`，但**没剥 `trunc nsw`** | `sum_all` 的 `chunk` 从 constexpr 改成**运行期参数**：constexpr 上界会让 LLVM 把 4 轮循环整个展开再诱发加宽；运行期上界不会 |
| 3 | `NameError: Cannot access global variable _OP_MAX from within @jit'ed function` | Triton 3.4 不允许 @jit 函数引用普通模块级全局（要么 `tl.constexpr(...)` 实例化，要么用字面量） | 内核里用字面量 `0/1/2`，并在模块顶部注明对应关系 |

**给 corex-triton 的建议（两条都能在 `rewrite.py` 里做，改完本仓库的绕法可以留着也无害）**：
`(trunc|zext|sext)\s+nsw` 一并剥掉（`nsw` 只是「不溢出」的承诺，剥掉语义更宽、安全）；
`splat (T <const>)` 展开成显式 `<N x T …>` 向量常量。

### 7.3 正确性（口径：与 CPU 双精度参照比；"逐字节"只用在能逐字节的地方）

**规约类不承诺逐字节相等**，判据是「误差 ≤ 8 × 顺序 fp32 累加自己的误差」——
后者是 CPU 上一一步步 `fp32` 累加与双精度的差，就是「累加顺序不同」的自然量级。
所以下表每个 sum/mean 都带一个「顺序 fp32 误差」和「限」两列。

| 用例 | max_abs | 顺序 fp32 误差 | 结论 |
| --- | --- | --- | --- |
| `sum_last` cols=512 / 1000 / 4097（dyadic 数据） | **0.000e+00** ×3 | 0（数据精确） | 通过。注意：这组数据（0.5 的倍数、和 < 2^24）的和**本来就是精确的**，所以 max_abs=0 只证明算法对，不证明累加顺序无误差 |
| `sum_last` cols=4097（**非 dyadic**，0.1 的倍数） | **2.403e-05** | 1.607e-04 | 通过，且比「顺序 fp32 累加」自己的误差**小 6.7 倍**（成对/树形累加更准） |
| `sum_last` cols=1000（非 dyadic） | 4.387e-06 | 3.843e-05 | 通过，小 8.8 倍 |
| `mean_last` cols=512 / 1000 / 4097 | 0 / 4.768e-09 / 1.398e-09 | — | 通过 |
| `max_last` cols=512 / 1000 / 4097 | **逐字节相等** ×3 | — | max 与顺序无关，必须精确 |
| `softmax_last` cols=512 / 1000 / 4097 | **1.149e-08 / 6.279e-09 / 1.961e-09** | — | 通过（门槛 1e-6）。另报「每行和偏离 1」= 9.352e-08 / 6.306e-08 / 8.402e-08，用来暴露 max 或 denom 算错 |
| `softmax_last` **原地**（out==x） | 与 out-of-place **逐字节相同** ×3 | — | 三遍扫描对每个 lane 是「先读自己、再写自己」，原地安全 |
| `sum_all` n=4096 / 100000 / 1M | **0.000e+00** ×3 | 0 | 通过 |
| `sum_all` n=1M，block=256 / 4096（G 被夹住的回归） | 0.000e+00 / 0.000e+00 | — | 通过（这两个变体是 §7.5 那三个 bug 的回归用例） |

矩阵乘分两档判，这是刻意的：

| 用例 | 判据 | 结果 |
| --- | --- | --- |
| **dyadic 输入**（±0.5/±1/±2）8³、17×19×23、37×53×29、64³、100×33×65 | 逐字节相等 | **5/5 全等** |
| bmm dyadic (2,8,8,8)、(3,13,17,11)、(4,32,16,24) | 逐字节相等 | **3/3 全等** |
| 非 dyadic 32³ / 17×19×23 / 37×53×29 | max_abs ≤ 8×顺序 fp32 误差 | 5.770e-07 / 4.673e-07 / 6.342e-07（限 9.224e-06 / 6.534e-06 / 8.061e-06） |
| 256³ / 512³ 抽查 4 个元素 | 相对误差（只取 \|ref\|>1e-3 的元素） | 8.385e-07 / 2.036e-06 |

> 为什么 dyadic 那一档值钱：fp32 累加在 dyadic 输入上是**精确**的，所以「逐字节相等」可以当
> 硬判据用 —— 它抓的是**布局/索引错**（行主序算错一维、批量偏移算错），那才是矩阵乘最容易
> 出的错。§7.5 的第一个 bug 就是被这一档抓出来的，而当时小尺寸全绿。

norm / index：

| 用例 | 结果 |
| --- | --- |
| `rms_norm` cols=512 / 1000 / 4097 | max_rel **1.423e-07 / 1.112e-07 / 1.621e-07** |
| `layernorm` 同上 | max_rel **1.722e-07 / 1.658e-07 / 1.908e-07** |
| 单遍方差（`E[x²]-E[x]²`）vs 两遍方差的差 | 0 / 6.821e-13 / 3.467e-11 —— 这批数据上单遍算法没有可测代价；「均值远大于标准差」的输入才会有相消误差（`_layernorm_kernel` 注释里写了） |
| 常量行：`layernorm` | 输出**全 0**（精确） |
| 常量行：`rms_norm` | 与 `w` 的 max_abs 6.819e-08 |
| `gather_rows` 7×1000 / 5×4097 / 33×512（乱序 + 重复下标） | **逐字节相等**，i32 与 i64 下标都是 |

### 7.4 性能

矩阵乘（口径：每轮刷 256 MiB L2、事件中位数；`flops = 2MNK`）：

| 尺寸 | 耗时 | GFLOP/s | 对照 |
| --- | --- | --- | --- |
| mm 256³ | 177.7 µs | **188.8** | Triton FMA 基线 105 → **1.80x**；厂商 TCU 1577 → 0.120x（只当天花板，见 §9 的口径约定） |
| mm 512³ | 1267.5 µs | **211.8** | — |
| bmm B=8, 256³ | 1271.2 µs | 211.2 | 与 mm 同量级，批偏移没有额外代价 |

分块扫描（mm 256³，定默认值用）：

| (BM,BN,BK) | warps | stages | GFLOP/s |
| --- | --- | --- | --- |
| **(16,16,16)** | **1** | **1** | **189.1** ← 默认值 |
| (16,16,32) | 1 | 1 | 150.8 |
| (16,16,32) | 1 | 3 | 153.5 |
| (32,32,32) | 1 | 1 | 141.3 |
| (32,32,32) | 4 | 1 | 100.9 |
| (64,64,16) | 4 | 1 | 141.8 |
| (16,32,32) | 2 | 1 | 116.2 |
| (16,16,64) | 2 | 1 | 126.3 |

结论与 `corex-triton/docs/00-FACTS.md` §15.2 一致：**FMA 路径吃「更多并发块 + 更少每块工作」**，
与矩阵单元正好相反（(16,16,16) 比 (32,32,32,4 warps) 快 87%）。多级流水（stages=3）没有收益，
因为这张卡没有 cp.async。

规约 / softmax / norm / index（1M 量级，L2 刷）：

| 算子 | 形状 | 耗时 | GB/s | 流量口径 |
| --- | --- | --- | --- | --- |
| `sum_last` | 1024×1000 | 28.6 µs | 143.3 | 读 3.91 MiB + 写 4 KiB |
| `mean_last` | 1024×1000 | 28.4 µs | 144.2 | 同上 |
| `max_last` | 1024×1000 | 27.0 µs | 151.9 | 同上 |
| `softmax_last` | 1024×1000 | 42.5 µs | 192.7 | 读 1 + 写 1（**源码读 3 遍**，行 4 KB 留在 L1/L2，DRAM 上仍 1 遍） |
| `sum_all` | n=1M | 34.1 µs | 122.9 | 读 4.00 MiB（两段式：256 个部分和 + 1 次归约） |
| `sum_all` | n=4M | 74.1 µs | **226.4** | 同上 —— 尺寸翻 4 倍带宽接近翻倍，说明 1M 那档仍被启动地板压着（同 §6.5 的结论） |
| `rms_norm` | 1024×1000 | 38.2 µs | 321.3 | 读 x + 读 w + 写 out |
| `layernorm` | 1024×1000 | 39.0 µs | 315.1 | 同上 |
| `gather_rows` | 1024×1000（表 4096 行） | 34.5 µs | 237.3 | 读 1 行 + 写 1 行 |

（norm/index 这一组跑了两遍，两次差异 ≤ 1%，属同一噪声量级。）

### 7.5 我在这一批里修掉的三个「静默算错」型 bug

写在这里是因为**它们都不会崩、只会算错**，而且都被不同的判据抓出来：

| # | bug | 为什么不显形 | 被谁抓到 |
| --- | --- | --- | --- |
| 1 | `b_ptrs += BLOCK_K` 应该是 `+= BLOCK_K * N`（b 是 K×N，沿 k 走 BLOCK_K **行** = BLOCK_K*N 个元素） | `K ≤ BLOCK_K` 时循环只跑一轮、指针根本不递增 → 小尺寸全对。写成 `+= BLOCK_K` 只在 N==1 时才对 | **dyadic + K=64** 那一档（`mm 64x64x64` 从全绿变成 max_abs=2.5） |
| 2 | `sum_all` 的 G 必须 ≤ BLOCK | 第二段是单程序按 BLOCK 装载部分和再 `tl.sum`，G > BLOCK 时后面的部分和被 `offs < g` 掩码挡掉 → 少算 | 我原来的 `_MAX_PARTS` 恰好等于 `BLOCK`（都是 1024）所以正常路径全对，改 `block=` 参数才暴露；现在 G 同时受 `min(_MAX_PARTS, block)` 约束 |
| 3 | `chunk` 必须是 BLOCK 的整数倍 | 内核按 `for j in range(0, chunk, BLOCK)` 扫，最后一个窗口会向上取整到 BLOCK；chunk 不是整数倍时相邻程序**读重叠的一段**，等于重复累加 | `sum_all n=100000`（max_abs 17.5）与 `n=1M block=4096`（max_abs 187）。修法：chunk 先按 G 算、再向上取到 BLOCK 的整数倍 |

教训：**「大尺寸 + 非整除」这两个条件必须同时出现在用例里**。第 1 个 bug 在「小尺寸」下隐身，
第 3 个在「整除」下隐身。

### 7.6 bf16：四个算子显式 `NotImplementedError`，`gather_rows` 反而能跑

| 算子（bf16） | 结果 |
| --- | --- |
| `sum_last` / `softmax_last` / `rms_norm` / `mm` | `NotImplementedError`，消息里带 llc 原文：`LLVM ERROR: Cannot select: … f32 = fp_extend …`（这批内核读 bf16 要**加宽**到 fp32，bi 后端选不出 `fp_extend`；与元素级那批的 `fp_round`（收窄）是同一个缺口的两个方向） |
| `gather_rows` | **编得过并跑通** —— 它是纯搬运，不做 bf16↔fp32 转换，所以不碰这两个指令 |

契约执行情况（`probes/180` §3）：**5/5**，四个抛 `NotImplementedError`、`gather_rows` 正常返回。
没有任何一条静默降精度。

### 7.7 结论与下一步

| # | 结论 |
| --- | --- |
| 1 | 规约/softmax/norm/index/矩阵乘全部在真机上跑通，三个探针 `PASS`，全程无 torch |
| 2 | **矩阵乘 256³ 188.8 GFLOP/s**：是 Triton FMA 基线（105）的 **1.80x**，是厂商 TCU（1577）的 **0.12x** —— 后者是矩阵单元的成绩，接 TCU 之前不可能追上（要落到 `TritonGPU→LLVM` 那层，见 `corex-triton/docs/00-FACTS.md` §13） |
| 3 | 规约类误差比「朴素顺序 fp32 累加」**小 6.7–8.8 倍**（树形/成对累加更准），max 与 gather 是逐字节精确 |
| 4 | 三个「上游 IR 比 LLVM 16 新」的坑（§7.2）：`splat` 非零向量常量、`trunc nsw`、@jit 看不到普通全局。前两个建议 corex-triton 在 `rewrite.py` 里补，补完我这边的绕法留着也无害 |
| 5 | bf16 见 §8：bi 后端不认标准 `fpext`/`fptrunc`，只认自己的 intrinsic；改写层接上之后**元素级已 33/33 逐字节相等** |

下一步：① `mma`/TCU 的 lowering（决定矩阵乘能不能从 189 走到四位数）；
② 规约类目前是「一行一个程序」，`rows` 很大（比如 10 万行）时并行度够、但每程序只搬 4 KB，
启动地板会吃掉效率，可以改成「一个程序处理多行」；③ softmax 三遍扫描可换成在线（online）算法
把读次数压到 1。

---

## 8. bf16 通路（`src/corex_torch/_bf16.py` + 各算子的算术位置）

### 8.1 问题：bi 后端不认标准 LLVM 的转换指令

Triton 生成的 bf16 内核里，bf16↔fp32 的转换是一条**标准 LLVM 指令**：

| 方向 | Triton 发的 IR | 厂商 llc（LLVM 16 + bi 后端）的反应 |
| --- | --- | --- |
| 展宽 bf16→f32 | `fpext bfloat %x to float` / `fpext <8 x bfloat> %x to <8 x float>` | `LLVM ERROR: Cannot select: f32 = fp_extend` |
| 收窄 f32→bf16 | `fptrunc float %x to bfloat` / `<8 x float> → <8 x bfloat>` | `LLVM ERROR: Cannot select: bf16 = fp_round` |

所以问题不在"bf16 算术"，而在**两条转换指令**。bi 后端其实**有**转换指令，只是不叫这两个名字。

### 8.2 解法：在 biir 进 llc 之前把名字换掉（`_bf16.rewrite_bf16_conversions`）

厂商自己的 CUDA 头（`cuda_bf16.hpp`）走的是 intrinsic，落到：

```
%n = call float  @llvm.bi.bfloat2float(bfloat %x)
%n = call bfloat @llvm.bi.float2bfloat(float %x)
```

于是本仓库在 `_triton.CorexTorchBackend.make_cubin`（biir → llc 之间）做纯文本改写，**不动
corex-triton 一个字节**，命中数写进 `metadata['corex_rewrite_counts']['bf16_conversions']`。
内核侧因此可以写**自然形式**：

```python
f = tl.load(x_ptr + off).to(tl.float32)   # → @llvm.bi.bfloat2float
tl.store(y_ptr + off, f.to(tl.bfloat16))  # → @llvm.bi.float2bfloat
```

**⚠ 两个必须知道的坑（都是本仓库踩出来的）**

| # | 坑 | 证据 |
| --- | --- | --- |
| 1 | **标量 intrinsic 是真实现，向量重载不是**。`@llvm.bi.bfloat2float.v8bf16` 这种带后缀的向量形态 llc **收下并 rc=0**，但它没有 lowering 实现，只剩一个外部符号，被 `lld --no-undefined` 拦下：`lld: error: undefined symbol: llvm.bi.bfloat2float.v8bf16` | `probes/195` 三级对照：V1 原样 `llc=FAIL`、V2 换向量 intrinsic `llc=OK lld=FAIL`、V3 再标量化 `llc=OK lld=OK elf=6104 B` |
| 2 | **判「intrinsic 支不支持」不能只看 `llc` 返回码**。LLVM 允许把未实现的外部调用原样留在目标文件里。必须跑 `llc` **加** `lld --no-undefined`（`corex-triton` 的 `make_cubin` 正好带这个 flag） | `probes/194` 只看 llc，得出「向量重载可用」的**错误**结论；`probes/195` 用链接判据把它纠正为「只认声明、没实现」 |

结论：**向量形式必须标量化**（`extractelement` → 标量 intrinsic 调用 → `insertelement` 链），
标量化之后 llc 与 lld 双过。

### 8.3 各算子的算术位置（这是设计决定表，不是实现细节）

| 算子 | 算术在哪 | bf16 转换出现在哪 | 为什么 |
| --- | --- | --- | --- |
| `add/sub/mul/div`、`*_scalar`、`fill`、`copy` | 内核里自然写，Triton 在 **fp32** 里算，只有输出一次舍回 bf16 | 展宽（读）+ 收窄（写） | 逐元素没有累加，一次舍入就是 IEEE RNE，与 CPU bf16 参照**逐字节相等** |
| `silu` | **fp32**（`ex2.approx.f32` 没有 bf16 版本） | 展宽 + 收窄 | bf16 下做不出 exp |
| `gather_rows` | 纯搬运，不做任何转换 | **无** | 没有算术就没有舍入，逐字节 |
| `sum_last/mean_last/max_last/sum_all` | **fp32 累加** | 向量展宽（读）+ 标量收窄（写） | 纯 bf16 累加 1000 项相对误差到 1e-2 量级，模型直接废 |
| `softmax_last` | **fp32**（max / exp / 归一） | 同上 | 行和偏离 1 要留在 1e-7 量级 |
| `rms_norm/layernorm` | **fp32**（平方和/方差 + 归一） | 同上 | 同上 |
| `mm/bmm` | **fp32 累加**，bf16 进 bf16 出 | 向量展宽 + 向量收窄 | 工业界标准；纯 bf16 累加在 K=1024 时相对误差约 1e-1，没有可用性 |

### 8.4 为什么内核侧**不该**自己用位运算造转换（两个死点）

bf16 就是 fp32 的高 16 位，所以「u16 缓冲区 + 整数位运算」在算术上完全可行（`probes/190` 证实
展宽方向 `fp_extend=0`、求和正确）。但这条路在**向量化之后**撞上两种「比 LLVM 16 新」的语法，
两种都是上游 LLVM 17+ 才有、厂商 llc 16 解析不了的：

| # | 现象 | 触发点 | 探针 |
| --- | --- | --- | --- |
| 1 | `llc: error: expected type`，指到 `%48 = trunc **nuw** i32 %47 to i16` | 逐元素 `.to(tl.uint16)` 收窄；LLVM 17 起 `trunc` 有了 `nuw`/`nsw` 标记 | `probes/190`、`probes/191` §1 |
| 2 | `llc: error: expected value token`，指到 `%111 = lshr <2 x i32> %110, **splat (i32 16)**` | 对**向量**做 `>> 16` / `<< 16`（向量常量会被打成 `splat`）；标量移位没事 | `probes/191` §4 |

也就是说：**位运算是把「一条缺指令」换成「两条上游新语法」**，一共要打三个补丁才够，而且每换一种
写法（标量/向量、i16/i32、成对打包）都可能冒出新的。相比之下 intrinsic 是**一条**能一次解决、
且数值与硬件一致的替换。这就是这套改写层存在的理由。

判据上的教训也一致：`probes/191` 里「成对打包的 `fill` 过了、`add` 挂了」——
过的那个是**标量**移位，挂的那个是**向量**移位，说明这类问题的边界不在算法，在**向量化**。

### 8.5 体检：改写命中数与「进 llc 的那份 IR」

口径很重要：`kernel.asm['biir']` 是**改写前**的文本，`fpext`/`fptrunc` 出现在里面是正常的
（`make_biir` 的产物）。要看的是「**再跑一遍改写器之后**还剩几条」——那才是喂给 llc 的文本。

`probes/192` §6 实测（bf16 元素级内核）：

```
('binary','bf16',0,…) cubin= 7192  bf16_conversions=24  改写后剩余{}=空
('binary','bf16',1,…) cubin= 7216  bf16_conversions=24  改写后剩余{}=空
('binary','bf16',2,…) cubin= 7008  bf16_conversions= 0  改写后剩余{}=空   ← mul 天然是 bf16 算术，无需转换
('binary','bf16',3,…) cubin= 7608  bf16_conversions=24  改写后剩余{}=空
('copy',  'bf16',…)   cubin= 5752  bf16_conversions= 0  改写后剩余{}=空   ← 纯搬运
('fill',  'bf16',…)   cubin= 4200  bf16_conversions= 1  改写后剩余{}=空
('scalar','bf16',0,…) cubin= 6048  bf16_conversions=16  改写后剩余{}=空
('scalar','bf16',2,…) cubin= 6048  bf16_conversions=16  改写后剩余{}=空
('silu',  'bf16',…)   cubin= 6712  bf16_conversions=16  改写后剩余{}=空
elementwise 合计 bf16_conversions=121，改写后可疑语法=0
```

`fpext / fptrunc / trunc nuw / trunc nsw / splat / fp_extend / fp_round` 在改写后的文本里
**全部为 0**。

`probes/192` §6 全量（bf16 内核，含规约/矩阵乘/index）：

```
--- elementwise：11 个内核 ---  (bf16 部分)
  ('binary','bf16',0..3)  cubin=7008..7608  bf16_conversions=24/24/0/24
  ('scalar','bf16',0/2)   cubin= 6048       bf16_conversions=16/16
  ('fill',  'bf16')       cubin= 4200       bf16_conversions= 1
  ('silu',  'bf16')       cubin= 6712       bf16_conversions=16
  ('copy',  'bf16')       cubin= 5752       bf16_conversions= 0   ← 纯搬运
  elementwise 合计 bf16_conversions=121，改写后可疑语法=0
--- reduce：5 个内核 ---
  ('softmax',  'bf16')    cubin=13000  bf16_conversions=26
  ('layernorm','bf16')    cubin=11992  bf16_conversions=25
  ('rms_norm', 'bf16')    cubin=11120  bf16_conversions=25
  ('row_reduce','bf16',0) cubin= 6104  bf16_conversions= 2
  ('row_reduce','bf16',2) cubin= 5960  bf16_conversions= 3
  reduce 合计 bf16_conversions=81，改写后可疑语法=0
--- matmul：3 个内核 ---
  ('bf16',16,16,16,False,1,1) cubin=11536  bf16_conversions=17
  ('bf16',16,16,16,True, 1,1) cubin=11816  bf16_conversions=17   ← bmm
  ('fp32',16,16,16,False,1,1) cubin=11208  bf16_conversions= 0
  matmul 合计 bf16_conversions=34，改写后可疑语法=0
--- index：1 个内核 ---
  ('bf16','i32',1024,4)   cubin= 5920   bf16_conversions= 0   ← 纯搬运
全部 bf16_conversions 命中 236，改写后可疑语法 0
```

注意 `bf16_conversions=0` 的那几个内核不是"改写没生效"，而是**它们本来就不需要转换**：
`mul`（bf16 乘 bf16 存 bf16）、`copy`、`gather_rows` 全程不出现 fp32 值；fp32 内核自然也不需要。

### 8.6 正确性：元素级逐字节、累加类与「顺序 bf16 累加」对照

**元素级（`probes/192` §2）**：`add / add(alpha=0.5) / sub / mul / div / add_scalar / mul_scalar /
fill(0.5) / fill(-2.0) / copy` × n=4096 / **4097** / **1000** = **33/33 全部逐字节相等**，
参照是「先 `to(bfloat16)` 再按 bf16 语义算」的 CPU 值。`silu` 的 max_abs 也是 **0.000e+00**——
bf16 的 8 位尾数把 `ex2.approx` 的误差整个吸收了（它的相对粒度是 `2^-9 ≈ 2e-3`）。

**累加类（§3）**：输入先舍成 bf16，参照用双精度。判据不是「相等」，而是
「误差 ≤ 行和量级的一个 bf16 ulp」，并与**纯 bf16 顺序累加**（每一步都舍回 bf16，即"如果内核真按
bf16 硬算"的样子）对比——那一列才是"为什么必须 fp32 累加"的证据。

| 用例 | cols | max_abs / max_rel | 顺序 bf16 累加的误差 | 我们好多少 |
| --- | --- | --- | --- | --- |
| `sum_last` | 1000 | 5.000e-01 | **1.235e+02** | **247×** |
| `sum_last` | 512 | 5.000e-01 | **6.450e+01** | **129×** |
| `sum_last` | 4097 | **0.000e+00** | **4.750e+02** | 精确 |
| `max_last` | 1000 / 512 / 4097 | **逐字节相等 ×3** | — | max 与顺序无关 |
| `softmax_last` | 1000 / 512 / 4097 | 1.213e-04 / 3.214e-04 / 5.613e-05（行和偏离 1：8.33e-04 / 1.26e-03 / 1.31e-03） | — | bf16 输出粒度 |
| `rms_norm` | 1000 / 512 / 4097 | 3.828e-03 / 3.773e-03 / 3.764e-03 | — | 同上 |
| `layernorm` | 1000 / 512 / 4097 | 3.837e-03 / 3.859e-03 / 3.801e-03 | — | 同上 |

`sum_last` 的 5.000e-01 不是误差失控，而是**输出本身是 bf16**：行和量级 ~1e3，bf16 在 1e3 附近的
ulp 就是 4~8，0.5 是一个 ulp 的一小部分。同一个数在 fp32 输出时是 `2.403e-05`（§7.3）。
`§3 15/15`。

**矩阵乘 / index（§4）**：

| 用例 | 判据 | 结果 |
| --- | --- | --- |
| `gather_rows` 7×1000 / 5×4097 | 逐字节相等 | **True / True** |
| `mm` 32×32×**K=256** | max_rel（\|ref\|>1e-2） | **3.673e-03**；纯 bf16 顺序累加是 **2.292e+00** → **好 624×** |
| `mm` 16×16×**K=1024** | 同上 | **3.622e-03**；纯 bf16 顺序累加是 **7.663e-02** → **好 21×** |
| `bmm` B=2 16×16×128 | 同上 | **0.000e+00**（逐字节相等） |

K 从 256 涨到 1024，我们的相对误差基本不动（3.67e-03 → 3.62e-03），因为累加在 fp32 里做；
而「纯 bf16 累加」的误差随 K 增长（2.3 → 0.077 的绝对值差异来自数据量级不同，但趋势是单调累积）。
**这就是 §8.3 里"累加必须 fp32"的实测依据。**

### 8.7 性能：bf16 vs fp32（同尺寸并列）

| 算子 | dtype | 耗时 | 流量 | 吞吐 | 口径 |
| --- | --- | --- | --- | --- | --- |
| `add` n=1M | fp32 | 42.6 µs | 12.00 MiB | **295.1 GB/s** | L2 每轮刷 |
| `add` n=1M | bf16 | 28.4 µs | 6.00 MiB | **221.4 GB/s** | L2 每轮刷 |
| `mm` 256³ | fp32 | 176.6 µs | — | **190.0 GFLOP/s** | L2 每轮刷 |
| `mm` 256³ | bf16 | 174.2 µs | — | **192.7 GFLOP/s** | L2 每轮刷 |

**两个不同的结论，别混着说：**

1. **逐元素（带宽受限）**：**流量减半，但 GB/s 没有翻倍**——42.6 → 28.4 µs 只快了 33%。
   原因是**启动地板不随 dtype 变小**：这 28.4 µs 里有约 10 µs 是空 kernel 开销
   （`corex-triton/docs/00-FACTS.md` §12：9.9 µs）。扣掉地板后 bf16 的有效带宽约 310 GB/s，
   与 fp32 的 380 GB/s 同量级，两者都贴着 `probes/170` 量出来的平台上限。
   **bf16 带宽数字低不是退化，是分母里有个固定成本。**
2. **矩阵乘（算力受限）**：bf16 几乎没有收益（190.0 → 192.7 GFLOP/s，差 1.4%）。因为这条路径是
   **fp32 FMA** 打满的，瓶颈在乘加吞吐而不是访存；bf16 只省了输入输出的带宽（256³ 时
   a/b/c 一共 768 KiB，本来也全在 L2 里）。**要真正提速必须接矩阵单元（TCU），不是换 dtype。**

### 8.8 结论

| # | 结论 |
| --- | --- |
| 1 | bf16 之前不可用的根因**不是算术**，而是两条转换指令的名字：bi 后端只认 `llvm.bi.bfloat2float` / `llvm.bi.float2bfloat` |
| 2 | 在 biir→llc 之间做文本改写即可，内核侧写自然形式，**不用动 corex-triton** |
| 3 | 元素级 bf16 与 CPU bf16 参照**逐字节相等**（33/33，含 1000/4097 尾块）；`silu` 的 max_abs 也是 0 |
| 4 | 累加类（sum/softmax/norm/mm）按 §8.3 保留 **fp32 累加**：`sum_last` 比纯 bf16 累加准 **129–247 倍**，`mm` K=256 时准 **624 倍**、K=1024 时准 **21 倍** |
| 5 | 向量重载**只认声明没实现**，必须标量化；判据必须是 `llc` + `lld --no-undefined` 两级（`probes/194` 的「可用」结论是错的，`probes/195` 已更正） |
| 6 | 内核侧不要自己用位运算造转换：那会把「一条缺指令」换成 `trunc nuw` + `splat` 两条上游新语法（§8.4） |
| 7 | 性能：逐元素 bf16 快 33%（流量减半，但启动地板不变，所以 GB/s 不翻倍）；矩阵乘 bf16 无收益（算力受限，不是带宽受限） |

---

## 9. 第 5 批算子：relu / rsqrt / gelu / masked_fill / max.dim+argmax / log_softmax

`probes/196_new_ops.sh` **PASS**：§1 **63/63**、§2 **36/36**、bf16 改写命中 **130**、改写后可疑语法 **0**。

### 9.1 接口

```python
relu(a_ptr, out_ptr, n, dtype)                       # relu_ 就是 out_ptr == a_ptr
rsqrt(a_ptr, out_ptr, n, dtype)
gelu(a_ptr, out_ptr, n, dtype, mode="erf"|"tanh")
masked_fill(a_ptr, mask_ptr, out_ptr, n, dtype, value)   # mask_ptr 按 *i8 读，m != 0 判真
max_dim(a_ptr, val_ptr, idx_ptr, rows, cols, dtype)      # 值 + int64 下标
argmax_last(a_ptr, idx_ptr, rows, cols, dtype)           # 只出下标（与 max_dim 共用内核）
log_softmax_last(x_ptr, out_ptr, rows, cols, dtype)      # 与 softmax_last 共用内核
```

### 9.2 判据表（口径按"能不能逐字节"分类，不搞一刀切）

| 算子 | 判据 | 实测 | 为什么这个判据 |
| --- | --- | --- | --- |
| `relu` / `relu_` | **逐字节相等** | 9/9 组（3 dtype × 3 长度）全等 | `max(x,0)` 没有舍入 |
| `masked_fill` / `masked_fill_` | **逐字节相等** | 9/9 全等 | 选值不是算术；bf16/fp16 的 `value` 收窄是与 torch 同一次 RNE |
| `max_dim` 的值 | **逐字节相等** | 9/9 全等 | 输出就是输入里的某个元素，没有运算 |
| `max_dim` / `argmax` 的下标 | **完全相等** | 9/9 全对（含**重复最大值**用例） | 专门造了 `row[0] = row[1]`、`row[cols-1] = row[1]` 来验 aten 的「取第一个」语义 |
| `rsqrt` | max_rel | fp32 **0.000e+00**、fp16/bf16 0 | 见 §9.3 的说明（测试值特殊，别当普遍结论） |
| `gelu(erf)` | max_abs/max_rel | fp32 max_abs **5.960e-08**（≈1 ulp）、max_rel 1.020e-06；fp16/bf16 max_abs 0 | 走 libdevice `erf`，是超越函数，不能要求逐字节 |
| `gelu(tanh)` | max_abs/max_rel，**参照同一条公式** | fp32 max_abs **3.168e-08**、max_rel 2.037e-06 | 若拿精确 erf 版当参照，测到的是**近似误差**（实测 4.399e-04）而不是实现误差——这两件事必须分开 |
| `log_softmax` | max_abs + `exp` 回代的行和 | fp32 **1.036e-06**（行和偏离 1 仅 2.7e-07）；fp16 6.5e-3；bf16 5.7e-2 | 累加 + 超越函数；另加一条 `exp(log_softmax) == softmax` 的一致性检查（max_abs 3.9e-09 / 8.8e-06 / 2.4e-04） |

### 9.3 两个「换了实现」的决定（都是被工具链逼的，记在这里免得下次重试）

**① `rsqrt` 不能用 `tl.math.rsqrt`** —— 它在 Triton 3.4 里落成

```
%59 = tail call float @llvm.nvvm.rsqrt.approx.ftz.f(float %43)
```

而 corex-triton 的改写表**没有登记**这个 NVVM intrinsic，直接抛
`PortError: 未登记的 NVVM intrinsic 调用`。所以实现改成 `1.0 / tl.sqrt(x)`：`tl.sqrt` 是本仓库
`rms_norm`/`layernorm` 一直在用、已验证的路径（`probes/180`/`182`/`192`），代价是多一次
`div.rn` 舍入。**要更快的 rsqrt 必须给 corex-triton 补一条 `nvvm.rsqrt.approx.ftz.f` 模板。**

> ⚠ 数据提醒：本次 `rsqrt` 测出 `max_rel = 0`，但测试值只有 `[0.25,0.5,1,2,4,8,16,32]` 这类
> 特殊值（其中 `1/√2`、`√2` 只有两个无理数），**不能**据此说 `1/sqrt` 普遍与正确舍入值逐位相同。
> 它说明的是：在这组值上没观测到 `sqrt.approx` 的偏差（疑似 `sqrt` 在这张卡上是正确舍入的
> `sqrt.rn`，未单独验证）。

**② `gelu` 的 tanh 版不能调 `tl.math.tanh`** —— Triton 3.4 的 `tl.math` 里**没有** `tanh`
（实测 `hasattr(tl.math,"tanh")==False`，而 `erf`/`rsqrt`/`log` 都有）。所以 tanh 用
`tanh(z) = 2/(1+exp(-2z)) − 1` 现推：只依赖已经验证过的 `tl.exp`（`ex2.approx.f32`），
不必再赌 `libdevice` 模块在 jit 作用域里能否解析。多出的两次舍入被 tanh 近似本身
（4.4e-04）完全盖住。

### 9.4 `_log_softmax` 为什么加 constexpr 而不是第二个内核

`aten::_log_softmax` 与 `aten::_softmax` 只差**最后一遍写什么**：
`x - m - log(denom)` vs `exp(x−m)/denom`。前三遍（行最大值、`sum(exp)`）完全相同。
所以 `_softmax_kernel` 加一个 `LOG: tl.constexpr` 分叉，**共用一条内核**；`op` 槽位借来区分
（0=softmax，1=log-softmax），缓存键天然分开。两份内核会让「改一处忘一处」变成必然
（比如以后换 online 算法要改两处）。

`log_softmax` 用的是 aten 的**稳定形式**（先减行最大值再取 log），**不是** `log(softmax(x))`
——后者会先把小概率下溢到 0 再取 log，得到 `-inf`。

### 9.5 bf16 体检（本轮新增内核）

```
('gelu','bf16',0)        cubin=10448  bf16_conversions=16  改写后可疑语法={}
('gelu','bf16',1)        cubin= 7264  bf16_conversions=16  改写后可疑语法={}
('masked_fill','bf16')   cubin= 7288  bf16_conversions= 1  改写后可疑语法={}
('relu','bf16')          cubin= 5984  bf16_conversions=16  改写后可疑语法={}
('rsqrt','bf16')         cubin= 6376  bf16_conversions=16  改写后可疑语法={}
('max_dim','bf16',0/1)   cubin= 6896/7064  bf16_conversions= 6/7
('softmax','bf16',0/1)   cubin=13000/13264 bf16_conversions=26/26
新内核 bf16_conversions 合计 130，改写后可疑语法 0
```

bf16 在这批算子上**全部真支持**（没有一条走 `NotImplementedError`）：`relu`/`masked_fill` 逐字节，
`gelu`/`rsqrt` 在 bf16 粒度内精确（max_abs ≤ 5.9e-09），`max_dim`/`argmax` 的值与下标都精确。

### 9.6 结论

| # | 结论 |
| --- | --- |
| 1 | 6 组算子全部在真机跑通，`probes/196` PASS（63/63 + 36/36），覆盖 fp32/fp16/bf16 × 1000/4096/4097 |
| 2 | 「能逐字节就逐字节」：`relu`/`masked_fill`/`max_dim` 的值与下标、`argmax` 的下标**全部逐字节** |
| 3 | 「不能逐字节就说清为什么」：`gelu` 的两种形式各自对各自的公式比（精确版 max_abs 5.96e-08；tanh 版的近似误差 4.4e-04 单列，不混进实现误差） |
| 4 | 两个工具链坑：`tl.math.rsqrt` → 未登记的 `nvvm.rsqrt.approx.ftz.f`；`tl.math` 里**没有** `tanh`。两者都绕开了，但都值得回填 corex-triton/FACTS |
| 5 | `_log_softmax` 与 `_softmax` **共用一条内核**（`LOG` constexpr），`max.dim`/`argmax` 也共用一条（`WRITE_VAL` constexpr）——batch 越大越不该复制内核 |

---

## 10. 最后一批内核：`row_var` / `row_std` / `sum_axis` / `mean_axis`

`probes/197_var_axis.sh` **PASS**：§1 **27/27**、§2 **18/18**、bf16 改写命中 **17**、改写后可疑语法 **0**。

### 10.1 接口

```python
row_var(a_ptr, out_ptr, rows, cols, dtype, correction)   # correction: 1.0 样本 / 0.0 总体
row_std(a_ptr, out_ptr, rows, cols, dtype, correction)   # 与 row_var 共用内核（SQRT_OUT）
sum_axis(a_ptr, out_ptr, outer, red, inner, dtype)       # 单个中间维，输出 (outer, inner)
mean_axis(a_ptr, out_ptr, outer, red, inner, dtype)      # 与 sum_axis 共用内核（MEAN）
```

### 10.2 `row_var` 为什么必须两趟（附三组对照数字）

一趟恒等式 `var = (Σx² − n·mean²) / (n − c)` 代数上没错，但 `Σx²` 与 `n·mean²` 是两个
**量级相近的大数**，相减把有效位整段抵掉。`probes/197` 用「均值 1e6、标准差 0.4」的行实测：

| cols | 我们的两趟（max_abs） | 一趟公式（双精度，只剩公式误差） | 一趟公式（fp32 顺序累加，真实内核会拿到的数） | 两趟好多少 |
| --- | --- | --- | --- | --- |
| 1000 | **7.458e-11** | 1.101e-02 | **6.986e+06** | **9.4e16 倍** |
| 512 | **2.149e-06** | 2.150e-06 | **3.283e+06** | **1.5e12 倍** |
| 4097 | **3.352e-08** | 9.308e-04 | **1.901e+07** | **5.7e14 倍** |

注意第二列：**即使把一趟公式放到双精度里算，它也比我们的 fp32 两趟差**（512 那档两者持平）。
也就是说这不是"fp32 精度不够"，是**公式本身相消**——所以内核里不能图省那一次读。

为什么不是 Welford：Welford 要把 `(mean, M2, count)` 三元组做**跨 lane 的树形合并**（Triton 里得
给 `tl.reduce` 塞自定义 combine），而两趟已经把误差压到「输出舍入」这一档（fp32 的 7.5e-11 就是
`var≈0.156` 的 fp32 ulp 量级）。**等真出现「行大到装不进 L1/L2、第二趟要走 DRAM」时，换 Welford
才有收益**——那时两趟要多付一遍 DRAM 读，Welford 不用。这是有明确触发条件的升级路径，不是不做。

`row_std` 与 `row_var` 共用内核（`SQRT_OUT` constexpr），fp32 实测 max_rel 2.5e-08 ~ 1.1e-07。

### 10.3 `sum_axis` 为什么是 strided 内核而不是「permute 到最后一维」

归约中间维时，另一种做法是先把那一维 permute 到末尾、`contiguous()` 一次，再复用「最后一维
归约」。但那次 `contiguous` 对激活就是**多一遍全量 DRAM 往返**（读 + 写 + 再读）；这条内核让数据
**原地读一遍就出结果**：每个 lane 固定一个 `inner` 位置、跨 `red` 累加，访存是「跨 `inner` 长度
步进的连续段」，天然合并。

形状按 `(outer, red, inner)` 给：`inner` 是最内层连续维（torch 连续张量里 `x.sum(dim=-2)` 正好
是这个形状）。**多维组合（同时归约两个以上非相邻维）继续显式报错**——那需要通用步长向量，
是另一个量级的工作量，而模型里几乎用不到。

实测：`sum_axis` 在 dyadic 数据上 **逐字节相等**（3 dtype × 3 形状 = 9/9 全过），`mean_axis` 的
max_abs ≤ 2.2e-09（fp32）/ 2.9e-05（fp16）/ 1.4e-04（bf16），都是输出舍入量级。

### 10.4 同一族的第三个坑：`trunc nuw nsw`（这次用「指针行走」绕开）

`sum_axis` 第一版直接写 `r * inner + i`（`r` 是步长 1 的运行期循环变量），llc 报：

```
error: expected type（第 52 行：%44 = trunc nuw nsw i64 %indvars.iv to i32）
```

成因与 §7.2 的 `trunc nsw`、§8.4 的 `trunc nuw` **完全同族**：LLVM 的 `indvars` 把归纳变量加宽到
i64，再在每个用到它的地方截回 i32，而 `nuw`/`nsw` 是 LLVM 17 才加到 `trunc` 上的标记。

修法不是加补丁，而是**让 IR 里干脆没有归纳变量**：

```python
off = o.to(tl.int64) * red.to(tl.int64) * inner.to(tl.int64) + i
stride = inner.to(tl.int64)
for _ in range(0, red):
    v = tl.load(a_ptr + off, ...)
    acc += v
    off += stride          # ← 循环携带的 i64 向量自己往前挪
```

代价是每次迭代多一次向量加法，收益是**不再依赖「LLVM 不会加宽 IV」这种运气**。这条应写进
`corex-triton` 的写法约定：**运行期循环里避免 `r * stride` 形式的地址计算，改成指针行走。**

### 10.5 bf16 体检（本轮新增内核）

```
('row_var','bf16',0)   cubin= 9768  bf16_conversions=3  改写后可疑语法={}
('row_var','bf16',1)   cubin= 9776  bf16_conversions=3  改写后可疑语法={}   ← std
('sum_axis','bf16',0)  cubin= 7416  bf16_conversions=2  改写后可疑语法={}
('sum_axis','bf16',1)  cubin= 7640  bf16_conversions=9  改写后可疑语法={}   ← mean
合计 bf16_conversions=17，改写后可疑语法 0
```

bf16 全部真支持：`sum_axis` 逐字节、`mean_axis` 1.4e-04、`row_var` 相对误差 3.7e-04（都是输出
舍入量级，不是算法误差）。

### 10.6 未做：`masked_fill` 的广播 mask（优先级最低）

当前 `masked_fill` 要求 **mask 已经是全尺寸物化好的**（`*i8`，每元素 1 字节）。广播形状
（如 `(1, cols)` 广播到 `(rows, cols)`）需要内核知道广播的步长映射，那是另一条内核
（`mask_bcast`）。在此之前上层可以用 `mask.expand(shape).contiguous()` 兜住——代价是一次掩码
大小的拷贝，而掩码是 1 字节/元素，通常是特征图大小的 1/4~1/2。

### 10.7 结论

| # | 结论 |
| --- | --- |
| 1 | `row_var`/`row_std`/`sum_axis`/`mean_axis` 全部跑通，`probes/197` PASS（27/27 + 18/18），3 dtype × cols 1000/512/4097 |
| 2 | **方差必须两趟**：一趟公式在 fp32 下误差达 **1e6~1e7**（比待求量 0.156 大 7 个数量级），两趟好 **1e12~1e16 倍**；且一趟公式**即使在双精度下**也比我们的 fp32 两趟差（512 档持平）——问题是公式相消，不是精度不够 |
| 3 | Welford 的升级触发条件写清了：**行装不进 L1/L2、第二趟要走 DRAM** 时才值得换 |
| 4 | 中间维归约用 strided 内核、不做 permute+contiguous：后者对激活多一遍全量 DRAM 往返 |
| 5 | `trunc nuw nsw` 是同一族的第三次出现，修法是**指针行走**（消掉归纳变量），应写进 corex-triton 的写法约定 |

---

## 11. 一个测量陷阱：`bmm [4,1,16]@[4,16,6]` 的 2505 µs 是假的

`probes/023`（decode 算子逐项差分计时）报出：

| 算子 | 均值 µs/次 |
| --- | --- |
| `bmm [4,1,6]@[4,6,16]` | 85.6 |
| **`bmm [4,1,16]@[4,16,6]`** | **2505.0** ← 差 29 倍，不像内核慢 |

`probes/198_bmm_repro.sh` 用「200 次 + 循环外同步」复现，并**给出分布而不是只给均值**：

### 11.1 结论：内核与包装层都对形状完全不敏感（p50 13.5 µs）

| 形状 | p50 | p90 | 均值 | >100 µs 的迭代数 / 2000 |
| --- | --- | --- | --- | --- |
| `[4,1,16]@[4,16,6]`（"慢"的那条） | **13.48 µs** | 13.69 | 38.1 | 83 |
| `[4,1,6]@[4,6,16]`（"快"的那条） | **13.48 µs** | 13.67 | 34.7 | 83 |
| 另外 8 个形状（`K=1`、`N=1`、`B=1/8`、`M=2/16`、`K=N=6/16`） | **13.46–13.50 µs** | ≤13.91 | — | — |

- **裸 launch**（绕过我的 Python 包装层、直接 `_triton.launch`）p50 = **13.46 µs**，与包装层一致
  ⇒ 包装层开销 ≈ 0（全部成本就是一次 launch，和 §6.5 的 9.9 µs 启动地板对得上）。
- `_KERNEL_CACHE` 调用前后都是 1 条、**内核对象 identity 不变**、稳态单次调用不慢
  ⇒ **不存在每次重编**（缓存键里只有 dtype/分块/warps/stages，形状是运行期参数）。
- `malloc`+`free` 在 24 B ~ 64 KiB 各档约 7 µs（wall clock），与输出尺寸无关
  ⇒ 不是分配器。

### 11.2 2505 µs 是怎么来的：均值被罕见停顿污染

同一形状跑 2000 次：p50 **13.5 µs**，但 mean **33.5 µs**、p99 **512 µs**、max **675 µs**，
**4.1% 的迭代**超过 100 µs。用 50 次平均时，只要混进**一次 ~120 ms** 的停顿，均值就会被抬到
2.4 ms —— 这正是 2505 µs 的量级。

### 11.3 停顿归属：不是内核，是「launch 相关」的环境噪声

同一套计时方法做对照组：

| 被测对象 | p50 | >100 µs 的比例 |
| --- | --- | --- |
| 纯 Python 空循环 | 4.09 µs | **0 / 2000** |
| 只 `rt.synchronize()` | 21.3 µs | **0 / 2000** |
| **`fill`(64 个元素)** | 9.77 µs | **49 / 2000（2.5%）** |
| `bmm`(4 个 block) | 13.5 µs | 83 / 2000（4.1%） |

**只要真的发一次内核就有停顿（连 64 元素的 fill 也有），而空循环和纯同步一次都没有。**
所以停顿与「内核在跑」这件事相关，但与**工作量、形状**无关——最可能是这张物理卡被多租户共享时的
驱动抢占/时间片（沙箱队列是串行的，但物理卡 6/7 可能还有别的容器在用）。
关键点：**它是加性的、与形状无关，所以绝不会造成「同一形状量级、只差 K 方向」的 29 倍差**。

### 11.4 方法论教训（要改的是测量口径，不是代码）

1. **小算子的耗时必须报 p50（或 min-of-N），不能只报均值**。均值和 p50 差 2.5 倍是这张卡上的常态。
2. **迭代数要和停顿率匹配**：停顿率 ~4%，50 次里平均有 2 次命中；一次 120 ms 的停顿就足以
   把 2.4 ms 写进表里。要么把迭代数提到 ≥500，要么用 p90/p99。
3. **加对照组**：任何「某个形状/算子特别慢」的结论，先跑一条空循环和一条极小内核做基线，
   否则很容易把环境噪声记到某个算子头上。
4. 复核建议：把 `probes/023` 的统计量从均值换成 p50 重跑；若 `[4,1,16]@[4,16,6]` 的 p50 仍与
   `[4,1,6]@[4,6,16]` 差 29 倍，那才需要往下挖 aten 层（我这边 p50 已经证明两层都无差异）。

顺带一个交叉验证：`[4,1,6]@[4,6,16]` 的 85.6 µs 与我的账**完全对得上** ——
内核 13.5 + 分配 ~7 + aten/Python 分发约 50~60（`probes/023` 自己测出 `add` 里内核 33、总 82
⇒ 分发约 49）≈ 70~80 µs。也就是说**"快"的那一行是诚实的，"慢"的那一行是污染的**。

---

## 12. 给 C++ 快路径的稳定入口（`_ops_elementwise`）

把逐元素算子挪进 C++（模式照 `mm`：Python 编译 → `shim.load_kernel` → `set_*_kernel`）需要两样
东西；本文件提供四个公开入口，全部**语义化**，调用方不需要知道 `block`/`num_warps` 之类的内部参数。

### 12.1 入口

```python
from corex_torch import _ops_elementwise as elt

elt.fast_variants()                       # → 15 个 (op, variant) 组合（遍历这个，别自己写死列表）
elt.fast_op("add", "fp16", "scalar")      # → 实参顺序表（**不编译**，纯元数据）
elt.get_kernel("add", "fp16", "scalar")   # → CompiledKernel，.asm["cubin"] 即 load_kernel 的输入
elt.get_cubin("add", "fp16", "scalar")    # → bytes（上面那行的便捷写法）
elt.get_launch_meta("add", "fp16", "scalar")
# → {cubin, name, num_warps, num_stages, shared, block, args, n_ptrs, n_i32, n_f32, ...}
```

`variant` 只区分 **`"tensor"` / `"scalar"`**。`gelu` 的 erf/tanh 是**关键字 `mode`**，不是 variant
——两者是同一个 (op, variant) 下的两种数学形式，混进 variant 会让「变体」这个词失去意义。

### 12.2 为什么顺序表是「从签名数出来的」

`_signature()` 是内核签名的**单一真源**：`_build_kernel()` 用它编译，`fast_op()` 用它**派生**
`n_ptrs/n_i32/n_f32/args`。**两边不各写一份。**

这不是洁癖：实参顺序错了**不会报错**，只会算错——`n`（i32）与 `alpha`（f32）位置互换时，一个被
当成 float 解释、一个被当成长度，结果是「跑完了但全错」。手写常量表迟早和签名漂移，而漂移的
那天没人会注意到。所以顺序表必须是数出来的。

### 12.3 实参顺序表（`FASTPATH_OPS` 这批；三种 dtype 只差指针类型）

| op | variant | n_ptrs | n_i32 | n_f32 | has_alpha | alpha_effective | args（有序） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `add` | tensor | 3 | 1 | 1 | ✅ | ✅ | `a*, b*, out*, n:i32, alpha:f32` |
| `sub` | tensor | 3 | 1 | 1 | ✅ | ✅ | `a*, b*, out*, n:i32, alpha:f32` |
| `mul` | tensor | 3 | 1 | 1 | ✅ | ❌ | `a*, b*, out*, n:i32, alpha:f32` |
| `div` | tensor | 3 | 1 | 1 | ✅ | ❌ | `a*, b*, out*, n:i32, alpha:f32` |
| `add` | scalar | 2 | 1 | 1 | ❌ | ❌ | `a*, out*, n:i32, scalar:f32` |
| `sub` | scalar | 2 | 1 | 1 | ❌ | ❌ | `a*, out*, n:i32, scalar:f32` |
| `mul` | scalar | 2 | 1 | 1 | ❌ | ❌ | `a*, out*, n:i32, scalar:f32` |
| `div` | scalar | 2 | 1 | 1 | ❌ | ❌ | `a*, out*, n:i32, scalar:f32` |

其余已登记、本批不进快路径的：`fill` 1/1/1、`copy` 2/1/0、`silu`/`relu`/`rsqrt`/`gelu` 2/1/0、
`masked_fill` 3/1/1（`ptr_types` 里 `mask` 是 `*i8`）。

**`has_alpha` 与 `alpha_effective` 必须分开看**，这是本表最容易踩的地方：

* `has_alpha`：**内核签名里有没有 `alpha` 这个实参**。tensor 变体的 add/sub/mul/div 都有。
* `alpha_effective`：`alpha` 是否**真的改变结果**。只有 `add`/`sub` 的 tensor 变体是 True。
  `mul`/`div` 的那个槽位**仍然必须传**（签名要求的），传任何值都会被忽略——照签名传 `1.0` 最省事。
  `probes/199` §3 验过：`mul`/`div` 传 `alpha=0.0` 与 `1.0` 结果逐字节相同。

只留 `has_alpha` 的话很容易得出「mul 可以少传一个参数」的结论，然后 launch 参数个数不符
（`corex-triton` 的 driver 会报 `参数个数不符：签名要求 N 个`，还算不幸中的万幸）。

### 12.4 验收：`probes/199_fastop_api.sh` PASS 29/29

关键的一段是 **shim 模拟器**——它**只读 `fast_op()` 的返回值**，按表把实参填进
指针/i32/f32 三个并行数组再 `launch`，然后与「直调 Python 算子」比对：

```
add/sub/mul/div × tensor/scalar × fp32/fp16/bf16  → 24/24 全部逐字节相等
n = 4097（非 2 的幂、非 BLOCK 整数倍）
```

这就是 C++ 侧要做的事，所以等于先当了一次 C++。表若错位，24 组里必然有组算错。

### 12.5 建议用法（C++ 侧伪代码，对应 `_register_fast_ops()`）

```python
for (op, variant) in elt.fast_variants():
    if op not in elt.FASTPATH_OPS:
        continue
    for dtype in ("fp32", "fp16", "bf16"):
        m = elt.get_launch_meta(op, dtype, variant)     # cubin + 全部启动参数
        shim.load_kernel(op, variant, dtype, m["cubin"],
                         num_warps=m["num_warps"], num_stages=m["num_stages"],
                         n_ptrs=m["n_ptrs"], n_i32=m["n_i32"], n_f32=m["n_f32"],
                         args=m["args"])                # ← 顺序以 args 为准
        shim.set_grid(op, variant, dtype, "cdiv(n, block)", block=m["block"])
```

网格是 `cdiv(n, block)`（`fast_op()["grid_fn"]` 也能直接算，探针里就是这么用的）。
`block` 与 `num_warps` 从 `get_launch_meta()` 拿，别在 C++ 里再抄一遍默认值——抄的那份会漂移。
