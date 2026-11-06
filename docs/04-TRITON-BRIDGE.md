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
