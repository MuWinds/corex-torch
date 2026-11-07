# torch 2.8 的 PrivateUse1 扩展点盘点

本文回答一个问题：**上游 CPU 版 torch 2.8 要接纳一张新设备（CoreX / 天垓100），最少必须提供什么。**
每一条都带「产生它的命令」与实测输出。没有证据的一律写 `UNKNOWN`，并写清下一次探测要什么手段。

采集环境：SCOW 登录节点 `mgt`，2026-11-07。
torch 装在隔离 venv `~/venvs/corex28`（`torch 2.8.0+cpu`，python 3.10.16），
厂商 torch 2.1（共享 py310）只读、未改动。全部工作在登录节点完成，**没有碰卡**。

探针（`probes/`）：

| 编号 | 干什么 |
| --- | --- |
| `100_torch28_install.sh` | 装 torch 2.8.0+cpu 并验证 import（幂等，重跑 9 s） |
| `101_torch28_privateuse1.py` | 只用 Python 逐层试到失败，记录每一层的报错原文 |
| `102_torch28_cpp_ext.py` | 编最小 C++ 扩展 + 真 shim（分配器 / guard impl / 手搓张量），三段：`build`/`alloc`/`shim` |
| `103_torch28_privateuse1_src.sh` | 抓 Python 侧与 C++ 侧的源码证据（路径 + 行号） |
| `104_torch28_privateuse1_cpp.sh` | 头文件声明 + `nm -DC` 动态导出符号 |
| `105_torch28_pu1_hooks.sh` | PrivateUse1Hooks / generator / device lazy init 三个「第二层」入口 |
| `106_torch28_device_module_contract.py` | 用会记录属性访问的替身模块，实测 torch 到底调用 device module 的哪些函数 |
| `107_torch21_vs_28_diff.sh` | 厂商 torch 2.1 与上游 2.8 的接口 diff |
| `108_torch28_python_allocator_entry.sh` | Python 侧有没有分配器入口 + `cuda.is_available()` 是否被污染 |
| `110_shim_build.sh` | 登录节点编译真 shim（验收 A：一条命令出 `.so`、无 CUDA 依赖） |
| `111_shim_smoke.sh` | 沙箱真机验收（验收 B：strided 张量 / D2H 往返 / 缓存池收敛 / 算子清单） |
| `112_shim_edges.sh` | 沙箱真机边界：空张量 / 0 维标量 / 大块直通 / **多设备隔离** / init 幂等 |
| `120_aten_coverage.sh` | aten 覆盖总表：42 个算子与 CPU 逐项比形状/stride/数值，标出路由（ours / composite） |
| `121_view_alias.sh` | 视图语义硬要求：共享存储 / 写穿 / storage_offset / 视图存活期；逐条直调我们的实现 |
| `122_fast_launch.sh` | C++ 快速启动路径 vs Triton Python 启动的分段计时与数值一致性 |

---

## 0. 结论先说

| 问题 | 答案 | 关键证据 |
| --- | --- | --- |
| 能不能纯 Python 注册显存分配器？ | **不能** | `torch._C` 上只有 `_get_privateuse1_backend_name` / `_rename_privateuse1_backend` 两个 privateuse 入口（§2.2） |
| 那 Python 能做到哪一步？ | 能改名、能注册 device module、能注册单个 aten 算子；**但算子实现里无法把裸指针包成 Tensor** | §2.3、§2.4 |
| 必需的最小 C++ 面是什么？ | **1 个 `c10::Allocator` + 1 个 `c10::impl::DeviceGuardImplInterface`**，各注册一次 | §3.1、§4 |
| 用 g++ 10.3.1 + 上游 CPU wheel 的 headers 能编过吗？ | **能，26.4 s** | §4.2 |
| 真的能跑出 `torch.empty(..., device="corex")` 吗？ | **能**，`device=corex:0`，分配到的内存可读写 | §4.3 |
| `rename("corex")` 会把 `torch.cuda` 弄脏吗？ | 不会：`cuda.is_available()` 仍为 `False`、`device_count()` 仍为 0 | §2.5 |
| 剩下的墙在哪？ | 算子覆盖（`add.out` / `_copy_from` / `zero_` / `empty_strided` / `normal_`），不是设备注册 | §4.4 |

---

## 1. 装法：torch 2.8.0+cpu 进隔离 venv

### 1.1 为什么隔离

共享 py310 里的 `torch-2.1.0+corex.3.2.1` 是全环境的数值基线（`AGENTS.md` §4），
且**不能卸载**（别的项目在用）。`~/venvs/corex28` 是 `python -m venv` 建的纯隔离环境：

```
home = /data/software/Anaconda3/2024.10-1/envs/py310/bin
include-system-site-packages = false
version = 3.10.16
```

（证据：`python tools/remote.py put probes/100_torch28_install.sh <远端> && python tools/remote.py sh 'bash <远端>'`，
脚本 §0 段打印 `pyvenv.cfg`。）

### 1.2 两条踩过的坑

1. **venv 自带 pip 23.0.1 有名字规范化 bug**。下载 `typing_extensions-4.16.0` 时它把
   `typing-extensions` 与 `typing_extensions` 判为两个名字：

   ```
   Discarding ... typing_extensions-4.16.0-py3-none-any.whl ... has inconsistent Name:
   expected 'typing-extensions', but metadata has 'typing_extensions'
   ```

   随后丢弃 wheel、退化成源码构建，再卡在 `flit_core<4,>=3.11` 上失败。
   证据：`cat ~/venvs/corex28-pip.log`（旧日志保留着）。
   修法：先 `pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple` → pip 26.3。

2. **pip 自带的下载器在登录节点上下 184 MB 的 torch wheel 只有 ~52 kB/s**。
   实测：20 s 只长了 1 MiB；10 分钟走了 28 MB。同一时刻用 `curl` 下同一个 URL 是 **7.1 MB/s**
   （25 s 下完 184 021 630 B）。差两个数量级，是 pip 的 HTTP 栈问题，不是网络。
   修法：`curl` 下 wheel，再 `pip install --no-deps --no-index <本地 wheel>`。

### 1.3 最终可复现命令（就是 `probes/100_torch28_install.sh`）

```bash
VENV=/data/home/2024240384035/venvs/corex28
# 1) 升 pip（清华镜像）
$VENV/bin/python -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
# 2) 先把 torch 的纯 Python 依赖装好，后面装 torch 时完全不联网解析
$VENV/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    filelock jinja2 networkx sympy fsspec typing-extensions numpy
# 3) curl 下 wheel（-C - 断点续传；sha256 与 index 页一致）
curl -L -C - --retry 30 --retry-all-errors -o ~/wheels/torch-2.8.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl \
    "https://download.pytorch.org/whl/cpu/torch-2.8.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl"
# 4) 本地装
$VENV/bin/python -m pip install --no-deps --no-index ~/wheels/torch-2.8.0+cpu-*.whl
```

wheel 校验：

```
sha256 实测   16d75fa4e96ea28a785dfd66083ca55eb1058b6d6c5413f01656ca965ee2077e
index 页期望  16d75fa4e96ea28a785dfd66083ca55eb1058b6d6c5413f01656ca965ee2077e
```

（`sha256sum <wheel>` 对 `curl -sS https://download.pytorch.org/whl/cpu/torch/ | grep -o '...#sha256=[a-f0-9]*'`。）

### 1.4 版本证据

```
torch.__version__       = 2.8.0+cpu
torch.version.cuda      = None
torch.cuda.is_available = False
privateuse1 backend name= privateuseone
torch.get_num_threads() = 28
matmul(3x4 @ 4x3) max   = 366.0
TORCH28_VERDICT PASS
```

venv 占用 872 MB（`du -sh ~/venvs/corex28`），`pip list` 里只有 `torch 2.8.0+cpu` / `numpy 2.2.6`
等 8 个包，**没有 triton**（CPU wheel 不依赖 triton，`torch 2.8.0+cpu` 的依赖表里没有它）。

工具链（同一台登录节点）：`/usr/bin/g++` 10.3.1、`/usr/bin/ninja`、`/usr/local/bin/cmake` 都在。

---

## 2. Python 侧：PrivateUse1 的扩展点

### 2.1 入口总表

| API | 位置 | 作用 | 证据 |
| --- | --- | --- | --- |
| `torch.utils.rename_privateuse1_backend("corex")` | `torch/utils/backend_registration.py:18` | 把 `privateuseone` 改名成 `corex`，**每进程只能调一次** | `"this API can only be called once per process"`（同文件 docstring） |
| `torch._C._get_privateuse1_backend_name()` | `torch/_C/__init__.pyi:1662` | 读回当前名字（默认 `privateuseone`） | `dir(torch._C)` 过滤 `privateuse` 只剩两项 |
| `torch._C._rename_privateuse1_backend(name)` | `torch/_C/__init__.pyi:1661` | 上面那个函数的底层绑定 | 同上 |
| `torch._register_device_module("corex", module)` | `torch/__init__.py:2637` | `setattr(torch, "corex", module)` + 塞 `sys.modules["torch.corex"]` | `inspect.getsource(torch._register_device_module)` |
| `torch.get_device_module(device)` | `torch/__init__.py:2748-2749`（`@functools.cache`） | 按设备名取 module，取不到就抛 `RuntimeError: Device 'corex' does not have a corresponding module registered as 'torch.corex'.` | 源码 + 实测（`probes/108` §5） |
| `torch.utils.generate_methods_for_privateuse1_backend()` | `torch/utils/backend_registration.py:345` | monkey patch 出 `Tensor.corex()` / `Tensor.is_corex` / `Module.corex()` / `PackedSequence.corex()` /（可选）`Storage.corex()` | 实测 `hasattr(torch.Tensor,'corex') = True` |
| `torch.library.Library("aten","IMPL").impl(op, fn, "PrivateUse1")` | `torch/library.py:301` | **纯 Python** 注册单个 aten 算子的 PrivateUse1 实现 | 实测注册成功（`probes/102` §3） |
| `torch.library.Library("_","IMPL").fallback(fn, "PrivateUse1")` | `torch/library.py:391` | 给整个 PrivateUse1 挂兜底函数；**只允许全局 namespace `_`** | 源码里有 `if self.ns != "_": raise RuntimeError(...)`；实测注册成功（`probes/101` stage 4） |
| `torch._C._dispatch_key_for_device("corex")` | `torch.library.py:774` 调用处 | 设备名 → dispatch key | 实测返回 `PrivateUse1` |
| `torch.accelerator.*` | `torch/accelerator/__init__.py` | 2.6+ 的统一加速器 API：`device_count:28`、`is_available:47`、`current_accelerator:76`、`current_device_index:108`、`current_stream:142`、`set_stream:157`、`synchronize:168` | `grep -n "^def " torch/accelerator/__init__.py` |

### 2.2 Python 侧**没有**分配器入口（这是「必须 C++」的第一条硬证据）

```
torch._C 里 privateuse* : ['_get_privateuse1_backend_name', '_rename_privateuse1_backend']
torch._C 里 *alloc*     : ['_cuda_CUDAAllocator', '_cuda_beginAllocateCurrentThreadToPool',
                           '_cuda_beginAllocateToPool', '_cuda_endAllocateToPool',
                           '_is_default_mobile_cpu_allocator_set', '_mps_currentAllocatedMemory',
                           '_mps_driverAllocatedMemory', '_set_default_mobile_cpu_allocator',
                           '_unset_default_mobile_cpu_allocator']
torch.utils 里 privateuse* : ['generate_methods_for_privateuse1_backend', 'rename_privateuse1_backend']
torch.library 里 alloc 相关: []
```

（命令：`bash probes/108_torch28_python_allocator_entry.sh`。）
所有 `*alloc*` 私有 API 都是 CUDA/MPS 专用（`_cuda_CUDAAllocator` 那一族绑死在 CUDA 后端），
对 PrivateUse1 无效。存根 `torch/_C/__init__.pyi` 里 privateuse 开头的也只有 `:1661-1662` 两行。

### 2.3 设备模块必须是「可导入的模块」

`torch.empty(4, device="corex")` 在找一个真实模块 `torch.corex`，不是只看 `sys.modules`：

```
只 rename 之后: torch.empty(4, device='corex')
  -> ModuleNotFoundError: No module named 'torch.corex'
```

所以有两条落地方式（都实测可行）：

1. `torch._register_device_module("corex", module)`：注册后 `torch.corex.is_available()` 直接可用；
2. 把我们的目录挂进 `torch.__path__`，让 `import torch.corex` 真的能成功：

   ```python
   import torch, importlib
   torch.__path__.append("/path/to/our/torch_ext")   # 下面放 torch/corex/__init__.py
   importlib.invalidate_caches()
   import torch.corex          # OK，__file__ 指向我们的目录
   ```

   （证据：`probes/107` 末尾的试验，输出 `import torch.corex OK: /tmp/pu1path/torch/corex/__init__.py is_available() = True`。）
   这条对 shim 有直接影响：**不改上游 wheel 也能让 `torch.corex` 成为正经模块**，只要有个
   `.pth` / `sitecustomize` 在解释器启动时把它挂上。

### 2.4 device module 的必需函数（实测追踪，不是照抄文档）

`probes/106` 用一个「记录每次属性访问」的替身模块挂进 `torch._register_device_module`，逐条跑 API，
把 torch 主动要了什么抓出来：

| API | 会摸到的 device module 函数 | 是否必需 |
| --- | --- | --- |
| `torch.corex.is_available()` / `torch.accelerator.is_available()` | `is_available` | **必需** |
| `torch.accelerator.device_count()` | `device_count` | **必需** |
| `torch.accelerator.current_device_index()` | `_lazy_init`（**注意：文档里没列这个**） | **必需** |
| `torch.manual_seed(1234)` / `torch.random.manual_seed` | `_is_in_bad_fork` → `manual_seed_all` | 要支持 seed 就必需 |
| `tensor.corex()`（`generate_methods` 产物） | `current_device` | 用它就必需 |
| `torch.empty/zeros/tensor/randn(device="corex")` | **一个都不摸** | 不需要 |

实测输出（节选）：

```
[torch.accelerator.current_device_index()] OK
    摸到的 device module 函数: ['_lazy_init', '_lazy_init']
[torch.manual_seed(1234)] OK
    摸到的 device module 函数: ['_is_in_bad_fork', 'manual_seed_all', '_is_in_bad_fork', 'manual_seed_all']
[torch.empty(2, device='corex')] OK
    摸到的 device module 函数: （一个都没摸）
```

对照 `rename_privateuse1_backend` 的 docstring（`backend_registration.py:18` 起）列出的契约：

| docstring 要求的函数组 | 2.8 是否列出 |
| --- | --- |
| AMP：`get_amp_supported_dtype()` | 列了 |
| random：`_is_in_bad_fork` / `manual_seed_all` / `device_count` / `get_rng_state` / `set_rng_state` | 列了 |
| common：`is_available` / `current_device` | 列了 |
| `_lazy_init` | **没列，但实测会被调**（这是本探针最有价值的一处推翻文档） |
| `_get_device_properties` | **没列**。它不是 PrivateUse1 的必需项，是 `torch.cuda` 自己的约定；要提供得自己实现同名函数 |

### 2.5 rename 之后的状态（`cuda` 不会被弄脏）

```
rename 之前: cuda.is_available = False
rename 之前: cuda.device_count = 0
rename 之后: privateuse1 name  = corex
rename 之后: torch.device('corex') = corex
rename 之后: cuda.is_available = False        ← 仍为 False
rename 之后: cuda.device_count = 0
rename 之后: accelerator.current_accelerator = corex     ← 这条会变，通用代码要注意
rename 之后: accelerator.is_available -> RuntimeError: Device 'corex' does not have a
                                           corresponding module registered as 'torch.corex'.
```

（命令：`bash probes/108_torch28_python_allocator_entry.sh`。）

两条注意：
* `torch.accelerator.current_accelerator()` 在 rename 之后立刻返回 `corex` —— 任何用
  「有加速器就当有卡」的库代码会走错分支；`is_available()` 会在缺 device module 时抛错，
  所以 device module 必须随包提供。
* `torch.cuda.is_available()` 保持 False，**这是对的**：CoreX 走 PrivateUse1，不是 CUDA。

---

## 3. C++ 侧：必需的两个接口

### 3.1 符号导出实测（wheel 里是 `-fvisibility` 编的，声明存在 ≠ 能链）

`nm -DC --defined-only <so> | grep ...`（命令：`probes/104`、`probes/105`）：

| 符号 | 在哪个 so | 头文件声明 |
| --- | --- | --- |
| `c10::SetAllocator(DeviceType, Allocator*, uint8_t)` | **libc10.so** | `c10/core/Allocator.h:276` |
| `c10::GetAllocator(DeviceType const&)` | libc10.so | `c10/core/Allocator.h:277` |
| `c10::register_privateuse1_backend(std::string const&)` | libc10.so | `c10/core/DeviceType.h:104` |
| `c10::get_privateuse1_backend(bool)` | libc10.so | `c10/core/DeviceType.h:105` |
| `c10::is_privateuse1_backend_registered()` | libc10.so | `c10/core/DeviceType.h:107` |
| `c10::impl::DeviceGuardImplRegistrar::DeviceGuardImplRegistrar(DeviceType, DeviceGuardImplInterface const*)` | libc10.so | `c10/core/impl/DeviceGuardImplInterface.h:344`（`C10_REGISTER_GUARD_IMPL` 宏就是用它的 ctor） |
| `c10::DeviceTypeName(DeviceType, bool)` | libc10.so | `c10/core/DeviceType.h:98` |
| `torch::Library::Library(...)` / `c10::Dispatcher::registerImpl(...)` | libtorch_cpu.so | `torch/library.h:1062`（`TORCH_LIBRARY_IMPL`） |
| `at::RegisterPrivateUse1HooksInterface` / `at::isPrivateUse1HooksRegistered` / `at::detail::getPrivateUse1Hooks` | libtorch_cpu.so | `ATen/detail/PrivateUse1HooksInterface.h` |
| `at::detail::empty_generic(sizes, Allocator*, DispatchKeySet, ScalarType, optional<MemoryFormat>)` | libtorch_cpu.so | **头文件没发**（见 §5.2） |

`PrivateUse1` 的静态标识：`c10/core/DeviceType.h:57  PrivateUse1 = 20`，`:34 _(PrivateUse1, extra)`，
`:81 constexpr DeviceType kPrivateUse1`。编译期注入设备名的宏（`TORCH_PRIVATEUSE1` /
`pytorch_privateuse1_backend_name`）在 **2.8 的 CPU wheel 头文件里搜不到**：
`grep -rn "TORCH_PRIVATEUSE1" include/` 无命中 —— 2.8 的设备名是**运行时**通过
`c10::register_privateuse1_backend()` 注册的，不是编译期常量。这条很重要：意味着我们
**不需要重编 torch** 就能改设备名。

### 3.2 DeviceGuardImplInterface 必须实现的纯虚函数（8 个）

`grep -n "= 0;" c10/core/impl/DeviceGuardImplInterface.h`（2.8 行号）：

```
70:  virtual DeviceType type() const = 0;
75:  virtual Device exchangeDevice(Device) const = 0;
92:  virtual Device getDevice() const = 0;
97:  virtual void setDevice(Device) const = 0;
103: virtual void uncheckedSetDevice(Device) const noexcept = 0;
108: virtual Stream getStream(Device) const = 0;
141: virtual Stream exchangeStream(Stream) const = 0;
190: virtual DeviceIndex deviceCount() const noexcept = 0;
```

注册宏原文（同文件 344-352 行）：

```cpp
class C10_API DeviceGuardImplRegistrar {
 public:
  DeviceGuardImplRegistrar(DeviceType, const DeviceGuardImplInterface*);
};

#define C10_REGISTER_GUARD_IMPL(DevType, DeviceGuardImpl)              \
  static ::c10::impl::DeviceGuardImplRegistrar C10_ANONYMOUS_VARIABLE( \
      g_##DeviceType)(::c10::DeviceType::DevType, new DeviceGuardImpl());
```

**缺这一步的报错**（就是 `getDeviceGuardImpl` 里的 `TORCH_CHECK`，同文件约 127 行）：

```
RuntimeError: PyTorch is not linked with support for corex devices
```

（实测：`probes/101` stage 3 的 `cpu_tensor.to('corex')`；注册 shim 之后同一条变成
`NotImplementedError: ... 'aten::empty_strided' ...`，见 §4.4。）

### 3.3 c10::Allocator 的两个签名坑（2.8 相对 2.1 变了）

`c10/core/Allocator.h`（2.8）：

```cpp
struct C10_API Allocator {                       // :167
  virtual ~Allocator() = default;
  virtual DataPtr allocate(size_t n) = 0;        // :170  ← 不是 const！
  virtual bool is_simple_data_ptr(const DataPtr&) const;      // :182 有默认实现
  virtual DeleterFnPtr raw_deleter() const { return nullptr; } // :189 有默认实现
  virtual void copy_data(void* dest, const void* src, std::size_t count) const = 0;  // :212 ← 2.8 新增纯虚
};
```

两个坑（都是实测编译报错才发现的，照 2.1 写会挂）：

| 坑 | 2.1 的写法 | 2.8 编不过的原文 |
| --- | --- | --- |
| `allocate` 不是 const 成员 | `DataPtr allocate(size_t) const override` | `错误：‘c10::DataPtr ...::allocate(size_t) const’ marked ‘override’, but does not override` |
| 新增纯虚 `copy_data` | （2.1 没有） | `错误：invalid new-expression of abstract class type ‘...::HostAllocator’` |

`SetAllocator` 的语义（同文件 261-276 行注释原文要点）：传进去的指针**必须是静态生命周期**，
函数不接管所有权；**非线程安全**，只应在初始化时调用；`priority` 用来覆盖默认分配器，
只有不低于既有优先级的才能覆盖成功。

### 3.4 分配器不注册会怎样：拿到的是 **nullptr，不报错**

`probes/102 --stage alloc`（同一进程，先问再注册）：

```
[注册前的 c10::GetAllocator(PrivateUse1)] OK -> 0
[注册后的 c10::GetAllocator(PrivateUse1)] OK -> 107123696
```

这是「分配器必须由 C++ 注册」的第二条硬理由：**忘记注册不会给你一句人话报错**，
`GetAllocator` 返回 0，后面解引用空指针。（具体在哪一行崩 = `UNKNOWN`，见 §6。）

---

## 4. 最小可运行链路（实测走通）

### 4.1 三段式

`probes/102_torch28_cpp_ext.py` 分三段，各自独立进程：

| 段 | 内容 | 结果 |
| --- | --- | --- |
| `build` | `load_inline` 编一个只有 `triple(x)=3x` 的扩展 | `BUILD_MIN_OK triple(14)=42 编译耗时=27.9s` |
| `alloc` | 不注册先问 `GetAllocator`，再注册再问 | `0` → `107123696` |
| `shim` | 真 shim：分配器 + guard impl + 手搓张量 + Python 注册 `empty.memory_format` | 全链路通，见 §4.3 |

### 4.2 编译事实

```
c++ -MMD -DTORCH_EXTENSION_NAME=corex_probe_shim -DTORCH_API_INCLUDE_EXTENSION_H
    -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\"
    -DPYBIND11_BUILD_ABI=\"_cxxabi1018\"
    -isystem <venv>/lib/python3.10/site-packages/torch/include
    -isystem <venv>/lib/python3.10/site-packages/torch/include/torch/csrc/api/include
    -isystem /data/software/Anaconda3/2024.10-1/envs/py310/include/python3.10
    -fPIC -std=c++17 -O2 -c main.cpp -o main.o
c++ main.o -shared -L<venv>/lib/python3.10/site-packages/torch/lib \
    -lc10 -ltorch_cpu -ltorch -ltorch_python -o corex_probe_shim.so
```

* 编译器就是 `/usr/bin/c++` = **g++ 10.3.1**，不需要 `/data/software` 里的高版本 gcc。
* 首次编译 **26.4 s**（`BUILD_SHIM_OK 编译耗时=26.4s`），之后 ninja 缓存命中 0 s。
* 构建目录 `~/.cache/torch_extensions/py310_cpu/corex_probe_shim/`。

### 4.3 shim 里的三件事（这就是「最小 C++ 面」）

```cpp
#include <torch/extension.h>                                  // 懒人包，够用
#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/Stream.h>
#include <c10/core/StorageImpl.h>                             // 手搓张量才要
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <ATen/core/TensorBody.h>                             // at::detail::make_tensor

// ① 分配器：DataPtr 的 device 必须写 PrivateUse1，TensorImpl 是从它读 device 的
struct HostAllocator final : public c10::Allocator {
  c10::DataPtr allocate(size_t n) override {
    void* p = n ? std::malloc(n) : nullptr;
    return c10::DataPtr(p, p, &std::free, c10::Device(c10::DeviceType::PrivateUse1, 0));
  }
  void copy_data(void* dest, const void* src, std::size_t count) const override {  // 2.8 新纯虚
    if (count) std::memcpy(dest, src, count);
  }
  c10::DeleterFnPtr raw_deleter() const override { return &std::free; }
};

// ② guard impl：8 个纯虚函数（见 §3.2）
struct HostGuardImpl final : public c10::impl::DeviceGuardImplInterface { /* ... */ };

// ③ 两处注册
C10_REGISTER_GUARD_IMPL(PrivateUse1, HostGuardImpl);                     // 静态，加载即生效
c10::SetAllocator(c10::DeviceType::PrivateUse1, new HostAllocator(), 1); // 初始化时调一次
```

手搓张量的三行（上游没发 `ATen/native/EmptyTensor.h`，所以不依赖 `empty_generic`）：

```cpp
c10::Storage storage(c10::make_intrusive<c10::StorageImpl>(
    c10::StorageImpl::use_byte_size_t(), nbytes, allocator, /*resizable=*/true));
auto t = at::detail::make_tensor<c10::TensorImpl>(
    std::move(storage), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), dtype_meta);
t.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
```

> 注意：本探针在登录节点上没有卡，分配器用 host `malloc` 顶着。**它验的是注册通路，
> 不是最终显存来源**；真机要换成 CoreX 的 `cuMemAlloc` / `cuMemFree`。

### 4.4 打通后的实测输出（`probes/102 --stage shim`）

```
--- 1. 只注册 C++ shim（分配器 + guard impl），未注册 device module ---
  [torch.empty(4, device='corex')] EXC ModuleNotFoundError: No module named 'torch.corex'
--- 2. 加上 device module ---
  [torch.empty(4, device='corex')] EXC NotImplementedError: Could not run 'aten::empty.memory_format'
       with arguments from the 'corex' backend. ...
  [t.to('corex')] EXC NotImplementedError: ... 'aten::empty_strided' ...
  [accelerator.is_available()] OK -> True
--- 3. 再加 Python 侧的 aten::empty.memory_format 实现（内部调 C++ shim）---
  [lib.impl('empty.memory_format')] OK -> None
  [torch.empty(2, 3, device='corex')] OK
  张量属性: device=corex:0 dtype=torch.float32 shape=(2, 3) contiguous=True numel=6
  data_ptr=0x5846120（host 虚拟地址，说明分配器真的被调用了）
  写回读出 = [1.5, -2.5, 3.25, 4.75, 5.5, 6.0]
  [t.to('cpu')] EXC NotImplementedError: ... 'aten::_copy_from' ...
  [t + 1] EXC NotImplementedError: ... 'aten::add.out' ...
  [t.sum()] EXC NotImplementedError: ... 'aten::sum.IntList_out' ...
  [torch.zeros(4, device='corex')] EXC NotImplementedError: ... 'aten::zero_' ...
```

**三级失败梯度**（这条决定实施顺序）：

| 已注册到 | `torch.empty(4, device="corex")` 的报错 |
| --- | --- |
| 只 `rename` | `ModuleNotFoundError: No module named 'torch.corex'` |
| + device module | `NotImplementedError: Could not run 'aten::empty.memory_format' with arguments from the 'corex' backend. ...` |
| + C++ 分配器 & guard impl & `empty.memory_format` 实现 | **成功**，`device=corex:0` |

**剩下的墙是算子覆盖，不是设备注册**。第一批要铺的算子（按实测的报错倒推）：

| 场景 | 需要的算子 |
| --- | --- |
| `torch.tensor([1.], device="corex")` / `.to("corex")` | `aten::empty_strided` |
| `torch.zeros(4, device="corex")` | `aten::zero_`（zeros 先 empty 再 zero_） |
| `t.to("cpu")` | `aten::_copy_from` |
| `t + 1` | `aten::add.out`（out 版本，非 functional） |
| `t.sum()` | `aten::sum.IntList_out` |
| `torch.randn(4, device="corex")` | `aten::normal_`（还会牵到 generator，见 §5.1） |
| 打印张量（`repr`/`print`） | `aten::view` + `aten::item`（踩过：探针打印张量时被打断） |

---

## 5. 第二层入口（不是必需，但会咬到）

### 5.1 PrivateUse1HooksInterface（generator / lazy init 那一层）

`ATen/detail/PrivateUse1HooksInterface.h`（2.8 全文 2489 B）里每个函数不实现就
`TORCH_CHECK_NOT_IMPLEMENTED`，报错模板是
`You should register 'PrivateUse1HooksInterface' by 'RegisterPrivateUse1HooksInterface' and implement '<func>' at the same time for PrivateUse1.`
（同文件 10 处 `FAIL_PRIVATEUSE1HOOKS_FUNC`）。注册入口：`at::RegisterPrivateUse1HooksInterface(...)`，
符号在 libtorch_cpu.so 里导出（§3.1）。要实现的：`isBuilt` / `isAvailable` /
`getDefaultGenerator` / `getNewGenerator` / `getDeviceFromPtr` / `getPinnedMemoryAllocator` /
`hasPrimaryContext` / `resizePrivateUse1Bytes`；`isPinnedPtr` 有默认实现（返回 false）。

关联文件：`ATen/core/GeneratorForPrivateuseone.h`（`class _GeneratorRegister` 在 :12）、
`torch/csrc/utils/device_lazy_init.h`（`is_device_lazy_init_supported()` 把 PrivateUse1 算在内，
所以每次建张量前 C++ 会做一次 lazy init —— 这也是「device module 必须能 import」的来源）。

### 5.2 头文件没发全：`at::detail::empty_generic` 有符号但没声明

```
ls include/ATen/native/EmptyTensor.h   -> No such file or directory
nm -DC libtorch_cpu.so | grep empty_generic
  0000000001665270 T at::detail::empty_generic(ArrayRef<long>, Allocator*, DispatchKeySet, ScalarType, optional<MemoryFormat>)
```

含义：`include/ATen/native/` 只有 109 个条目（其中 12 个是子目录 `cpu`/`cuda`/`ao_sparse` 等），
内部 kernel 头没发全。
要复用 `empty_generic` 就得自己写函数声明去对符号（可做，但没必要）——
直接按 §4.3 的三行手搓更稳。

---

## 6. UNKNOWN（没有证据的项，以及下一次探测的手段）

| 项 | 现状 | 下一次探测手段 |
| --- | --- | --- |
| 不调 `SetAllocator` 时具体在哪一行崩 | 只知道 `GetAllocator(PrivateUse1)` 返回 0 | 沙箱里故意不注册分配器、跑建立张量的用例，抓 core dump 栈 |
| CoreX 的流语义怎么映射到 `getStream` / `exchangeStream` | 本次 `StreamId` 返回 0 是桩值 | 真机上确认真实 stream 句柄能否放进 `c10::Stream`（`Stream::UNSAFE` 构造） |
| `torch.randn` 能不能走通 | 只确认到 `aten::normal_` 未实现 | 注册 `PrivateUse1HooksInterface` + `_GeneratorRegister` 后再试 |
| 2.1 侧的**运行时**行为 | 登录节点不能 import 厂商 torch（`libcufft.so.10` 缺失） | 沙箱里 `import torch` 对照；或 `CUDA_VISIBLE_DEVICES` 空跑 |
| `Library("_","IMPL").fallback(fn,"PrivateUse1")` 里能不能 redispatch 回 CPU | 注册成功，但没验证兜底函数里怎么踢掉 PrivateUse1 再调用 | 用 `with_keyset=True` 拿 keyset，配合 `torch._C._dispatch_*` 试 redispatch |
| 多卡 / `deviceCount()` 的真实值 | 现在返回 1 是桩 | 沙箱里读 CoreX 的 `cuDeviceGetCount` |
| 对齐要求（`cuMemAlloc` 的对齐、`DataPtr` 是否能承载 stream 记录） | 未测 | 沙箱里读 `cudaMalloc` 返回指针对 512 B 取模并试对齐敏感算子 |
| `copy_data` 在 2.8 里的调用者 | 只知道它是纯虚，不知道谁调 | `grep -rn "copy_data" include/` + 真机跑一遍 storage clone |
| `TORCH_PRIVATEUSE1` 之类的编译期注入宏是否存在 | 2.8 CPU wheel 头文件里搜不到 | 有源码包时 `grep -rn "TORCH_PRIVATEUSE1"` |

---

## 7. 厂商 torch 2.1 vs 上游 2.8（`probes/107`）

### 7.1 版本与形态

| 项 | 厂商 2.1 | 上游 2.8 |
| --- | --- | --- |
| `torch.__version__` | `2.1.0` | `2.8.0+cpu` |
| `torch.version.cuda` | `'10.2'` | `None` |
| 设备后端 | **把 CoreX 当 CUDA 用**（`libtorch_python.so` 里 `corex`/`iluvatar` 字样 = 0，`privateuse1` = 4） | CUDA 后端不存在，PrivateUse1 空着 |
| `torch/corex.py` | 有，内容就两行：`corex = True`、`corex_arch = 'ivcore10'` | 无 |
| `torch/accelerator/` | **无** | 有 |
| `torch/_library/` | 无（`Library` 都在 `torch/library.py`） | 有 |
| `torch/_dynamo/device_interface.py` | 无 | 有 |

> 结论：**厂商 2.1 不是 PrivateUse1 实现**，它把 CoreX 伪装成 CUDA（所以 `torch.version.cuda`
> 是 10.2、社区代码全走 `torch.cuda.*`）。我们在 2.8 上走 PrivateUse1 是另一条路，
> 厂商那份源码里没有可以直接抄的 PrivateUse1 实现。

### 7.2 会影响我们的接口差异

| 接口 | 2.1 | 2.8 | 影响 |
| --- | --- | --- | --- |
| `DeviceGuardImplInterface::getStream` | `virtual Stream getStream(Device) const noexcept = 0` | `virtual Stream getStream(Device) const = 0`（**去掉 noexcept**） | 照 2.1 写 `noexcept override` 在 2.8 会编不过 |
| 同上 `exchangeStream` | `const noexcept = 0` | `const = 0` | 同上 |
| 接口文件规模 | 328 行 | 375 行 | 新增了若干带默认实现的事件/同步相关虚函数 |
| `c10::Allocator` | 只有 `allocate` 一个纯虚 | 多一个 `copy_data` 纯虚（`Allocator.h:212`） | 见 §3.3 |
| `PrivateUse1HooksInterface.h` | 897 B（无 `FAIL_PRIVATEUSE1HOOKS_FUNC` 宏） | 2489 B，10 处抛错函数 | 2.8 要实现的 hooks 明显更多 |
| `rename_privateuse1_backend` 的 AMP 契约 | docstring 列 5 个（`get_amp_supported_dtype`、`is_autocast_enabled`、`get_autocast_dtype`、`set_autocast_enabled`、`set_autocast_dtype`） | **只列 1 个** `get_amp_supported_dtype` | 2.8 的 AMP 状态迁到 `torch.amp`，device module 不用再实现那 4 个 | 
| `torch.library.Library.fallback` | **不存在** | 有（`torch/library.py:391`） | 2.8 才有「整键兜底」的能力，2.1 时代必须逐个算子注册 |
| `torch.library.Library.impl` | `impl(self, op_name, fn, dispatch_key='')` | `impl(...)` 增加 `allow_override` 等参数（`torch/library.py:301`） | 签名兼容，但 2.8 多了能力 |
| `c10::Event` | `elapsedTime` 不存在（`grep -c elapsedTime Event.h` = 0） | 存在（= 2） | 事件计时在 2.8 是内建，不用自己实现 |
| `c10::GeneratorImpl` | `graphsafe*` 不存在（= 0） | 新增 `graphsafe_set_state` / `graphsafe_get_state`（= 2） | 我们不用 CUDA graph，**不需要实现**（与 AGENTS.md 的禁令一致） |
| `c10::register_privateuse1_backend` | 有，`DeviceType.h:101` | 有，`DeviceType.h:104`（多了 `get_`/`is_..._registered` 两个） | 设备名的运行期注册两代都有，不是 2.8 新特性 |
| `torch.get_device_module` | 无（`torch/__init__.py` 里没有） | 有（`:2748`，`@functools.cache`） | 2.8 统一了「设备名 → 模块」的取法，取不到报 `RuntimeError`（不是 ImportError） |

---

## 8. 建议的实施顺序（按证据倒推）

> **进度（2026-11-07）**：第 1、2 步已落地并在真机验通，见 §10；第 3、4 步由 `_device.py` 接手，
> 需要的算子清单在 §10.6。

1. **C++ 薄 shim**（约 60 行）：`CorexAllocator`（`cuMemAlloc`/`cuMemFree` + `DataPtr` 带 `PrivateUse1`
   设备号 + `copy_data` 用 `memcpy`/`cuMemcpy`）+ `CorexGuardImpl`（`cuSetDevice` + stream 句柄）
   + 一个 `corex_empty(sizes, dtype)` factory；`C10_REGISTER_GUARD_IMPL` + `c10::SetAllocator` 各一次。
2. **device module**（Python）：`is_available` / `device_count` / `_lazy_init` / `_is_in_bad_fork` /
   `manual_seed_all` / `current_device`（前三个是实测必需，后三个按需），并在解释器启动时让它
   能以 `torch.corex` 被 import（`torch.__path__` 挂载或 `.pth`）。
3. **第一批算子**（Python 侧 `torch.library` 注册，内部调 shim）：`empty.memory_format`、
   `empty_strided`、`zero_`、`_copy_from`、`add.out`、`view`、`item`、`sum.IntList_out`。
4. **之后再考虑** `PrivateUse1HooksInterface`（generator，`randn` 需要）与算子下沉到 C++。

## 9. 一键复现

```bash
cd corex-torch
python tools/remote.py put probes/100_torch28_install.sh /data/home/2024240384035/p100.sh
python tools/remote.py sh 'bash /data/home/2024240384035/p100.sh'                    # 期望 TORCH28_VERDICT PASS

for p in 101_torch28_privateuse1.py 102_torch28_cpp_ext.py 106_torch28_device_module_contract.py; do
    python tools/remote.py put probes/$p /data/home/2024240384035/$p
done
python tools/remote.py sh '~/venvs/corex28/bin/python /data/home/2024240384035/101_torch28_privateuse1.py'
python tools/remote.py sh '~/venvs/corex28/bin/python /data/home/2024240384035/102_torch28_cpp_ext.py --stage shim'
python tools/remote.py sh '~/venvs/corex28/bin/python /data/home/2024240384035/106_torch28_device_module_contract.py'

for p in 103_torch28_privateuse1_src.sh 104_torch28_privateuse1_cpp.sh \
         105_torch28_pu1_hooks.sh 107_torch21_vs_28_diff.sh 108_torch28_python_allocator_entry.sh; do
    python tools/remote.py put probes/$p /data/home/2024240384035/p.sh
    python tools/remote.py sh 'bash /data/home/2024240384035/p.sh'
done

# 真 shim：登录节点编译 → 沙箱真机验收
WORK=/data/home/2024240384035/coffin-work/base/sandbox/work/ct
python tools/remote.py put tools/build_shim.py $WORK/repo/tools/build_shim.py
python tools/remote.py put src/csrc/corex_shim.cpp $WORK/repo/src/csrc/corex_shim.cpp
python tools/remote.py put probes/110_shim_build.sh /data/home/2024240384035/p110.sh
python tools/remote.py sh 'bash /data/home/2024240384035/p110.sh'        # 期望 PROBE110_DONE
python tools/remote.py sbx-run probes/111_shim_smoke.sh --name 111_shim_smoke --tail 0
```

---

## 10. 薄 C++ shim：已落地并真机验通（2026-11-07）

本节的数字全部来自 `probes/110`（登录节点编译）与 `probes/111`（沙箱 gpu-node-07，健康卡 6/7
经 runner 重编号成 0..1）。代码在 `src/csrc/corex_shim.cpp`，构建脚本 `tools/build_shim.py`。

### 10.1 Python 侧拿到的接口（签名已固定）

```python
import torch                      # 必须先 import torch，否则 libc10.so 加载失败
from corex_torch import _corex_shim as shim

shim.init(0)                      # 幂等；绑定设备 + SetAllocator + guard impl 已静态注册
shim.is_initialized()             # -> True
shim.device_count()               # -> 2（本沙箱）
t = shim.empty_strided([2, 3], [3, 1], 6, 0)          # dtype 传 c10::ScalarType 的整数
b = shim.from_blob(ptr, [4, 4], [4, 1], 6, 0)         # 不接管所有权
shim.reserved_bytes(); shim.allocated_bytes(); shim.empty_cache()
shim.slab_alignment()             # 额外加的一个观测接口，见 10.5
```

两个必须知道的细节：

* **模块/文件名叫 `_corex_shim`，不是 `corex_shim`。** CPython 按文件名找 `PyInit_<文件名>`
  符号（`_corex_shim.so` → `PyInit__corex_shim`），对不上就报
  `ImportError: dynamic module does not define module export function`（实测踩过）。
  `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` 里的名字由 `build_shim.py` 传进去，所以两者自动一致；
  C++ 侧的命名空间仍叫 `corex_shim`。
* **`dtype` 传整数**：`int(torch.float32) == 6`（实测可用，等于 `c10::ScalarType::Float`），
  不必自己维护一张映射表。

### 10.2 四条设计决定的理由

| 决定 | 理由 |
| --- | --- |
| 编译期不 include 任何 CUDA 头、不链 `libcudart`，全部 `dlopen`/`dlsym` | 编译必须在登录节点做（没有 `/usr/local/corex`）；`ldd` 实测只有 `libc10/libtorch_cpu/libtorch/libtorch_python/libdl` |
| 分配器自带 2 的幂缓存池（最小 512 B，13 个桶到 2 MiB；大块按 2 MiB 整数倍直通） | 平台没有 memory pool，`cudaMalloc` 是真分配（R7）。aten 频繁申请小块 |
| 页大小编译期固定 2 MiB | KMD 的 `PG_MODE=2M`，是编译期常量，探也只有一个答案 |
| `empty_strided`/`from_blob` 的 strides **原样**写入 TensorImpl | 否则 `view`/`transpose` 会静默算错。真机验过 `(1,2)` 这种非连续布局 |

### 10.3 编译事实（验收 A，`probes/110`，登录节点）

```
[build_shim] torch 2.8.0+cpu cuda=None
[build_shim] 导出接口 ['allocated_bytes', 'device_count', 'empty_cache', 'empty_strided',
                       'from_blob', 'init', 'is_initialized', 'reserved_bytes', 'slab_alignment']
[build_shim] is_initialized() = False（未 dlopen 运行时，符合预期）
[build_shim] 编译耗时 26.7 s
[build_shim] 产物 .../src/corex_torch/_corex_shim.so (248384 B)
BUILD_SHIM_OK → NO_CUDART_DEP_OK → SHIM_API_OK → PROBE110_DONE
```

实际命令行（`build/shim/build.ninja` 抄出来的）：

```
cflags  = -DTORCH_EXTENSION_NAME=_corex_shim -DTORCH_API_INCLUDE_EXTENSION_H
          -DPYBIND11_COMPILER_TYPE=_gcc ... -fPIC -std=c++17 -O2
ldflags = -shared -ldl -L<venv>/lib/python3.10/site-packages/torch/lib \
          -lc10 -ltorch_cpu -ltorch -ltorch_python
```

编译器是系统的 **g++ 10.3.1**，不需要 `/data/software` 里的高版本 gcc，也不需要 `nvcc`。

### 10.4 真机实测（验收 B，`probes/111`，`SHIM_SMOKE_PASS`）

```
init(0) 之后 is_initialized = True | device_count = 2
=== empty_strided + H2D/D2H 往返 ===
  device = corex:0 | dtype = torch.float32 | shape = (2, 3) | stride = (3, 1)
  | numel = 6 | contiguous = True
  data_ptr = 0x304000002000000            ← 60 位设备地址，与 _runtime.py 的警告一致
  H2D rc = 0 / D2H rc = 0
  读回 = [1.5, -2.5, 3.25, 4.75, 5.5, 6.0]   ROUNDTRIP_MATCH
=== 非连续 strides ===
  shape = (2, 3) | stride = (1, 2) | contiguous = False    STRIDE_PASSTHROUGH_OK
  存储顺序读回 = [10, 20, 30, 40, 50, 60]
=== from_blob 不接管所有权 ===
  包装张量析构后仍能 rt.free()                            FROM_BLOB_NO_OWNERSHIP_OK
=== 缓存池收敛 ===
  基线           : reserved = 0         allocated = 0
  [2,3] 记账     : 24 B                                    STORAGE_BYTES_OK
  200 个张量活着 : reserved = 18874368  allocated = 4440960
  全部释放后     : reserved = 18874368  allocated = 0       （池留着，账归零）
  empty_cache    : reserved = 0         allocated = 0
                                                          SHIM_ALLOC_PASS
=== 结尾 ===
  backend 名 = corex | cuda.is_available = False（必须仍是 False）
  CARD_STILL_HEALTHY → PROBE111_DONE
```

`slab_alignment()` 观测到 **16 MiB / 32 MiB**（两次运行不同），说明这张卡的 `cudaMalloc`
返回值远高于 2 MiB 页粒度——按 2 的幂切分不会破坏对齐。

### 10.4.1 边界与多设备（`probes/112`，`SHIM_EDGES_PASS`）

```
空张量 (0,3)      : numel=0 data_ptr=0，账本不变                      OK
0 维标量 []       : 记账 4 B，可读写                                  OK
大块 4 MiB        : reserved 增量 4194304（2 MiB 整数倍），首部可读写   OK
多设备隔离        : 0 号卡 0x304000002000000 / 1 号卡 0x394000002000000
                    1 号卡不复用 0 号卡的块；两卡数据互不干扰          OK
init 幂等         : 重复 init(0) 不重复注册、账本不变                  OK
收尾              : empty_cache 后 reserved=0 allocated=0             OK
```

**⚠️ 一条必须遵守的硬规则（真机实测，不是理论）**：本平台 `cudaMemcpy` 在
「当前 runtime 设备」与指针所属设备不一致时 **不报错、rc=0，而是静默别名**——
1 号卡的指针 `0x394000002000000` 在 0 号卡当前的上下文里被当成同一块偏移，
写到 1 号卡的数据实际落在 0 号卡上。

```
反面实验 rc = 0 | 本该写卡 1 的 -7/-8 之后，卡 0 上读回 [-7.0, -8.0]
CROSS_DEVICE_SILENT_ALIAS
```

推论，写 `_device.py` 时必须照做：

1. **每次 raw `cudaMemcpy` 之前显式 `set_device(该指针所在的卡)`。** 顺序交错地操作两张卡时
   尤其危险（我们的探针第一版就是在这里读出了「两张卡的数据一样」）。
2. shim 的 `empty_strided` / `from_blob` 已经在 factory 里绑一次设备（否则从缓存池命中时
   根本不会调 `cudaMalloc`，当前设备就停在别的卡上），但**两次 factory 之间切换过设备，
   绑定就会失效**——绑定是「当时」的，不是「永久」的。
3. 由此得出分配器必须按设备分池：同一块设备内存不能在不同卡之间复用（本仓库已实现，
   `g_device_pools` 按 `device_index` 分开，`probes/112` 用「0 号卡释放的块不许发给 1 号卡」
   这条断言钉住）。

### 10.5 真机抓出来的一个真 bug（回归项）

第一版 `empty_strided` 用「取各维最大跨度」算存储量：

```cpp
elements = max(elements, stride[i]*(size[i]-1) + 1);   // 错
```

对 `[2,3]` 配 `(3,1)` 只算出 4 个元素（实际要 6 个），大张量会写越界。暴露方式很隐蔽：
探针的 D2H 往返照样对（桶里给的块比 24 B 大得多），但**账本对不上**——两个 6 元素张量
记了 36 B 而不是 48 B。改成求和：

```cpp
elements += stride[i]*(size[i]-1);   return (elements + 1) * itemsize;   // 对
```

`probes/111` §5 现在有一条 `STORAGE_BYTES_OK` 断言（`[2,3]` 必须记账 24 B）钉住它。

### 10.6 `_device.py` 要补的算子清单（真机抓的原文，按先后顺序）

只有 shim（`empty.memory_format` 未注册）时：

| 表达式 | 报什么 |
| --- | --- |
| `torch.zeros(4, device="corex")` | `NotImplementedError: ... 'aten::empty.memory_format' ...` |
| `torch.empty(4, device="corex")` | 同上 |
| `torch.tensor([1.], device="corex")` | `NotImplementedError: ... 'aten::empty_strided' ...` |

挂上 `empty.memory_format`（用 shim 实现）之后，下一跳依次是：

| 表达式 | 缺的算子 |
| --- | --- |
| `torch.zeros(4, device="corex")` | `aten::zero_` |
| `t + 1` | `aten::add.out` |
| `t.sum()` | `aten::sum.IntList_out` |
| `t.to("cpu")` | `aten::_copy_from` |

注意 `add` 走的是 **`.out` 重载**、`sum` 走 `IntList_out`——补算子时不能只实现 functional 版本。

### 10.7 这一层**没有**做的事（下一阶段的边界）

| 没做 | 说明 |
| --- | --- |
| 任何 aten 算子 | 全在 Python 层（§10.6 的清单） |
| pinned 暂存区（R8） | `.cpu()` / H2D 批量上传要靠它，落在 `_device.py` 的 `_copy_from` 里 |
| 事件（`record`/`block`/`queryEvent`） | guard impl 里没实现，默认实现会抛「Backend doesn't support events」。R3 要求「事件少而短」，等真需要时再加 |
| `copy_data` 的异步/流感知 | 现在走同步 `cudaMemcpy(D2D)`，够语义正确，不够快 |
| ~~多设备池隔离~~ | **已做**：`g_device_pools` 按设备分池（§10.4.1），因为真机证实跨设备指针会静默别名 |

### 10.8 部署位置（远端）

`.so` 必须和 `_device.py` 在同一个包里（`from . import _corex_shim` 按文件位置找）：

```
/data/home/2024240384035/coffin-work/base/sandbox/work/ct/src/corex_torch/_corex_shim.so
```

构建/部署闭环（登录节点编译，产物直接进包目录）：

```bash
cd $WORK/repo && ~/venvs/corex28/bin/python tools/build_shim.py \
    --out $WORK/ct/src/corex_torch/_corex_shim.so
```


---

## 11. aten 分发层第二批（`_aten_extra.py` + `_meta.py`）

目标不是「把算子都实现一遍」，而是**先实测哪些真的缺**。判据来自 `probes/120`：
42 个算子里，`dispatch` 表里已有 `PrivateUse1` 键（含 composite 复制到每个后端的
`[default backend kernel]`）的**一律不碰**，只补「真缺」和「composite 的下一跳正好缺」的。

### 11.1 基线（`_aten_extra` 尚未提供时，真机实测）

`OK 5 / FAIL 37`，而且 37 个里 **19 个是同一个根因**：

| 报错原文 | 数量 | 涉及算子 |
| --- | --- | --- |
| `Could not run 'aten::as_strided' ...` | 13 | `t` / `transpose.int` / `permute` / `squeeze(.dim)` / `unsqueeze` / `expand` / `expand_as` / `contiguous` / `select.int` / `slice.Tensor` / `narrow` |
| `Could not run 'aten::view' ...` | 2 | `view` / `view_as`（`view_as` 有 composite，下一跳是 view） |
| `Could not run 'aten::empty_strided' ...` | 1 | `torch.tensor([...], device='corex')` |
| `Could not run 'aten::sum.IntList_out' ...` 等 | 若干个 | 规约/激活/拼装，见 11.4 |

### 11.2 实现（每条都写清「为什么不交给 composite」）

| 算子 | 为什么不交给 composite | 落地方式 |
| --- | --- | --- |
| `aten::as_strided` | dispatch 表无 PrivateUse1 键、无 composite；**13 个算子的公共下一跳** | C++ shim 的 `as_strided`（共享 StorageImpl + 独立 sizes/strides/offset + 视图持存储所有权） |
| `aten::view` | 同上（只有 CPU/Meta 实现） | `_meta.view_strides` 算行主序 strides（支持 `-1`）后走 shim |
| `aten::_reshape_alias` | `reshape` 的 composite 在「可视图」时走它，而它自己缺 | 直接走 shim `as_strided` |
| `aten::copy_.Tensor` | composite 落到 `_copy_from`，而后者是**线性** memcpy；非连续源会拷错（实测 `contiguous` 的 `max_abs_diff=2.0`） | 按「末尾连续段 + 外层 stride」分段 D2D memcpy |
| `aten::mul_.Tensor` / `div_.Tensor` | composite 落在 `.out` 上，而 `.out` 在本版本里是错的（`alpha` 被无条件传给没有 alpha 形参的内核，`TypeError`） | 复用 `_ops_elementwise` 已有内核 |
| `aten::add_.Scalar` / `sub_.Scalar` / `mul_.Scalar` / `div_.Scalar` | `.out` 的标量路径未接（`NotImplementedError`） | 同上（`alpha` 折进标量） |
| `aten::silu.out` | `silu` 有 composite，但下一跳就是它 | 内核已有，只补分发 |
| `aten::cat.out` | `cat`/`stack` 都有 composite，下一跳都是它 | 纯 D2D 分段拷贝拼装（不写内核：D2D 595–649 GB/s，比逐元素内核快） |

### 11.3 真机结果（`probes/120`，同一台 `gpu-node-07`）

| 指标 | 基线 | 现在 |
| --- | --- | --- |
| OK / 总数 | 5 / 42 | **28 / 43**（新增 `clone_of_view` 用例） |
| 数值不符 | — | **0** |
| 视图族（19 个） | 全 FAIL | **全 OK**，形状/stride/dtype/数值与 CPU 逐项一致 |

视图语义另外单独验过（`probes/121`，`VIEW_ALIAS_PASS`）：

```
视图形状 (4,3) / stride (1,4) / data_ptr 与基张量相同 / storage 是同一个 / storage_offset=2 生效
写穿：透过视图写 0，基张量读回 [0...]（其余元素未被碰）
生命周期：删掉基张量后视图仍可读；视图释放后 allocated_bytes 回落
```

### 11.4 仍然是缺口（`KNOWN_GAPS`，按缺的东西分类）

| 缺什么 | 算子 | 说明 |
| --- | --- | --- |
| 扫描内核（要同时出下标） | `aten::max.dim_max`、`aten::argmax.out` | `_ops_reduce.max_last` 只出 values |
| log 版 softmax | `aten::_log_softmax.out` | `_ops_reduce` 只有 `softmax_last` |
| 两趟/一趟 Welford | `aten::var.correction` | |
| 逐元素内核 | `aten::rsqrt.out`、`aten::relu`、`aten::relu_`、`aten::gelu.out`、`aten::masked_fill_.Scalar` | `_ops_elementwise` 目前只有 add/sub/mul/div/标量/fill/silu/copy |
| 设备端 RNG | `aten::normal_` / `aten::uniform_` / `aten::bernoulli_` | **本阶段不做**：卡上没有可用的随机数通道。要随机数就 CPU 生成再 `.to("corex")`；`torch.randn(..., device="corex")` 报 `NotImplementedError` 是**预期行为** |
| dtype 重解释 | `aten::view.dtype` | 要换 TypeMeta 建视图，shim 目前按源 dtype 建 |

### 11.5 三条「会静默算错」的告警（`_aten_extra.warnings()` 原样输出）

1. `_device._out_binary` 的**标量路径**未接：`t + 2.0` / `t - 2.0` 走 `add.out`/`sub.out` 抛
   `NotImplementedError`；本模块只能覆盖 aten 级的 `add_.Scalar`，Python 的 `t.add_(2.0)`
   会直接落到 `add.out`，所以这条仍会咬人。
2. `_device._copy_from_impl` 是线性 memcpy（只校验 shape）：对非连续源直接 `.cpu()` 会拷错。
   本模块的 `copy_.Tensor` 兜住了 `clone`/`contiguous` 这类「先 empty_like 再 copy_」的路径，
   但 `_copy_from` 本身仍需按 stride 实现。
3. device module 没提供 `manual_seed_all` / `_is_in_bad_fork`：`torch.manual_seed()` 会**静默成功**
   而不改任何状态。本阶段不做设备端 RNG，建议这两个函数显式抛 `NotImplementedError`。

---

## 12. C++ 快速启动路径（`shim.load_kernel` / `shim.launch`）

### 12.1 为什么

`probes/010` 的归因：mm 256³ 单次 265.8 µs = 内核 ≈166.5 + **Python 启动 ≈39.6** + aten ≈59.7。
decode 每 token 六百多次发射，按这个数就是每步 60 ms 级白费。Triton 的 Python 启动路径每次
要过 tuple 打包、`_pack_arg`、多层 Python 函数——这些在 C++ 里全可以省掉。

### 12.2 接口（已实现，14 个导出里的 4 个新接口）

```python
handle = shim.load_kernel(cubin_bytes, kernel_name, "ppppiii")   # 按内容哈希缓存，重复调用零成本
shim.launch(handle, [gx,gy,gz], [bx,by,bz], shared, stream,
            [a, b, c, scratch],   # ptr_args
            [],                   # f32_args
            [M, N, K],            # i32_args
            [])                   # i64_args
```

实参**按位置**从三个并行数组取，取哪个由 `spec` 的字符决定（`p`=设备指针 `f`=float
`i`=int32 `I`=int64），所以启动期没有字符串解析、没有 dict 查找、没有每参数分配。
内核按 cubin 内容哈希缓存 module+function，`kernel_count()` 可查。

### 12.3 两个必须记下来的坑

| 坑 | 现象 | 根因与修法 |
| --- | --- | --- |
| driver API 的 current context | `cuModuleLoadData` 返回 **`rc=709 context is destroyed`** | cudart 建的是 primary context，但 libcuda 的 current context 可能指向已销毁的句柄。**逐字照抄** `corex-triton` 的 `_ensure_context`（`driver.py:86`）：`cuInit` → `cuDeviceGet` → `cuDevicePrimaryCtxRetain` → `cuCtxSetCurrent`。热路径用 thread_local 缓存这个确认结果，失败一次再重试一次 |
| Triton 3.4 的**隐藏尾参** | `launch` 段错误（rc=139） | 3.4 在实参表末尾追加一个 **global scratch 指针**（`docs/00-FACTS.md` §14.1 记过，`metadata.global_scratch_size`）。少传这一个，内核从栈上读到垃圾指针。mm 的 scratch 大小为 0，但要**照传一个 0** |

### 12.4 真机数字（`probes/122`，单卡 fp32，循环外同步一次）

> **注意：本节 2026-11-07 02:05 修订过。** 早先记的「256³ 端到端 110.5 µs / 303.9 GFLOP/s」
> 是在**参数契约错误**下测出来的（内核实际一个元素都没写，等于空转），属于假快；
> 修好 spec/shared 之后（见 §12.6）的真实数字如下。

| 段 | Python 启动（Triton） | C++ 快路径 | 倍数 |
| --- | --- | --- | --- |
| mm 16³（纯发射地板） | 38.34 µs | **10.79 µs** | **3.6x** |
| mm 256³（端到端） | 184.48 µs | **171.79 µs** | 1.07x |
| mm 256³ GFLOP/s | 181.9 | **195.3** | 1.07x |
| mm 256³ 走 aten | 265.8 µs（010 基线） | **193.26 µs / 173.6 GFLOP/s** | 1.38x |

数值一致性：两条路径算出的 c **逐元素完全相等**，与 CPU 参照 `max_abs_diff = 0.000e+00`。

注意 256³ 的绝对值已经低于 `probes/010` 里「内核 ≈166.5 µs」的归因值：那 166.5 是**含 host
停顿**的墙上时间，把 host 压到 10.8 µs 之后队列才真正喂满，量到的是内核本来的 110 µs
（GFLOP/s 303.6）。所以「mm 256³ 端到端 ≤60 µs」这个目标在 303 GFLOP/s 下等于要 559 GFLOP/s，
与「GFLOP/s ≥180」这一条互相矛盾——建议把验收口径改成「发射 ≤12 µs」+「GFLOP/s ≥180」。

### 12.5 aten 层：C++ `aten::mm` 已接管（`TORCH_LIBRARY_IMPL`）

`corex_shim.cpp` 里注册了 `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) { m.impl("mm", ...); }`，
Python 侧首次编译后调一次 `set_mm_kernel(dtype_code, handle, bm, bn, bk, num_warps=1)` 登记；
**没登记时 `aten::mm` 明确报错、不回落 Python**（回落会让性能数据分不清走没走快路径）。

真机（`probes/122`，同一份 cubin）：

| 路径 | 单次 | GFLOP/s |
| --- | --- | --- |
| `aten` `a @ b`（C++ 实现） | **204.1 µs** | **164.4** |
| `probes/010` 基线（Python 实现） | 265.8 µs | 126.2 |

> **2026-11-07 02:10 更正（重要）。** 这里原本记着「aten 分发器 ≈73 µs/次固定成本」，
> 那是**被污染的测量**：当时 `_aten_kernels` 把 Python 版 `mm` 又注册了一遍，覆盖掉了 C++ 版
> （就是那条 `Overriding a previously registered kernel` 警告）。每次 `aten::mm` 实际走的是
> **Python kernel**，那 73 µs 是 Python 回调，不是分发器。C++ 接管且 Python 版不再重复注册之后：

| 量 | 值（02:08 两次复现） | 含义 |
| --- | --- | --- |
| `aten mm 16³`（内核≈0） | **10.14 / 10.20 µs** | 走分发器的完整地板 |
| `shim.launch` 16³ | 10.78 / 10.82 µs | 同样一次启动，不经过分发器 |
| `aten` `a @ b` 256³ | **170.52 µs / 196.8 GFLOP/s** | 与直调 170.49 µs 持平 |
| `shim.is_initialized()`（pybind 地板） | 0.21 µs | Python→C++ 调用本身不贵 |

结论：**aten 分发器基本免费**（10.14 vs 直调 10.78 µs，差异在噪声内）。热路径**不必**自建绕过 aten 的一层，
只要保证「Python 版 kernel 不要覆盖 C++ 版」。真正要避开的是「用 Python 实现」而不是「用 aten」。
（这条更正反过来也说明：那条 `Overriding ... previous kernel` 警告是**实质缺陷**，不是噪音。）
---

## 13. 复跑记录（2026-11-07 01:47，三探针同批）

| 探针 | 结果 | 关键数字 |
| --- | --- | --- |
| `120_aten_coverage` | rc=0，**OK 31/43，数值不符 0** | 视图族 19 个 + `contiguous`/`clone_of_view`/`add_.Scalar` 全绿（后三个是 `_copy_from` 改成按 stride 搬之后转绿的） |
| `121_view_alias` | rc=0，`VIEW_ALIAS_PASS` | 非连续源的 `contiguous()`/`clone()` 与 CPU 逐元素一致；`copy_.Tensor` 直调一致 |
| `122_fast_launch` | rc=0，`FAST_LAUNCH_PASS` | 发射 39.5→10.8 µs；256³ 189.1→110.7 µs（303.9 GFLOP/s）；aten 204.1 µs |

剩余 12 个 FAIL 全部是**内核缺口**（不是分发问题）：`sum.dim_IntList`（只支持全归约/最后一维）、
`mean.dim`、`max.dim`、`argmax`、`log_softmax`、`var.correction`、`rsqrt`、`relu`/`relu_`、
`gelu`、`masked_fill_.Scalar`。

### 12.6 那两个「静默算错」的参数契约（2026-11-07 02:05 定位并修复）

`aten::mm` 的 C++ 实现在**非方阵上算出垃圾**（`probes/009`：`mm 64x128 @ 128x256` 的
`max_rel=2.4e+04`），而 256³ 「看起来正常」。根因**不是** grid 的 X/Y 交换，而是两个参数契约：

| # | 错法 | 后果 | 修法 |
| --- | --- | --- | --- |
| 1 | kwargs 的**位置**：Triton 3.4 把 `global scratch` 指针追加在参数表**最后**，早先按「第 4 个」传（spec `"ppppiii"`） | 内核收到的 `M == 0` → 网格里所有 block 的掩码全假 → **一个元素都不写**，而 `cuLaunchKernel` 返回 0 | spec 必须是 `"pppiiip"`（scratch 在最后）；C++ 侧现在**校验 spec**，只接受 `"pppiii"` / `"pppiiip"` |
| 2 | **动态 shared 传 0**：`tl.dot` 要做布局转换，`metadata.shared` 实测 2048 B | 写进去了，但值错（`max_rel` 到 1e+01 量级），同样不报错 | `load_kernel(cubin, name, spec, shared_bytes=metadata.shared)`；`set_mm_kernel(..., shared=-1)` 可用 -1 表示「用装载时记的」，或显式覆盖 |

**为什么之前没被发现**：验收只测了 256³ 且主要看吞吐；更糟的是我在 `probes/122` 里用
「两条路径写同一个缓冲再互相比」——池会回收上一轮已经算对的块，于是「没写完」的区域
留着正确数值，**假通过**。修法是把输出所在的池块先填哨兵再释放（`probes/123`），
任何没被写到的元素都会暴露成哨兵值。

修完的真机验收（`probes/123`，`COREX_FAST_MM=1`，五个形状全覆盖，**逐元素对 CPU**）：

```
OK 64x128 @ 128x256      aten max_abs=0.000e+00 残留哨兵=0   direct 同
OK 16x16 @ 16x64         aten max_abs=0.000e+00 残留哨兵=0   direct 同
OK 256x256x256           aten max_abs=0.000e+00 残留哨兵=0   direct 同
OK 128x64 @ 64x32        aten max_abs=0.000e+00 残留哨兵=0   direct 同
OK 37x53 @ 53x29（非整除） aten max_abs=0.000e+00 残留哨兵=0   direct 同
ALL_OK
```

**教训（写进团队口径）**：「方阵 + 只报吞吐」的验收会整类漏掉 dispatch/grid/参数错误——
方阵上 X/Y 交换与维度错位都不可见，而吞吐与正确性无关。判据必须是
**形状矩阵 × 与 CPU 逐元素比对 × 污染池**，吞吐只是附加信息。这跟「只看 llc 不看 lld」
是同一类问题：判据选错，测试就变成安慰剂。

---

## 14. 全量回归（2026-11-07 02:10，最新 .so 340512 B + ENABLE_FAST_MM 默认 True）

| 探针 | rc | 结果 | 通过 / 失败 / 数值不符 |
| --- | --- | --- | --- |
| `120_aten_coverage` | 0 | `ATEN_COVERAGE_DONE` | **37/43 · 6 · 0** |
| `121_view_alias` | 0 | `VIEW_ALIAS_PASS` | 全绿 |
| `122_fast_launch` | 0 | `FAST_LAUNCH_PASS` | 全绿 |
| `123_mm_registration` | 0 | `ALL_OK` | 5 形状 × 2 路径全 0 误差 |
| `008_device_end_to_end` | 0 | `DEVICE_VERDICT PARTIAL` | 9/10 · 1 · 0 |
| `009_torch_ops` | 0 | `TORCH_OPS_VERDICT PASS` | **9/9** |
| `017_torch_ops_batch5` | 0 | `OPS5_VERDICT PASS` | **12/12** |

### 14.1 失败项定性

| 失败项 | 探针 | 定性 | 说明 |
| --- | --- | --- | --- |
| `sum.dim_IntList` / `sum.IntList_out` / `mean.dim` | 120 | **内核缺口** | 归约内核只支持「全归约」或「最后一维」，`dim=[0]` 报 `NotImplementedError` |
| `max.dim` / `argmax` | 120 | **内核缺口** | 同上，只支持最后一维（最后一维的 `max.dim`/`argmax` 已由 017 验过全绿） |
| `var.correction` | 120 | **分发缺口** | 该重载连 composite 都没有；可用 mean→平方差→mean 拼（Lead 归口） |
| 非连续**目的**张量的 `_copy_from` | 008 | **设备层缺口** | `_copy_from` 已支持非连续**源**，但目的端非连续仍报错；可复用 `_aten_extra._copy_impl` 的分段搬运 |

### 14.2 关键数字（同批实测）

- 发射地板：Triton Python **39.57 µs** → C++ **10.78 µs**（3.7x）
- mm 256³：Python 实现 183.19 µs / 183.2 GFLOP/s → C++ 直调 **170.49 µs / 196.8 GFLOP/s**
- mm 256³ 走 aten：**170.52 µs / 196.8 GFLOP/s**（与直调持平，见 §12.5 的更正）
- `009` 的 mm 64×128@128×256 与注意力小块：`max_rel = 0.000e+00` / `1.618e-07`

---

## 15. 搬运路径重构：「调用次数比字节数贵」（2026-11-07 02:45）

### 15.1 根因

`probes/022` 的 decode 是 17.1 ms/步、约 31 个算子（≈550 µs/算子），而发射地板只有 10.8 µs。
Lead 用 **p50 + 500 次**把范围锁到 `_copy_impl` 的**非连续源**：
`[4,6,16].transpose(1,2).contiguous()` 约 2 ms ≈ **5 µs/元素**。

两个原因叠加：

1. `_runtime.memcpy` 在 `stream=None` 时走**同步** `cudaMemcpy`，本平台实测**每次 5–13 µs**
   ——所以「拷贝次数」而不是「字节数」决定了耗时；
2. `_copy_impl` 原来只找「**目的**后缀连续段」。转置后目的后缀在源上不连续（`src_stride[2]=16`），
   于是 `run` 退化成 1，整个拷贝变成「每元素一次同步 memcpy」。

### 15.2 改法

| 层 | 改动 | 为什么 |
| --- | --- | --- |
| shim | 新增 `memcpy2d(dest, dest_pitch, src, src_pitch, width, height, device)`（`cudaMemcpy2D` D2D） | 同步 memcpy 每次 5–13 µs，必须把 N 条合成 1 条 |
| `_copy_impl` | 重排成 5 条路径（见下） | 每升一条路径，拷贝次数就降一个量级 |
| `cat` | 每个输入一条 `memcpy2d`（原来 `outer` 条） | KV cache 的 `[4,L,16]` 拼接 8 条 → 2 条 |
| `cat` | stride 检查改用 torch 语义：**size-1 维不约束 stride** | `[4,1,16]`、stride `(16,64,1)` 的 `is_contiguous()` 是 True，旧检查把它拒了（`probes/022` 撞到） |

`_copy_impl` 的路径顺序：

1. 两边都连续 → **1 条** `cudaMemcpy`；
2. 广播源（前导维 stride==0，`expand` 的形态）→ 每条前导下标 1 条；
3. **A** 目的后缀连续 → **1 条** `memcpy2D`（`slice+contiguous`、`cat` 相邻块）；
4. **B0** 源是均匀步长视图且目的连续 → **1 条** `memcpy2D`（`x[..., 0::2]`）；
5. **B** 源有一个最内层连续维 → `numel/该维长度` 条 `memcpy2D`（转置走这条）；
6. **C** 两侧都没有连续段 → 逐元素兜底（保留只为「不静默算错」，量级很难看）。

### 15.3 实测（`probes/124`，p50 / 500 次，带两条基线）

| 用例 | 旧 | 现在 | 门槛 |
| --- | --- | --- | --- |
| `[4,6,16].transpose(1,2).contiguous()`（384 元素） | ≈2000 µs | **91.7 µs** | ≤400（终态 ≤100，已达标） |
| `[1,64][..., 0::2].contiguous()`（32 元素） | ≈156 µs | **48.6 µs** | ≤60 |
| `cat [4,6,16]+[4,1,16] dim=1`（384 元素） | 117 µs | **26.0 µs** | ≤150 |
| `copy_` 连续源 1 KiB / 64 KiB / 1 MiB | — | 10.15 / 9.96 / 10.03 µs | 恒为 1 条 → 与大小无关 |
| `copy_` 广播源（expand，4 条） | — | 31.4 / 32.3 / 32.5 µs | 4 条 → 与大小无关 |
| `cat dim=0`（2 条） | — | 28.7 / 28.9 / 30.4 µs | 2 条 → 与大小无关 |
| 基线：空循环 / 只分配 64 元素 | — | 0.15 µs / 1.48 µs | 划清测试框架地板 |

「三档大小耗时基本不变」这一列是**关键证据**：说明这条路径已经纯开销驱动，与字节数无关。

### 15.4 对端到端的影响

`probes/022`（6 步增量 decode + KV cache）：**17.1 ms/步 → 9.2 ms/步**，判决仍 `PASS`
（逐 token logits 全局相对最差 6.62e-06、argmax 6/6 一致、KV cache 两次运行逐字节相同）。

剩下 9.2 ms/步 ÷ 31 算子 ≈ 300 µs/算子，离「纯 Python 分发地板（≈70–80 µs）」还有距离，
下一批收益在**算子个数**与 `_out_binary` 的分段搬上，不在本模块。

### 15.5 同批回归

| 探针 | 结果 |
| --- | --- |
| `120_aten_coverage` | **OK 39/43**，数值不符 0（剩 `sum.IntList_out` / `max.dim` / `argmax` / `var.correction`） |
| `009_torch_ops` | `PASS` 9/9 |
| `017_torch_ops_batch5` | `PASS` 12/12 |
| `021_axis_reduce` | `PASS` |
| `022_decode_loop` | `PASS`（9.2 ms/步） |
| `124_copy_cat_perf` | `PASS` 13/13 用例 |

---

## 16. 逐元素算子的 C++ 接管（`load_fast_op`，2026-11-07 03:25）

照 `aten::mm` 的模式，把 `add/sub/mul/div` 的 out / inplace / scalar 三条路径挪进 C++。

### 16.1 接口（由 Lead 定稿）

```python
shim.load_fast_op(op, variant, dtype, cubin, num_warps, num_stages,
                  n_ptrs, n_i32, n_f32, alpha_effective, block, name="", shared=0)
```
`name`/`shared` 是可选尾参（默认空/0）。实参表由内核组从签名**数出来**
（`_ops_elementwise.get_launch_meta()`），C++ 侧不抄默认值——抄的那份会漂移，
而漂移的表现是「跑完了但全错」。

### 16.2 三个必须记下来的坑

| 坑 | 现象 | 修法 |
| --- | --- | --- |
| **out 变体的 kernel 签名里 `out` 在最后** | `.so` 加载期直接 `terminate`（`Mismatch in kernel C++ signatures`） | `Tensor& impl(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& out)`；不是 ATen native 那种 out-first |
| **cubin 里读不出函数符号** | `load_fast_op` 报「在 cubin 的 .symtab 里找不到函数符号」，而 Python 侧 `except: continue` 把它吞掉了 → 注册表空、C++ 实现报「还没登记内核」 | 改成候选名探测：调用方给的 `name` → `_binary_kernel`/`_scalar_kernel` → ELF 符号表。实测 `_binary_kernel` 就够（`probes/126` 第 3.5 节） |
| **Python 注册会覆盖 C++ 注册** | `aten add.out` 仍是 42 µs（`_device._out_binary` 的 Python 实现把 C++ 盖住了，那条 `Overriding ... previous kernel` 警告就是它） | `shim.elementwise_op_names()` 给 Python 侧一份「C++ 已接管」清单，Python 侧必须据此跳过 |

### 16.3 实测（`probes/125`，p50 / 500 次，逐字节比对）

| 路径 | fp32 | fp16 | bf16 | 判据 |
| --- | --- | --- | --- | --- |
| `add_.Tensor`（C++） | **1.96 µs** | **1.98 µs** | **1.94 µs** | ≤12 ✓（Python 基线 78.8） |
| `sub_.Tensor` / `mul_.Tensor`（C++） | 2.07 / 1.97 µs | 2.09 / 1.95 µs | 2.05 / 1.93 µs | ✓ |
| `add.out` / `sub.out` / `mul.out` / `div.out` | 42–44 µs | 42–43 µs | 42–44 µs | ✗ 仍是 Python（被 `_device` 覆盖） |

`probes/126` 用同一份 meta 手动登记后量 `add_.Tensor`：**10.9 / 10.7 / 10.6 µs**
（含探针自己的 `copy_from`），逐字节一致——C++ 路径本身是对的。

### 16.4 仍卡住的两点（等 Lead 决定）

1. **out 变体被 `_device._out_binary` 覆盖**：Python 后注册、后生效，C++ 拿不到。
   需要在 `_device._register_ops()` 里用 `shim.elementwise_op_names()` 跳过这 8 个 schema。
2. **标量变体走不到**：`view.add_(2.5)` 实测最终派发到 `aten::add_.Tensor`，`other` 是
   **0-dim 张量**，形状与 self 不同 → C++ 的「同形」快路径拒绝。两条路可选：
   - (a) C++ 支持 0-dim 张量：要把它读成标量，得做一次 4 字节 D2H（约 10 µs），收益打折；
   - (b) 标量这 4 个 schema 留给 Python 实现（`_aten_extra` 已经能处理），
         即从 `elementwise_op_names()` 里去掉 `*.Scalar`。

### 16.5 补测（2026-11-07 03:18）：0 维张量走 host 读值，scalar 变体也进 C++

按 Lead 的选择 (a) 实现，但把「付 D2H 代价」的形态压到最小——实测 `view.add_(2.5)` 造出来的
0 维张量**在 CPU 上**，值本来就在 host 内存里，所以直接 `memcpy` 读，**一次 D2H 都不做**：

| 形态 | 处理 | p50（fp32） |
| --- | --- | --- |
| Tensor，同形连续 | tensor 变体内核 | **1.82–1.96 µs** |
| Python float 标量（`add_.Scalar` 等） | scalar 变体内核，值在 host | **11.2–11.8 µs** |
| Tensor 但 0 维（`view.add_(2.5)` → `add_.Tensor`） | 0 维在 CPU → 直接读 host；在 corex → 一次 D2H | 同 scalar 变体 |

三条纪律都落实在 `read_scalar_from_device()`：按 `other.scalar_type()` 解释字节（`add_(2.5)` 常见
是 0 维 fp32 配 fp16 self）、`n` 仍是输出 `numel`、读不了就报错不猜。

### 16.6 `.out` 为什么还是 42 µs —— 不是 C++ 慢，是**它根本没走 C++**

`probes/127` 的 dispatch 表是决定性证据：

```
aten::add.out   → PrivateUse1: registered at /dev/null:639 [kernel]          ← 走了 Python 注册
aten::add_.Tensor → PrivateUse1: registered at .../corex_shim.cpp:1793 [kernel]  ← 走了 C++
```

`/dev/null:<行号>` 是 `torch.library.Library.impl()`（Python 侧）注册的标志，`corex_shim.cpp:<行号>`
才是 `.so` 里的 `TORCH_LIBRARY_IMPL`。也就是说 `_device._out_binary` 仍把 4 个 `.out` 注册成
Python 实现，C++ 被盖住。量到的同一个 op 两条路径差 10 倍（48.0 µs vs 4.5 µs）就是它。

**排查这类问题的通用手法**：`torch._C._dispatch_dump_table("aten::xxx")` 看那一行的注册来源，
不要靠猜「是不是 C++ 里有额外开销」。

### 16.7 广播/非连续实参：物化（2026-11-07 03:32）

`.out` 的实参在真实代码里几乎不可能永远「同形 + 连续」，RoPE 就是最硬的例子：

```python
out[..., 0::2] = x1 * c - x2 * s     # out 是视图（stride (T*H,H,2)），x1/x2 也是步长视图
```

所以 C++ 侧补了三种「物化」路径，全部走池分配器、不每次 `cudaMalloc`：

| 情形 | 做法 | 实测（p50） |
| --- | --- | --- |
| `other` 单广播维（`[4,16,64] *= [4,1,64]`） | 每个广播下标一条 `memcpy2D`（行间按外层真实间距走） | **9.50 µs**（fp32/fp16 同） |
| `other` 多广播维且都在前导（`[4,16,64] *= [64]`） | 先写首块、再**倍增复制**（调用数 `repeats` → `ceil(log2(repeats))`） | **6.01 µs**（逐块循环是 512 µs） |
| 输入是步长视图（`x[..., 0::2]`） | 均匀步长 → 一条带 pitch 的 `memcpy2D` | 计入上面的数 |
| `out` 是步长视图（`out[..., 0::2]`） | 先算进连续临时张量，再按 stride 写回（均匀步长 → 一条 `memcpy2D`） | 同上 |

能力声明：`shim.elementwise_supports_broadcast()` 返回 `True`，Python 侧据此让位 `.out`。

### 16.8 让位之后的端到端

| 指标 | 让位前 | 让位后 |
| --- | --- | --- |
| `aten add.out`（p50） | 42–44 µs | **2.23 µs** |
| `add_.Tensor`（p50） | 78.8 µs（Python 实现） | **1.84 µs** |
| `add_.Scalar`（p50） | 同上 | 11.6–12.1 µs（0 维标量在 CPU，读 host，零 D2H） |
| `probes/022` decode | 17.1 → 9.2 ms/步 | **4.9 ms/步**（仍 PASS） |

`probes/125` 40 个用例剩 1 个「超门槛 0.07 µs」：`fp32 add_.Scalar` p50 = 12.07 µs / 门槛 12 µs，
数值逐字节正确，属噪声边界（同批 fp16/bf16 是 11.6/11.7 µs）。

### 16.9 一个必须记住的坑

**非连续 `out`/`other` 一开始直接报错，把 decode 打崩了**（`probes/022` 在 `apply_rope` 上
`RuntimeError: 只支持连续张量`）。教训：给快路径让位之前，要先确认它的**能力覆盖**真实调用形态——
同形+连续只是最简单的一档，广播（`cos[:, None, :]`）、步长视图（`0::2`）、非连续输出，
在 decode 里每一个都会出现。判据不是「我实现了哪个」，是「真实调用会不会踩到没实现的那个」。
