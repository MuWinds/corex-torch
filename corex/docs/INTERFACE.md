# 接口：分配器、guard、device module、`DEVICE_NAME` 别名、aten 接管边界

这份文档回答：**要让 torch 2.8 认一个 `cuda`（其实是 corex）设备，必须实现哪些接口、各自契约是什么、算子接管的边界怎么划。**

证据与命令在 [`00-FACTS.md`](00-FACTS.md) §9–§11、§20；路线与不做什么在 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

---

## 1. 总览：三处注册 + 一个别名 + 一层算子

| 件 | 谁注册 | 缺了会怎样 |
| --- | --- | --- |
| `c10::Allocator` | C++，`c10::SetAllocator(PrivateUse1, …)` | `GetAllocator` 返回 **nullptr、不报错**，之后解空指针 |
| `DeviceGuardImplInterface` | C++，`C10_REGISTER_GUARD_IMPL(PrivateUse1, …)`（静态） | `PyTorch is not linked with support for corex devices` |
| device module `torch.corex` | Python，`torch._register_device_module(DEVICE_NAME, module)` | `ModuleNotFoundError: No module named 'torch.corex'` |
| 设备名 `"cuda"` | 编译期补丁（C++ 3 处 + Python 4 处），最后 `rename_privateuse1_backend("cuda")` | 客户代码 `model.cuda()` / `device="cuda"` 走不进我们的后端 |
| 算子实现 | 三条并存：C++ `TORCH_LIBRARY_IMPL`、Python `torch.library`、composite + CPU 兜底 | `NotImplementedError: Could not run 'aten::X' with arguments from the 'corex' backend` |

## 2. 分配器（C++）

`c10::Allocator` 在 2.8 有两个必须照做的签名（照 2.1 写会安不上）：

```cpp
struct Allocator {
  virtual DataPtr allocate(size_t n) = 0;                       // 注意：不是 const
  virtual void copy_data(void*, const void*, std::size_t) const = 0;  // 2.8 新增纯虚
};
```

契约要点：

- **`DataPtr` 的 device 必须写 `c10::Device(PrivateUse1, idx)`**——`TensorImpl` 是从它读 device 的，
  写错了张量会被当成 CPU 张量。
- `SetAllocator` 传进去的指针**必须是静态生命周期**，函数不接管所有权；**非线程安全**，
  只在初始化时调一次。
- **缓存池是必需的，不是优化**：平台没有 memory pool（`cudaMallocAsync` 连声明都没有），
  `cudaMalloc` 是真分配（3–12 µs/次）。实现是 2 的幂桶（最小 512 B，13 档到 2 MiB），
  大块按 2 MiB 整数倍直通；`empty_cache()` 才还回去。
- 页大小编译期固定 **2 MiB**（KMD 的 `PG_MODE=2M`）——探也只有一个答案。
- `slab_alignment()` 是观测接口：实测返回 16 MiB / 32 MiB（两次运行不同），
  说明 `cudaMalloc` 的返回地址远高于 2 MiB 粒度，按 2 的幂切分不会破坏对齐。

**多设备的硬规则**：`cudaMemcpy` 在「当前 runtime 设备」与指针所属设备不一致时**不报错、rc=0，
而是静默别名**（把 1 号卡的指针当成 0 号卡的同一偏移）。所以每次拷贝前必须 `setDevice`，
这条写进了 shim 的 `memcpy2d` 与池的分配路径。

## 3. guard（C++）

`c10::impl::DeviceGuardImplInterface` 的 8 个纯虚（2.8 行号见 [`00-FACTS.md`](00-FACTS.md) §9）：

```
type / exchangeDevice / getDevice / setDevice / uncheckedSetDevice / getStream / exchangeStream / deviceCount
```

- 注册宏是 `C10_REGISTER_GUARD_IMPL(PrivateUse1, XxxGuardImpl)`，它是**静态对象构造**，加载即生效。
- `getStream`/`exchangeStream` 用的是 `c10::Stream`；本仓库把 CoreX 的流句柄塞进
  `Stream::UNSAFE`，`StreamId` 不做语义解释（默认流返回 0）。
- 一个坑：`PyTorch is not linked with support for corex devices` 这条报错来自
  `getDeviceGuardImpl` 的 `TORCH_CHECK`，不是链接错误——看到它先查注册，别去查 `DT_NEEDED`。

## 4. device module（Python）

`torch.<name>` 必须是**一个真的能 import 的模块**（C++ 的 lazy init 会 `import torch.corex`），
不是只塞 `sys.modules`。两种落地方式都验过：`torch._register_device_module("corex", module)`，
或把目录挂进 `torch.__path__` 让 `import torch.corex` 真能成功。本仓库两种都做（后者保证
`__file__` 指向我们的目录）。

实际会被 torch 摸到的函数（`probes/106` 用「记录属性访问」的替身模块抓的）：

| 入口 | 需要的函数 | 备注 |
| --- | --- | --- |
| `is_available()` | `is_available` | 必需 |
| `accelerator.device_count()` | `device_count` | 必需 |
| `accelerator.current_device_index()` | **`_lazy_init`** | **文档没列，但实测会被调** |
| `manual_seed(…)` | `_is_in_bad_fork` → `manual_seed_all` | 要支持 seed 就必需 |
| `tensor.cuda()` / `.corex()` | `current_device` | 用 `generate_methods_for_privateuse1_backend()` 时必需 |
| `torch.empty/zeros/tensor/randn(device=…)` | **一个都不摸** | 不需要 |

另外两条：`rename_privateuse1_backend` 每进程只能调一次，且必须在任何张量创建之前；
`get_device_properties` **不是** PrivateUse1 的必需项（它是 `torch.cuda` 自己的约定），
但客户代码会调，所以我们在 device module 里提供了同名函数。

## 5. `DEVICE_NAME = "cuda"` 别名

`torch/corex/*` 里 `DEVICE_NAME = "cuda"`（`_device.py:41`、`_aten_kernels.py:28`、`__init__.py:42`）。
它不是「改个字符串」那么简单，背后是 9 个文件约 150 行的补丁（清单见
[`00-FACTS.md`](00-FACTS.md) §20）。三条契约：

1. **两个名字都解析到 PrivateUse1**：`"cuda"` 与 `"corex"` 都通（老代码 `device="corex"` 继续可用），
   但 `torch.device("corex").type == "cuda"`——因为注册名只有一个。
2. **C++ 侧的 `is_cuda()` 必须跟着改**：`Device::is_cuda()` 与 `TensorImpl::is_cuda()` 都要对
   PrivateUse1 返回真。客户代码里大量的 `if x.is_cuda:` 才走对支路；只改字符串不改变这个判断。
3. **`torch.cuda.*` 是委托，不是实现**：`torch/cuda/__init__.py` 末尾把
   `is_available/device_count/current_device/set_device/synchronize/Stream/Event/current_stream`
   转给 `torch.corex`，并把 `Tensor.cuda()` 接到 `.to("cuda")`（C++ 的 `at::Tensor::cuda()`
   用枚举而不是字符串，绕不过别名）。`torch.cuda.amp`、`torch.cuda.CUDAGraph` 这类
   「必须真有 CUDA」的功能**没有**实现。

**别和 Triton 的后端名混**：triton 那边的目标名仍然是 `corex`（`GPUTarget("corex", 70, 32)`）。

## 6. shim 的 Python API

`from torch.corex import _corex_shim as shim`（模块名取自 `_device.SHIM_MODULE`；**必须先 `import torch`**，否则 `libc10.so` 找不到）。
导出个数**以源码 `m.def(` 计数为准**（当前 22 处，见 [`00-FACTS.md`](00-FACTS.md) §11）。契约：

```python
shim.init(0)                     # 幂等；绑定设备，SetAllocator + guard impl 已静态注册
shim.is_initialized()            # -> bool
shim.device_count()              # -> int
shim.empty_strided(sizes, strides, nbytes, dtype_int, device=0)
shim.from_blob(ptr, sizes, strides, dtype_int, device=0)     # 不接管所有权
shim.reserved_bytes() / allocated_bytes() / empty_cache() / slab_alignment()
shim.load_kernel(cubin, name, num_warps, num_stages, shared, spec)
shim.launch(...)                 # 参数表见下节的 spec 契约
shim.set_mm_kernel(dtype, cubin, ...) / mm_kernel_dtypes() / mm_kernel_info(dtype)
shim.load_fast_op(op, variant, dtype, cubin, num_warps, num_stages,
                  n_ptrs, n_i32, n_f32, alpha_effective, block, name="", shared=0)
shim.elementwise_supports_broadcast() / elementwise_op_names() / elementwise_kernel_info(op, variant, dtype)
shim.memcpy2d(dst, dst_pitch, src, src_pitch, width, height)  # 带 setDevice 保护
```

- `dtype` 传 `c10::ScalarType` 的**整数**：`int(torch.float32) == 6`，不必自建映射表。
- `empty_strided`/`from_blob` 的 strides **原样**写进 TensorImpl，否则 `view`/`transpose` 会静默算错。
- 模块名是 **`_corex_shim`**（`PyInit_` 符号由文件名决定），C++ 命名空间仍叫 `corex_shim`。

## 7. aten 接管边界

三条注册方式并存，**优先级由「谁后注册」决定**——这是最容易踩的地方：

| 方式 | 标志 | 用途 | 限制 |
| --- | --- | --- | --- |
| C++ `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` | dispatch 表里显示 `corex_shim.cpp:<行号>` | 热路径（`add/sub/mul/div` 的 out/inplace/scalar、`mm`） | 只能手写 schema/实参顺序，写错就是「跑完但全错」 |
| Python `torch.library.Library("aten","IMPL").impl(...)` | 显示 `/dev/null:<行号>` | 大量算子、原型验证 | **后注册会盖住 C++**（`Overriding … previous kernel` 警告就是它） |
| composite + `cpu_fallback` | 走 `_to_copy` 往返 | 未实现算子、冷门算子 | 慢且会产生隐式拷贝，必须记账 |

**让位协议**：C++ 声明能力，Python 据此跳过。当前两个闸门是
`shim.elementwise_supports_broadcast()`（广播/步长视图物化能力）与 `shim.elementwise_op_names()`
（「C++ 已接管」清单）。判据不能写成「我实现了哪个」，要写成「真实调用会不会踩到没实现的那个」——
`.out` 第一版只支持「同形 + 连续」就让位，把 decode 的 `apply_rope` 直接打崩了（三个实参都是视图），
见 [`00-FACTS.md`](00-FACTS.md) §17。

**双键规则**：`CompositeImplicitAutograd` 类算子（`rms_norm`、`sdpa`、`layer_norm`）必须**同时**挂
`PrivateUse1` 与 `AutogradPrivateUse1`，只挂前者会停在 `Autogradcorex` 层报「算子不存在」。

**归属判据只有 dispatch 表**：

```python
torch._C._dispatch_dump_table("aten::add.out")     # 看那一行的注册来源，不要靠猜
```

## 8. 算子层的位置参数契约（`_device.py` ↔ `_ops_*`）

`torch/corex/_ops_elementwise.py` / `_ops_reduce.py` / `_ops_matmul.py` / `_ops_index.py`
**刻意不 import torch**：它们只认「设备指针 + 形状 + dtype 整数」，这样内核层可以脱离框架单测
（`probes/199` 用一个**不许用内部知识**的 shim 模拟器验了 24 组，逐字节正确）。

- **位置参数不能改**：调用方（`_device.py`）按固定顺序传，改了不会报错，只会「算错」。
- 实参顺序表由内核组从签名**数出来**（`get_launch_meta()` → `n_ptrs/n_i32/n_f32/alpha_effective/block`），
  C++ 侧**不抄默认值**——抄的那份会漂移，而漂移的表现就是「跑完了但全错」。
- 两个概念不能混：`has_alpha`（签名里有没有 alpha 槽位）与 `alpha_effective`（alpha 是否真改变结果）。
  mul/div 的槽位**仍然必须传**（传 0.0 与 1.0 输出逐字节相同），少传就是实参个数不符 → 参数错位。
- launch spec：Triton 3.4 把 global scratch 指针追加在参数表**最后**，所以 spec 是
  **`"pppiiip"`**（3 指针 + 3 个 i32 + 末尾 scratch），动态 shared 必须带 `metadata.shared`。
  两个都错的后果分别是「一个元素都不写」与「写了但值错」，都不报错。

## 9. 失败梯度（照这个顺序排查）

| 已注册到 | `torch.empty(4, device="cuda")` 的表现 |
| --- | --- |
| 只 `rename` | `ModuleNotFoundError: No module named 'torch.corex'` |
| + device module | `NotImplementedError: Could not run 'aten::empty.memory_format' with arguments from the 'corex' backend` |
| + C++ 分配器 & guard impl & `empty.memory_format` 实现 | 成功，`device=cuda:0` |

再往后每一次「算子未实现」都是同一句话的不同算子名；**不要**把它当成设备层问题。
