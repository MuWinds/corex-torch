"""探针 102：在 torch 2.8 CPU wheel 之上，用 ``torch.utils.cpp_extension`` 编一个最小
C++ shim，验证「内存分配器能不能绕过 C++」这个问题。

为什么做这个探测：101 已经证明纯 Python 走到 ``ModuleNotFoundError`` / ``NotImplementedError``
就没路了，但「没路」不等于「必须写 C++」——也有可能是我们注册姿势不对。所以这里正面试一次：

1. 先确认登录节点能编 torch 扩展（系统 g++，不碰卡）；
2. 再编一个真的 shim：``c10::Allocator`` + ``c10::impl::DeviceGuardImplInterface``
   + ``C10_REGISTER_GUARD_IMPL`` + ``c10::SetAllocator``，然后从 Python 只注册
   ``aten::empty.memory_format`` 的 PrivateUse1 实现，看 ``torch.empty(4, device="corex")``
   会不会真的返回一个张量。

这台机器上没有卡，所以分配器用 host ``malloc`` 顶着——本探针要验的是**注册通路**，
不是最终的内存来源。最终实现要走 CoreX 的 ``cuMemAlloc``（沙箱里做）。

跑法（在 corex28 venv 里，登录节点直接跑；编译约 1 分钟）::

    ~/venvs/corex28/bin/python probes/102_torch28_cpp_ext.py --stage build
    ~/venvs/corex28/bin/python probes/102_torch28_cpp_ext.py --stage shim

Module: ``corex-torch/probes``。
"""

from __future__ import annotations

import ctypes
import sys
import time

BACKEND = "corex"

#: 最小 C++ shim。只做四件事：host 分配器、guard impl 的八个纯虚函数、
#: 用 StorageImpl + TensorImpl 手搓一个 PrivateUse1 张量、把三者注册进 c10。
#: 注释写清「为什么」：这些都是 c10 的 ABI，写法错了会静默出错而不是编不过。
SHIM_CPP = r"""
#include <torch/extension.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/Stream.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <ATen/core/TensorBody.h>

#include <cstdlib>
#include <cstring>

namespace corex_shim {

// 分配器：数据指针的 device 必须写成 PrivateUse1，TensorImpl 是从这里读 device 的；
// 写成 CPU 会让张量变成 cpu 设备，后面所有 dispatch 全错——这一步静默，不报错。
//
// 两个签名坑（torch 2.8 与 2.1 不同，实测编译错才发现的）：
//  1) allocate 在 2.8 里**不是** const 成员（写成 const 会报 marked override but does not override）；
//  2) 2.8 的 Allocator 多了一个纯虚 copy_data(dest, src, count) const，不实现就编成抽象类。
struct HostAllocator final : public c10::Allocator {
  c10::DataPtr allocate(size_t n) override {
    void* p = n ? std::malloc(n) : nullptr;
    return c10::DataPtr(p, p, &std::free, c10::Device(c10::DeviceType::PrivateUse1, 0));
  }
  void copy_data(void* dest, const void* src, std::size_t count) const override {
    if (count) {
      std::memcpy(dest, src, count);
    }
  }
  c10::DeleterFnPtr raw_deleter() const override { return &std::free; }
};

// guard impl：八个纯虚函数见 c10/core/impl/DeviceGuardImplInterface.h 的 `= 0;`。
// 真实后端要在这里调 cuSetDevice / cuStreamCreate；现在只是记录一个线程局部的设备号，
// 流统一返回 StreamId 0（CoreX 的默认流语义要等沙箱里实测，不能照抄 CUDA）。
struct HostGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  c10::DeviceType type() const override { return c10::DeviceType::PrivateUse1; }

  c10::Device exchangeDevice(c10::Device d) const override {
    auto old = c10::Device(c10::DeviceType::PrivateUse1, index());
    index() = d.index();
    return old;
  }
  c10::Device getDevice() const override {
    return c10::Device(c10::DeviceType::PrivateUse1, index());
  }
  void setDevice(c10::Device d) const override { index() = d.index(); }
  void uncheckedSetDevice(c10::Device d) const noexcept override { index() = d.index(); }
  c10::Stream getStream(c10::Device d) const override {
    return c10::Stream(c10::Stream::UNSAFE, d, 0);
  }
  c10::Stream exchangeStream(c10::Stream s) const override {
    return c10::Stream(c10::Stream::UNSAFE, s.device(), 0);
  }
  c10::DeviceIndex deviceCount() const noexcept override { return 1; }

  static c10::DeviceIndex& index() {
    static thread_local c10::DeviceIndex idx = 0;
    return idx;
  }
};

// 静态注册：加载 .so 的那一刻把 guard impl 塞进 device_guard_impl_registry。
// 102 的实测里，缺了这一步的后果是 `RuntimeError: PyTorch is not linked with
// support for corex devices`（见 101 stage 3）。
C10_REGISTER_GUARD_IMPL(PrivateUse1, HostGuardImpl);

void register_allocator() {
  // priority=1：允许覆盖可能已存在的低优先级默认分配器（Allocator.h 的注释：
  // 只有不低于既有优先级的才能覆盖）。
  static bool once = [] {
    c10::SetAllocator(c10::DeviceType::PrivateUse1, new HostAllocator(), /*priority=*/1);
    return true;
  }();
  (void)once;
}

// 手搓张量：上游 CPU wheel 里没有把 aten 的 empty_generic 头文件发出来
// （include/ATen/native/EmptyTensor.h 不存在，但符号 at::detail::empty_generic
// 在 libtorch_cpu.so 里是导出的），所以这里不依赖它，直接按 c10 的 ABI 组装。
at::Tensor empty_pu1(std::vector<int64_t> sizes, c10::ScalarType dtype) {
  register_allocator();
  auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
  auto dtype_meta = c10::scalarTypeToTypeMeta(dtype);
  int64_t nbytes = static_cast<int64_t>(dtype_meta.itemsize());
  for (auto s : sizes) {
    nbytes *= s;
  }
  c10::Storage storage(c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), nbytes, allocator, /*resizable=*/true));
  auto t = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      dtype_meta);
  t.unsafeGetTensorImpl()->set_sizes_contiguous(c10::IntArrayRef(sizes));
  return t;
}

// 从 Python 侧把张量里的数据读出来，证明分配器真的把内存交出去了。
int64_t tensor_data_ptr(const at::Tensor& t) {
  return reinterpret_cast<int64_t>(t.data_ptr());
}

void register_shim() {
  register_allocator();
}

// 故意不做任何注册，直接问 c10 要 PrivateUse1 的分配器：用来判断「不调 SetAllocator
// 会怎样」——这是「分配器必须由 C++ 注册」最直接的证据（Python 侧根本没有这个入口）。
int64_t get_allocator_ptr() {
  auto* allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
  return reinterpret_cast<int64_t>(allocator);
}

}  // namespace corex_shim

// load_inline 生成的 pybind 绑定代码写在全局作用域里，所以要把导出的符号引到全局；
// 不引会报 "'empty_pu1' was not declared in this scope; did you mean 'corex_shim::...'"。
using corex_shim::empty_pu1;
using corex_shim::get_allocator_ptr;
using corex_shim::register_shim;
using corex_shim::tensor_data_ptr;
"""

MINIMAL_CPP = r"""
#include <torch/extension.h>
int64_t triple(int64_t x) { return 3 * x; }
"""


def _build_minimal() -> None:
    """第一关：系统 g++ 能不能编 torch 扩展（与设备无关，纯工具链验证）。"""
    from torch.utils import cpp_extension

    print("torch", cpp_extension.torch.__version__)
    print("include_paths:", cpp_extension.include_paths()[:3], "...")
    print("library_paths:", cpp_extension.library_paths()[:3], "...")
    t0 = time.time()
    mod = cpp_extension.load_inline(
        name="corex_probe_min",
        cpp_sources=MINIMAL_CPP,
        functions=["triple"],
        extra_cflags=["-O2"],
        verbose=True,
    )
    dt = time.time() - t0
    print(f"BUILD_MIN_OK triple(14)={mod.triple(14)} 编译耗时={dt:.1f}s")
    print("build dir:", cpp_extension._get_build_directory("corex_probe_min", verbose=False))


def _build_shim():
    """第二关：编真正的 shim（分配器 + guard impl + 手搓张量）。"""
    from torch.utils import cpp_extension

    t0 = time.time()
    mod = cpp_extension.load_inline(
        name="corex_probe_shim",
        cpp_sources=SHIM_CPP,
        functions=["register_shim", "empty_pu1", "tensor_data_ptr", "get_allocator_ptr"],
        extra_cflags=["-O2", "-std=c++17"],
        verbose=True,
    )
    print(f"BUILD_SHIM_OK 编译耗时={time.time() - t0:.1f}s")
    return mod


def _try(label: str, fn) -> object:
    """跑一段并打印结果；异常只记类型与首行。

    注意：成功分支里的 ``repr()`` 也要兜住——PrivateUse1 张量的 ``repr`` 会走
    ``aten::view`` / ``aten::item``，这些算子还没实现，直接把探针打断（实测踩过）。
    """
    try:
        value = fn()
    except BaseException as exc:  # noqa: BLE001 - 就是要抓所有失败形态
        first = str(exc).strip().splitlines()
        print(f"  [{label}] EXC {type(exc).__name__}: {(first[0] if first else '')[:400]}")
        return None
    try:
        shown = repr(value)
    except BaseException as exc:  # noqa: BLE001 - repr 依赖算子，可能也失败
        shown = f"<{type(value).__name__} 对象，repr 失败: {type(exc).__name__}>"
    print(f"  [{label}] OK -> {shown}"[:400])
    return value


def _fake_device_module():
    """最小 device module：只为让 torch 的 factory 函数能通过 module 查找。"""
    import torch

    class FakeCorexModule:
        """桩模块，接口清单来自 rename_privateuse1_backend 的 docstring。"""

        is_available = staticmethod(lambda: True)
        device_count = staticmethod(lambda: 1)
        current_device = staticmethod(lambda: 0)
        _is_in_bad_fork = staticmethod(lambda: False)
        manual_seed_all = staticmethod(lambda seed: None)
        set_rng_state = staticmethod(lambda state, device=None: None)
        _lazy_init = staticmethod(lambda: None)

        @staticmethod
        def get_rng_state(device=None):
            return torch.zeros(1, dtype=torch.uint8)

        @staticmethod
        def get_amp_supported_dtype():
            return [torch.float32]

    return FakeCorexModule


def stage_shim() -> int:
    """按顺序：只注册 C++ shim → 加 device module → 加 empty 的 Python 实现。"""
    import torch

    mod = _build_shim()
    mod.register_shim()
    print("--- 1. 只注册 C++ shim（分配器 + guard impl），未注册 device module ---")
    torch.utils.rename_privateuse1_backend(BACKEND)
    _try("torch.empty(4, device='corex')", lambda: torch.empty(4, device=BACKEND))

    print("--- 2. 加上 device module ---")
    torch._register_device_module(BACKEND, _fake_device_module())
    _try("torch.empty(4, device='corex')", lambda: torch.empty(4, device=BACKEND))
    _try("t.to('corex')", lambda: torch.zeros(2).to(BACKEND))
    _try("accelerator.is_available()", lambda: torch.accelerator.is_available())

    print("--- 3. 再加 Python 侧的 aten::empty.memory_format 实现（内部调 C++ shim）---")
    lib = torch.library.Library("aten", "IMPL")

    def _empty_impl(size, *, dtype=None, layout=None, device=None,
                    pin_memory=None, memory_format=None):
        dt = torch.float32 if dtype is None else dtype
        return mod.empty_pu1(list(size), dt)

    _try(
        "lib.impl('empty.memory_format')",
        lambda: lib.impl("empty.memory_format", _empty_impl, "PrivateUse1"),
    )
    t = _try("torch.empty(2, 3, device='corex')", lambda: torch.empty(2, 3, device=BACKEND))
    if t is not None:
        print(f"  张量属性: device={t.device} dtype={t.dtype} shape={tuple(t.shape)} "
              f"contiguous={t.is_contiguous()} numel={t.numel()}")
        ptr = mod.tensor_data_ptr(t)
        print(f"  data_ptr=0x{ptr:x}（host 虚拟地址，说明分配器真的被调用了）")
        # 用 ctypes 往这块内存写 8 个 float，再读回来：证明它是一块可用内存，
        # 而不是空壳 DataPtr。
        buf = (ctypes.c_float * 6)(1.5, -2.5, 3.25, 4.75, 5.5, 6.0)
        ctypes.memmove(ptr, buf, ctypes.sizeof(buf))
        back = (ctypes.c_float * 6).from_address(ptr)
        print(f"  写回读出 = {[round(v, 3) for v in back]}")
        _try("t.device.type", lambda: t.device.type)
        _try("t.to('cpu')", lambda: t.to("cpu"))
        _try("t + 1", lambda: t + 1)
        _try("t.sum()", lambda: t.sum())
        _try("torch.zeros(4, device='corex')", lambda: torch.zeros(4, device=BACKEND))

    print("--- 4. 结论口径 ---")
    return 0


def stage_alloc() -> int:
    """不注册任何东西，先问 c10 要 PrivateUse1 的分配器；再看注册之后是什么。

    这一段单独起进程的原因：一旦 ``register_shim()`` 跑过，分配器就被塞进 c10 的全局表，
    后面再问就永远是有值的了。
    """
    mod = _build_shim()
    _try("注册前的 c10::GetAllocator(PrivateUse1)", mod.get_allocator_ptr)
    mod.register_shim()
    ptr = _try("注册后的 c10::GetAllocator(PrivateUse1)", mod.get_allocator_ptr)
    print(f"  （非 0 即表示 c10 的分配器表里已经有 PrivateUse1 的表项：{ptr}）")
    return 0


def main(argv: list[str]) -> int:
    if "--stage" in argv:
        stage = argv[argv.index("--stage") + 1]
        if stage == "build":
            _build_minimal()
            return 0
        if stage == "shim":
            return stage_shim()
        if stage == "alloc":
            return stage_alloc()
        raise SystemExit(f"未知 stage: {stage}")
    # 默认三关都跑（各自独立进程，避免 rename 一次性的限制互相干扰）。
    import subprocess

    rc = 0
    for stage in ("build", "alloc", "shim"):
        print(f"===== stage {stage} =====")
        p = subprocess.run([sys.executable, __file__, "--stage", stage],
                           capture_output=True, text=True, timeout=900)
        print(p.stdout.rstrip())
        if p.returncode != 0:
            rc = p.returncode
            print(p.stderr.rstrip()[-4000:])
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
