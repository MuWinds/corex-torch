// corex_shim.cpp —— 把 CoreX（天数智芯 天垓100）作为 torch 2.8 的 PrivateUse1 设备接进去的薄 C++ 层。
//
// 为什么需要这一层（结论有实测支撑，见 docs/03-TORCH28-API.md §0）：
//   torch 2.8 的 PrivateUse1 有两个入口**只存在于 C++**，Python 侧没有绑定：
//     1) `c10::SetAllocator(DeviceType::PrivateUse1, ...)` —— `torch._C` 上 privateuse 开头的
//        只有 `_get_privateuse1_backend_name` / `_rename_privateuse1_backend`，没有分配器入口；
//     2) `C10_REGISTER_GUARD_IMPL(PrivateUse1, ...)` —— 缺它时 `.to("corex")` 报
//        `RuntimeError: PyTorch is not linked with support for corex devices`
//        （就是 `c10::impl::getDeviceGuardImpl` 里的 TORCH_CHECK）。
//   除此之外的算子分发**可以**放在 Python（`torch.library.Library("aten","IMPL").impl`），
//   所以这一层的职责只有三件：分配器、guard impl、把裸指针包成 Tensor 的 factory。
//
// 为什么不 include 任何 CUDA 头、也不链 libcudart：
//   **编译必须发生在登录节点**（没有 /usr/local/corex、没有 GPU 驱动），而运行时库只在计算
//   节点上存在。所以所有 `cuda*` 入口一律走 `dlopen`/`dlsym`（见 CorexApi）。
//
// 为什么分配器要自带缓存池（R7，docs/05-DRIVER-CONSTRAINTS.md）：
//   本平台没有 memory pool（`cudaMallocAsync` 连声明都没有），`cudaMalloc` 是**真分配**：
//   页大小编译期固定 2 MiB、走 KMD 的 CreateAllocation。aten 会频繁申请/释放小块张量，
//   把 `cudaMalloc` 直接挂到分配器上会被分配开销吃掉。所以按 2 的幂分桶、大块一次拿、
//   桶内切分，`empty_cache()` 才还给驱动。
//
// 两条硬禁令（AGENTS.md §2）：不碰 CUDA graph、不碰 cp.async。本文件不出现任何二者。

#include <torch/extension.h>

#include <ATen/core/TensorBody.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/Stream.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace corex_shim {
namespace {

// 常量：页大小与分桶下界

/// 驱动侧页大小，编译期固定 2 MiB（`PG_MODE=2M`）。
///
/// 为什么写死而不是运行时探测：页大小在本平台是 KMD 编译期常量，探也只有一个答案；
/// 写死可以让分配器在 init() 之前就能算。
constexpr size_t kPageBytes = 2u * 1024u * 1024u;

/// 桶的最小粒度 512 B。比它小的请求一律按 512 B 记账——更低尺寸的碎片（标量、索引、
/// mask）再细分的元数据开销会超过收益。
constexpr size_t kMinBucketBytes = 512u;

/// 大块直通阈值：超过一页的请求不进桶，直接 `cudaMalloc` 一个页的整数倍。
/// 理由：跨页切分会让「一块板一个桶」的记账变复杂，而大张量本来就不多。
constexpr size_t kBigThresholdBytes = kPageBytes;

/// 桶的数量：512 B 起 2 的幂，到 2 MiB，共 13 个。
constexpr int kBucketCount = 13;

/// 桶下标 → 块大小。
size_t bucket_size(int bucket) {
  return kMinBucketBytes << bucket;
}

/// 申请字节数 → 桶下标；超过一页返回 `-1`（走大块直通）。
int bucket_for(size_t nbytes) {
  if (nbytes > kBigThresholdBytes) {
    return -1;
  }
  int bucket = 0;
  size_t size = kMinBucketBytes;
  while (size < nbytes && bucket + 1 < kBucketCount) {
    size <<= 1;
    ++bucket;
  }
  return bucket;
}

// 运行时绑定：dlopen libcudart，一切 cuda* 走 dlsym

/// CoreX 运行时函数表。字段名与 CUDA runtime 一致，方便和 `_runtime.py` 的 ctypes 表对照。
struct CorexApi {
  int (*cudaSetDevice)(int) = nullptr;
  int (*cudaGetDeviceCount)(int*) = nullptr;
  int (*cudaMalloc)(void**, size_t) = nullptr;
  int (*cudaFree)(void*) = nullptr;
  int (*cudaMemcpy)(void*, const void*, size_t, int) = nullptr;
  int (*cudaDeviceSynchronize)() = nullptr;
  int (*cudaStreamSynchronize)(void*) = nullptr;
  /// `cudaGetErrorString` 返回 `const char*`，单独用正确签名的函数指针接。
  const char* (*cudaGetErrorStringStr)(int) = nullptr;
};

/// 全局唯一的函数表。`init()` 之前是空的，所有使用点都必须先过 `api()`。
CorexApi g_api;

/// 是否已经成功 dlopen 过。
bool g_api_loaded = false;

/// `init()` 是否跑完（幂等判断）。
bool g_initialized = false;

/// 保护 `g_api_loaded` / `g_initialized` / `g_api` 的互斥量。
std::mutex g_mutex;

/// 把 `dlsym` 的 `void*` 转成函数指针。
///
/// 为什么要这个模板：C++ 标准不允许 `void*` 直接转函数指针；集中一处 `reinterpret_cast`，
/// 别的调用点就不用重复写。
template <typename Fn>
Fn bind_symbol(void* handle, const char* name) {
  void* sym = ::dlsym(handle, name);
  TORCH_CHECK(sym != nullptr, "corex_shim: libcudart 里找不到符号 ", name,
              "（CoreX 3.2.1 的兼容层应该导出它，见 docs/05-DRIVER-CONSTRAINTS.md §0）");
  return reinterpret_cast<Fn>(sym);
}

/// 候选库路径。顺序：`$COREX_HOME/lib64` → `$COREX_HOME/lib` → 裸 soname。
///
/// 为什么最后要试裸 soname：沙箱里 runner 可能已经把 libcudart 放进 LD_LIBRARY_PATH，
/// 这时按名字 `dlopen` 能命中，而且与 Python 侧 ctypes 打开的是同一个库实例。
std::vector<std::string> cudart_candidates() {
  const char* env = std::getenv("COREX_HOME");
  const std::string home =
      (env != nullptr && env[0] != '\0') ? std::string(env) : std::string("/usr/local/corex");
  const char* soname = "libcudart.so.10.2";
  return {home + "/lib64/" + soname, home + "/lib/" + soname, soname};
}

/// 打开 libcudart 并绑定全部函数（前置条件：`g_mutex` 已持）。
///
/// 失败时把**试过的每个路径**都写进报错——登录节点没有 /usr/local/corex，
/// 这种时候「找不到库」的原文必须能直接告诉人缺的是什么。
void load_api_locked() {
  if (g_api_loaded) {
    return;
  }
  std::string tried;
  void* handle = nullptr;
  for (const std::string& path : cudart_candidates()) {
    handle = ::dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle != nullptr) {
      tried += " [hit] " + path;
      break;
    }
    tried += " [miss] " + path;
  }
  TORCH_CHECK(handle != nullptr,
              "corex_shim: 打不开 CoreX 运行时 libcudart.so.10.2。试过：", tried,
              "。用环境变量 COREX_HOME 指向 CoreX 安装根目录（默认 /usr/local/corex）。");

  g_api.cudaSetDevice = bind_symbol<int (*)(int)>(handle, "cudaSetDevice");
  g_api.cudaGetDeviceCount = bind_symbol<int (*)(int*)>(handle, "cudaGetDeviceCount");
  g_api.cudaMalloc = bind_symbol<int (*)(void**, size_t)>(handle, "cudaMalloc");
  g_api.cudaFree = bind_symbol<int (*)(void*)>(handle, "cudaFree");
  g_api.cudaMemcpy = bind_symbol<int (*)(void*, const void*, size_t, int)>(handle, "cudaMemcpy");
  g_api.cudaDeviceSynchronize = bind_symbol<int (*)()>(handle, "cudaDeviceSynchronize");
  g_api.cudaStreamSynchronize = bind_symbol<int (*)(void*)>(handle, "cudaStreamSynchronize");
  g_api.cudaGetErrorStringStr = bind_symbol<const char* (*)(int)>(handle, "cudaGetErrorString");
  g_api_loaded = true;
}

/// 取函数表；未加载则先加载（guard impl 可能在任何时刻被 torch 调到）。
const CorexApi& api() {
  std::lock_guard<std::mutex> lock(g_mutex);
  load_api_locked();
  return g_api;
}

/// 把 runtime 返回码翻成带错误串原文的异常。
///
/// 为什么单独抽出来：CoreX 有「返回 0 但不改出参」的接口（`_runtime.py` 的 Constraints），
/// 所以关键出参还要调用方自己再断言一次，不能只信 rc。
void check_rc(const char* what, int rc) {
  if (rc == 0) {
    return;
  }
  const char* msg = nullptr;
  if (g_api_loaded && g_api.cudaGetErrorStringStr != nullptr) {
    msg = g_api.cudaGetErrorStringStr(rc);
  }
  TORCH_CHECK(false, "corex_shim: ", what, " 失败 rc=", rc, " ", msg != nullptr ? msg : "");
}

// 每线程的设备与流

/// 每个线程当前的设备号。
///
/// 为什么要有这个东西：`c10::Allocator::allocate(size_t)` **没有设备参数**，torch 的约定是
/// 「按当前设备分配」，当前设备来自 DeviceGuardImplInterface 的 thread-local。所以分配器
/// 必须问自己这一份，而不是猜 0 号卡。
thread_local c10::DeviceIndex tls_device_index = 0;

/// 每个线程当前的流句柄。0 = 默认流（即 `cudaStream_t` 的 nullptr）。
///
/// 为什么用 `c10::Stream::id()` 直接存裸句柄：StreamId 是 int64，本平台 `cudaStream_t`
/// 是 8 字节指针，塞得进去且无损。
thread_local int64_t tls_stream_handle = 0;

/// 上一回真的调过 `cudaSetDevice` 的设备号，避免每次分配都过一次 runtime。
thread_local int tls_runtime_device = -1;

/// 保证当前线程的 runtime 设备与 `device_index` 一致。
void bind_runtime_device(int device_index) {
  if (tls_runtime_device == device_index) {
    return;
  }
  check_rc("cudaSetDevice", api().cudaSetDevice(device_index));
  tls_runtime_device = device_index;
}

// 分配器：2 的幂分桶 + 2 MiB 板切分

/// 一块从驱动拿到的内存。板切分成等大的块，块给桶用。
struct Slab {
  void* base = nullptr;  ///< 驱动返回的基址。
  size_t bytes = 0;      ///< 板大小（页的整数倍）。
  size_t block = 0;      ///< 每块大小。
  int bucket = -1;       ///< 所属桶；`-1` 表示单块大板（不切分）。
  size_t live = 0;       ///< 已借出的块数；0 表示整块板可以还给驱动。
};

/// 每个用户指针的记账。
struct BlockInfo {
  size_t requested = 0;  ///< 申请者要的字节数（`allocated_bytes` 按它累加）。
  size_t block = 0;      ///< 实际块大小。
  int bucket = -1;       ///< `-1` = 大块直通。
  size_t slab = 0;       ///< 所属板下标（大块无意义）。
  int device = 0;        ///< 这块属于哪张卡——归还时必须回到对应设备的池。
};

/// 一台设备的账本 + 缓存池。
///
/// 为什么按设备分开：`cudaMalloc` 出来的内存在**分配它的那张卡**上，跨卡复用指针会让
/// 内核在错误的卡上访存，症状是"数据看起来对，偶尔结果错"——本仓库的沙箱里
/// `device_count = 2`（物理 6、7），这个坑是真实存在的，不是理论问题。
struct DevicePool {
  /// 构造：把每个桶的空闲表准备好，省得用花括号列表初始化踩坑。
  DevicePool() : free_blocks(static_cast<size_t>(kBucketCount)) {}

  /// 每桶的空闲块，(指针, 所属板下标)。
  std::vector<std::vector<std::pair<void*, size_t>>> free_blocks;
  std::unordered_map<size_t, std::vector<void*>> big_free;  ///< 大块缓存：大小 → 指针。
  std::vector<Slab> slabs;                                  ///< 所有板。
  size_t reserved = 0;                                      ///< 从驱动拿到的总字节（未还）。
  size_t allocated = 0;                                     ///< 已借出的总字节。
  size_t slab_alignment = 0;                                ///< 首次分配观测到的基址对齐。
};

/// 所有设备的池。`std::map` 而不是 `unordered_map`：要保证 **引用稳定**（拿到的
/// `DevicePool&` 在别人插新设备时不失效），`map` 的节点地址天然稳定。
std::map<int, DevicePool> g_device_pools;

/// 全局记账表：用户指针 → 记账。跨设备共用一张，因为 deleter 只收到指针，
/// 得靠它反查这块属于哪张卡。
std::unordered_map<void*, BlockInfo> g_blocks;

/// 保护 `g_device_pools` / `g_blocks` 的互斥量。
///
/// 为什么用一把全局锁而不是每设备一把：分配/释放都是微秒级，跨设备争用不存在
/// （一个线程同时只在一张卡上分配），一把锁换来「账本永远自洽」。
std::mutex g_pool_mutex;

/// 取某台设备的池（前置条件：`g_pool_mutex` 已持）。
DevicePool& pool_for_locked(int device_index) {
  return g_device_pools[device_index];
}

/// 释放回调：把块还回对应设备的桶或大块缓存。
///
/// 为什么用「查表」而不是把桶号塞进 ctx：`c10::DataPtr` 的 deleter 只收到 ctx，而 torch 在
/// raw 路径上会把数据指针当 ctx 传（`raw_allocate` 里断言 data == ctx）。为了两条路径都对，
/// 统一用「用户指针 → 记账表」查，代价是一次哈希。
void free_block(void* ctx) {
  if (ctx == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  auto it = g_blocks.find(ctx);
  if (it == g_blocks.end()) {
    // 没登记说明不是我们发的（from_blob 的指针就是这种）。静默返回是有意的：
    // 析构路径不能抛异常。
    return;
  }
  const BlockInfo info = it->second;
  g_blocks.erase(it);
  DevicePool& p = pool_for_locked(info.device);
  p.allocated -= info.requested;
  if (info.bucket >= 0) {
    if (info.slab < p.slabs.size() && p.slabs[info.slab].live > 0) {
      --p.slabs[info.slab].live;
    }
    p.free_blocks[static_cast<size_t>(info.bucket)].emplace_back(ctx, info.slab);
  } else {
    p.big_free[info.block].push_back(ctx);
  }
}

/// `from_blob` 用的空 deleter：外部指针的所有权不在我们手里。
void noop_deleter(void*) {}

/// 从驱动拿一块，并把记账登记好（前置条件：`g_pool_mutex` 已持、`p` 是目标设备的池）。
///
/// @return 可供使用的设备指针。
void* acquire_locked(DevicePool& p, size_t nbytes, int device_index) {
  const int bucket = bucket_for(nbytes);
  void* ptr = nullptr;
  size_t slab_index = 0;

  if (bucket < 0) {
    // 大块：按页的整数倍直通。
    const size_t bytes = ((nbytes + kPageBytes - 1) / kPageBytes) * kPageBytes;
    auto& cache = p.big_free[bytes];
    if (!cache.empty()) {
      ptr = cache.back();
      cache.pop_back();
    } else {
      bind_runtime_device(device_index);
      check_rc("cudaMalloc", api().cudaMalloc(&ptr, bytes));
      p.reserved += bytes;
    }
    BlockInfo info;
    info.requested = nbytes;
    info.block = bytes;
    info.bucket = -1;
    info.device = device_index;
    g_blocks[ptr] = info;
    p.allocated += nbytes;
    return ptr;
  }

  const size_t block = bucket_size(bucket);
  auto& free_list = p.free_blocks[static_cast<size_t>(bucket)];
  if (!free_list.empty()) {
    // 池里有现成的块：直接借出，并把板的 live 计数加一。
    const std::pair<void*, size_t> entry = free_list.back();
    free_list.pop_back();
    ptr = entry.first;
    slab_index = entry.second;
    if (slab_index < p.slabs.size()) {
      ++p.slabs[slab_index].live;
    }
  } else {
    // 桶空了：向驱动要一整块板（页），切成等大的块，第 0 块给本次申请。
    const size_t blocks_per_slab = std::max<size_t>(1, kPageBytes / block);
    const size_t bytes = blocks_per_slab * block;
    bind_runtime_device(device_index);
    check_rc("cudaMalloc", api().cudaMalloc(&ptr, bytes));
    p.reserved += bytes;
    // 记下基址的最低置位——这就是驱动给我们的对齐粒度，探针会把它打出来对账。
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    p.slab_alignment = address & (~address + 1u);

    Slab slab;
    slab.base = ptr;
    slab.bytes = bytes;
    slab.block = block;
    slab.bucket = bucket;
    slab.live = 1;  // 第 0 块马上借出去
    slab_index = p.slabs.size();
    p.slabs.push_back(slab);
    for (size_t i = 1; i < blocks_per_slab; ++i) {
      free_list.emplace_back(static_cast<char*>(ptr) + i * block, slab_index);
    }
  }
  BlockInfo info;
  info.requested = nbytes;
  info.block = block;
  info.bucket = bucket;
  info.slab = slab_index;
  info.device = device_index;
  g_blocks[ptr] = info;
  p.allocated += nbytes;
  return ptr;
}

/// 把一张板还给驱动（前置条件：`g_pool_mutex` 已持，且板上没有借出的块）。
void release_slab_locked(DevicePool& p, size_t slab_index) {
  Slab& slab = p.slabs[slab_index];
  if (slab.base == nullptr) {
    return;
  }
  // 板里残留的空闲块必须从空闲表摘掉，否则以后会拿到已还给驱动的指针。
  if (slab.bucket >= 0) {
    auto& free_list = p.free_blocks[static_cast<size_t>(slab.bucket)];
    const char* base = static_cast<const char*>(slab.base);
    free_list.erase(std::remove_if(free_list.begin(), free_list.end(),
                                   [&](const std::pair<void*, size_t>& candidate) {
                                     return candidate.second == slab_index;
                                   }),
                    free_list.end());
    (void)base;
  }
  check_rc("cudaFree", api().cudaFree(slab.base));
  p.reserved -= slab.bytes;
  slab.base = nullptr;
  slab.bytes = 0;
  slab.live = 0;
}

/// 本进程的 PrivateUse1 分配器。`SetAllocator` 只接指针、不接管所有权，
/// 所以这个实例必须是静态生命周期的。
///
/// 两个签名坑是 torch 2.8 独有的（2.1 上写不出来，实测编译报错才知道）：
///   1) `allocate` 在 2.8 **不是 const 成员**（写成 const 会报 marked 'override',
///      but does not override）；
///   2) 2.8 的 `Allocator` **新增了一个纯虚** `copy_data(dest, src, count) const`，
///      不实现就编成抽象类。
class CorexAllocator final : public c10::Allocator {
 public:
  /// 按当前设备分配 `nbytes`。
  ///
  /// 为什么设备号要从 thread-local 取：见 `tls_device_index` 的注释——这个接口没有
  /// device 参数，是 c10 的设计。
  c10::DataPtr allocate(size_t nbytes) override {
    const int device_index = static_cast<int>(tls_device_index);
    const c10::Device device(c10::DeviceType::PrivateUse1, device_index);
    if (nbytes == 0) {
      // 0 字节不求驱动：numel==0 的张量是合法的（aten 里到处都是空张量），
      // 而 `cudaMalloc(0)` 在多数实现里返回错误码。
      return c10::DataPtr(nullptr, nullptr, &noop_deleter, device);
    }
    void* ptr = nullptr;
    {
      std::lock_guard<std::mutex> lock(g_pool_mutex);
      DevicePool& p = pool_for_locked(device_index);
      ptr = acquire_locked(p, nbytes, device_index);
    }
    return c10::DataPtr(ptr, ptr, &free_block, device);
  }

  /// 分配器之间的数据搬运（2.8 新增的纯虚）。
  ///
  /// 不能退化成 host `memcpy`：本平台 `UNIFIED_ADDRESSING = 0`
  /// （05-DRIVER-CONSTRAINTS §1），设备指针在 host 上不可解引用。所以走
  /// `cudaMemcpy` 的 device-to-device（kind=3）。
  void copy_data(void* dest, const void* src, std::size_t count) const override {
    if (count == 0) {
      return;
    }
    check_rc("cudaMemcpy(D2D)", api().cudaMemcpy(dest, src, count, /*kind=*/3));
  }

  /// 不走 raw 接口：返回 nullptr 表示 raw allocate/deallocate 不可用。
  ///
  /// 为什么不做：raw 路径上 torch 会把数据指针当 ctx 传回来，而我们的块归属是靠
  /// 「指针 → 记账表」查的；torch 允许后端不要这个能力。
  c10::DeleterFnPtr raw_deleter() const override { return nullptr; }
};

/// 全局分配器实例（`SetAllocator` 要求静态生命周期）。
CorexAllocator g_allocator;

// DeviceGuardImplInterface：八个纯虚函数

/// CoreX 的设备守卫实现。
///
/// 2.8 相对 2.1 的签名变化（照 2.1 写编不过）：
///   * `getStream` / `exchangeStream` 在 2.8 **去掉了 noexcept**；
///   * 其余六个签名不变。
struct CorexGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  /// 在 torch 里这一类设备始终是 PrivateUse1（对外叫 corex 是改名的结果）。
  c10::DeviceType type() const override { return c10::DeviceType::PrivateUse1; }

  /// 切设备并返回旧设备。
  c10::Device exchangeDevice(c10::Device device) const override {
    const c10::Device old(c10::DeviceType::PrivateUse1, tls_device_index);
    tls_device_index = device.index();
    return old;
  }

  /// 当前设备。
  c10::Device getDevice() const override {
    return c10::Device(c10::DeviceType::PrivateUse1, tls_device_index);
  }

  /// 设置当前设备（真的绑到 runtime 上）。
  void setDevice(c10::Device device) const override {
    tls_device_index = device.index();
    bind_runtime_device(static_cast<int>(tls_device_index));
  }

  /// 不做检查地设置当前设备（noexcept：这条路径上不许抛）。
  void uncheckedSetDevice(c10::Device device) const noexcept override {
    // 只更新 thread-local，不下发 cudaSetDevice：noexcept 语义下不能因为 runtime 报错
    // 就终止进程，真正的绑定留到第一次分配时做。
    tls_device_index = device.index();
  }

  /// 当前流（句柄 0 = 默认流）。
  c10::Stream getStream(c10::Device device) const override {
    return c10::Stream(c10::Stream::UNSAFE, device, tls_stream_handle);
  }

  /// 换流并返回旧流。
  c10::Stream exchangeStream(c10::Stream stream) const override {
    const c10::Stream old = getStream(stream.device());
    tls_stream_handle = stream.id();
    return old;
  }

  /// 设备数。
  c10::DeviceIndex deviceCount() const noexcept override {
    int count = 0;
    const int rc = api().cudaGetDeviceCount(&count);
    if (rc != 0 || count <= 0) {
      return 0;
    }
    return static_cast<c10::DeviceIndex>(count);
  }

  /// 等一条流上的工作完成（`torch.accelerator.synchronize` / `Stream.synchronize` 会走到这里）。
  void synchronizeStream(const c10::Stream& stream) const override {
    if (stream.id() == 0) {
      check_rc("cudaDeviceSynchronize", api().cudaDeviceSynchronize());
      return;
    }
    check_rc("cudaStreamSynchronize",
             api().cudaStreamSynchronize(
                 reinterpret_cast<void*>(static_cast<intptr_t>(stream.id()))));
  }

  /// 等一台设备上的所有工作完成。
  void synchronizeDevice(const c10::DeviceIndex /*device_index*/) const override {
    check_rc("cudaDeviceSynchronize", api().cudaDeviceSynchronize());
  }
};

// 静态注册：加载本 .so 的那一刻把 guard impl 塞进 c10 的 device_guard_impl_registry。
// 缺这一步的报错是 `RuntimeError: PyTorch is not linked with support for corex devices`。
C10_REGISTER_GUARD_IMPL(PrivateUse1, CorexGuardImpl);

}  // namespace

// 公开接口（Python 侧按这些名字调，签名不要改）

/// 初始化：绑定设备、注册分配器；幂等。
///
/// 为什么注册放在这里而不是静态初始化里：`SetAllocator` 的注释明说「非线程安全、
/// 只在初始化时调用」，而静态初始化期的调用顺序不由我们控制。
///
/// @param device_index 目标设备号（沙箱 runner 已用 CUDA_VISIBLE_DEVICES 重编号成 0..N-1）。
void init(int64_t device_index) {
  std::lock_guard<std::mutex> lock(g_mutex);
  load_api_locked();
  if (g_initialized) {
    return;
  }
  int count = 0;
  check_rc("cudaGetDeviceCount", g_api.cudaGetDeviceCount(&count));
  TORCH_CHECK(count > 0, "corex_shim: cudaGetDeviceCount 返回 0，没有可用设备");
  TORCH_CHECK(device_index >= 0 && device_index < count, "corex_shim: device_index=",
              device_index, " 越界（可见设备数 ", count, "，注意 runner 已重编号）");
  check_rc("cudaSetDevice", g_api.cudaSetDevice(static_cast<int>(device_index)));
  tls_runtime_device = static_cast<int>(device_index);
  tls_device_index = static_cast<c10::DeviceIndex>(device_index);
  // priority=1：允许覆盖可能已存在的低优先级默认分配器（Allocator.h 注释：只有不低于
  // 既有优先级的才能覆盖成功）。
  c10::SetAllocator(c10::DeviceType::PrivateUse1, &g_allocator, /*priority=*/1);
  g_initialized = true;
}

/// `init()` 是否已经完成。
bool is_initialized() { return g_initialized; }

/// 可见设备数。
int64_t device_count() {
  int count = 0;
  const int rc = api().cudaGetDeviceCount(&count);
  if (rc != 0) {
    return 0;
  }
  return static_cast<int64_t>(count);
}

namespace {

/// 由 sizes/strides 算出存储需要多少字节。
///
/// 为什么要用「求和」而不是「取各维最大跨度」：存储量由**最大元素下标**决定，而最大下标
/// = Σ stride[i]*(size[i]-1)（storage_offset 恒为 0）。取各维最大值是错的：`[2,3]` 配
/// `(3,1)` 逐维算是 max(3*1+1, 1*2+1)=4 个元素，实际最后一个元素在 1*3+2*1=5 号下标，
/// 需要 6 个——少算就会写越界。这个错在 `probes/111` 的账本上暴露过
/// （16+20=36 与「两个 6 元素张量应占 48 B」对不上）。
size_t storage_bytes_for(c10::IntArrayRef sizes, c10::IntArrayRef strides, size_t itemsize) {
  size_t elements = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] <= 0) {
      return 0;  // 空张量
    }
    elements += static_cast<size_t>(strides[i]) * static_cast<size_t>(sizes[i] - 1);
  }
  return (elements + 1) * itemsize;
}

/// 按 c10 的 ABI 手搓一个 PrivateUse1 张量。
///
/// 为什么不用 `at::detail::empty_generic`：上游 CPU wheel **没有发这个头**
/// （`include/ATen/native/EmptyTensor.h` 不存在，虽然符号在 libtorch_cpu.so 里导出）。
/// 自己按 StorageImpl + TensorImpl 组装，少一个「函数声明得自己写」的坑。
///
/// @param storage_ptr 设备指针；`owned == false` 表示外部指针（from_blob，不接管）。
at::Tensor make_tensor_from_ptr(void* storage_ptr, size_t nbytes, bool owned,
                                c10::IntArrayRef sizes, c10::IntArrayRef strides,
                                c10::ScalarType dtype, int64_t device_index) {
  const c10::Device device(c10::DeviceType::PrivateUse1, device_index);
  const caffe2::TypeMeta type_meta = c10::scalarTypeToTypeMeta(dtype);
  c10::DataPtr data_ptr(storage_ptr, storage_ptr, owned ? &free_block : &noop_deleter, device);
  // resizable 只在「这块内存属于我们」时为真；from_blob 的指针不能 resize，
  // 否则 resize_ 会去调一个不属于我们的释放函数。
  c10::Storage storage(c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), static_cast<int64_t>(nbytes), std::move(data_ptr),
      owned ? static_cast<c10::Allocator*>(&g_allocator) : nullptr, owned));
  auto tensor = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), type_meta);
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
  return tensor;
}

}  // namespace

/// 建一个 corex 设备上的 strided 张量（对应 `aten::empty_strided`）。
///
/// @param size 各维长度。
/// @param stride 各维步长（**原样使用**，允许非连续；view/transpose 的正确性靠它）。
/// @param dtype `c10::ScalarType` 的整数值（python 侧 `int(torch.float32)` 得 6）。
/// @param device_index 设备号。
at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride, int64_t dtype,
                         int64_t device_index) {
  TORCH_CHECK(size.size() == stride.size(),
              "corex_shim: empty_strided 的 size 与 stride 维数不一致（", size.size(), " vs ",
              stride.size(), "）");
  const auto scalar_type = static_cast<c10::ScalarType>(dtype);
  const size_t itemsize = c10::scalarTypeToTypeMeta(scalar_type).itemsize();
  const size_t nbytes = storage_bytes_for(size, stride, itemsize);
  void* ptr = nullptr;
  // 无条件绑一次设备（内部只在设备变化时下发 cudaSetDevice）：
  // 从**缓存池**拿块时不会走 cudaMalloc，如果只在 cudaMalloc 前绑，当前线程的 runtime
  // 设备就可能还停在别的卡上；调用方紧接着用 cudaMemcpy 搬数据时，就会拿错误的 context
  // 去解释这个指针。`probes/112` §1.4 就是这么暴露出来的。
  bind_runtime_device(static_cast<int>(device_index));
  if (nbytes > 0) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    DevicePool& p = pool_for_locked(static_cast<int>(device_index));
    ptr = acquire_locked(p, nbytes, static_cast<int>(device_index));
  }
  return make_tensor_from_ptr(ptr, nbytes, /*owned=*/true, size, stride, scalar_type,
                              device_index);
}

/// 用外部设备指针建张量（**不接管所有权**；给「Triton 写好的缓冲包成 Tensor」用）。
at::Tensor from_blob(int64_t ptr, at::IntArrayRef size, at::IntArrayRef stride, int64_t dtype,
                     int64_t device_index) {
  TORCH_CHECK(size.size() == stride.size(), "corex_shim: from_blob 的 size 与 stride 维数不一致");
  const auto scalar_type = static_cast<c10::ScalarType>(dtype);
  const size_t itemsize = c10::scalarTypeToTypeMeta(scalar_type).itemsize();
  const size_t nbytes = storage_bytes_for(size, stride, itemsize);
  // 同 empty_strided：包装外部指针后调用方马上要用 cudaMemcpy，先把 runtime 绑到这台设备。
  bind_runtime_device(static_cast<int>(device_index));
  return make_tensor_from_ptr(reinterpret_cast<void*>(static_cast<intptr_t>(ptr)), nbytes,
                              /*owned=*/false, size, stride, scalar_type, device_index);
}

/// 已经向驱动要到的字节数（含缓存中的空闲块，所有设备求和）。
int64_t reserved_bytes() {
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  size_t total = 0;
  for (const auto& entry : g_device_pools) {
    total += entry.second.reserved;
  }
  return static_cast<int64_t>(total);
}

/// 当前已经借出去（张量持有中）的字节数，所有设备求和。
int64_t allocated_bytes() {
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  size_t total = 0;
  for (const auto& entry : g_device_pools) {
    total += entry.second.allocated;
  }
  return static_cast<int64_t>(total);
}

/// 把缓存里空闲的板与大块还给驱动（`torch.corex.empty_cache()` 用）。
///
/// 只还「没有任何借出块」的板——这是安全线：还有张量活着的板必须等析构。
void empty_cache() {
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  for (auto& device_entry : g_device_pools) {
    DevicePool& p = device_entry.second;
    for (size_t i = 0; i < p.slabs.size(); ++i) {
      if (p.slabs[i].base != nullptr && p.slabs[i].live == 0) {
        release_slab_locked(p, i);
      }
    }
    for (auto& entry : p.big_free) {
      for (void* ptr : entry.second) {
        check_rc("cudaFree", api().cudaFree(ptr));
        p.reserved -= entry.first;
      }
      entry.second.clear();
    }
  }
}

/// 首次分配后观测到的 slab 基址对齐（字节，所有设备取最大）；0 表示还没分配过。
int64_t slab_alignment() {
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  size_t observed = 0;
  for (const auto& entry : g_device_pools) {
    observed = std::max(observed, entry.second.slab_alignment);
  }
  return static_cast<int64_t>(observed);
}

}  // namespace corex_shim

// pybind 绑定代码必须写在全局作用域：PYBIND11_MODULE 展开出的是 `extern "C"` 的
// `PyInit_<name>`，放在 namespace 里会变成某个命名空间的成员，import 时找不到入口。
using corex_shim::allocated_bytes;
using corex_shim::device_count;
using corex_shim::empty_cache;
using corex_shim::empty_strided;
using corex_shim::from_blob;
using corex_shim::init;
using corex_shim::is_initialized;
using corex_shim::reserved_bytes;
using corex_shim::slab_alignment;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() =
      "corex_shim: CoreX(天垓100) 的 torch 2.8 PrivateUse1 接入层"
      "（分配器 + guard impl + factory）";
  m.def("init", &init, pybind11::arg("device_index") = 0,
        "初始化：绑定设备、注册分配器与 guard impl；幂等");
  m.def("is_initialized", &is_initialized, "init() 是否已完成");
  m.def("device_count", &device_count, "可见设备数");
  m.def("empty_strided", &empty_strided, pybind11::arg("size"), pybind11::arg("stride"),
        pybind11::arg("dtype"), pybind11::arg("device_index"),
        "在 corex 上建 strided 张量（strides 原样使用）");
  m.def("from_blob", &from_blob, pybind11::arg("ptr"), pybind11::arg("size"),
        pybind11::arg("stride"), pybind11::arg("dtype"), pybind11::arg("device_index"),
        "用外部设备指针建张量，不接管所有权");
  m.def("reserved_bytes", &reserved_bytes, "已向驱动要到的字节数（含缓存）");
  m.def("allocated_bytes", &allocated_bytes, "已借出的字节数");
  m.def("empty_cache", &empty_cache, "把空闲的板/大块还给驱动");
  m.def("slab_alignment", &slab_alignment, "首次分配观测到的 slab 基址对齐（字节）");
}
