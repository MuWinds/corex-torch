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
  /// 2-D 拷贝：一次调用搬 `height` 行、每行 `width` 字节，行间按 pitch 走。
  /// 为什么要它：同步 `cudaMemcpy` 在本平台实测每次 5–13 µs，**调用次数比字节数贵**；
  /// 拼接/带行间隙的拷贝用一条 memcpy2D 替掉 N 条 memcpy，是这批性能问题的主解法。
  int (*cudaMemcpy2D)(void*, size_t, const void*, size_t, size_t, size_t, int) = nullptr;
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
  g_api.cudaMemcpy2D =
      bind_symbol<int (*)(void*, size_t, const void*, size_t, size_t, size_t, int)>(handle,
                                                                                "cudaMemcpy2D");
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

/// `cudaMemcpy2D` 的 device-to-device：一次搬 `height` 行、每行 `width` 字节。
///
/// 为什么要它：同步 `cudaMemcpy` 在本平台实测每次 5–13 µs
/// （`probes/124`），所以**调用次数比字节数贵**。拼接、带行间隙的拷贝、以及
/// 「沿一维跳着取」的转置，都能用一条 memcpy2D 替掉 N 条 memcpy。
///
/// @param dest 目的首地址；@param dest_pitch 目的行间距（字节）
/// @param src 源首地址；@param src_pitch 源行间距（字节）
/// @param width 每行字节数；@param height 行数；@param device_index 目标卡
void memcpy2d(int64_t dest, int64_t dest_pitch, int64_t src, int64_t src_pitch, int64_t width,
              int64_t height, int64_t device_index) {
  bind_runtime_device(device_index);
  TORCH_CHECK(width >= 0 && height >= 0, "corex_shim: memcpy2d 的 width/height 不能为负");
  if (width == 0 || height == 0) {
    return;
  }
  check_rc("cudaMemcpy2D(D2D)",
           api().cudaMemcpy2D(reinterpret_cast<void*>(dest), static_cast<size_t>(dest_pitch),
                              reinterpret_cast<const void*>(src), static_cast<size_t>(src_pitch),
                              static_cast<size_t>(width), static_cast<size_t>(height),
                              /*kind=*/3));
}

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

/// 建一个**共享同一块存储**的 strided 视图（对应 `aten::as_strided`）。
///
/// 为什么必须落在 C++：视图要三样东西同时成立——共享 StorageImpl（写穿可见）、
/// 各自独立的 sizes/strides/storage_offset、以及**生命周期跟谁绑**（视图活着时底层存储不能被
/// 释放）。这三样都只存在于 `c10::TensorImpl` + `c10::Storage` 的 ABI 里。而
/// `aten::as_strided` 正是 Python 侧缺的那个算子：`probes/120` 的基线里 19 个算子都报
/// `Could not run 'aten::as_strided'`——view / t / transpose / permute / squeeze / unsqueeze /
/// expand / expand_as / contiguous / select / slice / narrow / embedding 全部由它解锁。
///
/// 实现方式与 ATen 的 `as_strided` 同构：拷一份 `Storage`（intrusive_ptr 复制 = 同一个
/// StorageImpl、引用计数 +1，所以视图天生持有存储所有权），再用
/// `set_sizes_and_strides(size, stride, storage_offset)` 改元数据。
///
/// @param self 源张量。
/// @param size 视图形状。
/// @param stride 视图 strides（原样使用，允许 0 = 广播）。
/// @param storage_offset 相对存储起点的**元素**偏移（不是字节）。
at::Tensor as_strided(const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
                      int64_t storage_offset) {
  TORCH_CHECK(size.size() == stride.size(),
              "corex_shim: as_strided 的 size 与 stride 维数不一致（", size.size(), " vs ",
              stride.size(), "）");
  TORCH_CHECK(self.device().is_privateuseone(), "corex_shim: as_strided 只接 corex 张量，收到 ",
              self.device());
  const c10::TensorImpl* src = self.unsafeGetTensorImpl();
  const c10::Storage& src_storage = src->storage();
  TORCH_CHECK(storage_offset >= 0, "corex_shim: as_strided 的 storage_offset 必须 >= 0，收到 ",
              storage_offset);

  // 越界检查（与 ATen 同口径）：视图能碰到的最大元素下标必须落在存储范围内。
  // 少了这一步，越界读会安静地读到别人的显存——本平台没有统一地址空间保护。
  if (size.size() > 0) {
    int64_t max_index = 0;
    for (size_t i = 0; i < size.size(); ++i) {
      TORCH_CHECK(size[i] >= 0, "corex_shim: as_strided 的 size 不能为负");
      TORCH_CHECK(stride[i] >= 0, "corex_shim: as_strided 暂不支持负 stride（", stride[i], "）");
      if (size[i] > 0) {
        max_index += stride[i] * (size[i] - 1);
      }
    }
    const int64_t itemsize = static_cast<int64_t>(src->itemsize());
    const int64_t available = static_cast<int64_t>(src_storage.nbytes()) / itemsize;
    TORCH_CHECK(storage_offset + max_index < available,
                "corex_shim: as_strided 越界：末元素下标 ", storage_offset + max_index,
                " 超出存储元素数 ", available, "（storage.nbytes=", src_storage.nbytes(),
                ", itemsize=", itemsize, "）");
  }

  // 拷一份 Storage：intrusive_ptr 复制 → 指向同一个 StorageImpl，视图因此持有存储所有权。
  c10::Storage storage(src_storage);
  auto out = at::detail::make_tensor<c10::TensorImpl>(std::move(storage), src->key_set(),
                                                     src->dtype());
  out.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride, storage_offset);
  return out;
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

// ---------------------------------------------------------------------------
// 快速启动路径：cuModuleLoadData + cuLaunchKernel，绕开 Python 的实参打包
// ---------------------------------------------------------------------------

namespace {

/// CUDA driver API（`libcuda.so.1`）的函数表。与 runtime 表分开：启动走 driver，
/// 分配/拷贝走 runtime，两套句柄语义不同，混在一张表里容易把 `CUstream` 当 `cudaStream_t` 用。
struct CudaApi {
  int (*cuModuleLoadData)(void** module, const void* image) = nullptr;
  int (*cuModuleGetFunction)(void** function, void* module, const char* name) = nullptr;
  int (*cuLaunchKernel)(void* function, unsigned int gx, unsigned int gy, unsigned int gz,
                        unsigned int bx, unsigned int by, unsigned int bz,
                        unsigned int shared_mem_bytes, void* stream, void** params,
                        void** extra) = nullptr;
  int (*cuGetErrorString)(int code, const char** str) = nullptr;
  // context 管理：driver API 只认 libcuda 的「当前 context」，而本进程的 primary context
  // 是 cudart 建的；别的线程/别的层切换过之后，这里可能拿到一个已销毁的 handle。
  int (*cuInit)(unsigned int flags) = nullptr;
  int (*cuDeviceGet)(int* device, int ordinal) = nullptr;
  int (*cuCtxGetCurrent)(void** ctx) = nullptr;
  int (*cuCtxSetCurrent)(void* ctx) = nullptr;
  int (*cuCtxGetDevice)(int* device) = nullptr;
  int (*cuDevicePrimaryCtxRetain)(void** ctx, int device) = nullptr;
  int (*cudaGetDeviceLite)(int* device) = nullptr;
};

CudaApi g_cuda;

bool g_cuda_loaded = false;

/// 一个已装载的内核：module + function + 实参类型串。
///
/// 为什么把 spec 存进句柄而不是每次传：启动是热路径，`std::string` 的比较/拷贝都要避免；
/// 装载时算一次，之后每次 launch 只按字符走一遍。
struct KernelEntry {
  void* module = nullptr;
  void* function = nullptr;
  std::string spec;   ///< 逐字符：`p`=设备指针(int64) `f`=float `i`=int32 `I`=int64
  std::string name;
  /// 动态 shared 字节数。**不能想当然给 0**：Triton 的 `tl.dot` 会为布局转换申请 shared，
  /// 少给会让内核读写到「它以为存在」的 shared，结果错但不报错（2026-11-07 非方阵 mm 就是这么错的）。
  int64_t shared = 0;
};

std::mutex g_kernel_mutex;
std::unordered_map<size_t, KernelEntry*> g_kernels;   ///< 内容哈希 → 内核
std::string g_last_error;

/// FNV-1a：只用来做「同一份 cubin + 同一个内核名 + 同一份 spec」的去重键。
///
/// 为什么不用 std::string 当键：cubin 是几十 KB 的二进制，做键要反复比较整串；
/// 内容哈希一次算完，碰撞概率对「一个进程里几十个内核」这个量级可以忽略。
size_t fnv1a(const void* data, size_t size, size_t seed) {
  const unsigned char* bytes = static_cast<const unsigned char*>(data);
  size_t hash = seed;
  for (size_t i = 0; i < size; ++i) {
    hash ^= static_cast<size_t>(bytes[i]);
    hash *= static_cast<size_t>(1099511628211ULL);
  }
  return hash;
}

/// 前置声明：`ensure_context_locked` 要用它，而它的定义在下面（带 libcuda 错误串）。
void check_cu(const char* what, int code);

/// 打开 libcuda 并绑定四个入口（前置条件：`g_mutex` 已持）。
void load_cuda_api_locked() {
  if (g_cuda_loaded) {
    return;
  }
  const char* env = std::getenv("COREX_HOME");
  const std::string home =
      (env != nullptr && env[0] != '\0') ? std::string(env) : std::string("/usr/local/corex");
  const std::vector<std::string> candidates = {
      home + "/lib64/libcuda.so.1", home + "/lib/libcuda.so.1",
      home + "/lib64/libcuda.so",   home + "/lib/libcuda.so",
      "libcuda.so.1",               "libcuda.so"};
  std::string tried;
  void* handle = nullptr;
  for (const std::string& path : candidates) {
    handle = ::dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle != nullptr) {
      tried += " [hit] " + path;
      break;
    }
    tried += " [miss] " + path;
  }
  TORCH_CHECK(handle != nullptr, "corex_shim: 打不开 CUDA driver API（libcuda.so.1）。试过：",
              tried);
  g_cuda.cuModuleLoadData = bind_symbol<int (*)(void**, const void*)>(handle, "cuModuleLoadData");
  g_cuda.cuModuleGetFunction =
      bind_symbol<int (*)(void**, void*, const char*)>(handle, "cuModuleGetFunction");
  g_cuda.cuLaunchKernel =
      bind_symbol<int (*)(void*, unsigned int, unsigned int, unsigned int, unsigned int,
                          unsigned int, unsigned int, unsigned int, void*, void**, void**)>(
          handle, "cuLaunchKernel");
  g_cuda.cuGetErrorString = bind_symbol<int (*)(int, const char**)>(handle, "cuGetErrorString");
  g_cuda.cuInit = bind_symbol<int (*)(unsigned int)>(handle, "cuInit");
  g_cuda.cuDeviceGet = bind_symbol<int (*)(int*, int)>(handle, "cuDeviceGet");
  g_cuda.cuCtxGetCurrent = bind_symbol<int (*)(void**)>(handle, "cuCtxGetCurrent");
  g_cuda.cuCtxSetCurrent = bind_symbol<int (*)(void*)>(handle, "cuCtxSetCurrent");
  g_cuda.cuCtxGetDevice = bind_symbol<int (*)(int*)>(handle, "cuCtxGetDevice");
  g_cuda.cuDevicePrimaryCtxRetain =
      bind_symbol<int (*)(void**, int)>(handle, "cuDevicePrimaryCtxRetain");
  // `cudaGetDevice` 在 runtime 表里没留（分配器不需要它），这里单独从同一个 libcudart 取；
  // `dlopen` 同一路径返回同一句柄，拿到的函数与 cudart 侧完全一致。
  for (const std::string& path : cudart_candidates()) {
    void* rt = ::dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (rt != nullptr) {
      g_cuda.cudaGetDeviceLite = bind_symbol<int (*)(int*)>(rt, "cudaGetDevice");
      break;
    }
  }
  g_cuda_loaded = true;
}

/// 保证当前线程上有一个**有效**的 libcuda current context。
///
/// 为什么需要：`cuModuleLoadData` / `cuLaunchKernel` 只认 libcuda 的 current context，
/// 而本进程的 primary context 是 cudart 建的。实测踩过：直接调 `cuModuleLoadData` 返回
/// `rc=709 context is destroyed`（探针 122 第一次跑就是）。
///
/// 顺序**逐字照抄** corex-triton 的 `_ensure_context`（`driver.py:86`）：
/// `cuInit` → `cuDeviceGet` → `cuDevicePrimaryCtxRetain` → `cuCtxSetCurrent`。
/// 为什么不用 `cuCtxGetCurrent` 先看一眼：厂商库上「拿得到的非空句柄」不一定可用，
/// 而 retain 出来的是 primary context，与 cudart / corex-triton 看到的是同一个，重复 set 无害。
void ensure_context_locked() {
  load_cuda_api_locked();
  check_cu("cuInit", g_cuda.cuInit(0));
  int device = 0;
  if (g_cuda.cudaGetDeviceLite != nullptr) {
    g_cuda.cudaGetDeviceLite(&device);
  }
  int cuda_device = 0;
  check_cu("cuDeviceGet", g_cuda.cuDeviceGet(&cuda_device, device));
  void* primary = nullptr;
  check_cu("cuDevicePrimaryCtxRetain", g_cuda.cuDevicePrimaryCtxRetain(&primary, cuda_device));
  check_cu("cuCtxSetCurrent", g_cuda.cuCtxSetCurrent(primary));
}

/// 对外：让当前线程的 context 就位（`launch`/`load_kernel` 内部也会做，幂等且很便宜）。
void ensure_context() {
  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_context_locked();
}

/// 本线程是否已经确认过 context 有效。
///
/// 为什么用 thread_local 缓存：一次完整检查要过 `g_mutex` + 两个 driver 调用，量级 1 µs，
/// 而热路径的目标是 10 µs 级——每次都查会把收益吃掉。缓存的失效条件是「launch 返回非 0」，
/// 那时清标志并重试一次（见 `launch`），既省又不会卡在失效状态上。
thread_local bool tls_context_ready = false;

/// 轻量版：只在还没确认过时才真的去查一次。
void ensure_context_cached() {
  if (tls_context_ready) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!tls_context_ready) {
    ensure_context_locked();
    tls_context_ready = true;
  }
}

/// 取 driver 表；未加载先加载。
const CudaApi& cuda_api() {
  std::lock_guard<std::mutex> lock(g_mutex);
  load_cuda_api_locked();
  return g_cuda;
}

/// 把 driver 返回码翻成带原文的异常，并记进 `g_last_error` 供 `last_error()` 取。
void check_cu(const char* what, int code) {
  if (code == 0) {
    return;
  }
  const char* msg = nullptr;
  if (g_cuda_loaded && g_cuda.cuGetErrorString != nullptr) {
    g_cuda.cuGetErrorString(code, &msg);
  }
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_last_error = std::string(what) + " 失败 rc=" + std::to_string(code) + " " +
                   (msg != nullptr ? msg : "");
  }
  TORCH_CHECK(false, "corex_shim: ", what, " 失败 rc=", code, " ", msg != nullptr ? msg : "");
}

}  // namespace

/// 装载一个 cubin 里的内核，返回句柄（按内容哈希缓存，重复调用零成本）。
///
/// 为什么要有这条：Triton 的 Python 启动路径每次调用要过 tuple 打包、`_pack_arg`、
/// 一层层 Python 函数（真机实测纯启动 39.6 µs）。这里把「装模块 + 找函数」在装载期做完，
/// 启动期只剩一次 `cuLaunchKernel`。
///
/// @param cubin cubin 的字节（Triton 的 `compiled.asm["cubin"]`）。
/// @param kernel_name cubin 里的内核名（Triton 的 `compiled.name`）。
/// @param spec 实参类型串：`p`=设备指针 `f`=float `i`=int32 `I`=int64，按内核形参顺序。
/// @return 句柄（>0），传给 `launch`。
int64_t load_kernel(const std::string& cubin, const std::string& kernel_name,
                    const std::string& spec, int64_t shared_bytes) {
  TORCH_CHECK(!cubin.empty(), "corex_shim: load_kernel 收到空 cubin");
  TORCH_CHECK(!kernel_name.empty(), "corex_shim: load_kernel 收到空内核名");
  size_t key = fnv1a(cubin.data(), cubin.size(), static_cast<size_t>(1469598103934665603ULL));
  key = fnv1a(kernel_name.data(), kernel_name.size(), key);
  key = fnv1a(spec.data(), spec.size(), key);
  key = fnv1a(&shared_bytes, sizeof(shared_bytes), key);

  std::lock_guard<std::mutex> lock(g_kernel_mutex);
  auto found = g_kernels.find(key);
  if (found != g_kernels.end()) {
    return reinterpret_cast<int64_t>(found->second);
  }
  const CudaApi& api_ref = cuda_api();
  // 装载前必须有一个有效的 current context（否则 rc=709 context is destroyed）。
  ensure_context();
  auto* entry = new KernelEntry();
  entry->spec = spec;
  entry->name = kernel_name;
  entry->shared = shared_bytes;
  check_cu("cuModuleLoadData", api_ref.cuModuleLoadData(&entry->module, cubin.data()));
  check_cu("cuModuleGetFunction",
           api_ref.cuModuleGetFunction(&entry->function, entry->module, kernel_name.c_str()));
  g_kernels[key] = entry;
  return reinterpret_cast<int64_t>(entry);
}

/// 启动一个已装载的内核。**热路径**：不做字符串解析、不做 dict 查找、不分配每参数内存。
///
/// 实参按位置从三个并行数组里取，取哪个数组由 `spec` 的字符决定（Python 侧在编译期算好
/// 一次、缓存）。这样 Python 侧每次调用只需要构造三四个 list，C++ 侧只往 `void* params[]`
/// 里填指针——不再有 Triton 的 `_pack_arg` 与多层 Python 调用。
///
/// @param handle `load_kernel` 的返回值。
/// @param grid 3 元组（不足的补 1）。
/// @param block 3 元组。
/// @param shared 动态共享内存字节数。
/// @param stream `CUstream` 句柄（0 = legacy 默认流，与本平台 launch 的语义一致）。
/// @param ptr_args `p` 类实参（设备指针，按调用顺序）。
/// @param f32_args `f` 类实参（double 进、float 出）。
/// @param i32_args `i` 类实参。
/// @param i64_args `I` 类实参。
void launch(int64_t handle, std::vector<int64_t> grid, std::vector<int64_t> block,
            int64_t shared, int64_t stream, std::vector<int64_t> ptr_args,
            std::vector<double> f32_args, std::vector<int64_t> i32_args,
            std::vector<int64_t> i64_args) {
  auto* entry = reinterpret_cast<KernelEntry*>(static_cast<intptr_t>(handle));
  TORCH_CHECK(entry != nullptr && entry->function != nullptr,
              "corex_shim: launch 收到无效句柄（先 load_kernel）");
  TORCH_CHECK(grid.size() == 3 && block.size() == 3, "corex_shim: grid/block 必须是 3 元组");

  const std::string& spec = entry->spec;
  // 三个数组的量必须正好用完：多传/少传都是 Python 侧的类型串算错了，宁可当场报错。
  size_t need_p = 0, need_f = 0, need_i = 0, need_I = 0;
  for (char c : spec) {
    switch (c) {
      case 'p': ++need_p; break;
      case 'f': ++need_f; break;
      case 'i': ++need_i; break;
      case 'I': ++need_I; break;
      default: TORCH_CHECK(false, "corex_shim: spec 里有未知类型字符 '", c, "'");
    }
  }
  TORCH_CHECK(need_p == ptr_args.size() && need_f == f32_args.size() &&
                  need_i == i32_args.size() && need_I == i64_args.size(),
              "corex_shim: 实参数与 spec 不符（spec=", spec, " 需要 p", need_p, " f", need_f,
              " i", need_i, " I", need_I, "；收到 p", ptr_args.size(), " f",
              f32_args.size(), " i", i32_args.size(), " I", i64_args.size(), "）");

  // 存储体一次分配好：params 里存的是「指向这些槽位的指针」，vector 扩容会让指针失效，
  // 所以按最终大小一次性 resize，之后只按下标写。
  std::vector<void*> params(spec.size(), nullptr);
  std::vector<int64_t> slots_p(need_p);
  std::vector<float> slots_f(need_f);
  std::vector<int32_t> slots_i(need_i);
  std::vector<int64_t> slots_I(need_I);
  size_t ip = 0, iff = 0, ii = 0, iI = 0;
  for (size_t index = 0; index < spec.size(); ++index) {
    switch (spec[index]) {
      case 'p':
        slots_p[ip] = ptr_args[ip];
        params[index] = &slots_p[ip];
        ++ip;
        break;
      case 'f':
        slots_f[iff] = static_cast<float>(f32_args[iff]);
        params[index] = &slots_f[iff];
        ++iff;
        break;
      case 'i':
        slots_i[ii] = static_cast<int32_t>(i32_args[ii]);
        params[index] = &slots_i[ii];
        ++ii;
        break;
      case 'I':
        slots_I[iI] = i64_args[iI];
        params[index] = &slots_I[iI];
        ++iI;
        break;
      default: break;
    }
  }
  // 热路径：context 用 thread_local 缓存，正常情况下这里是一次判断。
  ensure_context_cached();
  int rc = cuda_api().cuLaunchKernel(
      entry->function, static_cast<unsigned int>(grid[0]), static_cast<unsigned int>(grid[1]),
      static_cast<unsigned int>(grid[2]), static_cast<unsigned int>(block[0]),
      static_cast<unsigned int>(block[1]), static_cast<unsigned int>(block[2]),
      static_cast<unsigned int>(shared),
      reinterpret_cast<void*>(static_cast<intptr_t>(stream)), params.data(), nullptr);
  if (rc != 0) {
    // 失败一次就把 context 重新确认一遍再试一次：典型原因是别的层换过 context，
    // 直接报错会让上层以为内核/参数有问题，查错成本远高于一次重试。
    tls_context_ready = false;
    ensure_context_cached();
    rc = cuda_api().cuLaunchKernel(
        entry->function, static_cast<unsigned int>(grid[0]), static_cast<unsigned int>(grid[1]),
        static_cast<unsigned int>(grid[2]), static_cast<unsigned int>(block[0]),
        static_cast<unsigned int>(block[1]), static_cast<unsigned int>(block[2]),
        static_cast<unsigned int>(shared),
        reinterpret_cast<void*>(static_cast<intptr_t>(stream)), params.data(), nullptr);
  }
  check_cu("cuLaunchKernel", rc);
}

/// 上一次失败的原文（`launch`/`load_kernel` 失败后取一次即可）。
std::string last_error() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_last_error;
}

/// 已装载的内核数（探针用来确认缓存命中）。
int64_t kernel_count() {
  std::lock_guard<std::mutex> lock(g_kernel_mutex);
  return static_cast<int64_t>(g_kernels.size());
}

// ---------------------------------------------------------------------------
// aten::mm 的 C++ 实现：绕掉 Python 分发与 _empty_impl（probes/010 里 D 段那 59.7 µs）
// ---------------------------------------------------------------------------

namespace {

/// 一套登记给 `aten::mm` 用的内核。
///
/// 为什么按 dtype 分别登记：fp32/fp16/bf16 的最优分块不一定相同，而且内核本身是按
/// (dtype, BLOCK_M, BLOCK_N, BLOCK_K, warps) 编出来的不同 cubin——一张表只装一套会让
/// 第二种 dtype 静默用错内核。
struct MmKernel {
  int64_t handle = 0;     ///< `load_kernel` 的句柄。
  int64_t block_m = 0;    ///< 分块行数。
  int64_t block_n = 0;    ///< 分块列数。
  int64_t block_k = 0;    ///< 分块归约深度。
  int64_t num_warps = 1;  ///< 每块 warp 数（blockDim = warps*32）。
  /// 动态 shared 字节数；`-1` = 用装载时记下的那个值。
  /// 为什么允许在这里覆盖：Python 侧可能忘了给 `load_kernel` 传 metadata.shared，
  /// 而 shared 给 0 会让内核读写到「它以为存在」的 shared，**结果错但不报错**。
  int64_t shared = -1;
};

std::mutex g_mm_mutex;
std::unordered_map<int, MmKernel> g_mm_kernels;   ///< c10::ScalarType → 内核

}  // namespace

/// 登记一套 `aten::mm` 用的内核（Python 侧首次编译完成后调一次）。
///
/// @param dtype_code `c10::ScalarType` 的整数值（fp32=6 / fp16=5 / bf16=15）。
/// @param handle `load_kernel` 返回的句柄。
/// @param block_m/block_n/block_k 分块参数，必须与编译内核时用的 constexpr 一致。
/// @param num_warps 每块 warp 数；`blockDim = num_warps * 32`。
void set_mm_kernel(int64_t dtype_code, int64_t handle, int64_t block_m, int64_t block_n,
                   int64_t block_k, int64_t num_warps, int64_t shared) {
  TORCH_CHECK(handle != 0, "corex_shim: set_mm_kernel 收到 0 句柄（先 load_kernel）");
  TORCH_CHECK(block_m > 0 && block_n > 0 && block_k > 0 && num_warps > 0,
              "corex_shim: set_mm_kernel 的分块参数必须为正");
  std::lock_guard<std::mutex> lock(g_mm_mutex);
  MmKernel kernel;
  kernel.handle = handle;
  kernel.block_m = block_m;
  kernel.block_n = block_n;
  kernel.block_k = block_k;
  kernel.num_warps = num_warps;
  kernel.shared = shared;
  g_mm_kernels[static_cast<int>(dtype_code)] = kernel;
}

/// 已登记的 (dtype, block_m, block_n, block_k, num_warps, shared) 明细（诊断用）。
std::vector<std::vector<int64_t>> mm_kernel_info() {
  std::lock_guard<std::mutex> lock(g_mm_mutex);
  std::vector<std::vector<int64_t>> out;
  for (const auto& entry : g_mm_kernels) {
    const auto* kernel = reinterpret_cast<const KernelEntry*>(static_cast<intptr_t>(entry.second.handle));
    out.push_back({static_cast<int64_t>(entry.first), entry.second.block_m,
                  entry.second.block_n, entry.second.block_k, entry.second.num_warps,
                  kernel != nullptr ? kernel->shared : 0});
  }
  std::sort(out.begin(), out.end());
  return out;
}

/// 已登记的 dtype 列表（探针与错误信息用）。
std::vector<int64_t> mm_kernel_dtypes() {
  std::lock_guard<std::mutex> lock(g_mm_mutex);
  std::vector<int64_t> out;
  for (const auto& entry : g_mm_kernels) {
    out.push_back(static_cast<int64_t>(entry.first));
  }
  std::sort(out.begin(), out.end());
  return out;
}

/// `aten::mm` 的 PrivateUse1 实现：`c = a @ b`（a: M×K，b: K×N，行主序连续）。
///
/// **为什么不回落**：没登记内核时直接报错，不退回 Python 实现。理由（Lead 定的口径）：
/// 回落会让「走了快路径」和「没走」两种状态在性能数据里分不清——而这一层的全部意义就是
/// 省掉那 100 µs 的 host 开销。
at::Tensor mm_impl(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.device().is_privateuseone() && b.device().is_privateuseone(),
              "corex 的 mm 要求两个张量都在 corex 上，收到 ", a.device(), " / ", b.device());
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "corex 的 mm 只支持 2 维：收到 ", a.dim(), " / ",
              b.dim());
  TORCH_CHECK(a.scalar_type() == b.scalar_type(),
              "corex 的 mm 要求两边 dtype 相同：", a.scalar_type(), " vs ", b.scalar_type());
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
              "corex 的 mm 暂不支持非连续张量（先 .contiguous()）");

  const int64_t M = a.size(0);
  const int64_t K = a.size(1);
  const int64_t N = b.size(1);
  TORCH_CHECK(b.size(0) == K, "corex 的 mm 归约维不一致：a 的列数 ", K, "，b 的行数 ",
              b.size(0));

  MmKernel kernel;
  {
    std::lock_guard<std::mutex> lock(g_mm_mutex);
    const auto found = g_mm_kernels.find(static_cast<int>(a.scalar_type()));
    TORCH_CHECK(found != g_mm_kernels.end(),
                "corex 的 aten::mm 还没登记内核（dtype=", static_cast<int>(a.scalar_type()),
                "）。先在 Python 侧编译并登记：\\n"
                "  k = _ops_matmul._get_kernel(...)\\n"
                "  h = shim.load_kernel(k.asm['cubin'], k.name, 'ppppiii')\\n"
                "  shim.set_mm_kernel(dtype_code, h, block_m, block_n, block_k, num_warps)");
    kernel = found->second;
  }

  std::vector<int64_t> sizes{M, N};
  std::vector<int64_t> strides{N, 1};
  at::Tensor out = empty_strided(sizes, strides, static_cast<int64_t>(a.scalar_type()),
                                 a.device().index());
  if (M <= 0 || N <= 0 || K <= 0) {
    return out;                       // 空矩阵：分配即可，不启动内核（与 aten 语义一致）
  }
  const auto* entry = reinterpret_cast<const KernelEntry*>(static_cast<intptr_t>(kernel.handle));
  TORCH_CHECK(entry != nullptr, "corex 的 aten::mm 收到无效内核句柄");

  // 参数类型串必须与这份 C++ 实现的固定布局对齐。这是**踩过的坑**：
  // Triton 3.4 把 global scratch 指针追加在参数表**最后**（不是中间），早先把 scratch
  // 放在第 4 位会让内核拿到的 M == 0，于是网格里所有 block 的掩码全假、一个元素都不写，
  // 而 `cuLaunchKernel` 返回 0——表面完全正常。
  const std::string& spec = entry->spec;
  TORCH_CHECK(spec == "pppiii" || spec == "pppiiip",
              "corex 的 aten::mm 要求内核 spec 为 \"pppiii\"（内核不带 global scratch）"
              "或 \"pppiiip\"（scratch 指针在**最后**），实际收到 \"", spec, "\"");

  const int64_t grid_x = (M + kernel.block_m - 1) / kernel.block_m;
  const int64_t grid_y = (N + kernel.block_n - 1) / kernel.block_n;
  std::vector<int64_t> ptr_args{
      reinterpret_cast<int64_t>(a.data_ptr()),
      reinterpret_cast<int64_t>(b.data_ptr()),
      reinterpret_cast<int64_t>(out.data_ptr()),
  };
  if (spec == "pppiiip") {
    ptr_args.push_back(0);          // global scratch 指针：mm 不需要，位置在最后
  }
  // shared：登记时显式给的优先，否则用装载时记下的 metadata.shared。
  // 给 0 的后果同样是「结果错但不报错」（tl.dot 的布局转换要用 shared）。
  const int64_t shared = kernel.shared >= 0 ? kernel.shared : entry->shared;
  launch(kernel.handle, {grid_x, grid_y, 1}, {kernel.num_warps * 32, 1, 1}, shared, 0,
         ptr_args, {}, {M, N, K}, {});
  return out;
}

// ---------------------------------------------------------------------------
// 逐元素算子的 C++ 实现（照 aten::mm 的模式，接口由 Lead 定稿）
// ---------------------------------------------------------------------------

namespace {

/// 变体：Tensor 实参 or Scalar 实参。
enum class ElemVariant { Tensor = 0, Scalar = 1 };

/// 一套登记好的逐元素内核。`n_ptrs/n_i32/n_f32` 全部来自 Python 侧按内核签名**数出来**的
/// 实参表（`_ops_elementwise.fast_op()`），C++ 这边**不抄默认值**——抄的那份会漂移，
/// 而漂移的表现是「跑完了但全错」（长度和值互相错位）。
struct ElemKernel {
  int64_t handle = 0;
  int64_t num_warps = 1;
  int64_t block = 1024;
  int64_t n_ptrs = 0;
  int64_t n_i32 = 0;
  int64_t n_f32 = 0;
  bool alpha_effective = false;   ///< 只有 add/sub 的 tensor 变体会让 alpha 改变结果。
  int64_t shared = 0;             ///< 动态 shared 字节数（elementwise 一般是 0）。
};

using ElemKey = std::tuple<int, int, int>;   ///< (op_code, variant, dtype_code)

std::mutex g_elem_mutex;
std::map<ElemKey, ElemKernel> g_elem_kernels;

/// 按候选名列表装载内核：装一次 module，逐个试 `cuModuleGetFunction`，第一个成功的算数。
///
/// 为什么要「候选名」：`cuModuleGetFunction` 只认名字，而 `load_fast_op` 的入参只有
/// cubin 字节。实测这些 cubin 的 `.symtab` 里读不出可用的函数符号（`probes/126`：
/// ELF 解析拿不到名字），所以按优先级试：
///   1. 调用方显式给的名字（Python 侧 `meta["name"]` 就有，最可靠）；
///   2. Triton 两个变体固定的函数名（`_binary_kernel` / `_scalar_kernel`）；
///   3. 从 ELF 符号表猜出来的那个（有就用）。
int64_t load_kernel_any_name(const std::string& cubin, const std::vector<std::string>& names,
                             const std::string& spec, int64_t shared_bytes) {
  TORCH_CHECK(!cubin.empty(), "corex_shim: 收到空 cubin");
  TORCH_CHECK(!names.empty(), "corex_shim: 候选内核名列表为空");
  size_t key = fnv1a(cubin.data(), cubin.size(), static_cast<size_t>(1469598103934665603ULL));
  for (const std::string& name : names) {
    key = fnv1a(name.data(), name.size(), key);
  }
  key = fnv1a(spec.data(), spec.size(), key);
  key = fnv1a(&shared_bytes, sizeof(shared_bytes), key);

  std::lock_guard<std::mutex> lock(g_kernel_mutex);
  const auto found = g_kernels.find(key);
  if (found != g_kernels.end()) {
    return reinterpret_cast<int64_t>(found->second);
  }
  const CudaApi& api_ref = cuda_api();
  ensure_context();
  auto* entry = new KernelEntry();
  entry->spec = spec;
  entry->shared = shared_bytes;
  check_cu("cuModuleLoadData", api_ref.cuModuleLoadData(&entry->module, cubin.data()));
  int last_rc = 0;
  for (const std::string& name : names) {
    void* function = nullptr;
    last_rc = api_ref.cuModuleGetFunction(&function, entry->module, name.c_str());
    if (last_rc == 0 && function != nullptr) {
      entry->function = function;
      entry->name = name;
      break;
    }
  }
  if (entry->function == nullptr) {
    delete entry;
    TORCH_CHECK(false, "corex_shim: cubin 里找不到候选内核（试过 ", names.size(),
                " 个名字，最后一个 rc=", last_rc, "）");
  }
  g_kernels[key] = entry;
  return reinterpret_cast<int64_t>(entry);
}
/// 从 cubin（ELF64 可重定位对象）里找出内核符号名。
///
/// 为什么需要：`cuModuleGetFunction` 只认名字，而 `load_fast_op` 的入参只有 cubin 字节；
/// 与其让调用方多传一个「必须和 cubin 完全一致」的名字，不如从符号表读出来。
/// 做法：遍历 `.symtab` 里 `STT_FUNC` 且 size>0 的符号——Triton 生成的 cubin 里
/// 真正被 launch 的那个就是最大的那个函数符号。
std::string kernel_name_from_cubin(const std::string& cubin) {
  TORCH_CHECK(cubin.size() > 64, "corex_shim: cubin 太小，不是 ELF");
  const auto* raw = reinterpret_cast<const unsigned char*>(cubin.data());
  TORCH_CHECK(raw[0] == 0x7f && raw[1] == 'E' && raw[2] == 'L' && raw[3] == 'F',
              "corex_shim: cubin 不是 ELF 文件（魔数不符）");
  auto u16 = [raw](size_t at) {
    uint16_t value = 0;
    std::memcpy(&value, raw + at, sizeof(value));
    return value;
  };
  auto u32 = [raw](size_t at) {
    uint32_t value = 0;
    std::memcpy(&value, raw + at, sizeof(value));
    return value;
  };
  auto u64 = [raw](size_t at) {
    uint64_t value = 0;
    std::memcpy(&value, raw + at, sizeof(value));
    return value;
  };
  const size_t shoff = static_cast<size_t>(u64(0x28));
  const size_t shentsize = u16(0x3A);
  const size_t shnum = u16(0x3C);
  TORCH_CHECK(shoff != 0 && shentsize >= 64 && shnum != 0, "corex_shim: cubin 没有节头表");

  constexpr uint32_t kSymTab = 2;          // SHT_SYMTAB
  constexpr unsigned char kFunc = 2;       // STT_FUNC
  std::string best_name;
  size_t best_size = 0;
  for (size_t index = 0; index < shnum; ++index) {
    const size_t at = shoff + index * shentsize;
    if (at + 64 > cubin.size()) {
      break;
    }
    if (u32(at + 4) != kSymTab) {
      continue;
    }
    const size_t sym_off = static_cast<size_t>(u64(at + 24));
    const size_t sym_size = static_cast<size_t>(u64(at + 32));
    const size_t sym_entsize = static_cast<size_t>(u64(at + 56));
    const size_t link = u32(at + 40);
    if (sym_entsize != 24 || link >= shnum) {
      continue;
    }
    const size_t str_at = shoff + link * shentsize;
    const size_t str_off = static_cast<size_t>(u64(str_at + 24));
    const size_t str_size = static_cast<size_t>(u64(str_at + 32));
    for (size_t sym = 0; sym + 24 <= sym_size; sym += 24) {
      const size_t entry = sym_off + sym;
      if (entry + 24 > cubin.size()) {
        break;
      }
      if ((raw[entry + 4] & 0x0f) != kFunc) {
        continue;
      }
      const uint64_t name_at = u32(entry);
      const uint64_t func_size = u64(entry + 16);
      if (name_at == 0 || name_at >= str_size) {
        continue;
      }
      const size_t abs = str_off + static_cast<size_t>(name_at);
      if (abs >= cubin.size()) {
        continue;
      }
      const std::string name(cubin.c_str() + abs);   // 符号名以 NUL 结尾
      if (name.empty()) {
        continue;
      }
      if (func_size > best_size) {                   // 取最大的函数符号
        best_size = static_cast<size_t>(func_size);
        best_name = name;
      }
    }
  }
  TORCH_CHECK(!best_name.empty(), "corex_shim: 在 cubin 的 .symtab 里找不到函数符号");
  return best_name;
}

int op_code_from_name(const std::string& op) {
  if (op == "add") return 0;
  if (op == "sub") return 1;
  if (op == "mul") return 2;
  if (op == "div") return 3;
  TORCH_CHECK(false, "corex_shim: 未知的算子名 '", op, "'（只认 add/sub/mul/div）");
}

int dtype_code_from_name(const std::string& dtype) {
  if (dtype == "fp16") return 5;
  if (dtype == "fp32") return 6;
  if (dtype == "bf16") return 15;
  TORCH_CHECK(false, "corex_shim: 未知的 dtype 名 '", dtype, "'（只认 fp16/fp32/bf16）");
}

std::string op_display_name(int op_code, ElemVariant variant) {
  static const char* kOps[4] = {"add", "sub", "mul", "div"};
  std::string out(kOps[op_code]);
  out += (variant == ElemVariant::Tensor) ? ".out(tensor)" : ".out(scalar)";
  return out;
}

/// 取某个 (op, variant, dtype) 的内核；没登记就明确报错（**不回落**）。
ElemKernel elem_kernel_for(int op_code, ElemVariant variant, const at::Tensor& a,
                           const char* op_name) {
  std::lock_guard<std::mutex> lock(g_elem_mutex);
  const auto found = g_elem_kernels.find(ElemKey{op_code, static_cast<int>(variant),
                                                 static_cast<int>(a.scalar_type())});
  TORCH_CHECK(found != g_elem_kernels.end(),
              "corex 的 ", op_name, "（dtype=", static_cast<int>(a.scalar_type()),
              "，", (variant == ElemVariant::Tensor ? "tensor" : "scalar"),
              " 变体）还没登记 C++ 内核；先用 shim.load_fast_op 登记，或用 Python 实现");
  return found->second;
}

}  // namespace

/// 装载并登记一个逐元素内核（Python 侧在 `_register_fast_ops()` 里按 (op, variant, dtype) 调）。
///
/// 实参表由 Python 侧从内核签名数出来（`_ops_elementwise.fast_op()`）：
///   * tensor 变体：`a*, b*, out*, n:i32, alpha:f32`
///   * scalar 变体：`a*, out*, n:i32, scalar:f32`
/// 末尾再追加 Triton 3.4 的 global scratch 指针。**`alpha_effective` 与「有没有 alpha 槽位」
/// 是两件事**：mul/div 的 tensor 变体也有 alpha 槽位，只是值不影响结果——槽位必须照传。
///
/// @param op `add`/`sub`/`mul`/`div`。
/// @param variant `tensor`/`scalar`。
/// @param dtype `fp16`/`fp32`/`bf16`。
/// @param cubin Triton 的 `compiled.asm["cubin"]`。
/// @param num_warps 每块 warp 数（blockDim = warps*32）。
/// @param num_stages 流水级数（本平台恒为 1；这里只做记录）。
/// @param n_ptrs/n_i32/n_f32 实参表里各类实参的个数。
/// @param alpha_effective 只有 add/sub 的 tensor 变体为真。
/// @param block 内核的 BLOCK constexpr（网格 = cdiv(n, block)）。
/// @param shared 动态 shared 字节数（`metadata.shared`，elementwise 一般是 0）。
void load_fast_op(const std::string& op, const std::string& variant, const std::string& dtype,
                  const std::string& cubin, int64_t num_warps, int64_t num_stages, int64_t n_ptrs,
                  int64_t n_i32, int64_t n_f32, bool alpha_effective, int64_t block,
                  const std::string& name, int64_t shared) {
  (void)num_stages;                       // 只在需要复现编译配置时才有意义
  const int op_code = op_code_from_name(op);
  const int dtype_code = dtype_code_from_name(dtype);
  ElemVariant variant_kind;
  if (variant == "tensor") {
    variant_kind = ElemVariant::Tensor;
  } else if (variant == "scalar") {
    variant_kind = ElemVariant::Scalar;
  } else {
    TORCH_CHECK(false, "corex_shim: 未知的变体 '", variant, "'（只认 tensor/scalar）");
  }
  TORCH_CHECK(n_ptrs > 0 && n_i32 >= 0 && n_f32 >= 0, "corex_shim: 实参个数不能为负");
  TORCH_CHECK(block > 0 && num_warps > 0, "corex_shim: block/num_warps 必须为正");

  // 实参类型串：指针 → i32 → f32，最后是 global scratch 指针（Triton 3.4 固定追加）。
  std::string spec;
  spec.append(static_cast<size_t>(n_ptrs), 'p');
  spec.append(static_cast<size_t>(n_i32), 'i');
  spec.append(static_cast<size_t>(n_f32), 'f');
  spec.push_back('p');

  // 候选名顺序：调用方给的 → Triton 变体固定名 → ELF 猜出来的（拿不到就算了）。
  std::vector<std::string> names;
  if (!name.empty()) {
    names.push_back(name);
  }
  names.push_back(variant_kind == ElemVariant::Tensor ? "_binary_kernel" : "_scalar_kernel");
  names.push_back("_binary_kernel");
  names.push_back("_scalar_kernel");
  try {
    const std::string from_elf = kernel_name_from_cubin(cubin);
    if (!from_elf.empty()) {
      names.push_back(from_elf);
    }
  } catch (const std::exception&) {
    // 解析不出来不算错：候选名里已经有可用的了。
  }
  const int64_t handle = load_kernel_any_name(cubin, names, spec, shared);

  std::lock_guard<std::mutex> lock(g_elem_mutex);
  ElemKernel kernel;
  kernel.handle = handle;
  kernel.num_warps = num_warps;
  kernel.block = block;
  kernel.n_ptrs = n_ptrs;
  kernel.n_i32 = n_i32;
  kernel.n_f32 = n_f32;
  kernel.alpha_effective = alpha_effective;
  kernel.shared = shared;
  g_elem_kernels[ElemKey{op_code, static_cast<int>(variant_kind), dtype_code}] = kernel;
}

/// 已登记逐元素内核明细 `[[op, variant, dtype, block, warps, n_ptrs, n_i32, n_f32], ...]`。
std::vector<std::vector<int64_t>> elementwise_kernel_info() {
  std::lock_guard<std::mutex> lock(g_elem_mutex);
  std::vector<std::vector<int64_t>> out;
  for (const auto& item : g_elem_kernels) {
    out.push_back({std::get<0>(item.first), std::get<1>(item.first), std::get<2>(item.first),
                   item.second.block, item.second.num_warps, item.second.n_ptrs,
                   item.second.n_i32, item.second.n_f32,
                   item.second.alpha_effective ? 1 : 0});
  }
  return out;
}

/// C++ 已接管的逐元素 schema 名。Python 侧据此**跳过**同名 Python 实现——
/// 否则 Python 注册会覆盖 C++ 注册，静默退回 Python 实现（mm 上踩过一次）。
/// C++ 的逐元素快路径是否支持广义广播（True：`.out` 的广播实参会被物化，不报错）。
///
/// 为什么要有这个探针：Python 侧(namely `_device`)靠它决定「能不能把 `.out` 让给 C++」。
/// 能力没覆盖就贸然让位 = 拿正确性换速度（`x1 * 2.0` 曾经因此直接抛错）。
bool elementwise_supports_broadcast() { return true; }

std::vector<std::string> elementwise_op_names() {
  return {"add.out",      "sub.out",      "mul.out",      "div.out",
          "add_.Tensor",  "sub_.Tensor",  "mul_.Tensor",  "div_.Tensor",
          "add_.Scalar",  "sub_.Scalar",  "mul_.Scalar",  "div_.Scalar"};
}

namespace {

/// 把一块**连续**的源写回一个**可能非连续**的目的张量（逻辑布局一一对应）。
///
/// 为什么需要：`.out` 的 `out` 在真实代码里常常是视图——RoPE 就是
/// `out[..., 0::2] = x1 * c - x2 * s`，`out` 的 stride 是 `(T*H, H, 2)`。内核只写线性内存，
/// 所以先在连续临时张量里算，再按 stride 写回。
///
/// 三条路径（按成本从低到高）：目的其实连续 → 1 条 memcpy；目的外层「彼此紧挨」→ 1 条
/// memcpy2D；目的是**均匀步长视图**（`0::2` 这类）→ 1 条带 pitch 的 memcpy2D；都不是才逐元素。
void write_back_strided(const at::Tensor& src, at::Tensor& dst) {
  const int64_t ndim = dst.dim();
  const int64_t total = dst.numel();
  if (total == 0) {
    return;
  }
  const int64_t item = dst.element_size();
  bind_runtime_device(dst.device().index());
  const int64_t dst_base = reinterpret_cast<int64_t>(dst.data_ptr());
  const int64_t src_base = reinterpret_cast<int64_t>(src.const_data_ptr());
  std::vector<int64_t> sizes(dst.sizes().begin(), dst.sizes().end());
  std::vector<int64_t> strides(ndim);
  for (int64_t dim = 0; dim < ndim; ++dim) {
    strides[dim] = dst.stride(dim);
  }
  int64_t run = 1;
  int64_t run_dims = 0;
  for (int64_t dim = ndim - 1; dim >= 0; --dim) {
    if (strides[dim] == run) {
      run *= sizes[dim];
      ++run_dims;
    } else {
      break;
    }
  }
  if (run_dims == ndim) {
    check_rc("cudaMemcpy(写回连续)",
             api().cudaMemcpy(reinterpret_cast<void*>(dst_base),
                              reinterpret_cast<const void*>(src_base), total * item, /*kind=*/3));
    return;
  }
  std::vector<int64_t> outer_dims;
  for (int64_t dim = 0; dim < ndim - run_dims; ++dim) {
    outer_dims.push_back(dim);
  }
  bool packed = true;
  for (size_t k = 0; k + 1 < outer_dims.size(); ++k) {
    if (strides[outer_dims[k]] != sizes[outer_dims[k + 1]] * strides[outer_dims[k + 1]]) {
      packed = false;
      break;
    }
  }
  if (packed) {
    int64_t height = 1;
    for (int64_t dim : outer_dims) {
      height *= sizes[dim];
    }
    memcpy2d(dst_base, strides[outer_dims[0]] * item, src_base, run * item, run * item, height,
             dst.device().index());
    return;
  }
  // 均匀步长视图（`x[..., 0::2]`）：每个非 size-1 维的 stride 都等于 step × 尾积。
  bool uniform = true;
  int64_t step = 0;
  int64_t tail = 1;
  for (int64_t dim = ndim - 1; dim >= 0; --dim) {
    if (sizes[dim] > 1) {
      if (strides[dim] % tail != 0) {
        uniform = false;
        break;
      }
      const int64_t ratio = strides[dim] / tail;
      if (step == 0) {
        step = ratio;
      } else if (step != ratio) {
        uniform = false;
        break;
      }
    }
    tail *= sizes[dim];
  }
  if (uniform && step > 1) {
    memcpy2d(dst_base, step * item, src_base, item, item, total, dst.device().index());
    return;
  }
  // 兜底：逐外层下标搬 run 个元素（慢但正确）。
  const int64_t run_bytes = run * item;
  std::vector<int64_t> index(outer_dims.size(), 0);
  while (true) {
    int64_t dst_off = 0;
    int64_t src_off = 0;
    int64_t inner = 1;
    for (size_t k = 0; k < outer_dims.size(); ++k) {
      dst_off += index[k] * strides[outer_dims[k]];
      src_off += index[k] * inner;
      inner *= sizes[outer_dims[k]];
    }
    check_rc("cudaMemcpy(写回块)",
             api().cudaMemcpy(reinterpret_cast<void*>(dst_base + dst_off * item),
                              reinterpret_cast<const void*>(src_base + src_off * item), run_bytes,
                              /*kind=*/3));
    bool carried = true;
    for (int64_t k = static_cast<int64_t>(outer_dims.size()) - 1; k >= 0; --k) {
      if (++index[k] < sizes[outer_dims[k]]) {
        carried = false;
        break;
      }
      index[k] = 0;
    }
    if (carried) {
      break;
    }
  }
}
/// 按广播规则把 `src` 物化成一个与 `proto` 同形的连续临时张量（走池分配器）。
///
/// 为什么需要：`.out` 的实参可能是广播视图（`x1 * cos[:, None, :]`）。内核是线性下标写的、
/// 不认广播，所以要么物化、要么报错。Lead 定的口径：物化（正确性优先），但要把成本压住。
///
/// 实现要点：
///   * 只支持**一个广播维**的快路径（decode 里全部是这种形态：`[B,1,H]` 对 `[B,T,H]`），
///     每个广播下标一条 `memcpy2D`，把该下标下的整块（`above × below`）一次搬完；
///   * 多个广播维或源在上层不连续时走通用逐块循环（慢，但正确）；
///   * 临时张量用 `empty_strided` 分配 → 走自建的桶缓存，不每次 `cudaMalloc`。
at::Tensor materialize_broadcast(const at::Tensor& src, const at::Tensor& proto,
                                 const char* op_name) {
  const int64_t ndim = proto.dim();
  const int64_t src_ndim = src.dim();
  TORCH_CHECK(src_ndim <= ndim, "corex 的 ", op_name, " 的广播源维数比输出多（", src_ndim,
              " > ", ndim, "）");
  std::vector<int64_t> sizes(proto.sizes().begin(), proto.sizes().end());
  std::vector<int64_t> dst_strides(ndim, 1);
  int64_t acc = 1;
  for (int64_t dim = ndim - 1; dim >= 0; --dim) {
    dst_strides[dim] = acc;
    acc *= sizes[dim];
  }
  std::vector<int64_t> src_sizes(ndim, 1);
  std::vector<int64_t> src_strides(ndim, 0);
  for (int64_t dim = 0; dim < ndim; ++dim) {
    const int64_t src_dim = dim - (ndim - src_ndim);
    if (src_dim < 0) {
      continue;                                  // 源里没有这一维 → 广播
    }
    src_sizes[dim] = src.size(src_dim);
    src_strides[dim] = (src.size(src_dim) == 1) ? 0 : src.stride(src_dim);
    TORCH_CHECK(src.size(src_dim) == 1 || src.size(src_dim) == sizes[dim],
                "corex 的 ", op_name, " 的形状不可广播：", proto.sizes(), " vs ", src.sizes());
  }
  at::Tensor tmp = empty_strided(sizes, dst_strides, static_cast<int64_t>(proto.scalar_type()),
                                 proto.device().index());
  const int64_t item = proto.element_size();
  bind_runtime_device(proto.device().index());
  const int64_t total = tmp.numel();
  if (total == 0) {
    return tmp;
  }

  std::vector<int64_t> bcast_dims;
  for (int64_t dim = 0; dim < ndim; ++dim) {
    if (src_sizes[dim] == 1 && sizes[dim] > 1) {
      bcast_dims.push_back(dim);
    }
  }

  if (bcast_dims.empty()) {
    // 形状一致：把源按 stride 搬进连续临时张量。只有「末尾连续段 + 外层」两种情形，
    // 用 memcpy2D 一次搬完（dst 连续 → dpitch = 段字节数）。
    int64_t run = 1;
    int64_t run_dims = 0;
    for (int64_t dim = ndim - 1; dim >= 0; --dim) {
      if (src_sizes[dim] == sizes[dim] && src_strides[dim] == run) {
        run *= sizes[dim];
        ++run_dims;
      } else {
        break;
      }
    }
    int64_t outer = 1;
    for (int64_t dim = 0; dim < ndim - run_dims; ++dim) {
      outer *= sizes[dim];
    }
    const int64_t block_bytes = run * item;
    const int64_t dst_pitch = dst_strides[ndim - run_dims - 1] * item;
    const int64_t src_pitch = src_strides[ndim - run_dims - 1] * item;
    memcpy2d(reinterpret_cast<int64_t>(tmp.data_ptr()), outer > 1 ? dst_pitch : block_bytes,
             reinterpret_cast<int64_t>(src.const_data_ptr()), outer > 1 ? src_pitch : block_bytes,
             block_bytes, outer, proto.device().index());
    return tmp;
  }

  // 单广播维快路径：源可看成 [above, 1, below] 且 below/above 连续。
  if (bcast_dims.size() == 1) {
    const int64_t dim = bcast_dims[0];
    int64_t below_elems = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
      below_elems *= sizes[d];
    }
    int64_t above_elems = 1;
    for (int64_t d = 0; d < dim; ++d) {
      above_elems *= sizes[d];
    }

    const int64_t block_bytes = below_elems * item;
    const int64_t dst_pitch = dst_strides[dim] * item;      // 广播维相邻下标之间的间距
    // 源在「广播维之上」是否连续：stride[dim-1] 应等于 below_elems（`cos[:, None, :]` 就是）
    const bool upper_contiguous = (dim == 0) || (src_strides[dim - 1] == below_elems);
    if (upper_contiguous) {
      // 每条 memcpy2D 搬「above 行 × below 个元素」：行间按**外层维的真实间距**走，
      // 不是块大小——早先这里用了 block_bytes 当 dpitch，对 `[4,16,64] *= [4,1,64]`
      // 会把 4 行按 64 个元素的间距写，结果整块错位（probes/128 抓到）。
      const int64_t dst_row_pitch = (dim > 0) ? dst_strides[dim - 1] * item : block_bytes;
      const int64_t src_row_pitch = (dim > 0) ? src_strides[dim - 1] * item : block_bytes;
      for (int64_t index = 0; index < sizes[dim]; ++index) {
        memcpy2d(reinterpret_cast<int64_t>(tmp.data_ptr()) + index * dst_pitch, dst_row_pitch,
                 reinterpret_cast<int64_t>(src.const_data_ptr()), src_row_pitch, block_bytes,
                 above_elems, proto.device().index());
      }
      return tmp;
    }
  }

  // 多广播维且都在前导位置（典型：`[T,H] *= [H]` 的 bias）：源是一整块连续数据，在输出里
  // 逐块重复。用「先写首块、再倍增复制」把调用数从 repeats 降到 ceil(log2(repeats))。
  // 为什么值得单独做：逐块循环对 `[4,16,64] *= [64]` 是 64 条同步 memcpy ≈ 512 µs（实测），
  // 倍增之后是 7 条 ≈ 40 µs。
  {
    bool leading = !bcast_dims.empty() && bcast_dims.front() == 0;
    for (size_t k = 0; leading && k < bcast_dims.size(); ++k) {
      leading = (bcast_dims[k] == static_cast<int64_t>(k));
    }
    int64_t block_elems = 1;
    bool remainder_ok = leading;
    if (leading) {
      for (int64_t dim = static_cast<int64_t>(bcast_dims.size()); dim < ndim; ++dim) {
        block_elems *= sizes[dim];
        remainder_ok = remainder_ok && (src_sizes[dim] == sizes[dim]);
      }
      remainder_ok = remainder_ok && src.is_contiguous() && (src.numel() == block_elems);
    }
    if (remainder_ok) {
      int64_t repeats = 1;
      for (size_t k = 0; k < bcast_dims.size(); ++k) {
        repeats *= sizes[bcast_dims[k]];
      }
      const int64_t block_bytes = block_elems * item;
      check_rc("cudaMemcpy(广播首块)",
               api().cudaMemcpy(tmp.data_ptr(), src.const_data_ptr(), block_bytes, /*kind=*/3));
      int64_t filled = 1;
      while (filled < repeats) {
        const int64_t chunk = (filled < repeats - filled) ? filled : (repeats - filled);
        memcpy2d(reinterpret_cast<int64_t>(tmp.data_ptr()) + filled * block_bytes, block_bytes,
                 reinterpret_cast<int64_t>(tmp.data_ptr()), block_bytes, block_bytes, chunk,
                 proto.device().index());
        filled += chunk;
      }
      return tmp;
    }
  }
  // 通用兜底：逐外层下标搬 run 个元素（慢但正确）。
  int64_t run = 1;
  int64_t run_dims = 0;
  for (int64_t dim = ndim - 1; dim >= 0; --dim) {
    if (src_sizes[dim] == sizes[dim] && src_strides[dim] == run) {
      run *= sizes[dim];
      ++run_dims;
    } else {
      break;
    }
  }
  const int64_t block_bytes = run * item;
  std::vector<int64_t> outer_dims;
  for (int64_t dim = 0; dim < ndim - run_dims; ++dim) {
    outer_dims.push_back(dim);
  }
  std::vector<int64_t> index(outer_dims.size(), 0);
  while (true) {
    int64_t dst_off = 0;
    int64_t src_off = 0;
    for (size_t k = 0; k < outer_dims.size(); ++k) {
      dst_off += index[k] * dst_strides[outer_dims[k]];
      src_off += index[k] * src_strides[outer_dims[k]];
    }
    check_rc("cudaMemcpy(广播块)",
             api().cudaMemcpy(reinterpret_cast<void*>(reinterpret_cast<int64_t>(tmp.data_ptr()) +
                                                      dst_off * item),
                              src.const_data_ptr() + src_off * item, block_bytes, /*kind=*/3));
    bool carried = true;
    for (int64_t k = static_cast<int64_t>(outer_dims.size()) - 1; k >= 0; --k) {
      if (++index[k] < sizes[outer_dims[k]]) {
        carried = false;
        break;
      }
      index[k] = 0;
    }
    if (carried) {
      break;
    }
  }
  return tmp;
}
/// 把一个 0 维 / 单元素张量读成 host 标量（**一次** `cudaMemcpy` D2H，4 或 8 字节）。
///
/// 三条纪律（Lead 定的口径）：
///   1. 按 `t.scalar_type()` 解释字节，**不能按 self 的 dtype 硬读**——`add_(2.5)` 常见是
///      0 维 fp32 配 fp16 self，硬读会读到垃圾；读出来统一转 double，再交给吃 f32 的内核。
///   2. 只有 `dim()==0 && numel()==1` 才允许走这里，其它形状不匹配一律报错。
///   3. D2H 会打断流水，所以它只出现在这一个分支里，不给其它路径加同步。
double read_scalar_from_device(const at::Tensor& t) {
  TORCH_CHECK(t.is_contiguous(), "corex: 0 维张量不连续");
  const size_t nbytes = static_cast<size_t>(t.element_size());
  alignas(8) unsigned char buffer[16] = {0};
  TORCH_CHECK(nbytes <= sizeof(buffer), "corex: 0 维张量的元素大小异常（", nbytes, " 字节）");
  if (t.device().is_cpu()) {
    // **最常见的情形**：`view.add_(2.5)` 造出来的 0 维标量张量在 CPU 上，值本来就在 host
    // 内存里 —— 直接读，不做 D2H、不打断流水。这是 Lead 说的「Python float 标量不要付 D2H」。
    std::memcpy(buffer, t.const_data_ptr(), nbytes);
  } else if (t.device().is_privateuseone()) {
    // 设备上的 0 维张量：只有这一种情形要一次 4/8 字节 D2H，作为最后手段。
    bind_runtime_device(t.device().index());
    check_rc("cudaMemcpy(D2H 标量)",
             api().cudaMemcpy(buffer, t.const_data_ptr(), nbytes, /*kind=*/2));
  } else {
    TORCH_CHECK(false, "corex: 0 维张量必须在 CPU 或 corex 上（", t.device(), "）");
  }
  switch (t.scalar_type()) {
    case at::kHalf:
      return static_cast<float>(*reinterpret_cast<const c10::Half*>(buffer));
    case at::kBFloat16:
      return static_cast<float>(*reinterpret_cast<const c10::BFloat16*>(buffer));
    case at::kFloat: {
      float value = 0.0F;
      std::memcpy(&value, buffer, sizeof(value));
      return static_cast<double>(value);
    }
    case at::kDouble: {
      double value = 0.0;
      std::memcpy(&value, buffer, sizeof(value));
      return value;
    }
    case at::kInt: {
      int32_t value = 0;
      std::memcpy(&value, buffer, sizeof(value));
      return static_cast<double>(value);
    }
    case at::kLong: {
      int64_t value = 0;
      std::memcpy(&value, buffer, sizeof(value));
      return static_cast<double>(value);
    }
    default:
      TORCH_CHECK(false, "corex: 不支持把 dtype ", t.scalar_type(), " 的 0 维张量读成标量");
  }
}
/// 公共实现：同形 + 连续 + 同设备 + 同 dtype 才走快路径；否则**明确报错**。
at::Tensor& elementwise_run(int op_code, ElemVariant variant, const char* op_name,
                            at::Tensor& out, const at::Tensor& a, const at::Tensor& b,
                            double alpha, double scalar_value);

at::Tensor& elementwise_run(int op_code, ElemVariant variant, const char* op_name,                            at::Tensor& out, const at::Tensor& a, const at::Tensor& b,
                            double alpha, double scalar_value) {
  TORCH_CHECK(out.device().is_privateuseone() && a.device().is_privateuseone(),
              "corex 的 ", op_name, " 要求张量都在 corex 上");
  TORCH_CHECK(a.scalar_type() == out.scalar_type(),
              "corex 的 ", op_name, " 要求同 dtype：", a.scalar_type(), " vs ", out.scalar_type());
  // 非连续输入：物化一块连续临时张量再算。真实代码里这就是 RoPE 的
  // `x[..., 0::2]`——它是步长视图，内核按线性下标写，不物化就只能报错（正确性优先）。
  if (!a.is_contiguous()) {
    const at::Tensor dense = materialize_broadcast(a, a, op_name);
    return elementwise_run(op_code, variant, op_name, out, dense, b, alpha, scalar_value);
  }
  // out 可能是视图（RoPE 的 `out[..., 0::2]`）：先在连续临时张量里算，再按 stride 写回。
  if (!out.is_contiguous()) {
    std::vector<int64_t> sizes(out.sizes().begin(), out.sizes().end());
    std::vector<int64_t> strides(out.dim(), 1);
    int64_t acc = 1;
    for (int64_t dim = out.dim() - 1; dim >= 0; --dim) {
      strides[dim] = acc;
      acc *= sizes[dim];
    }
    at::Tensor temp = empty_strided(sizes, strides, static_cast<int64_t>(out.scalar_type()),
                                    out.device().index());
    elementwise_run(op_code, variant, op_name, temp, a, b, alpha, scalar_value);
    write_back_strided(temp, out);
    return out;
  }
  if (variant == ElemVariant::Tensor) {
    if (b.sizes() != a.sizes()) {
      if (b.dim() == 0 && b.numel() == 1) {
        // 形态 3：0 维张量。`view.add_(2.5)` 在 torch 里最终派发到 `aten::add_.Tensor`，
        // 而 `other` 是 0 维标量张量（实测在 CPU 上 → 直接读 host，零 D2H）。
        TORCH_CHECK(b.device().is_cpu() || b.device().is_privateuseone(), "corex 的 ", op_name,
                    " 的 0 维张量必须在 CPU 或 corex 上（", b.device(), "）");
        const double value = read_scalar_from_device(b);
        // add/sub 的语义是 `self ± alpha * other`，alpha 折进标量；mul/div 没有 alpha。
        const double folded = (op_code == 0 || op_code == 1) ? alpha * value : value;
        return elementwise_run(op_code, ElemVariant::Scalar, op_name, out, a, a, 1.0, folded);
      }
      // 形态 4：广义广播（`x1 * cos[:, None, :]`）——物化一块与输出同形的临时张量再走同形
      // 快路径。为什么物化而不是报错：报错会把「能算但慢」变成「不能算」，那是拿正确性换
      // 速度。临时张量走池分配器，不每次 cudaMalloc。
      const at::Tensor materialized = materialize_broadcast(b, a, op_name);
      return elementwise_run(op_code, ElemVariant::Tensor, op_name, out, a, materialized,
                             alpha, 0.0);
    }
    TORCH_CHECK(b.device().is_privateuseone() && b.scalar_type() == a.scalar_type(),
                "corex 的 ", op_name, " 的 C++ 快路径要求 other 在 corex、同 dtype");
    if (!b.is_contiguous()) {
      const at::Tensor dense = materialize_broadcast(b, a, op_name);
      return elementwise_run(op_code, variant, op_name, out, a, dense, alpha, scalar_value);
    }
  }
  TORCH_CHECK(a.sizes() == out.sizes(), "corex 的 ", op_name, " 要求 out 与 self 同形");

  const ElemKernel kernel = elem_kernel_for(op_code, variant, a, op_name);
  const int64_t n = a.numel();
  if (n == 0) {
    return out;
  }
  const auto* entry = reinterpret_cast<const KernelEntry*>(static_cast<intptr_t>(kernel.handle));
  TORCH_CHECK(entry != nullptr, "corex 的 ", op_name, " 收到无效内核句柄");
  TORCH_CHECK(static_cast<int64_t>(entry->spec.size()) ==
                  kernel.n_ptrs + kernel.n_i32 + kernel.n_f32 + 1 &&
                  entry->spec.back() == 'p',
              "corex 的 ", op_name, " 的内核 spec 与登记的实参个数不符（", entry->spec, "）");

  std::vector<int64_t> ptr_args;
  ptr_args.push_back(reinterpret_cast<int64_t>(a.data_ptr()));
  if (variant == ElemVariant::Tensor) {
    ptr_args.push_back(reinterpret_cast<int64_t>(b.data_ptr()));
  }
  ptr_args.push_back(reinterpret_cast<int64_t>(out.data_ptr()));
  ptr_args.push_back(0);                       // global scratch（Triton 3.4 固定尾参）
  std::vector<double> f32_args;
  if (variant == ElemVariant::Tensor) {
    // alpha 槽位**永远**要传（mul/div 只是值不影响结果）。
    f32_args.push_back(kernel.alpha_effective ? alpha : 1.0);
  } else {
    f32_args.push_back(scalar_value);
  }
  const int64_t grid = (n + kernel.block - 1) / kernel.block;
  launch(kernel.handle, {grid, 1, 1}, {kernel.num_warps * 32, 1, 1}, kernel.shared, 0, ptr_args,
         f32_args, {n}, {});
  return out;
}

}  // namespace

/// `aten::add.out`（tensor 变体）：`out = self + alpha * other`。
at::Tensor& add_out_impl(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha,
                         at::Tensor& out) {
  return elementwise_run(0, ElemVariant::Tensor, "add.out", out, self, other, alpha.to<double>(),
                         0.0);
}

/// `aten::sub.out`（tensor 变体）：`out = self - alpha * other`。
at::Tensor& sub_out_impl(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha,
                         at::Tensor& out) {
  return elementwise_run(1, ElemVariant::Tensor, "sub.out", out, self, other, alpha.to<double>(),
                         0.0);
}

/// `aten::mul.out`（tensor 变体）。
at::Tensor& mul_out_impl(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  return elementwise_run(2, ElemVariant::Tensor, "mul.out", out, self, other, 1.0, 0.0);
}

/// `aten::div.out`（tensor 变体）。
at::Tensor& div_out_impl(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  return elementwise_run(3, ElemVariant::Tensor, "div.out", out, self, other, 1.0, 0.0);
}

/// `aten::add_.Tensor`。
at::Tensor& add_inplace_impl(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  return elementwise_run(0, ElemVariant::Tensor, "add_", self, self, other, alpha.to<double>(),
                         0.0);
}

/// `aten::sub_.Tensor`。
at::Tensor& sub_inplace_impl(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  return elementwise_run(1, ElemVariant::Tensor, "sub_", self, self, other, alpha.to<double>(),
                         0.0);
}

/// `aten::mul_.Tensor`。
at::Tensor& mul_inplace_impl(at::Tensor& self, const at::Tensor& other) {
  return elementwise_run(2, ElemVariant::Tensor, "mul_", self, self, other, 1.0, 0.0);
}

/// `aten::div_.Tensor`。
at::Tensor& div_inplace_impl(at::Tensor& self, const at::Tensor& other) {
  return elementwise_run(3, ElemVariant::Tensor, "div_", self, self, other, 1.0, 0.0);
}

/// `aten::add_.Scalar`：就地 `self += alpha * other`（other 是标量）。
at::Tensor& add_scalar_inplace_impl(at::Tensor& self, const at::Scalar& other,
                                    const at::Scalar& alpha) {
  return elementwise_run(0, ElemVariant::Scalar, "add_.Scalar", self, self, self,
                         alpha.to<double>(), alpha.to<double>() * other.to<double>());
}

/// `aten::sub_.Scalar`。
at::Tensor& sub_scalar_inplace_impl(at::Tensor& self, const at::Scalar& other,
                                    const at::Scalar& alpha) {
  return elementwise_run(1, ElemVariant::Scalar, "sub_.Scalar", self, self, self,
                         alpha.to<double>(), alpha.to<double>() * other.to<double>());
}

/// `aten::mul_.Scalar`。
at::Tensor& mul_scalar_inplace_impl(at::Tensor& self, const at::Scalar& other) {
  return elementwise_run(2, ElemVariant::Scalar, "mul_.Scalar", self, self, self, 1.0,
                         other.to<double>());
}

/// `aten::div_.Scalar`。
at::Tensor& div_scalar_inplace_impl(at::Tensor& self, const at::Scalar& other) {
  return elementwise_run(3, ElemVariant::Scalar, "div_.Scalar", self, self, self, 1.0,
                         other.to<double>());
}

}  // namespace corex_shim

// aten 的分发键注册必须在全局作用域做（`TORCH_LIBRARY_IMPL` 展开出的是一个静态注册对象）。
// 注意：**一旦加载本 .so，这些算子就由 C++ 接手**；没登记内核时它们明确报错、不回落
// Python 实现（刻意设计，见 elementwise_run 的注释）。out 变体的 kernel 签名里 **out 在最后**
// （torch 2.8 的约定，不是 ATen native 函数的 out-first 形状——写反了会在 .so 加载期
// 抛 "Mismatch in kernel C++ signatures" 直接 abort）。
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("mm", TORCH_FN(corex_shim::mm_impl));
  m.impl("add.out", TORCH_FN(corex_shim::add_out_impl));
  m.impl("sub.out", TORCH_FN(corex_shim::sub_out_impl));
  m.impl("mul.out", TORCH_FN(corex_shim::mul_out_impl));
  m.impl("div.out", TORCH_FN(corex_shim::div_out_impl));
  m.impl("add_.Tensor", TORCH_FN(corex_shim::add_inplace_impl));
  m.impl("sub_.Tensor", TORCH_FN(corex_shim::sub_inplace_impl));
  m.impl("mul_.Tensor", TORCH_FN(corex_shim::mul_inplace_impl));
  m.impl("div_.Tensor", TORCH_FN(corex_shim::div_inplace_impl));
  m.impl("add_.Scalar", TORCH_FN(corex_shim::add_scalar_inplace_impl));
  m.impl("sub_.Scalar", TORCH_FN(corex_shim::sub_scalar_inplace_impl));
  m.impl("mul_.Scalar", TORCH_FN(corex_shim::mul_scalar_inplace_impl));
  m.impl("div_.Scalar", TORCH_FN(corex_shim::div_scalar_inplace_impl));
}
// pybind 绑定代码必须写在全局作用域：PYBIND11_MODULE 展开出的是 `extern "C"` 的
// `PyInit_<name>`，放在 namespace 里会变成某个命名空间的成员，import 时找不到入口。
using corex_shim::allocated_bytes;
using corex_shim::device_count;
using corex_shim::empty_cache;
using corex_shim::empty_strided;
using corex_shim::from_blob;
using corex_shim::init;
using corex_shim::is_initialized;
using corex_shim::as_strided;
using corex_shim::kernel_count;
using corex_shim::last_error;
using corex_shim::launch;
using corex_shim::mm_kernel_dtypes;
using corex_shim::elementwise_kernel_info;
using corex_shim::load_fast_op;
using corex_shim::elementwise_op_names;
using corex_shim::elementwise_supports_broadcast;
using corex_shim::memcpy2d;

using corex_shim::mm_kernel_info;
using corex_shim::load_kernel;
using corex_shim::reserved_bytes;
using corex_shim::set_mm_kernel;
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
  m.def("as_strided", &as_strided, pybind11::arg("self"), pybind11::arg("size"),
        pybind11::arg("stride"), pybind11::arg("storage_offset") = 0,
        "建共享同一存储的 strided 视图（aten::as_strided 的底层）");
  m.def("reserved_bytes", &reserved_bytes, "已向驱动要到的字节数（含缓存）");
  m.def("allocated_bytes", &allocated_bytes, "已借出的字节数");
  m.def("empty_cache", &empty_cache, "把空闲的板/大块还给驱动");
  m.def("slab_alignment", &slab_alignment, "首次分配观测到的 slab 基址对齐（字节）");
  m.def("load_kernel", &load_kernel, pybind11::arg("cubin"), pybind11::arg("kernel_name"),
        pybind11::arg("spec"), pybind11::arg("shared_bytes") = 0,
        "装载 cubin 里的内核并返回句柄（内容哈希缓存）；spec 见 launch；"
        "shared_bytes 传 metadata.shared，不能省（tl.dot 要 shared）");
  m.def("launch", &launch, pybind11::arg("handle"), pybind11::arg("grid"),
        pybind11::arg("block"), pybind11::arg("shared"), pybind11::arg("stream"),
        pybind11::arg("ptr_args"), pybind11::arg("f32_args"), pybind11::arg("i32_args"),
        pybind11::arg("i64_args"),
        "热路径启动：按 spec 展开实参，直接 cuLaunchKernel");
  m.def("last_error", &last_error, "上一次 load_kernel/launch 失败的原文");
  m.def("kernel_count", &kernel_count, "已装载（缓存）的内核数");
  m.def("set_mm_kernel", &set_mm_kernel, pybind11::arg("dtype_code"), pybind11::arg("handle"),
        pybind11::arg("block_m"), pybind11::arg("block_n"), pybind11::arg("block_k"),
        pybind11::arg("num_warps") = 1, pybind11::arg("shared") = -1,
        "登记 aten::mm 用的内核（按 dtype）；shared 传 metadata.shared（-1 = 用 load_kernel 记的）");
  m.def("mm_kernel_dtypes", &mm_kernel_dtypes, "已登记 mm 内核的 dtype 列表");
  m.def("load_fast_op", &load_fast_op, pybind11::arg("op"), pybind11::arg("variant"),
        pybind11::arg("dtype"), pybind11::arg("cubin"), pybind11::arg("num_warps"),
        pybind11::arg("num_stages"), pybind11::arg("n_ptrs"), pybind11::arg("n_i32"),
        pybind11::arg("n_f32"), pybind11::arg("alpha_effective"), pybind11::arg("block"),
        pybind11::arg("name") = "", pybind11::arg("shared") = 0,
        "装载并登记一个逐元素内核（实参表由 Python 按签名数出来）");
  m.def("elementwise_kernel_info", &elementwise_kernel_info,
        "已登记逐元素内核 [[op, dtype, block, warps, has_alpha], ...]");
  m.def("elementwise_supports_broadcast", &elementwise_supports_broadcast,
        "C++ 的逐元素快路径是否支持广义广播（会把广播实参物化）");
  m.def("elementwise_op_names", &elementwise_op_names,
        "C++ 已接管的逐元素算子名（Python 侧据此跳过自己的实现）");
  m.def("memcpy2d", &memcpy2d, pybind11::arg("dest"), pybind11::arg("dest_pitch"),
        pybind11::arg("src"), pybind11::arg("src_pitch"), pybind11::arg("width"),
        pybind11::arg("height"), pybind11::arg("device_index"),
        "2-D D2D 拷贝：一次搬 height 行、每行 width 字节（行间按 pitch）");
  m.def("mm_kernel_info", &mm_kernel_info,
        "已登记 mm 内核明细 [[dtype, bm, bn, bk, warps, shared], ...]");
}
