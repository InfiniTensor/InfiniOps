#ifndef INFINI_OPS_TRITON_JIT_H_
#define INFINI_OPS_TRITON_JIT_H_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <type_traits>
#include <vector>

#include "cache.h"
#include "data_type.h"
#include "runtime.h"
#include "tensor.h"

namespace infini::ops {

// ---- device support ----

template <Device::Type kDev>
inline constexpr bool kJitSupported = false;

template <>
inline constexpr bool kJitSupported<Device::Type::kNvidia> = true;

template <typename Op, Device::Type kDev, bool = kJitSupported<kDev>>
struct JitOperatorBase : Op {
  using Op::Op;
};

template <typename Op, Device::Type kDev>
struct JitOperatorBase<Op, kDev, false> {};

// ---- compilation & launch ----

template <Device::Type kDev>
TargetInfo CurrentTarget() {
  TargetInfo target;
  target.type = Device::StringFromType(kDev);
  int dev_id = 0;
  if (Runtime<kDev>::GetDevice(&dev_id) != Runtime<kDev>::kSuccess)
    return target;
  target.id = dev_id;
  int major = 0, minor = 0;
  Runtime<kDev>::DeviceGetAttribute(
      &major, Runtime<kDev>::kDevAttrComputeCapabilityMajor, dev_id);
  Runtime<kDev>::DeviceGetAttribute(
      &minor, Runtime<kDev>::kDevAttrComputeCapabilityMinor, dev_id);
  target.arch = major * 10 + minor;
  Runtime<kDev>::DeviceGetAttribute(&target.warp_size,
                                    Runtime<kDev>::kDevAttrWarpSize, dev_id);
  return target;
}

template <Device::Type kDev>
bool Load(const TargetInfo& target, const char* binary_data, size_t binary_size,
          const KernelMeta& meta, typename Driver<kDev>::Function& func,
          typename Driver<kDev>::Module& mod) {
  (void)binary_size;

  if (Driver<kDev>::ModuleLoadData(&mod, binary_data) != Driver<kDev>::kSuccess)
    return false;

  if (Driver<kDev>::ModuleGetFunction(&func, mod, meta.name.c_str()) !=
      Driver<kDev>::kSuccess) {
    Driver<kDev>::ModuleUnload(mod);
    return false;
  }

  if (meta.shared > 49152) {
    int optin = 0;
    Runtime<kDev>::DeviceGetAttribute(
        &optin, Runtime<kDev>::kDevAttrMaxSharedMemoryPerBlockOptin, target.id);
    int st = 0;
    Driver<kDev>::FuncGetAttribute(
        &st, Driver<kDev>::kFuncAttributeSharedSizeBytes, func);
    if (optin < st || meta.shared > static_cast<unsigned>(optin - st)) {
      Driver<kDev>::ModuleUnload(mod);
      return false;
    }
    Driver<kDev>::FuncSetCacheConfig(func,
                                     Driver<kDev>::kFuncCachePreferShared);
    if (Driver<kDev>::FuncSetAttribute(
            func, Driver<kDev>::kFuncAttributeMaxDynamicSharedSizeBytes,
            optin - st) != Driver<kDev>::kSuccess) {
      Driver<kDev>::ModuleUnload(mod);
      return false;
    }
  }

  return true;
}

template <Device::Type kDev>
typename Driver<kDev>::Function GetKernel(const char* op_name,
                                          const char* signature_str,
                                          void* stream, const JitConfig& opts,
                                          unsigned* out_shared) {
  TargetInfo target = CurrentTarget<kDev>();

  auto r = CacheQuery<kDev>(op_name, signature_str, opts.num_warps,
                            opts.num_stages, target.arch, target.id);
  if (r.mem_hit) {
    *out_shared = r.shared;
    return r.func;
  }

  KernelMeta meta;
  std::string binary_data;
  if (!ReadArtifacts(r.out_prefix, &meta, &binary_data)) {
    int ret = CompileKernel(target, op_name, r.out_prefix.c_str(),
                            opts.num_warps, opts.num_stages, signature_str);
    if (ret != 0) return nullptr;
    if (!ReadArtifacts(r.out_prefix, &meta, &binary_data)) return nullptr;
  }

  if (meta.global_scratch_size > 0 || meta.profile_scratch_size > 0) {
    fprintf(stderr, "triton jit: scratch not supported yet\n");
    return nullptr;
  }

  typename Driver<kDev>::Function func;
  typename Driver<kDev>::Module mod;
  if (!Load<kDev>(target, binary_data.data(), binary_data.size(), meta, func,
                  mod))
    return nullptr;

  unsigned shared = meta.shared;
  KernelCacheEntry<kDev> mine{func, shared};
  KernelCacheEntry<kDev> winner;
  if (KernelCacheLookup<kDev>(r.mem_key, &winner)) {
    Driver<kDev>::ModuleUnload(mod);
    func = winner.func;
    shared = winner.shared;
  } else {
    KernelCacheInsert<kDev>(r.mem_key, mine);
  }

  *out_shared = shared;
  return func;
}

template <Device::Type kDev>
int LaunchKernel(const char* op_name, const char* signature_str, void* stream,
                 Grid grid, const JitConfig& config, void** args) {
  unsigned shared = 0;
  auto func = GetKernel<kDev>(op_name, signature_str, stream, config, &shared);
  if (!func) return -1;

  TargetInfo target = CurrentTarget<kDev>();
  return Driver<kDev>::LaunchKernel(
      func, grid.x, grid.y, grid.z, config.num_warps * target.warp_size, 1, 1,
      shared, static_cast<typename Driver<kDev>::Stream>(stream), args,
      nullptr);
}

// ---- specialization helpers ----

inline const char* SpecPtr(uintptr_t v) { return v % 16 == 0 ? ":16" : ""; }

template <typename T>
const char* SpecInt(T v) {
  if (v == 1) return ":1";
  if ((v & 15) == 0) return ":16";
  return "";
}

// ---- `DataType` → Triton string ----

inline const char* DataTypeToTritonType(DataType dt) {
  switch (dt) {
    case DataType::kFloat16:
      return "fp16";
    case DataType::kBFloat16:
      return "bf16";
    case DataType::kFloat32:
      return "fp32";
    case DataType::kFloat64:
      return "fp64";
    case DataType::kInt8:
      return "i8";
    case DataType::kInt16:
      return "i16";
    case DataType::kInt32:
      return "i32";
    case DataType::kInt64:
      return "i64";
    case DataType::kUInt8:
      return "u8";
    case DataType::kUInt16:
      return "u16";
    case DataType::kUInt32:
      return "u32";
    case DataType::kUInt64:
      return "u64";
  }
  return "fp32";
}

// ---- C++ scalar type → Triton string ----

template <typename T>
const char* ScalarTypeToTritonType() {
  if constexpr (std::is_same_v<T, float>)
    return "fp64";
  else if constexpr (std::is_same_v<T, double>)
    return "fp64";
  else if constexpr (std::is_same_v<T, bool>)
    return "i32";
  else if constexpr (std::is_integral_v<T>) {
    if constexpr (sizeof(T) == 1) return std::is_signed_v<T> ? "i8" : "u8";
    if constexpr (sizeof(T) == 2) return std::is_signed_v<T> ? "i16" : "u16";
    if constexpr (sizeof(T) == 4) return std::is_signed_v<T> ? "i32" : "u32";
    if constexpr (sizeof(T) == 8) return std::is_signed_v<T> ? "i64" : "u64";
  }
  return "i32";
}

// ---- arguments parser ----

struct ArgPack {
  std::vector<void*> ptrs;

  std::deque<uint64_t> storage;

  std::string sig;

  template <typename T>
  void* Store(T v) {
    static_assert(sizeof(T) <= sizeof(uint64_t),
                  "scalar arg wider than `uint64_t`");
    uint64_t slot = 0;
    std::memcpy(&slot, &v, sizeof(T));
    storage.push_back(slot);
    return &storage.back();
  }
};

inline void PushArg(const Tensor& t, ArgPack& pack) {
  auto ptr = reinterpret_cast<uintptr_t>(t.data());
  pack.sig +=
      std::string("*") + DataTypeToTritonType(t.dtype()) + SpecPtr(ptr) + ",";
  pack.ptrs.push_back(pack.Store(ptr));
}

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void PushArg(T v, ArgPack& pack) {
  const char* s = SpecInt(v);
  pack.sig += std::string(ScalarTypeToTritonType<T>()) + s + ",";
  if (std::strcmp(s, ":1") != 0) pack.ptrs.push_back(pack.Store(v));
}

inline void PushArg(float v, ArgPack& pack) {
  pack.ptrs.push_back(pack.Store(v));
  pack.sig += "fp32,";
}

inline void PushArg(double v, ArgPack& pack) {
  pack.ptrs.push_back(pack.Store(v));
  pack.sig += "fp64,";
}

// ---- launch wrapper ----

template <Device::Type kDev, typename... Args>
int LaunchJit(const char* op, void* stream, Grid grid, const JitConfig& config,
              Args&&... args) {
  ArgPack pack;
  pack.sig.reserve(256);
  (PushArg(std::forward<Args>(args), pack), ...);
  for (const auto& [name, val] : config.constexprs)
    pack.sig += name + "=" + std::to_string(val) + ",";
  if (!pack.sig.empty()) pack.sig.pop_back();

  void* scratch = pack.Store<uint64_t>(0);
  pack.ptrs.push_back(scratch);
  pack.ptrs.push_back(scratch);

  return LaunchKernel<kDev>(op, pack.sig.c_str(), stream, grid, config,
                            pack.ptrs.data());
}

template <Device::Type kDev, typename GridFn, typename... Args>
int LaunchJitAutotune(const char* op, void* stream,
                      const AutotuneConfig& config,
                      const std::vector<Tensor::Size>& key,
                      const std::vector<DataType>& dtype, GridFn grid_fn,
                      Args&&... args) {
  TargetInfo target = CurrentTarget<kDev>();

  std::string cache_key = op;
  for (auto d : key) cache_key += "|" + std::to_string(d);
  for (auto dt : dtype)
    cache_key += "|" + std::string(DataTypeToTritonType(dt));
  cache_key += "|sm" + std::to_string(target.arch);

  ArgPack pack;
  pack.sig.reserve(256);
  (PushArg(std::forward<Args>(args), pack), ...);

  std::vector<Grid> grids;
  grids.reserve(config.candidates.size());
  for (const auto& c : config.candidates) grids.push_back(grid_fn(c));

  JitConfig best =
      AutotuneBench(op, config.candidates, pack.sig, pack.ptrs, grids,
                    config.warmup, config.rep, cache_key.c_str(), target);

  Grid grid = grid_fn(best);

  for (const auto& [name, val] : best.constexprs)
    pack.sig += name + "=" + std::to_string(val) + ",";
  if (!pack.sig.empty()) pack.sig.pop_back();

  void* scratch = pack.Store<uint64_t>(0);
  pack.ptrs.push_back(scratch);
  pack.ptrs.push_back(scratch);

  return LaunchKernel<kDev>(op, pack.sig.c_str(), stream, grid, best,
                            pack.ptrs.data());
}

}  // namespace infini::ops

#endif
