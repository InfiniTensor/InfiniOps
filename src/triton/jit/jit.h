#ifndef INFINI_OPS_TRITON_JIT_H_
#define INFINI_OPS_TRITON_JIT_H_

#include <cassert>
#include <cstdint>
#include <cstring>
#include <deque>
#include <string>
#include <type_traits>
#include <vector>

#include "config.h"
#include "data_type.h"
#include "tensor.h"

namespace infini::ops {

struct TritonConfig : Config {
  TritonConfig() = default;

  TritonConfig(unsigned num_warps, unsigned num_stages,
               std::vector<std::pair<std::string, int>> constexprs)
      : num_warps(num_warps),
        num_stages(num_stages),
        constexprs(std::move(constexprs)) {}

  unsigned num_warps = 4;

  unsigned num_stages = 3;

  std::vector<std::pair<std::string, int>> constexprs;

  bool autotune = false;

  std::vector<TritonConfig> configs;

  int warmup = 5;

  int rep = 50;

  bool IsAutotune() const { return autotune; }

  int At(const std::string& key) const {
    for (const auto& [k, v] : constexprs)
      if (k == key) return v;
    assert(false && "`constexpr` not found");
    return 0;
  }

  void ApplyDefaults(const TritonConfig& defaults) {
    for (const auto& [dk, dv] : defaults.constexprs) {
      bool found = false;
      for (const auto& [k, v] : constexprs)
        if (k == dk) {
          found = true;
          break;
        }
      if (!found) constexprs.push_back({dk, dv});
    }
  }
};

struct Grid {
  unsigned x = 1;

  unsigned y = 1;

  unsigned z = 1;
};

struct DeviceInfo {
  int id = 0;

  int arch = 0;
};

bool CompilerInit();

int CompileKernel(const char* op_name, const char* out_prefix, int num_warps,
                  int num_stages, int device_id, const char* signature);

int LaunchKernel(const char* op_name, const char* signature_str, void* stream,
                 Grid grid, TritonConfig config, void** args);

void* GetKernel(const char* op_name, const char* signature_str, void* stream,
                const TritonConfig& config, unsigned* out_shared);

DeviceInfo CurrentDevice();

TritonConfig AutotuneBench(const char* op_name,
                           const std::vector<TritonConfig>& configs,
                           const std::string& sig,
                           const std::vector<void*>& ptrs,
                           const std::vector<Grid>& grids, int warmup, int rep,
                           const char* key, int device_id);

// ---- specialization ----

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
    if constexpr (sizeof(T) == 8) return std::is_signed_v<T> ? "i64" : "i32";
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

template <typename... Args>
int LaunchJit(const char* op, void* stream, Grid grid, TritonConfig config,
              Args&&... args) {
  ArgPack pack;
  pack.sig.reserve(256);
  (PushArg(std::forward<Args>(args), pack), ...);
  for (const auto& [name, val] : config.constexprs)
    pack.sig += name + "=" + std::to_string(val) + ",";
  if (!pack.sig.empty()) pack.sig.pop_back();

  // Triton needs two trailing scratch pointers.
  void* scratch = pack.Store<uint64_t>(0);
  pack.ptrs.push_back(scratch);
  pack.ptrs.push_back(scratch);

  return LaunchKernel(op, pack.sig.c_str(), stream, grid, config,
                      pack.ptrs.data());
}

template <typename GridFn, typename... Args>
int LaunchJitAutotune(const char* op, void* stream, const TritonConfig& config,
                      const std::vector<Tensor::Size>& key_dims, DataType dtype,
                      GridFn grid_fn, Args&&... args) {
  std::string cache_key = op;
  for (auto d : key_dims) cache_key += "|" + std::to_string(d);
  cache_key += "|dt=" + std::to_string(static_cast<int>(dtype));

  ArgPack pack;
  pack.sig.reserve(256);
  (PushArg(std::forward<Args>(args), pack), ...);

  std::vector<Grid> grids;
  grids.reserve(config.configs.size());
  for (const auto& c : config.configs) grids.push_back(grid_fn(c));

  TritonConfig best = AutotuneBench(op, config.configs, pack.sig, pack.ptrs,
                                    grids, config.warmup, config.rep,
                                    cache_key.c_str(), CurrentDevice().id);

  Grid grid = grid_fn(best);

  for (const auto& [name, val] : best.constexprs)
    pack.sig += name + "=" + std::to_string(val) + ",";
  if (!pack.sig.empty()) pack.sig.pop_back();

  void* scratch = pack.Store<uint64_t>(0);
  pack.ptrs.push_back(scratch);
  pack.ptrs.push_back(scratch);

  return LaunchKernel(op, pack.sig.c_str(), stream, grid, best,
                      pack.ptrs.data());
}

}  // namespace infini::ops

#endif
