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

struct config_t : Config {
  config_t() = default;
  config_t(unsigned num_warps, unsigned num_stages,
           std::vector<std::pair<std::string, int>> constexprs)
      : num_warps(num_warps),
        num_stages(num_stages),
        constexprs(std::move(constexprs)) {}

  unsigned num_warps = 4;
  unsigned num_stages = 3;
  std::vector<std::pair<std::string, int>> constexprs;

  bool autotune = false;
  std::vector<config_t> configs;
  int warmup = 5;
  int rep = 50;

  bool is_autotune() const { return autotune; }

  int at(const std::string& key) const {
    for (const auto& [k, v] : constexprs)
      if (k == key) return v;
    assert(false && "constexpr not found");
    return 0;
  }

  void apply_defaults(const config_t& defaults) {
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

struct grid_t {
  unsigned x = 1, y = 1, z = 1;
};

struct device_info_t {
  int id = 0;
  int arch = 0;
};

bool compiler_init();

int compile_kernel(const char* op_name, const char* out_prefix, int num_warps,

                   int num_stages, int device_id, const char* signature);

int launch_kernel(const char* op_name, const char* signature_str, void* stream,
                  grid_t grid, config_t config, void** args);

void* get_kernel(const char* op_name, const char* signature_str, void* stream,
                 const config_t& config, unsigned* out_shared);

device_info_t current_device();

config_t autotune_bench(const char* op_name,
                        const std::vector<config_t>& configs,
                        const std::string& sig, const std::vector<void*>& ptrs,
                        const std::vector<grid_t>& grids, int warmup, int rep,
                        const char* key, int device_id);

// ---- specialization ----

inline const char* spec_ptr(uintptr_t v) { return v % 16 == 0 ? ":16" : ""; }

template <typename T>
const char* spec_int(T v) {
  if (v == 1) return ":1";
  if ((v & 15) == 0) return ":16";
  return "";
}

// ---- DataType → Triton string ----

inline const char* dtype_to_ttype(DataType dt) {
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
const char* cstype_to_ttype() {
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

struct arg_pack_t {
  std::vector<void*> ptrs;
  std::deque<uint64_t> storage;
  std::string sig;

  template <typename T>
  void* store(T v) {
    static_assert(sizeof(T) <= sizeof(uint64_t),
                  "scalar arg wider than 8 bytes");
    uint64_t slot = 0;
    std::memcpy(&slot, &v, sizeof(T));
    storage.push_back(slot);
    return &storage.back();
  }
};

inline void _push_arg(const Tensor& t, arg_pack_t& pack) {
  auto ptr = reinterpret_cast<uintptr_t>(t.data());
  pack.sig +=
      std::string("*") + dtype_to_ttype(t.dtype()) + spec_ptr(ptr) + ",";
  pack.ptrs.push_back(pack.store(ptr));
}

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void _push_arg(T v, arg_pack_t& pack) {
  const char* s = spec_int(v);
  pack.sig += std::string(cstype_to_ttype<T>()) + s + ",";
  if (std::strcmp(s, ":1") != 0) pack.ptrs.push_back(pack.store(v));
}

inline void _push_arg(float v, arg_pack_t& pack) {
  pack.ptrs.push_back(pack.store(v));
  pack.sig += "fp32,";
}

inline void _push_arg(double v, arg_pack_t& pack) {
  pack.ptrs.push_back(pack.store(v));
  pack.sig += "fp64,";
}

// ---- launch wrapper ----

template <typename... Args>
int launch_jit(const char* op, void* stream, grid_t grid, config_t config,
               Args&&... args) {
  arg_pack_t pack;
  pack.sig.reserve(256);
  (_push_arg(std::forward<Args>(args), pack), ...);
  for (const auto& [name, val] : config.constexprs)
    pack.sig += name + "=" + std::to_string(val) + ",";
  if (!pack.sig.empty()) pack.sig.pop_back();

  // triton need
  void* scratch = pack.store<uint64_t>(0);
  pack.ptrs.push_back(scratch);
  pack.ptrs.push_back(scratch);

  return launch_kernel(op, pack.sig.c_str(), stream, grid, config,
                       pack.ptrs.data());
}

template <typename GridFn, typename... Args>
int launch_jit_autotune(const char* op, void* stream, const config_t& config,
                        const std::vector<Tensor::Size>& key_dims,
                        DataType dtype, GridFn grid_fn, Args&&... args) {
  std::string cache_key = op;
  for (auto d : key_dims) cache_key += "|" + std::to_string(d);
  cache_key += "|dt=" + std::to_string(static_cast<int>(dtype));

  arg_pack_t pack;
  pack.sig.reserve(256);
  (_push_arg(std::forward<Args>(args), pack), ...);

  std::vector<grid_t> grids;
  grids.reserve(config.configs.size());
  for (const auto& c : config.configs) grids.push_back(grid_fn(c));

  config_t best = autotune_bench(op, config.configs, pack.sig, pack.ptrs, grids,
                                 config.warmup, config.rep, cache_key.c_str(),
                                 current_device().id);

  grid_t grid = grid_fn(best);

  for (const auto& [name, val] : best.constexprs)
    pack.sig += name + "=" + std::to_string(val) + ",";
  if (!pack.sig.empty()) pack.sig.pop_back();

  void* scratch = pack.store<uint64_t>(0);
  pack.ptrs.push_back(scratch);
  pack.ptrs.push_back(scratch);

  return launch_kernel(op, pack.sig.c_str(), stream, grid, best,
                       pack.ptrs.data());
}

}  // namespace infini::ops

#endif
