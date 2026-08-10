#ifndef INFINI_OPS_TRITON_JIT_BASE_H_
#define INFINI_OPS_TRITON_JIT_BASE_H_

#include <cassert>
#include <string>
#include <vector>

#include "config.h"
#include "device.h"
#include "driver.h"

namespace infini::ops {

// ---- types ----

struct JitConfig : Config {
  JitConfig() = default;

  JitConfig(unsigned num_warps, unsigned num_stages,
            std::vector<std::pair<std::string, int>> constexprs)
      : num_warps(num_warps),
        num_stages(num_stages),
        constexprs(std::move(constexprs)) {}

  bool autotune = false;

  unsigned num_warps = 4;

  unsigned num_stages = 3;

  std::vector<std::pair<std::string, int>> constexprs;

  int At(const std::string& key) const {
    for (const auto& [k, v] : constexprs)
      if (k == key) return v;
    assert(false && "`constexpr` not found");
    return 0;
  }

  void ApplyDefaults(const JitConfig& defaults) {
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

struct AutotuneConfig : public JitConfig {
  AutotuneConfig() { autotune = true; }

  std::vector<std::string> key;

  std::vector<JitConfig> candidates;

  int warmup = 25;

  int rep = 100;
};

struct Grid {
  unsigned x = 1;

  unsigned y = 1;

  unsigned z = 1;
};

struct TargetInfo {
  std::string type;

  int id = 0;

  int arch = 0;

  int warp_size = 0;
};

struct KernelMeta {
  std::string name;

  std::string binary_ext;

  unsigned shared = 0;

  unsigned num_warps = 0;

  int global_scratch_size = 0;

  int profile_scratch_size = 0;
};

// ---- declarations ----

bool CompilerInit();

int CompileKernel(const TargetInfo& target, const char* op_name,
                  const char* out_prefix, int num_warps, int num_stages,
                  const char* signature);

template <Device::Type kDev>
int LaunchKernel(const char* op_name, const char* signature_str, void* stream,
                 Grid grid, const JitConfig& config, void** args);

template <Device::Type kDev>
typename Driver<kDev>::Function GetKernel(const char* op_name,
                                          const char* signature_str,
                                          void* stream, const JitConfig& config,
                                          unsigned* out_shared);

template <Device::Type kDev>
TargetInfo CurrentTarget();

JitConfig AutotuneBench(const char* op_name,
                        const std::vector<JitConfig>& configs,
                        const std::string& sig, const std::vector<void*>& ptrs,
                        const std::vector<Grid>& grids, int warmup, int rep,
                        const char* key, const TargetInfo& target);

}  // namespace infini::ops

#endif
