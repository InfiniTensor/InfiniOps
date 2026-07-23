#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infini::ops {

enum class HostRangeLayer {
  kBindingBody,
  kTensorConversion,
  kDispatchCall,
  kOperatorCall,
  kCacheKey,
  kCacheLookup,
  kCacheConstruct,
  kOperatorInvoke,
  kBackendSubmit,
  kCalibrationDepth1,
  kCalibrationDepth2,
  kCalibrationDepth3,
  kCount,
};

const char *HostRangeLayerName(HostRangeLayer layer);

struct HostRangeSummary {
  HostRangeLayer layer;
  std::size_t count;
  double inclusive_mean;
  double inclusive_median;
  double self_mean;
  double self_median;
};

class HostRangeProfiler {
 public:
  static bool IsCompiled();
  static void Start();
  static std::vector<HostRangeSummary> Stop();
  static std::vector<HostRangeSummary> Calibrate(std::size_t iterations);
};

class HostRangeScope {
 public:
  explicit HostRangeScope(HostRangeLayer layer);
  ~HostRangeScope() noexcept;

  HostRangeScope(const HostRangeScope &) = delete;
  HostRangeScope &operator=(const HostRangeScope &) = delete;
  HostRangeScope(HostRangeScope &&) = delete;
  HostRangeScope &operator=(HostRangeScope &&) = delete;

 private:
  bool active_{false};
  const void *owner_{nullptr};
  std::uint64_t session_id_{0};
  HostRangeLayer layer_{HostRangeLayer::kCount};
  std::size_t depth_{0};
};

}  // namespace infini::ops

#if defined(INFINI_OPS_ENABLE_HOST_RANGE_PROFILING)
#define INFINI_OPS_HOST_RANGE_SCOPE_CONCAT_IMPL(lhs, rhs) lhs##rhs
#define INFINI_OPS_HOST_RANGE_SCOPE_CONCAT(lhs, rhs) \
  INFINI_OPS_HOST_RANGE_SCOPE_CONCAT_IMPL(lhs, rhs)
#define INFINI_OPS_HOST_RANGE_SCOPE(layer)                              \
  ::infini::ops::HostRangeScope INFINI_OPS_HOST_RANGE_SCOPE_CONCAT(     \
      infini_ops_host_range_scope_, __COUNTER__)(layer)
#else
#define INFINI_OPS_HOST_RANGE_SCOPE(layer)
#endif
