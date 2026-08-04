#ifndef INFINI_OPS_OPERATOR_H_
#define INFINI_OPS_OPERATOR_H_

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "config.h"
#include "dispatcher.h"
#include "handle.h"
#include "host_range_profiler.h"
#include "tensor.h"

#ifdef WITH_TUNING
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include "runtime.h"
#include "tuning_manager.h"
#include "tuning_signature.h"
#endif

namespace infini::ops::detail {

struct CacheKey {
  std::size_t hash;

  std::vector<Tensor> tensors;

  std::size_t scalar_hash;

  template <typename... Args>
  static CacheKey Build(const Args&... args) {
    CacheKey key;
    key.hash = 0;
    key.scalar_hash = 0;
    (key.Absorb(args), ...);
    return key;
  }

 private:
  void Absorb(const Tensor& t) {
    HashCombine(hash, t);
    tensors.push_back(t);
  }

  void Absorb(const std::vector<Tensor>& ts) {
    HashCombine(hash, ts.size());
    for (const auto& t : ts) {
      HashCombine(hash, t);
      tensors.push_back(t);
    }
  }

  template <typename T>
  void Absorb(const T& v) {
    HashCombine(hash, v);
    HashCombine(scalar_hash, v);
  }
};

template <typename Functor, typename... Args, auto... implementation_indices>
auto DispatchImplementation(std::size_t implementation_index, Functor&& func,
                            std::string_view context_str,
                            List<implementation_indices...>, Args&&... args) {
  return DispatchFunc<std::size_t,
                      static_cast<std::size_t>(implementation_indices)...>(
      implementation_index, std::forward<Functor>(func), context_str,
      std::forward<Args>(args)...);
}

template <auto... values>
std::vector<std::size_t> ListToVector(List<values...>) {
  return {static_cast<std::size_t>(values)...};
}

template <typename ValueType, auto... values>
bool ListContains(ValueType value, List<values...>) {
  return ((value == static_cast<ValueType>(values)) || ...);
}

template <typename TensorLike, typename = void>
class IsTensorLike : public std::false_type {};

template <typename TensorLike>
class IsTensorLike<
    TensorLike,
    std::void_t<decltype(std::declval<const TensorLike&>().data()),
                decltype(std::declval<const TensorLike&>().shape()),
                decltype(std::declval<const TensorLike&>().strides()),
                decltype(std::declval<const TensorLike&>().dtype()),
                decltype(std::declval<const TensorLike&>().device())>>
    : public std::true_type {};

template <typename T, typename std::enable_if_t<
                          IsTensorLike<std::decay_t<T>>::value, int> = 0>
Tensor AsCallArg(const T& tensor) {
  return Tensor{tensor};
}

template <typename T, typename std::enable_if_t<
                          !IsTensorLike<std::decay_t<T>>::value, int> = 0>
const T& AsCallArg(const T& value) {
  return value;
}

template <typename Key, typename TensorLike, typename Args, typename = void>
class HasMakeReturnValueImpl : public std::false_type {};

template <typename Key, typename TensorLike, typename... Args>
class HasMakeReturnValueImpl<
    Key, TensorLike, std::tuple<Args...>,
    std::void_t<decltype(Key::MakeReturnValue(std::declval<const TensorLike&>(),
                                              std::declval<const Args&>()...))>>
    : public std::true_type {};

template <typename Key, typename... Args>
class HasMakeReturnValueImpl<Key, Tensor, std::tuple<Args...>>
    : public std::false_type {};

template <typename Key, typename TensorLike, typename... Args>
class HasMakeReturnValue
    : public HasMakeReturnValueImpl<Key, std::decay_t<TensorLike>,
                                    std::tuple<Args...>> {};

inline bool TraceOperatorCallsEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("INFINI_OPS_TRACE_CALLS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

template <typename Key>
constexpr std::string_view OperatorName() {
#if defined(__clang__) || defined(__GNUC__)
  std::string_view name{__PRETTY_FUNCTION__};
  constexpr std::string_view marker{"Key = "};
  const auto start = name.find(marker) + marker.size();
  const auto end = name.find_first_of(";]", start);
  name = name.substr(start, end - start);
#else
  std::string_view name{"unknown"};
#endif
  constexpr std::string_view namespace_prefix{"infini::ops::"};
  if (name.rfind(namespace_prefix, 0) == 0) {
    name.remove_prefix(namespace_prefix.size());
  }
  return name;
}

template <typename Key>
void TraceOperatorCall(const CacheKey& key, const Config& config) {
  if (!TraceOperatorCallsEnabled()) return;

  const auto device =
      key.tensors.empty()
          ? std::string_view{"unknown"}
          : Device::StringFromType(key.tensors.front().device().type());
  constexpr auto operator_name = OperatorName<Key>();
  std::fprintf(stderr,
               "[INFINI_OPS_TRACE_CALLS] {\"operator_name\": \"%.*s\", "
               "\"device_type\": \"%.*s\", \"implementation\": %zu}\n",
               static_cast<int>(operator_name.size()), operator_name.data(),
               static_cast<int>(device.size()), device.data(),
               config.implementation_index());
}

}  // namespace infini::ops::detail

template <>
struct std::hash<infini::ops::detail::CacheKey> {
  std::size_t operator()(const infini::ops::detail::CacheKey& key) const {
    return key.hash;
  }
};

template <>
struct std::equal_to<infini::ops::detail::CacheKey> {
  bool operator()(const infini::ops::detail::CacheKey& a,
                  const infini::ops::detail::CacheKey& b) const {
    if (a.scalar_hash != b.scalar_hash) return false;
    if (a.tensors.size() != b.tensors.size()) return false;
    std::equal_to<infini::ops::Tensor> eq;
    for (std::size_t i = 0; i < a.tensors.size(); ++i) {
      if (!eq(a.tensors[i], b.tensors[i])) return false;
    }
    return true;
  }
};

namespace infini::ops {

#ifdef WITH_TUNING
namespace detail {

// 通用提取算子名称：从模板类型 Key 中提取短名（如 "RmsNorm"）
// 使用编译器内置宏 __PRETTY_FUNCTION__ 或 __FUNCSIG__
template <typename Key>
std::string ExtractOperatorName() {
#if defined(__GNUC__) || defined(__clang__)
  // GCC/Clang: __PRETTY_FUNCTION__ 包含完整函数签名
  // 例如: "std::string infini::ops::detail::ExtractOperatorName() [Key = infini::ops::RmsNorm]"
  std::string_view sig = __PRETTY_FUNCTION__;

  // 查找 "Key = " 后的类型名
  auto key_pos = sig.find("Key = ");
  if (key_pos == std::string_view::npos) return "UnknownOp";

  key_pos += 6;  // 跳过 "Key = "
  auto end_pos = sig.find_first_of("]>;", key_pos);
  std::string full_name(sig.substr(key_pos, end_pos - key_pos));

  // 提取最后一个 "::" 之后的短名
  auto last_colon = full_name.rfind("::");
  if (last_colon != std::string::npos) {
    return full_name.substr(last_colon + 2);
  }
  return full_name;
#elif defined(_MSC_VER)
  // MSVC: __FUNCSIG__ 类似
  std::string_view sig = __FUNCSIG__;
  auto key_pos = sig.find("Key=");
  if (key_pos == std::string_view::npos) return "UnknownOp";
  key_pos += 4;
  auto end_pos = sig.find_first_of("]>,", key_pos);
  std::string full_name(sig.substr(key_pos, end_pos - key_pos));
  auto last_colon = full_name.rfind("::");
  if (last_colon != std::string::npos) {
    return full_name.substr(last_colon + 2);
  }
  return full_name;
#else
  return "UnknownOp";
#endif
}

// 设备同步：等待该设备上此前提交的所有异步任务真正执行完毕，
// 这样基于 CPU 计时器（std::chrono）的测速才准确（GPU 提交是异步的）。
// 通过 DispatchFunc 把运行期 dev_type 派发到编译期的 Runtime<kDev>，
// CPU 与各 GPU 后端都提供 DeviceSynchronize()（见 InfiniRT runtime_.h）。
inline void SyncDevice(Device::Type dev_type) {
  if (!ListContains(dev_type, ActiveDevices<void>{})) {
    return;
  }
  DispatchFunc<ActiveDevices<void>>(
      dev_type,
      [](auto device_tag) {
        constexpr Device::Type kDev = decltype(device_tag)::value;
        infini::rt::runtime::Runtime<kDev>::DeviceSynchronize();
      },
      "SyncDevice");
}

// 读取整数型环境变量，缺省或非法时返回 fallback。
inline int EnvInt(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  int parsed = std::atoi(v);
  return parsed > 0 ? parsed : fallback;
}

// 从参数列表中找出首个张量参数的设备类型（与 Operator::Make 的推断一致）。
// 支持 Tensor 与 vector<Tensor>；其余参数跳过。找不到则返回 kCount。
inline Device::Type FirstDeviceTypeHelper(bool& found) {
  found = false;
  return Device::Type::kCount;
}

template <typename First, typename... Rest>
Device::Type FirstDeviceTypeHelper(bool& found, const First& first,
                                   const Rest&... rest) {
  if constexpr (std::is_same_v<std::decay_t<First>, Tensor>) {
    found = true;
    return first.device().type();
  } else if constexpr (std::is_same_v<std::decay_t<First>,
                                      std::vector<Tensor>>) {
    if (!first.empty()) {
      found = true;
      return first.front().device().type();
    }
    return FirstDeviceTypeHelper(found, rest...);
  } else {
    return FirstDeviceTypeHelper(found, rest...);
  }
}

template <typename... Args>
Device::Type FirstDeviceType(const Args&... args) {
  bool found = false;
  return FirstDeviceTypeHelper(found, args...);
}

}  // namespace detail
#endif

template <typename Key>
struct CacheKeyBuilder {
  template <typename... Args>
  detail::CacheKey operator()(const Config& config, const Args&... args) const {
    return detail::CacheKey::Build(config.implementation_index(), args...);
  }
};

// 声明函数：ResolveConfig / ResolveConfigOnline
template <typename Key, typename... Args>
Config ResolveConfig(const Config& config, Device::Type dev_type,
                     const Args&... args);

template <typename Key, typename... Args>
Config ResolveConfigOnline(const Handle& handle, const Config& config,
                           const Args&... args);

template <typename Key, Device::Type kDev>
struct ActiveImplementations;

class OperatorBase {
 public:
  virtual ~OperatorBase() = default;

  virtual std::size_t workspace_size_in_bytes() const { return 0; }

  void set_handle(const Handle& handle) { handle_ptr_ = handle.Clone(); }

  void set_config(const Config& config) { config_ptr_ = config.Clone(); }

  void set_stream(void* stream) { stream_ = stream; }

  void set_workspace(void* workspace) { workspace_ = workspace; }

  void set_workspace_size_in_bytes(std::size_t workspace_size_in_bytes) {
    workspace_size_in_bytes_ = workspace_size_in_bytes;
  }

 protected:
  std::unique_ptr<Handle> handle_ptr_;

  std::unique_ptr<Config> config_ptr_;

  void* stream_{nullptr};

  void* workspace_{nullptr};

  std::size_t workspace_size_in_bytes_{0};
};

template <typename Key, Device::Type device_type = Device::Type::kCount,
          std::size_t implementation_index = 0>
class Operator : public OperatorBase {
 public:
  // Invalidate the operator cache. Cached operators are destroyed on the next
  // `Call()` invocation. Intended for test isolation; production code should
  // never call this.
  static void clear_cache() {
    cache_generation_.fetch_add(1, std::memory_order_relaxed);
  }

  template <typename... Args>
  static std::unique_ptr<Operator> Make(const Config& config,
                                        const Tensor tensor, Args&&... args) {
    // 在构造算子前解析配置：如果启用自动选择，查询调优缓存
    Config resolved = ResolveConfig<Key>(config, tensor.device().type(), tensor, args...);
    return MakeWithDevice(resolved, tensor.device().type(), tensor,
                          std::forward<Args>(args)...);
  }

  template <typename... Args>
  static std::unique_ptr<Operator> Make(const Tensor tensor, Args&&... args) {
    return Make(DefaultConfig(tensor.device().type()), tensor,
                std::as_const(args)...);
  }

  template <typename... Args>
  static std::unique_ptr<Operator> Make(const Config& config,
                                        const std::vector<Tensor> tensors,
                                        Args&&... args) {
    assert(!tensors.empty() && "operator tensor list input cannot be empty");

    // 同样在构造前解析配置
    Config resolved = ResolveConfig<Key>(config, tensors.front().device().type(), tensors, args...);
    return MakeWithDevice(resolved, tensors.front().device().type(), tensors,
                          std::forward<Args>(args)...);
  }

  template <typename... Args>
  static std::unique_ptr<Operator> Make(const std::vector<Tensor> tensors,
                                        Args&&... args) {
    assert(!tensors.empty() && "operator tensor list input cannot be empty");

    return Make(DefaultConfig(tensors.front().device().type()), tensors,
                std::as_const(args)...);
  }

  template <typename... Args>
  static void Call(const Handle& handle, const Config& config,
                   const Args&... args) {
    [[maybe_unused]] HostRangeScope host_range_operator_call{
        HostRangeLayer::kOperatorCall};

    static thread_local std::unordered_map<detail::CacheKey,
                                           std::unique_ptr<Operator>>
        cache;
    static thread_local std::size_t generation{0};

    const auto cache_generation =
        cache_generation_.load(std::memory_order_relaxed);
    if (generation != cache_generation) {
      cache.clear();
      generation = cache_generation;
    }

    const Config effective_config =
        ResolveConfigOnline<Key>(handle, config, args...);

#if defined(INFINI_OPS_ENABLE_HOST_RANGE_PROFILING)
    auto key = [&]() {
      HostRangeScope host_range_cache_key{HostRangeLayer::kCacheKey};
      return CacheKeyBuilder<Key>{}(effective_config, args...);
    }();
    detail::TraceOperatorCall<Key>(key, effective_config);

    auto it = [&]() {
      HostRangeScope host_range_cache_lookup{HostRangeLayer::kCacheLookup};
      return cache.find(key);
    }();

    if (it == cache.end()) {
      HostRangeScope host_range_cache_construct{
          HostRangeLayer::kCacheConstruct};
      auto new_op = Make(effective_config, args...);
      it = cache.emplace(std::move(key), std::move(new_op)).first;
    }
#else
    auto key = CacheKeyBuilder<Key>{}(effective_config, args...);
    detail::TraceOperatorCall<Key>(key, effective_config);

    auto it{cache.find(key)};

    if (it == cache.end()) {
      it = cache.emplace(std::move(key), Make(effective_config, args...)).first;
    }
#endif

    auto& op{it->second};

    [[maybe_unused]] HostRangeScope host_range_operator_invoke{
        HostRangeLayer::kOperatorInvoke};
    return (*op)(handle, args...);
  }

  template <typename... Args>
  static void Call(const Tensor tensor, const Args&... args) {
    return Call({}, DefaultConfig(tensor.device().type()), tensor, args...);
  }

  template <
      typename TensorLike, typename... Args,
      typename std::enable_if_t<
          detail::HasMakeReturnValue<Key, TensorLike, Args...>::value, int> = 0>
  static auto Call(const TensorLike& tensor, const Args&... args) {
    return CallReturning(tensor, args...);
  }

  static std::vector<std::size_t> active_implementation_indices(
      Device::Type dev_type) {
    if (!detail::ListContains(dev_type, ActiveDevices<Key>{})) {
      return {};
    }

    std::vector<std::size_t> result;
    DispatchFunc<ActiveDevices<Key>>(
        dev_type,
        [&](auto device_tag) {
          constexpr Device::Type kDev = decltype(device_tag)::value;
          result = detail::ListToVector(
              typename ActiveImplementations<Key, kDev>::type{});
        },
        "Operator::active_implementation_indices");
    return result;
  }

  template <typename... Args>
  void operator()(const Handle& handle, const Args&... args) {
    set_handle(handle);
    set_stream(handle.stream());
    set_workspace(handle.workspace());
    set_workspace_size_in_bytes(handle.workspace_size_in_bytes());

    return operator()(args...);
  }

  template <typename... Args>
  void operator()(const Args&... args) const {
    return (*static_cast<const Key*>(this))(args...);
  }

 protected:
  static constexpr Device::Type device_type_{device_type};

  static constexpr std::size_t implementation_index_{implementation_index};

 private:
  template <auto first, auto... rest>
  static constexpr std::size_t FirstActiveImplementationIndex(
      List<first, rest...>) {
    return static_cast<std::size_t>(first);
  }

  static std::size_t FirstActiveImplementationIndex(List<>) {
    assert(false && "operator has no active implementation for this device");
    std::abort();
  }

  static std::size_t DefaultImplementationIndex(Device::Type dev_type) {
    std::size_t default_index{0};

    DispatchFunc<ActiveDevices<Key>>(
        dev_type,
        [&](auto device_tag) {
          constexpr Device::Type kDev = decltype(device_tag)::value;
          default_index = FirstActiveImplementationIndex(
              typename ActiveImplementations<Key, kDev>::type{});
        },
        "Operator::DefaultImplementationIndex");

    return default_index;
  }

  static Config DefaultConfig(Device::Type dev_type) {
    Config config;
    config.set_implementation_index(DefaultImplementationIndex(dev_type));

    return config;
  }

  template <typename TensorLike, typename... Args>
  static auto CallReturning(const TensorLike& tensor, const Args&... args) {
    auto out = Key::MakeReturnValue(tensor, args...);
    Key::Call(detail::AsCallArg(tensor), detail::AsCallArg(args)...,
              detail::AsCallArg(out));
    return out;
  }

  template <typename... Args>
  static std::unique_ptr<Operator> MakeWithDevice(
      const Config& config, Device::Type dispatch_device_type, Args&&... args) {
    std::unique_ptr<Operator> op_ptr;
    auto cache_args = std::forward_as_tuple(args...);

    DispatchFunc<ActiveDevices<Key>>(
        dispatch_device_type,
        [&](auto device_tag) {
          constexpr Device::Type kDev = decltype(device_tag)::value;
          detail::DispatchImplementation(
              config.implementation_index(),
              [&](auto implementation_tag) {
                constexpr std::size_t kImplementationIndex =
                    decltype(implementation_tag)::value;
                if constexpr (std::is_constructible_v<
                                  Operator<Key, kDev, kImplementationIndex>,
                                  Args...>) {
                  std::apply(
                      [&](auto&... cached_args) {
                        op_ptr = std::make_unique<
                            Operator<Key, kDev, kImplementationIndex>>(
                            cached_args...);
                      },
                      cache_args);
                } else {
                  assert(false &&
                         "operator is not implemented for this device and "
                         "implementation index");
                }
              },
              "Operator::Make(implementation_index)",
              typename ActiveImplementations<Key, kDev>::type{});
        },
        "Operator::Make");

    op_ptr->set_config(config);

    return op_ptr;
  }

  static inline std::atomic<std::size_t> cache_generation_{0};
};

// Maximum number of implementation slots per (operator, device) pair.
// Increase this value when adding operators with more implementations.
constexpr std::size_t kMaxImplementations = 32;

// SFINAE-based implementation detection. A partial specialization
// `Operator<Key, kDev, N>` inherits from `Key` (the operator base class),
// while the unspecialized primary template inherits only from `OperatorBase`.
// `std::is_base_of` distinguishes the two at compile time, eliminating the
// need for manual `registry.h` files.
template <typename Key, Device::Type kDev, std::size_t N,
          bool = std::is_base_of_v<Key, Operator<Key, kDev, N>>>
struct ActiveImplementationsImpl {
  using type = List<>;
};

template <typename Key, Device::Type kDev, std::size_t N>
struct ActiveImplementationsImpl<Key, kDev, N, true> {
  using type = List<N>;
};

namespace detail {

template <typename Key, Device::Type kDev, typename Seq>
struct ActiveImplementationsHelper;

template <typename Key, Device::Type kDev, std::size_t... ns>
struct ActiveImplementationsHelper<Key, kDev, std::index_sequence<ns...>> {
  using type = typename Flatten<
      typename ActiveImplementationsImpl<Key, kDev, ns>::type...>::type;
};

}  // namespace detail

template <typename Key, Device::Type kDev>
struct ActiveImplementations {
  using type = typename detail::ActiveImplementationsHelper<
      Key, kDev, std::make_index_sequence<kMaxImplementations>>::type;
};

// 解析配置：如果启用自动选择且编译时启用了调优，则查询最优实现。
template <typename Key, typename... Args>
Config ResolveConfig(const Config& config, Device::Type dev_type,
                     const Args&... args) {
#ifdef WITH_TUNING
  // 仅当用户未显式指定实现时才启用自动选择
  if (config.auto_select()) {
    auto indices = Operator<Key>::active_implementation_indices(dev_type);
    if (!indices.empty()) {
      // 通用地从参数中提取形状和类型，构建调优签名
      auto signature = TuningSignature::Build(args...);

      // 从模板类型 Key 提取算子短名（如 "RmsNorm"），查询调优缓存
      auto op_name = detail::ExtractOperatorName<Key>();
      auto tuned_index =
          TuningManager::Instance().Lookup(op_name, dev_type, signature);

      Config resolved = config;
      if (tuned_index.has_value()) {
        // 检查调优结果是否在当前编译的可用实现列表中
        bool is_valid = std::find(indices.begin(), indices.end(),
                                  *tuned_index) != indices.end();
        if (is_valid) {
          resolved.set_implementation_index(*tuned_index);
        } else {
          // 警告：调优数据指向的实现在本次编译中不存在（如编译选项不同）
          std::cerr << "[Tuning] Warning: tuned implementation " << *tuned_index
                    << " for " << op_name << " on "
                    << Device::StringFromType(dev_type)
                    << " is not available (compiled indices:";
          for (auto idx : indices) std::cerr << " " << idx;
          std::cerr << "), falling back to " << indices.front() << std::endl;
          resolved.set_implementation_index(indices.front());
        }
      } else {
        // 未找到调优数据，回退到第一个可用实现
        resolved.set_implementation_index(indices.front());
      }
      return resolved;
    }
  }
#endif
  // 未启用调优，或用户已显式指定实现（auto_select_=false），原样返回
  return config;
}

#ifdef WITH_TUNING
// 基准测试单个实现：用固定的实现索引构造算子并运行若干次，返回最快耗时（秒）。
// 预热 1 次让设备进入稳定状态，正式测 5 次取最小值（默认，可用环境变量覆盖）。
template <typename Key, typename... Args>
double BenchmarkImplementation(const Handle& handle, Device::Type dev_type,
                               std::size_t impl_index, const Args&... args) {
  // 用显式索引构造该实现（set_implementation_index 会关闭 auto_select，因此不会递归触发调优）
  Config fixed;
  fixed.set_implementation_index(impl_index);

  auto op = Operator<Key>::Make(fixed, args...);
  if (!op) {
    return std::numeric_limits<double>::infinity();
  }

  const int warmup = detail::EnvInt("INFINI_OPS_TUNING_WARMUP", 1);
  const int repeat = detail::EnvInt("INFINI_OPS_TUNING_REPEAT", 5);

  // 预热
  for (int i = 0; i < warmup; ++i) {
    (*op)(handle, args...);
  }
  detail::SyncDevice(dev_type);

  // 测速：逐次计时取最小，减少系统抖动干扰
  double best = std::numeric_limits<double>::infinity();
  for (int i = 0; i < repeat; ++i) {
    auto start = std::chrono::steady_clock::now();
    (*op)(handle, args...);
    detail::SyncDevice(dev_type);
    auto end = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration<double>(end - start).count();
    best = std::min(best, elapsed);
  }
  return best;
}
#endif

// 解析调优配置：
//   1) 未开 WITH_TUNING 或用户已指定实现 → 原样返回；
//   2) 查缓存命中 → 直接采用记录的最优实现；
//   3) 未命中 → 现场基准测试所有候选实现，选最快者，写盘记录并采用。
template <typename Key, typename... Args>
Config ResolveConfigOnline(const Handle& handle, const Config& config,
                           const Args&... args) {

#ifdef WITH_TUNING
  if (config.auto_select() && TuningManager::Instance().IsEnabled()) {
    // 从首个张量参数推断设备类型
    Device::Type dev_type = detail::FirstDeviceType(args...);
    auto indices = Operator<Key>::active_implementation_indices(dev_type);

    // 只有一个候选实现时无需测速，直接用它
    if (indices.size() == 1) {
      Config resolved = config;
      resolved.set_implementation_index(indices.front());
      return resolved;
    }

    if (!indices.empty()) {
      auto signature = TuningSignature::Build(args...);
      auto op_name = detail::ExtractOperatorName<Key>();

      // 先查已有记录
      auto tuned = TuningManager::Instance().Lookup(op_name, dev_type, signature);

      std::size_t chosen;
      if (tuned.has_value() &&
          std::find(indices.begin(), indices.end(), *tuned) != indices.end()) {
        // 命中且有效
        chosen = *tuned;
      } else {
        // 未命中（或记录失效）：现场基准测试所有候选实现
        chosen = indices.front();
        double best_time = std::numeric_limits<double>::infinity();
        for (auto idx : indices) {
          double t = BenchmarkImplementation<Key>(handle, dev_type, idx,
                                                   args...);
          if (t < best_time) {
            best_time = t;
            chosen = idx;
          }
        }
        // 记录并立即写盘，供后续调用与后续进程复用
        TuningManager::Instance().Record(op_name, dev_type, signature, chosen);
        std::cout << "[Tuning] " << op_name << " on "
                  << Device::StringFromType(dev_type) << ": benchmarked "
                  << indices.size() << " impls, chose index " << chosen
                  << " (" << best_time * 1e6 << " us)" << std::endl;
      }

      Config resolved = config;
      resolved.set_implementation_index(chosen);
      return resolved;
    }
  }
#endif
  (void)handle;
  return config;
}

}  // namespace infini::ops

#endif
