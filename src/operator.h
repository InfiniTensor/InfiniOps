#ifndef INFINI_OPS_OPERATOR_H_
#define INFINI_OPS_OPERATOR_H_

#include <cassert>
#include <memory>
#include <optional>
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

template <typename Key>
struct CacheKeyBuilder {
  template <typename... Args>
  detail::CacheKey operator()(const Config& config, const Args&... args) const {
    return detail::CacheKey::Build(config.implementation_index(), args...);
  }
};

template <typename Key, Device::Type kDev>
struct ActiveImplementations;

class OperatorBase {
 public:
  virtual ~OperatorBase() = default;

  virtual std::size_t workspace_size_in_bytes() const { return 0; }

  void set_handle(const Handle& handle) { handle_ = handle; }

  void set_config(const Config& config) { config_ = config; }

  void set_stream(void* stream) { stream_ = stream; }

  void set_workspace(void* workspace) { workspace_ = workspace; }

  void set_workspace_size_in_bytes(std::size_t workspace_size_in_bytes) {
    workspace_size_in_bytes_ = workspace_size_in_bytes;
  }

 protected:
  Handle handle_;

  Config config_;

  void* stream_{nullptr};

  void* workspace_{nullptr};

  std::size_t workspace_size_in_bytes_{0};
};

template <typename Key, Device::Type device_type = Device::Type::kCount,
          std::size_t implementation_index = 0>
class Operator : public OperatorBase {
 public:
  // Invalidate the operator cache.  Cached operators are destroyed on the
  // next `call()` invocation.  Intended for test isolation — production
  // code should never call this.
  static void clear_cache() {
#if defined(INFINI_OPS_ENABLE_HOST_RANGE_PROFILING)
    HostRangeProfiler::InvalidateOperatorCaches();
#else
    ++cache_generation_;
#endif
  }

  template <typename... Args>
  static std::unique_ptr<Operator> Make(const Config& config,
                                        const Tensor tensor, Args&&... args) {
    return MakeWithDevice(config, tensor.device().type(), tensor,
                          std::forward<Args>(args)...);
  }

  template <typename... Args>
  static std::unique_ptr<Operator> Make(const Tensor tensor, Args&&... args) {
    return Make({}, tensor, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static std::unique_ptr<Operator> Make(const Config& config,
                                        const std::vector<Tensor> tensors,
                                        Args&&... args) {
    assert(!tensors.empty() && "operator tensor list input cannot be empty");

    return MakeWithDevice(config, tensors.front().device().type(), tensors,
                          std::forward<Args>(args)...);
  }

  template <typename... Args>
  static std::unique_ptr<Operator> Make(const std::vector<Tensor> tensors,
                                        Args&&... args) {
    return Make({}, tensors, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static void Call(const Handle& handle, const Config& config,
                   const Args&... args) {
    INFINI_OPS_HOST_RANGE_SCOPE(HostRangeLayer::kOperatorCall);

    static thread_local std::unordered_map<detail::CacheKey,
                                           std::unique_ptr<Operator>>
        cache;
    static thread_local std::size_t generation{0};

    const auto current_generation = [&]() {
#if defined(INFINI_OPS_ENABLE_HOST_RANGE_PROFILING)
      return HostRangeProfiler::OperatorCacheGeneration();
#else
      return cache_generation_;
#endif
    }();

    if (generation != current_generation) {
      cache.clear();
      generation = current_generation;
    }

#if defined(INFINI_OPS_ENABLE_HOST_RANGE_PROFILING)
    auto key = [&]() {
      INFINI_OPS_HOST_RANGE_SCOPE(HostRangeLayer::kCacheKey);
      return CacheKeyBuilder<Key>{}(config, args...);
    }();

    auto it = [&]() {
      INFINI_OPS_HOST_RANGE_SCOPE(HostRangeLayer::kCacheLookup);
      return cache.find(key);
    }();

    if (it == cache.end()) {
      INFINI_OPS_HOST_RANGE_SCOPE(HostRangeLayer::kCacheConstruct);
      auto new_op = Make(config, args...);
      it = cache.emplace(std::move(key), std::move(new_op)).first;
    }
#else
    auto key = CacheKeyBuilder<Key>{}(config, args...);

    auto it{cache.find(key)};

    if (it == cache.end()) {
      it = cache.emplace(std::move(key), Make(config, args...)).first;
    }
#endif

    auto& op{it->second};

    INFINI_OPS_HOST_RANGE_SCOPE(HostRangeLayer::kOperatorInvoke);
    return (*op)(handle, args...);
  }

  template <typename... Args>
  static void Call(const Tensor tensor, const Args&... args) {
    return Call({}, {}, tensor, args...);
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

  // Profiling builds use a process-wide generation owned by
  // `HostRangeProfiler` so pybind and libinfiniops share invalidation state.
#if !defined(INFINI_OPS_ENABLE_HOST_RANGE_PROFILING)
  static inline std::size_t cache_generation_{0};
#endif
};

// Maximum number of implementation slots per (operator, device) pair.
// Increase this value when adding operators with more implementations.
constexpr std::size_t kMaxImplementations = 16;

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

}  // namespace infini::ops

#endif
