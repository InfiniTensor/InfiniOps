#ifndef INFINI_OPS_OPERATOR_H_
#define INFINI_OPS_OPERATOR_H_

#include <cassert>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "config.h"
#include "dispatcher.h"
#include "handle.h"
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

// Forward declaration — defined after `Operator` using SFINAE auto-detection.
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
  template <typename... Args>
  static auto Make(const Config& config, const Tensor tensor, Args&&... args) {
    std::unique_ptr<Operator> op_ptr;

    DispatchFunc<ActiveDevices<Key>>(
        tensor.device().type(),
        [&](auto device_tag) {
          constexpr Device::Type kDev = decltype(device_tag)::value;
          detail::DispatchImplementation(
              config.implementation_index(),
              [&](auto implementation_tag) {
                constexpr std::size_t kImplementationIndex =
                    decltype(implementation_tag)::value;
                if constexpr (std::is_constructible_v<
                                  Operator<Key, kDev, kImplementationIndex>,
                                  const Tensor&, Args...>) {
                  op_ptr = std::make_unique<
                      Operator<Key, kDev, kImplementationIndex>>(
                      tensor, std::forward<Args>(args)...);
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

  template <typename... Args>
  static auto Make(const Tensor tensor, Args&&... args) {
    return Make({}, tensor, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static auto Call(const Handle& handle, const Config& config, Args&&... args) {
    static std::unordered_map<detail::CacheKey, std::unique_ptr<Operator>>
        cache;

    auto key = detail::CacheKey::Build(config.implementation_index(), args...);

    auto it{cache.find(key)};

    if (it == cache.end()) {
      // Pass args as lvalue refs so they remain valid for the `operator()` call
      // below. Forwarding rvalue temporaries into `Make()` would leave the args
      // in a moved-from (empty) state before operator() can use them.
      it = cache.emplace(std::move(key), Make(config, args...)).first;
    }

    auto& op{it->second};

    return (*op)(handle, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static auto Call(const Tensor tensor, Args&&... args) {
    return Call({}, {}, tensor, std::forward<Args>(args)...);
  }

  static std::vector<std::size_t> active_implementation_indices(
      Device::Type dev_type) {
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
  auto operator()(const Handle& handle, Args&&... args) {
    set_handle(handle);
    set_stream(handle.stream());
    set_workspace(handle.workspace());
    set_workspace_size_in_bytes(handle.workspace_size_in_bytes());

    return operator()(std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto operator()(Args&&... args) const {
    return (*static_cast<const Key*>(this))(std::forward<Args>(args)...);
  }

 protected:
  static constexpr Device::Type device_type_{device_type};

  static constexpr std::size_t implementation_index_{implementation_index};
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
