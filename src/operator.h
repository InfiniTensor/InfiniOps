#ifndef INFINI_OPS_OPERATOR_H_
#define INFINI_OPS_OPERATOR_H_

#include <cassert>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

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

class OperatorBase {
 public:
  virtual ~OperatorBase() = default;

  virtual std::size_t workspace_size_in_bytes() const { return 0; }

  void set_handle(const Handle& handle) { handle_ = handle; }

  void set_stream(void* stream) { stream_ = stream; }

  void set_workspace(void* workspace) { workspace_ = workspace; }

  void set_workspace_size_in_bytes(std::size_t workspace_size_in_bytes) {
    workspace_size_in_bytes_ = workspace_size_in_bytes;
  }

 protected:
  Handle handle_;

  void* stream_{nullptr};

  void* workspace_{nullptr};

  std::size_t workspace_size_in_bytes_{0};
};

template <typename Key, Device::Type device_type = Device::Type::kCount>
class Operator : public OperatorBase {
 public:
  template <typename... Args>
  static auto make(const Tensor tensor, Args&&... args) {
    std::unique_ptr<Operator> op_ptr;

    DispatchFunc<ActiveDevices>(
        tensor.device().type(),
        [&](auto tag) {
          constexpr Device::Type kDev = decltype(tag)::value;
          if constexpr (std::is_constructible_v<Operator<Key, kDev>,
                                                const Tensor&, Args...>) {
            op_ptr = std::make_unique<Operator<Key, kDev>>(
                tensor, std::forward<Args>(args)...);
          } else {
            assert(false && "operator is not implemented for this device");
          }
        },
        "Operator::make");

    return op_ptr;
  }

  template <typename... Args>
  static auto call(const Handle& handle, void* stream, void* workspace,
                   std::size_t workspace_size_in_bytes, Args&&... args) {
    static std::unordered_map<detail::CacheKey, std::unique_ptr<Operator>>
        cache;

    auto key = detail::CacheKey::Build(args...);

    auto it{cache.find(key)};

    if (it == cache.end()) {
      it = cache.emplace(std::move(key), make(std::forward<Args>(args)...))
               .first;
    }

    auto& op{it->second};

    auto resolved_stream{stream ? stream : handle.stream()};
    auto resolved_workspace{workspace ? workspace : handle.workspace()};
    auto resolved_workspace_size{workspace_size_in_bytes
                                     ? workspace_size_in_bytes
                                     : handle.workspace_size_in_bytes()};

    op->set_handle(handle);
    op->set_stream(resolved_stream);
    op->set_workspace(resolved_workspace);
    op->set_workspace_size_in_bytes(resolved_workspace_size);

    return (*op)(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static auto call(const Tensor tensor, Args&&... args) {
    return call({}, nullptr, nullptr, 0, tensor, std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto operator()(Args&&... args) const {
    return (*static_cast<const Key*>(this))(std::forward<Args>(args)...);
  }

 protected:
  static constexpr Device::Type device_type_{device_type};
};

}  // namespace infini::ops

#endif
