#ifndef INFINI_OPS_OPERATOR_H_
#define INFINI_OPS_OPERATOR_H_

#include <cassert>
#include <memory>
#include <type_traits>
#include <unordered_map>

#include "dispatcher.h"
#include "tensor.h"

namespace infini::ops {

class OperatorBase {
 public:
  virtual ~OperatorBase() = default;

  static void set_stream(void* stream) { stream_ = stream; }

 protected:
  inline static thread_local void* stream_{nullptr};
};

template <typename Key, Device::Type device = Device::Type::kCount>
class Operator : public OperatorBase {
 public:
  template <typename... Args>
  static auto make(const Tensor tensor, Args&&... args) {
    std::unique_ptr<Operator> op_ptr;

    DispatchFunc<ActiveDevices>(
        tensor.device().type(),
        [&](auto tag) {
          constexpr Device::Type dev = decltype(tag)::value;
          if constexpr (std::is_constructible_v<Operator<Key, dev>,
                                                const Tensor&, Args...>) {
            op_ptr = std::make_unique<Operator<Key, dev>>(
                tensor, std::forward<Args>(args)...);
          } else {
            assert(false && "operator is not implemented for this device");
          }
        },
        "Operator::make");

    return op_ptr;
  }

  template <typename... Args>
  static auto call(void* stream, Args&&... args) {
    static std::unordered_map<std::size_t, std::unique_ptr<Operator>> cache;

    std::size_t hash{0};

    (hash_combine(hash, args), ...);

    auto it{cache.find(hash)};

    if (it == cache.end()) {
      it = cache.emplace(hash, make(std::forward<Args>(args)...)).first;
    }

    return (*it->second)(stream, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static auto call(const Tensor tensor, Args&&... args) {
    return call(stream_, tensor, std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto operator()(void* stream, Args&&... args) const {
    return (*static_cast<const Key*>(this))(stream,
                                            std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto operator()(const Tensor tensor, Args&&... args) const {
    return operator()(stream_, tensor, std::forward<Args>(args)...);
  }
};

}  // namespace infini::ops

#endif
