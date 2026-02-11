#ifndef INFINI_OPS_OPERATOR_H_
#define INFINI_OPS_OPERATOR_H_

#include <cassert>
#include <memory>

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

    switch (tensor.device().type()) {
      case Device::Type::kNvidia:
        op_ptr = std::make_unique<Operator<Key, Device::Type::kNvidia>>(
            tensor, std::forward<Args>(args)...);
        break;
      default:
        assert(false &&
               "constructor dispatching not implemented for this device");
    }

    return op_ptr;
  }

  template <typename... Args>
  static auto call(void* stream, Args&&... args) {
    // TODO: Cache the created `Operator`.
    return (*make(std::forward<Args>(args)...))(stream,
                                                std::forward<Args>(args)...);
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
