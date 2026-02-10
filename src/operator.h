#ifndef INFINI_OPS_OPERATOR_H_
#define INFINI_OPS_OPERATOR_H_

#include <cassert>
#include <memory>

#include "tensor.h"

namespace infini::ops {

template <typename Key, Device::Type device = Device::Type::kCount>
class Operator {
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

    op_ptr->device_ = tensor.device();

    return op_ptr;
  }

  template <typename... Args>
  auto operator()(Args&&... args) const {
    switch (device_.type()) {
      case Device::Type::kNvidia:
        return (*static_cast<const Operator<Key, Device::Type::kNvidia>*>(
            this))(std::forward<Args>(args)...);
    }

    assert(false && "`operator()` dispatching not implemented for this device");
  }

 private:
  Device device_;
};

}  // namespace infini::ops

#endif
