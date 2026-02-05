#ifndef INFINI_OPS_OPERATOR_H_
#define INFINI_OPS_OPERATOR_H_

#include <cassert>
#include <memory>

#include "handle.h"

namespace infini::ops {

template <typename Key, Device device = Device::kCount>
class Operator {
 public:
  template <typename... Args>
  static auto make(const Handle& handle, Args&&... args) {
    std::unique_ptr<Operator> op_ptr;

    switch (handle.device()) {
      case Device::kNvidia:
        op_ptr = std::make_unique<Operator<Key, Device::kNvidia>>(
            std::forward<Args>(args)...);
        break;
      default:
        assert(false &&
               "constructor dispatching not implemented for this device");
    }

    op_ptr->device_ = handle.device();

    return op_ptr;
  }

  template <typename... Args>
  auto operator()(Args&&... args) const {
    switch (device_) {
      case Device::kNvidia:
        return (*static_cast<const Operator<Key, Device::kNvidia>*>(this))(
            std::forward<Args>(args)...);
    }

    assert(false && "`operator()` dispatching not implemented for this device");
  }

 private:
  Device device_;
};

}  // namespace infini::ops

#endif
