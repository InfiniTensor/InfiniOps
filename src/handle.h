#ifndef INFINI_OPS_HANDLE_H_
#define INFINI_OPS_HANDLE_H_

#include "device.h"

namespace infini::ops {

class Handle {
 public:
  Handle(Device device) : device_{device} {}

  const Device& device() const { return device_; }

 private:
  Device device_;
};

}  // namespace infini::ops

#endif
