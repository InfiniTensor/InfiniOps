#ifndef INFINI_OPS_DRIVER_H_
#define INFINI_OPS_DRIVER_H_

#include <infini/rt.h>

namespace infini::ops {

template <Device::Type device_type>
using Driver = infini::rt::driver::Driver<device_type>;

template <typename Derived>
using DeviceDriver = infini::rt::driver::DeviceDriver<Derived>;

}  // namespace infini::ops

#endif
