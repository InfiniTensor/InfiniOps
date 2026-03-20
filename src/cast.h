#ifndef INFINI_OPS_CAST_H_
#define INFINI_OPS_CAST_H_

#include "data_type.h"
#include "device.h"

namespace infini::ops {

namespace detail {

template <Device::Type device_type, typename Dst, typename Src>
struct CastHelper {
  Dst operator()(const Src& x) const { return static_cast<Dst>(x); };
};

}  // namespace detail

template <Device::Type device_type, typename Dst, typename Src>
Dst Cast(Src&& x) {
  return detail::CastHelper<device_type, Dst, Src>{}(x);
}

}  // namespace infini::ops

#endif
