#ifndef INFINI_OPS_DISPATCHER_H_
#define INFINI_OPS_DISPATCHER_H_

#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

#include "common/traits.h"
#include "data_type.h"
#include "device.h"

namespace infini::ops {

// -----------------------------------------------------------------------------
// Core Generic Runtime Dispatchers
// -----------------------------------------------------------------------------

namespace detail {

// Implements the dispatch body over a resolved `List<head, tail...>`.
template <typename ValueType, typename Functor, typename... Args, auto head,
          auto... tail>
auto DispatchFuncImpl(ValueType value, Functor &&func,
                      std::string_view context_str, List<head, tail...>,
                      Args &&...args) {
  using ReturnType = decltype(std::forward<Functor>(func)(
      ValueTag<static_cast<ValueType>(head)>{}, std::forward<Args>(args)...));

  // Path for void functions.
  if constexpr (std::is_void_v<ReturnType>) {
    bool handled = ((value == static_cast<ValueType>(tail)
                         ? (std::forward<Functor>(func)(
                                ValueTag<tail>{}, std::forward<Args>(args)...),
                            true)
                         : false) ||
                    ... ||
                    (value == static_cast<ValueType>(head)
                         ? (std::forward<Functor>(func)(
                                ValueTag<head>{}, std::forward<Args>(args)...),
                            true)
                         : false));

    if (!handled) {
      std::cerr << "dispatch error (void): value " << static_cast<int>(value)
                << " not supported in the context: " << context_str << "\n";
      std::abort();
    }
  }
  // Path for non-void functions.
  else {
    std::optional<ReturnType> result;
    bool handled = ((value == static_cast<ValueType>(tail)
                         ? (result.emplace(std::forward<Functor>(func)(
                                ValueTag<tail>{}, std::forward<Args>(args)...)),
                            true)
                         : false) ||
                    ... ||
                    (value == static_cast<ValueType>(head)
                         ? (result.emplace(std::forward<Functor>(func)(
                                ValueTag<head>{}, std::forward<Args>(args)...)),
                            true)
                         : false));

    if (handled) {
      return *result;
    }
    std::cerr << "dispatch error (non-void): value " << static_cast<int>(value)
              << " not supported in the context: " << context_str << "\n";
    std::abort();
    return ReturnType{};
  }
}

// Deduces `head`/`tail` from a `List` type via partial specialization,
// then forwards to `DispatchFuncImpl`.
template <typename ValueType, typename Functor, typename FilteredList,
          typename ArgsTuple>
struct DispatchFuncUnwrap;

template <typename ValueType, typename Functor, auto head, auto... tail,
          typename... Args>
struct DispatchFuncUnwrap<ValueType, Functor, List<head, tail...>,
                          std::tuple<Args...>> {
  static auto call(ValueType value, Functor &&func,
                   std::string_view context_str, Args &&...args) {
    return DispatchFuncImpl(value, std::forward<Functor>(func), context_str,
                            List<head, tail...>{}, std::forward<Args>(args)...);
  }
};

// Empty-list specialization.
template <typename ValueType, typename Functor, typename... Args>
struct DispatchFuncUnwrap<ValueType, Functor, List<>, std::tuple<Args...>> {
  static auto call(ValueType value, Functor &&, std::string_view context_str,
                   Args &&...) {
    std::cerr << "dispatch error: no allowed values registered for value "
              << static_cast<int64_t>(value)
              << " in the context: " << context_str << "\n";
    std::abort();
  }
};

}  // namespace detail

// (Single Dispatch) Dispatches a runtime value to a compile-time functor.
template <typename ValueType, ValueType... all_values, typename Functor,
          typename... Args>
auto DispatchFunc(ValueType value, Functor &&func,
                  std::string_view context_str = "", Args &&...args) {
  using FilteredPack = typename Filter<Functor, std::tuple<Args...>, List<>,
                                       all_values...>::type;

  return detail::DispatchFuncUnwrap<
      ValueType, Functor, FilteredPack,
      std::tuple<Args...>>::call(value, std::forward<Functor>(func),
                                 context_str, std::forward<Args>(args)...);
}

// -----------------------------------------------------------------------------
// High-Level Specialized Dispatchers
// -----------------------------------------------------------------------------

namespace detail {

// Device-aware DataType adapter - maps (DataType, Device) to device-specific.
template <Device::Type device, typename Functor>
struct DeviceDataTypeAdapter {
  Functor &func;

  template <auto dtype, typename... Args>
  auto operator()(ValueTag<dtype>, Args &&...args) const {
    using T = typename TypeMap<static_cast<DataType>(dtype), device>::type;
    return func(TypeTag<T>{}, std::forward<Args>(args)...);
  }
};

// Helper to unpack List types for device-aware dispatch.
template <Device::Type device, typename Functor, typename... Args,
          auto... items>
auto DispatchFuncListAliasDeviceAwareImpl(DataType dtype, Functor &&func,
                                          std::string_view context_str,
                                          List<items...>, Args &&...args) {
  return DispatchFunc<device, items...>(dtype, std::forward<Functor>(func),
                                        context_str,
                                        std::forward<Args>(args)...);
}

}  // namespace detail

// Single-device, multiple-datatype Dispatch with explicit datatype list.
template <Device::Type device, DataType... allowed_dtypes, typename Functor,
          typename... Args>
auto DispatchFunc(DataType dtype, Functor &&func,
                  std::string_view context_str = "", Args &&...args) {
  detail::DeviceDataTypeAdapter<device, std::remove_reference_t<Functor>>
      adapter{func};
  return DispatchFunc<DataType, allowed_dtypes...>(dtype, adapter, context_str,
                                                   std::forward<Args>(args)...);
}

// Single-device, multiple-datatype Dispatch with List type (e.g., AllTypes).
template <Device::Type device, typename ListType, typename Functor,
          typename... Args,
          typename = std::enable_if_t<IsListType<ListType>::value>>
auto DispatchFunc(DataType dtype, Functor &&func,
                  std::string_view context_str = "", Args &&...args) {
  return detail::DispatchFuncListAliasDeviceAwareImpl<device>(
      dtype, std::forward<Functor>(func), context_str, ListType{},
      std::forward<Args>(args)...);
}

// -----------------------------------------------------------------------------
// Generic Dispatchers (no device context required)
// -----------------------------------------------------------------------------

// Generic Device::Type dispatcher - passes ValueTag with Device::Type value.
template <typename ListType, typename Functor, typename... Args,
          typename = std::enable_if_t<IsListType<ListType>::value>>
auto DispatchFunc(Device::Type device_type, Functor &&func,
                  std::string_view context_str = "", Args &&...args) {
  return detail::DispatchFuncUnwrap<
      Device::Type, std::remove_reference_t<Functor>, ListType,
      std::tuple<Args...>>::call(device_type, std::forward<Functor>(func),
                                 context_str, std::forward<Args>(args)...);
}

// Generic DataType dispatcher - converts ValueTag to TypeTag using CPU as
// default device.
template <typename ListType, typename Functor, typename... Args,
          typename = std::enable_if_t<IsListType<ListType>::value>>
auto DispatchFunc(DataType dtype, Functor &&func,
                  std::string_view context_str = "", Args &&...args) {
  detail::DeviceDataTypeAdapter<Device::Type::kCpu,
                                std::remove_reference_t<Functor>>
      adapter{func};
  return detail::DispatchFuncUnwrap<
      DataType, decltype(adapter), ListType,
      std::tuple<Args...>>::call(dtype, std::move(adapter), context_str,
                                 std::forward<Args>(args)...);
}

}  // namespace infini::ops

#endif
