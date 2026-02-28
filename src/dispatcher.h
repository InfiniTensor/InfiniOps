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

namespace {

// Implements the dispatch body over a resolved List<Head, Tail...>.
template <typename ValueType, typename Functor, typename... Args, auto Head,
          auto... Tail>
auto DispatchFuncImpl(ValueType value, Functor &&func,
                      std::string_view context_str, List<Head, Tail...>,
                      Args &&...args) {
  using ReturnType = decltype(std::forward<Functor>(func)(
      ValueTag<static_cast<ValueType>(Head)>{}, std::forward<Args>(args)...));

  // Path for Void Functions
  if constexpr (std::is_void_v<ReturnType>) {
    bool handled = ((value == static_cast<ValueType>(Tail)
                         ? (std::forward<Functor>(func)(
                                ValueTag<Tail>{}, std::forward<Args>(args)...),
                            true)
                         : false) ||
                    ... ||
                    (value == static_cast<ValueType>(Head)
                         ? (std::forward<Functor>(func)(
                                ValueTag<Head>{}, std::forward<Args>(args)...),
                            true)
                         : false));

    if (!handled) {
      std::cerr << "Dispatch error (void): Value " << static_cast<int>(value)
                << " not supported in context: " << context_str << "\n";
      std::abort();
    }
  }
  // Path for Non-Void Functions
  else {
    std::optional<ReturnType> result;
    bool handled = ((value == static_cast<ValueType>(Tail)
                         ? (result.emplace(std::forward<Functor>(func)(
                                ValueTag<Tail>{}, std::forward<Args>(args)...)),
                            true)
                         : false) ||
                    ... ||
                    (value == static_cast<ValueType>(Head)
                         ? (result.emplace(std::forward<Functor>(func)(
                                ValueTag<Head>{}, std::forward<Args>(args)...)),
                            true)
                         : false));

    if (handled) {
      return *result;
    }
    // TODO(lzm): change to logging.
    std::cerr << "Dispatch error (non-void): Value " << static_cast<int>(value)
              << " not supported in context: " << context_str << "\n";
    std::abort();
    return ReturnType{};
  }
}

// Deduces Head/Tail from a List type via partial specialization,
// then forwards to DispatchFuncImpl.
template <typename ValueType, typename Functor, typename FilteredList,
          typename ArgsTuple>
struct DispatchFuncUnwrap;

template <typename ValueType, typename Functor, auto Head, auto... Tail,
          typename... Args>
struct DispatchFuncUnwrap<ValueType, Functor, List<Head, Tail...>,
                          std::tuple<Args...>> {
  static auto call(ValueType value, Functor &&func,
                   std::string_view context_str, Args &&...args) {
    return DispatchFuncImpl(value, std::forward<Functor>(func), context_str,
                            List<Head, Tail...>{}, std::forward<Args>(args)...);
  }
};

// Empty-list specialization
template <typename ValueType, typename Functor, typename... Args>
struct DispatchFuncUnwrap<ValueType, Functor, List<>, std::tuple<Args...>> {
  static auto call(ValueType value, Functor &&, std::string_view context_str,
                   Args &&...) {
    // TODO(lzm): change to logging.
    std::cerr << "Dispatch error: no allowed values registered for value "
              << static_cast<int64_t>(value) << " in context: " << context_str
              << "\n";
    std::abort();
  }
};

}  // namespace

// (Single Dispatch) Dispatches a runtime value to a compile-time functor.
template <typename ValueType, ValueType... AllValues, typename Functor,
          typename... Args>
auto DispatchFunc(ValueType value, Functor &&func,
                  std::string_view context_str = "", Args &&...args) {
  using FilteredPack =
      typename Filter<Functor, std::tuple<Args...>, List<>, AllValues...>::type;

  return DispatchFuncUnwrap<ValueType, Functor, FilteredPack,
                            std::tuple<Args...>>::call(value,
                                                       std::forward<Functor>(
                                                           func),
                                                       context_str,
                                                       std::forward<Args>(
                                                           args)...);
}

// (Multi-Dispatch) Dispatches a vector of runtime values to a compile-time
// functor.
// Base Case: All dimensions resolved
template <typename Functor, typename... Args, auto... Is>
auto DispatchFunc(const std::vector<int64_t> &values, size_t /*index*/,
                  Functor &&func, std::string_view /*context_str*/, List<Is...>,
                  Args &&...args) {
  return std::forward<Functor>(func)(List<Is...>{},
                                     std::forward<Args>(args)...);
}

// Forward declaration of the recursive multi-dispatch overload.
template <typename FirstList, typename... RestLists, typename Functor,
          typename... Args, auto... Is>
auto DispatchFunc(const std::vector<int64_t> &values, size_t index,
                  Functor &&func, std::string_view context_str, List<Is...>,
                  Args &&...args);

// Adapter used in the recursive multi-dispatch case: given a resolved value Val
// recurse into the next dimension.
template <typename RestListsPack, typename Functor, auto... Is>
struct MultiDispatchRecurseAdapter;

template <typename... RestLists, typename Functor, auto... Is>
struct MultiDispatchRecurseAdapter<TypePack<RestLists...>, Functor, Is...> {
  const std::vector<int64_t> &values;
  size_t next_index;
  Functor &func;
  std::string_view context_str;

  template <auto Val, typename... Args>
  auto operator()(ValueTag<Val>, Args &&...args) const {
    return DispatchFunc<RestLists...>(values, next_index, func, context_str,
                                      List<Is..., Val>{},
                                      std::forward<Args>(args)...);
  }
};

template <typename RestListsPack, typename Functor, typename... Args,
          auto... Is, auto... Allowed>
auto MultiDispatchFirstDim(const std::vector<int64_t> &values, size_t index,
                           Functor &func, std::string_view context_str,
                           List<Is...>, List<Allowed...>, Args &&...args) {
  static_assert(sizeof...(Allowed) > 0,
                "`DispatchFunc` dimension list is empty");
  using EnumType = std::common_type_t<decltype(Allowed)...>;

  MultiDispatchRecurseAdapter<RestListsPack, Functor, Is...> adapter{
      values, index + 1, func, context_str};

  return DispatchFunc<EnumType, Allowed...>(
      static_cast<EnumType>(values.at(index)), adapter, context_str,
      std::forward<Args>(args)...);
}

// (Multi-Dispatch) Recursive Case
template <typename FirstList, typename... RestLists, typename Functor,
          typename... Args, auto... Is>
auto DispatchFunc(const std::vector<int64_t> &values, size_t index,
                  Functor &&func, std::string_view context_str, List<Is...>,
                  Args &&...args) {
  return MultiDispatchFirstDim<TypePack<RestLists...>>(
      values, index, func, context_str, List<Is...>{}, FirstList{},
      std::forward<Args>(args)...);
}

// -----------------------------------------------------------------------------
// High-Level Specialized Dispatchers
// -----------------------------------------------------------------------------
// These provide cleaner and more convenient APIs for common InfiniOps types.

namespace {

// Bridges the generic value dispatch layer to the DataType-specific type
// dispatch layer.
template <typename Functor>
struct DataTypeAdapter {
  Functor &func;

  template <auto DT, typename... Args>
  auto operator()(ValueTag<DT>, Args &&...args) const {
    using T = TypeMapType<static_cast<DataType>(DT)>;
    return func(TypeTag<T>{}, std::forward<Args>(args)...);
  }
};

template <typename Functor>
struct DataTypeMultiAdapter {
  Functor &func;

  template <auto... DTs, typename... Args>
  auto operator()(List<DTs...>, Args &&...args) const {
    return func(TypeTag<TypeMapType<static_cast<DataType>(DTs)>>{}...,
                std::forward<Args>(args)...);
  }
};

template <typename Functor>
struct DeviceAdapter {
  Functor &func;

  template <auto Dev, typename... Args>
  auto operator()(ValueTag<Dev>, Args &&...args) const {
    return func(ValueTag<Dev>{}, std::forward<Args>(args)...);
  }
};

template <typename Functor>
struct DeviceMultiAdapter {
  Functor &func;

  template <auto... Ds, typename... Args>
  auto operator()(List<Ds...>, Args &&...args) const {
    return func(ValueTag<Ds>{}..., std::forward<Args>(args)...);
  }
};

}  // namespace

// DataType Dispatch
template <DataType... allowed_dtypes, typename Functor, typename... Args>
auto DispatchFunc(DataType dtype, Functor &&func,
                  std::string_view context_str = "", Args &&...args) {
  DataTypeAdapter<std::remove_reference_t<Functor>> adapter{func};
  return DispatchFunc<DataType, allowed_dtypes...>(dtype, adapter, context_str,
                                                   std::forward<Args>(args)...);
}

// DataType Multi-Dispatch
template <typename... Lists, typename Functor, typename... Args>
auto DispatchFunc(std::initializer_list<DataType> dtypes, Functor &&func,
                  std::string_view context_str = "", Args &&...args) {
  std::vector<int64_t> v;
  for (auto d : dtypes) v.push_back(static_cast<int64_t>(d));

  DataTypeMultiAdapter<std::remove_reference_t<Functor>> adapter{func};
  return DispatchFunc<Lists...>(v, 0, adapter, context_str, List<>{},
                                std::forward<Args>(args)...);
}

// Device Dispatch
template <auto... AllowedDevices, typename Functor, typename... Args>
auto DispatchFunc(Device::Type device, Functor &&func,
                  std::string_view context_str = "", Args &&...args) {
  DeviceAdapter<std::remove_reference_t<Functor>> adapter{func};
  return DispatchFunc<Device::Type,
                      static_cast<Device::Type>(AllowedDevices)...>(
      device, adapter, context_str, std::forward<Args>(args)...);
}

// Device Multi-Dispatch
template <typename... Lists, typename Functor, typename... Args>
auto DispatchFunc(std::initializer_list<Device::Type> devices, Functor &&func,
                  std::string_view context_str = "", Args &&...args) {
  std::vector<int64_t> v;
  for (auto d : devices) v.push_back(static_cast<int64_t>(d));

  DeviceMultiAdapter<std::remove_reference_t<Functor>> adapter{func};
  return DispatchFunc<Lists...>(v, 0, adapter, context_str, List<>{},
                                std::forward<Args>(args)...);
}

template <typename ValueType, typename Functor, typename... Args, auto... Is>
auto DispatchFuncListAliasImpl(ValueType value, Functor &&func,
                               std::string_view context_str, List<Is...>,
                               Args &&...args) {
  return DispatchFunc<static_cast<std::decay_t<ValueType>>(Is)...>(
      value, std::forward<Functor>(func), context_str,
      std::forward<Args>(args)...);
}

// Interface for generic List Aliases
template <typename ListType, typename ValueType, typename Functor,
          typename... Args,
          typename = std::enable_if_t<IsListType<ListType>::value>>
auto DispatchFunc(ValueType value, Functor &&func,
                  std::string_view context_str = "", Args &&...args) {
  return DispatchFuncListAliasImpl(value, std::forward<Functor>(func),
                                   context_str, ListType{},
                                   std::forward<Args>(args)...);
}

}  // namespace infini::ops

#endif
