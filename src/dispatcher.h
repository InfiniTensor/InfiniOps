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

// (Single Dispatch) Dispatches a runtime value to a compile-time functor.
template <typename ValueType, ValueType... all_values, typename Functor,
          typename... Args>
auto DispatchFunc(ValueType value, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  using FilteredPack =
      typename Filter<Functor, std::tuple<Args...>, List<>, all_values...>::type;

  return [&]<auto head, auto... tail>(List<head, tail...>) {
    using ReturnType =
        decltype(std::forward<Functor>(func)
                     .template operator()<static_cast<ValueType>(head)>(
                         std::forward<Args>(args)...));

    // Path for Void Functions
    if constexpr (std::is_void_v<ReturnType>) {
      bool handled =
          ((value == static_cast<ValueType>(tail)
                ? (std::forward<Functor>(func).template operator()<tail>(
                       std::forward<Args>(args)...),
                   true)
                : false) ||
           ... ||
           (value == static_cast<ValueType>(head)
                ? (std::forward<Functor>(func).template operator()<head>(
                       std::forward<Args>(args)...),
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
      bool handled =
          ((value == static_cast<ValueType>(tail)
                ? (result.emplace(
                       std::forward<Functor>(func).template operator()<tail>(
                           std::forward<Args>(args)...)),
                   true)
                : false) ||
           ... ||
           (value == static_cast<ValueType>(head)
                ? (result.emplace(
                       std::forward<Functor>(func).template operator()<head>(
                           std::forward<Args>(args)...)),
                   true)
                : false));

      if (handled) {
        return *result;
      }
      // TODO(lzm): change to logging.
      std::cerr << "Dispatch error (non-void): Value "
                << static_cast<int>(value)
                << " not supported in context: " << context_str << "\n";
      std::abort();
      return ReturnType{};
    }
  }(FilteredPack{});
}

// (Multi-Dispatch) Dispatches a vector of runtime values to a compile-time
// functor.
// Base Case: All dimensions resolved.
template <typename Functor, typename... Args, auto... is>
auto DispatchFunc(const std::vector<int64_t>& values, size_t index,
                  Functor&& func, std::string_view context_str, List<is...>,
                  Args&&... args) {
  return std::forward<Functor>(func).template operator()<is...>(
      std::forward<Args>(args)...);
}

// (Multi-Dispatch) Recursive Case
template <typename FirstList, typename... RestLists, typename Functor,
          typename... Args, auto... is>
auto DispatchFunc(const std::vector<int64_t>& values, size_t index,
                  Functor&& func, std::string_view context_str, List<is...>,
                  Args&&... args) {
  return [&]<auto... allowed>(List<allowed...>) {
    static_assert(sizeof...(allowed) > 0,
                  "`DispatchFunc` dimension list is empty");
    using EnumType = std::common_type_t<decltype(allowed)...>;

    return DispatchFunc<EnumType, allowed...>(
        static_cast<EnumType>(values.at(index)),
        [&]<EnumType val>(Args&&... inner_args) {
          return DispatchFunc<RestLists...>(
              values, index + 1, std::forward<Functor>(func), context_str,
              List<is..., val>{}, std::forward<Args>(inner_args)...);
        },
        context_str, std::forward<Args>(args)...);
  }(FirstList{});
}

// -----------------------------------------------------------------------------
// High-Level Specialized Dispatchers
// -----------------------------------------------------------------------------
// These provide cleaner and more convenient APIs for common InfiniOps types.

// DataType Dispatch
template <DataType... allowed_dtypes, typename Functor, typename... Args>
auto DispatchFunc(DataType dtype, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  return DispatchFunc<DataType, allowed_dtypes...>(
      dtype,
      [&]<DataType dt>(Args&&... inner_args) {
        using T = TypeMapType<dt>;
        return std::forward<Functor>(func).template operator()<T>(
            std::forward<Args>(inner_args)...);
      },
      context_str, std::forward<Args>(args)...);
}

// DataType Multi-Dispatch
template <typename... Lists, typename Functor, typename... Args>
auto DispatchFunc(std::initializer_list<DataType> dtypes, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  std::vector<int64_t> v;
  for (auto d : dtypes) v.push_back(static_cast<int64_t>(d));

  return DispatchFunc<Lists...>(
      v, 0,
      [&func]<DataType... dts>(Args&&... inner_args) {
        return std::forward<Functor>(func).template
        operator()<TypeMapType<dts>...>(std::forward<Args>(inner_args)...);
      },
      context_str, List<>{}, std::forward<Args>(args)...);
}

// Device Dispatch
template <Device::Type... allowed_devices, typename Functor, typename... Args>
auto DispatchFunc(Device::Type device, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  return DispatchFunc<Device::Type, allowed_devices...>(
      device,
      [&]<Device::Type dev>(Args&&... inner_args) {
        return std::forward<Functor>(func).template operator()<dev>(
            std::forward<Args>(inner_args)...);
      },
      context_str, std::forward<Args>(args)...);
}

// Device Multi-Dispatch
template <typename... Lists, typename Functor, typename... Args>
auto DispatchFunc(std::initializer_list<Device::Type> devices, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  std::vector<int64_t> v;
  for (auto d : devices) v.push_back(static_cast<int64_t>(d));

  return DispatchFunc<Lists...>(
      v, 0,
      [&func]<Device::Type... ds>(Args&&... inner_args) {
        return std::forward<Functor>(func).template operator()<ds...>(
            std::forward<Args>(inner_args)...);
      },
      context_str, List<>{}, std::forward<Args>(args)...);
}

// Interface for generic List Aliases, which unpacks a list.
template <typename ListType, typename ValueType, typename Functor,
          typename... Args>
auto DispatchFunc(ValueType value, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  return [&]<auto... is>(List<is...>) {
    return DispatchFunc<static_cast<std::decay_t<decltype(value)>>(is)...>(
        value, std::forward<Functor>(func), context_str,
        std::forward<Args>(args)...);
  }(ListType{});
}

}  // namespace infini::ops

#endif
