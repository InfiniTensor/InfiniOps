#ifndef INFINI_OPS_DISPATCHER_H_
#define INFINI_OPS_DISPATCHER_H_

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
template <typename ValueType, ValueType... AllValues, typename Functor,
          typename... Args>
auto DispatchFunc(ValueType value, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  using FilteredPack =
      typename Filter<Functor, std::tuple<Args...>, List<>, AllValues...>::type;

  return [&]<auto... Pruned>(List<Pruned...>) {
    using ReturnType =
        decltype(std::forward<Functor>(func)
                     .template operator()<static_cast<ValueType>(0)>(
                         std::forward<Args>(args)...));

    bool handled = false;

    if constexpr (std::is_void_v<ReturnType>) {
      handled =
          ((value == static_cast<ValueType>(Pruned)
                ? (std::forward<Functor>(func).template operator()<Pruned>(
                       std::forward<Args>(args)...),
                   true)
                : false) ||
           ...);
    } else {
      std::optional<ReturnType> result;
      handled =
          ((value == static_cast<ValueType>(Pruned)
                ? (result.emplace(
                       std::forward<Functor>(func).template operator()<Pruned>(
                           std::forward<Args>(args)...)),
                   true)
                : false) ||
           ...);
      return *result;
    }
    if (!handled) {
      // TODO(lzm): change to logging
      std::cerr << "Dispatch error: Value " << static_cast<int>(value)
                << " not supported in the context: " << context_str << "\n";
      std::abort();
    }
  }(FilteredPack{});
}

// (Multi-Dispatch) Dispatches a vector of runtime values to a compile-time
// functor.
// Base Case: All dimensions resolved
template <typename Functor, typename... Args, auto... Is>
auto DispatchFunc(const std::vector<int64_t>& values, size_t index,
                  Functor&& func, std::string_view context_str, List<Is...>,
                  Args&&... args) {
  return std::forward<Functor>(func).template operator()<Is...>(
      std::forward<Args>(args)...);
}

// (Multi-Dispatch) Recursive Case
template <typename FirstList, typename... RestLists, typename Functor,
          typename... Args, auto... Is>
auto DispatchFunc(const std::vector<int64_t>& values, size_t index,
                  Functor&& func, std::string_view context_str, List<Is...>,
                  Args&&... args) {
  return [&]<auto... Allowed>(List<Allowed...>) {
    static_assert(sizeof...(Allowed) > 0,
                  "DispatchFunc dimension list is empty!");
    using EnumType = std::common_type_t<decltype(Allowed)...>;

    return DispatchFunc<EnumType, Allowed...>(
        static_cast<EnumType>(values.at(index)),
        [&]<EnumType Val>(Args&&... inner_args) {
          return DispatchFunc<RestLists...>(
              values, index + 1, std::forward<Functor>(func), context_str,
              List<Is..., Val>{}, std::forward<Args>(inner_args)...);
        },
        context_str, std::forward<Args>(args)...);
  }(FirstList{});
}

// -----------------------------------------------------------------------------
// High-Level Specialized Dispatchers
// -----------------------------------------------------------------------------
// These provide cleaner and more convenient APIs for common InfiniOps types.

// DataType Dispatch
template <DataType... AllowedDTypes, typename Functor, typename... Args>
auto DispatchFunc(DataType dtype, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  return DispatchFunc<DataType, AllowedDTypes...>(
      dtype,
      [&]<DataType DT>(Args&&... inner_args) {
        using T = TypeMap_t<DT>;
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
      [&func]<DataType... DTs>(Args&&... inner_args) {
        return std::forward<Functor>(func).template
        operator()<TypeMap_t<DTs>...>(std::forward<Args>(inner_args)...);
      },
      context_str, List<>{}, std::forward<Args>(args)...);
}

// Device Dispatch
template <Device::Type... AllowedDevices, typename Functor, typename... Args>
auto DispatchFunc(Device::Type device, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  return DispatchFunc<Device::Type, AllowedDevices...>(
      device,
      [&]<Device::Type Dev>(Args&&... inner_args) {
        return std::forward<Functor>(func).template operator()<Dev>(
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
      [&func]<Device::Type... Ds>(Args&&... inner_args) {
        return std::forward<Functor>(func).template operator()<Ds...>(
            std::forward<Args>(inner_args)...);
      },
      context_str, List<>{}, std::forward<Args>(args)...);
}

// Interface for generic List Aliases, which unpacks a list
template <typename ListType, typename ValueType, typename Functor,
          typename... Args>
auto DispatchFunc(ValueType value, Functor&& func,
                  std::string_view context_str = "", Args&&... args) {
  return [&]<auto... Is>(List<Is...>) {
    return DispatchFunc<static_cast<std::decay_t<decltype(value)>>(Is)...>(
        value, std::forward<Functor>(func), context_str,
        std::forward<Args>(args)...);
  }(ListType{});
}

}  // namespace infini::ops

#endif
