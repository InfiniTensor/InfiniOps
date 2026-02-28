#ifndef INFINI_OPS_COMMON_TRAITS_H_
#define INFINI_OPS_COMMON_TRAITS_H_

#include <tuple>
#include <type_traits>

namespace infini::ops {

// --------------------- Type/Device Lists ---------------------
// A generic container for a sequence of compile-time values.
template <auto... Items>
struct List {};

// ListGet<I>(List<Items...>{}) extracts the Ith value from a List tag.
template <std::size_t I, auto Head, auto... Tail>
constexpr auto ListGetImpl(List<Head, Tail...>) {
  if constexpr (I == 0)
    return Head;
  else
    return ListGetImpl<I - 1>(List<Tail...>{});
}

template <std::size_t I, auto... Items>
constexpr auto ListGet(List<Items...> l) {
  return ListGetImpl<I>(l);
}

template <typename... Ts>
struct TypePack {};

// -----------------------------------------------------------------------------
// Tags
// -----------------------------------------------------------------------------
// Tags are passed as regular function arguments to user functors instead of
// template parameters. This lets users write plain C++17 `[](auto tag)` lambdas
// rather than C++20 template lambdas (`[]<typename T>()`).
//
// TypeTag<T>:   carries a C++ type. Recover with `typename
// decltype(tag)::type`. ValueTag<V>:  carries a compile-time value. Recover
// with `decltype(tag)::value`.
template <typename T>
struct TypeTag {
  using type = T;
};

template <auto V>
struct ValueTag {
  using value_type = decltype(V);
  static constexpr auto value = V;
};

// -----------------------------------------------------------------------------
// List Queries
// -----------------------------------------------------------------------------

// Check at compile-time if a Value exists within a construct (e.g., List<>).
// Example: static_assert(ContainsValue<SupportedTiles, 32>);
template <typename T, auto Value>
struct Contains;

template <auto Value, auto... Items>
struct Contains<List<Items...>, Value>
    : std::disjunction<std::bool_constant<Value == Items>...> {};

template <typename T, auto Value>
inline constexpr bool ContainsValue = Contains<T, Value>::value;

// Check at compile-time if a type T is present in a variadic list of types Ts.
// Example: static_assert(IsTypeInList<T, float, int>);
template <typename T, typename... Ts>
inline constexpr bool IsTypeInList = (std::is_same_v<T, Ts> || ...);

// Trait to detect whether T is a List<...> specialization.
template <typename T>
struct IsListType : std::false_type {};

template <auto... Items>
struct IsListType<List<Items...>> : std::true_type {};

// -----------------------------------------------------------------------------
// List Operations
// -----------------------------------------------------------------------------

// Concatenates two List types into a single List.
// Example: ConcatType<List<1, 2>, List<3, 4>> is List<1, 2, 3, 4>.
template <typename L1, typename L2>
struct Concat;

template <auto... I1, auto... I2>
struct Concat<List<I1...>, List<I2...>> {
  using type = List<I1..., I2...>;
};

template <typename L1, typename L2>
using ConcatType = typename Concat<L1, L2>::type;

template <typename... Lists>
struct Flatten;

template <auto... Items>
struct Flatten<List<Items...>> {
  using type = List<Items...>;
};

template <typename L1, typename L2, typename... Rest>
struct Flatten<L1, L2, Rest...> {
  using type = typename Flatten<ConcatType<L1, L2>, Rest...>::type;
};

// -----------------------------------------------------------------------------
// Invocability Detection (SFINAE)
// -----------------------------------------------------------------------------

// Checks if a Functor can be called with a ValueTag<Value> and Args...
template <typename Functor, auto Value, typename = void, typename... Args>
struct IsInvocable : std::false_type {};

template <typename Functor, auto Value, typename... Args>
struct IsInvocable<Functor, Value,
                   std::void_t<decltype(std::declval<Functor>()(
                       ValueTag<Value>{}, std::declval<Args>()...))>,
                   Args...> : std::true_type {};

template <typename Functor, auto Value, typename... Args>
inline constexpr bool IsInvocableValue =
    IsInvocable<Functor, Value, void, Args...>::value;

// -----------------------------------------------------------------------------
// Filtering Logic
// -----------------------------------------------------------------------------

// Recursive template to filter values based on Functor support at compile-time.
template <typename Functor, typename ArgsTuple, typename Result,
          auto... Remaining>
struct Filter;

// Base case: All values processed.
template <typename Functor, typename... Args, auto... Filtered>
struct Filter<Functor, std::tuple<Args...>, List<Filtered...>> {
  using type = List<Filtered...>;
};

// Recursive step: Test the 'Head' value and accumulate if supported.
template <typename Functor, typename... Args, auto... Filtered, auto Head,
          auto... Tail>
struct Filter<Functor, std::tuple<Args...>, List<Filtered...>, Head, Tail...> {
  using type = typename std::conditional_t<
      IsInvocableValue<Functor, Head, Args...> &&
          !ContainsValue<List<Filtered...>, Head>,
      Filter<Functor, std::tuple<Args...>, List<Filtered..., Head>, Tail...>,
      Filter<Functor, std::tuple<Args...>, List<Filtered...>, Tail...>>::type;
};

// Interface to filter a List type directly.
template <typename Functor, typename ArgsTuple, typename ListType>
struct FilterList;

template <typename Functor, typename... Args, auto... Items>
struct FilterList<Functor, std::tuple<Args...>, List<Items...>> {
  using type =
      typename Filter<Functor, std::tuple<Args...>, List<>, Items...>::type;
};

}  // namespace infini::ops

#endif
