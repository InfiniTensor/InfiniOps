#ifndef INFINI_OPS_COMMON_TRAITS_H_
#define INFINI_OPS_COMMON_TRAITS_H_

#include <tuple>
#include <type_traits>

namespace infini::ops {

// A generic container for a sequence of compile-time values.
template <auto... Items>
struct List {};

// -----------------------------------------------------------------------------
// List Queries
// -----------------------------------------------------------------------------

// Check at compile-time if a Value exists within a construct (e.g., List<>).
// Example: static_assert(Contains_v<SupportedTiles, 32>);
template <typename T, auto Value>
struct Contains;

template <auto Value, auto... Items>
struct Contains<List<Items...>, Value>
    : std::disjunction<std::bool_constant<Value == Items>...> {};

template <typename T, auto Value>
inline constexpr bool Contains_v = Contains<T, Value>::value;

// Check at compile-time if a type T is present in a variadic list of types Ts.
// Example: static_assert(IsTypeInList<T, float, int>);
template <typename T, typename... Ts>
inline constexpr bool IsTypeInList = (std::is_same_v<T, Ts> || ...);

// -----------------------------------------------------------------------------
// List Operations
// -----------------------------------------------------------------------------

// Concatenates two List types into a single List.
// Example: Concat_t<List<1, 2>, List<3, 4>> is List<1, 2, 3, 4>.
template <typename L1, typename L2>
struct Concat;

template <auto... I1, auto... I2>
struct Concat<List<I1...>, List<I2...>> {
  using type = List<I1..., I2...>;
};

template <typename L1, typename L2>
using Concat_t = typename Concat<L1, L2>::type;

// -----------------------------------------------------------------------------
// Invocability Detection (SFINAE)
// -----------------------------------------------------------------------------

// Checks if a Functor's template operator()<Value> can be called with Args.
template <typename Functor, auto Value, typename = void, typename... Args>
struct IsInvocable : std::false_type {};

template <typename Functor, auto Value, typename... Args>
struct IsInvocable<
    Functor, Value,
    std::void_t<decltype(std::declval<Functor>().template operator()<Value>(
        std::declval<Args>()...))>,
    Args...> : std::true_type {};

template <typename Functor, auto Value, typename... Args>
inline constexpr bool IsInvocable_v =
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
      IsInvocable_v<Functor, Head, Args...> &&
          !Contains_v<List<Filtered...>, Head>,
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
