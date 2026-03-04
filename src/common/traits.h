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

// Check at compile-time if a value exists within a construct (e.g., List<>).
// Example: static_assert(ContainsValue<SupportedTiles, 32>);
template <typename T, auto value>
struct Contains;

template <auto value, auto... items>
struct Contains<List<items...>, value>
    : std::disjunction<std::bool_constant<value == items>...> {};

template <typename T, auto value>
inline constexpr bool ContainsValue = Contains<T, value>::value;

// Check at compile-time if a type T is present in a variadic list of types Ts.
// Example: static_assert(IsTypeInList<T, float, int>);
template <typename T, typename... Ts>
inline constexpr bool IsTypeInList = (std::is_same_v<T, Ts> || ...);

// -----------------------------------------------------------------------------
// List Operations
// -----------------------------------------------------------------------------

// Concatenates two List types into a single List.
// Example: ConcatType<List<1, 2>, List<3, 4>> is List<1, 2, 3, 4>.
template <typename List1, typename List2>
struct Concat;

template <auto... i1, auto... i2>
struct Concat<List<i1...>, List<i2...>> {
  using type = List<i1..., i2...>;
};

template <typename List1, typename List2>
using ConcatType = typename Concat<List1, List2>::type;

// -----------------------------------------------------------------------------
// Invocability Detection (SFINAE)
// -----------------------------------------------------------------------------

// Checks if a Functor's template operator()<value> can be called with Args.
template <typename Functor, auto value, typename = void, typename... Args>
struct IsInvocable : std::false_type {};

template <typename Functor, auto value, typename... Args>
struct IsInvocable<
    Functor, value,
    std::void_t<decltype(std::declval<Functor>().template operator()<value>(
        std::declval<Args>()...))>,
    Args...> : std::true_type {};

template <typename Functor, auto value, typename... Args>
inline constexpr bool IsInvocableValue =
    IsInvocable<Functor, value, void, Args...>::value;

// -----------------------------------------------------------------------------
// Filtering Logic
// -----------------------------------------------------------------------------

// Recursive template to filter values based on Functor support at compile-time.
template <typename Functor, typename ArgsTuple, typename Result,
          auto... remaining>
struct Filter;

// Base case: All values processed.
template <typename Functor, typename... Args, auto... filtered>
struct Filter<Functor, std::tuple<Args...>, List<filtered...>> {
  using type = List<filtered...>;
};

// Recursive step: Test the head value and accumulate if supported.
template <typename Functor, typename... Args, auto... filtered, auto head,
          auto... tail>
struct Filter<Functor, std::tuple<Args...>, List<filtered...>, head, tail...> {
  using type = typename std::conditional_t<
      IsInvocableValue<Functor, head, Args...> &&
          !ContainsValue<List<filtered...>, head>,
      Filter<Functor, std::tuple<Args...>, List<filtered..., head>, tail...>,
      Filter<Functor, std::tuple<Args...>, List<filtered...>, tail...>>::type;
};

// Interface to filter a List type directly.
template <typename Functor, typename ArgsTuple, typename ListType>
struct FilterList;

template <typename Functor, typename... Args, auto... items>
struct FilterList<Functor, std::tuple<Args...>, List<items...>> {
  using type =
      typename Filter<Functor, std::tuple<Args...>, List<>, items...>::type;
};

}  // namespace infini::ops

#endif
