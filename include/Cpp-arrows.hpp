// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65

#pragma once

#include <experimental/type_traits>
#include <functional>
#include <type_traits>
#include <utility>

using std::experimental::is_detected_v;

namespace tf {
// Tools for categorical notation of arrows ............... f[[[1
  template <typename... Ts>
  struct Doms {};

  template <typename Dom, typename Cod>
  struct Hom : public std::function<Cod(Dom)> {
    using std::function<Cod(Dom)>::function;
  };

  template <typename Cod, typename... Ts>
  struct Hom<Doms<Ts...>, Cod>
      : public std::function<Cod(Ts...)> {
    using std::function<Cod(Ts...)>::function;
  };

  namespace impl {
    template <typename F, typename Ret, typename Arg1,
        typename... Rest>
    Arg1 function_signature_helper(Ret (F::*)(Arg1, Rest...));

    template <typename F, typename Ret, typename Arg1,
        typename... Rest>
    Arg1 function_signature_helper(
        Ret (F::*)(Arg1, Rest...) const);

    template <typename F>
    struct first_parameter {
      using type =
          decltype(function_signature_helper(&F::operator()));
    };
  } // namespace impl

  template <typename F>
  using Dom = typename impl::first_parameter<F>::type;

  template <typename F>
  using Cod = typename std::invoke_result_t<F, Dom<F>>;
// ........................................................ f]]]1
// Identity function ...................................... f[[[1
  template <typename T>
  constexpr decltype(auto) id(T &&x) {
    return std::forward<T>(x);
  }
// ........................................................ f]]]1
// Arrow composition ...................................... f[[[1
  template <typename F, typename... Fs>
  constexpr decltype(auto) compose(F f, Fs... fs) {
    if constexpr (sizeof...(fs) < 1)
      return [f](auto &&x) -> decltype(auto) {
        return std::invoke(f, std::forward<decltype(x)>(x));
      };
    else
      return [f, fs...](auto &&x) -> decltype(auto) {
        return std::invoke(
            f, compose(fs...)(std::forward<decltype(x)>(x)));
      };
  }
// ........................................................ f]]]1
// Currying ............................................... f[[[1
  template <typename F>
  constexpr decltype(auto) curry(F f) {
    if constexpr (std::is_invocable_v<F>)
      return std::invoke(f);
    else
      return [f](auto &&x) {
        return curry(
            // perfectly capture x here:
            [f, x](auto &&...xs)
                -> decltype(std::invoke(f, x, xs...)) {
              return std::invoke(f, x, xs...);
            });
      };
  }
// ........................................................ f]]]1
} // namespace tf
