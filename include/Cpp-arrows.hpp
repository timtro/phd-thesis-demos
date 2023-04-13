// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65

#pragma once

#include <experimental/type_traits>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace tf {
  // Tools for categorical notation of arrows ............. f[[[1
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
    template <typename T>
    struct function_traits;

    template <typename T>
    struct function_traits
        : public function_traits<decltype(&T::operator())> {};

    template <typename Ret, typename... Args>
    struct function_traits<Ret (*)(Args...)> {
      using return_type = Ret;
      using arg_types = std::tuple<Args...>;
    };

    template <typename Ret, typename T, typename... Args>
    struct function_traits<Ret (T::*)(Args...)> {
      using return_type = Ret;
      using arg_types = std::tuple<Args...>;
      using struct_type = T;
    };

    template <typename Ret, typename T, typename... Args>
    struct function_traits<Ret (T::*)(Args...) const> {
      using return_type = Ret;
      using arg_types = std::tuple<Args...>;
      using struct_type = T;
    };

    template <typename Ret, typename T, typename... Args>
    struct function_traits<Ret (T::*)(Args...) volatile> {
      using return_type = Ret;
      using arg_types = std::tuple<Args...>;
      using struct_type = T;
    };

    template <typename Ret, typename T, typename... Args>
    struct function_traits<Ret (T::*)(Args...) const volatile> {
      using return_type = Ret;
      using arg_types = std::tuple<Args...>;
      using struct_type = T;
    };

    template <typename Ret, typename... Args>
    struct function_traits<std::function<Ret(Args...)>> {
      using return_type = Ret;
      using arg_types = std::tuple<Args...>;
    };

  } // namespace impl

  template <typename F>
  using Dom = typename std::tuple_element_t<0,
      typename impl::function_traits<F>::arg_types>;

  template <typename F>
  using Cod = typename impl::function_traits<F>::return_type;

  // ...................................................... f]]]1
  // Identity function .................................... f[[[1
  template <typename T>
  constexpr decltype(auto) id(T &&x) {
    return std::forward<T>(x);
  }
  // ...................................................... f]]]1
  // Arrow composition .................................... f[[[1

  template <typename Fn>
  constexpr auto compose(Fn fn) {
    return [fn](Dom<Fn> x) -> Cod<Fn> { return fn(x); };
  }

  template <typename Fn, typename Gn>
  constexpr auto compose(Fn fn, Gn gn) {
    return [fn, gn](Dom<Gn> x) -> Cod<Fn> { return fn(gn(x)); };
  }

  template <typename Fn, typename Gn, typename Hn>
  constexpr auto compose(Fn fn, Gn gn, Hn hn) {
    return [fn, gn, hn](Dom<Hn> x) -> Cod<Fn> {
      return fn(gn(hn(x)));
    };
  }

  template <typename Fn, typename Gn, typename Hn, typename In>
  constexpr auto compose(Fn fn, Gn gn, Hn hn, In in) {
    return [fn, gn, hn, in](Dom<In> x) -> Cod<Fn> {
      return fn(gn(hn(in(x))));
    };
  }

  // template <typename F, typename... Fs>
  // constexpr decltype(auto) compose(F f, Fs... fs) {
  //   if constexpr (sizeof...(fs) < 1)
  //     return [f](Dom<F> &&x) -> Cod<F> {
  //       return std::invoke(f, std::forward<decltype(x)>(x));
  //     };
  //   else
  //     return [f, fs...](Dom<F> &&x) -> Cod<F> {
  //       return std::invoke(
  //           f, compose(fs...)(std::forward<decltype(x)>(x)));
  //     };
  // }
  // ...................................................... f]]]1
  // Currying ............................................. f[[[1

  template <typename F>
  constexpr auto curry(F f) {
    if constexpr (std::is_invocable_v<F>) {
      return std::invoke(f);
    } else if constexpr (
        std::tuple_size_v<
            typename impl::function_traits<F>::arg_types> == 1) {
      return [f](Dom<F> x) -> Cod<F> { return f(x); };
    } else if constexpr (
        std::tuple_size_v<
            typename impl::function_traits<F>::arg_types> == 2) {

      using Dm0 = std::tuple_element_t<0,
          typename impl::function_traits<F>::arg_types>;
      using Dm1 = std::tuple_element_t<1,
          typename impl::function_traits<F>::arg_types>;

      return [fn = f](Dm0 d0) -> Hom<Dm1, Cod<F>> {
        return [&fn, d0](Dm1 d1) {
          return std::invoke(fn, d0, d1);
        };
      };
    } else if constexpr (
        std::tuple_size_v<
            typename impl::function_traits<F>::arg_types> == 3) {

      using Dm0 = std::tuple_element_t<0,
          typename impl::function_traits<F>::arg_types>;
      using Dm1 = std::tuple_element_t<1,
          typename impl::function_traits<F>::arg_types>;
      using Dm2 = std::tuple_element_t<2,
          typename impl::function_traits<F>::arg_types>;

      return [fn = f](Dm0 d0) -> Hom<Dm1, Hom<Dm2, Cod<F>>> {
        return [&fn, d0](Dm1 d1) -> Hom<Dm2, Cod<F>> {
          return [&fn, &d0, d1](Dm2 d2) -> Cod<F> {
            return std::invoke(fn, d0, d1, d2);
          };
        };
      };
    } else if constexpr (
        std::tuple_size_v<
            typename impl::function_traits<F>::arg_types> == 4) {

      using Dm0 = std::tuple_element_t<0,
          typename impl::function_traits<F>::arg_types>;
      using Dm1 = std::tuple_element_t<1,
          typename impl::function_traits<F>::arg_types>;
      using Dm2 = std::tuple_element_t<2,
          typename impl::function_traits<F>::arg_types>;
      using Dm3 = std::tuple_element_t<3,
          typename impl::function_traits<F>::arg_types>;

      return [fn = f](Dm0 d0)
                 -> Hom<Dm1, Hom<Dm2, Hom<Dm3, Cod<F>>>> {
        return [&fn, d0](Dm1 d1) -> Hom<Dm2, Hom<Dm3, Cod<F>>> {
          return [&fn, &d0, d1](Dm2 d2) -> Hom<Dm3, Cod<F>> {
            return [&fn, &d0, &d1, d2](Dm3 d3) -> Cod<F> {
              return std::invoke(fn, d0, d1, d2, d3);
            };
          };
        };
      };
    }
  }
  // ...................................................... f]]]1
} // namespace tf
