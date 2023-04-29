// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65
#pragma once

#include <catch2/catch.hpp>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <variant>

#include <list>

#include "Cpp-arrows.hpp"

using tf::Cod;
using tf::compose;
using tf::curry;
using tf::Dom;
using tf::Doms;
using tf::Hom;

// For convenience with functions of std::function type,
//   wrap tf::id in a std::function.
template <typename T>
auto id = Hom<T, T>{tf::id<T>};

// Demos of structure in Cpp .............................. f[[[1
// Identity functor ....................................... f[[[2

template <typename T>
using IdType = T;

struct Id {

  template <typename T>
  using Of = IdType<T>;

  template <typename Fn>
  static auto fmap(Fn fn) -> Hom<Of<Dom<Fn>>, Of<Cod<Fn>>> {
    return fn;
  };
};

// ........................................................ f]]]2
// Optional functor ....................................... f[[[2

namespace Optional {
  template <typename T>
  using Of = std::optional<T>;

  template <typename Fn>
  static auto fmap(Fn fn) -> Hom<Of<Dom<Fn>>, Of<Cod<Fn>>> {
    using T = Dom<Fn>;
    using U = Cod<Fn>;
    return [fn](Of<T> ot) -> Of<U> {
      if (ot)
        return fn(ot.value());
      else
        return std::nullopt;
    };
  }
} // namespace Optional

// ........................................................ f]]]2
// std::vector based List-functor ......................... f[[[2

namespace Vector {
  template <typename T>
  using Of = std::vector<T>;

  template <typename Fn>
  auto fmap(Fn fn) -> Hom<Of<Dom<Fn>>, Of<Cod<Fn>>> {
    using T = Dom<Fn>;
    using U = Cod<Fn>;
    return [fn](Of<T> t_s) {
      Of<U> u_s;
      u_s.reserve(t_s.size());

      std::transform(
          cbegin(t_s), cend(t_s), std::back_inserter(u_s), fn);
      return u_s;
    };
  };
}; // namespace Vector

// ........................................................ f]]]2
// Constant functor ....................................... f[[[2

template <typename T>
struct Always {
  template <typename>
  using Given = T;
};

template <typename T>
struct Const {

  template <typename U>
  using Of = typename Always<T>::template Given<U>;

  template <typename Fn>
  static auto fmap(Fn) -> Hom<T, T> {
    return id<T>;
  };
};

// ........................................................ f]]]2
// shared_ptr functor ..................................... f[[[2

namespace sptr {
  template <typename T>
  using Of = std::shared_ptr<T>;

  template <typename Fn>
  auto map(Fn fn) -> Hom<Of<Dom<Fn>>, Of<Cod<Fn>>> {
    return [fn](Of<Dom<Fn>> x_ptr) -> Of<Cod<Fn>> {
      const auto result = fn(*x_ptr);
      using Result = std::remove_cv_t<decltype(result)>;

      return std::make_shared<Result>(result);
    };
  }
} // namespace sptr

// ........................................................ f]]]2
// Natural Transformations ................................ f[[[2

template <typename T>
auto len(Vector::Of<T> t_s) -> Const<std::size_t>::Of<T> {
  return t_s.size();
}

// ........................................................ f]]]2
// CCC in Cpp ............................................. f[[[2
// Categorical product bifunctor .......................... f[[[3

template <typename T, typename U>
struct P : std::pair<T, U> {
  using std::pair<T, U>::pair;
};

template <typename T, typename U>
P(T, U) -> P<T, U>;

namespace std {
  template <typename T, typename U>
  struct tuple_element<0, P<T, U>> {
    using type = T;
  };

  template <typename T, typename U>
  struct tuple_element<1, P<T, U>> {
    using type = U;
  };
} // namespace std

template <typename T, typename U>
auto proj_l(P<T, U> tu) -> T {
  return std::get<0>(tu);
}

template <typename T, typename U>
auto proj_r(P<T, U> tu) -> U {
  return std::get<1>(tu);
}

// clang-format off
template <typename Fn,          typename Gn,
          typename T = Dom<Fn>, typename U = Dom<Gn>,
          typename X = Cod<Fn>, typename Y = Cod<Gn>>
// clang-format on
auto fanout(Fn fn, Gn gn) -> Hom<T, P<X, Y>> {
  static_assert(std::is_same_v<T, U>);
  return [fn, gn](auto t) -> P<X, Y> {
    static_assert(std::is_invocable_v<Fn, decltype(t)>);
    static_assert(std::is_invocable_v<Gn, decltype(t)>);

    return {fn(t), gn(t)};
  };
}

// clang-format off
template <typename Fn,          typename Gn,
          typename T = Dom<Fn>, typename U = Dom<Gn>,
          typename X = Cod<Fn>, typename Y = Cod<Gn>>
// clang-format on
auto prod(Fn fn, Gn gn) -> Hom<P<T, U>, P<X, Y>> {
  return [fn, gn](P<T, U> tu) -> P<X, Y> {
    auto [t, u] = tu;
    return {fn(t), gn(u)};
  };
}

struct Pair {

  template <typename T, typename U>
  using Of = P<T, U>;

  template <typename Fn, typename Gn>
  static auto bimap(Fn f, Gn g)
      -> Hom<P<Dom<Fn>, Dom<Gn>>, P<Cod<Fn>, Cod<Gn>>> {
    return prod(std::forward<Fn>(f), std::forward<Gn>(g));
  }
};

// ........................................................ f]]]3
// Product associator ..................................... f[[[3

template <typename T, typename U, typename V>
auto associator_fd(P<T, P<U, V>> t_uv) -> P<P<T, U>, V> {
  auto [t, uv] = t_uv;
  auto [u, v] = uv;

  return {{t, u}, v};
}

template <typename T, typename U, typename V>
auto associator_rv(P<P<T, U>, V> tu_v) -> P<T, P<U, V>> {
  auto [tu, v] = tu_v;
  auto [t, u] = tu;

  return {t, {u, v}};
}

// ........................................................ f]]]3
// Product Unitor ......................................... f[[[3

struct PUnit { // monoidal unit for P
  bool operator==(const PUnit) const { return true; }
};

template <typename T>
auto l_unitor_fw(P<PUnit, T> it) -> T {
  return std::get<1>(it);
}

template <typename T>
auto l_unitor_rv(T t) -> P<PUnit, T> {
  return {PUnit{}, t};
}

template <typename T>
auto r_unitor_fw(P<T, PUnit> ti) -> T {
  return std::get<0>(ti);
}

template <typename T>
auto r_unitor_rv(T t) -> P<T, PUnit> {
  return {t, PUnit{}};
}

// ........................................................ f]]]3
// Product braiding, self-inverse ......................... f[[[3

template <typename T, typename U>
auto braid(P<T, U> tu) -> P<U, T> {
  auto [t, u] = tu;
  return {u, t};
}

// ........................................................ f]]]3
// covariant hom functor .................................. f[[[3

template <typename T>
struct HomFrom {
  template <typename U>
  using HomTo = Hom<T, U>;
};

template <typename T>
struct CHom {
  template <typename U>
  using Of = typename HomFrom<T>::template HomTo<U>;

  template <typename Fn>
  static auto fmap(Fn fn) -> Hom<Hom<T, Dom<Fn>>, Hom<T, Cod<Fn>>> {
    return [fn](auto gn) { return compose(fn, gn); };
  };
};

// ........................................................ f]]]3
// Cartesian closure laws ................................. f[[[3

template <typename T, typename U>
auto ev(P<Hom<T, U>, T> fn_and_arg) {
  auto [fn, x] = fn_and_arg;
  return fn(x);
}

// clang-format off
template <typename Fn, typename TU = Dom<Fn>,
          typename T = std::tuple_element_t<0, TU>,
          typename U = std::tuple_element_t<1, TU>,
          typename V = Cod<Fn>>
// clang-format on
auto pcurry(Fn fn) -> Hom<T, Hom<U, V>> {
  return [fn](T t) -> Hom<U, V> {
    return [fn, t](U u) -> V { return fn({t, u}); };
  };
}

// clang-format off
template <typename Fn,             typename T = Dom<Fn>,
          typename UtoV = Cod<Fn>, typename U = Dom<UtoV>,
                                   typename V = Cod<UtoV>>
// clang-format on
auto puncurry(Fn fn) -> Hom<P<T, U>, V> {
  return [fn](P<T, U> p) -> V { return fn(p.first)(p.second); };
}

// ........................................................ f]]]3
// ........................................................ f]]]2
// Cocartesian monoid in Cpp .............................. f[[[2
// Categorical coproduct bifunctor ........................ f[[[3

template <typename T, typename U>
struct S : std::variant<T, U> {
  using std::variant<T, U>::variant;

  S() = delete;
};

template <typename T, typename U>
S(T, U) -> S<T, U>;

template <std::size_t N, typename S>
struct sum_term;

template <typename T, typename U>
struct sum_term<0, S<T, U>> {
  using type = T;
};

template <typename T, typename U>
struct sum_term<1, S<T, U>> {
  using type = U;
};

template <std::size_t N, typename S>
using sum_term_t = typename sum_term<N, S>::type;

struct Never { // Monoidal unit for S
  Never() = delete;
  Never(const Never &) = delete;
  virtual ~Never();

  bool operator==(const Never &) const {
    throw std::domain_error(
        "`Never` instances should not exist, "
        "and someone must have done something perverse.");
  }
};

template <typename T, typename U>
auto inject_l(T t) -> S<T, U> {
  return S<T, U>(std::in_place_index<0>, t);
}

template <typename T, typename U>
auto inject_r(U t) -> S<T, U> {
  return S<T, U>(std::in_place_index<1>, t);
}

// clang-format off
template <typename Fn,          typename Gn,
          typename T = Dom<Fn>, typename U = Dom<Gn>,
          typename V = Cod<Fn>>
// clang-format on
auto fanin(Fn fn, Gn gn) -> Hom<S<T, U>, V> {

  static_assert(std::is_same_v<V, Cod<Gn>>);

  return [fn, gn](S<T, U> t_or_u) -> V {
    static_assert(std::is_invocable_v<Fn, T>);
    static_assert(std::is_invocable_v<Gn, U>);

    if (t_or_u.index() == 0)
      return fn(std::get<0>(t_or_u));
    else
      return gn(std::get<1>(t_or_u));
  };
}

// ((A → B), (C → D)) → (A + C → B + D)
// clang-format off
template <typename Fn,          typename Gn,
          typename T = Dom<Fn>, typename U = Dom<Gn>,
          typename X = Cod<Fn>, typename Y = Cod<Gn>>
// clang-format on
auto coprod(Fn fn, Gn gn) -> Hom<S<T, U>, S<X, Y>> {
  using TorU = S<T, U>;
  using XorY = S<X, Y>;

  return [fn, gn](TorU t_or_u) -> XorY {
    if (t_or_u.index() == 0)
      return inject_l<X, Y>(fn(std::get<0>(t_or_u)));
    else
      return inject_r<X, Y>(gn(std::get<1>(t_or_u)));
  };
}

struct Either {
  template <typename T, typename U>
  using Of = P<T, U>;

  template <typename Fn, typename Gn>
  static auto bimap(Fn fn, Gn gn) {
    using T = Dom<Fn>;
    using U = Dom<Gn>;
    using X = Cod<Fn>;
    using Y = Cod<Gn>;
    using TorU = S<T, U>;
    using XorY = S<X, Y>;

    return [fn, gn](TorU t_or_u) -> XorY {
      if (t_or_u.index() == 0)
        return inject_l<X, Y>(fn(std::get<0>(t_or_u)));
      else
        return inject_r<X, Y>(gn(std::get<1>(t_or_u)));
    };
  };

  template <typename Fn, typename U>
  static auto lmap(Fn fn) {
    return [fn](Of<Dom<Fn>, U> tu) { return bimap(fn, id<U>); };
  };

  template <typename Gn, typename T>
  static auto rmap(Gn gn) {
    return [gn](Of<T, Dom<Gn>> tu) { return bimap(id<T>, gn); };
  };
};

// ........................................................ f]]]3
// Coproduct associator ................................... f[[[3

template <typename T, typename U, typename V>
auto associator_co_fd(S<T, S<U, V>> t_uv) -> S<S<T, U>, V> {
  if (t_uv.index() == 0) {
    if constexpr (!std::is_same_v<T, Never>)
      return inject_l<S<T, U>, V>(std::get<0>(t_uv));
  } else {
    auto &uv = std::get<1>(t_uv);
    if (uv.index() == 0) {
      if constexpr (!std::is_same_v<U, Never>)
        return inject_l<S<T, U>, V>(std::get<0>(uv));
    } else {
      if constexpr (!std::is_same_v<V, Never>)
        return inject_r<S<T, U>, V>(std::get<1>(uv));
    }
  }
  throw std::domain_error("Recieved a variant with no value.");
}

template <typename T, typename U, typename V>
auto associator_co_rv(S<S<T, U>, V> tu_v) -> S<T, S<U, V>> {
  if (tu_v.index() == 0) {
    auto &tu = std::get<0>(tu_v);
    if (tu.index() == 0) {
      if constexpr (!std::is_same_v<T, Never>)
        return inject_l<T, S<U, V>>(std::get<0>(tu));
    } else {
      if constexpr (!std::is_same_v<U, Never>)
        return inject_r<T, S<U, V>>(std::get<1>(tu));
    }
  } else {
    if constexpr (!std::is_same_v<V, Never>)
      return inject_r<T, S<U, V>>(std::get<1>(tu_v));
  }
  throw std::domain_error("Recieved a variant with no value.");
}

// ........................................................ f]]]3
// Corpdocut unitor ....................................... f[[[3

template <typename T>
struct S<T, Never> : std::variant<T, Never> {
  using std::variant<T, Never>::variant;

  S() : std::variant<T, Never>{inject_l<T, Never>(T{})} {}

  S(const S &other)
      : std::variant<T, Never>(
            std::in_place_type<T>, std::get<T>(other)) {}
};

template <typename T>
struct S<Never, T> : std::variant<Never, T> {
  using std::variant<Never, T>::variant;

  S() : std::variant<Never, T>{inject_r<Never, T>(T{})} {}

  S(const S &other)
      : std::variant<Never, T>(
            std::in_place_type<T>, std::get<T>(other)) {}
};

template <typename T>
auto l_unitor_co_fw(S<Never, T> just_t) -> T {
  return std::get<1>(just_t);
}

template <typename T>
auto l_unitor_co_rv(T t) -> S<Never, T> {
  return inject_r<Never, T>(t);
}

template <typename T>
auto r_unitor_co_fw(S<T, Never> just_t) -> T {
  return std::get<0>(just_t);
}

template <typename T>
auto r_unitor_co_rv(T t) -> S<T, Never> {
  return inject_l<T, Never>(t);
}

// ........................................................ f]]]3
// Coproduct symmetric braiding ........................... f[[[3

template <typename T, typename U>
auto braid_co(S<T, U> t_or_u) -> S<U, T> {
  if (t_or_u.index() == 0)
    return inject_r<U, T>(std::get<0>(t_or_u));
  else
    return inject_l<U, T>(std::get<1>(t_or_u));
}

// ........................................................ f]]]3
// ........................................................ f]]]2
// BiCCC: currying/product equations ...................... f[[[2
// Product distributes over coproduct ..................... f[[[3

// BUG: I am not going to pessimise the `distributor`
//   against T or U or X = Never, because the nesting is just too
//   hard to read for thesis code, and I only need to get the
//   point across.

template <typename T, typename U, typename X>
auto factorise(S<P<T, X>, P<U, X>> tx_ux) -> P<S<T, U>, X> {
  // clang-format off
  const auto universal_factorise = fanin(
        prod(inject_l<T, U>, id<X>),
        prod(inject_r<T, U>, id<X>)
      );
  // clang-format on

  return universal_factorise(tx_ux);
}

template <typename T, typename U, typename Z>
auto expand(P<S<T, U>, Z> t_or_u_and_x) -> S<P<T, Z>, P<U, Z>> {
  // clang-format off
  // tz : T $×$ Z → (T $×$ Z) + (U $×$ Z)
  const auto tz =
      pcurry(compose(
              inject_l<P<T, Z>, P<U, Z>>,
              id<P<T, Z>>
            ));
  // uz : U $×$ Z → (T $×$ Z) + (U $×$ Z)
  const auto uz =
      pcurry(compose(
              inject_r<P<T, Z>, P<U, Z>>,
              id<P<U, Z>>
            ));

  // tz_uz : (T + U) → Z $⊸$ (T $×$ Z) + (U $×$ Z)
  const auto tz_uz = fanin(tz, uz);
  // universal_expand : (T + U) $×$ Z  $⊸$  (T $×$ Z) + (U $×$ Z)
  const auto universal_expand = puncurry(tz_uz);
  // clang-format on

  return universal_expand(t_or_u_and_x);
}

// ........................................................ f]]]3
// ........................................................ f]]]2
// Isomorphism between C++ function arguments and tuples .. f[[[2

template <typename Fn>
auto to_unary(Fn &&f) {
  return [f = std::forward<Fn>(f)](auto &&args) mutable {
    return std::apply(f, std::forward<decltype(args)>(args));
  };
}

template <typename Fn>
auto to_n_ary(Fn &&f) {
  return [f = std::forward<Fn>(f)](auto &&...args) mutable {
    return f(
        std::make_tuple(std::forward<decltype(args)>(args)...));
  };
}

// ........................................................ f]]]2
// MP<T> functor fixpoint ................................. f[[[2
// List = List = Mu<MP<T>::template Left> ................. f[[[3

template <template <typename> class F>
struct Mu : F<Mu<F>> {
  explicit Mu(F<Mu<F>> f) : F<Mu<F>>(f) {}
};

template <template <typename> class F>
auto in(F<Mu<F>> f) -> Mu<F> {
  return Mu<F>{f};
}

template <template <typename> class F>
auto out(Mu<F> f) -> F<Mu<F>> {
  return f;
}

template <typename T>
struct MP {
  template <typename U>
  using Left = S<PUnit, P<std::shared_ptr<U>, T>>;

  using Right = T; // not really usable.
};

// Helpers ................................................ f[[[4
template <typename T>
using maybe_pair_element_t = typename sum_term_t<1, T>::second_type;

template <typename Lst>
using snoclist_element_type = typename std::remove_reference<
    decltype(std::get<1>(out(std::declval<Lst>())))>::type::
    second_type;

template <typename T, typename U>
constexpr bool has_pair(S<PUnit, P<T, U>> const &i_or_val) {
  return i_or_val.index() == 1;
}
// ........................................................ f]]]4

template <typename T>
using SnocList = Mu<MP<T>::template Left>;

template <typename T>
auto nil = in<MP<T>::template Left>(PUnit{});

template <typename T>
auto snoc(SnocList<T> lst, T t) -> SnocList<T> {
  return in<MP<T>::template Left>(
      P{std::make_shared<SnocList<T>>(lst), t});
}

template <typename Lst, typename T = snoclist_element_type<Lst>>
auto unsafe_head(Lst lst) -> T {
  return std::get<1>(out(lst)).second;
}

template <typename Lst, typename T = snoclist_element_type<Lst>>
auto operator==(Lst const &lhs, Lst const &rhs) -> bool {
  auto l = out(lhs);
  auto r = out(rhs);

  while (has_pair(l) && has_pair(r)) {
    auto [lhs_tail, lhs_val] = std::get<1>(l);
    auto [rhs_tail, rhs_val] = std::get<1>(r);
    if (!(lhs_val == rhs_val)) {
      return false;
    }
    l = *lhs_tail;
    r = *rhs_tail;
  }

  // If one of l or r still hold a pair at this point, then lhs
  // and rhs are different lengths and are not equal.
  return !has_pair(l) && !has_pair(r);
}

template <typename Lst, typename T = snoclist_element_type<Lst>>
auto operator<<(std::ostream &os, Lst const &lst)
    -> std::ostream & {
  auto l = out(lst);

  while (has_pair(l)) {
    auto [tail, val] = std::get<1>(l);
    os << val << " ";
    l = *tail;
  }

  return os;
}

// ........................................................ f]]]3
// Isomorphism between List and std::vector ............... f[[[3
template <typename Lst>
auto to_vector(const Lst lst)
    -> std::vector<snoclist_element_type<Lst>> {
  using T = snoclist_element_type<Lst>;
  static_assert(std::is_same_v<Lst, SnocList<T>>);

  std::vector<T> output;
  auto outermost = out(lst);
  while (has_pair(outermost)) {
    auto [tail, head] = std::get<1>(outermost);
    output.push_back(head);
    outermost = out(*tail);
  }

  std::reverse(output.begin(), output.end());

  return output;
}

template <typename T>
auto to_snoclist(const std::vector<T> &vec) -> SnocList<T> {

  SnocList<T> accumulator = in<MP<T>::template Left>(PUnit{});

  for (auto it = vec.cbegin(); it != vec.cend(); ++it) {
    accumulator = snoc(accumulator, *it);
  }

  return accumulator;
}

// ........................................................ f]]]3
// List<T>-catamorphisms .................................. f[[[3

template <typename T>
struct SnocF {

  template <typename U>
  using Of = typename MP<T>::template Left<U>;

  template <typename Fn>
  static auto fmap(Fn fn) -> Hom<Of<Dom<Fn>>, Of<Cod<Fn>>> {
    return [fn](Of<Dom<Fn>> i_or_p) -> Of<Cod<Fn>> {
      using Elem = maybe_pair_element_t<Of<Dom<Fn>>>;

      return coprod(id<PUnit>, prod(sptr::map(fn), id<Elem>))(
          i_or_p);
    };
  }

  template <typename Carrier>
  using Alg = Hom<typename MP<T>::template Left<Carrier>, Carrier>;

  template <typename Carrier>
  static auto cata(Alg<Carrier> alg) -> Hom<SnocList<T>, Carrier> {
    return [alg](SnocList<T> ts) {
      return alg(fmap(cata<Carrier>(alg))(out(ts)));
      // clang-format off
      //     alg   $∘$   $\Ffmap{\ttF}(\catam{\ttVar{alg}})$   $∘$  out
      // clang-format on
    };
  }
};

template <typename T, typename Carrier>
using SnocAlg = typename SnocF<T>::template Alg<Carrier>;

template <typename T, typename Carrier>
auto make_cata(SnocAlg<T, Carrier> &&alg) {
  return SnocF<T>::template cata<Carrier>(
      std::forward<decltype(alg)>(alg));
}

// ........................................................ f]]]3
// ........................................................ f]]]2
// ........................................................ f]]]1
