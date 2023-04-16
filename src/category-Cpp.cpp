// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65
#include <bits/utility.h>
#include <catch2/catch.hpp>
#include <cctype>
#include <optional>
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

#include "test-tools.hpp"
using tst::A; // Tag for unit type
using tst::B; // Tag for unit type
using tst::C; // Tag for unit type
using tst::D; // Tag for unit type
using tst::f; // f : A → B
using tst::g; // g : B → C
using tst::h; // h : C → D
// using tst::id;

// tfunc test cases ....................................... f[[[1
TEST_CASE(
    "Polymorphic identity function should perfectly "
    "forward and …",
    "[id], [interface]") {
  SECTION("… identify lvalues", "[mathematical]") {
    auto a = A{};
    auto b = B{};
    REQUIRE(tf::id(a) == A{});
    REQUIRE(tf::id(b) == B{});
  }
  SECTION("… preserve lvalue-refness") {
    REQUIRE(std::is_lvalue_reference<
        decltype(tf::id(std::declval<int &>()))>::value);
  }
  SECTION("… identify rvalues", "[mathematical]") {
    REQUIRE(tf::id(A{}) == A{});
    REQUIRE(tf::id(B{}) == B{});
  }
  SECTION("… preserve rvalue-refness") {
    REQUIRE(std::is_rvalue_reference<
        decltype(tf::id(std::declval<int &&>()))>::value);
  }
}

// For convenience with functions of std::function type,
//   wrap tf::id in a std::function.
template <typename T>
auto id = Hom<T, T>{tf::id<T>};

TEST_CASE("compose(f) == f ∘ id_A == id_B ∘ f.", //
    "[compose], [mathematical]") {
  REQUIRE(compose(f)(A{}) == compose(f, id<A>)(A{}));
  REQUIRE(compose(f, id<A>)(A{}) == compose(id<B>, f)(A{}));
}

TEST_CASE("Compose two C-functions", "[compose], [interface]") {
  auto id2 = compose(id<A>, id<A>);
  REQUIRE(id2(A{}) == A{});
}

template <typename T>
struct Getable {
  T N;
  Getable(T N) : N(N) {}
  [[nodiscard]] T get() const { return N; }
};

TEST_CASE(
    "Return of a composed function should preserve the "
    "rvalue-refness of the outer function", //
    "[compose], [interface]") {
  B b{};
  auto ref_to_b = [&b](A) -> B & { return b; };
  auto fog = compose(ref_to_b, id<A>);
  REQUIRE(std::is_lvalue_reference<decltype(fog(A{}))>::value);
}

TEST_CASE(
    "Curried non-variadic functions should bind "
    "arguments, one at a time, from left to right.", //
    "[curry], [non-variadic], [interface]") {

  // abcd : (A, B, C, D) → (A, B, C, D)
  // a_b_c_d : A → B → C → D → (A, B, C, D)
  auto abcd = [](A a, B b, C c, D d) {
    return std::tuple{a, b, c, d};
  };
  auto a_b_c_d = curry(abcd);

  auto [a, b, c, d] = a_b_c_d(A{})(B{})(C{})(D{});
  REQUIRE(a == A{});
  REQUIRE(b == B{});
  REQUIRE(c == C{});
  REQUIRE(d == D{});
}

TEST_CASE("Curried functions should…") {
  auto ABtoC = curry([](A, B) -> C { return {}; });
  auto BtoC = ABtoC(A{});

  SECTION("… compose with other callables.",
      "[curry], "
      "[compose], "
      "[interface]") {
    REQUIRE(compose(BtoC, f)(A{}) == C{});
  }
}

TEST_CASE(
    "Currying should work with the C++14 std outfix "
    "operators") {
  auto plus = curry(std::plus<int>{});
  auto increment = plus(1);
  REQUIRE(increment(0) == 1);
}
// ........................................................ f]]]1
// Demos of structure in Cpp .............................. f[[[1
// Basic category axioms .................................. f[[[2

TEST_CASE("Check associativity: (h.g).f == h.(g.f)") {
  REQUIRE(D{} == compose(h, g, f)(A{}));
  REQUIRE(
      compose(h, g, f)(A{}) == compose(h, compose(g, f))(A{}));
  REQUIRE(compose(compose(h, g), f)(A{}) ==
          compose(h, compose(g, f))(A{}));
}

TEST_CASE("f == f ∘ id_A == id_B ∘ f.") {
  REQUIRE(f(A{}) == compose(f, id<A>)(A{}));
  REQUIRE(f(A{}) == compose(id<B>, f)(A{}));
  REQUIRE(compose(f, id<A>)(A{}) == compose(id<B>, f)(A{}));
}

// ........................................................ f]]]2
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

TEST_CASE("Check the functor laws for IdF") {
  //    Alias for tst::A
  //             $↓$
  auto a = Id::Of<A>{};
  auto IdF_f = Id::fmap(f);
  auto IdF_g = Id::fmap(g);
  auto IdF_gf = Id::fmap<Hom<A, C>>(compose(g, f));
  auto IdF_idA = Id::fmap(id<A>);

  REQUIRE(compose(IdF_g, IdF_f)(a) == IdF_gf(a));
  REQUIRE(IdF_idA(a) == id<A>(a));
}
// ........................................................ f]]]2
// Optional functor ....................................... f[[[2

namespace Optional {
  template <typename T>
  using Of = std::optional<T>;

  template <typename Fn>
  static auto fmap(Fn f) -> Hom<Of<Dom<Fn>>, Of<Cod<Fn>>> {
    using T = Dom<Fn>;
    using U = Cod<Fn>;
    return [f](Of<T> ot) -> Of<U> {
      if (ot)
        return f(ot.value());
      else
        return std::nullopt;
    };
  }
} // namespace Optional

TEST_CASE("Basic behavioural tests of Optional::fmap") {
  auto a = Optional::Of<A>{A{}};
  auto not_a = Optional::Of<A>{};

  auto opt_f = Optional::fmap(f);
  auto opt_g = Optional::fmap(g);

  auto A_to_nullopt = [](Optional::Of<A>) -> Optional::Of<B> {
    return std::nullopt;
  };

  REQUIRE((compose(opt_g, opt_f)(a)).value() == C{});
  REQUIRE(compose(opt_g, opt_f)(not_a) == std::nullopt);
  REQUIRE(compose(opt_g, A_to_nullopt)(a) == std::nullopt);
}

TEST_CASE("Check the functor laws for Optional::fmap") {
  //        Alias for std::optional<A>
  //                 $↓$
  auto a = Optional::Of<A>{A{}};
  auto not_a = Optional::Of<A>{};

  auto opt_f = Optional::fmap(f);
  auto opt_g = Optional::fmap(g);
  auto opt_gf = Optional::fmap(compose(g, f));
  auto opt_idA = Optional::fmap(id<A>);

  REQUIRE(compose(opt_g, opt_f)(a) == opt_gf(a));
  REQUIRE(opt_idA(a) == id<Optional::Of<A>>(a));

  REQUIRE(compose(opt_g, opt_f)(not_a) == opt_gf(not_a));
  REQUIRE(opt_idA(not_a) == id<Optional::Of<A>>(not_a));
}

// ........................................................ f]]]2
// std::vector based List-functor ......................... f[[[2

namespace Vector {
  template <typename T>
  using Of = std::vector<T>;

  template <typename Fn>
  auto fmap(Fn f) -> Hom<Of<Dom<Fn>>, Of<Cod<Fn>>> {
    using T = Dom<Fn>;
    using U = Cod<Fn>;
    return [f](Of<T> t_s) {
      Of<U> u_s;
      u_s.reserve(t_s.size());

      std::transform(
          cbegin(t_s), cend(t_s), std::back_inserter(u_s), f);
      return u_s;
    };
  };
}; // namespace Vector

TEST_CASE("Check the functor laws for Vector::fmap") {
  //        Alias for std$∷$vector<A>
  //                 $↓$
  auto a_s = Vector::Of<A>{A{}, A{}, A{}};

  REQUIRE(compose(Vector::fmap(g), Vector::fmap(f))(a_s) ==
          Vector::fmap(compose(g, f))(a_s));

  REQUIRE(Vector::fmap(id<A>)(a_s) == id<Vector::Of<A>>(a_s));
}

// ........................................................ f]]]2
// Constant functor ....................................... f[[[2

template <typename T>
struct Always {
  template <typename>
  using given = T;
};

template <typename T>
struct Const {

  template <typename U>
  using Of = typename Always<T>::template given<U>;

  template <typename Fn>
  static auto fmap(Fn) -> Hom<T, T> {
    return id<T>;
  };
};

TEST_CASE("Functor axioms of Const<A>.") {
  // clang-format off
  REQUIRE(
    compose(Const<A>::fmap(g), Const<A>::fmap(f))(A{}) ==
      Const<A>::fmap(compose(g, f))(A{})
  );
  // clang-format on

  REQUIRE(Const<A>::fmap(id<A>)(A{}) == A{});
  // Interestingly, id<B> will be mapped to id<A> too:
  REQUIRE(Const<A>::fmap(id<B>)(A{}) == A{});
  // as will id<C>:
  REQUIRE(Const<A>::fmap(id<C>)(A{}) == A{});
}

// ........................................................ f]]]2
// Natural Transformations ................................ f[[[2

template <typename T>
auto len(Vector::Of<T> t_s) -> Const<std::size_t>::Of<T> {
  return t_s.size();
}

TEST_CASE("Test naturality square for len.") {
  constexpr std::size_t actual_length = 5;

  auto a_s = Vector::Of<A>(actual_length);

  // Does what it is supposed to:
  REQUIRE(len(a_s) == actual_length);

  // Satisfies the naturality square:
  // clang-format off
  REQUIRE(
    compose(len<B>, Vector::fmap(f))(a_s)
        ==
          compose(Const<std::size_t>::fmap(f), len<A>)(a_s)
  );
  // clang-format on
}

// ........................................................ f]]]2
// CCC in Cpp ............................................. f[[[2
// Categorical product bifunctor .......................... f[[[3

template <typename T, typename U>
using P = std::pair<T, U>;

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
auto prod(Fn f, Gn g) -> Hom<P<T, U>, P<X, Y>> {
  return [f, g](P<T, U> tu) -> P<X, Y> {
    auto [t, u] = tu;
    return {f(t), g(u)};
  };
}

TEST_CASE("P (via prod) is functorial in both factors.") {

  auto ab = P<A, B>{};

  REQUIRE(compose(prod(g, id<B>), prod(f, id<B>))(ab) ==
          prod(compose(g, f), id<B>)(ab));

  REQUIRE(compose(prod(id<A>, h), prod(id<A>, g))(ab) ==
          prod(id<A>, compose(h, g))(ab));

  REQUIRE(prod(id<A>, id<B>)(ab) == id<P<A, B>>(ab));
}

struct Pair {

  template <typename T, typename U>
  using Of = std::pair<T, U>;

  template <typename Fn, typename Gn>
  static auto bimap(Fn f, Gn g)
      -> Hom<P<Dom<Fn>, Dom<Gn>>, P<Cod<Fn>, Cod<Gn>>> {
    return prod(std::forward<Fn>(f), std::forward<Gn>(g));
  }
};

TEST_CASE("Pair is functorial in the left position.") {

  auto ab = Pair::Of<A, B>{};

  // clang-format off
  REQUIRE(
    compose(Pair::bimap(g, id<B>), Pair::bimap(f, id<B>))(ab) ==
              Pair::bimap(compose(g, f), id<B>)(ab)
  );

  REQUIRE(
    compose(Pair::bimap(id<A>, h), Pair::bimap(id<A>, g))(ab) ==
              Pair::bimap(id<A>, compose(h, g))(ab)
  );
  // clang-format on

  REQUIRE(Pair::bimap(id<A>, id<B>)(ab) == id<P<A, B>>(ab));
}

// This is cute, but doesn't really make the code more readable:
//
// template <typename Fn, typename Gn>
// auto operator*(Fn f, Gn g)
//     -> Hom<P<Dom<Fn>, Dom<Gn>>, P<Cod<Fn>, Cod<Gn>>> {
//   using TandU = P<Dom<Fn>, Dom<Gn>>;
//   using XandY = P<Cod<Fn>, Cod<Gn>>;
//
//   return [f, g](TandU tu) -> XandY {
//     auto [t, u] = tu;
//     return {f(t), g(u)};
//   };
// }

// clang-format off
template <typename Fn,          typename Gn,
          typename T = Dom<Fn>, typename U = Dom<Gn>,
          typename X = Cod<Fn>, typename Y = Cod<Gn>>
// clang-format on
auto fanout(Fn f, Gn g) -> Hom<T, P<X, Y>> {
  static_assert(std::is_same_v<T, U>);
  return [f, g](auto t) {
    static_assert(std::is_invocable_v<Fn, decltype(t)>);
    static_assert(std::is_invocable_v<Gn, decltype(t)>);

    return std::pair{f(t), g(t)};
  };
}

TEST_CASE("Coherence of $\\eqref{cd:cpp:binary-product}$") {
  auto a_to_b = [](A) { return B{}; };
  auto a_to_c = [](A) { return C{}; };

  auto a_to_bc = fanout(a_to_b, a_to_c);

  SECTION("Equation defining fanout.") {
    // clang-format off
    REQUIRE(
        fanout(
            compose(proj_l<B, C>, a_to_bc),
            compose(proj_r<B, C>, a_to_bc)) (A{}) 
              == a_to_bc(A{})
    );
    // clang-format on
  }

  SECTION("Commutativity of left and right triangles in "
          "$\\eqref{cd:cpp:binary-product}$") {
    REQUIRE(proj_l(a_to_bc(A{})) == a_to_b(A{}));
    REQUIRE(proj_r(a_to_bc(A{})) == a_to_c(A{}));
  }
}

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

TEST_CASE(
    "associator_fd and associator_rv are mutually "
    "inverse.") {
  // clang-format off
  auto associator_fd_rv = compose(
        associator_rv<A, B, C>,
        associator_fd<A, B, C>
      );
  auto associator_rv_fd = compose(
        associator_fd<A, B, C>,
        associator_rv<A, B, C>
      );
  // clang-format on

  auto a_bc = P<A, P<B, C>>{};
  REQUIRE(associator_fd_rv(a_bc) == id<P<A, P<B, C>>>(a_bc));
  auto ab_c = P<P<A, B>, C>{};
  REQUIRE(associator_rv_fd(ab_c) == id<P<P<A, B>, C>>(ab_c));
}

TEST_CASE("Associator diagram for std::pair") {
  auto start = P<A, P<B, P<C, D>>>{};

  // clang-format off
  auto cw_path = compose(
        associator_fd<P<A, B>, C, D>,
        associator_fd<A, B, P<C, D>>
      );

  auto ccw_path = compose(
        prod(associator_fd<A, B, C>, id<D>),
        associator_fd<A, P<B, C>, D>,
        prod(id<A>, associator_fd<B, C, D>)
      );
  // clang-format on

  REQUIRE(ccw_path(start) == cw_path(start));
};

// ........................................................ f]]]3
// Product Unitor ......................................... f[[[3

// unit-type is any singleton.
struct I {
  bool operator==(const I) const { return true; }
};

template <typename T>
auto l_unitor_fw(P<I, T> it) -> T {
  return std::get<1>(it);
}

template <typename T>
auto l_unitor_rv(T t) -> P<I, T> {
  return {I{}, t};
}

template <typename T>
auto r_unitor_fw(P<T, I> ti) -> T {
  return std::get<0>(ti);
}

template <typename T>
auto r_unitor_rv(T t) -> P<T, I> {
  return {t, I{}};
}

TEST_CASE("_fw and _rv are mutual inverses for L-/R-unitor") {
  auto ia = P<I, A>{};
  auto ai = P<A, I>{};

  // clang-format off
  REQUIRE(
      compose(l_unitor_rv<A>, l_unitor_fw<A>)(ia) ==
          id<P<I, A>>(ia)
  );
  REQUIRE(
      compose(l_unitor_fw<A>, l_unitor_rv<A>)(A{}) ==
          id<A>(A{})
  );

  REQUIRE(
      compose(r_unitor_rv<A>, r_unitor_fw<A>)(ai) ==
          id<P<A, I>>(ai)
  );
  REQUIRE(compose(r_unitor_fw<A>, r_unitor_rv<A>)(A{}) == 
          id<A>(A{})
  );
  // clang-format on
}

TEST_CASE("Unitor diagram") {
  auto a_ib = P<A, P<I, B>>{};

  // clang-format off
  auto cw_path = compose(
        prod(r_unitor_fw<A>, id<B>),
        associator_fd<A, I, B>
      );
  auto ccw_path = prod(id<A>, l_unitor_fw<B>);
  // clang-format on

  REQUIRE(cw_path(a_ib) == ccw_path(a_ib));
}

// ........................................................ f]]]3
// Product braiding, self-inverse ......................... f[[[3
template <typename T, typename U>
auto braid(P<T, U> tu) -> P<U, T> {
  auto [t, u] = tu;
  return {u, t};
}

TEST_CASE("Braiding is self-inverse") {
  auto ab = P<A, B>{};
  REQUIRE(braid(braid(ab)) == id<P<A, B>>(ab));
}

TEST_CASE("Braiding diagram 1") {
  auto ab_c = P<P<A, B>, C>{};

  // clang-format off
  auto cw_path = compose(
        prod(braid<C, A>, id<B>),
        associator_fd<C, A, B>,
        braid<P<A, B>, C>
      );
  auto ccw_path = compose(
        associator_fd<A, C, B>,
        prod(id<A>, braid<B, C>),
        associator_rv<A, B, C>
      );
  // clang-format on

  REQUIRE(cw_path(ab_c) == ccw_path(ab_c));
}

TEST_CASE("Braiding diagram 2") {
  auto a_bc = P<A, P<B, C>>{};

  // clang-format off
  auto cw_path = compose(
        prod(id<B>, braid<C, A>),
        associator_rv<B, C, A>,
        braid<A, P<B, C>>
      );
  auto ccw_path = compose(
        associator_rv<B, A, C>,
        prod(braid<A, B>, id<C>),
        associator_fd<A, B, C>
      );
  // clang-format on

  REQUIRE(cw_path(a_bc) == ccw_path(a_bc));
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
  static auto fmap(Fn f)
      -> Hom<Hom<T, Dom<Fn>>, Hom<T, Cod<Fn>>> {
    return [f](auto g) { return compose(f, g); };
  };
};

TEST_CASE("Functor laws for CHom—the covariant hom-functor") {

  // clang-format off
  // $\Ffmap{\ttName{Chom}}(\ttg) ∘ \Ffmap{\ttName{CHom}}(\ttf) = \Ffmap{\ttName{CHom}}(\ttg ∘ \ttf)$
  REQUIRE(
    compose(CHom<A>::fmap(g), CHom<A>::fmap(f))(id<A>)(A{}) ==
      CHom<A>::fmap<Hom<A, C>>(compose(g, f))(id<A>)(A{})
    );
  // clang-format on

  REQUIRE(CHom<A>::fmap(id<A>)(id<A>)(A{}) == id<A>(A{}));
  REQUIRE(CHom<A>::fmap(id<B>)(f)(A{}) == f(A{}));
  auto gof = compose(g, f);
  REQUIRE(CHom<A>::fmap(id<C>)(gof)(A{}) == gof(A{}));
}
// ........................................................ f]]]3
// Cartesian closure laws ................................. f[[[3

template <typename T, typename U>
auto ev(P<Hom<T, U>, T> fn_and_arg) {
  auto [f, x] = fn_and_arg;
  return std::invoke(f, x);
}

template <typename Fn, typename TU = Dom<Fn>,
    typename T = std::tuple_element_t<0, TU>,
    typename U = std::tuple_element_t<1, TU>,
    typename V = Cod<Fn>>
auto pcurry(Fn f) -> Hom<T, Hom<U, V>> {
  return [f](T t) -> Hom<U, V> {
    return [f, t](U u) -> V {
      return std::invoke(f, std::pair{t, u});
    };
  };
}

// LaTeX version:
// TEST_CASE("Commutativity of $\eqref{cd:cpp-exponential}$") {
TEST_CASE("Commutativity of $\\eqref{cd:cpp-exponential}$") {
  auto ab = P<A, B>{};
  auto k = [](P<A, B>) -> C { return {}; };

  auto cw = compose(ev<B, C>, prod(pcurry(k), id<B>));

  REQUIRE(cw(ab) == k(ab));
}

template <typename Fn>
auto puncurry(Fn f) {
  using T = Dom<Fn>;
  using UtoV = Cod<Fn>;
  using U = Dom<UtoV>;
  using V = Cod<UtoV>;

  return [f](std::pair<T, U> p) -> V {
    return std::invoke(std::invoke(f, p.first), p.second);
  };
}

TEST_CASE("pcurry and puncurry are inverse") {
  auto ab = P<A, B>{};
  auto k = [](P<A, B>) -> C { return {}; };

  REQUIRE(puncurry(pcurry(k))(ab) == C{});
}

TEST_CASE("Čubrić (1994) equations.") {
  auto cb_to_a = [](P<C, B>) { return A{}; };
  auto c_to_b_to_a = pcurry(cb_to_a);

  auto lhs1 = compose(ev<B, A>,
      fanout(
          compose(pcurry(cb_to_a), proj_l<C, B>), proj_r<C, B>));

  REQUIRE(lhs1(P<C, B>{}) == cb_to_a(P<C, B>{}));

  // clang-format off
  auto lhs2 = pcurry(Hom<P<C,B>, A>{
        compose(ev<B, A>,
          fanout(
            compose(c_to_b_to_a, proj_l<C, B>),
            proj_r<C, B>
          )
        )
        });
  // clang-format on

  REQUIRE(lhs2(C{})(B{}) == c_to_b_to_a(C{})(B{}));
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

struct Never { // Monoidal unit for S
  Never() = delete;
  Never(const Never &) = delete;
  bool operator==(const Never &) const { return false; }
};

template <typename T, typename U>
auto inject_l(T t) -> S<T, U> {
  return S<T, U>(std::in_place_index<0>, t);
}

template <typename T, typename U>
auto inject_r(U t) -> S<T, U> {
  return S<T, U>(std::in_place_index<1>, t);
}

template <typename Fn, typename Gn, typename T = Dom<Fn>,
    typename U = Dom<Gn>, typename V = Cod<Fn>>
auto fanin(Fn f, Gn g) -> Hom<S<T, U>, V> {

  static_assert(std::is_same_v<V, Cod<Gn>>);

  return [f, g](S<T, U> t_or_u) -> V {
    static_assert(std::is_invocable_v<Fn, T>);
    static_assert(std::is_invocable_v<Gn, U>);

    if (t_or_u.index() == 0)
      return std::invoke(f, std::get<0>(t_or_u));
    else
      return std::invoke(g, std::get<1>(t_or_u));
  };
}

// ((A → B), (C → D)) → (A + C → B + D)
template <typename Fn, typename Gn, typename T = Dom<Fn>,
    typename U = Dom<Gn>, typename X = Cod<Fn>,
    typename Y = Cod<Gn>>
auto coprod(Fn f, Gn g) -> Hom<S<T, U>, S<X, Y>> {

  using TorU = S<T, U>;
  using XorY = S<X, Y>;

  return [f, g](TorU t_or_u) -> XorY {
    if (t_or_u.index() == 0)
      return std::invoke(f, std::get<0>(t_or_u));
    else
      return std::invoke(g, std::get<1>(t_or_u));
  };
}

TEST_CASE("Equation defining fanin") {
  auto a_or_b_to_c = [](S<A, B>) { return C{}; };
  auto actual_a = inject_l<A, B>(A{});
  auto actual_b = inject_r<A, B>(B{});

  // clang-format off
  auto fanned = fanin(
              compose(a_or_b_to_c, inject_l<A, B>),
              compose(a_or_b_to_c, inject_r<A, B>)
            );
  // clang-format on

  REQUIRE(fanned(actual_a) == a_or_b_to_c(actual_a));
  REQUIRE(fanned(actual_b) == a_or_b_to_c(actual_b));
}

// LaTeX version
// TEST_CASE(
//     "Commutativity of left and right triangles in
//     $\eqref{cd:cpp:binary-coproduct}$") {
TEST_CASE(
    "Commutativity of left and right triangles in "
    "coproduct diagram") {
  auto a_to_c = [](A) { return C{}; };
  auto b_to_c = [](B) { return C{}; };

  auto left_triangle_path =
      compose(fanin(a_to_c, b_to_c), inject_l<A, B>);
  REQUIRE(left_triangle_path(A{}) == a_to_c(A{}));

  auto right_triangle_path =
      compose(fanin(a_to_c, b_to_c), inject_r<A, B>);
  REQUIRE(right_triangle_path(B{}) == b_to_c(B{}));
}

struct Either {
  template <typename T, typename U>
  using Of = P<T, U>;

  template <typename Fn, typename Gn>
  static auto bimap(Fn f, Gn g) {
    using T = Dom<Fn>;
    using U = Dom<Gn>;
    using X = Cod<Fn>;
    using Y = Cod<Gn>;
    using TorU = S<T, U>;
    using XorY = S<X, Y>;

    return [f, g](TorU t_or_u) -> XorY {
      if (t_or_u.index() == 0)
        return inject_l<X, Y>(
            std::invoke(f, std::get<0>(t_or_u)));
      else
        return inject_r<X, Y>(
            std::invoke(g, std::get<1>(t_or_u)));
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

TEST_CASE("S is functorial in the left- and right-position.") {
  // clang-format off
  auto actual_ab = std::vector<S<A, B>>{
      inject_l<A, B>(A{}),
      inject_r<A, B>(B{})
    };
  // clang-format on

  for (auto &x : actual_ab) {

    auto bimap_l_gf = compose(
        Either::bimap(g, id<B>), Either::bimap(f, id<B>));

    REQUIRE(
        bimap_l_gf(x) == Either::bimap(compose(g, f), id<B>)(x));

    auto bimap_r_hg = compose(
        Either::bimap(id<A>, h), Either::bimap(id<A>, g));

    REQUIRE(
        bimap_r_hg(x) == Either::bimap(id<A>, compose(h, g))(x));

    REQUIRE(Either::bimap(id<A>, id<B>)(x) == id<S<A, B>>(x));
  }
}

TEST_CASE("Coproduct of functions as expected") {

  // coprod(f, h) : S<A, C> $→$ S<B, D>
  REQUIRE(std::get<0>(coprod(f, h)(inject_l<A, C>(A{}))) == B{});
  REQUIRE(std::get<1>(coprod(f, h)(inject_r<A, C>(C{}))) == D{});

  SECTION("Also works on the diagonal") {
    // gf : A $→$ C
    auto gf = compose(g, f);
    // f_plus_gf : S<A, A> $→$ S<B, C>
    auto f_plus_gf = coprod(f, gf);

    REQUIRE(std::get<0>(f_plus_gf(inject_l<A, A>(A{}))) == B{});
    REQUIRE(std::get<1>(f_plus_gf(inject_r<A, A>(A{}))) == C{});
  }
}

// ........................................................ f]]]3
// Coproduct associator ................................... f[[[3

template <typename T, typename U, typename V>
auto coassociator_fd(S<T, S<U, V>> tl_ulv) -> S<S<T, U>, V> {
  if (tl_ulv.index() == 0) {
    if constexpr (std::is_same_v<T, Never>) {
      throw std::bad_variant_access();
    } else {
      return inject_l<S<T, U>, V>(std::get<0>(tl_ulv));
    }
  } else {
    auto &ulv = std::get<1>(tl_ulv);
    if (ulv.index() == 0) {
      if constexpr (std::is_same_v<U, Never>) {
        throw std::bad_variant_access();
      } else {
        return inject_l<S<T, U>, V>(std::get<0>(ulv));
      }
    } else {
      if constexpr (std::is_same_v<V, Never>) {
        throw std::bad_variant_access();
      } else {
        return inject_r<S<T, U>, V>(std::get<1>(ulv));
      }
    }
  }
}

template <typename T, typename U, typename V>
auto coassociator_rv(S<S<T, U>, V> tlu_lv) -> S<T, S<U, V>> {
  if (tlu_lv.index() == 0) {
    auto &tlu = std::get<0>(tlu_lv);
    if (tlu.index() == 0) {
      if constexpr (std::is_same_v<T, Never>) {
        throw std::bad_variant_access();
      } else {
        return inject_l<T, S<U, V>>(std::get<0>(tlu));
      }
    } else {
      if constexpr (std::is_same_v<U, Never>) {
        throw std::bad_variant_access();
      } else {
        return inject_r<T, S<U, V>>(
            inject_l<U, V>(std::get<1>(tlu)));
      }
    }
  } else {
    if constexpr (std::is_same_v<V, Never>) {
      throw std::bad_variant_access();
    } else {
      return inject_r<T, S<U, V>>(
          inject_r<U, V>(std::get<1>(tlu_lv)));
    }
  }
}

TEST_CASE(
    "coassociator_fd and coassociator_rv are mutually "
    "inverse.") {
  // clang-format off
  auto coassociator_fd_rv = compose(
        coassociator_rv<A, B, C>,
        coassociator_fd<A, B, C>
      );
  auto coassociator_rv_fd = compose(
        coassociator_fd<A, B, C>,
        coassociator_rv<A, B, C>
      );
  // clang-format on

  auto a_bc = inject_l<A, S<B, C>>(A{});
  REQUIRE(coassociator_fd_rv(a_bc) == id<S<A, S<B, C>>>(a_bc));
  auto ab_c = inject_r<S<A, B>, C>(C{});
  REQUIRE(coassociator_rv_fd(ab_c) == id<S<S<A, B>, C>>(ab_c));
}

TEST_CASE("Associator diagram for coproduct") {
  // clang-format off
  auto start_vals = std::vector<S<A, S<B, S<C, D>>>>{
    inject_l<A, S<B, S<C, D>>>(A{}),
    inject_r<A, S<B, S<C, D>>>(inject_l<B, S<C, D>>(B{})),
    inject_r<A, S<B, S<C, D>>>(inject_r<B, S<C, D>>(inject_l<C, D>(C{}))),
    inject_r<A, S<B, S<C, D>>>(inject_r<B, S<C, D>>(inject_r<C, D>(D{})))
  };

  auto cw_path = compose(
        coassociator_fd<S<A, B>, C, D>,
        coassociator_fd<A, B, S<C, D>>
      );

  auto ccw_path = compose(
        coprod(coassociator_fd<A, B, C>, id<D>),
        coassociator_fd<A, S<B, C>, D>,
        coprod(id<A>, coassociator_fd<B, C, D>)
      );
  // clang-format on
  for (auto &each : start_vals)
    REQUIRE(ccw_path(each) == cw_path(each));
};

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
auto l_counitor_fw(S<Never, T> just_t) -> T {
  return std::get<1>(just_t);
}

template <typename T>
auto l_counitor_rv(T t) -> S<Never, T> {
  return inject_r<Never, T>(t);
}

template <typename T>
auto r_counitor_fw(S<T, Never> just_t) -> T {
  return std::get<0>(just_t);
}

template <typename T>
auto r_counitor_rv(T t) -> S<T, Never> {
  return inject_l<T, Never>(t);
}

TEST_CASE("_fw and _rv are mutual inverses for L-/R-counitor") {
  auto ra = inject_r<Never, A>(A{});
  auto la = inject_l<A, Never>(A{});

  // clang-format off
  REQUIRE(
      compose(l_counitor_rv<A>, l_counitor_fw<A>)(ra) ==
          id<S<Never, A>>(ra)
  );
  REQUIRE(
      compose(l_counitor_fw<A>, l_counitor_rv<A>)(A{}) ==
          id<A>(A{})
  );

  REQUIRE(
      compose(r_counitor_rv<A>, r_counitor_fw<A>)(la) ==
          id<S<A, Never>>(la)
  );
  REQUIRE(compose(r_counitor_fw<A>, r_counitor_rv<A>)(A{}) ==
          id<A>(A{})
  );
  // clang-format on
}

TEST_CASE("Unitor diagram for coproduct") {
  auto a_or_rb = inject_l<A, S<Never, B>>(A{});

  // clang-format off
  auto cw_path = compose(
        coprod(r_counitor_fw<A>, id<B>),
        coassociator_fd<A, Never, B>
      );
  auto ccw_path = coprod(id<A>, l_counitor_fw<B>);
  // clang-format on

  coassociator_fd<A, Never, B>(a_or_rb);

  REQUIRE(cw_path(a_or_rb) == ccw_path(a_or_rb));
}

// ........................................................ f]]]3
// Coproduct symmetric braiding ........................... f[[[3

template <typename T, typename U>
auto cobraid(S<T, U> t_or_u) -> S<U, T> {
  if (t_or_u.index() == 0)
    return inject_r<U, T>(std::get<0>(t_or_u));
  else
    return inject_l<U, T>(std::get<1>(t_or_u));
}

TEST_CASE("Braiding of coproduct is self-inverse") {
  auto ab = inject_l<A, B>(A{});
  REQUIRE(cobraid(cobraid(ab)) == id<S<A, B>>(ab));
}

TEST_CASE("Braiding diagram 1 for coproduct") {
  auto start =
      compose(inject_l<S<A, B>, C>, inject_r<A, B>)(B{});

  // clang-format off
  auto cw_path = compose(
        coprod(cobraid<C, A>, id<B>),
        coassociator_fd<C, A, B>,
        cobraid<S<A, B>, C>
      );
  auto ccw_path = compose(
        coassociator_fd<A, C, B>,
        coprod(id<A>, cobraid<B, C>),
        coassociator_rv<A, B, C>
      );
  // clang-format on

  REQUIRE(cw_path(start) == ccw_path(start));
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
auto distributor_fw(S<P<T, X>, P<U, X>> tx_ux) -> P<S<T, U>, X> {
  if (tx_ux.index() == 0) {
    auto [t, x] = std::get<0>(tx_ux);
    return {inject_l<T, U>(t), x};
  } else {
    auto [u, x] = std::get<1>(tx_ux);
    return {inject_r<T, U>(u), x};
  }
}

template <typename T, typename U, typename X>
auto distributor_rv(P<S<T, U>, X> t_or_u_and_x)
    -> S<P<T, X>, P<U, X>> {
  auto [t_u, x] = t_or_u_and_x;
  if (t_u.index() == 0)
    return inject_l<P<T, X>, P<U, X>>({std::get<0>(t_u), x});
  else
    return inject_r<P<T, X>, P<U, X>>({std::get<1>(t_u), x});
}

TEST_CASE("Distributor is an isomorphism") {

  auto values_fw_rv = std::vector<S<P<A, C>, P<B, C>>>{
      inject_l<P<A, C>, P<B, C>>({A{}, C{}}),
      inject_r<P<A, C>, P<B, C>>({B{}, C{}})};

  auto fw_rv =
      compose(distributor_rv<A, B, C>, distributor_fw<A, B, C>);

  for (auto each : values_fw_rv) {
    REQUIRE(fw_rv(each) == id<S<P<A, C>, P<B, C>>>(each));
  }

  auto values_rv_fw = std::vector<P<S<A, B>, C>>{
      {inject_l<A, B>(A{}), C{}}, {inject_r<A, B>(B{}), C{}}};

  auto rv_fw =
      compose(distributor_fw<A, B, C>, distributor_rv<A, B, C>);

  for (auto each : values_rv_fw) {
    REQUIRE(rv_fw(each) == id<P<S<A, B>, C>>(each));
  }
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

// clang-format off
// LaTeX version:
// TEST_CASE("$\\ttf(\\ttT_1, \\ttT_2, \\ttT_3, …, \\ttT_n) ≅ \\ttf(\\std{tuple}\\Targ{\\ttT_1, \\ttT_2, \\ttT_3, …, \\ttT_n})$") {
// clang-format on
TEST_CASE("f(T₁, T₂, T₃, …, Tₙ) ≅ f(tuple<T₁, T₂, T₃, …, Tₙ>)") {
  auto f = [](A, B, C) -> D { return D{}; };

  REQUIRE(
      to_unary(f)(std::tuple<A, B, C>{}) == f(A{}, B{}, C{}));

  REQUIRE(
      to_n_ary(to_unary(f))(A{}, B{}, C{}) == f(A{}, B{}, C{}));
}

// ........................................................ f]]]2
// OP<T> functor fixpoint ................................. f[[[2

template <template <typename> class F>
struct Fix : F<Fix<F>> {
  explicit Fix(F<Fix<F>> f) : F<Fix<F>>(f) {}
};

template <template <typename> class F>
Fix<F> in(F<Fix<F>> f) {
  return Fix<F>{f};
}

template <template <typename> class F>
F<Fix<F>> out(Fix<F> f) {
  return f;
}

template <typename T>
using O = std::optional<T>;

template <typename T>
struct OP {
  template <typename U>
  using Fst = O<P<std::shared_ptr<U>, T>>;
};

template <typename T>
using List = Fix<OP<T>::template Fst>;

template <typename T>
auto snoc(List<T> l, T t) -> List<T> {
  return in<OP<T>::template Fst>(
      std::make_pair(std::make_shared<List<T>>(l), t));
}

template <typename Lst>
using list_element_type = typename std::remove_reference<
    decltype(*out(std::declval<Lst>()))>::type::second_type;

static_assert(std::is_same_v<list_element_type<List<int>>, int>);

template <typename Lst>
auto to_vector(const Lst l)
    -> std::vector<list_element_type<Lst>> {
  using T = list_element_type<Lst>;
  static_assert(std::is_same_v<Lst, List<T>>);

  std::vector<T> output;
  auto outermost = out(l);
  while (outermost != std::nullopt) {
    auto [tail, head] = outermost.value();
    output.push_back(head);
    outermost = out(*tail);
  }

  std::reverse(output.begin(), output.end());

  return output;
}

template <typename T>
auto to_snoclist(const std::vector<T> &vec) -> List<T> {

  List<T> accumulator = in<OP<T>::template Fst>(std::nullopt);

  for (auto it = vec.cbegin(); it != vec.cend(); ++it) {
    accumulator = snoc(accumulator, *it);
  }

  return accumulator;
}

template <typename Lst, typename T = list_element_type<Lst>>
auto operator==(Lst const &lhs, Lst const &rhs) -> bool {
  auto l = out(lhs);
  auto r = out(rhs);

  while (l.has_value() && r.has_value()) {
    auto [lhs_tail, lhs_val] = l.value();
    auto [rhs_tail, rhs_val] = r.value();
    if (!(lhs_val == rhs_val)) {
      return false;
    }
    l = *lhs_tail;
    r = *rhs_tail;
  }

  // If one of a or b still holds value at this point, then lhs
  // and rhs are different lengths and are not equal.
  return !l.has_value() && !r.has_value();
}

template <typename T>
auto nil = in<OP<T>::template Fst>(std::nullopt);

TEST_CASE(
    "Arbitrary nested optional-pairs isomorphic to "
    "lists") {

  auto list_as = snoc(snoc(snoc(nil<A>, A{}), A{}), A{});
  auto vec_as = std::vector{A{}, A{}, A{}};

  auto list_ints = snoc(snoc(snoc(nil<int>, 1), 2), 3);
  auto vec_ints = std::vector{1, 2, 3};

  REQUIRE(to_vector(list_as) == vec_as);
  REQUIRE(to_vector(list_ints) == vec_ints);

  REQUIRE(list_as == to_snoclist(vec_as));
  REQUIRE(list_ints == to_snoclist(vec_ints));

  REQUIRE(to_vector(to_snoclist(std::vector{1, 2, 3})) ==
          std::vector{1, 2, 3});
}

// List<T>-catamorphisms .................................. f[[[3

template <typename T>
struct ListF {

  template <typename U>
  using Of = typename OP<T>::template Fst<U>;

  template <typename Fn>
  static auto fmap(Fn fn) {
    return [fn](auto op)
               -> typename OP<T>::template Fst<Cod<Fn>> {
      if (op.has_value()) {
        auto [left, right] = op.value();
        auto result = fn(*left);
        return std::make_optional(std::make_pair(
            std::make_shared<decltype(result)>(result), right));
      } else {
        return std::nullopt;
      }
    };
  }

  template <typename Carrier>
  using Alg =
      Hom<typename OP<T>::template Fst<Carrier>, Carrier>;
};

ListF<int>::Alg<int> sum_alg = [](auto op) -> int {
  if (op.has_value()) {
    auto [l, r] = op.value();
    return *l + r;
  } else {
    return 0;
  }
};

auto len_alg = [](auto op) -> int {
  if (op.has_value()) {
    auto [l, r] = op.value();
    return *l + 1;
  } else {
    return 0;
  }
};

template <typename Carrier, typename Elem>
auto list_cata(typename ListF<Elem>::template Alg<Carrier> alg)
    -> Hom<List<Elem>, Carrier> {
  return [alg](List<Elem> ts) {
    return alg(ListF<Elem>::fmap(list_cata<Carrier, Elem>(alg))(
        out(ts)));
  };
}

TEST_CASE(
    "Testing catamorphisms with sum algebra on integer "
    "lists, and length algebras on integer- and "
    "A-lists") {
  auto list_ints = snoc(snoc(snoc(snoc(nil<int>, 1), 2), 3), 4);

  auto sum_int_list = list_cata<int, int>(sum_alg);
  auto sum = sum_int_list(list_ints);
  REQUIRE(sum == 1 + 2 + 3 + 4);

  auto len_int_list = list_cata<int, int>(len_alg);
  auto len = len_int_list(list_ints);
  REQUIRE(len == 4);

  auto list_as = snoc(snoc(nil<A>, A{}), A{});
  auto len_a_list = list_cata<int, A>(len_alg);
  REQUIRE(len_a_list(list_as) == 2);
}

// ........................................................ f]]]3
// ........................................................ f]]]2
// ........................................................ f]]]1
