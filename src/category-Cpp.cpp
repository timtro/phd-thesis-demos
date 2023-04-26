// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65
#include <catch2/catch.hpp>
#include <cctype>
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
  REQUIRE(compose(h, g, f)(A{}) == compose(h, compose(g, f))(A{}));
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
    compose(Const<A>::fmap(g), Const<A>::fmap(f))(A{})
      ==
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

TEST_CASE("ptr is functorial 'up to natural transformation'") {
  auto a_ptr = std::make_shared<A>();

  // clang-format off
  REQUIRE(
    *compose(sptr::map(g), sptr::map(f))(a_ptr)
        ==
            *sptr::map(compose(g, f))(a_ptr)
  );
  // clang-format on

  REQUIRE(*sptr::map(id<A>)(a_ptr) == *id<sptr::Of<A>>(a_ptr));

  REQUIRE(id<sptr::Of<A>>(a_ptr) == a_ptr);

  // Note that we have to dereference for the test.
  // This is not a functor without
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
auto prod(Fn fn, Gn gn) -> Hom<P<T, U>, P<X, Y>> {
  return [fn, gn](P<T, U> tu) -> P<X, Y> {
    auto [t, u] = tu;
    return {fn(t), gn(u)};
  };
}

TEST_CASE("P (via prod) is functorial in both factors.") {
  auto ab = P<A, B>{};

  // clang-format off
  REQUIRE(
    compose(prod(g, id<B>), prod(f, id<B>))(ab)
      ==
        prod(compose(g, f), id<B>)(ab)
  );

  REQUIRE(
    compose(prod(id<A>, h), prod(id<A>, g))(ab)
      ==
        prod(id<A>, compose(h, g))(ab)
  );
  // clang-format on

  REQUIRE(prod(id<A>, id<B>)(ab) == id<P<A, B>>(ab));
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

TEST_CASE("Associator diagram for P") {
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

struct I { // monoidal unit for P
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
  static auto fmap(Fn fn) -> Hom<Hom<T, Dom<Fn>>, Hom<T, Cod<Fn>>> {
    return [fn](auto gn) { return compose(fn, gn); };
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

// LaTeX version:
// TEST_CASE("Commutativity of $\eqref{cd:cpp-exponential}$") {
TEST_CASE("Commutativity of $\\eqref{cd:cpp-exponential}$") {
  auto ab = P<A, B>{};
  auto k = [](P<A, B>) -> C { return {}; };

  auto cw = compose(ev<B, C>, prod(pcurry(k), id<B>));

  REQUIRE(cw(ab) == k(ab));
}

// clang-format off
template <typename Fn,             typename T = Dom<Fn>,
          typename UtoV = Cod<Fn>, typename U = Dom<UtoV>,
                                   typename V = Cod<UtoV>>
// clang-format on
auto puncurry(Fn fn) -> Hom<P<T, U>, V> {
  return [fn](P<T, U> p) -> V { return fn(p.first)(p.second); };
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
      fanout(compose(pcurry(cb_to_a), proj_l<C, B>), proj_r<C, B>));

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

TEST_CASE(
    "(S, coprod) is functorial in the left- and "
    "right-position.") {
  // clang-format off
  auto actual_ab = std::vector<S<A, B>>{
      inject_l<A, B>(A{}),
      inject_r<A, B>(B{})
    };

  for (auto &x : actual_ab) {
    REQUIRE(
      compose(coprod(g, id<B>), coprod(f, id<B>))(x)
          ==
              coprod(compose(g, f), id<B>)(x)
    );

    REQUIRE(
      compose(coprod(id<A>, h), coprod(id<A>, g))(x)
          ==
              coprod(id<A>, compose(h, g))(x)
    );
    // clang-format on
    REQUIRE(coprod(id<A>, id<B>)(x) == id<S<A, B>>(x));
  }
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

TEST_CASE("S is functorial in the left- and right-position.") {
  // clang-format off
  auto actual_ab = std::vector<S<A, B>>{
      inject_l<A, B>(A{}),
      inject_r<A, B>(B{})
    };
  // clang-format on

  for (auto &x : actual_ab) {
    auto bimap_l_gf =
        compose(Either::bimap(g, id<B>), Either::bimap(f, id<B>));

    REQUIRE(
        bimap_l_gf(x) == Either::bimap(compose(g, f), id<B>)(x));

    auto bimap_r_hg =
        compose(Either::bimap(id<A>, h), Either::bimap(id<A>, g));

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

TEST_CASE(
    "coassociator_fd and coassociator_rv are mutually "
    "inverse.") {
  // clang-format off
  auto associator_co_fd_rv = compose(
        associator_co_rv<A, B, C>,
        associator_co_fd<A, B, C>
      );
  auto associator_co_rv_fd = compose(
        associator_co_fd<A, B, C>,
        associator_co_rv<A, B, C>
      );
  // clang-format on

  auto a_bc = inject_l<A, S<B, C>>(A{});
  REQUIRE(associator_co_fd_rv(a_bc) == id<S<A, S<B, C>>>(a_bc));
  auto ab_c = inject_r<S<A, B>, C>(C{});
  REQUIRE(associator_co_rv_fd(ab_c) == id<S<S<A, B>, C>>(ab_c));
}

TEST_CASE("Associator diagram for coproduct") {
  // clang-format off
  // All four values in S<A, S<B, S<C, D>>>:
  auto start_vals = std::vector<S<A, S<B, S<C, D>>>>{
    inject_l<A, S<B, S<C, D>>>(
             A{}
    ),
    inject_r<A, S<B, S<C, D>>>(
         inject_l<B, S<C, D>>(
                  B{}
      )
    ),
    inject_r<A, S<B, S<C, D>>>(
         inject_r<B, S<C, D>>(
              inject_l<C, D>(
                       C{}
        )
      )
    ),
    inject_r<A, S<B, S<C, D>>>(
         inject_r<B, S<C, D>>(
              inject_r<C, D>(
                          D{}
        )
      )
    )
  };

  auto cw_path = compose(
        associator_co_fd<S<A, B>, C, D>,
        associator_co_fd<A, B, S<C, D>>
      );

  auto ccw_path = compose(
        coprod(associator_co_fd<A, B, C>, id<D>),
        associator_co_fd<A, S<B, C>, D>,
        coprod(id<A>, associator_co_fd<B, C, D>)
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

TEST_CASE("_fw and _rv are mutual inverses for L-/R-counitor") {
  auto ra = inject_r<Never, A>(A{});
  auto la = inject_l<A, Never>(A{});

  // clang-format off
  REQUIRE(
      compose(l_unitor_co_rv<A>, l_unitor_co_fw<A>)(ra) ==
          id<S<Never, A>>(ra)
  );
  REQUIRE(
      compose(l_unitor_co_fw<A>, l_unitor_co_rv<A>)(A{}) ==
          id<A>(A{})
  );

  REQUIRE(
      compose(r_unitor_co_rv<A>, r_unitor_co_fw<A>)(la) ==
          id<S<A, Never>>(la)
  );
  REQUIRE(compose(r_unitor_co_fw<A>, r_unitor_co_rv<A>)(A{}) ==
          id<A>(A{})
  );
  // clang-format on
}

TEST_CASE("Unitor diagram for coproduct") {
  auto a_or_rb = inject_l<A, S<Never, B>>(A{});

  // clang-format off
  auto cw_path = compose(
        coprod(r_unitor_co_fw<A>, id<B>),
        associator_co_fd<A, Never, B>
      );
  auto ccw_path = coprod(id<A>, l_unitor_co_fw<B>);
  // clang-format on

  associator_co_fd<A, Never, B>(a_or_rb);

  REQUIRE(cw_path(a_or_rb) == ccw_path(a_or_rb));
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

TEST_CASE("Braiding of coproduct is self-inverse") {
  // clang-format off
  auto ab = std::vector<S<A, B>>{
    inject_l<A, B>(A{}),
    inject_r<A, B>(B{})
  };
  //clang-format on

  for (auto each: ab)
    REQUIRE(braid_co(braid_co(each)) == id<S<A, B>>(each));
}

TEST_CASE("Braiding diagram 1 for coproduct") {
  // clang-format off
  auto start_vals = std::vector<S<S<A, B>, C>>{
    inject_l<S<A, B>, C>(
      inject_l<A, B>(
               A{}
      )
    ),
    inject_l<S<A, B>, C>(
      inject_r<A, B>(
                  B{}
      )
    ),
    inject_r<S<A, B>, C>(
                      C{}
    ),
  };

  auto cw_path = compose(
        coprod(braid_co<C, A>, id<B>),
        associator_co_fd<C, A, B>,
        braid_co<S<A, B>, C>
      );
  auto ccw_path = compose(
        associator_co_fd<A, C, B>,
        coprod(id<A>, braid_co<B, C>),
        associator_co_rv<A, B, C>
      );
  // clang-format on
  for (auto each : start_vals)
    REQUIRE(cw_path(each) == ccw_path(each));
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

TEST_CASE("expand and factorise are mutually inverse") {
  // clang-format off
  SECTION("in the forward then reverse direction") {
    auto values = std::vector<S<P<A, C>, P<B, C>>>{
        inject_l<P<A, C>, P<B, C>>({A{}, C{}}),
        inject_r<P<A, C>, P<B, C>>({B{}, C{}})};

    auto fw_rv = compose(expand<A, B, C>, factorise<A, B, C>);

    for (auto each : values) {
      REQUIRE(fw_rv(each) == id<S<P<A, C>, P<B, C>>>(each));
    }
  }

  SECTION("and in the opposite direction") {
    auto values = std::vector<P<S<A, B>, C>>{
        {inject_l<A, B>(A{}), C{}},
        {inject_r<A, B>(B{}), C{}}
      };

    auto rv_fw = compose(factorise<A, B, C>, expand<A, B, C>);

    for (auto each : values)
      REQUIRE(rv_fw(each) == id<P<S<A, B>, C>>(each));
  }
  // clang-format on
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

  REQUIRE(to_unary(f)(std::tuple<A, B, C>{}) == f(A{}, B{}, C{}));

  REQUIRE(to_n_ary(to_unary(f))(A{}, B{}, C{}) == f(A{}, B{}, C{}));
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
  using Left = S<I, P<std::shared_ptr<U>, T>>;

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
constexpr bool has_pair(S<I, P<T, U>> const &i_or_val) {
  return i_or_val.index() == 1;
}
// ........................................................ f]]]4

template <typename T>
using SnocList = Mu<MP<T>::template Left>;

template <typename T>
auto nil = in<MP<T>::template Left>(I{});

template <typename T>
auto snoc(SnocList<T> lst, T t) -> SnocList<T> {
  return in<MP<T>::template Left>(
      P{std::make_shared<SnocList<T>>(lst), t});
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

TEST_CASE("Building `List`s") {

  auto list_ints = snoc(snoc(snoc(nil<int>, 1), 2), 3);
  auto another_list_ints =
      snoc(snoc(snoc(snoc(nil<int>, 1), 2), 3), 4);

  REQUIRE(std::is_same_v<SnocList<int>, decltype(list_ints)>);
  REQUIRE(std::is_same_v<snoclist_element_type<decltype(list_ints)>,
      int>);

  REQUIRE(list_ints == list_ints);
  // NB: require_FALSE:
  REQUIRE_FALSE(list_ints == another_list_ints);
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

  SnocList<T> accumulator = in<MP<T>::template Left>(I{});

  for (auto it = vec.cbegin(); it != vec.cend(); ++it) {
    accumulator = snoc(accumulator, *it);
  }

  return accumulator;
}

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

  REQUIRE(to_vector(to_snoclist(vec_ints)) == vec_ints);
  REQUIRE(to_snoclist(to_vector(list_ints)) == list_ints);
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

      return coprod(id<I>, prod(sptr::map(fn), id<Elem>))(i_or_p);
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

auto sum_alg = [](auto op) -> int {
  auto global_0 = [](I) -> int { return 0; };

  auto sum_pair = [](P<std::shared_ptr<int>, int> p) -> int {
    return *proj_l(p) + proj_r(p);
  };

  return fanin(global_0, sum_pair)(op);
  //     \______________________/
  //    $\vec{0} ▽ \ttName{sum\_pair}$
};

TEST_CASE("sum algebra on integer lists is as expected") {

  auto list_ints = snoc(snoc(snoc(snoc(nil<int>, 1), 2), 3), 4);
  auto sum_int_list = SnocF<int>::cata<int>(sum_alg);

  REQUIRE(sum_int_list(list_ints) == 1 + 2 + 3 + 4);
}


auto len_alg = [](auto op) -> int {
  // We don not care about the list-element type, so we deduce it:
  using ElemT = maybe_pair_element_t<decltype(op)>;

  auto global_0 = [](I) -> int { return 0; };
  auto add_one = [](P<std::shared_ptr<int>, ElemT> p) -> int {
    return *proj_l(p) + 1;
  };

  return fanin(global_0, add_one)(op);
};

TEST_CASE("len_alg-catamorphism is as expected…") {

  SECTION("… on an integer list") {
    auto list_ints = snoc(snoc(snoc(snoc(nil<int>, 1), 2), 3), 4);
    auto len_int_list = SnocF<int>::cata<int>(len_alg);

    REQUIRE(len_int_list(list_ints) == 4);
  }

  SECTION("… on an A-list") {
    auto list_as = snoc(snoc(nil<A>, A{}), A{});
    auto len_a_list = SnocF<A>::cata<int>(len_alg);

    REQUIRE(len_a_list(list_as) == 2);
  }
}

// ........................................................ f]]]3
// ........................................................ f]]]2
// ........................................................ f]]]1
