// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65
#include <catch2/catch.hpp>
#include <optional>
#include <type_traits>
#include <variant>

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

template <typename Derived,
    template <typename, typename...> typename TypeTemplate>
struct Functor {
  template <typename T>
  using Of = TypeTemplate<T>;

  template <typename Fn>
  static auto fmap(Fn f) -> Hom<Of<Dom<Fn>>, Of<Cod<Fn>>> {
    return Derived::fmap(f);
  }
};

template <typename Derived,
    template <typename, typename...> typename TypeTemplate>
struct Bifunctor {
  template <typename L, typename R>
  using Of = TypeTemplate<L, R>;

  template <typename Fn, typename Gn>
  static auto bimap(Fn f, Gn g) {
    return Derived::bimap(f, g);
  }
};

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

TEST_CASE("Sould be able to compose with PMFs", //
    "[compose], [interface]") {
  const auto a = A{};
  Getable getable_a{a};
  auto fog = compose(id<A>, &Getable<A>::get);
  REQUIRE(fog(&getable_a) == a);
}

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

struct Foo {
  D d_returner(A, B, C) { return {}; }
};

TEST_CASE("PMFs should curry", //
    "[curry], [non-variadic], [interface]") {
  Foo foo;
  auto foo_d_returner = curry(&Foo::d_returner);
  REQUIRE(foo_d_returner(&foo)(A{})(B{})(C{}) == D{});
  //                     ^ Always give pointer to object
}

TEST_CASE(
    "A curried non-variadic function should preserve the "
    "lvalue ref-ness of whatever is returned from the "
    "wrapped function.", //
    "[curry], [non-variadic], [interface]") {
  A a{};
  auto ref_to_a = [&a]() -> A & { return a; };
  REQUIRE(std::is_lvalue_reference<
      decltype(curry(ref_to_a))>::value);
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

struct Id : public Functor<Id, IdType> {
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

struct Optional : public Functor<Optional, std::optional> {
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
};

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
  auto opt_gf = Optional::fmap<Hom<A, C>>(compose(g, f));
  auto opt_idA = Optional::fmap(id<A>);

  REQUIRE(compose(opt_g, opt_f)(a) == opt_gf(a));
  REQUIRE(opt_idA(a) == id<Optional::Of<A>>(a));

  REQUIRE(compose(opt_g, opt_f)(not_a) == opt_gf(not_a));
  REQUIRE(opt_idA(not_a) == id<Optional::Of<A>>(not_a));
}

// ........................................................ f]]]2
// std::vector based List-functor ......................... f[[[2

struct Vector : public Functor<Vector, std::vector> {
  template <typename Fn>
  static auto fmap(Fn f) -> Hom<Of<Dom<Fn>>, Of<Cod<Fn>>> {
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
};

TEST_CASE("Check the functor laws for Vector::fmap") {
  //        Alias for std$∷$vector<A>
  //                 $↓$
  auto a_s = Vector::Of<A>{A{}, A{}, A{}};

  // clang-format off
  // $\Ffmap{\ttF}(\ttg) ∘ \Ffmap{\ttF}(\ttf) = \Ffmap{\ttF}(\ttg ∘ \ttf)$
  REQUIRE(
    compose(Vector::fmap(g), Vector::fmap(f))(a_s)
        == Vector::fmap<Hom<A, C>>(compose(g, f))(a_s));
  // clang-format on

  // $\Ffmap{\ttF}(\ttid⟨-⟩) = \ttid⟨\FOf{\ttf}{-}⟩$
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
struct Const
    : public Functor<Const<T>, Always<T>::template given> {
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
    compose(len<B>, Vector::fmap(f))(a_s) ==
          compose(Const<uint>::fmap(f), len<A>)(a_s)
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

struct Pair : public Bifunctor<Pair, P> {
  template <typename Fn, typename Gn>
  static auto bimap(Fn f, Gn g) {
    return [f, g](auto tu) {
      auto [t, u] = tu;
      return std::pair{f(t), g(u)};
    };
  }
};

TEST_CASE("std::pair is functorial in the left position.") {

  auto ab = Pair::Of<A, B>{};

  // clang-format off
  REQUIRE(
    compose(Pair::bimap(g, id<B>), Pair::bimap(f, id<B>))(ab) ==
              Pair::bimap<Hom<A, C>, Hom<B,B>>(compose(g, f), id<B>)(ab)
  );

  REQUIRE(
    compose(Pair::bimap(id<A>, h), Pair::bimap(id<A>, g))(ab) ==
              Pair::bimap<Hom<A,A>, Hom<B, D>>(id<A>, compose(h, g))(ab)
  );
  // clang-format on

  REQUIRE(Pair::bimap(id<A>, id<B>)(ab) == id<P<A, B>>(ab));
}

template <typename Fn, typename Gn>
auto prod(Fn f, Gn g)
    -> Hom<P<Dom<Fn>, Dom<Gn>>, P<Cod<Fn>, Cod<Gn>>> {
  using TandU = P<Dom<Fn>, Dom<Gn>>;
  using XandY = P<Cod<Fn>, Cod<Gn>>;

  return [f, g](TandU tu) -> XandY {
    auto [t, u] = tu;
    return {f(t), g(u)};
  };
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

TEST_CASE(
    "Product of functions behaves as expected with "
    "respect to  l/rmap.") {
  auto ac = P<A, C>{};
  REQUIRE(prod(f, id<C>)(ac) == Pair::bimap(f, id<C>)(ac));
  REQUIRE(prod(id<A>, h)(ac) == Pair::bimap(id<A>, h)(ac));
}

template <typename Fn, typename Gn>
auto fanout(Fn f, Gn g) -> Hom<Dom<Fn>, P<Cod<Fn>, Cod<Gn>>> {
  using T = Dom<Fn>;
  using U = Cod<Fn>;
  using V = Cod<Gn>;

  static_assert(std::is_same_v<T, Dom<Gn>>);

  return [f, g](T t) -> P<U, V> { return {f(t), g(t)}; };
}

TEST_CASE("Commutativity of $\\eqref{cd:cpp:binary-product}$") {
  auto A_to_B = [](A) { return B{}; };
  auto A_to_C = [](A) { return C{}; };

  auto A_to_BxC = fanout(A_to_B, A_to_C);

  REQUIRE(std::get<0>(A_to_BxC(A{})) == A_to_B(A{}));
  REQUIRE(std::get<1>(A_to_BxC(A{})) == A_to_C(A{}));
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
struct CHom
    : public Functor<CHom<T>, HomFrom<T>::template HomTo> {
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
// ........................................................ f]]]2
// Coproduct bifunctor .................................... f[[[2

template <typename T>
struct Left {
  T value;

  bool operator==(const Left<T> &other) const {
    return value == other.value;
  }
};

template <typename U>
struct Right {
  U value;

  bool operator==(const Right<U> &other) const {
    return value == other.value;
  }
};

struct Never { // Monoidal unit for LeftOrRight.
  Never() = delete;
  Never(const Never &) = delete;
  bool operator==(const Never &) const { return false; }
};

namespace util {
  template <typename T, typename... Types>
  constexpr bool holds_alternative(
      const std::variant<Types...> &v) noexcept {
    if constexpr (
        std::is_same_v<T, Left<Never>> ||
        std::is_same_v<T, Right<Never>>) {
      return false; // We will never see a Never value.
    } else {
      return std::holds_alternative<T>(v);
    }
  }
} // namespace util

template <typename T, typename U>
class LeftOrRight : public std::variant<Left<T>, Right<U>> {
public:
  using std::variant<Left<T>, Right<U>>::variant;

  [[nodiscard]] const T &left() const {
    return std::get<Left<T>>(*this).value;
  }

  [[nodiscard]] const U &right() const {
    return std::get<Right<U>>(*this).value;
  }
};

template <typename T, typename U>
using S = LeftOrRight<T, U>;

template <typename T, typename U>
auto inject_l(T t) -> S<T, U> {
  return Left<T>{t};
}

template <typename T, typename U>
auto inject_r(U t) -> S<T, U> {
  return Right<U>{t};
}

struct Either : public Bifunctor<Either, LeftOrRight> {
  template <typename Fn, typename Gn>
  static auto bimap(Fn f, Gn g) {
    using T = Dom<Fn>;
    using U = Dom<Gn>;
    using X = Cod<Fn>;
    using Y = Cod<Gn>;
    using TorU = S<T, U>;
    using XorY = S<X, Y>;

    return [f, g](TorU t_or_u) -> XorY {
      if (util::holds_alternative<Left<T>>(t_or_u))
        return Left<X>{std::invoke(f, t_or_u.left())};
      else
        return Right<Y>{std::invoke(g, t_or_u.right())};
    };
  };
};

TEST_CASE("P is functorial in the left- and right-position.") {

  auto just_a = inject_l<A, B>(A{});
  auto just_b = inject_r<A, B>(B{});

  // clang-format off
  REQUIRE(
    compose(Either::bimap(g, id<B>), Either::bimap(f, id<B>))(just_a) ==
              Either::bimap<Hom<A, C>, Hom<B,B>>(compose(g, f), id<B>)(just_a)
  );

  REQUIRE(
    compose(Either::bimap(g, id<B>), Either::bimap(f, id<B>))(just_b) ==
              Either::bimap<Hom<A, C>, Hom<B,B>>(compose(g, f), id<B>)(just_b)
  );

  REQUIRE(
    compose(Either::bimap(id<A>, h), Either::bimap(id<A>, g))(just_a) ==
              Either::bimap<Hom<A,A>, Hom<B, D>>(id<A>, compose(h, g))(just_a)
  );

  REQUIRE(
    compose(Either::bimap(id<A>, h), Either::bimap(id<A>, g))(just_b) ==
              Either::bimap<Hom<A,A>, Hom<B, D>>(id<A>, compose(h, g))(just_b)
  );
  // clang-format on

  REQUIRE(Either::bimap(id<A>, id<B>)(just_a) ==
          id<S<A, B>>(just_a));
  REQUIRE(Either::bimap(id<A>, id<B>)(just_b) ==
          id<S<A, B>>(just_b));
}

template <typename Fn, typename Gn>
auto fanin(Fn f, Gn g) -> Hom<S<Dom<Fn>, Dom<Gn>>, Cod<Fn>> {
  using T = Dom<Fn>;
  using U = Dom<Gn>;
  using X = Cod<Fn>;
  using TorU = S<T, U>;

  static_assert(std::is_same_v<X, Cod<Gn>>);

  return [f, g](TorU t_or_u) -> X {
    if (util::holds_alternative<Left<T>>(t_or_u))
      return std::invoke(f, t_or_u.left());
    else
      return std::invoke(g, t_or_u.right());
  };
}

TEST_CASE("fanin as expected") {
  auto A_to_bool = [](A) { return true; };
  auto B_to_bool = [](B) { return false; };

  auto really_a = inject_l<A, B>(A{});
  auto really_b = inject_r<A, B>(B{});

  auto AorB_to_bool = fanin(A_to_bool, B_to_bool);

  REQUIRE(AorB_to_bool(really_a) == true);
  REQUIRE(AorB_to_bool(really_b) == false);
}

// ((A → B), (C → D)) → (A + C → B + D)
template <typename Fn, typename Gn>
auto coprod(Fn f, Gn g)
    -> Hom<S<Dom<Fn>, Dom<Gn>>, S<Cod<Fn>, Cod<Gn>>> {
  using T = Dom<Fn>;
  using U = Dom<Gn>;
  using X = Cod<Fn>;
  using Y = Cod<Gn>;
  using TorU = S<T, U>;
  using XorY = S<X, Y>;

  return [f, g](TorU t_or_u) -> XorY {
    if (util::holds_alternative<Left<T>>(t_or_u))
      return Left<X>{std::invoke(f, t_or_u.left())};
    else
      return Right<Y>{std::invoke(g, t_or_u.right())};
  };
}

TEST_CASE("Coproduct of functions as expected") {

  auto gf = Hom<A, C>{compose(g, f)};

  REQUIRE((coprod(f, gf)(Left<A>{})).left() == B{});
  REQUIRE((coprod(f, gf)(Right<A>{})).right() == C{});

  REQUIRE((coprod(f, h)(Left<A>{})).left() == B{});
  REQUIRE((coprod(f, h)(Right<C>{})).right() == D{});
}

// ........................................................ f]]]2
// Coproduct associator ................................... f[[[2

template <typename T, typename U, typename V>
auto coassociator_fd(S<T, S<U, V>> tl_ulv) -> S<S<T, U>, V> {
  if (util::holds_alternative<Left<T>>(tl_ulv)) {
    if constexpr (std::is_same_v<T, Never>) {
      throw std::bad_variant_access();
    } else {
      return Left<S<T, U>>{Left<T>{tl_ulv.left()}};
    }
  } else {
    auto &ulv = tl_ulv.right();
    if (util::holds_alternative<Left<U>>(ulv)) {
      if constexpr (std::is_same_v<U, Never>) {
        throw std::bad_variant_access();
      } else {
        return Left<S<T, U>>{Right<U>{ulv.left()}};
      }
    } else {
      if constexpr (std::is_same_v<V, Never>) {
        throw std::bad_variant_access();
      } else {
        return Right<V>{ulv.right()};
      }
    }
  }
}

template <typename T, typename U, typename V>
auto coassociator_rv(S<S<T, U>, V> tlu_lv) -> S<T, S<U, V>> {
  if (util::holds_alternative<Left<S<T, U>>>(tlu_lv)) {
    auto &tlu = tlu_lv.left();
    if (util::holds_alternative<Left<T>>(tlu)) {
      if constexpr (std::is_same_v<T, Never>) {
        throw std::bad_variant_access();
      } else {
        return Left<T>{tlu.left()};
      }
    } else {
      if constexpr (std::is_same_v<U, Never>) {
        throw std::bad_variant_access();
      } else {
        return Right<S<U, V>>{Left<U>{tlu.right()}};
      }
    }
  } else {
    if constexpr (std::is_same_v<V, Never>) {
      throw std::bad_variant_access();
    } else {
      return Right<S<U, V>>{Right<V>{tlu_lv.right()}};
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

  auto a_bc = S<A, S<B, C>>{Left<A>{}};
  REQUIRE(coassociator_fd_rv(a_bc) == id<S<A, S<B, C>>>(a_bc));
  auto ab_c = S<S<A, B>, C>{Right<C>{}};
  REQUIRE(coassociator_rv_fd(ab_c) == id<S<S<A, B>, C>>(ab_c));
}

TEST_CASE("Associator diagram for coproduct") {
  auto start = S<A, S<B, S<C, D>>>{};

  // clang-format off
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

  REQUIRE(ccw_path(start) == cw_path(start));
};

// ........................................................ f]]]2
// Corpdocut unitor ....................................... f[[[2

template <typename T>
struct LeftOrRight<T, Never>
    : std::variant<Left<T>, Right<Never>> {
  using std::variant<Left<T>, Right<Never>>::variant;

  LeftOrRight()
      : std::variant<Left<T>, Right<Never>>{Left<T>{}} {}

  LeftOrRight(const LeftOrRight &other)
      : std::variant<Left<T>, Right<Never>>(
            std::in_place_type<Left<T>>, Left<T>{other.left()}) {
  }

  [[nodiscard]] const T &left() const {
    return std::get<Left<T>>(*this).value;
  }

  [[nodiscard]] const T &right() const {
    throw std::bad_variant_access();
  }
};

template <typename T>
struct LeftOrRight<Never, T>
    : std::variant<Left<Never>, Right<T>> {
  using std::variant<Left<Never>, Right<T>>::variant;

  LeftOrRight()
      : std::variant<Left<Never>, Right<T>>{Right<T>{}} {}

  LeftOrRight(const LeftOrRight &other)
      : std::variant<Left<Never>, Right<T>>(
            std::in_place_type<Right<T>>,
            Right<T>{other.right()}) {}

  [[nodiscard]] const T &left() const {
    throw std::bad_variant_access();
  }

  [[nodiscard]] const T &right() const {
    return std::get<Right<T>>(*this).value;
  }
};

template <typename T>
auto l_counitor_fw(S<Never, T> just_t) -> T {
  return just_t.right();
}

template <typename T>
auto l_counitor_rv(T t) -> S<Never, T> {
  return Right<T>{t};
}

template <typename T>
auto r_counitor_fw(S<T, Never> just_t) -> T {
  return just_t.left();
}

template <typename T>
auto r_counitor_rv(T t) -> S<T, Never> {
  return Left<T>{t};
}

TEST_CASE("_fw and _rv are mutual inverses for L-/R-counitor") {
  auto ra = S<Never, A>{Right<A>{}};
  auto la = S<A, Never>{Left<A>{}};

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
  auto a_or_rb = S<A, S<Never, B>>{Left<A>{}};

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

// ........................................................ f]]]2
// Coproduct symmetric braiding ........................... f[[[2

template <typename T, typename U>
auto cobraid(S<T, U> t_or_u) -> S<U, T> {
  if (util::holds_alternative<Left<T>>(t_or_u))
    return S<U, T>{Right<T>{t_or_u.left()}};
  else
    return S<U, T>{Left<U>{t_or_u.right()}};
}

TEST_CASE("Braiding of coproduct is self-inverse") {
  auto ab = S<A, B>{};
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
// ........................................................ f]]]2
// Product distributes over coproduct ..................... f[[[2

// BUG: I am not going to pessimise the `distributor`
//   against T,U,X = Never, because the nesting is just too hard
//   to read for thesis code, and I only need to get the point
//   across.

template <typename T, typename U, typename X>
auto distributor_fw(S<P<T, X>, P<U, X>> tx_ux) -> P<S<T, U>, X>{
  if (util::holds_alternative<Left<P<T, X>>>(tx_ux)) {
    auto [t, x] = tx_ux.left();
    return {{Left<T>{t}}, x};
  } else {
    auto [u, x] = tx_ux.right();
    return {{Right<U>{u}}, x};
  }
}

template <typename T, typename U, typename X>
auto distributor_rv(P<S<T, U>, X> t_or_u_and_x) -> S<P<T, X>, P<U, X>> {
  auto [t_u, x] = t_or_u_and_x;
  if (util::holds_alternative<Left<T>>(t_u))
    return Left<P<T, X>>{{t_u.left(), x}};
  else
    return Right<P<U, X>>{{t_u.right(), x}};
}

TEST_CASE("Distributor is an isomorphism") {

  auto values_fw_rv = std::vector<S<P<A, C>, P<B, C>>>{
    Left<P<A, C>>{{A{}, C{}}},
    Right<P<B, C>>{{B{}, C{}}},
  };

  auto fw_rv = compose(distributor_rv<A, B, C>, distributor_fw<A, B, C>);

  for (auto each: values_fw_rv){
    REQUIRE(fw_rv(each) == id<S<P<A, C>, P<B, C>>>(each));
  }

  auto values_rv_fw = std::vector<P<S<A, B>, C>>{
    {Left<A>{A{}}, C{}},
    {Right<B>{B{}}, C{}}
  };

  auto rv_fw = compose(distributor_fw<A, B, C>, distributor_rv<A, B, C>);

  for (auto each: values_rv_fw){
    REQUIRE(rv_fw(each) == id<P<S<A, B>, C>>(each));
  }

}

// ........................................................ f]]]2
// ........................................................ f]]]1
