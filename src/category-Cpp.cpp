// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65
#include <catch2/catch.hpp>
#include <type_traits>
#include <optional>

#include "test-tools.hpp"
using tst::A; // Tag for unit type
using tst::B; // Tag for unit type
using tst::C; // Tag for unit type
using tst::D; // Tag for unit type
using tst::f; // f : A → B
using tst::g; // g : B → C
using tst::h; // h : C → D

#include "Cpp-arrows.hpp"

using tf::Cod;
using tf::compose;
using tf::curry;
using tf::Dom;
using tf::Doms;
using tf::Hom;
using tf::id;

template <typename Derived,
    template <typename, typename...> typename TypeCtor>
struct Functor {
  template <typename T>
  using Of = TypeCtor<T>;

  template <typename Fn>
  static auto fmap(Fn f) -> Hom<Of<Dom<Fn>>, Of<Cod<Fn>>> {
    return Derived::fmap(f);
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
    REQUIRE(id(a) == A{});
    REQUIRE(id(b) == B{});
  }
  SECTION("… preserve lvalue-refness") {
    REQUIRE(std::is_lvalue_reference<
        decltype(id(std::declval<int &>()))>::value);
  }
  SECTION("… identify rvalues", "[mathematical]") {
    REQUIRE(id(A{}) == A{});
    REQUIRE(id(B{}) == B{});
  }
  SECTION("… preserve rvalue-refness") {
    REQUIRE(std::is_rvalue_reference<
        decltype(id(std::declval<int &&>()))>::value);
  }
}

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
  //                     ^ Always give pointer to object first.
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
  auto IdF_idA = Id::fmap<Hom<A, A>>(id<A>);

  REQUIRE(compose(IdF_g, IdF_f)(a) == IdF_gf(a));
  REQUIRE(IdF_idA(a) == id(a));
}
// ........................................................ f]]]2
// Constant functor ....................................... f[[[2

template <typename>
using ConstT = std::size_t;

struct ConstSizet : public Functor<ConstSizet, ConstT> {
  template <typename Fn>
  static auto fmap(Fn) -> Hom<Of<Dom<Fn>>, Of<Cod<Fn>>> {
    return [](std::size_t x) { return x; };
  };
};
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

TEST_CASE("Basic behavioural tests of Optional::fmap"){
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
  //        Alias for std::vector<A>
  //                 $↓$
  auto a = Optional::Of<A>{A{}};
  auto not_a = Optional::Of<A>{};

  auto opt_f = Optional::fmap(f);
  auto opt_g = Optional::fmap(g);
  auto opt_gf = Optional::fmap<Hom<A, C>>(compose(g, f));
  auto opt_idA = Optional::fmap<Hom<A, A>>(id<A>);

  REQUIRE(compose(opt_g, opt_f)(a) == opt_gf(a));
  REQUIRE(opt_idA(a) == id(a));

  REQUIRE(compose(opt_g, opt_f)(not_a) == opt_gf(not_a));
  REQUIRE(opt_idA(not_a) == id(not_a));
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
  //        Alias for std::vector<A>
  //                 $↓$
  auto a_s = Vector::Of<A>{A{}, A{}, A{}};
  auto vec_f = Vector::fmap(f);
  auto vec_g = Vector::fmap(g);
  auto vec_gf = Vector::fmap<Hom<A, C>>(compose(g, f));
  auto vec_idA = Vector::fmap<Hom<A, A>>(id<A>);

  REQUIRE(compose(vec_g, vec_f)(a_s) == vec_gf(a_s));
  REQUIRE(vec_idA(a_s) == id(a_s));
}
// ........................................................ f]]]2
// std::pair functorial on left and right sides. .......... f[[[2
template <typename F>
auto pair_lmap(F f) {
  return [f](auto tu) {
    auto [t, u] = tu;
    return std::pair{f(t), u};
  };
}

template <typename F>
auto pair_rmap(F f) {
  return [f](auto tu) {
    auto [t, u] = tu;
    return std::pair{t, f(u)};
  };
}

TEST_CASE(
    "std::pair is functorial in either the left or right "
    "position.") {
  GIVEN("A value in std::pair<A, C>") {
    auto ac = std::pair<A, C>{};

    THEN("pair_lmap(f) should act on the left (A) value.") {
      auto f_x_id = pair_lmap(f);
      REQUIRE(f_x_id(ac) == std::pair{B{}, C{}});
    }

    THEN("pair_lmap(f) should act on the right (C) value.") {
      auto id_x_h = pair_rmap(h);
      REQUIRE(id_x_h(ac) == std::pair{A{}, D{}});
    }
  }
}
// ........................................................ f]]]2
// covariant hom functor .................................. f[[[2
template <typename F>
auto chom_map(F f) {
  return [f](auto g) { return compose(f, g); };
}

TEST_CASE("chom_map") {
  auto hom_A_g = chom_map(g);

  REQUIRE(hom_A_g(f)(A{}) == C{});
}
// ........................................................ f]]]2
// Natural Transformations ................................ f[[[2
template <template <typename> typename F,
    template <typename> typename G>
struct NaturalTransformation {
  template <typename T>
  static auto transform(F<T> fa) -> G<T>;
};

struct len
    : public NaturalTransformation<Vector::Of, ConstSizet::Of> {
  template <typename T>
  static std::size_t transform(Vector::Of<T> t_s) {
    return t_s.size();
  }
};

TEST_CASE("Test naturality square for len.") {
  constexpr int actual_length = 5;
  auto a_s = Vector::Of<A>(actual_length);
  REQUIRE(compose(len::transform<B>, Vector::fmap(f))(a_s) ==
          actual_length);
  REQUIRE(compose(ConstSizet::fmap(f), len::transform<A>)(a_s) ==
          actual_length);
}
// ........................................................ f]]]2
// ........................................................ f]]]1
