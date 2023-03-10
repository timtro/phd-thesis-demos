// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65
#include <catch2/catch.hpp>
#include <type_traits>

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

template <template <typename> typename F>
struct Functor {
  template <typename X>
  using omap = F<X>;

  template <typename Fn>
  static auto fmap(Fn) -> Hom<F<Dom<Fn>>, F<Cod<Fn>>>;
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
// Cpp structures ......................................... f[[[1
// map : (A → B) → (vector<A> → vector<B>)
// template <typename A, typename B>
// auto vec_map(std::function<B(A)> f) {
//   return [f](std::vector<A> as) -> std::vector<B> {
//     std::vector<B> bs;
//     bs.reserve(as.size());
//
//     std::transform(
//         cbegin(as), cend(as), std::back_inserter(bs), f);
//     return bs;
//   };
// }

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
// std::vector functor with free func and Functor<>. ...... f[[[2

template <typename T>
using Vector = std::vector<T>;

struct VectorF : public Functor<Vector> {
  template <typename Fn>
  static auto fmap(Fn f) {
    using T = Dom<Fn>;
    using U = Cod<Fn>;
    return [f](Vector<T> t_s) {
      std::vector<U> u_s;
      u_s.reserve(t_s.size());

      std::transform(
          cbegin(t_s), cend(t_s), std::back_inserter(u_s), f);
      return u_s;
    };
  };
};

TEST_CASE(
    "Similar to the previous example using fmap on "
    "std::vector, we can use the Functor<Vector> "
    "constructed above to achieve the same ends.") {

  auto a_s = std::vector{A{}, A{}, A{}};
  auto vec_f = VectorF::fmap(f);
  auto vec_g = VectorF::fmap(g);
  auto vec_gf = VectorF::fmap<Hom<A, C>>(compose(g, f));

  REQUIRE(compose(vec_g, vec_f)(a_s) == vec_gf(a_s));
}

template <typename Fn>
auto vec_fmap(Fn f)
    -> Hom<std::vector<Dom<Fn>>, std::vector<Cod<Fn>>> {
  using T = Dom<Fn>;
  using U = Cod<Fn>;
  return [f](std::vector<T> t_s) {
    std::vector<U> u_s;
    u_s.reserve(t_s.size());

    std::transform(
        cbegin(t_s), cend(t_s), std::back_inserter(u_s), f);
    return u_s;
  };
}

TEST_CASE(
    "Given a function f : A → B and a function template "
    "fmap, then fmap(f) should should produce a function "
    "object that maps f over a std::vector<A> to give a "
    "std::vector<B>") {

  auto a_s = std::vector{A{}, A{}, A{}};
  auto vec_f = vec_fmap(f);
  auto vec_g = vec_fmap(g);
  auto vec_gf = vec_fmap<Hom<A, C>>(compose(g, f));

  REQUIRE(compose(vec_g, vec_f)(a_s) == vec_gf(a_s));
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

// ........................................................ f]]]1
