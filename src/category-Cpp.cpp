// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65
#include <catch2/catch.hpp>
#include <optional>
#include <type_traits>
#include <variant>

#include "test-tools.hpp"
using tst::A; // Tag for unit type
using tst::B; // Tag for unit type
using tst::C; // Tag for unit type
using tst::D; // Tag for unit type
using tst::f; // f : A → B
using tst::g; // g : B → C
using tst::h; // h : C → D
using tst::id;

#include "Cpp-arrows.hpp"

using tf::Cod;
using tf::compose;
using tf::curry;
using tf::Dom;
using tf::Doms;
using tf::Hom;

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

template <typename Derived,
    template <typename, typename...> typename TypeCtor>
struct Bifunctor {
  template <typename L, typename R>
  using Of = TypeCtor<L, R>;

  template <typename Fn>
  static auto lmap(Fn f) {
    return Derived::lmap(f);
  }

  template <typename Fn>
  static auto rmap(Fn f) {
    return Derived::rmap(f);
  }
};

// tfunc test cases ....................................... f[[[1
// TEST_CASE(
//     "Polymorphic identity function should perfectly "
//     "forward and …",
//     "[id], [interface]") {
//   SECTION("… identify lvalues", "[mathematical]") {
//     auto a = A{};
//     auto b = B{};
//     REQUIRE(id(a) == A{});
//     REQUIRE(id(b) == B{});
//   }
//   SECTION("… preserve lvalue-refness") {
//     REQUIRE(std::is_lvalue_reference<
//         decltype(id(std::declval<int &>()))>::value);
//   }
//   SECTION("… identify rvalues", "[mathematical]") {
//     REQUIRE(id(A{}) == A{});
//     REQUIRE(id(B{}) == B{});
//   }
//   SECTION("… preserve rvalue-refness") {
//     REQUIRE(std::is_rvalue_reference<
//         decltype(id(std::declval<int &&>()))>::value);
//   }
// }
//
// TEST_CASE("compose(f) == f ∘ id_A == id_B ∘ f.", //
//     "[compose], [mathematical]") {
//   REQUIRE(compose(f)(A{}) == compose(f, id<A>)(A{}));
//   REQUIRE(compose(f, id<A>)(A{}) == compose(id<B>, f)(A{}));
// }
//
// TEST_CASE("Compose two C-functions", "[compose], [interface]")
// {
//   auto id2 = compose(id<A>, id<A>);
//   REQUIRE(id2(A{}) == A{});
// }
//
// template <typename T>
// struct Getable {
//   T N;
//   Getable(T N) : N(N) {}
//   [[nodiscard]] T get() const { return N; }
// };
//
// TEST_CASE("Sould be able to compose with PMFs", //
//     "[compose], [interface]") {
//   const auto a = A{};
//   Getable getable_a{a};
//   auto fog = compose(id<A>, &Getable<A>::get);
//   REQUIRE(fog(&getable_a) == a);
// }
//
// TEST_CASE(
//     "Return of a composed function should preserve the "
//     "rvalue-refness of the outer function", //
//     "[compose], [interface]") {
//   B b{};
//   auto ref_to_b = [&b](A) -> B & { return b; };
//   auto fog = compose(ref_to_b, id<A>);
//   REQUIRE(std::is_lvalue_reference<decltype(fog(A{}))>::value);
// }
//
// TEST_CASE(
//     "Curried non-variadic functions should bind "
//     "arguments, one at a time, from left to right.", //
//     "[curry], [non-variadic], [interface]") {
//
//   // abcd : (A, B, C, D) → (A, B, C, D)
//   // a_b_c_d : A → B → C → D → (A, B, C, D)
//   auto abcd = [](A a, B b, C c, D d) {
//     return std::tuple{a, b, c, d};
//   };
//   auto a_b_c_d = curry(abcd);
//
//   auto [a, b, c, d] = a_b_c_d(A{})(B{})(C{})(D{});
//   REQUIRE(a == A{});
//   REQUIRE(b == B{});
//   REQUIRE(c == C{});
//   REQUIRE(d == D{});
// }
//
// struct Foo {
//   D d_returner(A, B, C) { return {}; }
// };
//
// TEST_CASE("PMFs should curry", //
//     "[curry], [non-variadic], [interface]") {
//   Foo foo;
//   auto foo_d_returner = curry(&Foo::d_returner);
//   REQUIRE(foo_d_returner(&foo)(A{})(B{})(C{}) == D{});
//   //                     ^ Always give pointer to object
//   first.
// }
//
// TEST_CASE(
//     "A curried non-variadic function should preserve the "
//     "lvalue ref-ness of whatever is returned from the "
//     "wrapped function.", //
//     "[curry], [non-variadic], [interface]") {
//   A a{};
//   auto ref_to_a = [&a]() -> A & { return a; };
//   REQUIRE(std::is_lvalue_reference<
//       decltype(curry(ref_to_a))>::value);
// }
//
// TEST_CASE("Curried functions should…") {
//   auto ABtoC = curry([](A, B) -> C { return {}; });
//   auto BtoC = ABtoC(A{});
//
//   SECTION("… compose with other callables.",
//       "[curry], "
//       "[compose], "
//       "[interface]") {
//     REQUIRE(compose(BtoC, f)(A{}) == C{});
//   }
// }
//
// TEST_CASE(
//     "Currying should work with the C++14 std outfix "
//     "operators") {
//   auto plus = curry(std::plus<int>{});
//   auto increment = plus(1);
//   REQUIRE(increment(0) == 1);
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
  REQUIRE(f(A{}) == compose(f, id)(A{}));
  REQUIRE(f(A{}) == compose(id, f)(A{}));
  REQUIRE(compose(f, id)(A{}) == compose(id, f)(A{}));
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
  auto IdF_idA = Id::fmap<Hom<A, A>>(id);

  REQUIRE(compose(IdF_g, IdF_f)(a) == IdF_gf(a));
  REQUIRE(IdF_idA(a) == id(a));
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
  //        Alias for std::vector<A>
  //                 $↓$
  auto a = Optional::Of<A>{A{}};
  auto not_a = Optional::Of<A>{};

  auto opt_f = Optional::fmap(f);
  auto opt_g = Optional::fmap(g);
  auto opt_gf = Optional::fmap<Hom<A, C>>(compose(g, f));
  auto opt_idA = Optional::fmap<Hom<A, A>>(id);

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
  //        Alias for std$∷$vector<A>
  //                 $↓$
  auto a_s = Vector::Of<A>{A{}, A{}, A{}};

  // $\FOf{\ttF}{\ttA} → \FOf{\ttF}{\ttB}$
  auto vec_f = Vector::fmap(f);

  // $\FOf{\ttF}{\ttB} → \FOf{\ttF}{\ttC}$
  auto vec_g = Vector::fmap(g);

  // $\FOf{\ttF}{\ttA} → \FOf{\ttF}{\ttC}$
  auto vec_gf = Vector::fmap<Hom<A, C>>(compose(g, f));

  // $\Ffmap{\ttF}(\ttg) ∘ \Ffmap{\ttF}(\ttf) = \Ffmap{\ttF}(\ttg
  // ∘ \ttf)$
  REQUIRE(compose(vec_g, vec_f)(a_s) == vec_gf(a_s));

  // vec_id$-$ : $\FOf{\ttF}{-} → \FOf{\ttF}{-}$
  auto vec_idA = Vector::fmap<Hom<A, A>>(id);
  auto vec_idB = Vector::fmap<Hom<B, B>>(id);
  auto vec_idC = Vector::fmap<Hom<C, C>>(id);

  auto b_s = vec_f(a_s);
  auto c_s = vec_g(b_s);

  REQUIRE(vec_idA(a_s) == id(a_s));
  REQUIRE(vec_idB(b_s) == id(b_s));
  REQUIRE(vec_idC(c_s) == id(c_s));
}

// ........................................................ f]]]2
// covariant hom functor .................................. f[[[2

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

  // $\FOf{\ttName{CHom}}{\ttA} → \FOf{\ttName{CHom}}{\ttB}$
  auto homA_f = CHom<A>::fmap(f);

  // $\FOf{\ttName{CHom}}{\ttB} → \FOf{\ttName{CHom}}{\ttC}$
  auto homA_g = CHom<A>::fmap(g);

  // $\FOf{\ttName{CHom}}{\ttA} → \FOf{\ttName{CHom}}{\ttC}$
  auto homA_gf = CHom<A>::fmap<Hom<A, C>>(compose(g, f));

  // $\Ffmap{\ttName{Chom}}(\ttg) ∘ \Ffmap{\ttName{CHom}}(\ttf) =
  // \Ffmap{\ttName{CHom}}(\ttg ∘ \ttf)$
  REQUIRE(homA_gf(id)(A{}) == compose(homA_g, homA_f)(id)(A{}));

  auto homA_idA = CHom<A>::fmap<Hom<A, A>>(id);
  auto homA_idB = CHom<A>::fmap<Hom<B, B>>(id);
  auto homA_idC = CHom<A>::fmap<Hom<C, C>>(id);
  auto gf = compose(g, f);

  REQUIRE(homA_idA(id)(A{}) == id(A{}));
  REQUIRE(homA_idB(f)(A{}) == f(A{}));
  REQUIRE(homA_idC(gf)(A{}) == gf(A{}));
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
    return id;
  };
};

TEST_CASE("Functor axioms of Const<A>.") {
  auto constA_f = Const<A>::fmap(f);
  auto constA_g = Const<A>::fmap(g);
  auto constA_gf = Const<A>::fmap(compose(g, f));
  REQUIRE(constA_gf(A{}) == compose(constA_g, constA_f)(A{}));

  REQUIRE(Const<A>::fmap(id)(A{}) == A{});
  REQUIRE(Const<A>::fmap(id)(A{}) == A{});
  REQUIRE(Const<A>::fmap(id)(A{}) == A{});
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
  REQUIRE(compose(len<B>, Vector::fmap(f))(a_s) ==
          compose(Const<uint>::fmap(f), len<A>)(a_s));
}

// ........................................................ f]]]2
// CCC in Cpp ............................................. f[[[2

template <typename T, typename U>
using P = std::pair<T, U>;

struct Pair : public Bifunctor<Pair, P> {
  template <typename Fn>
  static auto lmap(Fn f) {
    return [f](auto tu) {
      auto [t, u] = tu;
      return std::pair{f(t), u};
    };
  }

  template <typename Fn>
  static auto rmap(Fn f) {
    return [f](auto tu) {
      auto [t, u] = tu;
      return std::pair{t, f(u)};
    };
  }
};

TEST_CASE("std::pair is functorial in the left position.") {

  auto ac = Pair::Of<A, C>{};

  auto l_f = Pair::lmap(f);
  auto l_g = Pair::lmap(g);
  auto l_gf = Pair::lmap<Hom<A, C>>(compose(g, f));

  REQUIRE(compose(l_g, l_f)(ac) == l_gf(ac));

  auto l_idA = Pair::lmap<Hom<A, A>>(id);
  auto l_idB = Pair::lmap<Hom<B, B>>(id);
  auto l_idC = Pair::lmap<Hom<C, C>>(id);

  auto bc = l_f(ac);
  auto cc = l_g(bc);

  REQUIRE(l_idA(ac) == id(ac));
  REQUIRE(l_idB(bc) == id(bc));
  REQUIRE(l_idC(cc) == id(cc));
}

TEST_CASE("std::pair is functorial in the right position.") {

  auto ca = Pair::Of<C, A>{};

  auto r_f = Pair::rmap(f);
  auto r_g = Pair::rmap(g);

  auto r_gf = Pair::rmap<Hom<A, C>>(compose(g, f));

  REQUIRE(compose(r_g, r_f)(ca) == r_gf(ca));

  auto r_idA = Pair::rmap<Hom<A, A>>(id);
  auto r_idB = Pair::rmap<Hom<B, B>>(id);
  auto r_idC = Pair::rmap<Hom<C, C>>(id);

  auto cb = r_f(ca);
  auto cc = r_g(cb);

  REQUIRE(r_idA(ca) == id(ca));
  REQUIRE(r_idB(cb) == id(cb));
  REQUIRE(r_idC(cc) == id(cc));
}

template <typename F, typename G>
auto prod(F f, G g)
    -> Hom<P<Dom<F>, Dom<G>>, P<Cod<F>, Cod<G>>> {
  using TandU = P<Dom<F>, Dom<G>>;
  using XandY = P<Cod<F>, Cod<G>>;

  return [f, g](TandU tu) -> XandY {
    auto [t, u] = tu;
    return {f(t), g(u)};
  };
}

TEST_CASE("Product of functions as expected") {
  REQUIRE(prod(f, h)(P<A, C>{}) == P<B, D>{});
  REQUIRE(prod(f, Hom<C, C>{id})(P<A, C>{}) ==
          Pair::lmap(f)(P<A, C>{}));
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

// std::pair associator ................................... f[[[3

template <typename A, typename B, typename C>
auto associator_fd(P<A, P<B, C>> a_bc) -> P<P<A, B>, C> {
  auto [a, bc] = a_bc;
  auto [b, c] = bc;

  return {{a, b}, c};
}

template <typename A, typename B, typename C>
auto associator_rv(P<P<A, B>, C> ab_c) -> P<A, P<B, C>> {
  auto [ab, c] = ab_c;
  auto [a, b] = ab;

  return {a, {b, c}};
}

TEST_CASE("associator_fd and associator_rv are inverse.") {
  auto associator_fd_rv =
      compose(associator_rv<A, B, C>, associator_fd<A, B, C>);
  auto Pa_Pbc = P<A, P<B, C>>{};
  REQUIRE(associator_fd_rv(Pa_Pbc) == id(Pa_Pbc));
}

TEST_CASE("Associator diagram for std::pair") {
  auto PPPab_c_d = P<P<P<A, B>, C>, D>{}; // End
  // $↑$ associator_fd<P<A,B>, C, D>
  auto PabPcd = P<P<A, B>, P<C, D>>{};
  // $↑$ associator_fd<A, B, P<C,D>>
  auto Pa_Pb_Pcd = P<A, P<B, P<C, D>>>{}; // Start
  // $↓$ id $×$ associator_fd<B,C,D>
  auto Pa_PPbc_d = P<A, P<P<B, C>, D>>{};
  // $↓$ associator_fd<A, P<B, C>, D>
  auto PPa_Pbc_d = P<P<A, P<B, C>>, D>{};
  // $↓$ associator_fd<A,B,C> $×$ id
  // auto PPPab_c_d = P<P<P<A, B>, C>, D>{}; // End

  // Clockwise
  REQUIRE(associator_fd<A, B, P<C, D>>(Pa_Pb_Pcd) == PabPcd);
  REQUIRE(associator_fd<P<A, B>, C, D>(PabPcd) == PPPab_c_d);

  auto cw_path = compose(associator_fd<P<A, B>, C, D>,
      associator_fd<A, B, P<C, D>>);

  REQUIRE(cw_path(Pa_Pb_Pcd) == PPPab_c_d);

  // Anti-clockwise
  auto id_x_assoc_BCD =
      prod(Hom<A, A>{id}, std::function{associator_fd<B, C, D>});
  REQUIRE(id_x_assoc_BCD(Pa_Pb_Pcd) == Pa_PPbc_d);
  REQUIRE(associator_fd<A, P<B, C>, D>(Pa_PPbc_d) == PPa_Pbc_d);

  auto assoc_A_PBC_D_x_id =
      prod(std::function{associator_fd<A, B, C>}, Hom<D, D>{id});
  REQUIRE(assoc_A_PBC_D_x_id(PPa_Pbc_d) == PPPab_c_d);

  auto ccw_path = compose(assoc_A_PBC_D_x_id,
      associator_fd<A, P<B, C>, D>, id_x_assoc_BCD);
  REQUIRE(ccw_path(Pa_Pb_Pcd) == PPPab_c_d);

  // Now check composites
  REQUIRE(ccw_path(Pa_Pb_Pcd) == cw_path(Pa_Pb_Pcd));
};

// ........................................................ f]]]3
// std::pair unitor ....................................... f[[[3

// unit-type is any singleton.
struct I {
  bool operator==(const I) const { return true; }
};

template <typename A>
auto l_unitor_fw(P<I, A> ia) -> A {
  return ia.second;
}

template <typename A>
auto l_unitor_rv(A a) -> P<I, A> {
  return {I{}, a};
}

template <typename A>
auto r_unitor_fw(P<A, I> ai) -> A {
  return ai.first;
}

template <typename A>
auto r_unitor_rv(A a) -> P<A, I> {
  return {a, I{}};
}

TEST_CASE("L-/R-unitor inverse") {
  auto ia = P<I, A>{};
  auto ai = P<A, I>{};

  auto l_unitor_fw_rv = compose(l_unitor_rv<A>, l_unitor_fw<A>);
  REQUIRE(l_unitor_fw_rv(ia) == id(ia));

  auto r_unitor_fw_rv = compose(r_unitor_rv<A>, r_unitor_fw<A>);
  REQUIRE(r_unitor_fw_rv(ai) == id(ai));
}

TEST_CASE("Unitor diagram") {
  auto Pab = P<A, B>{}; // End
  // $↑$ runitor_x_id
  auto PPai_b = P<P<A, I>, B>{};
  // $↑$ associator<A,I,B>
  auto Pa_Pib = P<A, P<I, B>>{}; // Start
  // $↓$ id_x_lunitor
  // auto Pab = P<A, B>{};       // End

  // Clockwise path
  REQUIRE(associator_fd<A, I, B>(Pa_Pib) == PPai_b);

  auto runitor_x_id = prod(r_unitor_fw<A>, Hom<B, B>{id});
  REQUIRE(runitor_x_id(PPai_b) == Pab);

  auto cw_path = compose(runitor_x_id, associator_fd<A, I, B>);

  // Anti-clockwise path
  auto ccw_path = prod(Hom<A, A>{id}, l_unitor_fw<B>);
  REQUIRE(ccw_path(Pa_Pib) == Pab);

  // Check both paths are equivalent
  REQUIRE(cw_path(Pa_Pib) == ccw_path(Pa_Pib));
}

// ........................................................ f]]]3
// std::pair braiding, self-inverse. ...................... f[[[3
template <typename A, typename B>
auto braid(P<A, B> ab) -> P<B, A> {
  auto [a, b] = ab;
  return {b, a};
}
// ........................................................ f]]]3
// ........................................................ f]]]2
// Prod and Coprod of functions ........................... f[[[2

template <typename Fn, typename Gn>
auto fanin(Fn f, Gn g)
    -> Hom<std::variant<Dom<Fn>, Dom<Gn>>, Cod<Fn>> {
  using T = Dom<Fn>;
  using U = Dom<Gn>;
  using X = Cod<Fn>;
  using TorU = std::variant<T, U>;

  static_assert(std::is_same_v<X, Cod<Gn>>);

  return [f, g](TorU t_or_u) -> X {
    if (std::holds_alternative<T>(t_or_u))
      return std::invoke(f, std::get<T>(t_or_u));
    else
      return std::invoke(g, std::get<U>(t_or_u));
  };
}

TEST_CASE("fanin as expected") {
  auto A_to_bool = [](A) { return true; };
  auto B_to_bool = [](B) { return false; };

  auto really_a = std::variant<A, B>{A{}};
  auto really_b = std::variant<A, B>{B{}};

  auto AorB_to_bool = fanin(A_to_bool, B_to_bool);

  REQUIRE(AorB_to_bool(really_a) == true);
  REQUIRE(AorB_to_bool(really_b) == false);
}

// ((A → B), (C → D)) → (A + C → B + D)
template <typename F, typename G>
auto coprod(F f, G g) -> Hom<std::variant<Dom<F>, Dom<G>>,
    std::variant<Cod<F>, Cod<G>>> {
  using T = Dom<F>;
  using U = Dom<G>;
  using X = Cod<F>;
  using Y = Cod<G>;
  using TorU = std::variant<T, U>;
  using XorY = std::variant<X, Y>;

  return [f, g](TorU t_or_u) -> XorY {
    if (std::holds_alternative<T>(t_or_u))
      return std::invoke(f, std::get<T>(t_or_u));
    else
      return std::invoke(g, std::get<U>(t_or_u));
  };
}

TEST_CASE("Coproduct of functions as expected") {
  REQUIRE(std::holds_alternative<B>(coprod(f, h)(A{})));
  REQUIRE(std::holds_alternative<D>(coprod(f, h)(C{})));
}

// ........................................................ f]]]2
// Monoidal coproduct ..................................... f[[[2

struct Never { // Monoidal unit for variant.
  Never() = delete;
  Never(const Never &) = delete;
};

TEST_CASE("") {
  auto a = std::variant<Never, int>{4};
  REQUIRE(std::get<int>(a) == 4);
}

// ........................................................ f]]]2
// ........................................................ f]]]1
