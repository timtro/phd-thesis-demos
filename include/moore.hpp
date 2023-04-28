// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65
#pragma once

#include "Cpp-BiCCC.hpp"

namespace moore {

using State = int;
using Output = int;
using Input = int;

// Classical Moore Machine ................................ f[[[1
template <typename I, typename S, typename O>
struct MooreMachine {
  S s0;
  Hom<Doms<S, I>, S> tmap; // $S × I → S$
  Hom<S, O> rmap;          // $S → O$
};
// ........................................................ f]]]1
// MooreMachine to snoc algebra ........................... f[[[1
template <typename S>
auto moore_to_snoc_algebra(MooreMachine<Input, S, Output> mm)
    -> SnocF<Input>::Alg<S> {
  auto global_s0 = [mm](PUnit) -> S { return mm.s0; };
  auto state_trans = [mm](P<std::shared_ptr<S>, Input> p) -> int {
    auto [s, i] = p;
    return mm.tmap(*s, i);
  };

  return fanin(global_s0, state_trans);
}

// template <typename S>
// auto snoc_scanify(OPIAlgebra<S> alg) -> OPIAlgebra<std::vector<S>> {
//   return [alg](OPI<std::vector<S>> op) -> std::vector<S> {
//     if (!op)
//       return std::vector{alg(std::nullopt)};
//
//     auto [accum, val] = *op;
//     auto s0 = accum.back();
//     accum.push_back(alg(OPI<S>{{s0, val}}));
//     return accum;
//   };
// }
// ........................................................ f]]]1
// Moore Coalgebra ........................................ f[[[1

// M<S> = $(I ⊸ S, O)$
template <typename S>
using M = P<Hom<Input, S>, Output>;

//              M<𝑓>
//         M<A> ────🢒 M<B>
//
//          A ───────🢒 B
//               𝑓
template <typename A, typename B>
auto M_map(Hom<A, B> f) -> Hom<M<A>, M<B>> {
  return [f](const M<A> ma) -> M<B> {
    return {[f, ma](auto x) { return f(ma.first(x)); }, ma.second};
  };
}

// $\mathtt{MCoalg⟨S⟩} ≅ S → 𝘔⟨S⟩ = S → ( I ⊸ S, O)$
template <typename S>
using MCoalgebra = Hom<S, M<S>>;

template <typename I, typename S, typename O>
auto moore_to_coalgebra(MooreMachine<I, S, O> mm) -> MCoalgebra<S> {
  return [mm](S s) { return M<S>{curry(mm.tmap)(s), mm.rmap(s)}; };
}

}

// ........................................................ f]]]1
