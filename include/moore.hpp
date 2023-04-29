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
      -> SnocAlg<Input, S> {
    auto global_s0 = [mm](PUnit) -> S { return mm.s0; };
    auto state_trans = [mm](P<std::shared_ptr<S>, Input> p) -> int {
      auto [s, i] = p;
      return mm.tmap(*s, i);
    };

    return fanin(global_s0, state_trans);
  }

  template <typename I, typename S>
  auto snoc_scanify(SnocAlg<I, S> alg) -> SnocAlg<I, SnocList<S>> {
    return [alg=alg](auto unit_or_p) -> SnocList<S> {
      auto global_snoc_s0 = [&alg](PUnit) {
        return snoc(nil<S>, alg(PUnit{}));
        ;
      };

      auto accum_trans =
          [&alg](P<std::shared_ptr<SnocList<S>>, Input> p)
          -> SnocList<S> {
        auto [accum, val] = p;
        const auto s0 = std::get<1>(out((*accum))).second;
        return snoc(*accum, alg(P{std::make_shared<S>(s0), val}));
      };
      return fanin(global_snoc_s0, accum_trans)(unit_or_p);
    };
  }

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
      return {
          [f, ma](auto x) { return f(ma.first(x)); }, ma.second};
    };
  }

  // $\mathtt{MCoalg⟨S⟩} ≅ S → 𝘔⟨S⟩ = S → ( I ⊸ S, O)$
  template <typename S>
  using MCoalgebra = Hom<S, M<S>>;

  template <typename I, typename S, typename O>
  auto moore_to_coalgebra(MooreMachine<I, S, O> mm)
      -> MCoalgebra<S> {
    return [mm](S s) {
      return M<S>{curry(mm.tmap)(s), mm.rmap(s)};
    };
  }

} // namespace moore

// ........................................................ f]]]1
