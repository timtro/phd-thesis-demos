// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65

#include <functional>
#include <iostream>
#include <numeric> // for inclusive_scan
#include <optional>
#include <vector>

#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/view/exclusive_scan.hpp>
#include <range/v3/view/transform.hpp>
#include <rxcpp/rx.hpp>

namespace rx {
  using namespace rxcpp;
  using namespace rxcpp::operators;
  using namespace rxcpp::sources;
  using namespace rxcpp::util;
} // namespace rx

#include <catch2/catch.hpp>

#include "Cpp-arrows.hpp"

using tf::compose;
using tf::curry;
using tf::curry_variadic;
using tf::id;
using tf::pipe;

using State = int;
using Output = int;
using Input = int;

template<typename Sig>
using hom = std::function<Sig>;

// Classical Moore Machine ................................ f[[[1
template <typename I, typename S, typename O>
struct MooreMachine {
  S s0;
  hom<S(S, I)> tmap; // $S × I → S$
  hom<O(S)> rmap;    //     $S → O$
};
// ........................................................ f]]]1
// Moore Coalgebra ........................................ f[[[1

// M<S> = $(I ⊸ S, O)$
template <typename S>
using M = std::pair<hom<S(Input)>, Output>;

//              M<𝑓>
//         M<A> ────🢒 M<B>
//
//          A ───────🢒 B
//               𝑓
template <typename A, typename B>
auto M_map(hom<B(A)> f) -> hom<M<B>(M<A>)> {
  return [f](const M<A> ma) -> M<B> {
    return {
        [f, ma](auto x) { return f(ma.first(x)); }, ma.second};
  };
}

// $\mathtt{MCoalg⟨S⟩} ≅ S → 𝘔⟨S⟩ = S → ( I ⊸ S, O)$
template <typename S>
using MCoalgebra = hom<M<S>(S)>;

template <typename I, typename S, typename O>
auto moore_to_coalgebra(MooreMachine<I, S, O> mm)
    -> MCoalgebra<S> {
  return [mm](S s) {
    return M<S>{curry(mm.tmap)(s), mm.rmap(s)};
  };
}

// ........................................................ f]]]1
// Moore algebra .......................................... f[[[1

// $𝘖𝘗⟨S,I⟩ ≅ 𝟣 + S × I$
template <typename S, typename I>
using OP = std::optional<std::pair<S, I>>;

// $\mathtt{OPAlgebra} = \texttt{OP}⟨S, I⟩$
template <typename S, typename I>
using OPAlgebra = hom<S(OP<S, I>)>;

// $𝘖𝘗I⟨S⟩ ≅ 𝟣 + S × I$
template <typename S>
using OPI = std::optional<std::pair<S, Input>>;

// $\mathtt{OPIAlgebra} = \texttt{OPI}⟨S⟩$
template <typename S>
using OPIAlgebra = hom<S(OPI<S>)>;

template <typename S>
auto moore_to_algebra(MooreMachine<Input, S, Output> mm)
    -> OPIAlgebra<S> {
  return [mm](OPI<S> o_sxi) -> S {
    if (!o_sxi)
      return mm.s0;

    auto [s, i] = o_sxi.value();
    return mm.tmap(s, i);
  };
}

template <typename S>
auto make_cata(OPIAlgebra<S> alg) -> hom<S(std::vector<Input>)> {

  return [alg](std::vector<Input> i_s) -> S {
    auto s0 = alg(std::nullopt);

    auto accumulator = s0;
    for (auto &i : i_s)
      accumulator = alg({{accumulator, i}});

    return accumulator;
  };
}
// ........................................................ f]]]1
// List coalgebra stuff ................................... f[[[1
template <typename T, typename U>
using OPCoalgebra = hom<OP<T, U>(T)>;

template <typename T>
auto maybe_head_and_tail(std::vector<T> ts)
    -> std::optional<std::pair<T, std::vector<T>>> {
  if (ts.empty())
    return std::nullopt;

  const T head = ts[0];
  ts.erase(std::begin(ts));
  return {{head, ts}};
}

template <typename T, typename U>
auto ana_op(OPCoalgebra<T, U> coalg, T seed) -> std::vector<U> {
  std::vector<U> us;

  auto result_ab = coalg(seed);

  if (result_ab)
    us.push_back(result_ab->second);
  else
    return us;

  while (true) {
    result_ab = coalg(std::move(result_ab->first));
    if (result_ab)
      us.push_back(result_ab->second);
    else
      return us;
  }
}
// ........................................................ f]]]1
// Scanify ................................................ f[[[1
//
// scanify :: OPAlgebra<S,I> → OPAlgebra<S,I>
// transforms an algebra so that its catamorphism produces a
// scan.

// This commented version uses OPAlgebra, and templates on input
// type.
//
// template <typename S, typename I>
// auto scanify(OPAlgebra<S, I> alg)
//     -> OPAlgebra<std::vector<S>, I> {
//   return [alg](OP<std::vector<S>, I> op) -> std::vector<S> {
//     if (!op)
//       return std::vector{alg(std::nullopt)};
//
//     auto [accum, val] = *op;
//     auto s0 = accum.back();
//     accum.push_back(alg(OP<S, I>{{s0, val}}));
//     return accum;
//   };
// }

template <typename S>
auto scanify(OPIAlgebra<S> alg) -> OPIAlgebra<std::vector<S>> {
  return [alg](OPI<std::vector<S>> op) -> std::vector<S> {
    if (!op)
      return std::vector{alg(std::nullopt)};

    auto [accum, val] = *op;
    auto s0 = accum.back();
    accum.push_back(alg(OPI<S>{{s0, val}}));
    return accum;
  };
}
// ........................................................ f]]]1
// RxCpp operators for Moore machines ..................... f[[[1
template <typename I, typename S>
auto rx_scanl(S s0, hom<S(S, I)> f)
    -> hom<rx::observable<S>(rx::observable<I>)> {
  return [s0, f](rx::observable<I> obs) {
    return obs.scan(s0, f).start_with(s0);
  };
}

template <typename I, typename S, typename O>
auto rx_moore_machine(MooreMachine<I, S, O> mm)
    -> hom<rx::observable<S>(rx::observable<I>)> {
  return [mm](rx::observable<I> i) {
    return i | rx_scanl(mm.s0, mm.tmap) | rx::map(mm.rmap);
  };
}
// ........................................................ f]]]1
// Utilities .............................................. f[[[1
template <typename T>
auto drop_first(std::vector<T> ts) -> std::vector<T> {
  assert(!ts.empty());
  ts.erase(std::begin(ts));
  return ts;
}

template <typename T>
auto drop_last(std::vector<T> ts) -> std::vector<T> {
  assert(!ts.empty());
  ts.pop_back();
  return ts;
}

template <typename T>
auto make_vector_observable(std::vector<T> v)
    -> rx::observable<T> {
  return rx::observable<>::create<T>([=](rx::subscriber<T> s) {
    for (auto each : v) {
      s.on_next(each);
    }
    s.on_completed();
  });
}
// ........................................................ f]]]1

TEST_CASE(
    "Given a MooreMachine where,\n"
    "   S = O = I = int\n"
    "  s0 = 0\n"
    "   $f$ = (i, s) $↦$ s + i\n"
    "   $r$ = s $↦$ s,\n"
    "and given an input vector `i_s` and manually computed "
    "`running_sum`…") {
  const State s0 = 0;
  const auto f = [](State s, Input i) -> State { return s + i; };
  const auto r = id<State>;
  const auto mm = MooreMachine<Input, State, Output>{s0, f, r};

  const auto i_s = std::vector<Input>{0, 1, 2, 3, 4};
  // running_sum = $\set{s₀,\; r∘f(s_k,i_k)}_{k=0}^4$.
  const auto running_sum = // $\set{s₀,\; r∘f(s_k,i_k)}_{k=0}^4$.
      std::vector<Output>{0, 0, 1, 3, 6, 10};
  //                      $↑$
  //               Initial state
  AND_GIVEN(
      "a function that explicitly demonstrates the "
      "recursion of $f$ while generating a sequence of "
      "successive output values.") {

    auto manual_moore_machine =
        [&i_s, &mm]() -> std::vector<Output> {
      const auto [s0, f, r] = mm;
      return {
          // clang-format off
        r(s0),
        r(f(s0, i_s[0])),
        r(f(f(s0, i_s[0]), i_s[1])),
        r(f(f(f(s0, i_s[0]), i_s[1]), i_s[2])),
        r(f(f(f(f(s0, i_s[0]), i_s[1]), i_s[2]), i_s[3])),
        r(f(f(f(f(f(s0, i_s[0]), i_s[1]), i_s[2]), i_s[3]), i_s[4]))
          // clang-format on
      };
    };

    THEN("we should expect the running sum including the output "
         "of the initial state.") {
      REQUIRE(manual_moore_machine() == running_sum);
    }
  }

  AND_GIVEN(
      "A $𝘗^*_I$-algebra embodying $f$ and $s0$, and "
      "corresponding I/O response function "
      "$\\mathtt{phi} = r ∘ ⦇\\mathtt{alg}⦈$") {
    auto alg = moore_to_algebra(mm);
    auto phi = compose(mm.rmap, make_cata(alg));

    auto empty = std::vector<Input>();
    auto total =
        mm.rmap(std::accumulate(cbegin(i_s), cend(i_s), 0));

    THEN("phi applied to an empty list should produce the "
         "initial state.") {
      REQUIRE(phi(empty) == mm.rmap(mm.s0));
    }

    THEN("phi applied to i_s should produce its sum total.") {
      REQUIRE(phi(i_s) == total);
    }

    THEN("The scanified version of that algebra should produce "
         "a list (i.e., std::vector) of the running sum.") {
      auto running_sumer = make_cata(scanify(alg));
      REQUIRE(running_sumer(i_s) == running_sum);
    }
  }

  AND_GIVEN("an implementation of $ᵠ \\⦂ I^* → O$ in RxCpp") {

    auto rxcpp_scan = [&i_s, &mm]() -> std::vector<Output> {
      const auto [s0, f, r] = mm;
      auto oi = make_vector_observable(i_s);

      std::vector<Output> output;
      auto us = oi.scan(0, f).map(r);
      us.subscribe([&output](Output v) { output.push_back(v); });

      return output;
    };

    THEN("expect a running sum without output from the initial "
         "state.") {
      REQUIRE(rxcpp_scan() == drop_first(running_sum));
    }
  }

  AND_GIVEN("with rx_scnal") {

    auto custom_moore_scan =
        [&i_s, &mm]() -> std::vector<Output> {
      const auto [s0, f, r] = mm;
      auto oi = make_vector_observable(i_s);

      auto us = oi | rx_scanl(s0, f) | rx::map(r);

      std::vector<Output> output;
      us.subscribe([&output](Output v) { output.push_back(v); });

      return output;
    };

    THEN("expect a running sum without output from the initial "
         "state.") {
      REQUIRE(custom_moore_scan() == running_sum);
    }
  }

  AND_GIVEN("With rx_moore_machine") {

    auto custom_moore_scan =
        [&i_s, &mm]() -> std::vector<Output> {
      auto oi = make_vector_observable(i_s);
      auto oo = oi | rx_moore_machine(mm);

      std::vector<Output> output;
      oo.subscribe([&output](Output v) { output.push_back(v); });

      return output;
    };

    THEN("expect a running sum without output from the initial "
         "state.") {
      REQUIRE(custom_moore_scan() == running_sum);
    }
  }

  AND_GIVEN(
      "a function using std::inclusive_scan to accumulate "
      "state using f, and std::transform to map state to "
      "output.") {

    auto std_transform = [&i_s, &mm]() -> std::vector<Output> {
      const auto [s0, f, r] = mm;
      std::vector<int> output(i_s.size());
      std::inclusive_scan(
          cbegin(i_s), cend(i_s), begin(output), f, 0);
      std::transform(
          cbegin(output), cend(output), begin(output), r);

      return output;
    };

    THEN("expect a running sum without output from the initial "
         "state.") {
      REQUIRE(std_transform() == drop_first(running_sum));
    }
  }

  AND_GIVEN(
      "a function using range-v3's views::exclusive_scan "
      "combinator to accumulate state using f and then "
      "views::transform to map state to output.") {

    auto range_exclusive_scan =
        [&i_s, &mm]() -> std::vector<Output> {
      const auto [s0, f, r] = mm;

      using namespace ranges;
      const auto u_s =
          i_s | views::exclusive_scan(0, f) |
          views::transform(r);

      return std::vector(std::cbegin(u_s), std::cend(u_s));
    };

    THEN("the function should return a running sum including "
         "the initial state but without the last output "
         "value.") {
      REQUIRE(range_exclusive_scan() == drop_last(running_sum));
    }
  }
}
