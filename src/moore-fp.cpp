// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65

#include "Cpp-BiCCC.hpp"
#include <functional>
#include <iostream>
#include <numeric> // for inclusive_scan
#include <optional>
#include <vector>

#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/view/exclusive_scan.hpp>
#include <range/v3/view/transform.hpp>

#include "moore.hpp"
#include "rx_scanl.hpp"

#include <catch2/catch.hpp>

using moore::moore_to_coalgebra;
using moore::MooreMachine;
using moore::snoc_scanify;
using moore::M;
using moore::M_map;
using moore::MCoalgebra;
using moore::moore_to_coalgebra;

using State = moore::State;   // int
using Output = moore::Output; // int
using Input = moore::Input;   // int

// RxCpp operators for Moore machines ..................... f[[[1

template <typename I, typename S, typename O>
auto rx_moore_machine(MooreMachine<I, S, O> mm)
    -> Hom<rx::observable<I>, rx::observable<S>> {
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
auto make_vector_observable(std::vector<T> v) -> rx::observable<T> {
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
    auto alg = moore_to_snoc_algebra(mm);
    auto phi = compose(mm.rmap, SnocF<Input>::cata<State>(alg));

    auto total =
        mm.rmap(std::accumulate(cbegin(i_s), cend(i_s), 0));

    auto snoc_is = to_snoclist(i_s);

    THEN("phi applied to an empty list should produce the "
         "initial state.") {
      REQUIRE(phi(nil<Input>) == mm.rmap(mm.s0));
    }

    THEN("phi applied to i_s should produce its sum total.") {
      REQUIRE(phi(snoc_is) == total);
    }

    THEN("The scanified version of that algebra should produce "
         "a list (i.e., SnocList) of the running sum.") {
      auto running_sumer = SnocF<Input>::cata<SnocList<State>>(
          snoc_scanify<Input, State>(alg));
      REQUIRE(running_sumer(snoc_is) == to_snoclist(running_sum));
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

    auto custom_moore_scan = [&i_s, &mm]() -> std::vector<Output> {
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

    auto custom_moore_scan = [&i_s, &mm]() -> std::vector<Output> {
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
          i_s | views::exclusive_scan(0, f) | views::transform(r);

      return std::vector(std::cbegin(u_s), std::cend(u_s));
    };

    THEN("the function should return a running sum including "
         "the initial state but without the last output "
         "value.") {
      REQUIRE(range_exclusive_scan() == drop_last(running_sum));
    }
  }
}
