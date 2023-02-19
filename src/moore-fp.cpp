// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]

#include <functional>
#include <iostream>
#include <numeric> // for inclusive_scan
#include <optional>
#include <vector>

#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/view/exclusive_scan.hpp>
#include <range/v3/view/transform.hpp>
#include <rxcpp/rx.hpp>

#include <catch2/catch.hpp>

using State = int;
using Output = int;
using Input = int;

// Basic cCpp defs f[[[1
template <typename T>
constexpr decltype(auto) id(T &&x) {
  return std::forward<T>(x);
}

template <typename F, typename... Fs>
constexpr decltype(auto) compose(F f, Fs... fs) {
  if constexpr (sizeof...(fs) < 1)
    return [f](auto &&x) -> decltype(auto) {
      return std::invoke(f, std::forward<decltype(x)>(x));
    };
  else
    return [f, fs...](auto &&x) -> decltype(auto) {
      return std::invoke(
          f, compose(fs...)(std::forward<decltype(x)>(x)));
    };
}

template <typename F>
constexpr decltype(auto) curry(F f) {
  if constexpr (std::is_invocable_v<F>)
    return std::invoke(f);
  else
    return [f](auto &&x) {
      return curry(
          // perfectly capture x here:
          [f, x](auto &&...xs)
              -> decltype(std::invoke(f, x, xs...)) {
            return std::invoke(f, x, xs...);
          });
    };
}
//                                                                         f]]]1
// Classical Moore Machine f[[[1
template <typename I, typename S, typename O>
struct MooreMachine {
  S s0;
  std::function<S(S, I)> tmap;
  std::function<O(S)> rmap;
};
//                                                                         f]]]1
// Moore Coalgebra f[[[1

// M<S> = (I ⊸ S, O)
template <typename S>
using M = std::pair<std::function<S(Input)>, Output>;

//         M<𝑓>
//    M<A> ────🢒 M<B>
//
//     A ───────🢒 Λ
//          𝑓
template <typename A, typename B>
auto mmap(std::function<B(A)> f) -> std::function<M<B>(M<A>)> {
  return [f](const M<A> ma) -> M<B> {
    return {
        [f, ma](auto x) { return f(ma.first(x)); }, ma.second};
  };
}

// MCoalg = S → M<S> = S → ( I ⊸ S, O);
template <typename S>
using MCoalgebra = std::function<M<S>(S)>;

template <typename I, typename S, typename O>
auto moore_to_coalgebra(MooreMachine<I, S, O> mm)
    -> MCoalgebra<S> {
  return [mm](S s) {
    return M<S>{curry(mm.tmap)(s), mm.rmap(s)};
  };
}

//                                                                         f]]]1
// Moore algebra f[[[1
template <typename S, typename I>
using OP = std::optional<std::pair<S, I>>;

template <typename S, typename I>
using OPAlgebra = std::function<S(OP<S, I>)>;

template <typename I, typename S, typename O>
auto moore_to_algebra(MooreMachine<I, S, O> mm)
    -> OPAlgebra<S, I> {
  return [mm](OP<S, I> o_sxi) {
    if (!o_sxi)
      return mm.s0;

    auto [s, i] = *o_sxi;
    return mm.tmap(s, i);
  };
}

template <typename S, typename I>
auto make_cata(OPAlgebra<S, I> alg)
    -> std::function<S(std::vector<I>)> {

  return [alg](std::vector<I> i_s) -> S {
    auto s0 = alg(std::nullopt);

    if (i_s.empty())
      return s0;

    auto accumulator = s0;
    for (auto &i : i_s)
      accumulator = alg({{accumulator, i}});

    return accumulator;
  };
}
// f]]]1
// List co/algebra stuff f[[[1
template <typename T, typename U>
using OPCoalgebra = std::function<OP<T, U>(T)>;

template <typename X>
using OP_O = OP<X, Output>;

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
auto pr_ana(OPCoalgebra<T, U> coalg, T seed) -> std::vector<U> {
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
//                                                                         f]]]1
// SCANIFY f[[[1 scanify :: 𝘗ᵣAlgebra state value → 𝘗ᵣAlgebra
// (Snoc state) value scanify alg (Pᵣ Nothing) = nil ⧺ alg (Pᵣ
// Nothing) scanify alg (Pᵣ (Just ( accum , val))) = accum ⧺ alg
// (Pᵣ (Just (s0, val)))
//   where
//     s0 = snocHead accum
template <typename S, typename I>
auto scanify(OPAlgebra<S, I> alg)
    -> OPAlgebra<std::vector<S>, I> {
  return [alg](OP<std::vector<S>, I> op) -> std::vector<S> {
    if (!op)
      return std::vector{alg(std::nullopt)};

    auto [accum, val] = *op;
    auto s0 = accum.back();
    accum.push_back(alg(OP<S, I>{{s0, val}}));
    return accum;
  };
}
//                                                                         f]]]1
// Utilities f[[[1
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
//                                                                         f]]]1

TEST_CASE(
    "Given a MooreMachine where,\n"
    "   S = O = I = int\n"
    "  s0 = 0\n"
    "   f = (i, s) $↦$ s + i\n"
    "   r = s $↦$ s,\n"
    "and given an input vector `i_s` and manually computed "
    "`running_sum`…") {
  State s0 = 0;
  auto f = [](State s, Input i) -> State { return s + i; };
  auto r = id<State>;
  MooreMachine<Input, State, Output> mm = {s0, f, r};

  auto i_s = std::vector<Input>{0, 1, 2, 3, 4};
  // running_sum = $\set{s₀, r∘f(s_k,i_k)}_{k=0}^4$.
  auto running_sum = std::vector<Output>{0, 0, 1, 3, 6, 10};
  //                                     $↑$
  //                              Initial state

  AND_GIVEN(
      "a function that explicitly demonstrates the "
      "recursion of f while generating a sequence of "
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

  AND_GIVEN("A $𝘗^*_I$-algebra embodying $f$ and $s0$") {
    OPAlgebra<State, Input> alg = moore_to_algebra(mm);

    THEN("The algebra catamorphised over the input list should "
         "produce the total sum") {

      auto phi = compose(mm.rmap, make_cata(alg));
      REQUIRE(phi(i_s) == running_sum.back());
    }

    THEN("The scanified version of that algebra should produce "
         "a list (i.e., std::vector) of the running sum.") {
      auto running_sumer = make_cata(scanify(alg));
      REQUIRE(running_sumer(i_s) == running_sum);
    }
  }

  AND_GIVEN(
      "a function using the scan combinator from RxC++ to "
      "accumulate with f and then map r over the "
      "result.") {

    auto rxcpp_scan = [&i_s, &mm]() -> std::vector<Output> {
      const auto [s0, f, r] = mm;
      auto oi = rxcpp::observable<>::create<
          Input>([&](rxcpp::subscriber<Input> s) {
        for (auto each : i_s)
          s.on_next(each);
        s.on_completed();
      });

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
      const auto us =
          i_s | views::exclusive_scan(0, f) |
          views::transform(r);

      return std::vector(std::cbegin(us), std::cend(us));
    };

    THEN("the function should return a running sum including "
         "the initial state but without the last output "
         "value.") {
      REQUIRE(range_exclusive_scan() == drop_last(running_sum));
    }
  }
}
