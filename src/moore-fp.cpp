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

// Λ ≅ M<Λ> = (I ⊸ Λ, O) = (I ⊸ (I ⊸ (I ⊸ (⋯), O), O), O), O)
template <typename S>
struct Lambda : M<Lambda<S>> {
  Lambda(MCoalgebra<S> sigma, S s0) {
    const M<S> ms = sigma(s0);
    // .first and .second come from std::pair parentage:
    this->first = [=](Input i) {
      return Lambda(sigma, ms.first(i));
    };
    this->second = ms.second;
  }
};

template <typename S>
struct std::tuple_size<Lambda<S>> {
  static constexpr std::size_t value = 2;
};

template <typename S>
struct std::tuple_element<0, Lambda<S>> {
  using type = std::function<Lambda<S>(Input)>;
};

template <typename S>
struct std::tuple_element<1, Lambda<S>> {
  using type = Output;
};

template <std::size_t I, typename S>
auto get(const Lambda<S> &l) {
  if constexpr (I == 0)
    return l.first;
  else
    return l.second;
}
//                                                                         f]]]1
// Scanify solution stuff f[[[1
template <typename S, typename I>
using OP = std::optional<std::pair<S, I>>;

template <typename S, typename I>
using OPAlgebra = std::function<S(OP<S, I>)>;

// scanify :: 𝘗ᵣAlgebra state value → 𝘗ᵣAlgebra (Snoc state)
// value scanify alg (Pᵣ Nothing) = nil ⧺ alg (Pᵣ Nothing)
// scanify alg (Pᵣ (Just ( accum , val))) = accum ⧺ alg (Pᵣ (Just
// (s0, val)))
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

template <typename F>
auto flip(const F f) {
  return [f](auto a, auto b) { return f(b, a); };
}

template <typename F, typename T, typename C>
T foldr(const F &f, const T &z, const C &c) {
  return accumulate(crbegin(c), crend(c), z, flip(f));
}

// foldr f z []     = z
// foldr f z (x:xs) = f x (foldr f z xs)
template <typename S, typename I>
auto cata(OPAlgebra<S, I> alg, std::vector<I> vs) -> S {
  auto s0 = alg(std::nullopt);

  if (vs.empty())
    return s0;

  auto accumulator = s0;
  for (auto &v : vs)
    accumulator = alg({{accumulator, v}});

  return accumulator;
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
  auto r = [](State s) -> Output { return s; };
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

  AND_GIVEN(
      "a function that explicitly uses a $Λ$ value "
      "explicitly.") {

    auto moore_lambda_explicit =
        [&i_s, &mm]() -> std::vector<Output> {
      auto s0 = mm.s0;  // Cannot capture structured
      auto f = mm.tmap; //   bindings in C++17
      auto r = mm.rmap; //   lambda-closures.

      std::vector<int> output(i_s.size());

      MCoalgebra<State> sigma = [f, r](State s) -> M<State> {
        return {[s, f, r](Input i) { return f(s, i); }, r(s)};
      };

      auto [l0, o0] = Lambda(sigma, s0);
      auto [l1, o1] = l0(i_s[0]);
      auto [l2, o2] = l1(i_s[1]);
      auto [l3, o3] = l2(i_s[2]);
      auto [l4, o4] = l3(i_s[3]);
      auto [l5, o5] = l4(i_s[4]);

      return {o0, o1, o2, o3, o4, o5};
    };

    THEN("we should expect the running sum including the output "
         "of the initial state.") {

      REQUIRE(moore_lambda_explicit() == running_sum);
    }
  }

  AND_GIVEN(
      "a function which combines the Lambda structure and "
      "a $𝘗$-anamorphism to automatically map the input "
      "to the output, simulating inclusive scan.") {

    auto moore_lambda_list_ana =
        [&i_s, &mm]() -> std::vector<Output> {
      using LxI = std::pair<Lambda<State>, std::vector<Input>>;

      OPCoalgebra<LxI, Output> rho_exclusive =
          [](LxI lambda_and_inputs) -> OP_O<LxI> {
        auto [l, is] = lambda_and_inputs;
        auto opt_head_tail = maybe_head_and_tail(is);

        if (!opt_head_tail)
          return std::nullopt;

        auto [head, tail] = *opt_head_tail;
        return {{{l.first(head), tail}, l.second}};
      };

      OPCoalgebra<LxI, Output> rho_inclusive =
          [](LxI lambda_and_inputs) -> OP_O<LxI> {
        auto [l, is] = lambda_and_inputs;
        auto opt_head_tail = maybe_head_and_tail(is);

        if (!opt_head_tail)
          return std::nullopt;

        auto [head, tail] = *opt_head_tail;
        auto next_lambda = l.first(head);

        return {{{next_lambda, tail}, next_lambda.second}};
      };

      MCoalgebra<State> sigma =
          [f = mm.tmap, r = mm.rmap](State s) -> M<State> {
        return {[s, f, r](Input i) { return f(s, i); }, r(s)};
      };

      return pr_ana(
          rho_inclusive, std::pair{Lambda(sigma, mm.s0), i_s});
    };

    THEN("the function should return a running sum including "
         "the final state but without the initial.") {
      REQUIRE(moore_lambda_list_ana() == drop_first(running_sum));
    }
  }

  AND_GIVEN("A 𝘗-algebra embodying f and s0") {
    OPAlgebra<State, Input> alg =
        [&mm](OP<State, Input> p) -> State {
      if (!p)
        return mm.s0;

      auto [s, i] = *p;
      return mm.tmap(s, i);
    };

    THEN("The algebra catamorphised over the input list should "
         "produce the sum") {
      REQUIRE(cata(alg, i_s) == running_sum.back());
    }

    THEN("The scanified version of that algebra should produce "
         "a list (i.e., std::vector) of the running sum.") {
      auto scanified_alg = scanify(alg);
      REQUIRE(cata(scanified_alg, i_s) == running_sum);
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
          i_s | views::exclusive_scan(0, f) | views::transform(r);

      return std::vector(std::cbegin(us), std::cend(us));
    };

    THEN("the function should return a running sum including "
         "the initial state but without the last output "
         "value.") {
      REQUIRE(range_exclusive_scan() == drop_last(running_sum));
    }
  }
}
