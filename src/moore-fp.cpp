#include <functional>
#include <iostream>
#include <numeric> // for inclusive_scan
#include <optional>
#include <vector>

#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/view/exclusive_scan.hpp>
#include <range/v3/view/transform.hpp>
#include <rxcpp/rx.hpp>

template <class T>
auto operator<<(std::ostream &os, const std::vector<T> &v)
    -> std::ostream & {
  std::string separator = "[ ";
  for (const auto x : v) {
    std::cout << separator << x;
    separator = ", ";
  }
  std::cout << " ]";
  return os;
}

using State = int;
using Output = int;
using Input = int;

template <typename I, typename S, typename O>
struct MooreMachine {
  S s0;
  std::function<S(S, I)> tmap;
  std::function<O(S)> rmap;
};

// Moore Coalgebra tooling {{{1

// M<S> = (I ⊸ S, O)
template <typename S>
using M = std::pair<std::function<S(Input)>, Output>;

// MCoalg = S → M<S> = S → ( I ⊸ S, O);
template <typename S> using MCoalg = std::function<M<S>(S)>;

// Λ ≅ M<Λ> = (I ⊸ Λ, O) = (I ⊸ (I ⊸ (I ⊸ (⋯), O), O), O), O)
template <typename S> struct Lambda : M<Lambda<S>> {
  Lambda(MCoalg<S> sigma, S s0) {
    const M<S> ms = sigma(s0);
    // .first and .second come from std::pair parentage:
    this->first = [=](Input i) {
      return Lambda(sigma, ms.first(i));
    };
    this->second = ms.second;
  }
};

template <typename S> struct std::tuple_size<Lambda<S>> {
  static constexpr std::size_t value = 2;
};

template <typename S> struct std::tuple_element<0, Lambda<S>> {
  using type = std::function<Lambda<S>(Input)>;
};

template <typename S> struct std::tuple_element<1, Lambda<S>> {
  using type = Output;
};

template <std::size_t I, typename S>
auto get(const Lambda<S> &l) {
  if constexpr (I == 0)
    return l.first;
  else
    return l.second;
}

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
//                                                                          }}}1

// moore_machine_...(…) {{{1
// clang-format off
auto moore_machine_explicit(std::vector<Input> is,
  MooreMachine<Input, State, Output> mm) -> std::vector<Output> {
  const auto [s0, f, r] = mm;
  return {
      r(s0),
      r(f(s0, is[0])),
      r(f(f(s0, is[0]), is[1])),
      r(f(f(f(s0, is[0]), is[1]), is[2])),
      r(f(f(f(f(s0, is[0]), is[1]), is[2]), is[3])),
      r(f(f(f(f(f(s0, is[0]), is[1]), is[2]), is[3]), is[4]))};
}
// clang-format on

auto moore_machine_loop(const std::vector<Input> &is,
    MooreMachine<Input, State, Output> mm)
    -> std::vector<Output> {
  std::vector<int> output(is.size());
  const auto [s0, f, r] = mm;

  State current_state = s0;
  output.push_back(r(s0));
  for (const auto &i : is) {
    current_state = f(current_state, i);
    output.push_back(r(current_state));
  }

  return output;
}
//                                                                          }}}1
// Using Moore fixpoint {{{1
auto moore_lambda_explicit(const std::vector<Input> &is,
    MooreMachine<Input, State, Output> mm)
    -> std::vector<Output> {
  auto s0 = mm.s0;
  auto f = mm.tmap;
  auto r = mm.rmap;

  std::vector<int> output(is.size());

  MCoalg<State> sigma = [f, r](State s) -> M<State> {
    return {[s, f, r](Input i) { return f(s, i); }, r(s)};
  };

  auto [l0, o0] = Lambda(sigma, s0);
  auto [l1, o1] = l0(is[0]);
  auto [l2, o2] = l1(is[1]);
  auto [l3, o3] = l2(is[2]);
  auto [l4, o4] = l3(is[3]);
  auto [l5, o5] = l4(is[4]);

  return {o0, o1, o2, o3, o4, o5};
}

// unfold : ( (A → optional<pair<A, B>>), A ) → vector<B>
template <typename F, typename A> auto unfold(F f, A a0) {
  // will fail if `f` doesn't return a pair when given an A.
  using B =
      decltype(std::declval<std::invoke_result_t<F, A>>()
                   ->second);
  static_assert(std::is_same_v<std::invoke_result_t<F, A>,
      std::optional<std::pair<A, B>>>);

  std::vector<B> bs;

  auto result_ab = f(a0);
  if (result_ab)
    bs.push_back(result_ab->second);
  else
    return bs;

  while (true) {
    result_ab = f(std::move(result_ab->first));
    if (result_ab)
      bs.push_back(result_ab->second);
    else
      return bs;
  }
}

template <typename X>
using P_O = std::optional<std::pair<X, Output>>;

template <typename T>
auto maybe_head_and_tail(std::vector<T> ts)
    -> std::optional<std::pair<T, std::vector<T>>> {
  if (ts.empty())
    return std::nullopt;

  const T head = ts[0];
  ts.erase(std::begin(ts));
  return {{head, ts}};
}

auto moore_lambda_listana(const std::vector<Input> &is,
    MooreMachine<Input, State, Output> mm)
    -> std::vector<Output> {

  using LxI = std::pair<Lambda<State>, std::vector<Input>>;

  auto rho = [](LxI lambda_and_inputs) -> P_O<LxI> {
    auto [l, is] = lambda_and_inputs;
    auto opt_head_tail = maybe_head_and_tail(is);

    if (!opt_head_tail)
      return std::nullopt;

    auto [head, tail] = *opt_head_tail;
    return {{{l.first(head), tail}, l.second}};
  };

  MCoalg<State> sigma =
      [f = mm.tmap, r = mm.rmap](State s) -> M<State> {
    return {[s, f, r](Input i) { return f(s, i); }, r(s)};
  };

  return unfold(rho, std::pair{Lambda(sigma, mm.s0), is});
}
//                                                                          }}}1
// stdlib_cpp {{{1
auto stdlib_cpp(const std::vector<Input> &is,
    MooreMachine<Input, State, Output> mm)
    -> std::vector<Output> {
  const auto [s0, f, r] = mm;
  std::vector<int> output(is.size());
  std::inclusive_scan(cbegin(is), cend(is), begin(output), f, 0);
  std::transform(cbegin(output), cend(output), begin(output), r);

  return output;
}
//                                                                          }}}1
// rxcpp_scan {{{1
auto rxcpp_scan(const std::vector<Input> &is,
    MooreMachine<Input, State, Output> mm)
    -> std::vector<Output> {
  const auto [s0, f, r] = mm;
  auto oi = rxcpp::observable<>::create<
      Input>([&](rxcpp::subscriber<Input> s) {
    for (auto each : is)
      s.on_next(each);
    s.on_completed();
  });

  std::vector<Output> output;
  auto us = oi.scan(0, f).map(r);
  us.subscribe([&output](Output v) { output.push_back(v); });

  return output;
}
//                                                                          }}}1
// range_exclusive_scan {{{1
auto range_exclusive_scan(const std::vector<Input> &is,
    MooreMachine<Input, State, Output> mm)
    -> std::vector<Output> {
  const auto [s0, f, r] = mm;

  using namespace ranges;
  const auto us =
      is | views::exclusive_scan(0, f) | views::transform(r);

  return std::vector(std::cbegin(us), std::cend(us));
}
//                                                                          }}}1

// main() {{{1
int main() {
  std::vector<Input> is = {0, 1, 2, 3, 4};
  State s0 = 0;
  auto f = [](State c, Input i) -> State { return c + i; };
  auto r = [](State c) -> Output { return c; };

  MooreMachine<Input, State, Output> mm = {s0, f, r};

  // clang-format off
  std::cout << "moore_machine_explicit = "
            << moore_machine_explicit(is, mm)
            << std::endl
            << "moore_lambda_explicit  = "
            << moore_lambda_explicit(is, mm) 
            << std::endl
            << "moore_lambda_listana   = "
            << moore_lambda_listana(is, mm)
            << std::endl
            << "rxcpp_scan             = "
            << rxcpp_scan(is, mm)
            << " ← rxcpp’s scan drops initial value."
            << std::endl
            << "lib_cpp                = "
            << stdlib_cpp(is, mm)
            << " ← std::inclusive_scan drops initial value."
            << std::endl
            << "range_exclusive_scan   = "
            << range_exclusive_scan(is, mm)
            << " ← exclusive_scan drops last value."\
            << std::endl;
  // clang-format on
}
//                                                                          }}}1
