#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/view/exclusive_scan.hpp>
#include <range/v3/view/transform.hpp>
#include <rxcpp/rx.hpp>
#include <type_traits>
#include <utility>
#include <vector>

template <class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
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

using TransitionMap = std::function<State(State, Input)>;
using ReadoutMap = std::function<Output(State)>;

template <typename I, typename S, typename O>
struct MooreMachine {
  S s0;
  std::function<S(S, I)> tmap;
  std::function<O(S)> rmap;
};

// Moore Coalgebra tooling                                                  {{{1

// M<S> = (I ⊸ S, O)
template <typename S>
using M = std::pair<std::function<S(Input)>, Output>;

// MCoalg = s → M<S> = S → ( I ⊸ S, O);
template <typename S>
using MCoalg = std::function<M<S>(S)>;

// Λ ≅ M<Λ> = (I ⊸ Λ, O) = (I ⊸ (I ⊸ (I ⊸ (⋯), O), O), O), O)
template <typename S>
struct Lambda : M<Lambda<S>> {
  Lambda(MCoalg<S> sig, S s0) {
    const M<S> ms = sig(s0);
    // .first and .second come from std::pair parentage:
    this->first = [=](Input i) { return Lambda(sig, ms.first(i)); };
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
auto get(const Lambda<S>& l) {
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
    return {[f, ma](auto x) { return f(ma.first(x)); }, ma.second};
  };
}
//                                                                          }}}1

// moore_machine_...(…)                                                     {{{1
auto moore_machine_explicit(std::vector<Input> is,
                            MooreMachine<Input, State, Output> mm)
    -> std::vector<Output> {
  const auto [s0, f, r] = mm;
  return {r(s0),
          r(f(s0, is[0])),
          r(f(f(s0, is[0]), is[1])),
          r(f(f(f(s0, is[0]), is[1]), is[2])),
          r(f(f(f(f(s0, is[0]), is[1]), is[2]), is[3])),
          r(f(f(f(f(f(s0, is[0]), is[1]), is[2]), is[3]), is[4]))};
}

auto moore_machine_loop(const std::vector<Input>& is,
                        MooreMachine<Input, State, Output> mm)
    -> std::vector<Output> {
  std::vector<int> output(is.size());
  const auto [s0, f, r] = mm;

  State current_state = s0;
  output.push_back(r(s0));
  for (const auto& i : is) {
    current_state = f(current_state, i);
    output.push_back(r(current_state));
  }

  return output;
}
//                                                                          }}}1
// Using Moore fixpoint                                                     {{{1
auto fixpoint_coalg(const std::vector<Input>& is, MooreMachine<Input, State, Output> mm)
    -> std::vector<Output> {
  const auto s0 = mm.s0;
  const auto f = mm.tmap;
  const auto r = mm.rmap;

  std::vector<int> output(is.size());

  const MCoalg<State> sigma = [f, r](State s) -> M<State> {
    return {[s, f, r](Input i) { return f(s, i); }, r(s)};
  };

  const auto lambda = Lambda(sigma, s0);
  const auto [l0, o0] = lambda;
  const auto [l1, o1] = l0(is[0]);
  const auto [l2, o2] = l1(is[1]);
  const auto [l3, o3] = l2(is[2]);
  const auto [l4, o4] = l3(is[3]);
  const auto [l5, o5] = l4(is[4]);

  return {o0, o1, o2, o3, o4, o5};
}
//                                                                          }}}1
// stdlib_cpp                                                               {{{1
auto stdlib_cpp(const std::vector<Input>& is, MooreMachine<Input, State, Output> mm)
    -> std::vector<Output> {
  const auto [s0, f, r] = mm;
  std::vector<int> output(is.size());
  std::inclusive_scan(cbegin(is), cend(is), begin(output), f, 0);
  std::transform(cbegin(output), cend(output), begin(output), r);

  return output;
}
//                                                                          }}}1
// rxcpp_scan                                                               {{{1
auto rxcpp_scan(const std::vector<Input>& is, MooreMachine<Input, State, Output> mm)
    -> std::vector<Output> {
  const auto [s0, f, r] = mm;
  auto oi = rxcpp::observable<>::create<Input>([&](rxcpp::subscriber<Input> s) {
    for (auto each : is) s.on_next(each);
    s.on_completed();
  });

  std::vector<Output> output;
  auto us = oi.scan(0, f).map(r);
  us.subscribe([&output](Output v) { output.push_back(v); });

  return output;
}
//                                                                          }}}1
// range_exclusive_scan                                                     {{{1
auto range_exclusive_scan(const std::vector<Input>& is, MooreMachine<Input, State, Output> mm) -> std::vector<Output> {
  const auto [s0, f, r] = mm;

  using namespace ranges;
  const auto us = is | views::exclusive_scan(0, f) | views::transform(r);

  return std::vector(std::cbegin(us), std::cend(us));
}
//                                                                          }}}1

// main()                                                                   {{{1
int main() {
  std::vector<Input> is = {0, 1, 2, 3, 4};
  State s0 = 0;
  TransitionMap f = [](State c, Input x) -> State { return c + x; };
  ReadoutMap r = [](State c) -> Output { return c; };

  MooreMachine<Input, State, Output> mm = {s0, f, r};

  // clang-format off
  std::cout << "moore_machine_explicit = "
            << moore_machine_explicit(is, mm)
            << std::endl
            << "fixpoint_coalg         = "
            << fixpoint_coalg(is, mm) 
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
