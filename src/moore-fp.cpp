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

// Moore Coalgebra tooling                                                  {{{1

// M<S> = (I ⊸ S, O)
template <typename S>
using M = std::pair<std::function<S(Input)>, Output>;

// MCoalg = s → M<S> = S → ( I ⊸ S, O);
template <typename S>
using MCoalg = std::function<M<S>(S)>;

// Λ ≅ M<Λ> = (I ⊸ Λ, O) = (I ⊸ (I ⊸ (I ⊸ (⋯, O), O), O), O)
template <typename S>
struct Lambda : M<Lambda<S>> {
  const MCoalg<S> sigma;

  Lambda(MCoalg<S> sig, S s0) : sigma{sig} {
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

// list_by_hand(…)                                                          {{{1
auto list_by_hand(const std::vector<Input>& is, TransitionMap f, ReadoutMap r)
    -> std::vector<Output> {
  return {r(0),
          r(f(0, is[0])),
          r(f(f(0, is[0]), is[1])),
          r(f(f(f(0, is[0]), is[1]), is[2])),
          r(f(f(f(f(0, is[0]), is[1]), is[2]), is[3])),
          r(f(f(f(f(f(0, is[0]), is[1]), is[2]), is[3]), is[4]))};
}
//                                                                          }}}1
// Using Moore fixpoint                                                     {{{1
auto fixpoint_coalg(const std::vector<Input>& is, TransitionMap f, ReadoutMap r)
    -> std::vector<Output> {
  const State s0 = 0;
  const MCoalg<State> sigma = [f, r](State s) -> M<State> {
    return {[s, f, r](Input i) { return f(s, i); }, r(s)};
  };
  std::vector<int> output(is.size());

  const auto lambda = Lambda(sigma, s0);
  const auto [l0, o0] = lambda;
  const auto [l1, o1]   = l0(is[0]);
  const auto [l2, o2]   = l1(is[1]);
  const auto [l3, o3]   = l2(is[2]);
  const auto [l4, o4]   = l3(is[3]);
  const auto [l5, o5]   = l4(is[4]);

  return {o0, o1, o2, o3, o4, o5};
}
//                                                                          }}}1
// stdlib_cpp                                                               {{{1
auto stdlib_cpp(const std::vector<Input>& is, TransitionMap f, ReadoutMap r)
    -> std::vector<Output> {
  std::vector<int> output(is.size());
  std::inclusive_scan(cbegin(is), cend(is), begin(output), f, 0);
  std::transform(cbegin(output), cend(output), begin(output), r);

  return output;
}
//                                                                          }}}1
// rxcpp_scan                                                               {{{1
auto rxcpp_scan(const std::vector<Input>& is, TransitionMap f, ReadoutMap r)
    -> std::vector<Output> {
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
auto range_exclusive_scan(const std::vector<Input>& is, TransitionMap f,
                          ReadoutMap r) -> std::vector<Output> {
  using namespace ranges;
  const auto us = is | views::exclusive_scan(0, f) | views::transform(r);

  return std::vector(std::cbegin(us), std::cend(us));
}
//                                                                          }}}1

// main()                                                                   {{{1
int main() {
  const std::vector<Input> is = {0, 1, 2, 3, 4};

  constexpr auto f = [](State c, Input x) -> State { return c + x; };
  constexpr auto r = [](State c) -> Output { return c; };

  // clang-format off
  std::cout <<
    "by_hand_as_list      = " << list_by_hand(is, f, r)
      << std::endl;

  std::cout <<
    "fixpoint_coalg       = " << fixpoint_coalg(is, f, r) 
      << std::endl;

  std::cout << "rxcpp_scan           = " << rxcpp_scan(is, f, r)
            << " ← rxcpp’s scan drops initial value." << std::endl;

  std::cout << "lib_cpp              = " << stdlib_cpp(is, f, r)
            << " ← std::inclusive_scan drops initial value." << std::endl;

  std::cout << "range_exclusive_scan = " << range_exclusive_scan(is, f, r)
            << " ← exclusive_scan drops last value." << std::endl;
  // clang-format on
}
//                                                                          }}}1
