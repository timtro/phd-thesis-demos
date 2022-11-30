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

// Moore as σ                                                               {{{1
template <typename S>
using M = std::pair<std::function<S(Input)>, Output>;

template <typename S>
struct Sigma {
  TransitionMap tmap;
  ReadoutMap rout;
  auto operator()(S s) const -> M<S> {
    return {[s, this](Input i) { return tmap(s, i); }, rout(s)};
  }
};

template <typename S>
struct Lambda {
  Sigma<S> sig;
  M<Lambda> value;

  Lambda(Sigma<S> sig_, S s) : sig{sig_} {
    value = {[this, s](Input i) { return Lambda(sig, sig.tmap(s, i)); },
             sig.rout(s)};
  }
};

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
  return {
      r(0),
      r(f(0, is[0])),
      r(f(f(0, is[0]), is[1])),
      r(f(f(f(0, is[0]), is[1]), is[2])),
      r(f(f(f(f(0, is[0]), is[1]), is[2]), is[3])),
      r(f(f(f(f(f(0, is[0]), is[1]), is[2]), is[3]), is[4])),
      r(f(f(f(f(f(f(0, is[0]), is[1]), is[2]), is[3]), is[4]), is[5])),
      r(f(f(f(f(f(f(f(0, is[0]), is[1]), is[2]), is[3]), is[4]), is[5]),
          is[6])),
      r(f(f(f(f(f(f(f(f(0, is[0]), is[1]), is[2]), is[3]), is[4]), is[5]),
            is[6]),
          is[7])),
      r(f(f(f(f(f(f(f(f(f(0, is[0]), is[1]), is[2]), is[3]), is[4]), is[5]),
              is[6]),
            is[7]),
          is[8])),
      r(f(f(f(f(f(f(f(f(f(f(0, is[0]), is[1]), is[2]), is[3]), is[4]), is[5]),
                is[6]),
              is[7]),
            is[8]),
          is[9]))};
}
//                                                                          }}}1
// fixpoint_coalg                                                           {{{1
auto fixpoint_coalg(const std::vector<Input>& is, TransitionMap f,
                        ReadoutMap r) -> std::vector<Output> {
  const State s0 = 0;

  const auto [l0, o0] = Lambda{Sigma<State>{f, r}, s0}.value;
  const auto [l1, o1] = l0(is[0]).value;
  const auto [l2, o2] = l1(is[1]).value;
  const auto [l3, o3] = l2(is[2]).value;
  const auto [l4, o4] = l3(is[3]).value;
  const auto [l5, o5] = l4(is[4]).value;
  const auto [l6, o6] = l5(is[5]).value;
  const auto [l7, o7] = l6(is[6]).value;
  const auto [l8, o8] = l7(is[7]).value;
  const auto [l9, o9] = l8(is[8]).value;
  const auto [l10, o10] = l9(is[9]).value;
  return {o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10};
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

int main() {
  const std::vector<Input> is = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  constexpr auto f = [](State c, Input x) -> State { return c + x; };
  constexpr auto r = [](State c) -> Output { return c; };

  // clang-format off
  std::cout <<
    "by_hand_as_list(is)        = " << list_by_hand(is, f, r)
      << std::endl;

  std::cout <<
    "fixpoint_coalg(is)         = " << fixpoint_coalg(is, f, r) 
      << std::endl;

  std::cout << "rxcpp_scan(is)             = " << rxcpp_scan(is, f, r)
            << " ← rxcpp’s scan drops initial value." << std::endl;

  std::cout << "lib_cpp(is)                = " << stdlib_cpp(is, f, r)
            << " ← std::inclusive_scan drops initial value." << std::endl;

  std::cout << "range_exclusive_scan(is)   = " << range_exclusive_scan(is, f, r)
            << " ← exclusive_scan drops last value." << std::endl;
  // clang-format on
}
