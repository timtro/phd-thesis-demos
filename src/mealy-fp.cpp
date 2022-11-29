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

using State = int;
using Output = int;
using Input = int;

using TransitionMap = std::function<State(State, Input)>;
using ReadoutMap = std::function<Output(State)>;

template<typename S>
class MooreFixpoint;

template<typename S>
struct Lambda_val {
  std::function<MooreFixpoint<S>(Input)> lambda;
  Output output;
};

template<typename S>
class MooreFixpoint {
 public:
  Lambda_val<S> value;

  MooreFixpoint(S s, TransitionMap f, ReadoutMap r) : s(s), tmap(f), rout(r) {
    value = {[this, f, r, s](Input i) { return MooreFixpoint(tmap(s, i), f, r); },
             r(s)};
  }

  MooreFixpoint operator()(Input i) {
    const auto next_state = tmap(s, i);
    return MooreFixpoint(next_state, tmap, rout);
  }

 private:
  State s;
  TransitionMap tmap;
  ReadoutMap rout;
};

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

auto fixpoint_class_by_hand(const std::vector<Input>& is, TransitionMap f,
                            ReadoutMap r) -> std::vector<Output> {
  const auto Lambda = MooreFixpoint(0, f, r);

  const auto [l0, o0] = Lambda.value;
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

auto stdlib_cpp(const std::vector<Input>& is, TransitionMap f, ReadoutMap r)
    -> std::vector<Output> {
  std::vector<int> output(is.size());
  std::inclusive_scan(cbegin(is), cend(is), begin(output), f, 0);
  std::transform(cbegin(output), cend(output), begin(output), r);

  return output;
}

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

auto range_exclusive_scan(const std::vector<Input>& is, TransitionMap f,
                          ReadoutMap r) -> std::vector<Output> {
  using namespace ranges;
  const auto us = is | views::exclusive_scan(0, f) | views::transform(r);

  return std::vector(std::cbegin(us), std::cend(us));
}

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

int main() {
  const std::vector<Input> is = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  constexpr auto f = [](State c, Input x) -> State { return c + x; };
  constexpr auto r = [](State c) -> Output { return c; };

  // clang-format off
  std::cout <<
    "by_hand_as_list(is)        = " << list_by_hand(is, f, r)
      << std::endl;

  std::cout <<
    "fixpoint_class_by_hand(is) = " << fixpoint_class_by_hand(is, f, r) 
      << std::endl;

  std::cout << "rxcpp_scan(is)             = " << rxcpp_scan(is, f, r)
            << " ← rxcpp’s scan drops initial value." << std::endl;

  std::cout << "lib_cpp(is)                = " << stdlib_cpp(is, f, r)
            << " ← std::inclusive_scan drops initial value." << std::endl;

  std::cout << "range_exclusive_scan(is)   = " << range_exclusive_scan(is, f, r)
            << " ← exclusive_scan drops last value." << std::endl;
  // clang-format on
}
