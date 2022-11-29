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

using S = int;
using O = int;
using I = int;

using TransitionMap = std::function<S(S, I)>;
using ReadoutMap = std::function<O(S)>;

class MooreFixpoint;

struct Lambda_val {
  std::function<MooreFixpoint(I)> lambda;
  O output;
};

class MooreFixpoint {
 public:
  Lambda_val value;

  MooreFixpoint(S c, TransitionMap f, ReadoutMap r) : c(c), tmap(f), rout(r) {
    value = {[this, f, r, c](I i) { return MooreFixpoint(tmap(c, i), f, r); },
             r(c)};
  }

  MooreFixpoint operator()(I i) {
    const auto next_state = tmap(c, i);
    return MooreFixpoint(next_state, tmap, rout);
  }

 private:
  S c;
  TransitionMap tmap;
  ReadoutMap rout;
};

auto list_by_hand(const std::vector<I>& is, TransitionMap f, ReadoutMap r)
    -> std::vector<O> {
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

auto fixpoint_class_by_hand(const std::vector<I>& is, TransitionMap f,
                            ReadoutMap r) -> std::vector<O> {
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

auto stdlib_cpp(const std::vector<I>& is, TransitionMap f, ReadoutMap r)
    -> std::vector<O> {
  std::vector<int> output(is.size());
  std::inclusive_scan(cbegin(is), cend(is), begin(output), f, 0);
  std::transform(cbegin(output), cend(output), begin(output), r);

  return output;
}

auto rxcpp_scan(const std::vector<I>& is, TransitionMap f, ReadoutMap r)
    -> std::vector<O> {
  auto oi = rxcpp::observable<>::create<I>([&](rxcpp::subscriber<I> s) {
    for (auto each : is) s.on_next(each);
    s.on_completed();
  });

  std::vector<O> output;
  auto us = oi.scan(0, f).map(r);
  us.subscribe([&output](O v) { output.push_back(v); });

  return output;
}

auto range_exclusive_scan(const std::vector<I>& is, TransitionMap f,
                          ReadoutMap r) -> std::vector<O> {
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
  const std::vector<I> is = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  constexpr auto f = [](S c, I x) -> S { return c + x; };
  constexpr auto r = [](S c) -> O { return c; };

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
