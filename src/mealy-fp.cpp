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

using C = int;
using O = int;
using I = int;

C f(C c, I x) { return c + 1 * x; }

O r(C c) { return 1 * c; }

class MFP {
  C c;
  MFP(C c) : c{c} {}

 public:
  static std::pair<O, MFP> make_MFP(C c0) { return {r(c0), MFP{c0}}; }

  std::pair<O, MFP> operator()(I x) const { return make_MFP(f(c, x)); }
};

[[nodiscard]] std::vector<O> by_hand(std::vector<I> is) {
  const auto [o0, l0] = MFP::make_MFP(0);
  const auto [o1, l1] = l0(is[0]);
  const auto [o2, l2] = l1(is[1]);
  const auto [o3, l3] = l2(is[2]);
  const auto [o4, l4] = l3(is[3]);
  const auto [o5, l5] = l4(is[4]);
  const auto [o6, l6] = l5(is[5]);
  const auto [o7, l7] = l6(is[6]);
  const auto [o8, l8] = l7(is[7]);
  const auto [o9, l9] = l8(is[8]);
  const auto [o10, l10] = l9(is[9]);

  return {o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10};
}

[[nodiscard]] std::vector<O> by_hand_as_list(std::vector<I> is) {
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

std::vector<O> lib_cpp(std::vector<I> is) {
  std::vector<int> output(is.size());
  std::exclusive_scan(cbegin(is), cend(is), begin(output), 0, f);
  std::transform(cbegin(output), cend(output), begin(output), r);

  return output;
}

std::vector<O> rxcpp_scan(std::vector<I> is) {
  auto oi = rxcpp::observable<>::create<I>([&](rxcpp::subscriber<I> s) {
    for (auto each : is) s.on_next(each);
    s.on_completed();
  });

  std::vector<O> output;
  auto us = oi.scan(0, f).map(r);
  us.subscribe([&output](O v) { output.push_back(v); });

  return output;
}

std::vector<O> range_v3(std::vector<I> is) {
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

  std::cout << "by_hand(is)         = " << by_hand(is) << std::endl;
  std::cout << "by_hand_as_list(is) = " << by_hand_as_list(is) << std::endl;
  std::cout << "rxcpp_scan(is)      = " << rxcpp_scan(is) << std::endl;
  std::cout << "lib_cpp(is)         = " << lib_cpp(is) << std::endl;
  std::cout << "range_v3(is)        = " << range_v3(is) << std::endl;
}
