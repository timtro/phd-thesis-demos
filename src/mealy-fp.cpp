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

void by_hand(std::vector<I> is) {
  const auto [u0, l0] = MFP::make_MFP(0);
  const auto [u1, l1] = l0(is[0]);
  const auto [u2, l2] = l1(is[1]);
  const auto [u3, l3] = l2(is[2]);
  const auto [u4, l4] = l3(is[3]);
  const auto [u5, l5] = l4(is[4]);
  const auto [u6, l6] = l5(is[5]);
  const auto [u7, l7] = l6(is[6]);
  const auto [u8, l8] = l7(is[7]);
  const auto [u9, l9] = l8(is[8]);
  const auto [u10, l10] = l9(is[9]);

  std::cout << u0 << ' ' << u1 << ' ' << u2 << ' ' << u3 << ' ' << u4 << ' '
            << u5 << ' ' << u6 << ' ' << u7 << ' ' << u8 << ' ' << u9 << ' '
            << u10 << '\n';
}

void by_hand_as_list(std::vector<I> is) {
  std::vector<O> os = {
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
  for (auto each : os) std::cout << each << ' ';
  std::cout << '\n';
}

void lib_cpp(std::vector<I> is) {
  std::vector<int> output(is.size());
  std::exclusive_scan(cbegin(is), cend(is), begin(output), 0, f);
  std::transform(cbegin(output), cend(output), begin(output), r);
  for (auto each : output) std::cout << each << ' ';
  std::cout << '\n';
}

void rxcpp_scan(std::vector<I> is) {
  auto oi = rxcpp::observable<>::create<I>([&](rxcpp::subscriber<I> s) {
    for (auto each : is) s.on_next(each);
    s.on_completed();
  });

  auto us = oi.scan(0, f).map(r);
  us.subscribe([](O v) { std::cout << v << ' '; },
                 []() { printf("OnCompleted\n"); });
}

void range_v3(std::vector<I> is) {
  using namespace ranges;
  const auto us = is | views::exclusive_scan(0, f) | views::transform(r);
  std::cout << us << '\n';
}

int main() {
  const std::vector<I> is = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  by_hand(is);
  by_hand_as_list(is);
  rxcpp_scan(is);
  lib_cpp(is);
  range_v3(is);
}
