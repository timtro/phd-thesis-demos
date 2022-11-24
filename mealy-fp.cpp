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
using U = int;
using X = int;

C f(C c, X x) { return c + x * x; }

U r(C c) { return 2 * c; }

class MFP {
  C c;
  MFP(C c) : c{c} {}

 public:
  static std::pair<U, MFP> make_MFP(C c0) { return {r(c0), MFP{c0}}; }

  std::pair<U, MFP> operator()(X x) const { return make_MFP(f(c, x)); }
};

void by_hand(std::vector<X> xs) {
  const auto [u0, l0] = MFP::make_MFP(0);
  const auto [u1, l1] = l0(xs[0]);
  const auto [u2, l2] = l1(xs[1]);
  const auto [u3, l3] = l2(xs[2]);
  const auto [u4, l4] = l3(xs[3]);
  const auto [u5, l5] = l4(xs[4]);
  const auto [u6, l6] = l5(xs[5]);
  const auto [u7, l7] = l6(xs[6]);
  const auto [u8, l8] = l7(xs[7]);
  const auto [u9, l9] = l8(xs[8]);
  const auto [u10, l10] = l9(xs[9]);

  std::cout << u0 << ' ' << u1 << ' ' << u2 << ' ' << u3 << ' ' << u4 << ' '
            << u5 << ' ' << u6 << ' ' << u7 << ' ' << u8 << ' ' << u9 << ' '
            << u10 << '\n';
}

void by_hand_as_list(std::vector<X> xs) {
  std::vector<U> us = {
      r(0),
      r(f(0, xs[0])),
      r(f(f(0, xs[0]), xs[1])),
      r(f(f(f(0, xs[0]), xs[1]), xs[2])),
      r(f(f(f(f(0, xs[0]), xs[1]), xs[2]), xs[3])),
      r(f(f(f(f(f(0, xs[0]), xs[1]), xs[2]), xs[3]), xs[4])),
      r(f(f(f(f(f(f(0, xs[0]), xs[1]), xs[2]), xs[3]), xs[4]), xs[5])),
      r(f(f(f(f(f(f(f(0, xs[0]), xs[1]), xs[2]), xs[3]), xs[4]), xs[5]),
          xs[6])),
      r(f(f(f(f(f(f(f(f(0, xs[0]), xs[1]), xs[2]), xs[3]), xs[4]), xs[5]),
            xs[6]),
          xs[7])),
      r(f(f(f(f(f(f(f(f(f(0, xs[0]), xs[1]), xs[2]), xs[3]), xs[4]), xs[5]),
              xs[6]),
            xs[7]),
          xs[8])),
      r(f(f(f(f(f(f(f(f(f(f(0, xs[0]), xs[1]), xs[2]), xs[3]), xs[4]), xs[5]),
                xs[6]),
              xs[7]),
            xs[8]),
          xs[9]))};
  for (auto each : us) std::cout << each << ' ';
  std::cout << '\n';
}

void lib_cpp(std::vector<X> xs) {
  std::vector<int> output(xs.size());
  std::exclusive_scan(cbegin(xs), cend(xs), begin(output), 0, f);
  std::transform(cbegin(output), cend(output), begin(output), r);
  for (auto each : output) std::cout << each << ' ';
  std::cout << '\n';
}

void rxcpp_scan(std::vector<X> xs) {
  auto ox = rxcpp::observable<>::create<int>([&](rxcpp::subscriber<X> s) {
    for (auto each : xs) s.on_next(each);
    s.on_completed();
  });

  auto us = ox.scan(0, f).map(r);
  us.subscribe([](U v) { std::cout << v << ' '; },
                 []() { printf("OnCompleted\n"); });
}

void range_v3(std::vector<X> xs) {
  using namespace ranges;
  const auto us = xs | views::exclusive_scan(0, f) | views::transform(r);
  std::cout << us << '\n';
}

int main() {
  const std::vector<X> xs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  by_hand(xs);
  by_hand_as_list(xs);
  rxcpp_scan(xs);
  lib_cpp(xs);
  range_v3(xs);
}
