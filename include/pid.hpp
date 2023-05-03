// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65
#pragma once

#include "Cpp-BiCCC.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <string>

#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/view/exclusive_scan.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>

// For plot helpers at the end of the file.
#include "plotting/gnuplot-iostream.h"
#include <boost/format.hpp>

namespace chrono = std::chrono;
using namespace std::chrono_literals;
using namespace std::string_literals;

namespace util {
  template <typename T, typename A, typename Fn>
  auto vec_map(Fn f, const std::vector<T, A> &v) {
    using MapedToType = std::invoke_result_t<Fn, decltype(v[0])>;
    std::vector<MapedToType> result;
    result.reserve(v.size());

    for (const auto &each : v)
      result.push_back(f(each));

    return result;
  }

  template <typename Clock = chrono::steady_clock>
  inline constexpr auto double_to_duration(double t) {
    return chrono::duration_cast<typename Clock::duration>(
        chrono::duration<double>(t));
  }

  template <typename A>
  inline constexpr auto unchrono_sec(A t) {
    chrono::duration<double> tAsDouble = t;
    return tAsDouble.count();
  }
} // namespace util

template <typename T>
auto report_threshold_difference(const std::vector<T> &a,
    const std::vector<T> &b, T margin) -> uint {

  if (a.size() != b.size())
    throw std::invalid_argument("Input sequence size mismatch.");

  uint count = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (std::abs(a[i] - b[i]) > margin)
      count++;
  }

  return count;
}

auto root_mean_sqr_error(const std::vector<double> &simulated,
    const std::vector<double> &analytical) -> double {

  if (simulated.size() != analytical.size())
    throw std::invalid_argument("Input sequence size mismatch.");

  double sum_of_squares = std::inner_product(
      simulated.begin(), simulated.end(), analytical.begin(), 0.0,
      [](double accum, double val) { return accum + val; },
      [](double a, double b) { return (a - b) * (a - b); });

  return std::sqrt(
      sum_of_squares / static_cast<double>(simulated.size()));
}

template <typename T = double,
    typename Clock = chrono::steady_clock>
struct SignalPt {
  chrono::time_point<Clock> time;
  T value;
};

namespace Signal {
  template <typename A, typename F>
  [[nodiscard]] auto fmap(F f, const SignalPt<A> &a) {
    // using B = std::invoke_result_t<F, A>;
    return SignalPt{a.time, std::invoke(f, a.value)};
  }
} // namespace Signal

//             ┌   ·   ┐  // time
//             │ ┌ · ┐ │  // accumulated error
// PIDState =  │ │ · │ │  // error
//             └ └ · ┘ ┘  // control value
//             ↑ ↑
//             │ value
//             SignalPt
template <typename Clock = chrono::steady_clock>
struct PIDState {
  chrono::time_point<Clock> time;
  double err_accum;
  double error;
  double u;
};

template <typename Clock = chrono::steady_clock>
auto pid_algebra(double kp, double ki, double kd) -> Hom<
    Doms<PIDState<Clock>, SignalPt<double, Clock>>,
    PIDState<Clock>> {
  return [kp, ki, kd](PIDState<Clock> prev_c,
             SignalPt<double, Clock> cur_err) -> PIDState<Clock> {

    const chrono::duration<double> delta_t =
        cur_err.time - prev_c.time;
    if (delta_t <= chrono::seconds{0})
      return prev_c;

    const auto integ_err =
        std::fma(cur_err.value, delta_t.count(), prev_c.err_accum);

    const auto diff_err =
        (cur_err.value - prev_c.error) / delta_t.count();

    const auto u =
        kp * cur_err.value + ki * integ_err + kd * diff_err;

    return {cur_err.time, integ_err, cur_err.value, u};
  };
}

namespace sim {
  //            ┌ · ┐  // Position
  // SimState = │ · │  // Speed
  //            └ · ┘  // control variable
  using SimState = std::array<double, 3>;

  struct Plant {
    const double static_force;
    const double damp_coef;
    const double spring_coef;

    Plant(double force, double damp, double spring)
        : static_force(force), damp_coef(damp),
          spring_coef(spring) {}

    void operator()(const sim::SimState &x, sim::SimState &dxdt,
        double /*time*/) const {
      dxdt[0] = x[1];
      dxdt[1] =
          -spring_coef * x[0] - damp_coef * x[1] - x[2] +
          static_force;
      dxdt[2] = 100.; // Control variable dynamics are external to
                      // integration.
    }
  };

  inline double lyapunov(
      const SimState &s, const SimState &setpoint = {0, 0, 0}) {
    const auto error = setpoint[0] - s[0];
    return error * error + s[1] * s[1];
  }
} // namespace sim

template <typename Data, typename Fn>
void output_and_plot(const std::string title,
    const std::string filename, const Data &data, Fn ref_func,
    double margin) {
  const std::string tubecolour = "#6699ff55";

  { // Plot to screen
    Gnuplot gp;
    // gp << "set term cairolatex pdf transparent\n";
    // gp << "set output \"" << filename + ".tex" << "\"\n";
    gp << "set title '" << title << "'\n"
       << "plot '-' u 1:2:4 title 'acceptable margin: analytical $±"
       << boost::format("%.3f") % margin
       << "$' w filledcu fs solid fc rgb '" << tubecolour
       << "', '-' u 1:2 "
          "title 'test result' w l\n";

    auto range = util::vec_map(
        [&](auto x) {
          // with…
          const double t = x.first;
          const double analyt = ref_func(t);

          return std::make_tuple(
              t, analyt + margin, analyt, analyt - margin);
        },
        data);

    gp.send1d(range);
    gp.send1d(data);

    gp << "pause mouse close\n";
  }

  { // Output to CSV
    std::ofstream csv;
    csv.open(filename + ".csv");
    for (auto &each : data) {
      const double t = each.first;
      const double pos = each.second;
      csv << t << ", " << pos << ", " << ref_func(t) << std::endl;
    }
  }
}

namespace analyt {
  using std::cos;
  using std::exp;
  using std::sin;

  double test_a(double t) {
    return -0.27291 * exp(-5 * t) * sin(17.175 * t) -
           0.9375 * exp(-5 * t) * cos(17.175 * t) + 0.9375;
  }

  double test_b(double t) {
    return 0.042137 * exp(-10 * t) * sin(14.832 * t) -
           0.9375 * exp(-10 * t) * cos(14.832 * t) + 0.9375;
  }

  double test_c(double t) {
    return -0.86502 * exp(-3.9537 * t) * sin(4.2215 * t) -
           0.83773 * exp(-3.9537 * t) * cos(4.2215 * t) -
           0.16226 * exp(-2.0924 * t) + 0.99999;
  }

  double test_d(double t) {
    return -0.043992 * exp(-0.95693 * t) -
           0.017952 * exp(-5.899 * t) - 0.93805 * exp(-53.144 * t) +
           1.0;
  }

} // namespace analyt
