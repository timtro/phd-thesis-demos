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

  template <typename T>
  bool compare_vectors(
      const std::vector<T> &a, const std::vector<T> &b, T margin) {
    if (a.size() != b.size())
      return false;
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != Approx(b[i]).margin(margin)) {
        std::cout
            << a[i] << " @ idx[" << i << "] Should == " << b[i]
            << std::endl;
        return false;
      }
    }
    return true;
  }

} // namespace util

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
  double errSum;
  double error;
  double ctrlVal;
};

template <typename Clock = chrono::steady_clock>
auto pid_algebra(double kp, double ki, double kd) -> Hom<
    Doms<PIDState<Clock>, SignalPt<double, Clock>>,
    PIDState<Clock>> {
  return [kp, ki, kd](PIDState<Clock> prev,
             SignalPt<double, Clock> errSigl) -> PIDState<Clock> {
    const chrono::duration<double> deltaT =
        errSigl.time - prev.time;
    if (deltaT <= chrono::seconds{0})
      return prev;
    const auto errSum =
        std::fma(errSigl.value, deltaT.count(), prev.errSum);
    const auto dErr = (errSigl.value - prev.error) / deltaT.count();
    const auto ctrlVal =
        kp * errSigl.value + ki * errSum + kd * dErr;

    return {errSigl.time, errSum, errSigl.value, ctrlVal};
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
      dxdt[2] = 0.; // Control variable dynamics are external to
                    // integration.
    }
  };

  inline double lyapunov(
      const SimState &s, const SimState &setPoint = {0, 0, 0}) {
    const auto error = setPoint[0] - s[0];
    return error * error + s[1] * s[1];
  }
} // namespace sim

template <typename Data, typename Fn>
void plot_with_tube(const std::string title,
    const std::string filename, const Data &data, Fn ref_func,
    double margin) {
  const std::string tubecolour = "#6699ff55";

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

  gp << "pause mouse key\nwhile (mouse_char ne 'q') { pause mouse "
        "key; }\n";

  // {
  //   std::ofstream csv;
  //   csv.open(filename + ".csv");
  //   for (auto &each : data) {
  //     const double t = each.first;
  //     const double pos = each.second;
  //     csv << t << ", " << pos << ", " << ref_func(t) << std::endl;
  //   }
  // }
}

using std::cos;
using std::exp;
using std::sin;

namespace analyt {

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
