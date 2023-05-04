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
using Clock = chrono::steady_clock;

template <typename T>
struct SignalPt {
  chrono::time_point<Clock> time;
  T value;
};

struct PIDState {
  double err_accum;
  double error;
  double u;
};

//           ┌   ·   ┐  // time
// PState =  │ ┌ · ┐ │  // position
//           └ └ · ┘ ┘  // velocity
//           ^ ^
//           │ value
//           SignalPt
using PState = SignalPt<std::array<double, 2>>;

//           ┌   ·   ┐  // time
//           │ ┌ · ┐ │  // accumulated error
// CState =  │ │ · │ │  // error
//           └ └ · ┘ ┘  // control value
//           ^ ^
//           │ value
//           SignalPt
using CState = SignalPt<PIDState>;
using SetPt = double;
using ErrPt = SignalPt<double>;

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
      // clang-format off
      dxdt[0] = x[1];
      dxdt[1] = -spring_coef * x[0]
                  - damp_coef * x[1] 
                    + x[2] + static_force;

      dxdt[2] = 0.; // Control variable dynamics are external to
                    // integration.
      //clang-form on
    }
  };
} // namespace sim

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

  template <typename T>
  inline constexpr auto unchrono_sec(T t) {
    chrono::duration<double> t_doub = t;
    return t_doub.count();
  }
} // namespace util

template <typename Data, typename Fn>
void output_and_plot(const std::string title,
    const std::string filename, const Data &data, Fn ref_func,
    double margin) {
  const std::string tubecolour = "#6699ff55";

  { // Plot to screen
    Gnuplot gp;
    std::string dark_orange = "#ff8c00";
    std::string darker_orange = "#d55501";
    // gp << "set term cairolatex pdf transparent\n";
    // gp << "set output \"" << filename + ".tex" << "\"\n";
    gp << "set term wxt background \"#222222\"\n";
    gp << "set title textcolor rgb \"#cccccc\"\n";
    gp << "set border linecolor rgb \"#cccccc\"\n";
    gp << "set xtics textcolor rgb \"#cccccc\"\n";
    gp << "set xlabel textcolor rgb \"#cccccc\"\n";
    gp << "set ylabel textcolor rgb \"#cccccc\"\n";
    gp << "set key textcolor rgb \"#cccccc\" box linecolor rgb \"#cccccc\"\n";
    gp << "set style line 1 lc rgb \"#cccccc\" lt 1 lw 2\n";
    gp << "set title '" << title << "'\n"
       << "plot '-' u 1:3 w l lc rgb '" << darker_orange << "' title 'Analytical solution' "
       << ", '-' u 1:2 w l lc rgb '" << dark_orange << "' title 'test result'\n";

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
