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
      //clang-format on
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

} // namespace util

template <typename Clock = chrono::steady_clock>
inline constexpr auto double_to_duration(double t) {
  return chrono::duration_cast<typename Clock::duration>(
      chrono::duration<double, std::ratio<1>>(t));
}

template <typename T>
inline constexpr auto seconds_in(T t) {
  chrono::duration<double, std::ratio<1>> t_doub = t;
  return t_doub.count();
}

namespace tokyo {
  std::string bg_dark = "#1f2335";
  std::string bg = "#24283b";
  std::string bg_highlight = "#292e42";
  std::string terminal_black = "#414868";
  std::string fg = "#c0caf5";
  std::string fg_dark = "#a9b1d6";
  std::string fg_gutter = "#3b4261";
  std::string dark3 = "#545c7e";
  std::string comment = "#565f89";
  std::string dark5 = "#737aa2";
  std::string blue0 = "#3d59a1";
  std::string blue = "#7aa2f7";
  std::string cyan = "#7dcfff";
  std::string blue1 = "#2ac3de";
  std::string blue2 = "#0db9d7";
  std::string blue5 = "#89ddff";
  std::string blue6 = "#b4f9f8";
  std::string blue7 = "#394b70";
  std::string magenta = "#bb9af7";
  std::string magenta2 = "#ff007c";
  std::string purple = "#9d7cd8";
  std::string orange = "#ff9e64";
  std::string yellow = "#e0af68";
  std::string green = "#9ece6a";
  std::string green1 = "#73daca";
  std::string green2 = "#41a6b5";
  std::string teal = "#1abc9c";
  std::string red = "#f7768e";
  std::string red1 = "#db4b4b";
}

template <typename Data, typename Fn>
void output_and_plot(const std::string title,
    const std::string filename, const Data &data, Fn ref_func,
    double margin) {
  const std::string tubecolour = "#6699ff55";

  { // Plot to screen
    Gnuplot gp;
    std::string bg_colour = tokyo::bg_dark;
    std::string fg_colour = tokyo::fg_dark;
    std::string analyt_colour = tokyo::blue;
    std::string sim_colour = tokyo::magenta;
    // gp << "set term cairolatex pdf transparent\n";
    // gp << "set output \"" << filename + ".tex" << "\"\n";
    gp << "set term wxt background \"" << bg_colour << "\"\n";
    gp << "set title textcolor rgb \"" << fg_colour << "\"\n";
    gp << "set border linecolor rgb \"" << fg_colour << "\"\n";
    gp << "set xtics textcolor rgb \"" << fg_colour << "\"\n";
    gp << "set xlabel textcolor rgb \"" << fg_colour << "\"\n";
    gp << "set ylabel textcolor rgb \"" << fg_colour << "\"\n";
    gp << "set key textcolor rgb \"" << fg_colour << "\" box linecolor rgb \"" << fg_colour << "\"\n";
    gp << "set style line 1 lc rgb \"" << fg_colour << "\" lt 1 lw 2\n";
    gp << "set title '" << title << "'\n"
       << "plot '-' u 1:3 w l lc rgb '" << analyt_colour << "' title 'Analytical solution' "
       << ", '-' u 1:2 w l lc rgb '" << sim_colour << "' title 'test result'\n";

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
