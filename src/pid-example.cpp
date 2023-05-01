// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65
#include <boost/numeric/odeint.hpp>
#include <catch2/catch.hpp>
// #include <cmath>
// #include <iomanip>
// #include <iostream>

#include "pid.hpp"
#include "rx_scanl.hpp"

namespace ode = boost::numeric::odeint;
using CState = PIDState<>;                      // AKA U
using PState = SignalPt<std::array<double, 2>>; // AKA X
// NB:
// PState = SignalPt<std::array<double, 2>>
//          ┌                ┐
//          │        ┌   ┐   │
//        = │  · ,   │ · │   │
//          │        │ · │   │
//          │        └   ┘   │
//          └                ┘
//            ^ Time   ^ [Position, Speed]
//
// On the other hand, sim::SimState is just for Boost.odeint. It
// doesn't need time, but must be augmented with the control
// variable:
//
// sim::Pstate = std::array<double, 3>
//                ┌   ┐
//                │ · │  // Position
//             =  │ · │  // Speed
//                │ · │  // control variable for Boost.odeint.
//                └   ┘

constexpr double dt = 0.001; // seconds.
constexpr auto dts = util::double_to_duration(dt);
const auto now = chrono::steady_clock::now();

constexpr double mass = 1.;
constexpr double damp = 10. / mass;
constexpr double spring = 20. / mass;
constexpr double static_force = 1. / mass;
constexpr auto sim_duration = 2s; // seconds

// position_error : (PState, PState) → SignalPt<double>
inline SignalPt<double> position_error(
    const std::tuple<PState, PState> &ab) {
  const auto &[a, b] = ab;
  return {std::max(a.time, b.time), a.value[0] - b.value[0]};
}

struct WorldInterface {
  const rxcpp::subjects::behavior<PState> txSubject;
  const rxcpp::observable<PState> setPoint =
      // Setpoint to x = 1, for step response.
      rxcpp::observable<>::just(PState{now, {1., 0.}});

  WorldInterface(PState x0) : txSubject(x0) {}

  void controlled_step(CState u) {
    auto x = txSubject.get_value();
    sim::SimState xAugmented = {x.value[0], x.value[1], u.ctrlVal};

    if ((x.time - now) >= sim_duration)
      txSubject.get_subscriber().on_completed();

    // do_step uses the second argument for both input and output.
    stepper_.do_step(plant_, xAugmented, 0, dt);
    x.time += dts;
    x.value = {xAugmented[0], xAugmented[1]};
    txSubject.get_subscriber().on_next(x);
  };

  auto get_state_observable() { return txSubject.get_observable(); }
  auto time_elapsed() { return txSubject.get_value().time - now; }

private:
  ode::runge_kutta4<sim::SimState> stepper_;
  const sim::Plant plant_ = sim::Plant(static_force, damp, spring);
};

void step_response_test(const std::string test_title,
    const std::string filename, const double k_p, const double k_i,
    const double k_d,
    const std::function<double(double)> expected_fn,
    const double margin) {
  const CState u0 = {now, 0., 0., 0.};
  const PState x0 = {now, {0., 0.}};
  WorldInterface world_ix(x0);

  std::vector<PState> plant_states;
  world_ix.get_state_observable().subscribe([&](PState x) {
    plant_states.push_back(x);
  });

  // clang-format off
  const auto s_controls =
      world_ix.get_state_observable()
        | rx::combine_latest(world_ix.setPoint)
        | rx::map(&position_error)
        | rx::observe_on(rxcpp::identity_current_thread())
        | rx_scanl(u0, pid_algebra(k_p, k_i, k_d));
  // clang-format on

  s_controls.subscribe([&world_ix](CState u) {
    world_ix.controlled_step(u);
  });

  {
    auto simulated_positions = util::vec_map(
        [](auto x) { return x.value[0]; }, plant_states);
    auto theoretical_positions = util::vec_map(
        [&](auto x) {
          return expected_fn(util::unchrono_sec(x.time - now));
        },
        plant_states);

    const auto testData = util::vec_map(
        [](const auto &x) {
          return std::make_pair(
              util::unchrono_sec(x.time - now), x.value[0]);
        },
        plant_states);

    plot_with_tube(
        test_title, filename, testData, expected_fn, margin);

    REQUIRE(util::compare_vectors(
        simulated_positions, theoretical_positions, margin));
  }
}

TEST_CASE(
    "Given system and controller parameters, simulation "
    "should reproduce analytically computed step responses "
    "to within a margin of error. See src/calculations for "
    "details.") {
  SECTION("Test A (Proportional Control)") {
    constexpr double k_p = 300.;
    constexpr double k_i = 0.;
    constexpr double k_d = 0.;
    const auto title = "Test A; $(k_p, k_i, k_d) = (300., 0., 0.)$"s;
    const auto filename = "pid-test-a"s;
    step_response_test(
        title, filename, k_p, k_i, k_d, &analyt::test_a, 0.03);
  }

  SECTION("Test B (Proportional-Derivative Control)") {
    constexpr double k_p = 300.;
    constexpr double k_i = 0.;
    constexpr double k_d = 10.;
    const auto title = "Test B; $(k_p, k_i, k_d) = (300., 0., 10.)$"s;
    const auto filename = "pid-test-b"s;
    step_response_test(
        title, filename, k_p, k_i, k_d, &analyt::test_b, 0.03);
  }

  SECTION("Test C (Proportional-Integral Control)") {
    constexpr double k_p = 30.;
    constexpr double k_i = 70.;
    constexpr double k_d = 0.;
    const auto title = "Test C; $(k_p, k_i, k_d) = (30., 70., 0.)$"s;
    const auto filename = "pid-test-c"s;
    step_response_test(
        title, filename, k_p, k_i, k_d, &analyt::test_c, 0.03);
  }

  SECTION("Test D (Proportional-Integral-Derivative Control)") {
    constexpr double k_p = 350.;
    constexpr double k_i = 300.;
    constexpr double k_d = 50.;
    const auto title =
        "Test D; $(k_p, k_i, k_d) = (350., 300., 50.)$"s;
    const auto filename = "pid-test-d"s;
    step_response_test(
        title, filename, k_p, k_i, k_d, &analyt::test_d, 0.03);
  }
}
