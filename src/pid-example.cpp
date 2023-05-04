// vim: fdm=marker:fdc=2:fmr=f[[[,f]]]:tw=65
#include <boost/numeric/odeint.hpp>
#include <catch2/catch.hpp>
// #include <cmath>
// #include <iomanip>
// #include <iostream>

#include "pid.hpp"
#include "rx_scanl.hpp"

namespace ode = boost::numeric::odeint;

constexpr double dt = 0.001; // seconds.
constexpr auto dts = double_to_duration(dt);
const auto start = chrono::steady_clock::now();

constexpr double mass = 1.;
constexpr double damp = 10. / mass;
constexpr double spring = 20. / mass;
constexpr double static_force = 1. / mass;
constexpr auto sim_duration = 2s; // seconds

auto report_threshold_difference(const std::vector<double> &a,
    const std::vector<double> &b, double margin) -> uint {

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

struct WorldInterface {
  const rxcpp::subjects::behavior<PState> plant_subject;
  const rxcpp::observable<SetPt> setpoint =
      // Setpoint to x = 1, for step response.
      rxcpp::observable<>::just(1.);

  WorldInterface(PState x0) : plant_subject(x0) {}

  void controlled_step(double u) {
    auto x = plant_subject.get_value();
    sim::SimState x_sim = {x.value[0], x.value[1], u};

    if ((x.time - start) >= sim_duration)
      plant_subject.get_subscriber().on_completed();

    // do_step uses the second argument for both input and output.
    stepper_.do_step(plant_, x_sim, 0, dt);
    x.time += dts;
    x.value = {x_sim[0], x_sim[1]};
    plant_subject.get_subscriber().on_next(x);
  };

  auto get_plant_observable() {
    return plant_subject.get_observable();
  }

private:
  ode::runge_kutta4<sim::SimState> stepper_;
  const sim::Plant plant_ = sim::Plant(static_force, damp, spring);
};

inline auto position_error(PState const &x, SetPt const &setp)
    -> ErrPt {
  return {x.time, setp - x.value[0]};
}

auto pid_algebra(double k_p, double k_i, double k_d)
    -> Hom<Doms<CState, ErrPt>, CState> {
  return [k_p, k_i, k_d](CState prev_c, ErrPt cur_err) -> CState {
    const auto delta_t = seconds_in(cur_err.time - prev_c.time);

    // clang-format off
    if (delta_t <= 0) // Default to P-control if Δt ≤ 0
      return  {prev_c.time,
                { prev_c.value.error
                , prev_c.value.err_accum
                , k_p * cur_err.value
                }};
    // clang-format on

    const auto integ_err =
        std::fma(cur_err.value, delta_t, prev_c.value.err_accum);

    const auto diff_err =
        (cur_err.value - prev_c.value.error) / delta_t;

    const auto u =
        k_p * cur_err.value + k_i * integ_err + k_d * diff_err;

    return {cur_err.time, {integ_err, cur_err.value, u}};
  };
}

std::vector<std::pair<std::string, double>> rmses;

void step_response_test(const std::string test_title,
    const std::string filename, const double k_p, const double k_i,
    const double k_d,
    const std::function<double(double)> expected_fn,
    const double margin) {
  const CState c0 = {start, {0., 0., 0.}};
  const PState x0 = {start, {0., 0.}};
  WorldInterface world_ix(x0);

  std::vector<PState> plant_states;
  world_ix.get_plant_observable().subscribe([&](PState x) {
    plant_states.push_back(x);
  });

  // clang-format off
  const auto s_controls =
      world_ix.get_plant_observable()
        | rx::combine_latest(position_error, world_ix.setpoint)
        | rx::scan(c0, pid_algebra(k_p, k_i, k_d))
        | rx::map([](CState c) -> double { return c.value.u; });
  // clang-format on

  s_controls.subscribe([&world_ix](double u) {
    world_ix.controlled_step(u);
  });

  {
    auto simulated_positions = util::vec_map(
        [](auto x) { return x.value[0]; }, plant_states);
    auto theoretical_positions = util::vec_map(
        [&](auto x) {
          return expected_fn(seconds_in(x.time - start));
        },
        plant_states);

    const auto test_data = util::vec_map(
        [](const auto &x) {
          return std::make_pair(
              seconds_in(x.time - start), x.value[0]);
        },
        plant_states);

    output_and_plot(
        test_title, filename, test_data, expected_fn, margin);

    std::cout
        << "point-differences: "
        << report_threshold_difference(
               simulated_positions, theoretical_positions, margin)
        << std::endl;

    auto rmse = root_mean_sqr_error(
        simulated_positions, theoretical_positions);

    rmses.emplace_back(filename, rmse);

    REQUIRE(rmse < 0.01);
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

TEST_CASE(
    "Given system and controller parameters, simulation "
    "should reproduce analytically computed step responses "
    "to within a margin of error. See src/calculations for "
    "details.") {
  SECTION("Test A (Proportional Control)") {
    constexpr double k_p = 300.;
    constexpr double k_i = 0.;
    constexpr double k_d = 0.;
    const auto title =
        "Test A; $(k_p, k_i, k_d) = (300., 0., 0.)$"s;
    const auto filename = "pid-test-a"s;
    step_response_test(
        title, filename, k_p, k_i, k_d, &analyt::test_a, 0.01);
  }

  SECTION("Test B (Proportional-Derivative Control)") {
    constexpr double k_p = 300.;
    constexpr double k_i = 0.;
    constexpr double k_d = 10.;
    const auto title =
        "Test B; $(k_p, k_i, k_d) = (300., 0., 10.)$"s;
    const auto filename = "pid-test-b"s;
    step_response_test(
        title, filename, k_p, k_i, k_d, &analyt::test_b, 0.01);
  }

  SECTION("Test C (Proportional-Integral Control)") {
    constexpr double k_p = 30.;
    constexpr double k_i = 70.;
    constexpr double k_d = 0.;
    const auto title =
        "Test C; $(k_p, k_i, k_d) = (30., 70., 0.)$"s;
    const auto filename = "pid-test-c"s;
    step_response_test(
        title, filename, k_p, k_i, k_d, &analyt::test_c, 0.01);
  }

  SECTION("Test D (Proportional-Integral-Derivative Control)") {
    constexpr double k_p = 350.;
    constexpr double k_i = 300.;
    constexpr double k_d = 50.;
    const auto title =
        "Test D; $(k_p, k_i, k_d) = (350., 300., 50.)$"s;
    const auto filename = "pid-test-d"s;
    step_response_test(
        title, filename, k_p, k_i, k_d, &analyt::test_d, 0.01);
  }

  // Nastiness for RMSE table:
  std::cout << "       ";
  for (auto &each : rmses) {
    std::cout << each.first << "   ";
  }
  std::cout << std::endl << " RMSE  ";
  for (auto &each : rmses) {
    std::cout <<
        std::scientific << std::setprecision(3) << each.second << "    ";
  }
  std::cout << std::endl;
}
