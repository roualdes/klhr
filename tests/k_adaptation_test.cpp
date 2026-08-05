#include "k_adaptation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void check(const bool condition, const std::string_view expression,
           const int line) {
  if (!condition) {
    std::cerr << "line " << line << ": check failed: "
              << expression << '\n';
    ++failures;
  }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

// Period 13, deliberately longer than the deepest lag the refresh guard
// inspects. A short repeating pattern would make every multiple of its
// period look like a perfect cycle, which is the pathology the guard exists
// to reject -- a "healthy" fixture must not trip it.
klhr::KAdaptationObservation healthy_observation(
    const std::size_t index,
    const double squared_jump,
    const bool radius_available = false) {
  static constexpr double values[] = {
    0.0, 1.0, -0.5, 0.25, -1.0, 0.75, 0.5,
    -0.25, 1.25, -0.75, 0.1, -1.25, 0.6};
  const double value = values[index % 13];
  return {
    .acceptance_probability = 1.0,
    .standardized_squared_jump = squared_jump,
    .valid = true,
    .log_density = value,
    .radius_available = radius_available,
    .radius = value,
  };
}

// A clean cycle of the given period, for the periodicity tests.
klhr::KAdaptationObservation cyclic_observation(
    const std::size_t index,
    const std::size_t period,
    const double squared_jump) {
  const double value =
    static_cast<double>(index % period) - 0.5 * static_cast<double>(period - 1);
  return {
    .acceptance_probability = 1.0,
    .standardized_squared_jump = squared_jump,
    .valid = true,
    .log_density = value,
    .radius_available = true,
    .radius = value,
  };
}

void test_ladder() {
  CHECK(klhr::make_K_ladder(0) == std::vector<std::size_t>{0});
  CHECK(klhr::make_K_ladder(2) == std::vector<std::size_t>{0});
  CHECK(klhr::make_K_ladder(3) ==
        (std::vector<std::size_t>{0, 3}));
  CHECK(klhr::make_K_ladder(16) ==
        (std::vector<std::size_t>{0, 3, 7, 15}));
  CHECK(klhr::make_K_ladder(63) ==
        (std::vector<std::size_t>{0, 3, 7, 15, 31, 63}));

  const auto ladder = klhr::make_K_ladder(
    std::numeric_limits<std::size_t>::max());
  CHECK(!ladder.empty());
  CHECK(ladder.front() == 0);
  for (std::size_t i = 1; i < ladder.size(); ++i) {
    CHECK(ladder[i] % 2 == 1);
    CHECK(ladder[i] <=
          static_cast<std::size_t>(std::numeric_limits<int>::max()));
    if (i > 1) {
      CHECK(ladder[i] == 2 * ladder[i - 1] + 1);
    }
  }

  CHECK(klhr::nearest_K_ladder_index(
          std::vector<std::size_t>{0, 3, 7, 15, 31}, 16) == 3);
}

void test_post_transport_boundaries() {
  CHECK(klhr::post_transport_warmup_steps(20, 5) == 15);
  CHECK(klhr::post_transport_warmup_steps(5, 20) == 0);
  CHECK(klhr::post_transport_warmup_draw(5, 20, 5) == 0);
  CHECK(klhr::post_transport_warmup_draw(6, 20, 5) == 1);
  CHECK(klhr::post_transport_warmup_draw(20, 20, 5) == 15);
  CHECK(klhr::post_transport_warmup_draw(21, 20, 5) == 0);
}

void test_dedicated_schedule_and_freeze() {
  klhr::KWindowedAdaptation default_schedule({
    .warmup_steps = 850,
  });
  CHECK(default_schedule.closures() ==
        (std::vector<std::size_t>{50, 150, 350, 850}));

  klhr::KAdaptationConfig config{
    .initial_K = 3,
    .maximum_K = 7,
    .warmup_steps = 15,
    .windowsize = 2,
    .windowscale = 2,
    .minimum_window = 4,
  };
  klhr::KWindowedAdaptation adaptation(config);
  CHECK(adaptation.closures() ==
        (std::vector<std::size_t>{2, 6, 15}));

  std::vector<std::size_t> closed;
  for (std::size_t draw = 1; draw <= 15; ++draw) {
    const auto update = adaptation.observe(
      healthy_observation(draw - 1, 1.0));
    if (update.window_closed) {
      closed.push_back(draw);
    }
  }
  CHECK(closed == (std::vector<std::size_t>{2, 6, 15}));
  CHECK(adaptation.draw() == 15);
  CHECK(adaptation.windows_closed() == 3);
  CHECK(adaptation.finalized());

  const std::size_t frozen_K = adaptation.K();
  for (std::size_t i = 0; i < 10; ++i) {
    adaptation.observe(healthy_observation(i, 1000.0));
  }
  CHECK(adaptation.K() == frozen_K);
  CHECK(adaptation.draw() == 15);
  CHECK(adaptation.windows_closed() == 3);
}

void test_short_schedule_finalizes() {
  klhr::KAdaptationConfig config{
    .initial_K = 7,
    .maximum_K = 15,
    .warmup_steps = 3,
    .windowsize = 5,
    .minimum_window = 4,
  };
  klhr::KWindowedAdaptation adaptation(config);
  CHECK(adaptation.closures().empty());
  for (std::size_t draw = 0; draw < 3; ++draw) {
    adaptation.observe(healthy_observation(draw, 1.0));
  }
  CHECK(adaptation.finalized());
  CHECK(adaptation.windows_closed() == 1);
  CHECK(adaptation.K() == 7);
}

void test_fixed_mode_preserves_K() {
  klhr::KAdaptationConfig config{
    .enabled = false,
    .initial_K = 16,
    .warmup_steps = 20,
  };
  klhr::KWindowedAdaptation adaptation(config);
  CHECK(adaptation.finalized());
  CHECK(adaptation.K() == 16);
  adaptation.observe(healthy_observation(0, 1.0));
  CHECK(adaptation.draw() == 0);
  CHECK(adaptation.K() == 16);
}

void test_enabled_mode_snaps_to_ladder() {
  klhr::KWindowedAdaptation adaptation({
    .initial_K = 16,
    .maximum_K = 63,
    .warmup_steps = 20,
  });
  CHECK(adaptation.K() == 15);

  klhr::KWindowedAdaptation capped({
    .initial_K = 16,
    .maximum_K = 2,
    .warmup_steps = 20,
  });
  CHECK(capped.K() == 0);
}

void test_adjacent_candidates_share_one_window() {
  klhr::KAdaptationConfig config{
    .initial_K = 7,
    .maximum_K = 15,
    .warmup_steps = 32,
    .windowsize = 16,
    .windowscale = 1,
    .dimension = 1,
    .minimum_window = 4,
  };
  klhr::KWindowedAdaptation adaptation(config);
  for (std::size_t draw = 0; draw < 16; ++draw) {
    adaptation.observe(healthy_observation(draw, 1.0));
  }
  CHECK(adaptation.K() == 15);

  std::vector<std::size_t> attempts(16, 0);
  for (std::size_t draw = 0; draw < 16; ++draw) {
    const std::size_t K = adaptation.K();
    const double score = static_cast<double>(K + 1);
    adaptation.observe(healthy_observation(
      attempts[K]++, score, true));
  }
  CHECK(adaptation.finalized());
  CHECK(adaptation.K() == 15);
  CHECK(adaptation.last_summaries().size() == 3);
  CHECK(adaptation.last_summaries()[0].K == 3);
  CHECK(adaptation.last_summaries()[1].K == 7);
  CHECK(adaptation.last_summaries()[2].K == 15);
}

void test_lag_two_periodicity_moves_down() {
  klhr::KAdaptationConfig config{
    .initial_K = 15,
    .maximum_K = 15,
    .warmup_steps = 32,
    .windowsize = 16,
    .windowscale = 1,
    .dimension = 1,
    .minimum_window = 4,
  };
  klhr::KWindowedAdaptation adaptation(config);
  for (std::size_t draw = 0; draw < 16; ++draw) {
    adaptation.observe(healthy_observation(draw, 1.0));
  }
  CHECK(adaptation.K() == 15);

  std::vector<std::size_t> attempts(16, 0);
  for (std::size_t draw = 0; draw < 16; ++draw) {
    const std::size_t K = adaptation.K();
    const std::size_t index = attempts[K]++;
    if (K == 15) {
      const double periodic = index % 2 == 0 ? 1.0 : -1.0;
      adaptation.observe({
        .acceptance_probability = 1.0,
        .standardized_squared_jump = 100.0,
        .valid = true,
        .log_density = periodic,
        .radius_available = true,
        .radius = periodic,
      });
    } else {
      adaptation.observe(healthy_observation(index, 1.0, true));
    }
  }
  CHECK(adaptation.finalized());
  CHECK(adaptation.K() == 7);
}

// Selection used to apply a near-optimal band that broke ties toward the
// smaller K, so a larger level that was better by less than the band width
// could never win. Cost is flat in K now, so selection is a plain argmax and
// a small gain at the larger level is taken.
void test_small_gain_at_larger_K_is_taken() {
  klhr::KAdaptationConfig config{
    .initial_K = 7,
    .maximum_K = 15,
    .warmup_steps = 32,
    .windowsize = 16,
    .windowscale = 1,
    .dimension = 1,
    .minimum_window = 4,
  };
  klhr::KWindowedAdaptation adaptation(config);
  for (std::size_t draw = 0; draw < 16; ++draw) {
    adaptation.observe(healthy_observation(draw, 1.0));
  }

  std::vector<std::size_t> attempts(16, 0);
  for (std::size_t draw = 0; draw < 16; ++draw) {
    const std::size_t K = adaptation.K();
    // Monotone in K but all within the old 5% band of the best.
    const double score = K == 3 ? 10.0 : (K == 7 ? 10.2 : 10.4);
    adaptation.observe(healthy_observation(
      attempts[K]++, score, true));
  }
  CHECK(adaptation.finalized());
  CHECK(adaptation.K() == 15);
}

// A larger level that is genuinely worse must still lose.
void test_worse_larger_level_is_rejected() {
  klhr::KAdaptationConfig config{
    .initial_K = 7,
    .maximum_K = 15,
    .warmup_steps = 32,
    .windowsize = 16,
    .windowscale = 1,
    .dimension = 1,
    .minimum_window = 4,
  };
  klhr::KWindowedAdaptation adaptation(config);
  for (std::size_t draw = 0; draw < 16; ++draw) {
    adaptation.observe(healthy_observation(draw, 1.0));
  }

  std::vector<std::size_t> attempts(16, 0);
  for (std::size_t draw = 0; draw < 16; ++draw) {
    const std::size_t K = adaptation.K();
    const double score = K == 3 ? 50.0 : (K == 7 ? 10.0 : 1.0);
    adaptation.observe(healthy_observation(
      attempts[K]++, score, true));
  }
  CHECK(adaptation.finalized());
  CHECK(adaptation.K() == 3);
}

// A period-4 cycle has rho_2 near -1, so the old min(q1, q2/2) guard read it
// as excellent mixing and let it through. Inspecting more lags catches it:
// lag 4 has (1 - rho_4) near zero.
void test_longer_period_cycle_is_rejected() {
  klhr::KAdaptationConfig config{
    .initial_K = 15,
    .maximum_K = 15,
    .warmup_steps = 64,
    .windowsize = 32,
    .windowscale = 1,
    .dimension = 1,
    .minimum_window = 4,
  };
  klhr::KWindowedAdaptation adaptation(config);
  for (std::size_t draw = 0; draw < 32; ++draw) {
    adaptation.observe(healthy_observation(draw, 1.0, true));
  }
  CHECK(adaptation.K() == 15);

  std::vector<std::size_t> attempts(16, 0);
  for (std::size_t draw = 0; draw < 32; ++draw) {
    const std::size_t K = adaptation.K();
    const std::size_t index = attempts[K]++;
    if (K == 15) {
      // Large jumps, but the chain only ever visits four states in order.
      adaptation.observe(cyclic_observation(index, 4, 100.0));
    } else {
      adaptation.observe(healthy_observation(index, 1.0, true));
    }
  }
  CHECK(adaptation.finalized());
  CHECK(adaptation.K() == 7);
}

// Candidates are tested in several interleaved blocks with the order
// reversed on alternate rounds, so a linear drift through the window
// contributes equally to each of them.
void test_trials_are_interleaved_and_counterbalanced() {
  klhr::KAdaptationConfig config{
    .initial_K = 7,
    .maximum_K = 15,
    .warmup_steps = 512,
    .windowsize = 256,
    .windowscale = 1,
    .dimension = 1,
    .minimum_window = 16,
  };
  klhr::KWindowedAdaptation adaptation(config);
  for (std::size_t draw = 0; draw < 256; ++draw) {
    adaptation.observe(healthy_observation(draw, 1.0, true));
  }

  std::vector<std::size_t> order;
  std::vector<std::size_t> attempts(16, 0);
  for (std::size_t draw = 0; draw < 256; ++draw) {
    const std::size_t K = adaptation.K();
    if (order.empty() || order.back() != K) {
      order.push_back(K);
    }
    adaptation.observe(healthy_observation(attempts[K]++, 1.0, true));
  }

  // Contiguous allocation would give exactly one run per candidate.
  CHECK(order.size() >= 5);
  std::vector<std::size_t> reversed(order.rbegin(), order.rend());
  CHECK(order == reversed);

  // Equal shares, up to the leftover draws that fall to the incumbent.
  const auto& summaries = adaptation.last_summaries();
  CHECK(summaries.size() == 3);
  std::size_t lowest = std::numeric_limits<std::size_t>::max();
  std::size_t highest = 0;
  for (const auto& summary : summaries) {
    lowest = std::min(lowest, summary.attempts);
    highest = std::max(highest, summary.attempts);
    CHECK(summary.reliable);
    CHECK(std::isfinite(summary.log_density_integrated_time));
  }
  CHECK(highest - lowest <= 2);
}

// The ladder must actually reach the levels the fixed-K sweeps preferred.
void test_default_ladder_reaches_measured_optima() {
  const auto ladder = klhr::make_K_ladder(
    klhr::KAdaptationConfig{}.maximum_K);
  const auto has = [&](const std::size_t K) {
    return std::find(ladder.begin(), ladder.end(), K) != ladder.end();
  };
  CHECK(has(127));
  CHECK(has(255));
  CHECK(has(511));
  // Default start sits mid-ladder so either direction is reachable within
  // the handful of window closures a warmup provides.
  const std::size_t start = klhr::nearest_K_ladder_index(
    ladder, klhr::KAdaptationConfig{}.initial_K);
  CHECK(start >= 2);
  CHECK(start + 2 < ladder.size());
}

void test_acceptance_and_radius_safeguards() {
  klhr::KAdaptationConfig config{
    .initial_K = 7,
    .maximum_K = 15,
    .warmup_steps = 32,
    .windowsize = 16,
    .windowscale = 1,
    .dimension = 1,
    .minimum_window = 4,
  };
  klhr::KWindowedAdaptation low_acceptance(config);
  for (std::size_t draw = 0; draw < 16; ++draw) {
    low_acceptance.observe(healthy_observation(draw, 1.0));
  }
  std::vector<std::size_t> attempts(16, 0);
  for (std::size_t draw = 0; draw < 16; ++draw) {
    const std::size_t K = low_acceptance.K();
    auto observation = healthy_observation(
      attempts[K]++, 100.0, true);
    if (K != 3) {
      observation.acceptance_probability = 0.1;
    }
    low_acceptance.observe(observation);
  }
  CHECK(low_acceptance.finalized());
  CHECK(low_acceptance.K() == 3);

  klhr::KWindowedAdaptation periodic_radius(config);
  for (std::size_t draw = 0; draw < 16; ++draw) {
    periodic_radius.observe(healthy_observation(draw, 1.0));
  }
  attempts.assign(16, 0);
  for (std::size_t draw = 0; draw < 16; ++draw) {
    const std::size_t K = periodic_radius.K();
    const std::size_t index = attempts[K]++;
    auto observation = healthy_observation(index, 100.0, true);
    if (K != 3) {
      observation.radius = index % 2 == 0 ? 1.0 : -1.0;
    }
    periodic_radius.observe(observation);
  }
  CHECK(periodic_radius.finalized());
  CHECK(periodic_radius.K() == 3);
}

} // namespace

int main() {
  test_ladder();
  test_post_transport_boundaries();
  test_dedicated_schedule_and_freeze();
  test_short_schedule_finalizes();
  test_fixed_mode_preserves_K();
  test_enabled_mode_snaps_to_ladder();
  test_adjacent_candidates_share_one_window();
  test_lag_two_periodicity_moves_down();
  test_small_gain_at_larger_K_is_taken();
  test_worse_larger_level_is_rejected();
  test_longer_period_cycle_is_rejected();
  test_trials_are_interleaved_and_counterbalanced();
  test_default_ladder_reaches_measured_optima();
  test_acceptance_and_radius_safeguards();
  return failures == 0 ? 0 : 1;
}
