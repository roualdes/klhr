#pragma once

#include <windowedadaptation.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace klhr {

inline std::size_t post_transport_warmup_steps(
    const std::size_t warmup,
    const std::size_t initial_transport_steps) {
  return warmup > initial_transport_steps ?
    warmup - initial_transport_steps : 0;
}

// Returns zero outside post-transport warmup. Positive values are relative
// draws in the K-adaptation schedule.
inline std::size_t post_transport_warmup_draw(
    const std::size_t draw,
    const std::size_t warmup,
    const std::size_t initial_transport_steps) {
  if (draw <= initial_transport_steps || draw > warmup) {
    return 0;
  }
  return draw - initial_transport_steps;
}

// K=1 is the same independence kernel as K=0, so it is omitted. All
// nonzero levels are odd, avoiding the middle-rank self-transition atom.
inline std::vector<std::size_t> make_K_ladder(const std::size_t maximum) {
  const std::size_t cap = std::min<std::size_t>(
    maximum, static_cast<std::size_t>(std::numeric_limits<int>::max()));
  std::vector<std::size_t> ladder{0};
  std::size_t K = 3;
  while (K <= cap) {
    ladder.push_back(K);
    if (K > (cap - 1) / 2) {
      break;
    }
    K = 2 * K + 1;
  }
  return ladder;
}

inline std::size_t nearest_K_ladder_index(
    const std::vector<std::size_t>& ladder,
    const std::size_t K) {
  if (ladder.empty()) {
    return 0;
  }
  std::size_t best = 0;
  std::size_t best_distance = K;
  for (std::size_t i = 1; i < ladder.size(); ++i) {
    const std::size_t distance = ladder[i] > K ?
      ladder[i] - K : K - ladder[i];
    if (distance < best_distance) {
      best = i;
      best_distance = distance;
    }
  }
  return best;
}

struct KAdaptationConfig {
  bool enabled = true;
  // The ceiling is high because measured optima on the reference models sit
  // at K around 127 to 255, and each window only tests the incumbent's
  // immediate neighbours, so the ladder has to extend past the optimum for
  // the search to reach it.
  //
  // The starting level is deliberately *not* raised to match. Early in
  // warmup the line fit is at its worst, and a large K proposes at the far
  // antithetic quantile of that fit, which is exactly where a bad fit does
  // the most damage: on earnings, starting at 63 rather than 15 cost a
  // seed's worth of bulk-finding (36/40 against 37/40) with no gain, since
  // the adaptation converges to the same level from either start. Let the
  // search climb rather than beginning there.
  std::size_t initial_K = 15;
  std::size_t maximum_K = 511;
  std::size_t warmup_steps = 850;
  std::size_t windowsize = 50;
  std::size_t windowscale = 2;
  std::size_t dimension = 1;
  std::size_t minimum_window = 16;
  // Deepest lag the refresh guard inspects; two reproduces the original
  // lag-1/lag-2 test. Four catches period-3 and period-4 cycles that lag 2
  // alone reads as excellent mixing, and measured identically to two on the
  // reference models. Deeper than that the minimum over many noisy lag
  // estimates is biased low and starts rejecting healthy large-K
  // candidates: at eight lags, earnings fell from 168 to 108 min-ESS and
  // its selected K from a median of 47 down to 15.
  std::size_t refresh_lags = 4;
  // False disables interleaving, giving each candidate one contiguous run.
  bool interleave_trials = true;
  double minimum_acceptance = 0.5;
  double maximum_invalid_rate = 0.05;
  double minimum_refresh = 0.1;
  double tolerance = 1e-10;
};

struct KAdaptationObservation {
  double acceptance_probability = 0.0;
  double standardized_squared_jump = 0.0;
  bool valid = false;
  double log_density = std::numeric_limits<double>::quiet_NaN();
  bool radius_available = false;
  double radius = std::numeric_limits<double>::quiet_NaN();
};

struct KWindowSummary {
  std::size_t K = 0;
  std::size_t attempts = 0;
  double acceptance_probability = 0.0;
  double invalid_rate = 0.0;
  double soft_esjd = 0.0;
  double log_density_refresh = 0.0;
  bool radius_available = false;
  double radius_refresh = 0.0;
  // Diagnostic only, not used for selection. See ScalarLagStats.
  double log_density_integrated_time =
    std::numeric_limits<double>::quiet_NaN();
  bool reliable = false;
  bool safe = false;
};

struct KAdaptationUpdate {
  bool window_closed = false;
  bool finalized = false;
  std::size_t K = 0;
};

class KWindowedAdaptation {
public:
  explicit KWindowedAdaptation(
      const KAdaptationConfig& config = KAdaptationConfig{}) :
    config_(normalized_config_(config)),
    ladder_(make_K_ladder(config_.maximum_K)),
    windowed_adaptation_(config_.warmup_steps,
                         config_.windowsize,
                         config_.windowscale),
    candidate_windows_(ladder_.size()) {
    incumbent_index_ = nearest_K_ladder_index(ladder_, config_.initial_K);
    current_index_ = incumbent_index_;
    current_K_ = config_.initial_K;
    finalized_ = !config_.enabled || config_.warmup_steps == 0;
    if (!finalized_) {
      current_K_ = ladder_[current_index_];
      configure_window_(true);
    }
  }

  std::size_t K() const {
    return current_K_;
  }

  bool enabled() const {
    return config_.enabled && config_.warmup_steps > 0;
  }

  bool finalized() const {
    return finalized_;
  }

  std::size_t draw() const {
    return draw_;
  }

  std::size_t windows_closed() const {
    return windows_closed_;
  }

  const std::vector<std::size_t>& ladder() const {
    return ladder_;
  }

  const std::vector<std::size_t>& closures() const {
    return windowed_adaptation_.closures();
  }

  const std::vector<KWindowSummary>& last_summaries() const {
    return last_summaries_;
  }

  KAdaptationUpdate observe(const KAdaptationObservation& observation) {
    if (finalized_) {
      return {.window_closed = false,
              .finalized = true,
              .K = current_K_};
    }

    candidate_windows_[current_index_].update(observation);
    pooled_log_density_.update(observation.log_density);
    if (observation.radius_available) {
      pooled_radius_.update(observation.radius);
    }

    ++draw_;
    const bool scheduled_close = windowed_adaptation_.window_closed(draw_);
    const bool final_close = draw_ >= config_.warmup_steps;
    if (!scheduled_close && !final_close) {
      set_current_trial_(draw_ - window_start_);
      return {.window_closed = false,
              .finalized = false,
              .K = current_K_};
    }

    close_window_(final_close);
    return {.window_closed = true,
            .finalized = finalized_,
            .K = current_K_};
  }

private:
  struct ScalarMoments {
    std::size_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;

    void update(const double value) {
      if (!std::isfinite(value)) {
        return;
      }
      ++count;
      const double delta = value - mean;
      mean += delta / static_cast<double>(count);
      m2 += delta * (value - mean);
    }

    double variance() const {
      return count > 1 ? m2 / static_cast<double>(count - 1) : 0.0;
    }

    void reset() {
      count = 0;
      mean = 0.0;
      m2 = 0.0;
    }
  };

  // Squared successive differences at several lags. For a stationary series
  // E[(x_t - x_{t-k})^2] = 2 sigma^2 (1 - rho_k), so every lag reads off an
  // autocorrelation without storing the series: only a ring buffer of the
  // last max_lag values is kept, at O(max_lag) work per draw.
  struct ScalarLagStats {
    static constexpr std::size_t max_lag = 8;
    // A lag is only trusted once it has this many pairs, so short blocks
    // degrade gracefully to the lag-1/lag-2 test this replaced.
    static constexpr std::size_t minimum_pairs = 2;

    std::array<double, max_lag> recent{};
    std::array<double, max_lag> squared_difference{};
    std::array<std::size_t, max_lag> pairs{};
    std::size_t available = 0;
    std::size_t position = 0;
    std::size_t history = 0;

    void update(const double value) {
      if (!std::isfinite(value)) {
        return;
      }
      const std::size_t usable = std::min(available, max_lag);
      for (std::size_t k = 1; k <= usable; ++k) {
        const double earlier = recent[(position + max_lag - k) % max_lag];
        const double difference = value - earlier;
        const double squared = difference * difference;
        if (std::isfinite(squared)) {
          squared_difference[k - 1] += squared;
          ++pairs[k - 1];
        }
      }
      recent[position] = value;
      position = (position + 1) % max_lag;
      if (available < max_lag) {
        ++available;
      }
      ++history;
    }

    // A candidate's draws are spread over several blocks in the window, so
    // differences must never span a boundary: the level under test changes
    // there, and the gap is filled by other candidates' draws.
    void end_block() {
      available = 0;
      position = 0;
    }

    std::size_t trusted_lags() const {
      std::size_t lags = 0;
      while (lags < max_lag && pairs[lags] >= minimum_pairs) {
        ++lags;
      }
      return lags;
    }

    bool has_lags(const std::size_t minimum_count) const {
      return history >= minimum_count &&
        pairs[0] + 1 >= minimum_count &&
        pairs[1] + 2 >= minimum_count;
    }

    // Mixing rate, guarded against periodicity at any period the lags can
    // see. For a geometrically decaying chain (1 - rho_k)/k is flat in k, so
    // the minimum is just the mixing rate; for a chain cycling with period
    // p, lag p has 1 - rho_p near zero and the minimum collapses. That is
    // the pathology a near-deterministic overrelaxation level produces. With
    // only two trusted lags this reduces exactly to the previous
    // min(D*q1, 0.5*D*q2).
    double refresh(const std::size_t dimension,
                   const double reference_variance,
                   const double tolerance,
                   const std::size_t maximum_lags) const {
      if (!(reference_variance > tolerance) ||
          !std::isfinite(reference_variance)) {
        return 0.0;
      }
      const std::size_t lags = std::min(trusted_lags(), maximum_lags);
      if (lags < 2) {
        return 0.0;
      }
      const double D = static_cast<double>(std::max<std::size_t>(1, dimension));
      double smallest = std::numeric_limits<double>::infinity();
      for (std::size_t k = 1; k <= lags; ++k) {
        const double q = squared_difference[k - 1] /
          (2.0 * reference_variance * static_cast<double>(pairs[k - 1]));
        smallest = std::min(smallest, D * q / static_cast<double>(k));
      }
      return std::isfinite(smallest) ? std::max(0.0, smallest) : 0.0;
    }

    // Geyer initial-positive-sequence integrated autocorrelation time,
    // tau = 2 * sum_j (rho_2j + rho_2j+1) - 1, truncated at the first
    // non-positive pair. Reported for diagnostics only: it is deliberately
    // not the selection objective, because a near-deterministic two-cycle
    // has a tiny integrated time while being exactly the failure mode the
    // refresh guard exists to reject.
    double integrated_time(const double reference_variance,
                           const double tolerance) const {
      const double nan = std::numeric_limits<double>::quiet_NaN();
      if (!(reference_variance > tolerance) ||
          !std::isfinite(reference_variance)) {
        return nan;
      }
      const std::size_t lags = trusted_lags();
      if (lags < 2) {
        return nan;
      }
      const auto rho = [&](const std::size_t k) {
        return 1.0 - squared_difference[k - 1] /
          (2.0 * reference_variance * static_cast<double>(pairs[k - 1]));
      };
      double total = 0.0;
      for (std::size_t k = 0; k + 1 <= lags; k += 2) {
        const double pair_sum = (k == 0 ? 1.0 : rho(k)) + rho(k + 1);
        if (!(pair_sum > 0.0)) {
          break;
        }
        total += pair_sum;
      }
      return 2.0 * total - 1.0;
    }

    void reset() {
      recent.fill(0.0);
      squared_difference.fill(0.0);
      pairs.fill(0);
      available = 0;
      position = 0;
      history = 0;
    }
  };

  struct WindowStats {
    std::size_t attempts = 0;
    std::size_t valid = 0;
    double acceptance_probability = 0.0;
    std::vector<double> soft_esjd;
    ScalarLagStats log_density;
    ScalarLagStats radius;

    void update(const KAdaptationObservation& observation) {
      ++attempts;
      const double alpha = std::isfinite(observation.acceptance_probability) ?
        std::clamp(observation.acceptance_probability, 0.0, 1.0) : 0.0;
      acceptance_probability += alpha;
      valid += observation.valid;

      double jump = observation.standardized_squared_jump;
      if (!(jump >= 0.0) || !std::isfinite(jump)) {
        jump = 0.0;
      }
      soft_esjd.push_back(alpha * jump);
      log_density.update(observation.log_density);
      if (observation.radius_available) {
        radius.update(observation.radius);
      }
    }

    void end_block() {
      log_density.end_block();
      radius.end_block();
    }

    double robust_soft_esjd() const {
      if (soft_esjd.empty()) {
        return 0.0;
      }
      std::vector<double> ordered = soft_esjd;
      std::sort(ordered.begin(), ordered.end());
      const std::size_t cap_index = static_cast<std::size_t>(
        std::floor(0.95 * static_cast<double>(ordered.size() - 1)));
      const double cap = ordered[cap_index];
      double total = 0.0;
      for (const double value : ordered) {
        total += std::min(value, cap);
      }
      return total / static_cast<double>(ordered.size());
    }

    void reset() {
      attempts = 0;
      valid = 0;
      acceptance_probability = 0.0;
      soft_esjd.clear();
      log_density.reset();
      radius.reset();
    }
  };

  struct TrialBlock {
    std::size_t end = 0;
    std::size_t candidate = 0;
  };

  KAdaptationConfig config_;
  std::vector<std::size_t> ladder_;
  mcmcpp::WindowedAdaptation windowed_adaptation_;
  std::vector<WindowStats> candidate_windows_;
  ScalarMoments pooled_log_density_;
  ScalarMoments pooled_radius_;
  std::vector<std::size_t> active_indices_;
  std::vector<TrialBlock> trial_blocks_;
  std::vector<KWindowSummary> last_summaries_;
  std::size_t incumbent_index_ = 0;
  std::size_t current_index_ = 0;
  std::size_t current_K_ = 0;
  std::size_t draw_ = 0;
  std::size_t window_start_ = 0;
  std::size_t window_end_ = 0;
  std::size_t windows_closed_ = 0;
  bool calibration_window_ = true;
  bool finalized_ = false;

  static KAdaptationConfig normalized_config_(KAdaptationConfig config) {
    const auto int_max =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
    config.initial_K = std::min(config.initial_K, int_max);
    config.maximum_K = std::min(config.maximum_K, int_max);
    config.windowsize = std::max<std::size_t>(1, config.windowsize);
    config.windowscale = std::max<std::size_t>(1, config.windowscale);
    config.dimension = std::max<std::size_t>(1, config.dimension);
    config.minimum_window = std::max<std::size_t>(3, config.minimum_window);
    config.refresh_lags = std::clamp<std::size_t>(
      config.refresh_lags, 2, ScalarLagStats::max_lag);
    config.minimum_acceptance = std::isfinite(config.minimum_acceptance) ?
      std::clamp(config.minimum_acceptance, 0.0, 1.0) : 0.5;
    config.maximum_invalid_rate =
      std::isfinite(config.maximum_invalid_rate) ?
      std::clamp(config.maximum_invalid_rate, 0.0, 1.0) : 0.05;
    config.minimum_refresh = std::isfinite(config.minimum_refresh) ?
      std::max(0.0, config.minimum_refresh) : 0.1;
    if (!(config.tolerance > 0.0) || !std::isfinite(config.tolerance)) {
      config.tolerance = 1e-10;
    }
    return config;
  }

  std::size_t next_window_end_() const {
    for (const std::size_t closure : windowed_adaptation_.closures()) {
      if (closure > draw_) {
        return closure;
      }
    }
    return config_.warmup_steps;
  }

  void reset_window_stats_() {
    for (WindowStats& stats : candidate_windows_) {
      stats.reset();
    }
    pooled_log_density_.reset();
    pooled_radius_.reset();
    active_indices_.clear();
    trial_blocks_.clear();
  }

  void configure_window_(const bool calibration) {
    reset_window_stats_();
    calibration_window_ = calibration;
    window_start_ = draw_;
    window_end_ = next_window_end_();
    const std::size_t length = window_end_ > window_start_ ?
      window_end_ - window_start_ : 0;

    active_indices_.push_back(incumbent_index_);
    if (!calibration && length >= 2 * config_.minimum_window) {
      if (incumbent_index_ > 0) {
        active_indices_.push_back(incumbent_index_ - 1);
      }
      if (incumbent_index_ + 1 < ladder_.size()) {
        active_indices_.push_back(incumbent_index_ + 1);
      }
    }
    std::sort(active_indices_.begin(), active_indices_.end());

    if (active_indices_.size() == 3 &&
        length < 4 * config_.minimum_window) {
      // Prefer testing the lower neighbor when the window cannot support
      // both adjacent candidates with enough lag pairs.
      active_indices_.pop_back();
    }

    const std::size_t count = active_indices_.size();
    if (count <= 1) {
      if (length > 0 && count == 1) {
        trial_blocks_.push_back(
          {.end = length, .candidate = active_indices_.front()});
      }
      set_current_trial_(0);
      return;
    }

    // Candidates get equal shares, split into several interleaved blocks
    // rather than one contiguous run each. Warmup is not stationary, so a
    // contiguous layout confounds the comparison with whatever the chain
    // was doing at that point in the window. Reversing the order on
    // alternate rounds counterbalances a linear drift exactly, which is
    // stronger than randomising the order would be, and it stays
    // deterministic. Blocks are kept long enough to estimate the lags the
    // refresh guard uses, since differences cannot cross a boundary.
    const std::size_t per_candidate = length / count;
    std::size_t block = std::min<std::size_t>(4 * ScalarLagStats::max_lag,
                                              per_candidate);
    block = std::max(block, std::min(config_.minimum_window, per_candidate));
    const std::size_t rounds = (block > 0 && config_.interleave_trials) ?
      std::max<std::size_t>(1, per_candidate / block) : 1;

    std::size_t end = 0;
    for (std::size_t round = 0; round < rounds; ++round) {
      const std::size_t share = per_candidate / rounds +
        (round < per_candidate % rounds ? 1 : 0);
      if (share == 0) {
        continue;
      }
      const bool forward = (round + windows_closed_) % 2 == 0;
      for (std::size_t i = 0; i < count; ++i) {
        const std::size_t slot = forward ? i : count - 1 - i;
        end += share;
        trial_blocks_.push_back(
          {.end = end, .candidate = active_indices_[slot]});
      }
    }
    // Absorb the few draws that equal shares cannot cover into the final
    // block, rather than leaving a stray unbalanced run at every window end.
    if (!trial_blocks_.empty()) {
      trial_blocks_.back().end = length;
    }
    set_current_trial_(0);
  }

  void set_current_trial_(const std::size_t position) {
    const std::size_t previous = current_index_;
    std::size_t selected = incumbent_index_;
    for (const TrialBlock& block : trial_blocks_) {
      if (position < block.end) {
        selected = block.candidate;
        break;
      }
    }
    if (selected != previous && previous < candidate_windows_.size()) {
      // Leaving a block: drop the ring buffer so the next run of draws for
      // that candidate does not pair across the gap.
      candidate_windows_[previous].end_block();
    }
    current_index_ = selected;
    current_K_ = ladder_[current_index_];
  }

  KWindowSummary summarize_candidate_(
      const std::size_t index,
      const bool require_radius,
      const bool radius_evidence_available) const {
    const WindowStats& stats = candidate_windows_[index];
    KWindowSummary summary;
    summary.K = ladder_[index];
    summary.attempts = stats.attempts;
    if (stats.attempts == 0) {
      return summary;
    }

    const double attempts = static_cast<double>(stats.attempts);
    summary.acceptance_probability = stats.acceptance_probability / attempts;
    summary.invalid_rate =
      1.0 - static_cast<double>(stats.valid) / attempts;
    summary.soft_esjd = stats.robust_soft_esjd();
    summary.log_density_refresh = stats.log_density.refresh(
      config_.dimension, pooled_log_density_.variance(), config_.tolerance,
      config_.refresh_lags);
    summary.log_density_integrated_time =
      stats.log_density.integrated_time(
        pooled_log_density_.variance(), config_.tolerance);
    summary.radius_available = radius_evidence_available &&
      stats.radius.has_lags(config_.minimum_window);
    if (summary.radius_available) {
      summary.radius_refresh = stats.radius.refresh(
        config_.dimension, pooled_radius_.variance(), config_.tolerance,
        config_.refresh_lags);
    }

    summary.reliable =
      stats.attempts >= config_.minimum_window &&
      stats.log_density.has_lags(config_.minimum_window) &&
      (!require_radius || summary.radius_available);
    summary.safe = summary.reliable &&
      summary.acceptance_probability >= config_.minimum_acceptance &&
      summary.invalid_rate <= config_.maximum_invalid_rate &&
      summary.log_density_refresh >= config_.minimum_refresh &&
      (!require_radius ||
       summary.radius_refresh >= config_.minimum_refresh);
    return summary;
  }

  void select_incumbent_() {
    std::vector<const KWindowSummary*> safe;
    std::vector<const KWindowSummary*> reliable;
    for (const KWindowSummary& summary : last_summaries_) {
      if (summary.reliable) {
        reliable.push_back(&summary);
      }
      if (summary.safe) {
        safe.push_back(&summary);
      }
    }

    // Take the best safe candidate outright. There used to be a
    // near-optimal band that broke ties toward smaller K, which made sense
    // when each overrelaxation cost O(K) work to evaluate its transition
    // density. That density is gone from the Hastings ratio, so cost is now
    // flat in K -- measured at roughly 620k gradient evaluations for every
    // K from 0 to 1023 on earnings -- and the band only introduced a
    // systematic downward bias. On an exact tie prefer the larger level.
    if (!safe.empty()) {
      const KWindowSummary* selected = nullptr;
      for (const KWindowSummary* summary : safe) {
        if (selected == nullptr ||
            summary->soft_esjd > selected->soft_esjd ||
            (summary->soft_esjd == selected->soft_esjd &&
             summary->K > selected->K)) {
          selected = summary;
        }
      }
      if (selected != nullptr) {
        incumbent_index_ = nearest_K_ladder_index(ladder_, selected->K);
        return;
      }
    }

    // If every assessed candidate violates a safeguard, move only to the
    // smallest reliably tested level. Never freeze an untested fallback.
    if (!reliable.empty()) {
      const auto selected = *std::min_element(
        reliable.begin(), reliable.end(),
        [](const KWindowSummary* lhs, const KWindowSummary* rhs) {
          return lhs->K < rhs->K;
        });
      incumbent_index_ = nearest_K_ladder_index(ladder_, selected->K);
    }
  }

  void close_window_(const bool final_close) {
    ++windows_closed_;
    const bool require_radius = !calibration_window_;
    const bool radius_evidence_available =
      pooled_radius_.count >= config_.minimum_window;

    last_summaries_.clear();
    for (const std::size_t index : active_indices_) {
      last_summaries_.push_back(summarize_candidate_(
        index, require_radius, radius_evidence_available));
    }

    // The first window estimates the fixed diagonal metric and radius center
    // used by the first actual comparison window.
    if (!calibration_window_ &&
        (!require_radius || radius_evidence_available)) {
      select_incumbent_();
    }

    if (final_close) {
      finalized_ = true;
      current_index_ = incumbent_index_;
      current_K_ = ladder_[current_index_];
      return;
    }
    configure_window_(false);
  }
};

} // namespace klhr
