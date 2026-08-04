#pragma once

#include <windowedadaptation.hpp>

#include <algorithm>
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
  std::size_t initial_K = 16;
  std::size_t maximum_K = 63;
  std::size_t warmup_steps = 850;
  std::size_t windowsize = 50;
  std::size_t windowscale = 2;
  std::size_t dimension = 1;
  std::size_t minimum_window = 16;
  double minimum_acceptance = 0.5;
  double maximum_invalid_rate = 0.05;
  double minimum_refresh = 0.1;
  double near_optimal_fraction = 0.05;
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

  struct ScalarLagStats {
    std::size_t history = 0;
    double previous = 0.0;
    double previous2 = 0.0;
    std::size_t lag1_count = 0;
    std::size_t lag2_count = 0;
    double lag1_squared_difference = 0.0;
    double lag2_squared_difference = 0.0;

    void update(const double value) {
      if (!std::isfinite(value)) {
        return;
      }
      if (history >= 1) {
        const double difference = value - previous;
        const double squared = difference * difference;
        if (std::isfinite(squared)) {
          lag1_squared_difference += squared;
          ++lag1_count;
        }
      }
      if (history >= 2) {
        const double difference = value - previous2;
        const double squared = difference * difference;
        if (std::isfinite(squared)) {
          lag2_squared_difference += squared;
          ++lag2_count;
        }
      }
      previous2 = previous;
      previous = value;
      ++history;
    }

    bool has_lags(const std::size_t minimum_count) const {
      return history >= minimum_count &&
        lag1_count + 1 >= minimum_count &&
        lag2_count + 2 >= minimum_count;
    }

    double refresh(const std::size_t dimension,
                   const double reference_variance,
                   const double tolerance) const {
      if (!(reference_variance > tolerance) ||
          !std::isfinite(reference_variance) ||
          lag1_count == 0 || lag2_count == 0) {
        return 0.0;
      }
      const double q1 = lag1_squared_difference /
        (2.0 * reference_variance * static_cast<double>(lag1_count));
      const double q2 = lag2_squared_difference /
        (2.0 * reference_variance * static_cast<double>(lag2_count));
      const double D = static_cast<double>(std::max<std::size_t>(1, dimension));
      return std::max(0.0, std::min(D * q1, 0.5 * D * q2));
    }

    void reset() {
      history = 0;
      previous = 0.0;
      previous2 = 0.0;
      lag1_count = 0;
      lag2_count = 0;
      lag1_squared_difference = 0.0;
      lag2_squared_difference = 0.0;
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
    config.minimum_acceptance = std::isfinite(config.minimum_acceptance) ?
      std::clamp(config.minimum_acceptance, 0.0, 1.0) : 0.5;
    config.maximum_invalid_rate =
      std::isfinite(config.maximum_invalid_rate) ?
      std::clamp(config.maximum_invalid_rate, 0.0, 1.0) : 0.05;
    config.minimum_refresh = std::isfinite(config.minimum_refresh) ?
      std::max(0.0, config.minimum_refresh) : 0.1;
    config.near_optimal_fraction =
      std::isfinite(config.near_optimal_fraction) ?
      std::clamp(config.near_optimal_fraction, 0.0, 1.0) : 0.05;
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

    std::vector<std::pair<std::size_t, std::size_t>> candidates;
    if (active_indices_.size() == 1) {
      candidates.push_back({active_indices_.front(), length});
    } else if (active_indices_.size() == 2) {
      const std::size_t first = length / 2;
      candidates.push_back({active_indices_[0], first});
      candidates.push_back({active_indices_[1], length - first});
    } else {
      const std::size_t lower = length / 4;
      const std::size_t incumbent = length / 2;
      candidates.push_back({active_indices_[0], lower});
      candidates.push_back({active_indices_[1], incumbent});
      candidates.push_back({active_indices_[2], length - lower - incumbent});
    }

    if (windows_closed_ % 2 == 1) {
      std::reverse(candidates.begin(), candidates.end());
    }
    std::size_t end = 0;
    for (const auto& [candidate, count] : candidates) {
      if (count == 0) {
        continue;
      }
      end += count;
      trial_blocks_.push_back({.end = end, .candidate = candidate});
    }
    set_current_trial_(0);
  }

  void set_current_trial_(const std::size_t position) {
    for (const TrialBlock& block : trial_blocks_) {
      if (position < block.end) {
        current_index_ = block.candidate;
        current_K_ = ladder_[current_index_];
        return;
      }
    }
    current_index_ = incumbent_index_;
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
      config_.dimension, pooled_log_density_.variance(), config_.tolerance);
    summary.radius_available = radius_evidence_available &&
      stats.radius.has_lags(config_.minimum_window);
    if (summary.radius_available) {
      summary.radius_refresh = stats.radius.refresh(
        config_.dimension, pooled_radius_.variance(), config_.tolerance);
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

    if (!safe.empty()) {
      double best_score = 0.0;
      for (const KWindowSummary* summary : safe) {
        best_score = std::max(best_score, summary->soft_esjd);
      }
      const double threshold =
        (1.0 - config_.near_optimal_fraction) * best_score;
      const KWindowSummary* selected = nullptr;
      for (const KWindowSummary* summary : safe) {
        if (summary->soft_esjd >= threshold &&
            (selected == nullptr || summary->K < selected->K)) {
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
