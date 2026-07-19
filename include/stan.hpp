#pragma once

#include "gradient_sampler_options.hpp"

#include <Eigen/Dense>
#include <bridgestan.hpp>
#include <rng.hpp>
#include <welford.hpp>
#include <windowedadaptation.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace klhr {

struct StanOptions {
  std::uint64_t seed = 0;
  std::size_t warmup = 1'000;
  std::size_t initial_buffer = 75;
  std::size_t terminal_buffer = 50;
  std::size_t windowsize = 25;
  std::size_t windowscale = 2;
  double target_accept = 0.8;
  double initial_stepsize = 1.0;
  double min_stepsize = 1e-12;
  double max_stepsize = 1e7;
  double dual_gamma = 0.05;
  double dual_kappa = 0.75;
  double dual_t0 = 10.0;
  double variance_floor = 1e-12;
  double variance_ceiling = 1e12;
  std::size_t max_tree_depth = 10;
  double max_delta_H = 1'000.0;
};

// Standalone diagonal-metric multinomial NUTS sampler. The transition and
// step-size initialization follow stan-dev/stan's base_nuts and base_hmc.
class Stan {
public:
  std::size_t nfev_ = 0;
  double acceptance_rate_ = 0.0;
  double log_density_ = -std::numeric_limits<double>::infinity();

  double accept_stat_ = 0.0;
  bool divergent_ = false;
  std::size_t n_leapfrog_ = 0;
  std::size_t tree_depth_ = 0;
  double energy_ = std::numeric_limits<double>::infinity();

  Stan(std::string stan_file, std::string json_file,
       const StanOptions& options = StanOptions{}) :
    bsm_(std::move(stan_file), std::move(json_file)),
    rng_(options.seed),
    std_uniform_(0.0, 1.0),
    std_normal_(0.0, 1.0),
    opts_(normalized_options_(options)),
    metric_start_(detail::metric_start(opts_)),
    metric_end_(detail::metric_end(opts_)),
    metric_windowed_adaptation_(metric_end_ - metric_start_,
                                opts_.windowsize, opts_.windowscale),
    online_moments_(bsm_.dim()) {

    if (opts_.seed == 0) {
      std::random_device rd;
      const std::uint64_t r1 = rd();
      const std::uint64_t r2 = rd();
      opts_.seed = (r1 << 32) ^ r2;
      if (opts_.seed == 0) {
        opts_.seed = 1;
      }
      rng_.seed(opts_.seed);
    }

    if (dim() <= 0) {
      throw std::invalid_argument(
        "Stan: NUTS requires a positive unconstrained dimension");
    }

    std::uniform_int_distribution<unsigned int> uniform_uint;
    mcmcpp::bsrng bsrng = bsm_.make_rng(uniform_uint(rng_));
    theta_ = bsm_.param_initialize(bsrng);

    const Eigen::Index D = dim();
    grad_ = Eigen::VectorXd::Zero(D);
    variance_ = Eigen::VectorXd::Ones(D);
    sqrt_variance_ = Eigen::VectorXd::Ones(D);
    stepsize_ = opts_.initial_stepsize;

    if (!theta_.allFinite() ||
        !evaluate_(theta_, log_density_, grad_)) {
      throw std::runtime_error(
        "Stan: initial position, density, or gradient is not finite");
    }

    find_good_initial_stepsize(stepsize_);
  }

  Eigen::Index dim() const {
    return bsm_.dim();
  }

  std::uint64_t seed() const {
    return opts_.seed;
  }

  double stepsize() const {
    return stepsize_;
  }

  Eigen::VectorXd variance() const {
    return variance_;
  }

  double accept_stat() const {
    return accept_stat_;
  }

  bool divergent() const {
    return divergent_;
  }

  std::size_t n_leapfrog() const {
    return n_leapfrog_;
  }

  std::size_t tree_depth() const {
    return tree_depth_;
  }

  double energy() const {
    return energy_;
  }

  Eigen::VectorXd draw() {
    ++draw_;
    transition_();

    const double delta = accept_stat_ - acceptance_rate_;
    acceptance_rate_ += delta / static_cast<double>(draw_);

    adapt_warmup_();
    return bsm_.param_constrain(theta_);
  }

  double find_good_initial_stepsize(double initial = 1.0) {
    double epsilon = clamp_stepsize_(initial);
    // Stan's initialization search uses a fixed 0.8 crossing threshold,
    // independently of the configured dual-averaging target.
    const double log_target = std::log(0.8);
    const double initial_delta_H = trial_delta_H_(epsilon);
    const int direction = initial_delta_H > log_target ? 1 : -1;

    while (true) {
      const double delta_H = trial_delta_H_(epsilon);
      if ((direction == 1 && !(delta_H > log_target)) ||
          (direction == -1 && !(delta_H < log_target))) {
        break;
      }

      const double next = direction == 1 ? 2.0 * epsilon : 0.5 * epsilon;
      if (next > opts_.max_stepsize || !std::isfinite(next)) {
        throw std::runtime_error(
          "Stan: step size exceeded its upper bound; "
          "the posterior may be improper");
      }
      if (next < opts_.min_stepsize || !(next > 0.0)) {
        throw std::runtime_error(
          "Stan: no acceptably small step size could be found; "
          "the posterior may not be continuous");
      }
      epsilon = next;
    }

    set_stepsize_(epsilon);
    restart_stepsize_adaptation_();
    return stepsize_;
  }

  double find_good_stepsize(double initial = 1.0) {
    return find_good_initial_stepsize(initial);
  }

private:
  struct PhasePoint {
    Eigen::VectorXd position;
    Eigen::VectorXd momentum;
    Eigen::VectorXd gradient;
    double log_density = -std::numeric_limits<double>::infinity();
  };

  mcmcpp::bsmodel bsm_;
  mcmcpp::rng rng_;
  std::uniform_real_distribution<double> std_uniform_;
  std::normal_distribution<double> std_normal_;
  StanOptions opts_;
  std::size_t metric_start_;
  std::size_t metric_end_;
  mcmcpp::WindowedAdaptation metric_windowed_adaptation_;
  mcmcpp::WelfordAccumulator online_moments_;

  Eigen::VectorXd theta_;
  Eigen::VectorXd grad_;
  Eigen::VectorXd variance_;
  Eigen::VectorXd sqrt_variance_;
  double stepsize_ = 1.0;
  std::size_t draw_ = 0;

  std::size_t dual_counter_ = 0;
  double dual_s_bar_ = 0.0;
  double dual_x_bar_ = 0.0;
  double dual_mu_ = 0.0;

  static StanOptions normalized_options_(StanOptions options) {
    options.windowsize = std::max<std::size_t>(1, options.windowsize);
    options.windowscale = std::max<std::size_t>(1, options.windowscale);

    if (options.warmup < 20) {
      options.initial_buffer = options.warmup;
      options.terminal_buffer = 0;
    } else {
      const bool stages_fit =
        options.initial_buffer <= options.warmup &&
        options.windowsize <= options.warmup - options.initial_buffer &&
        options.terminal_buffer <=
          options.warmup - options.initial_buffer - options.windowsize;
      if (!stages_fit) {
        options.initial_buffer = static_cast<std::size_t>(
          0.15 * static_cast<double>(options.warmup));
        options.terminal_buffer = static_cast<std::size_t>(
          0.10 * static_cast<double>(options.warmup));
        options.windowsize =
          options.warmup - options.initial_buffer - options.terminal_buffer;
      }
    }

    if (!(options.target_accept > 0.0 && options.target_accept < 1.0) ||
        !std::isfinite(options.target_accept)) {
      options.target_accept = 0.8;
    }
    if (!(options.min_stepsize > 0.0) ||
        !std::isfinite(options.min_stepsize)) {
      options.min_stepsize = 1e-12;
    }
    if (!(options.max_stepsize >= options.min_stepsize) ||
        !std::isfinite(options.max_stepsize)) {
      options.max_stepsize = std::max(1e7, options.min_stepsize);
    }
    if (!(options.initial_stepsize > 0.0) ||
        !std::isfinite(options.initial_stepsize)) {
      options.initial_stepsize = 1.0;
    }
    options.initial_stepsize = std::clamp(
      options.initial_stepsize, options.min_stepsize, options.max_stepsize);
    if (!(options.dual_gamma > 0.0) ||
        !std::isfinite(options.dual_gamma)) {
      options.dual_gamma = 0.05;
    }
    if (!(options.dual_kappa > 0.0) ||
        !std::isfinite(options.dual_kappa)) {
      options.dual_kappa = 0.75;
    }
    if (!(options.dual_t0 > 0.0) ||
        !std::isfinite(options.dual_t0)) {
      options.dual_t0 = 10.0;
    }
    if (!(options.variance_floor > 0.0) ||
        !std::isfinite(options.variance_floor)) {
      options.variance_floor = 1e-12;
    }
    if (!(options.variance_ceiling >= options.variance_floor) ||
        !std::isfinite(options.variance_ceiling)) {
      options.variance_ceiling =
        std::max(1e12, options.variance_floor);
    }
    options.max_tree_depth =
      std::max<std::size_t>(1, options.max_tree_depth);
    if (!(options.max_delta_H >= 0.0) ||
        !std::isfinite(options.max_delta_H)) {
      options.max_delta_H = 1'000.0;
    }
    return options;
  }

  static double hamiltonian_(const PhasePoint& z) {
    return -z.log_density + 0.5 * z.momentum.squaredNorm();
  }

  static double log_sum_exp_(const double a, const double b) {
    if (a == -std::numeric_limits<double>::infinity()) {
      return b;
    }
    if (b == -std::numeric_limits<double>::infinity()) {
      return a;
    }
    if (a == std::numeric_limits<double>::infinity() ||
        b == std::numeric_limits<double>::infinity()) {
      return std::numeric_limits<double>::infinity();
    }
    const double maximum = std::max(a, b);
    return maximum + std::log1p(std::exp(std::min(a, b) - maximum));
  }

  static bool compute_criterion_(const Eigen::VectorXd& p_sharp_minus,
                                 const Eigen::VectorXd& p_sharp_plus,
                                 const Eigen::VectorXd& rho) {
    return p_sharp_plus.dot(rho) > 0.0 &&
      p_sharp_minus.dot(rho) > 0.0;
  }

  bool evaluate_(const Eigen::VectorXd& position,
                 double& log_density,
                 Eigen::VectorXd& gradient) {
    gradient.setZero(dim());
    ++nfev_;
    try {
      bsm_.log_density_gradient_noe(position, log_density, gradient);
    } catch (const mcmcpp::error&) {
      log_density = -std::numeric_limits<double>::infinity();
      gradient.setZero();
      return false;
    }
    if (!std::isfinite(log_density) || !gradient.allFinite()) {
      log_density = -std::numeric_limits<double>::infinity();
      gradient.setZero();
      return false;
    }
    return true;
  }

  bool leapfrog_(PhasePoint& z,
                 const Eigen::VectorXd& scaled_stepsize) {
    if (!scaled_stepsize.allFinite()) {
      return false;
    }

    z.momentum +=
      0.5 * scaled_stepsize.cwiseProduct(z.gradient);
    z.position += scaled_stepsize.cwiseProduct(z.momentum);
    if (!z.position.allFinite() || !z.momentum.allFinite() ||
        !evaluate_(z.position, z.log_density, z.gradient)) {
      return false;
    }
    z.momentum +=
      0.5 * scaled_stepsize.cwiseProduct(z.gradient);
    return z.momentum.allFinite();
  }

  bool build_tree_(const std::size_t depth,
                   PhasePoint& z,
                   PhasePoint& z_propose,
                   Eigen::VectorXd& p_sharp_beg,
                   Eigen::VectorXd& p_sharp_end,
                   Eigen::VectorXd& rho,
                   Eigen::VectorXd& p_beg,
                   Eigen::VectorXd& p_end,
                   const Eigen::VectorXd& scaled_stepsize,
                   const double H0,
                   std::size_t& n_leapfrog,
                   double& log_sum_weight,
                   double& sum_metro_prob) {
    if (depth == 0) {
      const bool valid = leapfrog_(z, scaled_stepsize);
      ++n_leapfrog;
      z_propose = z;

      double h = valid ? hamiltonian_(z) :
        std::numeric_limits<double>::infinity();
      if (std::isnan(h)) {
        h = std::numeric_limits<double>::infinity();
      }
      if (!valid || h - H0 > opts_.max_delta_H) {
        divergent_ = true;
      }

      double delta_H = H0 - h;
      if (std::isnan(delta_H)) {
        delta_H = -std::numeric_limits<double>::infinity();
      }
      log_sum_weight = log_sum_exp_(log_sum_weight, delta_H);
      if (delta_H > 0.0) {
        sum_metro_prob += 1.0;
      } else if (std::isfinite(delta_H)) {
        sum_metro_prob += std::exp(delta_H);
      }

      p_sharp_beg = z.momentum;
      p_sharp_end = z.momentum;
      rho += z.momentum;
      p_beg = z.momentum;
      p_end = z.momentum;
      return !divergent_;
    }

    double log_sum_weight_init =
      -std::numeric_limits<double>::infinity();
    Eigen::VectorXd p_init_end(z.momentum.size());
    Eigen::VectorXd p_sharp_init_end(z.momentum.size());
    Eigen::VectorXd rho_init = Eigen::VectorXd::Zero(rho.size());

    const bool valid_init = build_tree_(
      depth - 1, z, z_propose, p_sharp_beg, p_sharp_init_end,
      rho_init, p_beg, p_init_end, scaled_stepsize, H0, n_leapfrog,
      log_sum_weight_init, sum_metro_prob);
    if (!valid_init) {
      return false;
    }

    PhasePoint z_propose_final = z;
    double log_sum_weight_final =
      -std::numeric_limits<double>::infinity();
    Eigen::VectorXd p_final_beg(z.momentum.size());
    Eigen::VectorXd p_sharp_final_beg(z.momentum.size());
    Eigen::VectorXd rho_final = Eigen::VectorXd::Zero(rho.size());

    const bool valid_final = build_tree_(
      depth - 1, z, z_propose_final, p_sharp_final_beg, p_sharp_end,
      rho_final, p_final_beg, p_end, scaled_stepsize, H0, n_leapfrog,
      log_sum_weight_final, sum_metro_prob);
    if (!valid_final) {
      return false;
    }

    const double log_sum_weight_subtree =
      log_sum_exp_(log_sum_weight_init, log_sum_weight_final);
    log_sum_weight =
      log_sum_exp_(log_sum_weight, log_sum_weight_subtree);

    if (log_sum_weight_final > log_sum_weight_subtree) {
      z_propose = std::move(z_propose_final);
    } else {
      const double probability =
        std::exp(log_sum_weight_final - log_sum_weight_subtree);
      if (std_uniform_(rng_) < probability) {
        z_propose = std::move(z_propose_final);
      }
    }

    Eigen::VectorXd rho_subtree = rho_init + rho_final;
    rho += rho_subtree;

    bool persist_criterion =
      compute_criterion_(p_sharp_beg, p_sharp_end, rho_subtree);
    rho_subtree = rho_init + p_final_beg;
    persist_criterion &=
      compute_criterion_(p_sharp_beg, p_sharp_final_beg, rho_subtree);
    rho_subtree = rho_final + p_init_end;
    persist_criterion &=
      compute_criterion_(p_sharp_init_end, p_sharp_end, rho_subtree);
    return persist_criterion;
  }

  void transition_() {
    PhasePoint z{
      .position = theta_,
      .momentum = normal_rng_(dim()),
      .gradient = grad_,
      .log_density = log_density_,
    };

    PhasePoint z_fwd = z;
    PhasePoint z_bck = z;
    PhasePoint z_sample = z;
    PhasePoint z_propose = z;

    Eigen::VectorXd p_fwd_fwd = z.momentum;
    Eigen::VectorXd p_sharp_fwd_fwd = z.momentum;
    Eigen::VectorXd p_fwd_bck = z.momentum;
    Eigen::VectorXd p_sharp_fwd_bck = z.momentum;
    Eigen::VectorXd p_bck_fwd = z.momentum;
    Eigen::VectorXd p_sharp_bck_fwd = z.momentum;
    Eigen::VectorXd p_bck_bck = z.momentum;
    Eigen::VectorXd p_sharp_bck_bck = z.momentum;
    Eigen::VectorXd rho = z.momentum;

    const double H0 = hamiltonian_(z);
    const Eigen::VectorXd scaled_stepsize =
      stepsize_ * sqrt_variance_;
    double log_sum_weight = 0.0;
    double sum_metro_prob = 0.0;
    std::size_t n_leapfrog = 0;

    divergent_ = false;
    tree_depth_ = 0;

    while (tree_depth_ < opts_.max_tree_depth) {
      Eigen::VectorXd rho_fwd = Eigen::VectorXd::Zero(rho.size());
      Eigen::VectorXd rho_bck = Eigen::VectorXd::Zero(rho.size());
      bool valid_subtree = false;
      double log_sum_weight_subtree =
        -std::numeric_limits<double>::infinity();

      if (std_uniform_(rng_) > 0.5) {
        z = z_fwd;
        rho_bck = rho;
        p_bck_fwd = p_fwd_fwd;
        p_sharp_bck_fwd = p_sharp_fwd_fwd;

        valid_subtree = build_tree_(
          tree_depth_, z, z_propose,
          p_sharp_fwd_bck, p_sharp_fwd_fwd,
          rho_fwd, p_fwd_bck, p_fwd_fwd,
          scaled_stepsize, H0, n_leapfrog,
          log_sum_weight_subtree, sum_metro_prob);
        z_fwd = z;
      } else {
        z = z_bck;
        rho_fwd = rho;
        p_fwd_bck = p_bck_bck;
        p_sharp_fwd_bck = p_sharp_bck_bck;

        valid_subtree = build_tree_(
          tree_depth_, z, z_propose,
          p_sharp_bck_fwd, p_sharp_bck_bck,
          rho_bck, p_bck_fwd, p_bck_bck,
          -scaled_stepsize, H0, n_leapfrog,
          log_sum_weight_subtree, sum_metro_prob);
        z_bck = z;
      }

      if (!valid_subtree) {
        break;
      }

      ++tree_depth_;
      if (log_sum_weight_subtree > log_sum_weight) {
        z_sample = z_propose;
      } else {
        const double probability =
          std::exp(log_sum_weight_subtree - log_sum_weight);
        if (std_uniform_(rng_) < probability) {
          z_sample = z_propose;
        }
      }
      log_sum_weight =
        log_sum_exp_(log_sum_weight, log_sum_weight_subtree);

      rho = rho_bck + rho_fwd;
      bool persist_criterion =
        compute_criterion_(p_sharp_bck_bck, p_sharp_fwd_fwd, rho);
      Eigen::VectorXd rho_extended = rho_bck + p_fwd_bck;
      persist_criterion &= compute_criterion_(
        p_sharp_bck_bck, p_sharp_fwd_bck, rho_extended);
      rho_extended = rho_fwd + p_bck_fwd;
      persist_criterion &= compute_criterion_(
        p_sharp_bck_fwd, p_sharp_fwd_fwd, rho_extended);
      if (!persist_criterion) {
        break;
      }
    }

    n_leapfrog_ = n_leapfrog;
    accept_stat_ = n_leapfrog_ > 0 ?
      sum_metro_prob / static_cast<double>(n_leapfrog_) : 0.0;
    if (!std::isfinite(accept_stat_)) {
      accept_stat_ = 0.0;
    }
    accept_stat_ = std::clamp(accept_stat_, 0.0, 1.0);

    energy_ = hamiltonian_(z_sample);
    theta_ = std::move(z_sample.position);
    grad_ = std::move(z_sample.gradient);
    log_density_ = z_sample.log_density;
  }

  double trial_delta_H_(const double epsilon) {
    PhasePoint trial{
      .position = theta_,
      .momentum = normal_rng_(dim()),
      .gradient = grad_,
      .log_density = log_density_,
    };
    const double H0 = hamiltonian_(trial);
    const Eigen::VectorXd scaled_stepsize =
      epsilon * sqrt_variance_;
    const bool valid = leapfrog_(trial, scaled_stepsize);
    double h = valid ? hamiltonian_(trial) :
      std::numeric_limits<double>::infinity();
    if (std::isnan(h)) {
      h = std::numeric_limits<double>::infinity();
    }
    const double delta_H = H0 - h;
    return std::isnan(delta_H) ?
      -std::numeric_limits<double>::infinity() : delta_H;
  }

  void adapt_warmup_() {
    if (draw_ > opts_.warmup) {
      return;
    }

    update_stepsize_(accept_stat_);

    const bool in_metric =
      draw_ > metric_start_ && draw_ <= metric_end_;
    if (in_metric) {
      online_moments_.update(theta_);
      const std::size_t metric_draw = draw_ - metric_start_;
      if (metric_windowed_adaptation_.window_closed(metric_draw) ||
          draw_ == metric_end_) {
        update_variance_();
        online_moments_.reset();
        find_good_initial_stepsize(stepsize_);
      }
    }

    if (draw_ == opts_.warmup) {
      complete_stepsize_adaptation_();
    }
  }

  void update_variance_() {
    // WelfordAccumulator::variance() includes Stan's n / (n + 5)
    // shrinkage and its 1e-3 regularization term.
    Eigen::VectorXd updated = online_moments_.variance();
    for (Eigen::Index d = 0; d < updated.size(); ++d) {
      if (!std::isfinite(updated(d))) {
        throw std::runtime_error(
          "Stan: numerical overflow in metric adaptation");
      }
      updated(d) = std::clamp(
        updated(d), opts_.variance_floor, opts_.variance_ceiling);
    }
    variance_ = std::move(updated);
    sqrt_variance_ = variance_.cwiseSqrt();
  }

  void restart_stepsize_adaptation_() {
    dual_counter_ = 0;
    dual_s_bar_ = 0.0;
    dual_x_bar_ = 0.0;
    dual_mu_ = std::log(10.0 * stepsize_);
  }

  void update_stepsize_(double adapt_stat) {
    if (!std::isfinite(adapt_stat)) {
      adapt_stat = 0.0;
    }
    adapt_stat = std::min(1.0, adapt_stat);

    ++dual_counter_;
    const double counter = static_cast<double>(dual_counter_);
    const double eta = 1.0 / (counter + opts_.dual_t0);
    dual_s_bar_ = (1.0 - eta) * dual_s_bar_ +
      eta * (opts_.target_accept - adapt_stat);

    const double x =
      dual_mu_ - dual_s_bar_ * std::sqrt(counter) / opts_.dual_gamma;
    const double x_eta = std::pow(counter, -opts_.dual_kappa);
    dual_x_bar_ = (1.0 - x_eta) * dual_x_bar_ + x_eta * x;
    set_stepsize_(std::exp(x));
  }

  void complete_stepsize_adaptation_() {
    if (dual_counter_ > 0) {
      set_stepsize_(std::exp(dual_x_bar_));
    }
  }

  void set_stepsize_(const double epsilon) {
    stepsize_ = clamp_stepsize_(epsilon);
  }

  double clamp_stepsize_(double epsilon) const {
    if (std::isnan(epsilon)) {
      epsilon = opts_.initial_stepsize;
    } else if (!(epsilon > 0.0)) {
      epsilon = opts_.min_stepsize;
    } else if (!std::isfinite(epsilon)) {
      epsilon = opts_.max_stepsize;
    }
    return std::clamp(
      epsilon, opts_.min_stepsize, opts_.max_stepsize);
  }

  Eigen::VectorXd normal_rng_(const Eigen::Index D) {
    Eigen::VectorXd out(D);
    std::generate(
      out.data(), out.data() + D, [&](){ return std_normal_(rng_); });
    return out;
  }
};

} // namespace klhr
