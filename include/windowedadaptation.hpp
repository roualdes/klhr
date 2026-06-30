#pragma once

#include <cstddef>
#include <vector>

namespace klhr {

class WindowedAdaptation {
public:
  WindowedAdaptation(std::size_t warmup = 1'000,
                     std::size_t windowsize = 50,
                     std::size_t windowscale = 2) :
    windowsize_(windowsize),
    windowscale_(windowscale),
    warmup_(warmup),
    closewindow_(windowsize),
    idx_(0) {
    calculate_windows_();
  }

  bool window_closed(std::size_t draw) {
    if (warmup_ < windowsize_) {
      return false;
    }

    if (closures_.empty() || idx_ >= closures_.size()) {
      return false;
    }

    const bool closed = draw == closures_[idx_];
    if (closed && idx_ + 1 < closures_.size()) {
      ++idx_;
    }
    return closed;
  }

  void reset() {
    closewindow_ = windowsize_;
    idx_ = 0;
    closures_.clear();
    calculate_windows_();
  }

  const std::vector<std::size_t>& closures() const {
    return closures_;
  }

private:
  std::size_t windowsize_;
  std::size_t windowscale_;
  std::size_t warmup_;
  std::size_t closewindow_;
  std::size_t idx_;
  std::vector<std::size_t> closures_;

  void calculate_windows_() {
    const std::size_t original_windowsize = windowsize_;
    if (warmup_ > windowsize_) {
      for (std::size_t w = 0; w <= warmup_; ++w) {
        if (w == closewindow_) {
          closures_.push_back(w);
          calculate_next_window_();
        }
      }
    }
    windowsize_ = original_windowsize;
  }

  void calculate_next_window_() {
    windowsize_ *= windowscale_;
    const std::size_t nextclosewindow = closewindow_ + windowsize_;
    if (closewindow_ + windowscale_ * windowsize_ >= warmup_) {
      closewindow_ = warmup_;
    } else {
      closewindow_ = nextclosewindow;
    }
  }
};

} // namespace klhr
