#ifndef INFINI_OPS_CONFIG_H_
#define INFINI_OPS_CONFIG_H_

#include <cstddef>

namespace infini::ops {

class Config {
 public:
  std::size_t implementation_index() const { return implementation_index_; }

  void set_implementation_index(std::size_t implementation_index) {
    implementation_index_ = implementation_index;
    auto_select_ = false;
  }

  bool auto_select() const { return auto_select_; }

 private:
  std::size_t implementation_index_{0};
  bool auto_select_{true};
};

}  // namespace infini::ops

#endif
