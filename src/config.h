#ifndef INFINI_OPS_CONFIG_H_
#define INFINI_OPS_CONFIG_H_

#include <cstddef>
#include <memory>

#include "cloneable.h"

namespace infini::ops {

class Config {
 public:
  virtual ~Config() = default;

  virtual std::unique_ptr<Config> Clone() const {
    return std::make_unique<Config>(*this);
  }

  std::size_t implementation_index() const { return implementation_index_; }

  void set_implementation_index(std::size_t implementation_index) {
    implementation_index_ = implementation_index;
    // 用户显式指定实现索引时，关闭自动选择
    auto_select_ = false;
  }

  // 是否启用自动选择：默认为 true，当用户显式指定实现索引时设为 false
  bool auto_select() const { return auto_select_; }

 private:
  std::size_t implementation_index_{0};
  bool auto_select_{true};  // 默认启用自动选择
};

}  // namespace infini::ops

#endif
