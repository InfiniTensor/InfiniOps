#ifndef INFINI_OPS_CONFIG_H_
#define INFINI_OPS_CONFIG_H_

#include <cstddef>
#include <memory>

namespace infini::ops {

class Config {
 public:
  std::size_t implementation_index() const { return implementation_index_; }

  void set_implementation_index(std::size_t implementation_index) {
    implementation_index_ = implementation_index;
  }

  void set_extension(std::shared_ptr<Config> extension) {
    extension_ = std::move(extension);
  }

  std::shared_ptr<Config> extension() const { return extension_; }

 private:
  std::size_t implementation_index_{0};
  std::shared_ptr<Config> extension_{};
};

}  // namespace infini::ops

#endif
