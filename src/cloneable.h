#ifndef INFINI_OPS_CLONEABLE_H_
#define INFINI_OPS_CLONEABLE_H_

#include <memory>
#include <type_traits>

namespace infini::ops {

template <typename Base, typename Derived>
class Cloneable : public Base {
 public:
  std::unique_ptr<Base> Clone() const override {
    static_assert(std::is_final_v<Derived>,
                  "Cloneable requires a final derived class.");
    return std::make_unique<Derived>(static_cast<const Derived&>(*this));
  }
};

}  // namespace infini::ops

#endif
