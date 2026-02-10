#ifndef INFINI_OPS_DEVICE_H_
#define INFINI_OPS_DEVICE_H_

namespace infini::ops {

class Device {
 public:
  // TODO: Complete the list.
  enum class Type { kCpu, kNvidia, kCount };

  Device() = default;

  Device(const Type& type, const int& index = 0) : type_{type}, index_{index} {}

  const Type& type() const { return type_; }

  const int& index() const { return index_; }

 private:
  Type type_{Type::kCpu};

  int index_{0};
};

}  // namespace infini::ops

#endif
