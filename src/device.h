#ifndef INFINI_OPS_DEVICE_H_
#define INFINI_OPS_DEVICE_H_

namespace infini::ops {

class Device {
 public:
  // TODO: Complete the list.
  enum class Type {
    kCpu = 0,
    kNvidia = 1,
    kCambricon = 2,
    kAscend = 3,
    kMetax = 4,
    kMoore = 5,
    kIluvatar = 6,
    kKunlun = 7,
    kHygon = 8,
    kQy = 9,
    kCount
  };

  Device() = default;

  Device(const Type& type, const int& index = 0) : type_{type}, index_{index} {}

  static const Type& TypeFromString(const std::string& name) {
    // TODO: Handle `"cuda"` dispatching.
    static std::unordered_map<std::string, Type> name_to_type{
        {"cpu", Type::kCpu}, {"cuda", Type::kNvidia}};

    return name_to_type.at(name);
  }

  const Type& type() const { return type_; }

  const int& index() const { return index_; }

 private:
  Type type_{Type::kCpu};

  int index_{0};
};

}  // namespace infini::ops

#endif
