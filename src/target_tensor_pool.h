#ifndef INFINI_OPS_TARGET_TENSOR_POOL_H_
#define INFINI_OPS_TARGET_TENSOR_POOL_H_

#include <cassert>
#include <memory>
#include <utility>
#include <vector>

namespace infini::ops {

template <typename Adapter>
class TargetTensorPool {
  struct Entry {
    explicit Entry(std::unique_ptr<typename Adapter::State> state)
        : state{std::move(state)}, busy{true} {}

    std::unique_ptr<typename Adapter::State> state;

    bool busy;
  };

 public:
  class Lease {
   public:
    Lease() = default;

    Lease(const Lease&) = delete;

    Lease& operator=(const Lease&) = delete;

    Lease(Lease&& other) noexcept
        : entry_{std::exchange(other.entry_, nullptr)} {}

    Lease& operator=(Lease&& other) noexcept {
      if (this != &other) {
        Release();
        entry_ = std::exchange(other.entry_, nullptr);
      }

      return *this;
    }

    ~Lease() noexcept { Release(); }

    decltype(auto) Native() const { return Adapter::Native(*entry_->state); }

   private:
    explicit Lease(Entry* entry) : entry_{entry} {}

    void Release() noexcept {
      if (entry_ != nullptr) {
        entry_->busy = false;
        entry_ = nullptr;
      }
    }

    Entry* entry_ = nullptr;

    friend class TargetTensorPool;
  };

  TargetTensorPool() = default;

  TargetTensorPool(const TargetTensorPool&) = delete;

  TargetTensorPool& operator=(const TargetTensorPool&) = delete;

  TargetTensorPool(TargetTensorPool&&) noexcept = default;

  TargetTensorPool& operator=(TargetTensorPool&&) = delete;

  ~TargetTensorPool() {
#ifndef NDEBUG
    for (const auto& entry : entries_) {
      assert(!entry->busy &&
             "`TargetTensorPool` destroyed with an active lease.");
    }
#endif
  }

  template <typename TensorLike>
  Lease Acquire(const TensorLike& tensor) {
    static_assert(
        noexcept(Adapter::Rebind(std::declval<typename Adapter::State&>(),
                                 std::declval<const TensorLike&>())),
        "`Adapter::Rebind` must be `noexcept`.");

    for (auto& entry : entries_) {
      auto* entry_pointer = entry.get();

      if (entry_pointer->busy) {
        continue;
      }

      entry_pointer->busy = true;
      Adapter::Rebind(*entry_pointer->state, tensor);

      return Lease{entry_pointer};
    }

    auto state = Adapter::Create(tensor);
    auto entry = std::make_unique<Entry>(std::move(state));
    auto* entry_pointer = entry.get();
    entries_.push_back(std::move(entry));

    return Lease{entry_pointer};
  }

 private:
  std::vector<std::unique_ptr<Entry>> entries_;
};

}  // namespace infini::ops

#endif
