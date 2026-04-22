#ifndef INFINI_OPS_SMALL_VECTOR_H_
#define INFINI_OPS_SMALL_VECTOR_H_

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace infini::ops {

// Minimal, std::vector-compatible small-buffer-optimized vector for the
// handful of tiny arrays we copy on every op call (primarily `Tensor`'s
// shape and strides — up to 4-5 dims in practice). Keeps storage inline up
// to `N` elements; spills to the heap otherwise. Kept intentionally narrow
// — only the ops we use in InfiniOps (size/indexing/iteration/push_back)
// are implemented; no custom allocators or exception safety beyond what
// `std::terminate` provides.
template <typename T, std::size_t N>
class SmallVector {
  static_assert(std::is_trivially_copyable_v<T>,
                "SmallVector requires a trivially-copyable element type");

 public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = T*;
  using const_iterator = const T*;

  SmallVector() noexcept : data_{inline_storage()}, size_{0}, capacity_{N} {}

  explicit SmallVector(size_type count)
      : data_{inline_storage()}, size_{0}, capacity_{N} {
    resize(count);
  }

  SmallVector(size_type count, const T& value)
      : data_{inline_storage()}, size_{0}, capacity_{N} {
    resize(count, value);
  }

  SmallVector(std::initializer_list<T> init)
      : data_{inline_storage()}, size_{0}, capacity_{N} {
    reserve(init.size());
    for (const auto& v : init) {
      data_[size_++] = v;
    }
  }

  template <typename Iter,
            typename = std::enable_if_t<std::is_convertible_v<
                typename std::iterator_traits<Iter>::iterator_category,
                std::input_iterator_tag>>>
  SmallVector(Iter begin_it, Iter end_it)
      : data_{inline_storage()}, size_{0}, capacity_{N} {
    reserve(static_cast<size_type>(std::distance(begin_it, end_it)));
    for (auto it = begin_it; it != end_it; ++it) {
      data_[size_++] = static_cast<T>(*it);
    }
  }

  SmallVector(const SmallVector& other)
      : data_{inline_storage()}, size_{0}, capacity_{N} {
    assign_range(other.begin(), other.end());
  }

  SmallVector(SmallVector&& other) noexcept
      : data_{inline_storage()}, size_{0}, capacity_{N} {
    move_from(std::move(other));
  }

  // Interop with arbitrary vector-like containers (e.g. `std::vector`): we
  // accept any type whose elements convert to `T`. The `!is_same_v` guard
  // prevents this from shadowing the copy/move constructors above.
  template <typename Container,
            typename = std::enable_if_t<
                !std::is_same_v<std::decay_t<Container>, SmallVector> &&
                std::is_convertible_v<decltype(*std::declval<const Container&>().begin()),
                                      T>>>
  SmallVector(const Container& other)
      : data_{inline_storage()}, size_{0}, capacity_{N} {
    assign_range(other.begin(), other.end());
  }

  SmallVector& operator=(const SmallVector& other) {
    if (this == &other) return *this;
    clear_and_release();
    assign_range(other.begin(), other.end());
    return *this;
  }

  SmallVector& operator=(SmallVector&& other) noexcept {
    if (this == &other) return *this;
    clear_and_release();
    move_from(std::move(other));
    return *this;
  }

  template <typename Container>
  std::enable_if_t<std::is_convertible_v<
                       decltype(*std::declval<const Container&>().begin()), T>,
                   SmallVector&>
  operator=(const Container& other) {
    clear_and_release();
    assign_range(other.begin(), other.end());
    return *this;
  }

  ~SmallVector() {
    if (on_heap()) {
      ::operator delete(data_);
    }
  }

  size_type size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  pointer data() noexcept { return data_; }
  const_pointer data() const noexcept { return data_; }

  reference operator[](size_type i) noexcept { return data_[i]; }
  const_reference operator[](size_type i) const noexcept { return data_[i]; }
  reference front() noexcept { return data_[0]; }
  const_reference front() const noexcept { return data_[0]; }
  reference back() noexcept { return data_[size_ - 1]; }
  const_reference back() const noexcept { return data_[size_ - 1]; }

  iterator begin() noexcept { return data_; }
  const_iterator begin() const noexcept { return data_; }
  const_iterator cbegin() const noexcept { return data_; }
  iterator end() noexcept { return data_ + size_; }
  const_iterator end() const noexcept { return data_ + size_; }
  const_iterator cend() const noexcept { return data_ + size_; }

  void clear() noexcept { size_ = 0; }

  void push_back(const T& value) {
    if (size_ == capacity_) grow();
    data_[size_++] = value;
  }

  void push_back(T&& value) {
    if (size_ == capacity_) grow();
    data_[size_++] = std::move(value);
  }

  template <typename... Args>
  reference emplace_back(Args&&... args) {
    if (size_ == capacity_) grow();
    data_[size_++] = T(std::forward<Args>(args)...);
    return data_[size_ - 1];
  }

  void resize(size_type new_size, T fill = T{}) {
    reserve(new_size);
    for (size_type i = size_; i < new_size; ++i) data_[i] = fill;
    size_ = new_size;
  }

  void reserve(size_type new_capacity) {
    if (new_capacity <= capacity_) return;
    reallocate(new_capacity);
  }

  friend bool operator==(const SmallVector& lhs, const SmallVector& rhs) {
    if (lhs.size_ != rhs.size_) return false;
    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
  }

  friend bool operator!=(const SmallVector& lhs, const SmallVector& rhs) {
    return !(lhs == rhs);
  }

 private:
  T* inline_storage() noexcept {
    return reinterpret_cast<T*>(&inline_storage_buffer_);
  }

  const T* inline_storage() const noexcept {
    return reinterpret_cast<const T*>(&inline_storage_buffer_);
  }

  bool on_heap() const noexcept { return data_ != inline_storage(); }

  void grow() {
    reallocate(capacity_ ? capacity_ * 2 : 4);
  }

  void reallocate(size_type new_capacity) {
    auto* new_data = static_cast<T*>(::operator new(new_capacity * sizeof(T)));
    std::copy(data_, data_ + size_, new_data);
    if (on_heap()) ::operator delete(data_);
    data_ = new_data;
    capacity_ = new_capacity;
  }

  void clear_and_release() noexcept {
    if (on_heap()) ::operator delete(data_);
    data_ = inline_storage();
    size_ = 0;
    capacity_ = N;
  }

  template <typename Iter>
  void assign_range(Iter first, Iter last) {
    reserve(static_cast<size_type>(std::distance(first, last)));
    for (; first != last; ++first) {
      data_[size_++] = static_cast<T>(*first);
    }
  }

  void move_from(SmallVector&& other) noexcept {
    if (other.on_heap()) {
      // Steal the heap buffer — fast, zero copy.
      data_ = other.data_;
      capacity_ = other.capacity_;
      size_ = other.size_;
      other.data_ = other.inline_storage();
      other.size_ = 0;
      other.capacity_ = N;
    } else {
      // Other is inline — must copy into our inline buffer.
      size_ = other.size_;
      std::copy(other.data_, other.data_ + size_, data_);
      other.size_ = 0;
    }
  }

  T* data_;
  size_type size_;
  size_type capacity_;
  alignas(T) std::byte inline_storage_buffer_[N * sizeof(T)];
};

}  // namespace infini::ops

#endif
