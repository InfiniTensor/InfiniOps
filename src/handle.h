#ifndef INFINI_OPS_HANDLE_H_
#define INFINI_OPS_HANDLE_H_

#include <cstddef>

namespace infini::ops {

class Handle {
 public:
  void* stream() const { return stream_; }

  void* workspace() const { return workspace_; }

  std::size_t workspace_size_in_bytes() const {
    return workspace_size_in_bytes_;
  }

  void set_stream(void* stream) { stream_ = stream; }

  void set_workspace(void* workspace) { workspace_ = workspace; }

  void set_workspace_size_in_bytes(std::size_t workspace_size_in_bytes) {
    workspace_size_in_bytes_ = workspace_size_in_bytes;
  }

 private:
  void* stream_{nullptr};

  void* workspace_{nullptr};

  std::size_t workspace_size_in_bytes_{0};
};

}  // namespace infini::ops

#endif
