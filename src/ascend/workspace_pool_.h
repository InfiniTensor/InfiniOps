#ifndef INFINI_OPS_ASCEND_WORKSPACE_POOL__H_
#define INFINI_OPS_ASCEND_WORKSPACE_POOL__H_

#include <cassert>
#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "acl/acl.h"

namespace infini::ops::ascend {

struct WorkspaceArena {
  void* buf = nullptr;

  uint64_t capacity = 0;
};

class WorkspacePool {
 public:
  WorkspaceArena& Ensure(aclrtStream stream, uint64_t needed) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& arena = arenas_[stream];
    if (needed <= arena.capacity) return arena;
    if (arena.capacity > 0) {
      aclrtSynchronizeStream(stream);
      aclrtFree(arena.buf);
    }
    if (needed > 0) {
      auto ret = aclrtMalloc(&arena.buf, needed, ACL_MEM_MALLOC_NORMAL_ONLY);
      assert(ret == ACL_SUCCESS && "`WorkspacePool`: `aclrtMalloc` failed");
    }
    arena.capacity = needed;
    return arena;
  }

  ~WorkspacePool() {
    for (auto& [stream, arena] : arenas_) {
      if (arena.capacity > 0) aclrtFree(arena.buf);
    }
  }

 private:
  std::unordered_map<aclrtStream, WorkspaceArena> arenas_;

  std::mutex mutex_;
};

inline WorkspacePool& workspacePool() {
  static WorkspacePool pool;
  return pool;
}

}  // namespace infini::ops::ascend

#endif
