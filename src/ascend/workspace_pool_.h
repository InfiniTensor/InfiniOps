#ifndef INFINI_OPS_ASCEND_WORKSPACE_POOL__H_
#define INFINI_OPS_ASCEND_WORKSPACE_POOL__H_

#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <memory>
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
  WorkspaceArena& ensure(aclrtStream stream, uint64_t needed) {
    // Thread-local fast path: skip mutex when the same stream's arena already
    // has enough capacity.  After warmup (first call per operator), workspace
    // sizes are fixed and this path is always taken.
    //
    // NOTE: Only the most recent stream is cached.  If a single thread
    // alternates between multiple streams (e.g. TP>1 driven by one thread),
    // every stream switch falls back to the slow path.  Replace with a
    // small thread-local map if multi-stream-per-thread becomes common.
    thread_local aclrtStream last_stream = nullptr;
    thread_local WorkspaceArena* last_arena = nullptr;

    if (stream == last_stream && last_arena != nullptr &&
        needed <= last_arena->capacity) {
      return *last_arena;
    }

    // Slow path: look up arena in the map under lock.
    // Arenas are heap-allocated via `unique_ptr` so that pointers remain stable
    // across `unordered_map` rehashes (which invalidate value references).
    std::lock_guard<std::mutex> lock(mutex_);
    auto& slot = arenas_[stream];
    if (!slot) {
      slot = std::make_unique<WorkspaceArena>();
    }
    auto* arena = slot.get();
    if (needed > arena->capacity) {
      if (arena->capacity > 0) {
        aclrtSynchronizeStream(stream);
        aclrtFree(arena->buf);
      }
      if (needed > 0) {
        auto ret =
            aclrtMalloc(&arena->buf, needed, ACL_MEM_MALLOC_NORMAL_ONLY);
        assert(ret == ACL_SUCCESS && "`WorkspacePool`: `aclrtMalloc` failed");
      }
      arena->capacity = needed;
    }
    last_stream = stream;
    last_arena = arena;
    return *arena;
  }

  ~WorkspacePool() {
    for (auto& [stream, arena] : arenas_) {
      if (arena && arena->capacity > 0) {
        // The CANN runtime may already be torn down when this static
        // destructor runs.  aclrtGetDevice fails in that case — skip the
        // free to avoid glibc "double free" abort.
        int32_t dev_id = -1;
        if (aclrtGetDevice(&dev_id) == ACL_SUCCESS) {
          aclrtFree(arena->buf);
        } else {
          fprintf(stderr,
                  "[InfiniOps] `WorkspacePool`: CANN runtime already finalized, "
                  "skipping `aclrtFree` (%" PRIu64 " bytes leaked).\n",
                  arena->capacity);
        }
      }
    }
  }

 private:
  std::unordered_map<aclrtStream, std::unique_ptr<WorkspaceArena>> arenas_;

  std::mutex mutex_;
};

inline WorkspacePool& workspacePool() {
  static WorkspacePool pool;
  return pool;
}

}  // namespace infini::ops::ascend

#endif
