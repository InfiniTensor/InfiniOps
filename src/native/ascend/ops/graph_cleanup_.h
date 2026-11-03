#ifndef INFINI_OPS_ASCEND_GRAPH_CLEANUP__H_
#define INFINI_OPS_ASCEND_GRAPH_CLEANUP__H_

#include <functional>
#include <utility>
#include <vector>

namespace infini::ops::ascend {

class DeferredAclCleanupScope;

namespace detail {

inline thread_local DeferredAclCleanupScope* active_acl_cleanup_scope = nullptr;

}  // namespace detail

class DeferredAclCleanupScope {
 public:
  DeferredAclCleanupScope() : previous_(detail::active_acl_cleanup_scope) {
    detail::active_acl_cleanup_scope = this;
  }

  ~DeferredAclCleanupScope() {
    detail::active_acl_cleanup_scope = previous_;

    for (auto& cleanup : callbacks_) {
      cleanup();
    }
  }

  DeferredAclCleanupScope(const DeferredAclCleanupScope&) = delete;

  DeferredAclCleanupScope& operator=(const DeferredAclCleanupScope&) = delete;

  void Defer(std::function<void()> cleanup) {
    callbacks_.push_back(std::move(cleanup));
  }

  std::vector<std::function<void()>> Release() { return std::move(callbacks_); }

 private:
  DeferredAclCleanupScope* previous_;

  std::vector<std::function<void()>> callbacks_;
};

inline void DeferOrRunAclCleanup(std::function<void()> cleanup) {
  if (detail::active_acl_cleanup_scope) {
    detail::active_acl_cleanup_scope->Defer(std::move(cleanup));

    return;
  }

  cleanup();
}

}  // namespace infini::ops::ascend

#endif
