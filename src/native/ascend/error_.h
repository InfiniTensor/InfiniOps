#ifndef INFINI_OPS_ASCEND_ERROR__H_
#define INFINI_OPS_ASCEND_ERROR__H_

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

namespace infini::ops::ascend {

inline bool IsAscend310P() {
  const auto* soc = aclrtGetSocName();
  return soc != nullptr && std::strncmp(soc, "Ascend310P", 10) == 0;
}

inline void CheckAcl(int64_t status, const char* operation) {
#ifndef NDEBUG
  if (status != ACL_SUCCESS) {
    std::fprintf(stderr, "%s:%d: %s: `%s` failed with ACL status %lld.\n",
                 __FILE__, __LINE__, __func__, operation,
                 static_cast<long long>(status));
  }
  assert(status == ACL_SUCCESS && "`ACL` operation failed.");
#endif
  (void)status;
  (void)operation;
}

using AclTensorOwner = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;

inline AclTensorOwner OwnAclTensor(aclTensor* tensor) {
  assert(tensor != nullptr && "`aclCreateTensor` failed.");
  return AclTensorOwner(tensor, &aclDestroyTensor);
}

}  // namespace infini::ops::ascend

#endif
