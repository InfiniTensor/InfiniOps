#ifndef INFINI_OPS_ASCEND_ERROR__H_
#define INFINI_OPS_ASCEND_ERROR__H_

#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

namespace infini::ops::ascend {

inline bool IsAscend310P() {
  const auto* soc = aclrtGetSocName();
  return soc != nullptr && std::strncmp(soc, "Ascend310P", 10) == 0;
}

inline void CheckAcl(int64_t status, const char* operation) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(std::string(operation) +
                             " failed with ACL status " +
                             std::to_string(status));
  }
}

using AclTensorOwner = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;

inline AclTensorOwner OwnAclTensor(aclTensor* tensor) {
  assert(tensor != nullptr && "aclCreateTensor failed");
  return AclTensorOwner(tensor, &aclDestroyTensor);
}

}  // namespace infini::ops::ascend

#endif
