#ifndef INFINI_OPS_ASCEND_DEVICE_H_
#define INFINI_OPS_ASCEND_DEVICE_H_

#include <cassert>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "data_type.h"
#include "device.h"
#include "tensor.h"

namespace infini::ops::ascend {

inline constexpr auto kDeviceType{Device::Type::kAscend};

struct AscendBackend {
    using stream_t = aclrtStream;
    static constexpr auto malloc     = aclrtMalloc;
    static constexpr auto memcpy     = aclrtMemcpy;
    static constexpr auto free       = aclrtFree;
    static constexpr auto memcpyH2D  = ACL_MEMCPY_HOST_TO_DEVICE;  // Direction flag for aclrtMemcpy.
};

inline aclDataType toAclDtype(DataType dt) {
    switch (dt) {
        case DataType::kFloat16:  return ACL_FLOAT16;
        case DataType::kBFloat16: return ACL_BF16;
        case DataType::kFloat32:  return ACL_FLOAT;
        case DataType::kInt8:     return ACL_INT8;
        case DataType::kInt16:    return ACL_INT16;
        case DataType::kInt32:    return ACL_INT32;
        case DataType::kInt64:    return ACL_INT64;
        case DataType::kUInt8:    return ACL_UINT8;
        case DataType::kUInt16:   return ACL_UINT16;
        case DataType::kUInt32:   return ACL_UINT32;
        case DataType::kUInt64:   return ACL_UINT64;
        default:
            assert(false && "unsupported dtype for Ascend backend");
            return ACL_DT_UNDEFINED;
    }
}

// Returns true for integer (signed or unsigned) DataType values.
inline bool isIntegerDtype(DataType dt) {
    switch (dt) {
        case DataType::kInt8:
        case DataType::kInt16:
        case DataType::kInt32:
        case DataType::kInt64:
        case DataType::kUInt8:
        case DataType::kUInt16:
        case DataType::kUInt32:
        case DataType::kUInt64:
            return true;
        default:
            return false;
    }
}

struct WorkspaceArena {
    void* buf = nullptr;
    uint64_t capacity = 0;
};

class WorkspacePool {
 public:
    WorkspaceArena& ensure(aclrtStream stream, uint64_t needed) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& arena = arenas_[stream];
        if (needed <= arena.capacity) return arena;
        if (arena.capacity > 0) {
            aclrtSynchronizeStream(stream);
            aclrtFree(arena.buf);
        }
        if (needed > 0) {
            aclrtMalloc(&arena.buf, needed, ACL_MEM_MALLOC_NORMAL_ONLY);
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

inline aclTensor* buildAclTensor(const Tensor& t) {
    std::vector<int64_t> shape(t.shape().begin(), t.shape().end());
    std::vector<int64_t> strides(t.strides().begin(), t.strides().end());

    // Compute the minimum physical storage needed for this strided view.
    // For contiguous tensors this equals numel(); for non-contiguous (gapped)
    // tensors it may be larger; for broadcast (stride-0) tensors it may be
    // smaller.  Passing the view shape as the storage shape causes
    // "ViewShape overlap" errors in ACLNN for non-contiguous inputs.
    int64_t storage_elems = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 0) { storage_elems = 0; break; }
        if (strides[i] > 0 && shape[i] > 1) {
            storage_elems += static_cast<int64_t>(shape[i] - 1) * strides[i];
        }
    }
    std::vector<int64_t> storage_shape = {storage_elems};

    return aclCreateTensor(
        shape.data(),
        static_cast<int64_t>(shape.size()),
        toAclDtype(t.dtype()),
        strides.data(),
        /*storageOffset=*/0,
        ACL_FORMAT_ND,
        storage_shape.data(),
        static_cast<int64_t>(storage_shape.size()),
        const_cast<void*>(t.data()));
}

}  // namespace infini::ops::ascend

#endif
