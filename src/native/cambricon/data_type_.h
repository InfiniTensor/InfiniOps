#ifndef INFINI_OPS_CAMBRICON_DATA_TYPE__H_
#define INFINI_OPS_CAMBRICON_DATA_TYPE__H_

#include <infini/rt/cambricon/data_type_.h>
#include "common/traits.h"
#include "data_type.h"

namespace infini::ops {
template <DataType... Extra>
using CambriconBangFloatTypes =
    List<DataType::kFloat16,
#if !defined(INFINIOPS_CAMBRICON_BANG_BF16) || INFINIOPS_CAMBRICON_BANG_BF16
         DataType::kBFloat16,
#endif
         DataType::kFloat32, Extra...>;
}

// Native arithmetic is confined to genuine BANG compilation. The public RT
// type map uses the same complete storage types in host and native objects.
#if defined(__BANGCC__) || defined(INFINIOPS_CAMBRICON_NATIVE)
#include <bang_bf16.h>
#if __has_include(<bang_fp16.h>)
#include <bang_fp16.h>
#endif

namespace infini::ops::cambricon {
#if __has_include(<bang_fp16.h>)
using NativeHalf = __half;
#else
using NativeHalf = half;
#endif
template <typename StorageT>
struct NativeScalar {
  using type = StorageT;
};
template <>
struct NativeScalar<infini::rt::Float16> {
  using type = NativeHalf;
};
template <>
struct NativeScalar<infini::rt::BFloat16> {
  using type = __bang_bfloat16;
};
template <typename StorageT>
using NativeScalarType = typename NativeScalar<StorageT>::type;
}  // namespace infini::ops::cambricon
#endif

#endif
