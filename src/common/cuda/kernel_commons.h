#ifndef INFINI_OPS_COMMON_CUDA_KERNEL_COMMONS_H_
#define INFINI_OPS_COMMON_CUDA_KERNEL_COMMONS_H_

#ifdef WITH_NVIDIA
#include <cuda_runtime.h>
#elif defined(WITH_ILUVATAR)
#include <cuda_runtime.h>
#elif WITH_METAX  // TODO: Use `defined`.
#include <mcr/mc_runtime.h>
#endif

#include "data_type.h"

namespace infini::ops {

namespace detail {

template <typename T>
__host__ __device__ constexpr float ToFloatHelper(T&& x) {
  using PureT = std::remove_cv_t<std::remove_reference_t<T>>;
  if constexpr (std::is_same_v<PureT, TypeMapType<DataType::kBFloat16>>) {
    return __bfloat162float(x);
  } else if constexpr (std::is_same_v<PureT, half>) {
    return __half2float(x);
  } else {
    return static_cast<float>(std::forward<T>(x));
  }
}

template <typename Dst>
__host__ __device__ constexpr Dst FromFloatHelper(float f) {
  using PureDst = std::remove_cv_t<std::remove_reference_t<Dst>>;
  if constexpr (std::is_same_v<PureDst, TypeMapType<DataType::kBFloat16>>) {
    return __float2bfloat16(f);
  } else if constexpr (std::is_same_v<PureDst, half>) {
    return __float2half(f);
  } else {
    return static_cast<Dst>(f);
  }
}

}  // namespace detail

template <typename Dst, typename Src>
__host__ __device__ Dst Cast(Src&& x) {
  static_assert(!std::is_reference_v<Dst>,
                "`Cast` cannot return reference types");

  using PureSrc = std::remove_cv_t<std::remove_reference_t<Src>>;
  using PureDst = std::remove_cv_t<std::remove_reference_t<Dst>>;

  if constexpr (std::is_same_v<PureSrc, PureDst>) {
    return std::forward<Src>(x);
  }

  // Direct intrinsics
  if constexpr (std::is_same_v<PureSrc, TypeMapType<DataType::kBFloat16>>) {
    if constexpr (std::is_same_v<PureDst, int>) {
      return __bfloat162int_rn(x);
    } else if constexpr (std::is_same_v<PureDst, short>) {
      return __bfloat162short_rn(x);
    }
  } else if constexpr (std::is_same_v<PureSrc, half>) {
    if constexpr (std::is_same_v<PureDst, int>) {
      return __half2int_rn(x);
    } else if constexpr (std::is_same_v<PureDst, short>) {
      return __half2short_rn(x);
    }
  } else if constexpr (std::is_same_v<PureDst,
                                      TypeMapType<DataType::kBFloat16>>) {
    if constexpr (std::is_same_v<PureSrc, int>) {
      return __int2bfloat16_rn(x);
    } else if constexpr (std::is_same_v<PureSrc, double>) {
      return __double2bfloat16(x);
    }
  } else if constexpr (std::is_same_v<PureDst, half>) {
    if constexpr (std::is_same_v<PureSrc, int>) {
      return __int2half_rn(x);
    } else if constexpr (std::is_same_v<PureSrc, double>) {
      return __double2half(x);
    }
  }

  return detail::FromFloatHelper<PureDst>(
      detail::ToFloatHelper(std::forward<Src>(x)));
}

__forceinline__ __device__ __host__ size_t
IndexToOffset(size_t flat_index, size_t ndim, const size_t* shape,
              const ptrdiff_t* strides) {
  size_t res = 0;
  for (size_t i = ndim; i-- > 0;) {
    res += (flat_index % shape[i]) * strides[i];
    flat_index /= shape[i];
  }
  return res;
}

}  // namespace infini::ops

#endif
