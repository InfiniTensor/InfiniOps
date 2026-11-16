/***************************************************************************************************
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Helper macros for the CUTLASS library
*/

#pragma once
#include <hip/amd_detail/host_defines.h>
////////////////////////////////////////////////////////////////////////////////////////////////////


#ifdef CUTLASS_NAMESPACE
#define concat_tok(a, b) a ## b
#define mkcutlassnamespace(pre, ns) concat_tok(pre, ns)
#define cutlass mkcutlassnamespace(cutlass_, CUTLASS_NAMESPACE)
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
#if defined DCU_ASM
#define CUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#define CUTLASS_DEVICE __forceinline__ __device__
#elif defined(__CUDACC_RTC__)
#define CUTLASS_HOST_DEVICE __forceinline__ __device__
#define CUTLASS_DEVICE __forceinline__ __device__
#else
#define CUTLASS_HOST_DEVICE inline
#define CUTLASS_DEVICE inline
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
CUTLASS_HOST_DEVICE void __CUTLASS_UNUSED(T const &) 
{ }

#if defined(__GNUC__)
  #define CUTLASS_UNUSED(expr) __CUTLASS_UNUSED(expr)
#else
  #define CUTLASS_UNUSED(expr) do { ; } while (&expr != &expr)
#endif

#ifdef _MSC_VER
// Provides support for alternative operators 'and', 'or', and 'not'
#include <iso646.h>
#endif // _MSC_VER

#if !defined(__CUDACC_RTC__)
#include <assert.h>
#endif

#ifdef DCU_ASM
  #define CUTLASS_NOT_IMPLEMENTED() { printf("not implemented\n"); assert(0); }
#elif defined(__HIP_DEVICE_COMPILE__)
  #if defined(_MSC_VER)
    #define CUTLASS_NOT_IMPLEMENTED() { printf("%s not implemented\n", __FUNCSIG__); asm volatile ("brkpt;\n"); }
  #else
    #define CUTLASS_NOT_IMPLEMENTED() { printf("%s not implemented\n", __PRETTY_FUNCTION__); asm volatile ("brkpt;\n"); }
  #endif
#else
  #if defined(_MSC_VER)
    #define CUTLASS_NOT_IMPLEMENTED() assert(0 && __FUNCSIG__)
  #else
    #define CUTLASS_NOT_IMPLEMENTED() assert(0 && __PRETTY_FUNCTION__)
  #endif
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {


#ifndef CUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED
#define CUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED 0
#endif


// CUDA 10.1 introduces the mma instruction
#if !defined(CUTLASS_ENABLE_TENSOR_CORE_MMA)
#define CUTLASS_ENABLE_TENSOR_CORE_MMA 0
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CUTLASS_ASSERT(x) assert(x)

////////////////////////////////////////////////////////////////////////////////////////////////////

// CUTLASS_PRAGMA_(UNROLL|NO_UNROLL) optimization directives for the CUDA compiler.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__INTELLISENSE__)
  #if defined(__HIPCC__) || (defined(__clang__) && defined(__CUDA__))
    #define CUTLASS_PRAGMA_UNROLL _Pragma("unroll")
    #define CUTLASS_PRAGMA_NO_UNROLL _Pragma("unroll 1")
  #else
    #define CUTLASS_PRAGMA_UNROLL #pragma unroll
    #define CUTLASS_PRAGMA_NO_UNROLL #pragma unroll 1
  #endif

  #define CUTLASS_GEMM_LOOP CUTLASS_PRAGMA_NO_UNROLL

#else

    #define CUTLASS_PRAGMA_UNROLL
    #define CUTLASS_PRAGMA_NO_UNROLL _Pragma("nounroll")
    #define CUTLASS_GEMM_LOOP CUTLASS_PRAGMA_NO_UNROLL

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if !defined(__CUDACC_RTC__)
#define CUTLASS_THREAD_LOCAL thread_local
#else
#define CUTLASS_THREAD_LOCAL
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if (201700L <= __cplusplus)
#define CUTLASS_CONSTEXPR_IF_CXX17 constexpr
#define CUTLASS_CXX17_OR_LATER 1
#else
#define CUTLASS_CONSTEXPR_IF_CXX17
#define CUTLASS_CXX17_OR_LATER 0
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#define HAS_ATTRIBUTE_DEF(Attribute)                                           \
  template <typename T>                                                        \
  constexpr auto has_attribute_##Attribute(int)                                \
      -> decltype(std::declval<typename T::Attribute>(), std::true_type());    \
  template <typename T>                                                        \
  constexpr std::false_type has_attribute_##Attribute(...);                    \
  template <typename T>                                                        \
  constexpr bool has_attribute_##Attribute##_v =                               \
      decltype(has_attribute_##Attribute<T>(0))::value;

#define HAS_MEMBER_DEF(Member)                                                 \
  template <typename T>                                                        \
  static constexpr auto has_member_##Member(T &&)                              \
      -> decltype(typename std::decay<                                         \
                      decltype(std::declval<T>().Member)>::type(),             \
                  std::true_type()) {                                          \
    return std::true_type();                                                   \
  }                                                                            \
  static constexpr std::false_type has_member_##Member(...) {                  \
    return std::false_type();                                                  \
  }                                                                            \
  template <typename T>                                                        \
  static constexpr bool has_member_##Member##_v =                              \
      decltype(has_member_##Member(std::declval<T>()))::value;                 \
  template <typename Tclass, typename Tvalue>                                  \
  static constexpr auto get_constant_member_##Member(Tvalue default_value) {   \
    if constexpr (has_member_##Member##_v<Tclass>) {                           \
      return Tclass::Member;                                                   \
    } else {                                                                   \
      return default_value;                                                    \
    }                                                                          \
  }

////////////////////////////////////////////////////////////////////////////////////////////////////

}; // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////
