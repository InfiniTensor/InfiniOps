/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Basic include for CUTLASS.
*/

/*
  Note:  CUTLASS 3x increases the host compiler requirements to C++17. However, certain
         existing integrations of CUTLASS require C++11 host compilers.

         Until this requirement can be lifted, certain headers with this annotation are required
         to be remain consistent with C++11 syntax.

         C++11 compatibility is enforced by `cutlass_test_unit_core_cpp11`.
*/

#pragma once

#include "cutlass/detail/helper_macros.hpp"
#include "hip/hip_runtime.h"

// Support __shfl_sync while dtk do not have this head file
#if __has_include(<hip/amd_detail/amd_warp_sync_functions.h>)  // check if __shfl_sync is provided by dtk
  // has __shfl_sync
#else

// #define __hip_check_mask(MASK)                                                 \
//   do {                                                                         \
//     __hip_assert(MASK && "mask must be non-zero");                             \
//     bool done = false;                                                         \
//     while (__any(!done)) {                                                     \
//       if (!done) {                                                             \
//         auto chosen_mask = __hip_readfirstlane(MASK);                          \
//         if (MASK == chosen_mask) {                                             \
//           __hip_assert(MASK == __ballot(true) &&                               \
//                        "all threads specified in the mask"                     \
//                        " must execute the same operation with the same mask"); \
//           done = true;                                                         \
//         }                                                                      \
//       }                                                                        \
//     }                                                                          \
//   } while(0)

template <typename MaskT, typename T>
__device__ inline
T __shfl_sync(MaskT mask, T var, int srcLane,
              int width = __AMDGCN_WAVEFRONT_SIZE) {
  // dtk23.04 
  // static_assert(
  //     __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
  //     "The mask must be a 64-bit integer. "
  //     "Implicitly promoting a smaller integer is almost always an error.");
  // __hip_check_mask(mask);
  return __shfl(var, srcLane, width);
}  

#endif   // end check

#define WARP_SIZE_GPU 64
/*
  Note: use v_fma_mix_fp32
*/
// #define MIX_FP16XFP16_SM60
/*
  Note: use dot instruction
*/
#define MIX_FP16_DOT2

/*
  Note: default epilogue use acc-fp32 compute-fp32, compute finish convertTo fp16.
        Turn on this macro, which means we try to compute in fp16 data type.
        Also we should change epilogue_vector_length(in generator.py) to use v_cvt_pkrtz_f16_f32 instruction.
*/
// #define MIX_FP16_EPILOGUE
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/// Status code returned by CUTLASS operations
enum class Status {
  kSuccess,                    ///< Operation was successful.
  kErrorMisalignedOperand,     ///< operands fail alignment requirements.
  kErrorInvalidDataType,       ///< DataType fails requirement.
  kErrorInvalidLayout,         ///< Layout fails alignment requirement.
  kErrorInvalidProblem,        ///< Specified problem size is not supported by operator.
  kErrorNotSupported,          ///< Operation is not supported on current device.
  kErrorWorkspaceNull,         ///< The given workspace is null when it is required to be non-null.
  kErrorInternal,              ///< An error within CUTLASS occurred.
  kErrorArchMismatch,          ///< CUTLASS runs on a device that it was not compiled for.
  kErrorInsufficientDriver,    ///< CUTLASS runs with a driver that is too old.
  kErrorMemoryAllocation,      ///< Kernel launch failed due to insufficient device memory.
  kInvalid                     ///< Status is unspecified.
};

/// Convert cutlass status to status strings
CUTLASS_HOST_DEVICE
static char const* cutlassGetStatusString(cutlass::Status status) {
  switch (status) {
    case cutlass::Status::kSuccess:
      return "Success";
    case cutlass::Status::kErrorMisalignedOperand:
      return "Error Misaligned Operand";
    case cutlass::Status::kErrorInvalidDataType:
      return "Error Invalid Data Type";
    case cutlass::Status::kErrorInvalidLayout:
      return "Error Invalid Layout";
    case cutlass::Status::kErrorInvalidProblem:
      return "Error Invalid Problem";
    case cutlass::Status::kErrorNotSupported:
      return "Error Not Supported";
    case cutlass::Status::kErrorWorkspaceNull:
      return "Error Workspace Null";
    case cutlass::Status::kErrorInternal:
      return "Error Internal";
    case cutlass::Status::kErrorInsufficientDriver:
      return "Error Insufficient Driver";
    case cutlass::Status::kErrorArchMismatch:
      return "Error Architecture Mismatch";
    case cutlass::Status::kErrorMemoryAllocation:
      return "Error Memory Allocation failed";
    case cutlass::Status::kInvalid: break;
  }

  return "Invalid status";
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// static const int NumThreadsPerWarp = 32;
static const int NumThreadsPerWarp = WARP_SIZE_GPU;
static const int NumThreadsPerWarpGroup = 128;
static const int NumWarpsPerWarpGroup = NumThreadsPerWarpGroup / NumThreadsPerWarp;
static const int NumThreadsPerHalfWarp = NumThreadsPerWarp / 2;
static const int NumThreadsPerQuad = 4;
static const int NumThreadsPerQuadPair = NumThreadsPerQuad * 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper function to return true when called by thread 0 of threadblock 0.
CUTLASS_HOST_DEVICE bool thread0() {
  #if defined(__HIP_DEVICE_COMPILE__)
    return (!threadIdx.x && !threadIdx.y && !threadIdx.z) && (!blockIdx.x && !blockIdx.y && !blockIdx.z);
  #else
    return false;
  #endif
}

/// Returns a lane index in the warp. The threads in warp may not be convergent
CUTLASS_DEVICE
int canonical_lane_idx() { 
  #if defined(__HIP_DEVICE_COMPILE__)
    return threadIdx.x % NumThreadsPerWarp;
  #else
    return 0;
  #endif
}

/// Returns a warp-uniform value indicating the canonical warp index of the calling threads.
/// Threads within the warp must be converged.
CUTLASS_DEVICE
int canonical_warp_idx_sync() { 
  #if defined(__HIP_DEVICE_COMPILE__)
    #ifdef DCU_ASM
      return (threadIdx.x / NumThreadsPerWarp);
    #else
      return __shfl_sync((uint64_t)0xffffffff, threadIdx.x / NumThreadsPerWarp, 0);
    #endif
  #else
    return 0;
  #endif
}

/// Returns a warp index in the CTA. The threads in warp may not be convergent
/// As it doesn't sync the warp, it faster and allows forward progress
CUTLASS_DEVICE
int canonical_warp_idx() { 
  #if defined(__HIP_DEVICE_COMPILE__)
    return threadIdx.x / NumThreadsPerWarp;
  #else
    return 0;
  #endif
}

/// Returns a warp-uniform value indicating the canonical warp group index of the calling threads.
/// Threads within the warp must be converged.
CUTLASS_DEVICE
int canonical_warp_group_idx() {
  #if defined(__HIP_DEVICE_COMPILE__)
    // return __shfl_sync((uint64_t)0xffffffff, threadIdx.x / NumThreadsPerWarpGroup, 0);
    return __shfl(threadIdx.x / NumThreadsPerWarpGroup, 0);
  #else
    return 0;
  #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////
