/*
 * Copyright (c) 2025, InfiniTensor.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_rms_norm.h"

namespace ascend_kernel {

at::Tensor rms_norm(const at::Tensor &input, const at::Tensor &weight,
                    double eps) {
    // Input validation.
    TORCH_CHECK(input.dim() > 0,
                "rms_norm: input must have at least 1 dimension");
    TORCH_CHECK(input.scalar_type() == at::kHalf ||
                    input.scalar_type() == at::kFloat,
                "rms_norm: only float16 and float32 are supported, got ",
                input.scalar_type());
    TORCH_CHECK(weight.dim() == 1,
                "rms_norm: weight must be 1-dimensional");
    TORCH_CHECK(weight.size(0) == input.size(-1),
                "rms_norm: weight size (", weight.size(0),
                ") must match input last dim (", input.size(-1), ")");

    int64_t dimLength = input.size(-1);
    int64_t totalRows = input.numel() / dimLength;

    if (totalRows == 0 || dimLength == 0) {
        return at::empty_like(input);
    }

    at::Tensor x = input.contiguous();
    int64_t dtypeSize = x.element_size();

    // Hardware parameters.
    auto ascendc_platform =
        platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum =
        static_cast<int64_t>(ascendc_platform->GetCoreNumAiv());
    uint64_t ubSize;
    ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                                     ubSize);
    int64_t ubSizeLimit = static_cast<int64_t>(ubSize);

    // Alignment (32-byte boundary).
    int64_t alignElements = 32 / dtypeSize;
    int64_t dimLengthAlign =
        ((dimLength + alignElements - 1) / alignElements) * alignElements;

    // UB capacity check.
    // fp32: inQ(×2) + outQ(×2) + weight = 5 × dimLenAlign × 4 = coeff 20
    // fp16: inQ(×2) + outQ(×2) + xFp32 + tmpFp32 + weight
    //       = 2×dimLenAlign×2 ×2  + 3×dimLenAlign×4 = 8 + 12 = coeff 20
    int64_t bufferCoefficient = 20;
    int64_t maxDimLength =
        (ubSizeLimit - 1024) / bufferCoefficient;  // 1024 for reduce bufs.
    int64_t fpAlignElements = 32 / 4;  // fp32 alignment.
    maxDimLength =
        (maxDimLength / fpAlignElements) * fpAlignElements;
    TORCH_CHECK(dimLengthAlign <= maxDimLength,
                "rms_norm: dimLength ", dimLength,
                " (aligned ", dimLengthAlign,
                ") exceeds UB capacity (max ", maxDimLength, ")");

    // Padding.
    at::Tensor kernelInput;

    if (dimLength != dimLengthAlign) {
        kernelInput = x.reshape({totalRows, dimLength});
        kernelInput = at::constant_pad_nd(
            kernelInput, {0, dimLengthAlign - dimLength}, 0.0);
        kernelInput = kernelInput.contiguous();
    } else {
        kernelInput =
            x.reshape({totalRows, dimLengthAlign}).contiguous();
    }

    at::Tensor kernelOutput = at::empty_like(kernelInput);

    // Weight: always pass as fp32, padded to `dimLengthAlign`.
    at::Tensor weightFloat = weight.contiguous().to(at::kFloat);

    if (dimLength != dimLengthAlign) {
        weightFloat = at::constant_pad_nd(
            weightFloat, {0, dimLengthAlign - dimLength}, 0.0);
    }

    weightFloat = weightFloat.contiguous();

    // Block-level tiling (distribute rows across cores).
    int64_t usedCoreNum = std::min(totalRows, coreNum);
    int64_t formerLength =
        (totalRows + usedCoreNum - 1) / usedCoreNum;
    int64_t tailLength = formerLength - 1;
    int64_t formerNum = totalRows - tailLength * usedCoreNum;
    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

    // All EXEC_KERNEL_CMD args must be lvalues.
    float epsFloat = static_cast<float>(eps);
    int64_t dtypeSizeVal = dtypeSize;

    EXEC_KERNEL_CMD(rms_norm, blockDim,
                    kernelInput, weightFloat, kernelOutput,
                    totalRows, dimLength, dimLengthAlign,
                    formerNum, formerLength, tailLength,
                    epsFloat, dtypeSizeVal);

    // Remove padding and reshape back to original shape.
    at::Tensor output = kernelOutput;

    if (dimLength != dimLengthAlign) {
        output = output.narrow(-1, 0, dimLength).contiguous();
    }

    output = output.reshape(input.sizes());

    return output;
}

}  // namespace ascend_kernel
