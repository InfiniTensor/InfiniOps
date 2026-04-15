/*
 * Copyright (c) 2025, InfiniTensor.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_add_rms_norm.h"

namespace ascend_kernel {

std::vector<at::Tensor> add_rms_norm(const at::Tensor &x1,
                                     const at::Tensor &x2,
                                     const at::Tensor &weight, double eps) {
    // Input validation.
    TORCH_CHECK(x1.dim() > 0,
                "add_rms_norm: x1 must have at least 1 dimension");
    TORCH_CHECK(x1.sizes() == x2.sizes(),
                "add_rms_norm: x1 and x2 must have the same shape");
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(),
                "add_rms_norm: x1 and x2 must have the same dtype");
    TORCH_CHECK(x1.scalar_type() == at::kHalf ||
                    x1.scalar_type() == at::kFloat,
                "add_rms_norm: only float16 and float32 are supported, got ",
                x1.scalar_type());
    TORCH_CHECK(weight.dim() == 1,
                "add_rms_norm: weight must be 1-dimensional");
    TORCH_CHECK(weight.size(0) == x1.size(-1),
                "add_rms_norm: weight size (", weight.size(0),
                ") must match input last dim (", x1.size(-1), ")");

    int64_t dimLength = x1.size(-1);
    int64_t totalRows = x1.numel() / dimLength;

    if (totalRows == 0 || dimLength == 0) {
        return {at::empty_like(x1), at::empty_like(x1)};
    }

    at::Tensor inp1 = x1.contiguous();
    at::Tensor inp2 = x2.contiguous();
    int64_t dtypeSize = inp1.element_size();

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
    // fp16: inQ_x1(×2×2) + inQ_x2(×2×2) + outQ_y(×2×2) + outQ_xout(×2×2)
    //       + fp32Buf1(×4) + fp32Buf2(×4) + weight(×4) = 16 + 12 = 28
    // fp32: inQ_x1(×2×4) + inQ_x2(×2×4) + outQ_y(×2×4) + outQ_xout(×2×4)
    //       + weight(×4) = 32 + 4 = 36
    int64_t bufferCoefficient = (dtypeSize == 2) ? 28 : 36;
    int64_t maxDimLength =
        (ubSizeLimit - 1024) / bufferCoefficient;
    int64_t fpAlignElements = 32 / 4;
    maxDimLength =
        (maxDimLength / fpAlignElements) * fpAlignElements;
    TORCH_CHECK(dimLengthAlign <= maxDimLength,
                "add_rms_norm: dimLength ", dimLength,
                " (aligned ", dimLengthAlign,
                ") exceeds UB capacity (max ", maxDimLength, ")");

    // Padding.
    at::Tensor kernelInput1;
    at::Tensor kernelInput2;

    if (dimLength != dimLengthAlign) {
        kernelInput1 = inp1.reshape({totalRows, dimLength});
        kernelInput1 = at::constant_pad_nd(
            kernelInput1, {0, dimLengthAlign - dimLength}, 0.0);
        kernelInput1 = kernelInput1.contiguous();

        kernelInput2 = inp2.reshape({totalRows, dimLength});
        kernelInput2 = at::constant_pad_nd(
            kernelInput2, {0, dimLengthAlign - dimLength}, 0.0);
        kernelInput2 = kernelInput2.contiguous();
    } else {
        kernelInput1 =
            inp1.reshape({totalRows, dimLengthAlign}).contiguous();
        kernelInput2 =
            inp2.reshape({totalRows, dimLengthAlign}).contiguous();
    }

    at::Tensor kernelOutputY = at::empty_like(kernelInput1);
    at::Tensor kernelOutputXOut = at::empty_like(kernelInput1);

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

    EXEC_KERNEL_CMD(add_rms_norm, blockDim,
                    kernelInput1, kernelInput2, weightFloat,
                    kernelOutputY, kernelOutputXOut,
                    totalRows, dimLength, dimLengthAlign,
                    formerNum, formerLength, tailLength,
                    epsFloat, dtypeSizeVal);

    // Remove padding and reshape back to original shape.
    at::Tensor outputY = kernelOutputY;
    at::Tensor outputXOut = kernelOutputXOut;

    if (dimLength != dimLengthAlign) {
        outputY = outputY.narrow(-1, 0, dimLength).contiguous();
        outputXOut = outputXOut.narrow(-1, 0, dimLength).contiguous();
    }

    outputY = outputY.reshape(x1.sizes());
    outputXOut = outputXOut.reshape(x1.sizes());

    return {outputY, outputXOut};
}

}  // namespace ascend_kernel
