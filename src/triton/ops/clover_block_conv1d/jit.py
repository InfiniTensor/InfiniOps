# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang PR #46927, commit 058939af24d5cdc44bb4b8852d0ddfab555ed71e.
# Changes: InfiniOps JIT entry point, Hygon launch geometry and pool bounds.
import triton
import triton.language as tl


@triton.jit
def clover_block_conv1d_kernel(
    input_ptr,
    clean_state_ptr,
    cache_indices_ptr,
    weight_ptr,
    bias_ptr,
    right_weight_ptr,
    active_valid_ptr,
    output_ptr,
    input_stride_t,
    input_stride_c,
    state_stride_slot,
    state_stride_c,
    state_stride_w,
    weight_stride_c,
    weight_stride_w,
    right_weight_stride_c,
    right_weight_stride_w,
    output_stride_t,
    output_stride_c,
    NUM_CHANNELS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    INDEPENDENT_SYMMETRIC: tl.constexpr,
    BLOCK_T: tl.constexpr,
    NUM_TOKEN_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channel_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    token_program = tl.program_id(1)
    request_idx = token_program // NUM_TOKEN_TILES
    token_tile = token_program % NUM_TOKEN_TILES
    token_offsets = token_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    channel_mask = channel_offsets < NUM_CHANNELS
    token_mask = token_offsets < BLOCK_SIZE
    token_indices = request_idx * BLOCK_SIZE + token_offsets

    cache_index = tl.load(cache_indices_ptr + request_idx)
    request_active = (cache_index >= 0) & (cache_index < NUM_SLOTS)
    safe_cache_index = tl.maximum(cache_index, 0)
    output_active = (
        tl.load(active_valid_ptr + token_indices, mask=token_mask, other=0) != 0
    ) & request_active

    if HAS_BIAS:
        bias = tl.load(bias_ptr + channel_offsets, mask=channel_mask, other=0.0).to(
            tl.float32
        )
    else:
        bias = tl.zeros((BLOCK_C,), dtype=tl.float32)
    accumulator = tl.broadcast_to(bias[None, :], (BLOCK_T, BLOCK_C))

    for tap in tl.static_range(WIDTH):
        relative_source = token_offsets + tap - (WIDTH - 1)
        from_current = relative_source >= 0
        current_token = request_idx * BLOCK_SIZE + tl.maximum(relative_source, 0)
        current_valid = (
            tl.load(
                active_valid_ptr + current_token,
                mask=token_mask & from_current,
                other=0,
            )
            != 0
        ) & request_active
        current_value = tl.load(
            input_ptr
            + current_token[:, None] * input_stride_t
            + channel_offsets[None, :] * input_stride_c,
            mask=(token_mask & from_current & current_valid)[:, None]
            & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        history_offset = relative_source + (WIDTH - 1)
        history_value = tl.load(
            clean_state_ptr
            + safe_cache_index * state_stride_slot
            + channel_offsets[None, :] * state_stride_c
            + tl.maximum(history_offset, 0)[:, None] * state_stride_w,
            mask=(token_mask & ~from_current & request_active)[:, None]
            & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        source_value = tl.where(from_current[:, None], current_value, history_value)
        weight = tl.load(
            weight_ptr + channel_offsets * weight_stride_c + tap * weight_stride_w,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += source_value * weight[None, :]

    if INDEPENDENT_SYMMETRIC:
        right_accumulator = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
        for tap in tl.static_range(WIDTH - 1):
            distance = WIDTH - 1 - tap
            right_offset = token_offsets + distance
            right_token = request_idx * BLOCK_SIZE + right_offset
            right_valid = (
                tl.load(
                    active_valid_ptr + right_token,
                    mask=token_mask & (right_offset < BLOCK_SIZE),
                    other=0,
                )
                != 0
            ) & request_active
            right_value = tl.load(
                input_ptr
                + right_token[:, None] * input_stride_t
                + channel_offsets[None, :] * input_stride_c,
                mask=(token_mask & (right_offset < BLOCK_SIZE) & right_valid)[:, None]
                & channel_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(
                right_weight_ptr
                + channel_offsets * right_weight_stride_c
                + (distance - 1) * right_weight_stride_w,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            right_accumulator += right_value * weight[None, :]
        accumulator += right_accumulator

    output = accumulator * tl.sigmoid(accumulator)
    output = tl.where(output_active[:, None], output, 0.0)
    tl.store(
        output_ptr
        + token_indices[:, None] * output_stride_t
        + channel_offsets[None, :] * output_stride_c,
        output,
        mask=token_mask[:, None] & channel_mask[None, :],
    )


kernel = clover_block_conv1d_kernel
