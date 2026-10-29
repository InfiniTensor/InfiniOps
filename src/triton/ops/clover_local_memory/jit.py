# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang PR #46927, commit 058939af24d5cdc44bb4b8852d0ddfab555ed71e.
# Changes: InfiniOps JIT entry point, Hygon launch geometry and pool bounds.
import triton
import triton.language as tl


@triton.jit
def clover_local_memory_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    g_ptr,
    clean_state_ptr,
    cache_indices_ptr,
    active_valid_ptr,
    softmax_logit_scale_ptr,
    softmax_history_bias_ptr,
    softmax_history_c_ptr,
    output_ptr,
    query_stride_t,
    query_stride_h,
    query_stride_k,
    key_stride_t,
    key_stride_h,
    key_stride_k,
    value_stride_t,
    value_stride_h,
    value_stride_v,
    g_stride_t,
    g_stride_h,
    state_stride_slot,
    state_stride_h,
    state_stride_v,
    state_stride_k,
    output_stride_t,
    output_stride_h,
    output_stride_v,
    scale,
    softmax_reference_block_size,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    NUM_QUERY_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Compute the read-only BD-GDN preview output for one fixed-size block.

    Block-local softmax includes the gate-decayed clean readout as one
    history bucket. Its logit is history_bias plus
    history_c * log(valid_block_size / reference_block_size).

    One program handles one request, one value head, and one value-dimension
    tile. Persistent state is read directly in the [slot, H, V, K]
    layout and is never written.
    """
    value_tile = tl.program_id(0)
    query_program = tl.program_id(1)
    request_idx = query_program // NUM_QUERY_TILES
    query_tile = query_program % NUM_QUERY_TILES
    value_head = tl.program_id(2)

    token_offsets = query_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
    local_key_offsets = tl.arange(0, BLOCK_T)
    key_offsets = tl.arange(0, BLOCK_K)
    value_offsets = value_tile * BLOCK_V + tl.arange(0, BLOCK_V)
    token_mask = token_offsets < BLOCK_SIZE
    local_key_mask = local_key_offsets < BLOCK_SIZE
    key_mask = key_offsets < KEY_DIM
    value_mask = value_offsets < VALUE_DIM
    token_indices = request_idx * BLOCK_SIZE + token_offsets

    cache_index = tl.load(cache_indices_ptr + request_idx)
    request_active = (cache_index >= 0) & (cache_index < NUM_SLOTS)
    safe_cache_index = tl.maximum(cache_index, 0)
    active = (
        tl.load(active_valid_ptr + token_indices, mask=token_mask, other=0) != 0
    ) & request_active

    # Value heads may share Q/K heads. Clover maps pairs of value heads to
    # one of 16 Q/K heads without materializing repeat_interleave.
    query_head = value_head // (NUM_V_HEADS // NUM_Q_HEADS)
    key_head = value_head // (NUM_V_HEADS // NUM_K_HEADS)

    query_offsets = (
        token_indices[:, None] * query_stride_t
        + query_head * query_stride_h
        + key_offsets[None, :] * query_stride_k
    )
    key_offsets_transposed = (
        (request_idx * BLOCK_SIZE + local_key_offsets)[None, :] * key_stride_t
        + key_head * key_stride_h
        + key_offsets[:, None] * key_stride_k
    )
    value_matrix_offsets = (
        (request_idx * BLOCK_SIZE + local_key_offsets)[:, None] * value_stride_t
        + value_head * value_stride_h
        + value_offsets[None, :] * value_stride_v
    )

    query = tl.load(
        query_ptr + query_offsets,
        mask=token_mask[:, None] & key_mask[None, :] & request_active,
        other=0.0,
    )
    key_transposed = tl.load(
        key_ptr + key_offsets_transposed,
        mask=key_mask[:, None] & local_key_mask[None, :] & request_active,
        other=0.0,
    )
    values = tl.load(
        value_ptr + value_matrix_offsets,
        mask=local_key_mask[:, None] & value_mask[None, :] & request_active,
        other=0.0,
    )

    # Match the GDN epsilon and normalize before Tensor Core dot products.
    query_f32 = query.to(tl.float32)
    query_norm = tl.rsqrt(tl.sum(query_f32 * query_f32, axis=1) + 1.0e-6)
    query = (query_f32 * query_norm[:, None]).to(query.dtype)
    key_f32 = key_transposed.to(tl.float32)
    key_norm = tl.rsqrt(tl.sum(key_f32 * key_f32, axis=0) + 1.0e-6)
    key_transposed = (key_f32 * key_norm[None, :]).to(key_transposed.dtype)

    state_offsets = (
        safe_cache_index * state_stride_slot
        + value_head * state_stride_h
        + value_offsets[None, :] * state_stride_v
        + key_offsets[:, None] * state_stride_k
    )
    clean_state = tl.load(
        clean_state_ptr + state_offsets,
        mask=key_mask[:, None] & value_mask[None, :] & request_active,
        other=0.0,
    )

    scalar_offsets = token_indices * g_stride_t + value_head * g_stride_h
    g = tl.load(g_ptr + scalar_offsets, mask=token_mask & request_active, other=0.0).to(
        tl.float32
    )
    local_active = (
        tl.load(
            active_valid_ptr + request_idx * BLOCK_SIZE + local_key_offsets,
            mask=local_key_mask,
            other=0,
        )
        != 0
    ) & request_active

    history = tl.dot(query, clean_state.to(query.dtype))
    scores = tl.dot(query, key_transposed)
    logit_scale = tl.load(softmax_logit_scale_ptr + value_head).to(tl.float32)
    history_bias = tl.load(softmax_history_bias_ptr + value_head).to(tl.float32)
    history_c = tl.load(softmax_history_c_ptr + value_head).to(tl.float32)
    valid_block_size = tl.maximum(tl.sum(local_active, axis=0), 1.0)
    history_bias += history_c * tl.log(valid_block_size / softmax_reference_block_size)
    logits = scores * logit_scale
    logits = tl.where(local_active[None, :], logits, float("-inf"))
    row_max = tl.maximum(tl.max(logits, axis=1), history_bias)
    exp_local = tl.exp(logits - row_max[:, None])
    exp_history = tl.exp(history_bias - row_max)
    denominator = tl.sum(exp_local, axis=1) + exp_history
    local_weights = exp_local / denominator[:, None]
    history_weight = exp_history / denominator
    local = tl.dot(local_weights.to(values.dtype), values)
    output = history * (scale * tl.exp(g) * history_weight)[:, None] + local
    output = tl.where(active[:, None], output, 0.0)

    output_offsets = (
        token_indices[:, None] * output_stride_t
        + value_head * output_stride_h
        + value_offsets[None, :] * output_stride_v
    )
    tl.store(
        output_ptr + output_offsets,
        output,
        mask=token_mask[:, None] & value_mask[None, :],
    )


kernel = clover_local_memory_kernel
