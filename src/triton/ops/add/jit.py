import triton
import triton.language as tl


@triton.jit
def kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    output_shape_ptr,
    input_strides_ptr,
    other_strides_ptr,
    output_strides_ptr,
    is_input_contiguous,
    is_other_contiguous,
    is_output_contiguous,
    ndim,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)
    linear_offsets = (program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = linear_offsets < n_elements

    if (
        (is_input_contiguous != 0)
        and (is_other_contiguous != 0)
        and (is_output_contiguous != 0)
    ):
        input_value = tl.load(input_ptr + linear_offsets, mask=mask)
        other_value = tl.load(other_ptr + linear_offsets, mask=mask)
        tl.store(
            output_ptr + linear_offsets,
            input_value + other_value * alpha,
            mask=mask,
        )
    else:
        input_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
        other_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
        output_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
        remaining_index = linear_offsets

        for dimension_offset in range(ndim):
            dimension = ndim - 1 - dimension_offset
            dimension_size = tl.load(output_shape_ptr + dimension)
            dimension_index = remaining_index % dimension_size
            remaining_index = remaining_index // dimension_size
            input_offsets += dimension_index * tl.load(input_strides_ptr + dimension)
            other_offsets += dimension_index * tl.load(other_strides_ptr + dimension)
            output_offsets += dimension_index * tl.load(output_strides_ptr + dimension)

        if is_input_contiguous != 0:
            input_offsets = linear_offsets
        if is_other_contiguous != 0:
            other_offsets = linear_offsets
        if is_output_contiguous != 0:
            output_offsets = linear_offsets

        input_value = tl.load(input_ptr + input_offsets, mask=mask)
        other_value = tl.load(other_ptr + other_offsets, mask=mask)
        tl.store(
            output_ptr + output_offsets,
            input_value + other_value * alpha,
            mask=mask,
        )
