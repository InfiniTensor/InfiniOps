import pytest

from scripts.operator_categories import (
    BASIC_MATH,
    COMPARISON_LOGIC_BITWISE,
    FFT_SPECIAL,
    LINALG,
    NEURAL_NETWORK,
    REDUCTION_STAT_SORT,
    TENSOR_ORGANIZATION,
    category_for_operator,
    inventory_operator_name,
    is_native_operator,
    normalize_category_selectors,
    selectors_slug,
)


def test_category_for_operator_covers_report_categories():
    assert category_for_operator("add") == BASIC_MATH
    assert category_for_operator("eq") == COMPARISON_LOGIC_BITWISE
    assert category_for_operator("sum") == REDUCTION_STAT_SORT
    assert category_for_operator("mm") == LINALG
    assert category_for_operator("softmax") == NEURAL_NETWORK
    assert category_for_operator("gather") == TENSOR_ORGANIZATION
    assert category_for_operator("fft_fft") == FFT_SPECIAL


def test_internal_aten_names_are_mapped_to_inventory_names():
    assert inventory_operator_name("internal_fft_c2c", "_fft_c2c") == "aten_fft_c2c"
    assert inventory_operator_name("xlogy", "xlogy_") == "xlogy_inplace"


def test_native_operator_names_do_not_include_script_tests():
    assert is_native_operator("matmul")
    assert not is_native_operator("dev_scripts")


def test_category_selectors_accept_cli_slugs_and_chinese_names():
    assert normalize_category_selectors(("linalg", "神经网络常用算子")) == (
        LINALG,
        NEURAL_NETWORK,
    )
    assert selectors_slug(("comparison-logic-bitwise",)) == "comparison-logic-bitwise"


def test_unknown_category_selector_is_rejected():
    with pytest.raises(ValueError, match="unknown operator category"):
        normalize_category_selectors(("not-a-category",))
