"""Optional backend coverage and explicit acceptance requirements."""

from types import SimpleNamespace

import pytest

from tests import conftest


@pytest.mark.parametrize("tested_device", ["cpu", "musa"])
@pytest.mark.parametrize("strict", [False, True])
def test_missing_operator_requirement(monkeypatch, tested_device, strict):
    import infini.ops

    monkeypatch.setattr(infini.ops, "OnnxGather", None, raising=False)
    options = {"--devices": [], "--require-operator-set": strict}
    request = SimpleNamespace(
        config=SimpleNamespace(getoption=options.get),
        node=SimpleNamespace(
            callspec=SimpleNamespace(params={"device": tested_device}),
            get_closest_marker=lambda _: pytest.mark.operator_set(
                {"OnnxGather": 8}
            ).mark,
        ),
    )
    expected = pytest.fail.Exception if strict else pytest.skip.Exception
    with pytest.raises(expected, match="OnnxGather"):
        conftest.skip_op_without_platform_impl.__wrapped__(request)


@pytest.mark.parametrize("strict", [False, True])
def test_missing_operators_prevent_cpp_fixture_build(monkeypatch, strict):
    import infini.ops
    from tests.test_encoder_external_memory import external_memory_module

    monkeypatch.setattr(infini.ops, "OnnxGather", None, raising=False)
    options = {"--devices": ["cpu"], "--require-operator-set": strict}
    request = SimpleNamespace(config=SimpleNamespace(getoption=options.get))
    expected = pytest.fail.Exception if strict else pytest.skip.Exception
    with pytest.raises(expected, match="requires the retrieval operator set"):
        external_memory_module.__wrapped__(None, request)
