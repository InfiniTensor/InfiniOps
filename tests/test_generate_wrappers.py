import functools
import importlib.util
from pathlib import Path

import pytest


pytest.importorskip("clang.cindex")


@functools.lru_cache(maxsize=1)
def _load_generator():
    script = Path(__file__).parents[1] / "scripts" / "generate_wrappers.py"
    spec = importlib.util.spec_from_file_location("generate_wrappers", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def _generate_binding(op_name, tmp_path, monkeypatch, source):
    generator = _load_generator()
    src_dir = tmp_path / "src"
    base_dir = src_dir / "base"
    base_dir.mkdir(parents=True)
    (base_dir / f"{op_name}.h").write_text(source)
    monkeypatch.setattr(generator, "_SRC_DIR", src_dir)
    monkeypatch.setattr(generator, "_BASE_DIR", base_dir)
    operator = generator._OperatorExtractor()(op_name)

    return generator._generate_pybind11(operator)


def test_mha_varlen_fwd_requires_out_binding(tmp_path, monkeypatch):
    text = _generate_binding(
        "mha_varlen_fwd",
        tmp_path,
        monkeypatch,
        """
#include <cstdint>
#include <optional>

namespace infini::ops {

struct Tensor {};

template <typename T>
class Operator {};

class MhaVarlenFwd : public Operator<MhaVarlenFwd> {
 public:
  MhaVarlenFwd(const Tensor q, const Tensor k, const Tensor v, Tensor out,
               const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
               std::optional<Tensor> block_table, float softmax_scale,
               bool is_causal, int64_t num_splits = 0) {}

  virtual void operator()(const Tensor q, const Tensor k, const Tensor v,
                          Tensor out, const Tensor cu_seqlens_q,
                          const Tensor cu_seqlens_k,
                          std::optional<Tensor> block_table,
                          float softmax_scale, bool is_causal,
                          int64_t num_splits = 0) const = 0;
};

}  // namespace infini::ops
""",
    )

    assert 'py::arg("out"), py::arg("cu_seqlens_q")' in text
    assert 'py::arg("num_splits") = 0' in text
    assert 'py::arg("out") = py::none()' not in text
    assert "std::optional<py::object> out" not in text
    assert "OptionalTensorFromPybind11Handle(out)" not in text


def test_mha_fwd_kvcache_requires_out_binding(tmp_path, monkeypatch):
    text = _generate_binding(
        "mha_fwd_kvcache",
        tmp_path,
        monkeypatch,
        """
#include <cstdint>
#include <optional>

namespace infini::ops {

struct Tensor {};

template <typename T>
class Operator {};

class MhaFwdKvcache : public Operator<MhaFwdKvcache> {
 public:
  MhaFwdKvcache(const Tensor q, const Tensor kcache, const Tensor vcache,
                std::optional<Tensor> k, std::optional<Tensor> v, Tensor out,
                float softmax_scale, bool is_causal,
                int64_t num_splits = 0) {}

  virtual void operator()(const Tensor q, const Tensor kcache,
                          const Tensor vcache, std::optional<Tensor> k,
                          std::optional<Tensor> v, Tensor out,
                          float softmax_scale, bool is_causal,
                          int64_t num_splits = 0) const = 0;
};

}  // namespace infini::ops
""",
    )

    assert 'py::arg("out"), py::arg("softmax_scale")' in text
    assert 'py::arg("num_splits") = 0' in text
    assert 'py::arg("out") = py::none()' not in text
    assert "std::optional<py::object> out" not in text
    assert "OptionalTensorFromPybind11Handle(out)" not in text
