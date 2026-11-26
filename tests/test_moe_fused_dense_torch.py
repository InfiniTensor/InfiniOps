"""CPU regression for the Torch MoE backend's routing map and bias boundaries."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import include_paths

CPP = r'''
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <infini/ops.h>
at::Tensor run_moe(at::Tensor x, at::Tensor a, at::Tensor b, at::Tensor route,
                   at::Tensor ids, at::Tensor sorted, at::Tensor experts,
                   at::Tensor total, int64_t block, std::vector<at::Tensor> biases) {
  using namespace infini::ops;
  auto output = at::empty_like(x);
  const auto wrap = [](const at::Tensor &t) {
    auto type = t.scalar_type() == at::kFloat ? DataType::kFloat32
              : t.scalar_type() == at::kInt ? DataType::kInt32
              : t.scalar_type() == at::kHalf ? DataType::kFloat16 : DataType::kBFloat16;
    return Tensor(const_cast<void *>(t.data_ptr()),
                  std::vector<Tensor::Size>(t.sizes().begin(), t.sizes().end()),
                  type, Device{Device::Type::kCpu},
                  std::vector<Tensor::Size>(t.strides().begin(), t.strides().end()));
  };
  std::optional<Tensor> bias1, bias2;
  if (!biases.empty()) { bias1.emplace(wrap(biases.at(0))); bias2.emplace(wrap(biases.at(1))); }
  MoeFusedDense::Call(wrap(output), wrap(x), wrap(a), wrap(b), wrap(route), wrap(ids),
                      wrap(sorted), wrap(experts), wrap(total), static_cast<size_t>(block), bias1, bias2);
  return output;
}
PYBIND11_MODULE(_moe_dense_probe, m) { m.def("run_moe", &run_moe); }
'''


@unittest.skipUnless(os.environ.get('INFINI_ROOT'), 'requires an installed InfiniOps')
class MoeFusedDenseTorchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        folder = Path(cls.temp.name)
        source = folder / 'probe.cpp'
        source.write_text(CPP)
        extension = folder / ('_moe_dense_probe' + sysconfig.get_config_var('EXT_SUFFIX'))
        root = Path(os.environ['INFINI_ROOT'])
        torchlib = Path(torch.__file__).parent / 'lib'
        command = ['g++', '-shared', '-fPIC', '-std=c++17', '-O1', '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)), str(source), '-o', str(extension)]
        for directory in [*include_paths(), sysconfig.get_paths()['include'], str(root/'include')]:
            command.extend(['-I', directory])
        command.extend(['-L'+str(root/'lib'), '-linfiniops', '-L'+str(torchlib), '-ltorch', '-ltorch_cpu', '-ltorch_python', '-lc10', '-Wl,-rpath,'+str(root/'lib'), '-Wl,-rpath,'+str(torchlib)])
        subprocess.run(command, check=True)
        spec = importlib.util.spec_from_file_location('_moe_dense_probe', extension)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def check_case(self, remapped, bias):
        torch.manual_seed(123)
        dtype = torch.bfloat16
        x = (torch.randn(2, 8)*0.5).to(dtype)
        a = (torch.randn(2, 8, 8)*0.5).to(dtype)
        b = (torch.randn(2, 8, 4)*0.5).to(dtype)
        route = torch.tensor([[0.25, 0.75], [0.625, 0.375]], dtype=torch.float32)
        ids = torch.tensor([[0, 3], [2, 1]] if remapped else [[0, 1], [1, 0]], dtype=torch.int32)
        # Every local expert owns one padded block; remaining entries are sentinels.
        sorted_ids = torch.full((16,), 4, dtype=torch.int32)
        experts = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
        if remapped:
            sorted_ids[0], sorted_ids[4] = 2, 0
            mapping = {0: 1, 2: 0}
        else:
            sorted_ids[:2] = torch.tensor([0, 3]); sorted_ids[4:6] = torch.tensor([1, 2])
            mapping = {0: 0, 1: 1}
        biases = [(torch.randn(2, 8)*0.1).to(dtype), (torch.randn(2, 8)*0.1).to(dtype)] if bias else []
        actual = self.module.run_moe(x, a, b, route, ids, sorted_ids, experts, torch.tensor([8],dtype=torch.int32), 4, biases)
        expected = torch.zeros_like(x)
        for token in range(2):
            parts = []
            for slot in range(2):
                expert = mapping.get(int(ids[token, slot]))
                if expert is None:
                    continue
                gate_up = F.linear(x[token].float(), a[expert].float())
                if bias:
                    gate_up += biases[0][expert].float()
                gate_up = gate_up.to(dtype)
                activated = (F.silu(gate_up[:4].float())*gate_up[4:].float()).to(dtype)
                down = F.linear(activated.float(), b[expert].float())
                if bias:
                    down += biases[1][expert].float()
                parts.append((down*route[token, slot]).to(dtype).float())
            if parts:
                expected[token] = torch.stack(parts).sum(0).to(dtype)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.002)

    def test_tensor_parallel_routes(self):
        self.check_case(False, False)

    def test_remapped_local_experts_and_absent_routes(self):
        self.check_case(True, False)

    def test_bias_is_added_before_reduced_precision_rounding(self):
        self.check_case(True, True)


if __name__ == '__main__':
    unittest.main()
