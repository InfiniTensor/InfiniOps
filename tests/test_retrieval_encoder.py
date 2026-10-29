"""Exercise retrieval encoder operators together; this is not a full model."""

import math

import pytest
import torch

import infini.ops

from tests.utils import get_stream

pytestmark = pytest.mark.operator_set(
    {
        "OnnxGather": 8,
        "Cast": 8,
        "NativeLayerNorm": 8,
        "Matmul": 8,
        "Div": 8,
        "Sub": 8,
        "Mul": 8,
        "Add": 0,
        "Softmax": 8,
        "PermuteCopy": 8,
        "Erf": 8,
        "Tanh": 8,
        "Addmm": 8,
    }
)


def test_retrieval_attention_and_classifier(device):
    torch.manual_seed(17)
    width, heads = 32, 4
    table = torch.randn(97, width, device=device)
    weights = [torch.randn(width, width, device=device) * 0.05 for _ in range(3)]
    classifier = torch.randn(width, 1, device=device) * 0.05
    bias = torch.tensor([0.3], device=device)
    one = torch.tensor(1.0, device=device)
    negative = torch.tensor(-10000.0, device=device)
    scale = torch.tensor(math.sqrt(width // heads), device=device)

    def invoke(name, *args, implementation=8):
        getattr(infini.ops, name)(
            *args, implementation_index=implementation, stream=get_stream(device)
        )

    for batch, length in [(1, 1), (3, 17), (2, 83), (1, 1)]:
        ids = torch.randint(0, 97, (batch, length), device=device)
        mask = torch.arange(length, device=device) < max(1, length * 2 // 3)
        mask = mask.expand(batch, length).to(torch.int64)
        embedded = torch.empty((batch, length, width), device=device)
        invoke("onnx_gather", table, ids, 0, embedded)
        normalized = torch.empty_like(embedded)
        mean = torch.empty((batch, length, 1), device=device)
        rstd = torch.empty_like(mean)
        invoke(
            "native_layer_norm",
            embedded,
            [width],
            None,
            None,
            1e-5,
            normalized,
            mean,
            rstd,
        )
        projections = []
        for weight in weights:
            out = torch.empty_like(normalized)
            invoke("matmul", normalized, weight, out)
            projections.append(
                out.view(batch, length, heads, width // heads).transpose(1, 2)
            )
        query, key, value = projections
        scores = torch.empty((batch, heads, length, length), device=device)
        invoke("matmul", query, key.transpose(-2, -1), scores)
        scaled = torch.empty_like(scores)
        invoke("div", scores, scale, scaled)
        cast_mask = torch.empty((batch, length), device=device)
        invoke("cast", mask, cast_mask)
        inverted = torch.empty_like(cast_mask)
        invoke("sub", one, cast_mask, 1.0, inverted)
        penalty = torch.empty_like(inverted)
        invoke("mul", inverted, negative, penalty)
        masked = torch.empty_like(scores)
        invoke(
            "add",
            scaled,
            penalty[:, None, None, :].expand_as(scores),
            1.0,
            masked,
            implementation=0,
        )
        probabilities = torch.empty_like(scores)
        invoke("softmax", masked, -1, None, probabilities)
        attended = torch.empty((batch, heads, length, width // heads), device=device)
        invoke("matmul", probabilities, value, attended)
        contiguous = torch.empty((batch, length, heads, width // heads), device=device)
        invoke("permute_copy", attended, [0, 2, 1, 3], contiguous)
        states = contiguous.view(batch, length, width)
        # BGE exports GELU as Div / Erf / Add / Mul at opset 17.
        divided = torch.empty_like(states)
        invoke("div", states, torch.tensor(math.sqrt(2.0), device=device), divided)
        erf = torch.empty_like(states)
        invoke("erf", divided, erf)
        shifted = torch.empty_like(states)
        invoke("add", erf, one.expand_as(erf), 1.0, shifted, implementation=0)
        product = torch.empty_like(states)
        invoke("mul", states, shifted, product)
        activated = torch.empty_like(states)
        invoke("mul", product, torch.tensor(0.5, device=device), activated)
        cls = torch.empty((batch, width), device=device)
        invoke("onnx_gather", activated, torch.tensor(0, device=device), 1, cls)
        hidden = torch.empty_like(cls)
        invoke("tanh", cls, hidden)
        backing = torch.full((batch, 1, 2), -123.0, device=device)
        logits = backing[..., 0]
        address = logits.data_ptr()
        invoke("addmm", bias, hidden, classifier, 1.0, 1.0, logits)

        reference = torch.nn.functional.layer_norm(table[ids], [width], eps=1e-5)
        q, k, v = [
            (reference @ w).view(batch, length, heads, width // heads).transpose(1, 2)
            for w in weights
        ]
        ref_masked = (q @ k.transpose(-2, -1)) / scale + (
            1 - mask[:, None, None, :]
        ) * negative
        reference = (
            (ref_masked.softmax(-1) @ v)
            .transpose(1, 2)
            .contiguous()
            .view(batch, length, width)
        )
        reference = torch.nn.functional.gelu(reference)
        expected = reference[:, 0].tanh() @ classifier + bias
        torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-3)
        assert logits.data_ptr() == address
        assert torch.all(backing[..., 1] == -123.0)
        assert torch.isfinite(logits).all()
