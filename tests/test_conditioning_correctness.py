import unittest
import torch
import sys
from unittest.mock import MagicMock

# Mock necessary modules
sys.modules["comfy.model_management"] = MagicMock()
sys.modules["comfy.cli_args"] = MagicMock()
sys.modules["folder_paths"] = MagicMock()
sys.modules["latent_preview"] = MagicMock()

# Mock comfy submodules if needed
sys.modules["comfy.utils"] = MagicMock()
sys.modules["comfy.samplers"] = MagicMock()
sys.modules["comfy.sample"] = MagicMock()
sys.modules["comfy.sd"] = MagicMock()
sys.modules["comfy.controlnet"] = MagicMock()
sys.modules["comfy.clip_vision"] = MagicMock()
sys.modules["comfy.diffusers_load"] = MagicMock()

try:
    from nodes import ConditioningAverage
except ImportError:
    # If nodes cannot be imported directly, we might need to adjust sys.path
    import os
    sys.path.append(os.getcwd())
    from nodes import ConditioningAverage

class TestConditioningAverage(unittest.TestCase):
    def test_add_weighted_correctness(self):
        device = "cpu"
        dim = 768
        seq_len = 77
        batch_size = 1

        tensor1 = torch.randn(batch_size, seq_len, dim, device=device)
        pooled1 = torch.randn(batch_size, dim, device=device)
        cond1 = [[tensor1, {"pooled_output": pooled1}]]

        tensor2 = torch.randn(batch_size, seq_len, dim, device=device)
        pooled2 = torch.randn(batch_size, dim, device=device)
        cond2 = [[tensor2, {"pooled_output": pooled2}]]

        strength = 0.3

        # Original logic implementation for comparison
        # tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
        # t_to["pooled_output"] = torch.mul(pooled_output_to, conditioning_to_strength) + torch.mul(pooled_output_from, (1.0 - conditioning_to_strength))

        t0 = tensor2 # cond_from (cond2 passed as second arg)
        t1 = tensor1 # cond_to (cond1 passed as first arg)

        # Note: addWeighted(conditioning_to, conditioning_from, strength)
        # In implementation:
        # t1 comes from conditioning_to
        # t0 comes from conditioning_from

        expected_tw = torch.mul(t1, strength) + torch.mul(t0, (1.0 - strength))
        expected_pooled = torch.mul(pooled1, strength) + torch.mul(pooled2, (1.0 - strength))

        # Run actual node
        node = ConditioningAverage()
        result = node.addWeighted(cond1, cond2, strength)

        result_tw = result[0][0][0]
        result_pooled = result[0][0][1]["pooled_output"]

        # Check tolerance
        # Lerp might be slightly different than mul+add due to floating point precision,
        # but usually more accurate or very close.
        # Float32 epsilon is around 1e-7.

        # We use allclose
        self.assertTrue(torch.allclose(result_tw, expected_tw, atol=1e-5))
        self.assertTrue(torch.allclose(result_pooled, expected_pooled, atol=1e-5))

    def test_add_weighted_edge_cases(self):
        # Test strength 0.0 and 1.0
        device = "cpu"
        dim = 64
        seq_len = 10

        tensor1 = torch.randn(1, seq_len, dim)
        cond1 = [[tensor1, {}]]
        tensor2 = torch.randn(1, seq_len, dim)
        cond2 = [[tensor2, {}]]

        node = ConditioningAverage()

        # Strength 1.0 -> should be equal to cond1 (to)
        res_1 = node.addWeighted(cond1, cond2, 1.0)
        self.assertTrue(torch.allclose(res_1[0][0][0], tensor1))

        # Strength 0.0 -> should be equal to cond2 (from)
        res_0 = node.addWeighted(cond1, cond2, 0.0)
        self.assertTrue(torch.allclose(res_0[0][0][0], tensor2))

if __name__ == "__main__":
    unittest.main()
