import torch
import pytest
from comfy.utils import tiled_scale

def test_tiled_scale_correctness():
    C = 3
    H = 100
    W = 100
    # Use float32
    samples = torch.ones(1, C, H, W, dtype=torch.float32)

    def mock_model(x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")

    out = tiled_scale(samples, mock_model, tile_x=32, tile_y=32, overlap=8, upscale_amount=2, out_channels=C)

    # Check if all values are 1.0
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5)

def test_tiled_scale_caching_logic():
    # Test that caching handles different tile sizes (edges) correctly
    C = 3
    H = 64 # tile size 32, overlap 8.
    # Tile 1: 0-32.
    # Tile 2: 24-56.
    # Tile 3: 48-64 (size 16).
    W = 32

    samples = torch.randn(1, C, H, W)

    upscale = 2

    def mock_model(x):
        return torch.nn.functional.interpolate(x, scale_factor=upscale, mode="nearest")

    out = tiled_scale(samples, mock_model, tile_x=32, tile_y=32, overlap=8, upscale_amount=upscale, out_channels=C)

    assert out.shape == (1, C, H*upscale, W*upscale)
