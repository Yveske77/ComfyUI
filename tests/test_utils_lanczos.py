
import torch
import numpy as np
from PIL import Image
import comfy.utils
import pytest

def test_lanczos_conversion_correctness():
    device = torch.device("cpu") # Test on CPU for simplicity
    batch_size = 2
    channels = 3
    height = 64
    width = 64
    target_height = 32
    target_width = 32

    samples = torch.rand((batch_size, channels, height, width), device=device, dtype=torch.float32)

    # Run the function
    result = comfy.utils.lanczos(samples, target_width, target_height)

    # Check shape
    assert result.shape == (batch_size, channels, target_height, target_width)
    assert result.dtype == torch.float32

    # Replicate original logic to verify value correctness
    images_orig = [Image.fromarray(np.clip(255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images_orig = [image.resize((target_width, target_height), resample=Image.Resampling.LANCZOS) for image in images_orig]
    images_orig = [torch.from_numpy(np.array(image)).float().div_(255.0).movedim(-1, 0) for image in images_orig]
    expected_result = torch.stack(images_orig)

    # Compare
    # Note: there might be tiny differences if the implementation of lanczos inside comfy.utils changes slightly
    # (e.g. if I change how resize is called, which I'm not planning to).
    # But since I am only optimizing the pre-processing (Tensor->PIL), the result after resize might be slightly different
    # if the input to resize is slightly different (due to float vs uint8 conversion differences if any).
    # However, both paths convert to uint8 before resize.
    # The original path: float -> numpy float -> clip -> astype(uint8) -> PIL
    # Optimized path: float -> mul -> clamp -> to(uint8) -> numpy -> PIL
    # These should be identical for uint8 values.

    assert torch.allclose(result, expected_result, atol=1e-5)

if __name__ == "__main__":
    test_lanczos_conversion_correctness()
