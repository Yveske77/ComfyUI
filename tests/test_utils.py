
import torch
import numpy as np
from PIL import Image
import comfy.utils
import pytest

def lanczos_ref(samples, width, height):
    # Reference implementation (original)
    images = [Image.fromarray(np.clip(255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(np.array(image)).float().div_(255.0).movedim(-1, 0) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)

def test_lanczos_correctness():
    device = torch.device("cpu") # Test on CPU for consistency
    N, C, H, W = 2, 3, 64, 64
    target_W, target_H = 32, 32

    # Use deterministic random numbers
    torch.manual_seed(42)
    samples = torch.rand((N, C, H, W), device=device, dtype=torch.float32)

    # Expected result using reference implementation
    expected = lanczos_ref(samples, target_W, target_H)

    # Actual result using comfy.utils.lanczos (which will be optimized)
    actual = comfy.utils.lanczos(samples, target_W, target_H)

    # Check shape
    assert actual.shape == (N, C, target_H, target_W)

    # Check values. Since we are going through uint8 conversion, there might be
    # extremely minor differences if the float math on GPU/PyTorch differs
    # slightly from CPU/Numpy, but generally should be identical.
    # However, since I haven't applied the optimization yet, this test should pass exactly.
    # After optimization, we allow a small tolerance if necessary, but expect exact match.

    # Using allclose with a small tolerance just in case.
    # 1/255 is approx 0.0039.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_lanczos_correctness()
