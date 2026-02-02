import sys
import os
import torch
import numpy as np
from PIL import Image

# Ensure comfy is in path
sys.path.append(os.getcwd())
import comfy.utils

def legacy_full_lanczos(samples, width, height):
    images = [Image.fromarray(np.clip(255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(np.array(image)).float().div_(255.0).movedim(-1, 0) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)

def test_lanczos_correctness():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Random tensor in 0-1 range
    torch.manual_seed(42)
    tensor = torch.rand((2, 3, 64, 64), device=device)

    width, height = 32, 32

    res_legacy = legacy_full_lanczos(tensor, width, height)
    res_current = comfy.utils.lanczos(tensor, width, height)

    # Calculate difference
    # Note: Small floating point differences might occur due to operation order, but should be minimal.
    # Since we are converting to uint8 intermediate, the result is quantized.

    diff = (res_legacy - res_current).abs().max()
    print(f"Max difference: {diff}")

    # Tolerance: Since both convert to uint8 before resizing, they should be identical
    # unless the float->uint8 conversion logic differs at the boundary values.
    # My benchmark showed 0 diff, so I expect 0 or epsilon.
    assert diff < 1e-4, f"Difference too high: {diff}"

if __name__ == "__main__":
    test_lanczos_correctness()
    print("Test passed!")
