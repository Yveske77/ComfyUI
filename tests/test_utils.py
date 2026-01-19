
import torch
import unittest
import math
import comfy.utils

def resize_to_batch_size_reference(tensor, batch_size):
    in_batch_size = tensor.shape[0]
    if in_batch_size == batch_size:
        return tensor

    if batch_size <= 1:
        return tensor[:batch_size]

    output = torch.empty([batch_size] + list(tensor.shape)[1:], dtype=tensor.dtype, device=tensor.device)
    if batch_size < in_batch_size:
        scale = (in_batch_size - 1) / (batch_size - 1)
        for i in range(batch_size):
            output[i] = tensor[min(round(i * scale), in_batch_size - 1)]
    else:
        scale = in_batch_size / batch_size
        for i in range(batch_size):
            output[i] = tensor[min(math.floor((i + 0.5) * scale), in_batch_size - 1)]

    return output

class TestUtils(unittest.TestCase):
    def test_resize_to_batch_size(self):
        device = "cpu"
        shapes = [
            ((1, 4, 64, 64), 4),
            ((4, 4, 64, 64), 16),
            ((16, 4, 64, 64), 4),
            ((8, 1280, 8, 8), 16),
            ((1, 3, 512, 512), 2),
        ]

        for shape, target_batch in shapes:
            tensor = torch.randn(shape, device=device)
            expected = resize_to_batch_size_reference(tensor, target_batch)
            actual = comfy.utils.resize_to_batch_size(tensor, target_batch)

            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(torch.allclose(actual, expected), f"Mismatch for shape {shape} -> {target_batch}")

if __name__ == "__main__":
    unittest.main()
