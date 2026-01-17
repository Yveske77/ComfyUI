
import unittest
import torch
import math
import sys
import os

# Add the parent directory to sys.path to import comfy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfy.utils import resize_to_batch_size

def original_resize_to_batch_size_ref(tensor, batch_size):
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
    def setUp(self):
        self.device = torch.device("cpu")

    def test_resize_to_batch_size_downsample(self):
        tensor = torch.randn(10, 4, 4, device=self.device)
        batch_size = 5

        expected = original_resize_to_batch_size_ref(tensor, batch_size)
        result = resize_to_batch_size(tensor, batch_size)

        self.assertTrue(torch.allclose(result, expected), "Downsampling result does not match reference")

    def test_resize_to_batch_size_upsample(self):
        tensor = torch.randn(4, 4, 4, device=self.device)
        batch_size = 10

        expected = original_resize_to_batch_size_ref(tensor, batch_size)
        result = resize_to_batch_size(tensor, batch_size)

        self.assertTrue(torch.allclose(result, expected), "Upsampling result does not match reference")

    def test_resize_to_batch_size_equal(self):
        tensor = torch.randn(5, 4, 4, device=self.device)
        batch_size = 5
        result = resize_to_batch_size(tensor, batch_size)
        self.assertTrue(torch.allclose(result, tensor), "Equal batch size should return same tensor")

    def test_resize_to_batch_size_one(self):
        tensor = torch.randn(5, 4, 4, device=self.device)
        batch_size = 1
        result = resize_to_batch_size(tensor, batch_size)
        self.assertEqual(result.shape[0], 1)
        self.assertTrue(torch.allclose(result[0], tensor[0]), "Batch size 1 should return first element")

    def test_resize_to_batch_size_large(self):
        # Performance test / correctness check for larger tensors
        tensor = torch.randn(100, 16, 16, device=self.device)
        batch_size = 50

        expected = original_resize_to_batch_size_ref(tensor, batch_size)
        result = resize_to_batch_size(tensor, batch_size)

        self.assertTrue(torch.allclose(result, expected))

if __name__ == '__main__':
    unittest.main()
