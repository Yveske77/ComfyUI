
import unittest
import torch
from comfy.utils import resize_to_batch_size

class TestResizeToBatchSize(unittest.TestCase):
    def test_same_size(self):
        tensor = torch.randn(10, 4, 32, 32)
        out = resize_to_batch_size(tensor, 10)
        self.assertTrue(torch.equal(tensor, out))

    def test_downsample(self):
        # 10 -> 5
        tensor = torch.randn(10, 4, 32, 32)
        out = resize_to_batch_size(tensor, 5)
        self.assertEqual(out.shape[0], 5)
        self.assertEqual(out.shape[1:], tensor.shape[1:])
        # Verify first and last elements logic
        # scale = (10-1)/(5-1) = 9/4 = 2.25
        # i=0 -> 0
        # i=4 -> min(round(4*2.25), 9) = min(9,9) = 9
        self.assertTrue(torch.equal(out[0], tensor[0]))
        self.assertTrue(torch.equal(out[-1], tensor[-1]))

    def test_upsample(self):
        # 5 -> 10
        tensor = torch.randn(5, 4, 32, 32)
        out = resize_to_batch_size(tensor, 10)
        self.assertEqual(out.shape[0], 10)
        # scale = 5/10 = 0.5
        # i=0 -> floor(0.5*0.5) = 0
        # i=9 -> floor(9.5*0.5) = floor(4.75) = 4
        self.assertTrue(torch.equal(out[0], tensor[0]))
        self.assertTrue(torch.equal(out[-1], tensor[-1]))

    def test_batch_one(self):
        tensor = torch.randn(5, 4, 32, 32)
        out = resize_to_batch_size(tensor, 1)
        self.assertEqual(out.shape[0], 1)
        self.assertTrue(torch.equal(out[0], tensor[0]))

    def test_from_one(self):
        tensor = torch.randn(1, 4, 32, 32)
        out = resize_to_batch_size(tensor, 5)
        self.assertEqual(out.shape[0], 5)
        for i in range(5):
            self.assertTrue(torch.equal(out[i], tensor[0]))

if __name__ == '__main__':
    unittest.main()
