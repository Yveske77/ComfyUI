
import torch
import unittest
from comfy.utils import tiled_scale

class TestUtils(unittest.TestCase):
    def test_tiled_scale(self):
        device = torch.device("cpu")
        samples = torch.ones(1, 3, 64, 64, device=device) # small image

        # Identity upscale
        def identity(x):
            return x

        # upscale_amount=1, so output should be same as input
        output = tiled_scale(samples, identity, tile_x=32, tile_y=32, overlap=8, upscale_amount=1, output_device=device)

        # Check shape
        self.assertEqual(output.shape, samples.shape)

        # Check values (should be all ones)
        self.assertTrue(torch.allclose(output, samples))

    def test_tiled_scale_upscale(self):
        device = torch.device("cpu")
        samples = torch.ones(1, 3, 32, 32, device=device)

        upscale_amount = 2
        def upscale(x):
            return torch.nn.functional.interpolate(x, scale_factor=upscale_amount, mode='nearest')

        output = tiled_scale(samples, upscale, tile_x=16, tile_y=16, overlap=4, upscale_amount=upscale_amount, output_device=device)

        self.assertEqual(output.shape, (1, 3, 64, 64))
        self.assertTrue(torch.allclose(output, torch.ones_like(output)))

if __name__ == '__main__':
    unittest.main()
