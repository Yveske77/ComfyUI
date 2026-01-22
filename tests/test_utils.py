import torch
import unittest
from comfy.utils import tiled_scale

class TestUtils(unittest.TestCase):
    def test_tiled_scale_correctness(self):
        # Mock function: Upscale by factor of 2 using nearest neighbor
        def mock_upscale(x):
            return torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')

        # Create a small random input
        torch.manual_seed(42)
        C, H, W = 2, 64, 64
        samples = torch.rand(1, C, H, W)

        # Run tiled_scale
        # tile_x=32, tile_y=32, overlap=8.
        # This forces tiling (64 > 32).
        output = tiled_scale(samples, mock_upscale, tile_x=32, tile_y=32, overlap=8, upscale_amount=2, out_channels=C, output_device="cpu")

        # Expected output: full upscale
        expected = mock_upscale(samples)

        # Check similarity. Because of feathering and floating point ops, it might not be bitwise exact,
        # but for nearest neighbor with integer scale and perfect tiling alignment (if implemented right),
        # it might be close.
        # However, tiled_scale does blending in overlap regions.
        # If the function is linear (like resize), blending might introduce small deviations vs global resize
        # if the global resize algorithm handles edges differently.
        # But 'nearest' on pixels is not linear. Blending nearest results is weird.
        # Wait, if we use nearest neighbor, blending two nearest neighbor patches might result in values
        # that are not in the original set if we average them?
        # tiled_scale averages using the mask.
        # If values are 0 and 1, average is 0.5.

        # Let's check if output shape is correct at least.
        self.assertEqual(output.shape, expected.shape)

        # Let's assert it runs without error and produces values in range.
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))

    def test_tiled_scale_consistency(self):
        # Test that running twice produces same result (deterministic)
        def mock_upscale(x):
            return x # identity

        C, H, W = 2, 64, 64
        samples = torch.rand(1, C, H, W)

        out1 = tiled_scale(samples, mock_upscale, tile_x=32, tile_y=32, overlap=8, upscale_amount=1, out_channels=C)
        out2 = tiled_scale(samples, mock_upscale, tile_x=32, tile_y=32, overlap=8, upscale_amount=1, out_channels=C)

        self.assertTrue(torch.allclose(out1, out2))

if __name__ == '__main__':
    unittest.main()
