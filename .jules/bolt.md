## 2024-05-22 - PyTorch to Numpy/PIL Conversion Optimization
**Learning:** When converting PyTorch tensors to PIL images (e.g. for `lanczos` resizing), performing scaling (`* 255`), clamping, and casting to `uint8` (`.byte()`) on the source device (GPU) *before* moving to CPU/Numpy significantly reduces data transfer overhead and CPU usage compared to moving float32 data and doing the math in Numpy.
**Action:** Look for `cpu().numpy()` calls followed by math and casting. Move the math and casting to PyTorch before the `.cpu()` call.
