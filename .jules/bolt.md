## 2024-05-23 - Optimizing Tensor to PIL Conversion
**Learning:** Converting PyTorch tensors to PIL images by first moving `float32` data to CPU/NumPy is inefficient. Performing scaling, clamping, and casting to `uint8` on the device (GPU) before moving to CPU significantly reduces data transfer (4x reduction in size) and utilizes device parallelism.
**Action:** Always perform data type reduction on the source device before transferring to CPU for IO operations.
