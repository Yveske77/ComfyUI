## 2025-02-18 - Vectorization of Image Feathering
**Learning:** Nested Python loops for pixel-wise image manipulation (e.g. feathering mask generation) are a massive performance bottleneck (~20x slower than vectorized Ops). PyTorch broadcasting and `torch.min` can replace complex distance logic efficiently.
**Action:** Always inspect image processing nodes for `for x in range(width): for y in range(height):` patterns and replace them with vectorized tensor operations, ensuring to use `device=tensor.device` for compatibility.
