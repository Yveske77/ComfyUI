## 2024-03-24 - [Avoid Python loops for tensor construction]
**Learning:** Using Python loops (e.g., `for i in range(batch_size): output[i] = ...`) to construct or resize tensors is significantly slower than vectorized operations, even for small batch sizes (approx 2x slower for B=120).
**Action:** Replace iterative tensor population with `torch.linspace`, `torch.arange`, or `index_select` to leverage optimized C++ implementations.
