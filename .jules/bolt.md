## 2025-05-21 - [tiled_scale mask caching]
**Learning:** In tight loops with small tensor operations (like mask feathering), even simple allocations (`torch.ones_like`) and element-wise ops in Python can add up. Caching invariant tensors avoids this overhead.
**Action:** Look for repeated allocations of auxiliary tensors in loops, especially those that depend only on loop parameters or input shapes.
