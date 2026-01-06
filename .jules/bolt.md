## 2024-03-24 - Efficient Attention Memory Optimization
**Learning:** In PyTorch, using `torch.cat([chunks])` creates a peak memory spike because both the list of chunks and the final concatenated tensor exist simultaneously. Pre-allocating the result tensor with `torch.empty` and filling it via slice assignment (`res[slice] = chunk`) avoids this overhead and reduces fragmentation.
**Action:** When refactoring chunked processing loops, always prefer pre-allocation over accumulation + concatenation.
