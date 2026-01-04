## 2024-05-23 - Sub-quadratic Attention Optimization
**Learning:** Pre-allocating result tensors with `torch.empty` and using slice assignment is significantly more memory-efficient than accumulating results in a list and using `torch.cat`. `torch.cat` requires keeping all intermediate tensors in memory plus the final result, whereas slice assignment only requires one intermediate chunk + the result.
**Action:** Always prefer `torch.empty` + slice assignment over `torch.cat` when the final tensor size is known and we are iterating over chunks.
