## 2026-01-22 - Caching in Tiled Operations
**Learning:** In high-iteration loops like tiled processing, even cheap allocations (like `torch.ones_like`) accumulate significantly. Caching reusable tensors (masks) based on shape provided a measurable speedup (approx 1.7x in synthetic benchmark) with zero impact on correctness.
**Action:** Look for similar patterns in other tiled operations (e.g. VAE decoding if tiled) where masks or buffers are re-allocated per tile.
