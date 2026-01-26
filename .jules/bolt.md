## 2025-02-19 - Tensor Allocation in Loops
**Learning:** Repeatedly allocating and feathering masks in `tiled_scale` loop causes significant overhead (approx 25% of runtime).
**Action:** Cache auxiliary tensors that depend only on tile parameters (size, overlap) to avoid re-allocation and re-computation.
