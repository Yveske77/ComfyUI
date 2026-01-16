# Bolt's Journal

## 2024-05-22 - [Optimized Attention Slicing]
**Learning:** In `sub_quadratic_attention.py`, using `torch.cat` inside a loop to build the result tensor causes high peak memory usage and fragmentation.
**Action:** Pre-allocate the result tensor with `torch.empty` and use slice assignment (e.g., `res[:, i:j] = ...`). This avoids intermediate allocations and copies.

## 2024-05-22 - [Dead Code in Grid Calculation]
**Learning:** In `web/scripts/app.js`, the `calculateGrid` function had a `while` loop with a condition `columns * rows < n` that was mathematically unreachable because `columns` and `rows` were initialized using `Math.ceil(Math.sqrt(n))`.
**Action:** Remove the dead loop to clean up the code, although the performance impact is negligible since it never ran.

## 2024-05-22 - [Optimized Batch Resizing]
**Learning:** `torch.nn.functional.interpolate` can be slower than manual nearest-neighbor interpolation when exact alignment isn't strictly required or when dealing with specific tensor layouts.
**Action:** Implemented `resize_to_batch_size` in `comfy/utils.py` using `torch.linspace` for downsampling and `torch.arange` for upsampling. This achieved ~2.2x-2.7x speedup.

## 2024-05-22 - [Efficient Image Loading]
**Learning:** Loading images directly into float32 tensors can be memory intensive.
**Action:** Load as `uint8` numpy array, convert to tensor, and then normalize. This avoids large intermediate float32 numpy arrays.
