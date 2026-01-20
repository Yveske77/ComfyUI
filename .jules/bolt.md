## 2025-02-18 - Tiled Scale Mask Caching
**Learning:** `tiled_scale` in `comfy/utils.py` was re-allocating and re-computing feathering masks for every tile, causing significant overhead (especially allocation churn) when processing large images with many tiles.
**Action:** Implemented caching for feathering masks based on tile dimensions and leveraged broadcasting with single-channel masks to reduce memory usage and computation.
