## 2025-02-14 - Tiled Scale Optimization
**Learning:** `tiled_scale` recomputed masks for every tile, but tile shapes are highly repetitive.
**Action:** Use function-local dictionary caching for generated tensors that depend only on tile shape.
