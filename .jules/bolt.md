## 2024-05-23 - [ComfyUI Testing and Optimization]
**Learning:** Testing `nodes.py` in isolation requires mocking `comfy.cli_args` *before* imports to prevent `comfy.model_management` from failing CUDA initialization on CPU-only environments.
**Action:** When writing unit tests for ComfyUI nodes, use `unittest.mock` to set `comfy.cli_args.args.cpu = True` (and other defaults) before importing the node module.

**Learning:** `ImagePadForOutpaint` contained an O(N*M) Python loop for feathering which was a major bottleneck (2.4s for 1024x1024). Vectorization reduced it to ~0.01s.
**Action:** Inspect other image processing nodes for similar nested loop patterns involving pixel-wise operations.
