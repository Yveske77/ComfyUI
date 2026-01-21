## 2024-05-23 - Broadcasting Overhead in Cached Tensors
**Learning:** Caching a 1-channel mask to save memory caused a regression on CPU because `ps * mask` (3-channel * 1-channel) forced broadcasting at every step. Caching the full 3-channel mask (matching `ps` shape) restored performance while still avoiding re-computation of the mask content.
**Action:** When caching tensors to be used in frequent arithmetic operations, match the target shape to avoid repeated broadcasting costs, unless memory is extremely tight.
