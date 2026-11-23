# compile test folder structure

This inherited test layout describes compiler coverage and upstream CI selection.
Nabla's H20 host-planning record did not enable full-model `torch.compile`.
Run the relevant target-model path before making compiler or CI coverage claims.

- `compile/test_*.py` : various unit tests meant for testing particular code path/features. Future tests are most likely added here. New test files added here will be included in CI automatically
- `compile/fullgraph/` : full model tests, including all tests previously in compile/piecewise. These tests do not target particular features. New test files added here will be included in CI automatically
- `compile/distributed/` : tests that require multiple GPUs. New test files added here will **NOT** be included in CI automatically as these tests generally need to be manually configured to run in runners with particular number/type of GPUs.
