---
description: Run the repository integrity checks before claiming work is done
---
Run these checks in order and report each result in one line (pass or fail, with the failing detail):

1. `python -m pytest -q`
2. `python scripts/check_style.py` must report that the style check passed (no em dashes, no AI attribution in files or commit messages).
3. Confirm the manuscript or docs you edited read as formal research prose.
4. Leakage and determinism tests are included in step 1; confirm `test_no_test_leakage` and `test_sampling_is_deterministic_and_batch_invariant` passed.
5. `git status --short` and whether `data/external/` contains anything tracked (`git ls-files data/external` must list only `.gitkeep`).

If anything fails, stop and explain the cause before changing code. Do not fix failures by weakening tests.
