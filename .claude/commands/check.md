---
description: Run the repository integrity checks before claiming work is done
---
Run these checks in order and report each result in one line (pass or fail, with the failing detail):

1. `python -m pytest -q`
2. Em dash scan: `grep -rn $'—' --include='*.md' --include='*.py' --include='*.yaml' --include='*.cff' . | grep -v '^./.git/'` must print nothing.
3. AI attribution scan: `git log --format=%B | grep -inE 'co-authored-by|claude-session|generated with'` must print nothing, and the same pattern must not appear in tracked files other than CLAUDE.md and .claude/.
4. Leakage and determinism tests are included in step 1; confirm `test_no_test_leakage` and `test_sampling_is_deterministic_and_batch_invariant` passed.
5. `git status --short` and whether `data/external/` contains anything tracked (`git ls-files data/external` must list only `.gitkeep`).

If anything fails, stop and explain the cause before changing code. Do not fix failures by weakening tests.
