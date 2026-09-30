---
description: Record finished work in the WORKLOG and package it for the other Claude instance
argument-hint: [short summary of what was done]
---
Hand off the current work: $ARGUMENTS

1. Run the /check steps. Do not hand off with failing checks unless the WORKLOG entry says so explicitly.
2. Append a dated entry to `docs/WORKLOG.md` (label it `(local)`): what changed, commands run, exact numeric results with their source file, and the next step. Update the "Open tasks" checkboxes and claims.
3. Commit with a plain, descriptive message. Author Rohit Bharti. No AI attribution lines.
4. Create the bundle: `git bundle create ../handoff.bundle main` and print its path and `git log --oneline -5`.
5. Tell Rohit in two or three lines what to pass on and what the other instance should do next.
