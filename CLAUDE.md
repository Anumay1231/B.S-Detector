# CLAUDE.md

Project-specific instructions for Claude when working in this repository.

## Project

Speaker Verification module for the B.S. Detector project.

Repository: https://github.com/Anumay1231/B.S-Detector.git
Branch: main

## GIT WORKFLOW

1. Work only on the current `main` branch unless explicitly asked to
   create a feature branch.
2. Before making major changes, inspect `git status`.
3. Never use force push.
4. Never rewrite existing Git history unless explicitly asked.
5. Never commit secrets, API keys, `.env` files, virtual environments,
   model caches, datasets, audio files, generated outputs, or other files
   excluded by `.gitignore`.
6. After completing each requested development phase:
   - run the relevant tests
   - inspect `git diff`
   - inspect `git status`
   - commit the completed changes with a clear, descriptive commit message
   - push the commit to `origin/main`
7. Only push when tests for that phase pass.
8. If tests fail, do NOT commit or push the broken implementation. Report
   the failure instead.
9. After pushing, verify:
   ```
   git status
   git log --oneline -n 3
   ```
10. Never claim a push succeeded unless git actually reports successful
    push output.
11. If GitHub authentication fails, STOP and report it rather than trying
    to bypass authentication.
12. Keep commits logically separated by completed phases.

### Recommended commit message format

```
Phase 4: Add ECAPA-TDNN speaker encoder
Phase 5: Add speaker similarity verification
Phase 6: Add trial generation and calibration
Phase 7: Add verification thresholds
```

### Push cadence

Do not automatically push every small file edit. Push after a complete
requested phase or other meaningful milestone.
