# Session Log — 2026-06-24

## Summary

Repository initialization. Two commits, no substantive code.

## Commits

| Hash | Message |
|------|---------|
| `a074701` | Initial commit (README.md stub) |
| `d25dff6` | Create .gitignore |

## What Was Done

- Created the GitHub repository and made an initial commit with a one-line `README.md`.
- Added `.gitignore` covering Python bytecache (`__pycache__/`, `*.pyc`), virtual environments, outputs (`outputs/`, `*.ckpt`, `*.pt`), wandb cache, notebook checkpoints, and common editor/OS artifacts.

## What Was Not Done

No project structure, no source code, no config files, no dependencies. Everything beyond scaffolding was deferred to the next session.

## Open Questions at Session End

- Which CARD broadstreet version to use (settled on v4.0.1 in the next session).
- Whether to use pip or conda for dependency management (resolved: pip only, per CLAUDE.md).
- Model size tiers across environments (resolved in next session).
