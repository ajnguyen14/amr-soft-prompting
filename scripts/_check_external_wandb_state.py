"""Print the wandb state of the latest external-injection ablation run, or "" on failure.

Helper for scripts/check_memory_headroom.sh. Not a user-facing entry point.
"""

import wandb

ENTITY_PROJECT = "aidan-j-nguyen-san-jose-state-university/amr-soft-prompting"


def main() -> None:
    """Look up and print the latest external run's wandb state."""
    try:
        api = wandb.Api()
        runs = api.runs(
            ENTITY_PROJECT,
            filters={"config.model.injection_mode": "external"},
            order="-created_at",
        )
        print(runs[0].state if len(runs) else "")
    except Exception:
        print("")


if __name__ == "__main__":
    main()
