# BirdCLEF 2026 — Claude Code Guide

## Experimentation Workflow

Each experiment run follows this pattern:

### 1. Code change
Make a focused, single-purpose change to the experiment (e.g. unfreeze backbone, add label smoothing, change lr). Commit and push to `main`.

### 2. Run via `wm`
```bash
wm run exp_001
```
The `wm` framework automatically creates a snapshot branch and opens a PR on GitHub, capturing the exact code state for this run.

### 3. Post a PR comment
Immediately comment on the new PR with:
- W&B run link
- What changed from the previous run and why
- Full config table

### 4. Monitor the run
Poll metrics periodically using the W&B Python API:
```python
import wandb
api = wandb.Api()
run = api.run("josh-gree/birdclef-2026-take-2/<run_id>")
history = run.history(keys=["epoch_train_loss", "epoch_train_acc", "epoch_val_loss", "epoch_val_acc", "epoch"])
```

### 5. Kill or let complete
Kill early if the trajectory is clear (plateauing, clearly not going to improve, or overfitting). No need to run to completion if the outcome is obvious.

### 6. Post a summary comment on the PR
Once killed or complete, pull final metrics from the W&B API and post a summary comment containing:
- Full epoch-by-epoch results table
- Random chance baseline (1/num_classes) for context
- Verdict: did it converge? did it overfit? what was the ceiling?
- What it means for next steps

### 7. Update the plan doc
Record the run in the relevant plan doc in `/Users/josh-gree/vault/Projects/birdclef_2026/resources/`. Include:
- PR link and W&B link
- Key metrics (final val acc/loss, epochs run)
- One-line verdict

---

## Key Facts

- **Random chance baseline:** 1/234 ≈ 0.43% (234 classes in taxonomy)
- **Data:** processed memmap on Modal volume `birdclef-2026-processed`, mounted at `/data` — always copy `train.npy` to `/tmp` at job startup before training
- **Checkpoints:** saved per epoch to `run_dir/checkpoints/epoch_NNN.pt`, persisted to Modal volume `birdclef-2026-take-2-storage` under `exp_001/<wandb_run_id>/`
- **W&B project:** `josh-gree/birdclef-2026-take-2`
