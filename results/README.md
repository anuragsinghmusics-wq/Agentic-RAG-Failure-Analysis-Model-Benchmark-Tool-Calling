# Published run artifacts

This directory contains the curated, inspectable outputs from `run1`:

- `RESULTS.md` — generated report with aggregate metrics, failure taxonomy, and paired bootstrap comparisons.
- `results_summary.csv` — one row per dataset/model/condition.
- `failure_taxonomy.csv` — primary agentic failure shares.
- `figures/` — condition, failure, generalization, and QLoRA loss plots.
- `traces/` — JSONL traces used to calculate the published metrics.
- `logs/train_3B-ft.json` — optimizer-step loss history for the main adapter.

Trace files preserve the questions, model actions, search queries, retrieved titles, final answers, metrics, and conversation messages. They are useful for auditing individual failures without rerunning the GPU experiment.

The raw Kaggle bundle also contains embedding caches, SFT data, and the adapter checkpoint. Those generated intermediates are not duplicated here; the checkpoint policy is documented in [`../artifacts/README.md`](../artifacts/README.md).
