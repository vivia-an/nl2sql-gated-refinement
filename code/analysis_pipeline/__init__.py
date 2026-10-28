"""S1 end-to-end results pipeline: runs/<exp>/<run_id>/preds.jsonl -> results/*.json -> paper/results_macros.tex, tables, figures.

Order of commands: docs/ANALYSIS_RUNBOOK.md.  Host Python 3.6 can run make_macros / make_tables only (stdlib);
everything else needs numpy (+ scipy for the GLMM, matplotlib for figures) and runs on a worker machine venv."""
