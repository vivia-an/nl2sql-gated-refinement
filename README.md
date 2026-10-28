# Execution-gated refinement for text-to-SQL: code, analysis plan and result files

Code, frozen configurations, the statistical analysis plan with its dated amendments, result files and prompts for

> Li, S.; Song, X.; Wang, D. *What Does Execution-Gated Refinement Buy in Text-to-SQL? Cost, Accuracy Across Backbones, and Negative Results.* Applied Sciences (MDPI), <PAPER_DOI>.

Code archive (this repository, release v1.0.0): <ZENODO_CODE_DOI>. Run archive (stored model requests and responses, predictions): <ZENODO_ARCHIVE_DOI>. Source repository: <GITHUB_URL>.

The paper is an evaluation study. It asks what a generate-and-critique loop adds to a text-to-SQL system built on a frozen, API-hosted language model: what it costs, what it buys on backbones of different price, and which parts matter. It reports the tests that were pre-specified, the post hoc analyses (marked as such) and the components that did not help. This repository is not a software product; it exists so that the numbers in the paper can be traced to stored result files and re-derived.

Hardware: none beyond an ordinary computer. No GPU is used anywhere; all language-model calls go to a hosted API.

## What is in this repository

```
code/nl2sql/             the pipeline (modules M1-M7), baselines, execution and scoring, LLM client with request cache and usage log
code/analysis_pipeline/  the statistics: database-clustered bootstrap, Holm adjustment, McNemar, mixed model, routing metrics
code/prompts/            every prompt, verbatim
code/tools/              generators for the LaTeX number macros and tables, cost report, cache archive tool, post hoc analyses
code/tests/              unit tests (run without an API key and without benchmark data)
exp/configs/             configurations; exp/configs/frozen/ are the frozen ones (one YAML per system or arm); exp/frozen/ coefficient and store files
exp/dev_lock.json        hash lock of the configurations used on the development set; exp/models.yaml, exp/prices.yaml model ids and list prices
exp/splits/              tuning/hold-out split of the BIRD training databases (question and database identifiers only)
exp/queues/              the job lists that were executed (one JSON object per run: command, model, estimated calls)
docs/SAP.md             statistical analysis plan, amendments and the post hoc note; docs/EXPERIMENT_SPEC.md, TUNING.md, ...
results/                 flat result files (results/*.json, one per analysis stage) and the post hoc tables (results/posthoc2/*.csv)
paper/                   the number macros and generated tables of the article (results_macros*.tex, tables/, supp_tables/); no manuscript text
figs/                    the figures of the article and the data files they are drawn from
```

## What is not in this repository

- The manuscript and supplement sources (they are published by the journal). `paper/` holds only what the code generates: the number macros and the tables.
- Benchmark data. BIRD, Spider and the corrected-reference releases are not redistributed; download them from their sources. `DATA_LICENCES.md` lists the licence of each as stated upstream, what is kept here (identifiers and digests, and in `results/diag/` SQL text of BIRD training questions under CC BY-SA 4.0), and the attribution and modification notice.
- The archived model requests and responses and the per-question predictions of every run. They are deposited separately at <ZENODO_ARCHIVE_DOI> (see Level 2).
- API keys and any machine addresses. The scripts that drove the compute machines are not part of the release.
- The MDPI LaTeX class files.
- One baseline file (`macsql_const.py`, upstream has no licence file): `python code/tools/fetch_macsql_const.py` downloads it from the pinned upstream commit and checks its sha256.

## Reproduce the numbers

There are three levels. Each level needs more than the one before.

### Level 1: tables and number macros from the stored result files (no API key, no download)

Needs Python 3.6 or newer, standard library only.

```bash
python3 code/tools/make_tables.py                  # paper/tables/tab_{primary,e1,ablation,backbone,routing,pollution,taxonomy,rootstatus}.tex
python3 code/tools/make_macros.py --allow-missing  # paper/results_macros.tex   (\res{KEY}: confirmatory numbers)
python3 code/tools/make_macros_ts.py               # paper/results_macros_ts.tex (\tsres{KEY}: held-out tuning numbers)
python3 code/tools/make_xmacros.py --allow-missing # paper/results_macros_x.tex  (\xres{KEY}: post hoc and robustness numbers)
git status --short paper/                          # expected: only the "generated ..." header line of each macro file differs
```

Every number in the article is a `\res{KEY}`, `\tsres{KEY}` or `\xres{KEY}` reference into these macro files; no number is typed by hand. `\res` keys come from the confirmatory result files `results/*.json`, `\xres` keys from the post hoc and robustness files (`results/posthoc2`, `results/robust`, `results/ext1`, `results/diag`, `results/pf`, `results/r3`, `results/r4`, `results/r4_t0`), `\tsres` keys from the held-out tuning data (`results/ts`). `paper/res_keys_used.txt` lists the `\res` keys that the article references. The supplement's per-contrast tables are in `results/posthoc2/*.csv`.

### Level 2: result files from the stored predictions (no API key; needs the run archive)

The result files are computed from per-question predictions (`preds.jsonl`, one JSON object per question) and the archived model requests and responses of each run. The run archive is at <ZENODO_ARCHIVE_DOI> (`MANIFEST.tsv` lists every file with its sha256, `RUNS.tsv` every run). Unpack it so that the runs are at `<WORK_DIR>/runs/<experiment>/<run>/`:

```bash
tar xzf nl2sql_runs_1.0.0.tar.gz -C <WORK_DIR>             # creates <WORK_DIR>/runs/ and <WORK_DIR>/frozen/
sha256sum -c MANIFEST.sha256                                # optional: check the files
pip install -r code/requirements.txt                        # Python 3.12; CPU only; the versions are pinned, see below
cd code
python tools/cache_export.py verify <WORK_DIR>/runs         # request keys re-derived from the stored requests, archive hashes
python -m analysis_pipeline.run_all --runs <WORK_DIR>/runs --results <OUT_DIR> --B 10000 \
    --e4-roots <WORK_DIR>/runs/MIN/E4 --e9-dirs <WORK_DIR>/runs/MIN/E9/dev-qwen37plus <WORK_DIR>/runs/MIN/E9/dev-qwen37flash <WORK_DIR>/runs/MIN/E9/dev-qwen3527b <WORK_DIR>/runs/MIN/E9/pool-qwen37plus \
    --annot-dir ../results/e10 --decisions ../exp/selection_decisions_k4.json --route-coef ../exp/frozen/route_coef_hr.json --route-k 4 --tuning-results ../results
python tools/r4_t0.py --runs <WORK_DIR>/runs --out <OUT_DIR>/r4_t0 --B 10000        # temperature-0 re-runs (R4T0)
python tools/r3_seeds.py --runs <WORK_DIR>/runs --out <OUT_DIR>/r3 --B 10000         # seed and control re-runs (R3); r3_*.py: see docs/ANALYSIS_RUNBOOK.md
```

Which runs take part. The temperature-0 re-runs (`R4T0`) repeat the configuration of the headline runs, and the seed and control re-runs (`R3`) are other runs of the same systems. `run_all` therefore does not scan the archive and pick "the newest run of a configuration"; it analyses exactly the 91 run directories listed in `exp/replay_manifest.json` (`runs`: directory relative to `<WORK_DIR>/runs` and run id; `roles`: the five runs behind `results/primary.json`, which `tests/test_replay_manifest.py` checks against the released result file). If one of the listed directories is missing it stops with an error. `--replay-manifest none` analyses everything that is found (needed only for your own runs). `R3` and `R4T0` are read by their own entry points (`tools/r3_*.py`, `tools/r4_t0.py`), which name their run directories themselves.

What to expect. The replay recomputes every file of `results/` that the stages above write. With the pinned versions (`code/requirements.txt`: Python 3.12, numpy 1.26.4, scipy 1.15.2) Exact-reproduction scope, as measured on a CPU-only machine with numpy 1.26.4, scipy 1.15.2 and B = 10000. Level 1 is identical: the tables are byte-identical and the macro files differ only in the "generated" header line. Level 2: 6 317 of 6 390 values (17 result files) are identical to a relative 1e-9. The other 73 values are of three kinds. (a) 36 mixed-model values of the pollution experiment (`E4.*`, `P4.*` of `primary.json`, `P4.p_adj_se` of `Psec.json`) differ by up to 1.6e-2: the optimiser works on a null contrast and the result depends on the CPU and the library builds (on the authors' worker machine they reproduce exactly). (b) 35 interval ends of the bootstrap in `E10.json` differ by up to 0.7 percentage points: the order in which items are resampled depends on the string-hash seed of the Python process (`analysis_pipeline/e10.py` iterates over a set); the point estimates are identical. (c) Two provenance fields: `setup.devlock.hash` (the lock file grew after the analysis, entries of the third round were added) and `setup.plan.commit` (needs the git history of the authors' working repository). No verdict of a confirmatory test changes. The seed and control re-runs (R3) and the temperature-0 re-runs (R4T0) are identical (one value differs by 3e-11; the released `rv4t0_summary.json` additionally holds cost keys merged from separate cost files). Fields beginning with `_` (creation time, `git`, `sap_sha256`, the lock hash `setup.devlock.hash`, the machine paths) are provenance and are expected to differ. `setup.plan.commit` is empty without the git history of the authors' working repository (`--plan-commit` takes the hash).

`docs/ANALYSIS_RUNBOOK.md` lists the stages, the run roles and the scope guards (a tuning-set run cannot overwrite a confirmatory result file). The bootstrap seed is fixed (20261101).

### Level 3: re-run an experiment (needs an API key and costs money)

Everything that calls a language model needs an account at the model provider (the OpenAI-compatible endpoint named in `exp/models.yaml`) and the benchmark data under `NL2SQL_DATA` (`$NL2SQL_DATA/bird/dev/dev_20240627/dev.json`, `$NL2SQL_DATA/bird/train/train/train.json`, databases next to them). The key is never taken from a command line. It is read from the file named by the environment variable `NL2SQL_KEY_FILE` (default `~/.nl2sql/qwen.env`), which holds one line `QWEN_API_KEY=<key>`; if that file does not exist, from the environment variable `QWEN_API_KEY` (or `DASHSCOPE_API_KEY`).

```bash
export NL2SQL_ROOT=<WORK_DIR> NL2SQL_DATA=<BENCHMARK_DIR> NL2SQL_KEY_FILE=<PATH_TO_KEY_FILE>
cd code
python -m nl2sql.run --system full_gated --split bird_train_HT --config ../exp/configs/full_gated.yaml --model <model key from exp/models.yaml> --exp demo --limit 20
```

Expected cost: `exp/prices.yaml` holds the list prices (CNY per million tokens) on the day of the runs; `python code/tools/cost_report.py <RUNS_DIR>` turns a usage log into a cost. On the 1 534 BIRD development questions with the Plus backbone one arm cost about CNY 3.5 (single call), 10.6 (gated loop) and 17.4 (fixed loop), see `results/E1.json` (`*.cost_total.mean`); the seed and control re-runs of the third round cost about CNY 136 (`results/r3/cost.json`) and the temperature-0 re-runs about CNY 45 (`results/r4_t0/cost.json`). A 20-question demo costs a few CNY-cents to a few CNY.

The configurations of the full pipeline point to the frozen experience store `exp/frozen/corpus_burnin_39e2477c.sqlite` (36 MB, sha256 `39e2477c853c0738fa139e42c7fe700afb880966eaa35ac740eeed5cb25687b5`; built from BIRD training questions, hence not in git). It is in the run archive under `frozen/`; copy it to `exp/frozen/` before any run with a locked configuration.

The development set is only reachable through the lock (`exp/dev_lock.json`, `nl2sql.lock`): a run with an unregistered or edited configuration is refused. Model snapshot ids and aliases can be withdrawn by the provider; that is why the run archive exists. Re-runs will not reproduce the stored responses exactly: identical requests at temperature 0 did not return identical answers (paper, Data Availability and Supplement S17).

### Tests

```bash
cd code && python -m pytest tests -q        # no API key, no benchmark data; tests that need them are skipped
```

## Which paper statement rests on what

- Pre-specified tests, their decision rules and the verdict words: `docs/SAP.md` (Sections 13 to 15 and the amendment log).
- Post hoc analyses are labelled as such in the paper and in `docs/SAP.md` (Section 25 for the small-cluster readings). They are in `results/posthoc2/` and `results/robust/`.
- Annotation of failed answers is language-model labelling (two independent runs and an adjudication run), described in the paper and in `docs/ANNOTATION_GUIDE.md`; it is not human annotation.

## A note on paths and on files that keep their bytes

Provenance fields in some result files recorded absolute paths of the machines the runs were made on. In this release those paths are replaced by placeholders such as `<NL2SQL_ROOT>`; the sha256 values that other released files record for the changed files were updated accordingly.

Some files are byte-identical to the versions that were used, because their sha256 values are checked by the lock (`exp/dev_lock.json`, `nl2sql.lock`) or recorded in the metadata of every run (`run_meta.json` in the run archive): the lock file itself, `exp/configs/`, `code/prompts/`, `exp/diag_prompts/`, `exp/pf_prompts/`, `exp/models.yaml` and `exp/frozen/route_coef_hr.json`. A few of them contain remarks of the working environment that do not matter for the analysis: comment lines that refer to working notes that are not part of this release (`exp/configs/`), a key-file path and a remark on the machines in `exp/models.yaml` (the key file is chosen by `NL2SQL_KEY_FILE`, see Level 3), and the location of two input files in `exp/frozen/route_coef_hr.json`. Editing them would change the hashes and the lock would refuse the configurations.

`docs/SAP.md`, `docs/ANNOTATION_GUIDE.md` and `docs/ANNOTATION_PROMPT.md` are reading copies. They were generated from the working versions with the references to working notes, review rounds and machines removed (a paragraph at the top of `docs/SAP.md` says what was edited; sections, numbers, dates, scope labels, hashes and the two annotation prompts are unchanged). The sha256 values that result files record for these documents (`sap_sha256` in `results/primary.json` and the other files written by the primary stage, `files_sha256` and `frozen_texts_sha256` in `results/e10/annotation_runs.json`) refer to the working versions, which are not released:

| file | sha256 of the working version (recorded in result files) | sha256 of the reading copy in this repository |
|---|---|---|
| `docs/ANNOTATION_GUIDE.md` | `f9d427869d73a828637e1887234d17b61962695f4dc5ffb270f8cab633a8cd92` | `c65eb163c93f893c4b5bd244935a6dacbb4d9fc0ae8152babbebe614cf27bf7b` |
| `docs/ANNOTATION_PROMPT.md` | `1dfd34c4cfc32342015948dcf6d186e9aca143031333160e181874928ecf19e3` | `59d9c2fff8a6c191177fd129cc10dfb96c80b6ce503fa617eb01cdd28da70bc2` |
| `docs/SAP.md` | `3149e7dd624f9e418c8650acd5571f5c576c3cd8115544e14b798602745d8d3d` | `63fd3c7aa39cd123d23b461946b2340b80204327602a40fcfb35f9be163b2ed1` |

## Disclosure

The authors have filed a patent application on a method that underlies the pipeline; the paper states this under Conflicts of Interest and lists which of the evaluated components are not part of the application. Generative-AI use (code, drafting and editing, bibliographic checks, failure labelling) is disclosed in the paper's Acknowledgments.

## Licence

Code and documentation: MIT (`LICENSE`, Copyright (c) 2026 Shikai Li). Benchmark data and the benchmark-bearing files keep their own licences (`DATA_LICENCES.md`).

Run archive (<ZENODO_ARCHIVE_DOI>). It contains benchmark material: BIRD and Spider are published under CC BY-SA 4.0 and Spider-Realistic is treated as CC BY-SA 4.0 (the sources disagree: the Zenodo description says CC BY-SA 4.0, the metadata field says CC BY 4.0). Please attribute the benchmarks to their authors. The archive **modifies** this material: questions, evidence strings and schema text are embedded in the stored model requests and responses, and model outputs and execution results are attached; the modified material is shared under CC BY-SA 4.0. The databases under BIRD and Spider are not redistributed and not relicensed; they keep their upstream licences. The model responses are shared as research data with attribution to the provider (Alibaba Cloud Model Studio); no additional licence is claimed on the raw output text, and their use is subject to the provider's terms.

## Citation

See `CITATION.cff` (software) and the article above.
