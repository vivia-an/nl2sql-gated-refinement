# Data and third-party licence notes

This repository contains code, configurations, the analysis plan, result files and prompts. It does **not** contain benchmark
databases or the raw benchmark files; download them from the original sources and respect their licences.
The statements below were taken from the upstream project pages, README and LICENSE files between 2026-10-26 and 2026-10-28.
They are second-hand and are not legal advice; re-check them before relying on them.

## Summary

| Material | Licence | Note |
|---|---|---|
| Code, configurations, analysis plan, documentation (everything not listed below) | MIT (`LICENSE`) | |
| Files that contain benchmark material: SQL text and questions of BIRD training items in `results/diag/`, and the archived model requests (questions, evidence strings, schema text with sample values) in the separate run archive | CC BY-SA 4.0 | attribution and modification notice below |
| Raw text of language-model outputs (the stored responses in the run archive) | no licence added | shared as research data, source attributed |
| Benchmark databases and raw benchmark files | not included | keep their upstream licences |

## Benchmarks

| Data set | Used for | Licence as stated by its source | What is redistributed here |
|---|---|---|---|
| BIRD train and dev (Li et al., NeurIPS 2023), https://bird-bench.github.io | tuning (train, disjoint databases) and the confirmatory test (dev, version 20240627) | CC BY-SA 4.0. The project page states (news of 2024-04-27) that the data licence was changed to CC BY-SA 4.0; the datasheet of the NeurIPS paper still says CC BY-NC 4.0 and is superseded by that notice. | No databases or raw files. Question identifiers and execution digests; in `results/diag/` SQL text of BIRD **training** questions and perturbations derived from it; in the run archive the questions, evidence strings and schema text contained in the model requests. |
| BIRD Mini-Dev | overlap check with the corrected sets | CC BY-SA 4.0 | nothing |
| Corrected BIRD dev subset (Jin et al., VLDB 2026) | post hoc robustness check | CC BY-SA 4.0 (LICENSE of the upstream repository) | no corrected text; identifiers and digests only |
| Arcwise-Plat / BIRD-Platinum (ReViSQL release, Zhu et al.) | post hoc robustness check | The copy inside the Jin et al. repository falls under that repository's CC BY-SA 4.0 LICENSE; the ReViSQL repository itself has no LICENSE file. Status unclear. | **No corrected reference text** (the export contains no `original_*` or corrected-SQL fields) |
| Spider 1.0 (Yu et al.), https://yale-lily.github.io/spider | exemplar pool of the external-validity check | CC BY-SA 4.0 (official page) | exemplar questions that occur in archived requests |
| Spider-Realistic (Deng et al.) | external-validity check | The sources disagree: the description of the Zenodo record says CC BY-SA 4.0, its metadata field says CC BY 4.0. The set derives from Spider (CC BY-SA 4.0), so it is treated as CC BY-SA 4.0 here. | questions that occur in archived requests |
| Spider test-suite evaluation code | scoring on Spider-style data | Apache-2.0 | no (install from upstream) |
| Alpha-SQL prompt templates (`code/nl2sql/systems/baselines/_vendor/alphasql_templates/`) | baseline | MIT (licence file kept next to the templates) | yes, unchanged, with the licence file |
| MAC-SQL `core/const.py` | baseline prompts | The upstream repository has no licence file | **Not redistributed.** `python code/tools/fetch_macsql_const.py` downloads it from the upstream commit listed in `code/nl2sql/systems/baselines/_vendor/SOURCES.txt` and checks the sha256 in `docs/BASELINES.md`. |

Not used in the paper: Spider-Syn (MIT), Spider-DK (MIT), AmbiQT (MIT), LiveSQLBench-Base-Lite (licence statements on the dataset card disagree).

## Attribution

Please attribute the benchmarks to their authors:

- Li, J. et al. Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs (BIRD). NeurIPS 2023.
- Yu, T. et al. Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task. EMNLP 2018.
- Deng, X. et al. Structure-Grounded Pretraining for Text-to-SQL (Spider-Realistic). NAACL 2021.
- Jin et al. (corrected BIRD subsets), VLDB 2026.

## Modification notice (CC BY-SA 4.0, section 3(a)(1))

The benchmark material in the files named above was **modified**: questions, evidence strings and schema descriptions were rendered into
prompt text, sample values were added to the schema text, perturbations of training SQL were generated for the critic diagnostics
(`results/diag/`), and model outputs were attached. These files are adapted material and are shared under CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/). The databases that sit under BIRD and Spider keep the licences of their original
sources (the BIRD paper, appendix B.10, lists Public Domain, CC BY, CC BY-SA, GPL, CPOL and CC0 among them); this release does not relicense them.

## Model outputs

The stored language-model requests and responses were produced through the OpenAI-compatible endpoint of Alibaba Cloud Model Studio
(see `exp/models.yaml`) with a paid account of the authors (the Qwen models of version 1.0.0 of the run archive, and, in version 1.1.0, also `deepseek-v4.1-flash-1017`, which the same service offers under the same account); model identifiers and call dates are in the run metadata of the archive. They are
shared as research data so that the analyses can be replayed. No licence is added to the raw output text. The terms of the service provider
apply to their use; at the time of writing (checked 2026-10-28) they leave the rights in model output with the customer and do not allow using
the service or its output to train competing models. The responses are not human-written text and may contain errors.

**DeepSeek outputs (run archive 1.1.0, post hoc runs).** The responses of `deepseek-v4.1-flash-1017` were generated through Alibaba Cloud Model Studio under the same account terms as the Qwen responses. They are shared as research data on the same basis: no licence is added to the raw output text and none is claimed; the terms of the service and of the model developer, where they apply, govern further use.

## The run archive (Zenodo record https://doi.org/10.5281/zenodo.23735018; version 1.0.0 is https://doi.org/10.5281/zenodo.23735019, version 1.1.0 adds the post hoc runs)

- **Benchmark material, CC BY-SA 4.0 with attribution.** BIRD and Spider are published under CC BY-SA 4.0, and Spider-Realistic is treated as CC BY-SA 4.0 (see the table). The archive contains questions, evidence strings, schema text with sample values and exemplar questions from them. Please attribute the benchmarks to their authors (list above).
- **Modification notice.** The archive modifies the benchmark material: questions, evidence strings and schema text are embedded in the stored model requests and responses (rendered into prompts, with sample values added to the schema text), and model outputs and execution results are attached to them. The modified material is shared under CC BY-SA 4.0.
- **Underlying databases.** The databases under BIRD and Spider are not in the archive and are not relicensed here; they keep the licences of their original sources.
- **Model responses.** The responses are shared as research data, with attribution to the provider (Alibaba Cloud Model Studio, models named in the run metadata; for version 1.1.0 this includes the DeepSeek model named above). No additional licence is claimed on the raw output text. Their use is subject to the provider's terms.

## Code

See `LICENSE` (MIT). The method is the subject of a patent application filed by the authors; see the Conflicts of Interest statement of the article.
