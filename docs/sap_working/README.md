# Working files of the analysis plan

`docs/SAP.md` is a reading copy of the statistical analysis plan (see its first paragraph and "A note on paths and on files that keep their bytes" in the top-level README). This folder holds the working file `docs/SAP.md` of the authors' private repository at the four commits that the paper names (Section 3), **byte for byte**, so that the commit references and the SHA-256 recorded in the result files can be checked. The files are not edited: they keep the internal remarks of the working environment (paths of working notes that are not released, names of work packages in author lines, remarks on who proposed or ran a step and on the machines). The reading copy and these files agree on every section, amendment, date, scope label, number, margin, hash and decision rule that existed at the commit.

| file | commit (author date, UTC-4) | recorded in | sha256 |
|---|---|---|---|
| `SAP_7ea5cff.md` | `7ea5cff` (2026-10-25 13:31): SAP v1 written, before any development run | paper, Section 3 | `e0e9582918e74294ac0b1c2e42ec7f96d42e439f940e42811696b5acc1980dc1` |
| `SAP_900b6ea.md` | `900b6ea` (2026-10-26 05:47): freeze, stage 1 | `exp/dev_lock.json`, `sap_commit` of 8 entries | `abff302b2b4126272ca4a3ec28b21a9bf69a3d8773e0c0c1774d7f0c677358ab` |
| `SAP_899bdd2.md` | `899bdd2` (2026-10-26 06:05): freeze, stage 2 (the SAP is unchanged from `900b6ea`) | `exp/dev_lock.json`, `sap_commit` of 8 entries | `abff302b2b4126272ca4a3ec28b21a9bf69a3d8773e0c0c1774d7f0c677358ab` |
| `SAP_32de07d.md` | `32de07d` (2026-10-27 03:03): version read by the primary analysis | `exp/dev_lock.json`, `sap_commit` of 4 entries; `sap_sha256` in `results/primary.json` | `e3add9a4fdd7a5ff0e3a22b5f6744142ce38326d936316c2208c58e52984319a` |

Check: `sha256sum -c SHA256SUMS` in this folder, and compare the last line with `grep sap_sha256 results/primary.json`.

Commit dates are set by the authors and no external registry or timestamp holds these files; the paper therefore does not call the analysis pre-registered. Amendments made after `32de07d` are in the reading copy `docs/SAP.md` only.
