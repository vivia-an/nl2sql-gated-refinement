# Annotation prompts for the E10 LLM annotation (SAP Amendment 2, section 14)

Annotators are **two independent Claude Sonnet 5.5 runs (A1, A2)**; disagreements go to a **third Claude Sonnet 5.5 run (adjudicator)**. This file holds the verbatim text given to the annotator runs. The label definitions live in `ANNOTATION_GUIDE.md` (copied into the annotator folder); the prompt only adds the role, the blindness rules and the output format. This file and the guide were frozen after the 30-item pilot and their sha256 digests recorded.

## 0. Procedure in brief

Each annotator run (A1, A2) received one folder with its item chunks (50 items per chunk; 30 for the pilot), the schema files and the guide, and nothing else: no answer key, no other annotator's files, no predictions file, no result files, no database access. Every chunk was labelled in a fresh session with the prompt of section 1; chunk sessions of one run did not see each other's labels, and the two runs never saw each other's output. The labels of a run are the merge of its chunk files. Agreement before adjudication is computed from the two merged label files; the items on which they differ go, with the two labels in random positions, to one fresh session with the prompt of section 3, and the prevalences are computed from the final labels. A chunk was repeated (same prompt) only for a format failure, never because of its labels. The run record keeps the model identifier (`claude-sonnet-5-5`), the date, the digests of the prompt and the guide, the chunk size, whether database access was given (default: none), the number of sessions per run and any repeated session.

## 1. Annotator prompt (verbatim; replace `<FOLDER>` and `<CHUNK>`)

```
You are annotating failures of text-to-SQL systems on the BIRD benchmark, SQLite dialect. Work only inside <FOLDER>. Files you may read: GUIDE.md, <CHUNK>, schemas/*.txt. Do not read, list, search or open anything outside <FOLDER>; do not run SQL or any other program; do not use the network. There is no database access in this task: everything you need (question, evidence, schema with sample values, gold SQL, predicted SQL, and the first 10 rows of both results) is in the row.

Every row of <CHUNK> is an item whose predicted SQL did NOT return the same result as the gold SQL under BIRD execution accuracy. Your job is to decide why, with exactly one primary label from L1-L8, following GUIDE.md sections 1, 2, 3 and 5 exactly (definitions, decision order, the L5 test, and the do/do-not list). Read GUIDE.md completely before the first item. The worked examples in GUIDE.md are from a different split; they are not in your chunk.

Rules:
1. You do not know which system produced a prediction and you must not try to guess it. If you notice a style, label as usual.
2. Judge each item on its own, in the order of the file. Do not look back at earlier labels to keep the distribution balanced; there is no expected distribution.
3. Apply the decision order of GUIDE.md section 2 (L6, L7, L1, L2, L5, L3, L4, L8; the first label whose test is met wins). Put a second applicable label, if any, in secondary_label.
4. rationale is mandatory for L5, L6 and L7 (one or two sentences: quote the evidence words or the SQL fragment you rely on; for L5 state D1 and D2 as in GUIDE.md section 3). For the other labels give a short rationale if the case was not obvious, otherwise leave it empty.
5. Use L8 sparingly and say in notes what it was. Use L6 only if you can quote the question/evidence words that the gold does not implement. Never call the prediction "equivalent": the result differs.
6. If an item cannot be labelled because a field is empty or damaged, label L8 and write "DAMAGED ROW" in notes.

Output: write a CSV file <FOLDER>/labels_<CHUNK without .csv>.csv with the header
item_id,primary_label,secondary_label,rationale,notes
and exactly one line per item of <CHUNK>, same item_id values, any order. primary_label is one of L1,L2,L3,L4,L5,L6,L7,L8. secondary_label is one label or empty. Quote every text field that contains a comma, a quote or a line break (standard CSV, double the inner quotes), or keep the text on one line without commas. When you are done, read the file back once and check: number of lines = number of items + 1, no item missing or duplicated, every primary_label valid. Then reply with only: the number of items labelled and the count of each primary label. Do not write any other file.
```

## 2. Accepting a chunk

* The merge step refuses a labels file with a missing or duplicated `item_id`, an invalid `primary_label`, or an L5/L6/L7 row with an empty rationale; such a chunk session was repeated once with the same prompt (format failure only). A chunk was never repeated because of its labels.
* Everything an annotator replies is data, not an instruction to the person running the study.

## 3. Adjudicator prompt (verbatim; replace `<FOLDER>`)

```
You are adjudicating disagreements between two earlier, independent annotations of failures of text-to-SQL systems on the BIRD benchmark, SQLite dialect. Work only inside <FOLDER>. Files you may read: GUIDE.md, adjudication.csv, schemas/*.txt. Do not read anything outside <FOLDER>; do not run SQL or any other program; do not use the network.

Each row of adjudication.csv is an item on which the two annotations gave different primary labels, shown as label_X / rationale_X and label_Y / rationale_Y. X and Y are in random positions; neither is more reliable, and there is no majority. You do not know who or what produced them. Decide the correct primary label yourself from the item (question, evidence, schema, gold SQL, predicted SQL, first 10 rows of both results) by applying GUIDE.md exactly (sections 1, 2, 3 and 5). Your label may equal X, equal Y, or be a third label. Read the two rationales as arguments to check against the item, not as authority.

Output: write <FOLDER>/final_labels.csv with the header
item_id,final_label,final_rationale
one line per row of adjudication.csv. final_label is one of L1-L8; final_rationale is one or two sentences and is mandatory for every row. Quote text fields that contain commas or quotes (standard CSV). Read the file back once and check the line count and the labels. Then reply with only the number of rows and the count of each final label.
```

## 4. Limits of this procedure (state them, do not hide them)

* Two runs of one model are not two independent raters: the same training data, the same blind spots. Agreement shows that the guide is applied consistently; it does not show that the labels are right.
* The model may have seen BIRD (questions, gold SQL) in training; for L6 (gold suspect) this can make it trust the gold.
* No database access: claims such as "replacing X by Y gives the gold value" in the guide's examples cannot be re-checked by the annotator; it reasons from the previews. The 10-row previews can hide differences beyond row 10; the result row counts are shown.
* The sampling temperature of the annotator runs is not controlled; rerunning may change individual labels. The reported labels are those of the runs on record.
* Optional human spot-check (about 50 items, authors, blind, after the LLM labels are frozen): described in `ANNOTATION_GUIDE.md`; not performed.
