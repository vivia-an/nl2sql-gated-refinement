# Annotation guide: failure labels for the E10 sample (LLM annotation)

Status: draft for the 30-item pilot; frozen after the pilot (experiment specification E10). **Amendment 2 (2026-10-26, SAP section 14):** the annotators are two independent runs of Claude Sonnet 5.5 (runs A1 and A2), each blind to the system and to the automatic class, each seeing the items in its own random order, without access to the other run's output. Disagreements are adjudicated by a third Claude Sonnet 5.5 run that sees both labels in blinded positions (X/Y) and by nobody else. We call this "LLM annotation (two independent Claude Sonnet 5.5 runs)". It is **not** human annotation and **not** expert review, and the paper does not describe it as such. The exact instructions given to an annotator run are in `ANNOTATION_PROMPT.md`; the labels below are the same L1-L8 as before.

Limitation: both annotators are runs of the same language model. Their kappa measures the run-to-run consistency of one model under the guide, not the validity of the labels against human judgement; biases shared by the two runs are invisible to it, the model may have seen BIRD during training, and L6 (gold suspect) and L7 (ambiguous) are judgements by a model that did not build the databases. Prevalences are exploratory.

Planned add-on, not performed: the authors had considered labelling a random subset of about 50 of the 300 items after the language-model labels were frozen. This was not done; no human or expert annotation was performed, and the paper reports none.

## 1. What you get and what you do

Files come from `code/tools/make_annotation_sheet.py` (`annotation_A1.csv` / `annotation_A2.csv`, same 300 items in different random orders; `pilot_A*.csv` for the 30 pilot items). Each annotator run receives exactly one of the two files and the schema folder, nothing else (no `annotation_key.csv`, no `sheet_meta.json`, no other annotator's file). Every row has: `item_id, db_id, question, evidence, schema_ref, gold_sql, pred_sql, gold_result, pred_result` and the columns you fill: `primary_label, secondary_label, rationale, notes`.

- `schema_ref` is a path to the schema text (DDL with sample values) of that database, relative to the sheet folder.
- `gold_result` / `pred_result` show the column names, the row count (`+` = truncated) and the first 10 rows, or `ERROR: ...`.
- You are blind to the system that produced the prediction and to the automatic error class. The key file is not in your folder; do not search for it.
- Every item is a failure: the prediction did not match the gold result under BIRD EX. Your job is to say why, in one primary label.
- `primary_label`: exactly one of `L1` ... `L8`. `secondary_label`: optional, one label or empty. `rationale`: one or two sentences, mandatory for L5, L6, L7 (name the evidence sentence or the SQL fragment you rely on). `notes`: free text, not analysed.
- Do not execute SQL to "fix" the prediction. You may run read-only queries on the database to check a claim (for example whether two readings give the same rows); say so in `notes`. (In the LLM annotation of Amendment 2 there is no database access; the annotator works from the previews only, see `ANNOTATION_PROMPT.md`.)

## 2. Labels (primary, one per item)

| Label | Name | Definition (objective test) |
|---|---|---|
| L6 | gold suspect | The gold SQL contradicts the question or the evidence text, or its result is degenerate for the question (empty, NULL, a single constant where a list is asked, counts rows of an unrelated table). Test: quote the evidence/question words that the gold does not implement. |
| L7 | ambiguous question | After reading question and evidence, there are at least two readings that a careful reader would accept, the prediction implements one of them, and the gold implements another. Test: you can write the SQL of the other reading in your head and it would match the gold shape. Evidence that fixes the reading removes this option. |
| L1 | schema linking | The prediction uses a wrong table or wrong column (including a column of the same name in another table, or a column that exists but means something else) while the intended metric and the filters are otherwise as in the gold. Test: replacing the table/column by the gold one would repair the result. |
| L2 | join path | Tables and columns are right, but the join condition, join type, join chain, or the set of joined tables differs (missing join, extra join that multiplies or removes rows, wrong key). Test: same select/filter columns, different FROM clause. |
| L3 | filter / value | A predicate is wrong, missing or extra, or compares against a wrong literal, format, operator or boundary (`LIKE` vs equality, `>=` vs `>`, case, date format, NULL handling) where the evidence does not prescribe a different quantity. |
| L4 | aggregation / grouping / ordering | The aggregate function, DISTINCT, GROUP BY unit, HAVING, ORDER BY direction, LIMIT or tie handling differs from the gold in a way that does not change what concept is computed. |
| L5 | wrong criterion (exploratory) | See section 3. |
| L8 | other | None of the above fits (for example formatting of the answer, rounding, extra or missing output columns with otherwise equal values). Use sparingly and say what it was in `notes`. |

Decision order (first label whose test is met wins; apply L6 and L7 first because they say the failure is not the system's):
1. L6 if the gold itself is questionable.
2. L7 if the question has two defensible readings and evidence does not decide.
3. L1 if a wrong table or column is the cause.
4. L2 if the join path is the cause.
5. L5 if linking and joins are right, the filter columns are right, but the quantity that the question + evidence define (what is counted, divided, ranked, or the population it is computed over) is replaced by another one.
6. L3 if a predicate or literal is wrong.
7. L4 if aggregation, grouping or ordering details differ without changing the defined quantity.
8. L8 otherwise.

If two labels both apply, put the earlier one in the decision order into `primary_label` and the later one into `secondary_label`. Do not use the automatic class (you do not see it) and do not label by how many tokens differ.

## 3. L5 "wrong criterion" (exploratory)

Definition (experiment specification E10): an executable, plausible SQL with the right shape that implements a different metric or definition than the gold/evidence, while table and column linking are right. We report its prevalence as exploratory, weighted by stratum, with a confidence interval, and we report its agreement (kappa on L5-vs-rest) separately. Because it overlaps with L3, L4 and L7 on the boundary, a rationale is mandatory.

Operational test, all three must hold:
1. Linking is right: every table/column that the gold needs for the quantity is used by the prediction (or an equivalent one).
2. The prediction runs and returns the same output shape (same number of columns, a comparable kind of value).
3. You can state in one sentence a definition D1 (what the gold/evidence defines: for example "denominator = all clients", "rank complaint rows", "negation means nominee") and a different definition D2 that the prediction computes, and D1 is written in, or directly implied by, the evidence or the question words.

If D1 is not in the evidence and not implied by the question, the question is under-specified: use L7. If the gold contradicts D1 as written in the evidence, use L6.

Worked examples. The blocks are generated by script from the data and the stored predictions; they are quoted exactly. Provenance: all three are failures of the full system (`full_fixed`, qwen3.7-plus, seed 0) in the tuning-stage base run on the train held-out split HT-300, i.e. tuning-split items; no BIRD dev item is shown in this guide.

**Example A**  (`bird_train:law_episode:1279`, system `full_fixed`, run `P2_base/full_fixed-qwen37plus-s0`)

- question: What is the percentage of people who gave the "True North" episode a 1-star rating?
- evidence: the "True North" episode refers to title = 'True North'; 1-star refers to stars = 1; percentage = divide(count(episode_id where stars = 1), count(episode_id)) * 100% where title = 'True North'
- gold SQL:

```sql
SELECT CAST(SUM(CASE WHEN T2.stars = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.episode_id) FROM Episode AS T1 INNER JOIN Vote AS T2 ON T1.episode_id = T2.episode_id WHERE T1.title = 'True North' AND T1.episode_id = 'tt0629477'
```

- predicted SQL:

```sql
SELECT CAST(SUM(CASE WHEN T2.stars = 1 THEN T2.votes ELSE 0 END) AS REAL) * 100 / SUM(T2.votes)
FROM Episode AS T1
INNER JOIN Vote AS T2 ON T1.episode_id = T2.episode_id
WHERE T1.title = 'True North'
```

- gold result: 1 row(s): [(10.0,)]
- predicted result: 1 row(s): [(2.150537634408602,)]

**Example B**  (`bird_train:law_episode:1298`, system `full_fixed`, run `P2_base/full_fixed-qwen37plus-s0`)

- question: How many episodes have not won any Law & Order series awards?
- evidence: have not won any award refers to Result = 'Nominee'
- gold SQL:

```sql
SELECT COUNT(award_id) FROM Award WHERE Result = 'Nominee'
```

- predicted SQL:

```sql
SELECT COUNT(DISTINCT T1.episode_id)
FROM Episode AS T1
WHERE T1.series = 'Law and Order'
  AND T1.episode_id NOT IN (
    SELECT T2.episode_id
    FROM Award AS T2
    WHERE T2.result = 'Winner'
  )
```

- gold result: 1 row(s): [(10,)]
- predicted result: 1 row(s): [(20,)]

**Example C**  (`bird_train:retail_complains:285`, system `full_fixed`, run `P2_base/full_fixed-qwen37plus-s0`)

- question: What percentage of consumers from Houston disputed complaints?
- evidence: percentage = MULTIPLY(DIVIDE(SUM("Consumer disputed?" = 'Yes' AND city = 'Houston'), COUNT(client_id)), 1.0); Houston refers to city = 'Houston';
- gold SQL:

```sql
SELECT CAST(SUM(CASE WHEN T2.`Consumer disputed?` = 'Yes' AND T1.city = 'Houston' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.client_id) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID
```

- predicted SQL:

```sql
SELECT CAST(SUM(CASE WHEN e."Consumer disputed?" = 'Yes' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(c.client_id)
FROM client c
JOIN events e ON c.client_id = e.Client_ID
WHERE c.city = 'Houston'
```

- gold result: 1 row(s): [(0.6148853495025407,)]
- predicted result: 1 row(s): [(21.176470588235293,)]


Why L5. Example A: the evidence defines the percentage as the count of rows with `stars = 1` over the count of rows, both for the episode titled 'True North'; the prediction weights every row by its `votes` (2.15 instead of 10.0). Tables, columns and the title filter are right; replacing the vote-weighting by row counts in the prediction returns exactly the gold value 10.0 (checked by query). Note that the prediction is arguably the more natural reading of "percentage of people", but the evidence prescribes the quantity, and the gold implements the evidence as written, so this is not L6: L6 is for a gold that contradicts the evidence, not for evidence that is odd. Example B: the evidence defines "not won any award" as `Result = 'Nominee'`; the prediction replaces it with "no award whose result is 'Winner'" and adds a series filter, so the quantity differs although the same `Award` table is used. Example C: the evidence formula divides the number of disputed Houston clients by the count of all joined clients; the prediction moves the city into WHERE, which changes the denominator to the Houston clients only (21.18 against 0.61). Tables, columns and predicate are right, the defined quantity is not.

### Contrast cases (not L5)

**Contrast D (L6, not L5)**  (`bird_train:retail_complains:341`, system `full_fixed`, run `P2_base/full_fixed-qwen37plus-s0`)

- question: What percentage of complaints are from the elderly?
- evidence: elder refers to age < = 65; percentage refers to divide(sum(age < = 65) , count(client_id)) * 100%
- gold SQL:

```sql
SELECT CAST(SUM(CASE WHEN T1.age > 65 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.age) FROM client AS T1 INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID
```

- predicted SQL:

```sql
SELECT CAST(SUM(CASE WHEN T2.age <= 65 THEN 1.0 ELSE 0 END) AS REAL) * 100 / COUNT(T1."Complaint ID") AS percentage
FROM callcenterlogs AS T1
INNER JOIN client AS T2 ON T1."rand client" = T2.client_id
```

- gold result: 1 row(s): [(15.201332251590589,)]
- predicted result: 1 row(s): [(86.42172523961662,)]

**Contrast E (L1, not L5)**  (`bird_train:retail_complains:396`, system `full_fixed`, run `P2_base/full_fixed-qwen37plus-s0`)

- question: Among the complaints received in year 2015, what is total number of complaints timely response and closed with an explanation?
- evidence: in year 2015 refers to Date received LIKE '2015%'; timely response refers to Timely response?" = 'Yes'; closed with an explanation refers to Company response to consumer = 'Closed with explanation'
- gold SQL:

```sql
SELECT COUNT(T1.`Complaint ID`) FROM callcenterlogs AS T1 INNER JOIN events AS T2 ON T1.`Complaint ID` = T2.`Complaint ID` WHERE strftime('%Y', T1.`Date received`) = '2015' AND T2.`Timely response?` = 'Yes' AND T2.`Company response to consumer` = 'Closed with explanation'
```

- predicted SQL:

```sql
SELECT COUNT(*) 
FROM events 
WHERE "Date received" LIKE '2015%' 
  AND "Timely response?" = 'Yes' 
  AND "Company response to consumer" = 'Closed with explanation'
```

- gold result: 1 row(s): [(349,)]
- predicted result: 1 row(s): [(3136,)]


Why not L5. D: the evidence says "elder refers to age <= 65" but the gold computes `age > 65`, so the gold contradicts the evidence text: L6 (the prediction follows the evidence; the two results sum to 100). E: the prediction filters the year on `events."Date received"` while the gold joins `callcenterlogs` and filters its `Date received`; the criterion (complaints received in 2015 that were timely and closed with an explanation) is the same, the column of the same name in the other table is the difference: L1.

## 4. Process

1. Pilot: two fresh annotator runs (A1, A2) label the 30 pilot items (`pilot_A1.csv`, `pilot_A2.csv`) independently, in separate sessions. The two runs are compared, wording problems are resolved in this guide and in `ANNOTATION_PROMPT.md` once, and both texts are then frozen (the freeze date and the digests are recorded). Pilot runs are discarded; the main round uses fresh runs with the frozen text.
2. Main round: two new independent runs label the 300 items (A1 gets `annotation_A1.csv`, A2 gets `annotation_A2.csv`). The runs do not communicate; neither run is shown the other's output.
3. Agreement before adjudication, whatever the value: `python3 code/tools/annotation_kappa.py --a1 annotation_A1.csv --a2 annotation_A2.csv --key annotation_key.csv --out <dir>` writes percent agreement, unweighted Cohen kappa and Gwet AC1 on the primary label and on L5-vs-rest (item bootstrap CI), the confusion table, `adjudication.csv` (disagreeing items only; the two labels in random positions X and Y so the adjudicator does not know which run gave which) and the stratum-weighted prevalence tables with stratified-bootstrap CIs. (`make_annotation_sheet.py --adjudicate` gives the same preview of agreement and the same adjudication file.)
4. The adjudicator (a third Claude Sonnet 5.5 run, prompt in `ANNOTATION_PROMPT.md` section 3) fills `final_label` and `final_rationale` for every row of `adjudication.csv`. Items with equal labels keep that label. The adjudicator is told nothing about which run produced X or Y.
5. Prevalences use the post-stratification weights in `annotation_key.csv` (population size of the stratum / sample size of the stratum).
6. Reporting: kappa before adjudication, the limitation paragraph above, and the number of items for which the adjudicated label differs from both runs (if any).

## 5. Do and do not

- Do read the evidence first; most BIRD gold SQL implements the evidence literally, including odd evidence.
- Do not call a prediction "equivalent" when its result differs; every item here already failed EX. If you believe the prediction is right and the gold is wrong, the label is L6.
- Do not guess the system from the style of the SQL. If you notice it, label as usual.
- Do not invent an external ground truth; only the question, the evidence, the schema, the gold and both results count.
