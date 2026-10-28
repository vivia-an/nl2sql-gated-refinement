import csv, json, sys, collections
W = '<work>/'
final = {}
# final label = common label if runs agree, adjudicated label otherwise
a1 = {r['item_id']: r['primary_label'] for r in csv.DictReader(open(W + 'out/labels_A1.csv'))}
a2 = {r['item_id']: r['primary_label'] for r in csv.DictReader(open(W + 'out/labels_A2.csv'))}
adj = {r['item_id']: r['final_label'] for r in csv.DictReader(open(W + 'out/final_labels.csv'))}
for i in a1:
    final[i] = a1[i] if a1[i] == a2[i] else adj[i]
key = {r['item_id']: r for r in csv.DictReader(open(W + 'sheet/annotation_key.csv'))}
def load(p): return {json.loads(l)['qid']: json.loads(l) for l in open(p)}
F = load(W + 'remote/preds_fixed-s0.jsonl'); G = load(W + 'remote/preds_gated-s0.jsonl')
def cands(p):
    d = collections.defaultdict(list)
    for l in open(p):
        x = json.loads(l); d[x['qid']].append(x)
    return d
CF = cands(W + 'remote/cands_fixed-s0.jsonl'); CG = cands(W + 'remote/cands_gated-s0.jsonl')
disc = sorted(q for q in F if F[q]['ex'] != G[q]['ex'])
assert len(disc) == 28
EXEC = {'ok_nonempty', 'ok_empty'}
def rev(preds, cs, qids):
    """revision changed an executable SQL: root executable (status ok_*) and final_sql differs from root sql."""
    out = collections.Counter(); ids = {'helped': [], 'hurt': [], 'neutral': []}
    for q in qids:
        root = [c for c in cs[q] if c['action'] == 'root'][0]
        if root['status'] not in EXEC or preds[q]['final_sql'].strip() == root['sql'].strip():
            continue
        r0, r1 = root['ex'], preds[q]['ex']
        k = 'helped' if (r0 == 0 and r1 == 1) else 'hurt' if (r0 == 1 and r1 == 0) else 'neutral'
        out[k] += 1; ids[k].append(q)
    return dict(n_changed=sum(out.values()), helped=out['helped'], hurt=out['hurt'],
                neutral_both_correct_or_both_wrong=out['neutral']), ids
def repair(preds, cs, qids):
    out = collections.Counter()
    for q in qids:
        root = [c for c in cs[q] if c['action'] == 'root'][0]
        if root['status'] in EXEC or preds[q]['final_sql'].strip() == root['sql'].strip():
            continue
        r0, r1 = root['ex'], preds[q]['ex']
        out['helped' if (r0 == 0 and r1 == 1) else 'hurt' if (r0 == 1 and r1 == 0) else 'neutral'] += 1
    return dict(n_changed=sum(out.values()), helped=out['helped'], hurt=out['hurt'], neutral_both_correct_or_both_wrong=out['neutral'])
allq = sorted(F)
res = {}
for name, P, C in (('full_fixed', F, CF), ('full_gated', G, CG)):
    res.setdefault('revision_effect_all_dev_items', {})[name] = rev(P, C, allq)[0]
    res.setdefault('revision_effect_discordant_items', {})[name] = rev(P, C, disc)[0]
    res.setdefault('repair_of_nonexecutable_root_all_dev_items', {})[name] = repair(P, C, allq)
    res.setdefault('repair_of_nonexecutable_root_discordant_items', {})[name] = repair(P, C, disc)
# sheet lookup
inkey = collections.defaultdict(list)
for i, k in key.items():
    if i.startswith('A'):
        inkey[k['qid']].append((i, k['system']))
rows = []
cnt_full = collections.Counter(); cnt_zs = collections.Counter()
for q in disc:
    ents = inkey.get(q, [])
    r = dict(qid=q, fixed_ex=F[q]['ex'], gated_ex=G[q]['ex'], in_sheet_as_full_prediction=None, label_full=None, in_sheet_as_zs_prediction=None, label_zs=None)
    for i, s in ents:
        if s == 'full': r['in_sheet_as_full_prediction'] = i; r['label_full'] = final[i]; cnt_full[final[i]] += 1
        else: r['in_sheet_as_zs_prediction'] = i; r['label_zs'] = final[i]; cnt_zs[final[i]] += 1
    rows.append(r)
res['discordant'] = dict(
    n=len(disc), fixed_correct_gated_wrong=sum(F[q]['ex'] > G[q]['ex'] for q in disc), fixed_wrong_gated_correct=sum(F[q]['ex'] < G[q]['ex'] for q in disc),
    n_in_sheet_as_full_prediction=sum(1 for r in rows if r['label_full']), label_counts_full_prediction=dict(sorted(cnt_full.items())),
    n_in_sheet_as_zs_prediction=sum(1 for r in rows if r['label_zs']), label_counts_zs_prediction_not_the_full_systems=dict(sorted(cnt_zs.items())),
    note='The sheet is a stratified sample of failures of full_fixed (system full) and zs; a discordant item appears only if sampled. full rows are failures of full_fixed; a zs row labels the zs prediction of that question, not either full system. Final label = common label of A1/A2 or the adjudicated label.',
    items=rows)
res['definitions'] = dict(
    revision_changed_executable_sql='root candidate executes (status ok_nonempty or ok_empty) and the final SQL differs from the root SQL (text, stripped)',
    helped='root ex=0 -> final ex=1', hurt='root ex=1 -> final ex=0', neutral='root and final both correct or both wrong', repair_of_nonexecutable_root='root status is an error/timeout and the final SQL differs from the root SQL (reported separately; not a change of an executable SQL)',
    source='runs/MIN/M1/fixed-s0 and gated-s0 (preds.jsonl, cands.jsonl; dev; seed 0); ex is the stored evaluator flag')
json.dump(res, open(W + 'out/discordant_summary.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'discordant'}, indent=1)); d = dict(res['discordant']); d.pop('items'); print(json.dumps(d, indent=1))
