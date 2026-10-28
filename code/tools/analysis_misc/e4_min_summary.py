import os
import sys,json,glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from analysis_pipeline import e4
R=os.environ.get('NL2SQL_ROOT', os.path.expanduser('~/nl2sql_work')) + '/'
_r=e4.load_all([R+'runs/MIN/E4']); units=_r[0] if isinstance(_r,tuple) else _r; print(type(_r), [type(x) for x in _r] if isinstance(_r,tuple) else '')
print('units',len(units))
for pol in ('ws_full','memrl','append'):
    rows=e4.probe_rows(units,pol,40)
    y=rows['y'] if isinstance(rows,dict) else [r['y'] for r in rows]
    print(pol,'p40 final-probe EX',round(sum(y)/max(len(y),1),4),'n',len(y))
for a,b in (('ws_full','append'),('ws_full','memrl'),('memrl','append')):
    try:
        r=e4.policy_contrast(units,a,b,40,B=2000,fit=False)
        print(a,'-',b,json.dumps({k:v for k,v in r.items() if k in ('diff_pp','ci95','ci','pp','n','n_dbs','delta_pp','diff')},default=str)[:300] or list(r.keys()))
        if not any(k in r for k in ('diff_pp','ci95','ci','pp','n','n_dbs','delta_pp','diff')): print(list(r.keys()))
    except Exception as ex: print(a,b,'ERR',repr(ex)[:200])
for pol in ('ws_full','memrl','append'):
    try: print(pol,'evict diag',json.dumps(e4.eviction_diag(units,pol,40),default=str)[:300])
    except Exception as ex: print('diag ERR',repr(ex)[:150])
for m in ('qwen37flash','qwen3527b'):
    P=[json.loads(l) for l in open(R+'runs/X8/%s-zs/preds.jsonl'%m)]
    print(m,'zs EX',round(sum(p['ex'] or 0 for p in P)/len(P),4),len(P))
