"""Summarize actual completion state without mistaking staged cases for results."""
from pathlib import Path
from datetime import datetime,timezone
import csv
import json

OUT=Path(__file__).resolve().parent


def main():
    rows=[]
    for path in sorted((OUT/'runs').rglob('config.json')):
        cfg=json.loads(path.read_text());folder=path.parent
        field=folder.name=='fixed_geometry_inversion'
        result=folder/('optimizer_result.json' if field else 'result.json')
        state=folder/'status.json'
        d=json.loads(result.read_text()) if result.exists() else {}
        st=json.loads(state.read_text()) if state.exists() else {}
        row=dict(case=str(folder.relative_to(OUT)),status=st.get('stage','prepared'),
            complete=(folder/'COMPLETE').exists(),beta=cfg.get('beta',cfg.get('beta_l1','')),
            n_parameters=len(cfg.get('initial',cfg.get('initial_theta',[]))),
            data_objective=d.get('data_objective',''),rms_microstrain=d.get('rms_microstrain',''),
            theta=json.dumps(d.get('theta_log10_Kxx',d.get('theta',[]))),
            termination=d.get('termination',''),note=cfg.get('comparison_limit',''))
        if field and d:
            row['data_objective']=d.get('results',{}).get('weighted_objective_after','')
            row['rms_microstrain']=d.get('results',{}).get('unweighted_rms_after_microstrain','')
        rows.append(row)
    target=OUT/'run_summary.csv';temp=target.with_suffix('.tmp')
    with temp.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(target)
    counts={s:sum(r['status']==s for r in rows) for s in {r['status'] for r in rows}}
    summary=dict(as_of_utc=datetime.now(timezone.utc).isoformat(),counts=counts,
        completed=sum(r['complete'] for r in rows),staged_cases=len(rows),
        not_all_staged_cases_are_scheduled=True,
        scheduled_legacy='Only the six proposal-source v5 cases; the v6 factorial grid is staged, not silently rerun',
        scientific_scope='Three Kxx core experiments, fixed transverse permeability, same synthetic truth across geometry perturbations; no layerwise inversion')
    (OUT/'summary_status.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
