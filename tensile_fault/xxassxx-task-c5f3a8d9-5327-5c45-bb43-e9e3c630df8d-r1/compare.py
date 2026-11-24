from pathlib import Path
import csv, json, math, hashlib
root=Path(__file__).resolve().parent
results={}
rows_by_case={}
for case in ['baseline','shifted']:
    folder=root/case
    status=json.loads((folder/'status.json').read_text())
    rows=list(csv.DictReader((folder/'model_csv.csv').open()))
    assert status['exit_code']==0 and float(rows[-1]['time'])==61200
    assert len(rows)==5
    final=rows[-1]
    metrics={k:float(v) for k,v in final.items()}
    arrays={}
    for kind in ['pressure','strain','strain_rate']:
        file=folder/f'model_csv_fiber_{kind}_sampler_-1117.0ft_0004.csv'
        data=list(csv.DictReader(file.open()))
        assert len(data)==200
        assert all(math.isfinite(float(v)) for row in data for v in row.values())
        arrays[kind]=data
        for col in data[0]:
            if col.startswith(('pp','strain')):
                vals=[float(row[col]) for row in data]
                metrics[kind+'_'+col+'_min']=min(vals)
                metrics[kind+'_'+col+'_max']=max(vals)
                metrics[kind+'_'+col+'_max_abs']=max(map(abs,vals))
    results[case]=metrics
    rows_by_case[case]=arrays
delta={k:results['shifted'][k]-v for k,v in results['baseline'].items()}
profile_delta={}
for kind, data in rows_by_case['baseline'].items():
    for col in data[0]:
        if col.startswith(('pp','strain')):
            profile_delta[kind+'_'+col+'_max_abs_difference']=max(abs(float(a[col])-float(b[col])) for a,b in zip(data,rows_by_case['shifted'][kind]))
summary={'verified':True,'results':results,'shift_minus_baseline':delta,'profile_differences':profile_delta}
(root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
with (root/'comparison.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['metric','baseline','shifted','shift_minus_baseline'])
    for k,v in results['baseline'].items(): writer.writerow([k,v,results['shifted'][k],delta[k]])
files=[]
for p in sorted(root.rglob('*')):
    if p.is_file() and not any(x.startswith('.') for x in p.relative_to(root).parts) and p.name!='manifest.json':
        files.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(root/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps(summary,indent=2))
