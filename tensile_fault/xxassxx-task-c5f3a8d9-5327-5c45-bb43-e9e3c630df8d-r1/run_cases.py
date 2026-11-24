from pathlib import Path
import os, subprocess, json, time, shlex
root=Path(__file__).resolve().parent
env=os.environ.copy()
env.update(UCX_TLS='self',HWLOC_COMPONENTS='-linux',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),PYTHONDONTWRITEBYTECODE='1')
env['PATH']='/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin:'+env.get('PATH','')
exe='/rcp/rcp42/home/shenyaojin/Documents/bakken_mariner/moose_env/moose/modules/porous_flow/porous_flow-opt'
records=[]
for case in ['baseline','shifted']:
    folder=root/case
    if (folder/'status.json').exists():
        raise RuntimeError('Existing attempt: inspect before rerun')
    command=[exe,'-i','model.i','--n-threads=1']
    (folder/'command.txt').write_text('UCX_TLS=self HWLOC_COMPONENTS=-linux OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '+shlex.join(command)+'\n')
    start=time.monotonic()
    with (folder/'simulation.log').open('w') as log:
        try:
            result=subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=700)
            code=result.returncode
        except subprocess.TimeoutExpired:
            code=124
    record={'case':case,'exit_code':code,'wall_seconds':time.monotonic()-start,'single_process':True,'command':command}
    (folder/'status.json').write_text(json.dumps(record,indent=2)+'\n')
    records.append(record)
    print(json.dumps(record),flush=True)
(root/'run_status.json').write_text(json.dumps(records,indent=2)+'\n')
