"""Checkpointed sequential controller; no original run is relaunched in place."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def save(stage,**details):
    p=HERE/'controller_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**details),indent=2)+'\n');tmp.replace(p)


def main():
    lock=HERE/'CONTROLLER_RUNNING'
    if lock.exists():raise RuntimeError('A batch controller is already registered')
    lock.write_text(str(os.getpid()))
    # First fulfill the new paper's three-Kxx scope, then the final field control.
    # Historical two-zone configurations are staged separately, not silently
    # substituted for the three-Kxx experiments requested by the user.
    jobs=[('three_Kxx_main',[sys.executable,str(HERE/'suite.py'),'run-core','--only','clean_no_reg','shift_1m_no_reg','shift_2m_no_reg']),
          ('final_field_beta_zero',[sys.executable,str(HERE/'run_field_no_reg.py')]),
          ('three_Kxx_noise_and_L1_pairs',[sys.executable,str(HERE/'suite.py'),'run-core']),
          ('proposal_legacy_beta_zero',[sys.executable,str(HERE/'suite.py'),'run-legacy','--only',
              'exact_clean','exact_rfsdss_1pct','exact_rfsdss_2pct','exact_rfsdss_5pct','exact_rfsdss_10pct','pert_rfsdss_10pct'])]
    try:
        for label,command in jobs:
            save('running',current=label,command=command)
            env=os.environ.copy()
            env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg',MPLCONFIGDIR='/tmp/dss_no_reg_mpl')
            with (HERE/(label+'.log')).open('a') as log:
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            subprocess.run([sys.executable,str(HERE/'summarize.py')],env=env,check=True)
            if result.returncode:
                save('needs_attention',current=label,returncode=result.returncode,log=str(HERE/(label+'.log')))
                return result.returncode
        save('complete',completed=[j[0] for j in jobs])
        return 0
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':raise SystemExit(main())
