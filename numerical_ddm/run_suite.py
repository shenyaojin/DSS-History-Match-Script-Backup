"""Full reproducible convergence sequence, with resumable completed cases."""
import json,subprocess,sys,hashlib
from pathlib import Path
from run_validation import ROOT


def execute(processors=8,extensions=True):
    script=Path(__file__).with_name('run_validation.py')
    cases=[
      ('opening_h20_L800_t2',['--h','20']),
      ('opening_h10_L800_t2',['--h','10']),
      ('opening_h7p5_L800_t2',['--h','7.5']),
      ('opening_h10_L400_t2',['--h','10','--extent','400']),
      ('opening_h10_L1600_t2',['--h','10','--extent','1600']),
      ('opening_h10_L800_t1',['--h','10','--thickness','1']),
      ('opening_h10_L800_t4',['--h','10','--thickness','4']),
      ('opening_h10_L800_t2_b1',['--h','10','--band-cells','1']),
      ('opening_h10_L800_t2_b4',['--h','10','--band-cells','4']),
      ('opening_h10_L800_t2_q7',['--h','10','--quadrature','SEVENTH']),
    ]+[(f'{mode}_h10_L800_t2',['--h','10','--mode',mode]) for mode in ['s1','s2','mixed','zero','double']]
    cases.append(('mixed_h7p5_L800_t2',['--h','7.5','--mode','mixed']))
    executable_hash=hashlib.sha256((ROOT/'moose_env/moose/modules/combined/combined-opt').read_bytes()).hexdigest()
    for name,opts in cases:
        folder=ROOT/'output/numerical_ddm'/name
        if (folder/'metrics.json').exists() and (folder/'run.json').exists():
            previous=json.loads((folder/'run.json').read_text())
            if previous['success'] and previous.get('executable_sha256')==executable_hash:
                print('Retaining completed case:',name,flush=True)
                continue
        subprocess.run([sys.executable,str(script),'--np',str(processors),'--name',name,*opts],check=True)
    if extensions:
        from run_extensions import execute as run_extensions
        run_extensions(processors)
    from summarize import execute as summarize
    summarize()

if __name__=='__main__': execute()
