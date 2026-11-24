"""Final Phase7 control: identical data, geometry, bounds and optimizer; beta=0.

Read-only completed physical evaluations may be reused because the penalty is
outside the PDE. New evaluations are written only below this experiment folder.
"""
from pathlib import Path
import json
import os
import sys

os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/dss_no_reg_mpl')
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
SOURCE=ROOT/'scripts/DSS_history_match/real_rfsdss_fixed_zones_14373_14379_400000s_bg21_coupling/background_sensitivity'
ORIGINAL=SOURCE/'continuous_geometry_bg18'
CASE=OUT/'runs/field_phase7_no_reg'
sys.path.insert(0,str(ROOT/'fibeRIS/src'))
sys.path.insert(0,str(SOURCE))
import numpy as np
import continuous_geometry as model
import continuous_geometry_inversion as prior


class NoRegularization(prior.Inversion):
    def __init__(self):
        super().__init__()
        self.beta=0.
        # Recompute each cached scalar objective using beta=0, never reuse a
        # total objective or a regularized gradient from a previous run.
        self.cache={}
        self.reused=[]
        for folder in [self.initial_folder,*sorted((ORIGINAL/'permeability_evaluations').glob('k_*'))]:
            if not (folder/'COMPLETE').exists():continue
            m=json.loads((folder/'manifest.json').read_text())
            if [m[k] for k in ['center_ft','width_ft','h_y_max_m']] != [self.geometry[k] for k in ['center_ft','width_ft','h_y_max_m']]:continue
            theta=m['theta']
            self.cache[self.key(theta)]=self.read(folder,theta)
            self.reused.append(str(folder))
        cfg=json.loads((prior.RUN/'config.json').read_text())
        cfg.update(beta_l1=0.,regularization_note='User-requested zero-penalty control; bounds retained',
            objective='0.5 sum(w*(C*s_plus-d_plus)^2), t>0; no penalty',
            max_concurrent_forwards=1,mpi_ranks_per_forward=20,
            source_regularized_result=str(ORIGINAL/'fixed_geometry_inversion/optimizer_result.json'),
            reuse_policy='Only physical forward predictions reused read-only; objectives/gradients recomputed with beta=0',
            reused_forward_folders=self.reused)
        model.base.save(prior.RUN/'config.json',cfg)

    def gradient(self,theta,h):
        # The historical driver launched 2 x 20 ranks. Keep this control within
        # one 20-rank allocation; the finite-difference formula is unchanged.
        result=[]
        for j in range(2):
            points,coeff=prior.stencil(theta,j,h,self.bounds)
            result.append(sum(float(c)*self.get(p)['data'] for p,c in zip(points,coeff))*self.scale)
        return np.array(result)


def main():
    prior.verify_stencils()
    if (CASE/'fixed_geometry_inversion/COMPLETE').exists():
        print('Already complete');return
    CASE.mkdir(parents=True,exist_ok=True)
    (CASE/'selected_geometry.json').write_bytes((ORIGINAL/'selected_geometry.json').read_bytes())
    model.OUT=CASE
    prior.RUN=CASE/'fixed_geometry_inversion'
    prior.RUN.mkdir(exist_ok=True)
    lock=prior.RUN/'RUNNING'
    if lock.exists():raise RuntimeError('Refusing overlapping field controller')
    lock.write_text(f'pid={os.getpid()}\n')
    try:
        NoRegularization().run()
    except Exception as exc:
        model.base.status(prior.RUN/'status.json','failed',reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
