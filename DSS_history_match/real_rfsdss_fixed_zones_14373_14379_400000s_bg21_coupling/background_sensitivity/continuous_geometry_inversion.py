"""Fixed selected geometry: L-BFGS-B permeability inversion with profiled C.

The current executable lacks the historical custom adjoint VPP. Use explicit
bounded finite-difference stencils on the unchanged physical forward, verify
both permeability directions at h and h/2, and stop if either check fails.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import continuous_geometry as model

RUN=model.OUT/'fixed_geometry_inversion'


def stencil(theta,index,h,bounds):
    """Second-order centered or one-sided derivative, entirely within bounds."""
    p=float(theta[index]);lo,hi=bounds[index]
    if p-h<lo:
        offsets=[0,h,2*h];coeff=np.array([-3.,4.,-1.])/(2*h)
    elif p+h>hi:
        offsets=[0,-h,-2*h];coeff=np.array([3.,-4.,1.])/(2*h)
    else:
        offsets=[-h,h];coeff=np.array([-1.,1.])/(2*h)
    points=[]
    for offset in offsets:
        point=np.array(theta,dtype=float);point[index]+=offset
        assert lo-1e-12<=point[index]<=hi+1e-12
        points.append(point)
    return points,coeff


def verify_stencils():
    bounds=[[-21.,-12.],[-21.,-12.]]
    for x in [-21.,-17.,-12.]:
        theta=np.array([x,x])
        for j in range(2):
            points,coeff=stencil(theta,j,.02,bounds)
            derivative=sum(c*np.dot(p,p) for p,c in zip(points,coeff))
            np.testing.assert_allclose(derivative,2*x,rtol=0,atol=1e-9)


class Inversion:
    def __init__(self):
        self.selection=json.loads((model.OUT/'selected_geometry.json').read_text())
        assert self.selection['ready_for_inversion']
        self.geometry=self.selection['geometry']
        self.original=json.loads((model.HERE.parent/'inversion/config.json').read_text())
        self.bounds=self.original['parameter_bounds'][:2]
        self.theta0=np.asarray(self.geometry['theta'],dtype=float)
        self.scale=float(self.original['objective_scale'])
        self.beta=float(self.original['beta_l1'])
        self.delta=float(self.original['regularization_delta'])
        self.reference=float(self.original['regularization_reference_alpha'])
        self.cache={};self.gradients={};self.accepted=0
        self.initial_folder=Path(self.selection['folder'])
        self.cache[self.key(self.theta0)]=self.read(self.initial_folder,self.theta0)
        options={k:self.original['optimizer'][k] for k in ['ftol','gtol','maxls']}
        options['maxiter']=60
        self.options=options
        model.base.save(RUN/'config.json',dict(geometry=self.geometry,initial_theta=self.theta0.tolist(),
            background_alpha=-18,optimized_parameters=['alpha_xx','alpha_yy'],bounds=self.bounds,
            coupling='Unbounded analytical weighted least-squares optimum at every permeability evaluation',
            objective='0.5 sum(w*(C*s_plus-d_plus)^2) + unchanged smoothed L1 on xx/yy; t>0',
            core_sse_weight=3,core_depths_ft=self.geometry['central_channel_depths_ft'],
            regularization_reference_alpha=self.reference,regularization_delta=self.delta,beta_l1=self.beta,
            regularization_note='Retain previous inversion parameters; reference remains -21 although physical background is now -18',
            objective_scale=self.scale,optimizer='L-BFGS-B',optimizer_options=options,
            gradient_method='Second-order finite differences; center if interior, one-sided at bounds',
            gradient_step=.02,gradient_check_step=.01,
            gradient_reason='Current executable does not register PorousFlowOptimizationAnisotropicDiffusionInnerProduct; no source or binary rebuilt',
            max_concurrent_forwards=2,mpi_ranks_per_forward=20,n_forward_steps=129,max_steps=240,
            original_config_sha256=model.digest(model.HERE.parent/'inversion/config.json'),
            interpretation_limit='Pressure overshoot remains a separate unresolved numerical limitation'))

    @staticmethod
    def key(theta):
        return tuple(float(round(float(v),12)) for v in theta)

    def read(self,folder,theta):
        summary=json.loads((folder/'summary.json').read_text())
        row=summary['results'][0]
        d=np.asarray(theta)-self.reference
        denom=np.sqrt(d*d+self.delta*self.delta)
        reg=float(self.beta*np.sum(denom-self.delta))
        return dict(folder=folder,summary=summary,data=row['weighted_objective_after'],reg=reg,
            value=(row['weighted_objective_after']+reg)*self.scale,
            reg_gradient=self.beta*d/denom*self.scale)

    def get(self,theta):
        key=self.key(theta)
        if key not in self.cache:
            name=hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
            folder=model.prepare_case(f'permeability_evaluations/k_{name}',self.geometry['center_ft'],
                self.geometry['width_ft'],self.geometry['h_y_max_m'],theta=list(key))
            model.run_case(folder)
            self.cache[key]=self.read(folder,key)
        return self.cache[key]

    def gradient(self,theta,h):
        stencils=[stencil(theta,j,h,self.bounds) for j in range(2)]
        unique={self.key(p):p for points,_ in stencils for p in points}
        missing=[p for key,p in unique.items() if key not in self.cache]
        if missing:
            with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(self.get,missing))
        g=np.array([sum(float(c)*self.get(p)['data'] for p,c in zip(points,coeff))
            for points,coeff in stencils])*self.scale
        return g

    def evaluate(self,theta):
        key=self.key(theta)
        value=self.get(theta)
        if key not in self.gradients:
            model.base.status(RUN/'status.json','evaluating_gradient',theta=list(key),accepted_iterations=self.accepted)
            data_g=self.gradient(theta,.02)
            self.gradients[key]=data_g+value['reg_gradient']
            record=dict(theta=list(key),coupling_C=value['summary']['results'][0]['fitted_C'],
                scaled_objective=value['value'],data_objective=value['data'],regularization=value['reg'],
                gradient_scaled=self.gradients[key].tolist(),data_gradient_scaled=data_g.tolist(),
                forward_folder=str(value['folder']))
            with (RUN/'evaluations.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print('EVALUATION',json.dumps(record),flush=True)
        return value['value'],self.gradients[key]

    def validate(self):
        model.base.status(RUN/'status.json','two_direction_step_checks')
        g=self.gradient(self.theta0,.02);g_half=self.gradient(self.theta0,.01)
        checks=[]
        for j in range(2):
            tol=max(1e-7,.05*max(abs(g[j]),abs(g_half[j])))
            checks.append(dict(direction=np.eye(2)[j].tolist(),h=.02,h_half=.01,
                derivative_h=float(g[j]),derivative_half_h=float(g_half[j]),
                absolute_difference=float(abs(g[j]-g_half[j])),tolerance=float(tol),
                passed=bool(abs(g[j]-g_half[j])<=tol),
                note='FD step consistency, not an adjoint-vs-FD validation'))
        model.base.save(RUN/'gradient_check.json',checks)
        print('GRADIENT_CHECK',json.dumps(checks),flush=True)
        if not all(c['passed'] for c in checks):
            raise RuntimeError('Finite-difference derivative step check failed; optimization not started')
        self.gradients[self.key(self.theta0)]=g+self.get(self.theta0)['reg_gradient']
        (RUN/'GRADIENT_CHECK_PASSED').write_text('Both permeability FD directions stable at h=.02 and .01\n')

    def record(self,theta):
        self.accepted+=1
        assert self.accepted<=60
        value=self.get(theta);row=value['summary']['results'][0]
        record=dict(iteration=self.accepted,alpha_xx=float(theta[0]),alpha_yy=float(theta[1]),
            coupling_C=row['fitted_C'],scaled_objective=value['value'],
            weighted_rms_microstrain=row['weighted_rms_after_microstrain'],
            core_rms_microstrain=row['core_rms_after_microstrain'],forward_folder=str(value['folder']))
        history=RUN/'accepted_iterations.csv'
        pd.DataFrame([record]).to_csv(history,index=False,mode='a',header=not history.exists())
        model.base.save(RUN/'checkpoint.json',record)
        np.savetxt(RUN/'checkpoint_theta.txt',np.r_[theta,row['fitted_C']])
        shutil.copy2(value['folder']/'qc_bg_18_only.png',RUN/'qc_latest.png')
        model.base.status(RUN/'status.json','accepted_iteration',**record)
        print('ACCEPTED',json.dumps(record),flush=True)

    def run(self):
        shutil.copy2(self.initial_folder/'qc_bg_18_only.png',RUN/'qc_initial.png')
        self.validate()
        result=minimize(self.evaluate,self.theta0,jac=True,method='L-BFGS-B',bounds=self.bounds,
            callback=self.record,options=self.options)
        value=self.get(result.x);row=value['summary']['results'][0]
        limited=int(result.status)==1 and int(result.nit)==60
        record=dict(success=bool(result.success),status=int(result.status),message=str(result.message),
            iterations=int(result.nit),optimizer_evaluations=int(result.nfev),
            unique_physical_evaluations=len(self.cache),theta=np.r_[result.x,row['fitted_C']].tolist(),
            scaled_objective=float(result.fun),gradient_scaled=np.asarray(result.jac).tolist(),
            final_forward_folder=str(value['folder']),results=row,
            termination='iteration_limit' if limited else 'converged' if result.success else 'optimizer_failure')
        model.base.save(RUN/'optimizer_result.json',record)
        np.savetxt(RUN/'theta_final.txt',np.r_[result.x,row['fitted_C']])
        for source,target in [('bg_18_strain.csv','strain_final.csv'),('summary.json','final_evaluation.json'),
                              ('qc_bg_18_only.png','qc_final.png'),('qc_bg_18_only.pdf','qc_final.pdf')]:
            shutil.copy2(value['folder']/source,RUN/target)
        if not result.success and not limited:raise RuntimeError(str(result.message))
        (RUN/'COMPLETE').write_text(str(result.message)+'\n')
        model.base.status(RUN/'status.json','complete',**record)


if __name__=='__main__':
    verify_stencils()
    if RUN.exists():raise RuntimeError('Refusing duplicate inversion')
    RUN.mkdir()
    (RUN/'STARTED').write_text(f'pid={os.getpid()}\n')
    try:Inversion().run()
    except Exception as exc:
        model.base.status(RUN/'status.json','failed',reason=f'{type(exc).__name__}: {exc}')
        raise
