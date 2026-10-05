"""Field/V6 driver: independent xx, yy and a global positive attenuation C.

MOOSE evaluates F=0.5*||s-d/C||² with its existing rectified-strain adjoint.
Return J_data=C²*F, grad_alpha=C²*grad_alpha(F), grad_C=(Cs-d) dot s.
The original observation CSV is immutable. Only a private solver target changes.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import re
import shutil
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from fiberis.moose.runner import MooseRunner

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = HERE/'inversion'


def save_json(path, value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def status(stage, **details):
    save_json(RUN/'status.json',dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**details))


def plot_qc(folder, theta, label):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cfg=json.loads((RUN/'config.json').read_text())
    frame=pd.read_csv(folder/'strain.csv');signed=pd.read_csv(folder/'strain_signed.csv')
    times=np.sort(frame.measurement_time.unique());ys=np.sort(frame.measurement_ycoord.unique())
    md=cfg['md_center_ft']+(ys-cfg['y_center_m'])/.3048
    def matrix(df,column):
        return df.pivot(index='measurement_ycoord',columns='measurement_time',values=column).reindex(index=ys,columns=times).to_numpy()*1e6
    obs=matrix(frame,'measurement_values');sim=matrix(frame,'simulation_values');raw=matrix(signed,'simulation_values')
    native=np.load(HERE/'data/half_wave_positive_native.npz')
    zone=cfg['zones'][0];lo,hi=zone['lo_ft'],zone['hi_ft']
    core=np.where((md>=lo)&(md<hi))[0]
    shown=np.r_[core[0]-1,core,core[-1]+1]
    fig=plt.figure(figsize=(20,10),layout='constrained')
    grid=fig.add_gridspec(7,4,width_ratios=[2,2,.95,2.6],height_ratios=[1]*6+[.25])
    axes=[fig.add_subplot(grid[:6,i]) for i in range(3)]
    cmap=LinearSegmentedColormap.from_list('positive',['white','#ffaaaa','#e53935','#8b0000'])
    vmax=max(3.,float(sim.max()))
    for ax,values,tt,title in zip(axes[:2],[native['data']*1e6,sim],[native['taxis'],times],['Observed positive DSS (native)','C × simulated positive DSS (130 times)']):
        im=ax.pcolormesh(tt/1000,md,values,shading='nearest',cmap=cmap,vmin=0,vmax=vmax,rasterized=True)
        ax.set(title=title,xlabel='Time [1000 s]',xlim=(0,400),ylim=(14379,14373))
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        ax.axvspan(times[-1]/1000,400,color='.9')
        ax.axhline(lo,color='.3',ls='--',lw=.7);ax.axhline(hi,color='.3',ls='--',lw=.7)
    axes[0].set_ylabel('Measured depth [ft]');axes[1].tick_params(labelleft=False)
    depths=[14373,lo,lo,hi,hi,14379]
    for value,color,name in zip(theta[:2],['#d97d00','#155f9b'],['xx','yy']):
        axes[2].plot([-21,-21,value,value,-21,-21],depths,color=color,lw=2,label=name)
    axes[2].set(title='Permeability',xlabel='log10(k/m²)',ylim=(14379,14373),xlim=(-21.5,-11.5),xticks=[-21,-18,-15,-12])
    axes[2].tick_params(labelleft=False);axes[2].legend(fontsize=8);axes[2].grid(axis='x',alpha=.2)
    first=None;handles=[]
    low=min(-.3,float(raw[shown].min())*1.1);high=max(3.,float(sim[shown].max())*1.1)
    for i,j in enumerate(shown):
        ax=fig.add_subplot(grid[i,3],sharex=first)
        if first is None:first=ax
        ax.plot(native['taxis']/1000,native['data'][j]*1e6,color='.75',lw=.5)
        a,=ax.plot(times/1000,obs[j],color='black',lw=1,label='Observation (+)')
        b,=ax.plot(times/1000,sim[j],color='#d97d00',lw=1.3,label='C × simulation (+)')
        c,=ax.plot(times/1000,raw[j],color='#168b98',lw=.8,ls='--',label='C × simulation signed')
        handles=[a,b,c]
        ax.axhline(0,color='.6',lw=.4)
        ax.set(ylim=(low,high),xlim=(0,400))
        ax.text(.015,.88,f'{md[j]:.3f} ft',transform=ax.transAxes,va='top',fontsize=9)
        ax.grid(alpha=.15)
        if i==2:ax.set_ylabel('Strain [microstrain]')
        if i<5:ax.tick_params(labelbottom=False)
        else:ax.set_xlabel('Time [1000 s]')
    cax=fig.add_subplot(grid[6,:2]);fig.colorbar(im,cax=cax,orientation='horizontal',label='Positive strain [microstrain]')
    legend=fig.add_subplot(grid[6,2:]);legend.axis('off');legend.legend(handles=handles,loc='center',ncol=3,fontsize=9)
    fig.suptitle(f'S-well 14376 ft | {label} | alpha_xx={theta[0]:.6f}, alpha_yy={theta[1]:.6f}, C={theta[2]:.6g}\n'
                'Fixed zone and background; native-depth mesh; right: individual channel fits',fontsize=14)
    fig.savefig(RUN/f'qc_{label}.png',dpi=160);plt.close(fig)


class Inversion:
    def __init__(self):
        self.cfg=json.loads((RUN/'config.json').read_text())
        self.theta0=np.asarray(self.cfg['initial_theta'])
        self.indices=self.cfg['zone_layer_indices'][0]
        self.master=(RUN/'optimize.i').read_text()
        self.output=RUN/'solver_output';self.output.mkdir(exist_ok=True)
        self.runner=MooseRunner(str(ROOT/'moose_env/moose/modules/combined/combined-opt'),
                               '/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec')
        self.count=0;self.cache={};self.artifacts={}
        self.observations=pd.read_csv(RUN/'measurement_data.csv')

    def expand(self,theta):
        alpha=np.full(200,self.cfg['background_alpha'])
        alpha[self.indices]=theta[0]
        return np.r_[alpha,theta[1]]

    def regularization(self,theta):
        d=np.asarray(theta[:2])-self.cfg['regularization_reference_alpha']
        delta=self.cfg['regularization_delta'];denom=np.sqrt(d*d+delta*delta)
        beta=self.cfg['beta_l1']
        return float(beta*np.sum(denom-delta)),np.r_[beta*d/denom,0.]

    def reports(self,obj):
        def latest(pattern):
            files=list(self.output.glob(pattern))
            return max(files,key=lambda p:int(re.search(r'_(\d+)\.csv$',p.name)[1]))
        # The signed reporter name also ends in _data_; select the exact data suffix.
        files=[p for p in self.output.glob('*_forward0_*_data_*.csv') if '_signed_data_' not in p.name]
        report=max(files,key=lambda p:int(re.search(r'_data_(\d+)\.csv$',p.name)[1]))
        signed_report=latest('*_signed_data_*.csv')
        frame=pd.read_csv(report);raw=pd.read_csv(signed_report)
        supplied=pd.read_csv(RUN/'solver_measurement_data.csv')
        assert len(frame)==self.cfg['n_channels']*self.cfg['n_time'] and frame.measurement_time.nunique()==130
        keys=['measurement_time','measurement_xcoord','measurement_ycoord','measurement_zcoord','measurement_values']
        np.testing.assert_allclose(frame[keys],supplied[keys],rtol=1e-10,atol=1e-18)
        np.testing.assert_allclose(raw[keys],frame[keys],rtol=1e-10,atol=1e-18)
        assert np.isfinite(raw.simulation_values).all() and np.isfinite(frame.simulation_values).all()
        np.testing.assert_allclose(frame.simulation_values,np.maximum(raw.simulation_values,0),rtol=1e-9,atol=1e-16)
        residual=np.where(frame.measurement_time>0,frame.simulation_values-frame.measurement_values,0.)
        np.testing.assert_allclose(frame.misfit_values,residual,rtol=1e-9,atol=1e-18)
        assert np.isclose(.5*np.sum(residual**2),obj,rtol=1e-8,atol=1e-22)
        return report,signed_report

    def evaluate(self,theta,purpose='optimize'):
        key=tuple(theta)
        if key in self.cache:return self.cache[key]
        coupling=float(theta[2])
        if not np.isfinite(theta).all() or coupling<=0:
            raise ValueError('Parameters must be finite and C strictly positive')
        self.count+=1
        status(purpose,evaluation=self.count,theta=np.asarray(theta).tolist())
        print(f'EVALUATION {self.count} purpose={purpose} theta={theta}',flush=True)
        full=self.expand(theta)
        text,count=re.subn(r"(initial_condition\s*=\s*')[^']*(')",
            lambda m:m[1]+'; '.join(f'{v:.16g}' for v in full)+m[2],self.master,count=1)
        assert count==1 and len(full)==201
        text=text.replace('verbose = true','verbose = false').replace('console = true','console = false')
        path=RUN/'evaluate.i';path.write_text(text)
        supplied=self.observations.copy()
        supplied['measurement_values']/=coupling
        supplied.to_csv(RUN/'solver_measurement_data.csv',index=False)
        for old in self.output.glob('*.csv'):old.unlink()
        start=time.monotonic()
        ok,stdout,stderr=self.runner.run(str(path),str(self.output),num_processors=20,
            log_file_name='simulation.log',stream_output=False,clean_output_dir=False)
        (RUN/'last_solver_stderr.txt').write_text(stderr)
        if not ok:raise RuntimeError('MOOSE failed; no objective/gradient returned')
        moose_obj=float(pd.read_csv(self.output/'evaluate_out.csv')['OptimizationReporter/objective_value'].iloc[-1])
        report,raw=self.reports(moose_obj)
        frame=pd.read_csv(report);signed=pd.read_csv(raw)
        # Store physical and instrument-predicted strain separately, with original d.
        physical=frame.copy();physical_signed=signed.copy()
        observed=self.observations.measurement_values.to_numpy()
        use=frame.measurement_time.to_numpy()>0
        simulation=frame.simulation_values.to_numpy().copy()
        for df in [physical,physical_signed,frame,signed]:
            df['measurement_values']=observed
        frame['simulation_values']*=coupling
        signed['simulation_values']*=coupling
        for df in [physical,physical_signed,frame,signed]:
            df['misfit_values']=np.where(use,df.simulation_values-observed,0.)
        residual=frame.misfit_values.to_numpy()
        obj=float(.5*np.dot(residual,residual))
        np.testing.assert_allclose(obj,coupling**2*moose_obj,rtol=1e-8,atol=1e-22)
        gf=pd.read_csv(self.output/'evaluate_out_OptimizationReporter_0001.csv').iloc[-1]
        gx=gf[[f'grad_perm_{i+1}' for i in range(200)]].to_numpy(dtype=float)
        data_g=np.array([coupling**2*gx[self.indices].sum(),
                         coupling**2*float(gf['grad_kyy_A']),
                         float(np.dot(residual[use],simulation[use]))])
        reg,reg_g=self.regularization(theta);scale=self.cfg['objective_scale']
        value=(obj+reg)*scale;gradient=(data_g+reg_g)*scale
        assert np.isfinite(value) and np.isfinite(gradient).all()
        folder=RUN/'evaluations'/f'{self.count:04d}';folder.mkdir(parents=True)
        frame.to_csv(folder/'strain.csv',index=False)
        signed.to_csv(folder/'strain_signed.csv',index=False)
        physical.to_csv(folder/'strain_physical.csv',index=False)
        physical_signed.to_csv(folder/'strain_physical_signed.csv',index=False)
        supplied.to_csv(folder/'solver_measurement_data.csv',index=False)
        np.savetxt(folder/'gradient_scaled.txt',gradient);np.savetxt(folder/'data_gradient_unscaled.txt',data_g)
        ev=dict(evaluation=self.count,purpose=purpose,theta=np.asarray(theta).tolist(),
            data_objective=obj,moose_target_objective=moose_obj,coupling_C=coupling,
            regularization=reg,scaled_objective=float(value),
            gradient_scaled=gradient.tolist(),data_gradient_unscaled=data_g.tolist(),wall_seconds=time.monotonic()-start)
        save_json(folder/'evaluation.json',ev)
        row={k:ev[k] for k in ['evaluation','purpose','data_objective','regularization','scaled_objective','wall_seconds']}
        row.update(alpha_xx=float(theta[0]),alpha_yy=float(theta[1]),coupling_C=coupling,
                   gradient_norm=float(np.linalg.norm(gradient)))
        hist=RUN/'objective_history.csv';pd.DataFrame([row]).to_csv(hist,index=False,mode='a',header=not hist.exists())
        self.cache[key]=(float(value),gradient);self.artifacts[key]=folder
        print(json.dumps(row),flush=True)
        return self.cache[key]

    def record(self,theta,iteration):
        source=self.artifacts[tuple(theta)];target=RUN/'iterations'/f'{iteration:02d}'
        shutil.copytree(source,target)
        ev=json.loads((source/'evaluation.json').read_text())
        df=pd.read_csv(source/'strain.csv');use=df.measurement_time>0
        row=dict(iteration=iteration,evaluation=ev['evaluation'],alpha_xx=float(theta[0]),alpha_yy=float(theta[1]),coupling_C=float(theta[2]),
            data_objective=ev['data_objective'],regularization=ev['regularization'],scaled_objective=ev['scaled_objective'],
            residual_rms_microstrain=float(np.sqrt(np.mean(df.loc[use,'misfit_values']**2))*1e6))
        hist=RUN/'accepted_iteration_history.csv';pd.DataFrame([row]).to_csv(hist,index=False,mode='a',header=not hist.exists())
        np.savetxt(RUN/'checkpoint_theta.txt',theta)
        plot_qc(source,theta,f'iter_{iteration:02d}')
        print('ACCEPTED_ITERATION '+json.dumps(row),flush=True)

    def validate(self):
        initial=json.loads((self.artifacts[tuple(self.theta0)]/'evaluation.json').read_text())
        checks=[];h=.02;scale=self.cfg['objective_scale']
        for direction in np.eye(3)[:2]:
            values=[]
            for sign in [1,-1]:
                theta=self.theta0+sign*h*direction
                self.evaluate(theta,'gradient_check')
                values.append(json.loads((self.artifacts[tuple(theta)]/'evaluation.json').read_text())['data_objective']*scale)
            fd=(values[0]-values[1])/(2*h)
            adj=float(np.asarray(initial['data_gradient_unscaled'])@direction*scale)
            checks.append(dict(direction=direction.tolist(),h=h,fd=fd,adjoint=adj,
                relative_error=abs(fd-adj)/max(abs(fd),abs(adj),1e-8),
                passed=bool(abs(fd-adj)<=max(1e-7,.05*max(abs(fd),abs(adj))))))
            save_json(RUN/'gradient_check.json',checks)
            print('GRADIENT_CHECK '+json.dumps(checks[-1]),flush=True)
            if not checks[-1]['passed']:raise RuntimeError('Directional gradient check failed; inversion not started')
        # C changes the observation mapping, never the PDE solution; its finite
        # difference is exact using the same frozen physical forward response.
        frame=pd.read_csv(self.artifacts[tuple(self.theta0)]/'strain_physical.csv')
        use=frame.measurement_time>0
        s=frame.loc[use,'simulation_values'].to_numpy()
        d=frame.loc[use,'measurement_values'].to_numpy()
        c=float(self.theta0[2]);hc=min(1e-4,c*.1)
        plus=.5*np.sum(((c+hc)*s-d)**2)*scale
        minus=.5*np.sum(((c-hc)*s-d)**2)*scale
        fd=(plus-minus)/(2*hc);analytic=initial['data_gradient_unscaled'][2]*scale
        checks.append(dict(direction=[0,0,1],h=hc,fd=float(fd),adjoint=float(analytic),
                           method='Frozen forward, exact because C is absent from PDE',
                           absolute_error=float(abs(fd-analytic)),
                           passed=bool(abs(fd-analytic)<=max(1e-7,.05*max(abs(fd),abs(analytic))))))
        save_json(RUN/'gradient_check.json',checks)
        print('GRADIENT_CHECK '+json.dumps(checks[-1]),flush=True)
        if not checks[-1]['passed']:raise RuntimeError('Coupling gradient check failed; inversion not started')
        (RUN/'GRADIENT_CHECK_PASSED').write_text('Two permeability directional checks and coupling derivative check passed\n')

    def optimize(self):
        self.evaluate(self.theta0,'coupling_initialization')
        bootstrap=self.artifacts[tuple(self.theta0)]
        frame=pd.read_csv(bootstrap/'strain_physical.csv');use=frame.measurement_time>0
        s=frame.loc[use,'simulation_values'].to_numpy();d=frame.loc[use,'measurement_values'].to_numpy()
        if np.dot(s,s)<=0:raise RuntimeError('Zero positive initial strain; coupling is unidentifiable')
        best=float(np.dot(s,d)/np.dot(s,s))
        self.theta0[2]=np.clip(best,*self.cfg['coupling_bounds'])
        save_json(RUN/'coupling_initialization.json',dict(unconstrained_C=best,
            initial_theta=self.theta0.tolist(),bootstrap_evaluation=str(bootstrap.relative_to(RUN)),
            bounds=self.cfg['coupling_bounds']))
        self.evaluate(self.theta0,'initial');self.record(self.theta0,0)
        self.validate()
        accepted=0
        def callback(theta):
            nonlocal accepted
            accepted+=1
            assert accepted<=60
            self.record(theta,accepted)
        result=minimize(self.evaluate,self.theta0,jac=True,method='L-BFGS-B',
            bounds=[tuple(b) for b in self.cfg['parameter_bounds']],callback=callback,
            options={key:self.cfg['optimizer'][key] for key in ['maxiter','ftol','gtol','maxls']})
        limited=int(result.status)==1 and int(result.nit)==60
        record=dict(success=bool(result.success),status=int(result.status),message=str(result.message),
            iterations=int(result.nit),optimizer_evaluations=int(result.nfev),total_evaluations=self.count,
            theta=result.x.tolist(),objective=float(result.fun),
            coupling_C=float(result.x[2]),gradient_scaled=np.asarray(result.jac).tolist(),
            coupling_at_bound=bool(any(np.isclose(result.x[2],b,rtol=0,atol=1e-10) for b in self.cfg['coupling_bounds'])),
            termination='user_iteration_limit' if limited else 'converged' if result.success else 'optimizer_failure')
        save_json(RUN/'optimizer_result.json',record)
        folder=self.artifacts[tuple(result.x)]
        for name,target in [('strain.csv','strain_final.csv'),('strain_signed.csv','strain_signed_final.csv'),
                            ('strain_physical.csv','strain_physical_final.csv'),
                            ('strain_physical_signed.csv','strain_physical_signed_final.csv'),
                            ('evaluation.json','final_evaluation.json')]:
            shutil.copy2(folder/name,RUN/target)
        np.savetxt(RUN/'theta_final.txt',result.x)
        plot_qc(folder,result.x,'final')
        zone=self.cfg['zones'][0]
        pd.DataFrame([dict(lo_ft=zone['lo_ft'],hi_ft=zone['hi_ft'],alpha_xx=result.x[0],alpha_yy=result.x[1],
            kxx_m2=10**result.x[0],kyy_m2=10**result.x[1],kzz_m2=1e-21,coupling_C=result.x[2])]).to_csv(RUN/'permeability_final.csv',index=False)
        if not result.success and not limited:raise RuntimeError('Optimizer failed: '+str(result.message))
        (RUN/'COMPLETE').write_text(str(result.message)+'\n')
        status('complete',**record)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-input',action='store_true')
    args=parser.parse_args()
    os.environ['PATH']='/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin:'+os.environ['PATH']
    audit=json.loads((RUN/'preflight.json').read_text())
    for name,digest in audit['input_sha256'].items():
        assert hashlib.sha256((RUN/name).read_bytes()).hexdigest()==digest,name
    if (RUN/'RUNNING').exists() or (RUN/'objective_history.csv').exists():
        raise RuntimeError('Run already started; refusing duplicate or overwrite')
    model=Inversion()
    np.testing.assert_array_equal(model.expand(model.theta0),np.loadtxt(RUN/'initial_alpha.txt'))
    if args.check_input:
        ok,stdout,stderr=model.runner.run(str(RUN/'optimize.i'),str(RUN/'input_check'),num_processors=20,
            additional_args=['--check-input'],log_file_name='check.log',stream_output=False,clean_output_dir=False)
        (RUN/'input_check/stderr.log').write_text(stderr)
        if not ok:raise RuntimeError('Input check failed: '+stderr[-4000:])
        (RUN/'INPUT_CHECK_PASSED').write_text('MOOSE input check passed\n');return
    if not (RUN/'INPUT_CHECK_PASSED').exists():raise RuntimeError('Input check required before solve')
    with (RUN/'RUNNING').open('x') as f:f.write(f'pid={os.getpid()}\n')
    save_json(RUN/'launch_manifest.json',dict(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=audit['input_sha256'],
        authorization='User approved 14376-ft xx/yy=-17 and global attenuation C, then requested one inversion',
        mpi_ranks=20,gradient_check_required=True,maxiter=60))
    try:model.optimize()
    except Exception as exc:
        (RUN/'FAILED').write_text(f'{type(exc).__name__}: {exc}\n');status('failed',reason=str(exc));raise
    finally:(RUN/'RUNNING').unlink(missing_ok=True)


if __name__=='__main__':
    main()
