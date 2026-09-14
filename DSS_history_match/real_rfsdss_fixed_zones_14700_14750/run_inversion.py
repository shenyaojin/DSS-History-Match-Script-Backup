from pathlib import Path
import argparse, json, os, re, shutil, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from fiberis.moose.runner import MooseRunner

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]

def plot_qc(run, strain_csv, theta, label):
    import matplotlib.pyplot as plt
    cfg=json.loads((run/'config.json').read_text())
    df=pd.read_csv(strain_csv)
    time_values=np.sort(df.measurement_time.unique())
    ys=np.sort(df.measurement_ycoord.unique())
    md=14725+(ys-50)/.3048
    obs=df.pivot(index='measurement_ycoord',columns='measurement_time',values='measurement_values').reindex(index=ys,columns=time_values).to_numpy()*1e6
    sim=df.pivot(index='measurement_ycoord',columns='measurement_time',values='simulation_values').reindex(index=ys,columns=time_values).to_numpy()*1e6
    fig,axs=plt.subplots(1,4,figsize=(19,7),sharey=True,constrained_layout=True,gridspec_kw={'width_ratios':[2,2,2,1]})
    for ax,values,title in zip(axs[:3],[obs,sim,sim-obs],['Observation','Simulation (same transform)','Residual: simulation − observation']):
        im=ax.pcolormesh(time_values,md,values,shading='auto',cmap='bwr',vmin=-.5,vmax=.5,rasterized=True)
        ax.set(title=title,xlabel='Time [s]',ylim=(14750,14700),xlim=(0,40000))
    axs[0].set_ylabel('Measured depth [ft]')
    fig.colorbar(im,ax=axs[:3],label='Strain [με]',extend='both',shrink=.75)
    zones=cfg['zones'];edges=sorted({14700.,14750.,*[z[k] for z in zones for k in ['lo_ft','hi_ft']]})
    for label_profile,values,style in [('Initial',cfg['initial_theta'],'--'),(label,theta,'-')]:
        xp,yp=[],[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            mid=(lo+hi)/2
            value=next((values[i] for i,z in enumerate(zones) if z['lo_ft']<=mid<z['hi_ft']),-18.)
            xp.extend([value,value]);yp.extend([lo,hi])
        axs[3].plot(xp,yp,style,label=label_profile)
    axs[3].set(xlabel='α = log10(k / m²)',title='Fixed-zone permeability',xlim=(-18.5,-11.5))
    axs[3].legend(fontsize=8)
    fig.suptitle(f'POW-S | {run.name} | {label} | fixed geometry, 5 permeability parameters',fontsize=15)
    fig.savefig(run/f'qc_{label}.png',dpi=150);plt.close(fig)

class Inversion:
    def __init__(self,run,np_count):
        self.run=run;self.cfg=json.loads((run/'config.json').read_text())
        self.theta0=np.array(self.cfg['initial_theta']);self.indices=self.cfg['zone_layer_indices']
        self.master=(run/'optimize.i').read_text();self.np=np_count;self.count=0;self.cache={}
        self.runner=MooseRunner(str(REPO/'moose_env/moose/modules/combined/combined-opt'),'/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/mpiexec')
        self.output=run/'solver_output';self.output.mkdir(exist_ok=True)
        self.best=np.inf
    def expand(self,theta):
        alpha=np.full(200,self.cfg['background_alpha'])
        for value,indices in zip(theta,self.indices):alpha[indices]=value
        return alpha
    def find_report(self):
        # Adjoint replay writes files in reverse time order. Modification time
        # therefore selects t=0, not the full forward observation vector.
        files=list(self.output.glob('*_forward0_*_data_*.csv'))
        if not files:
            raise RuntimeError('No forward simulation observation reporter CSV found')
        report=max(files,key=lambda p:int(re.search(r'_data_(\d+)\.csv$',p.name)[1]))
        df=pd.read_csv(report)
        obj=pd.read_csv(self.output/'evaluate_out.csv')['OptimizationReporter/objective_value'].iloc[-1]
        # Reporter residuals exclude the initial frame in this transient solve.
        if not np.isclose(.5*np.sum(df.misfit_values**2),obj,rtol=1e-8,atol=1e-22):
            raise RuntimeError('Forward reporter residuals do not match the optimized objective')
        return report
    def evaluate(self,theta,purpose='optimize',force=False):
        key=tuple(theta)
        if key in self.cache and not force:return self.cache[key]
        self.count+=1
        print(f'EVALUATION {self.count} purpose={purpose} theta={theta}',flush=True)
        alpha=self.expand(theta)
        text=re.sub(r"(initial_condition\s*=\s*')[^']*(')",lambda m:m[1]+'; '.join(f'{v:.16g}' for v in alpha)+m[2],self.master,count=1)
        text=text.replace('verbose = true','verbose = false').replace('console = true','console = false')
        path=self.run/'evaluate.i';path.write_text(text)
        # Clear only previous solver CSVs in this dedicated case to prevent stale results.
        for p in sorted(self.output.glob('*.csv'),key=lambda p:p.stat().st_mtime_ns,reverse=True):p.unlink()
        started=time.monotonic()
        ok,stdout,stderr=self.runner.run(str(path),str(self.output),num_processors=self.np,log_file_name='simulation.log',stream_output=False,clean_output_dir=False)
        (self.run/'last_solver_stderr.txt').write_text(stderr)
        if not ok:raise RuntimeError('MOOSE failed; aborting rather than returning a false zero gradient')
        obj=pd.read_csv(self.output/'evaluate_out.csv')['OptimizationReporter/objective_value'].iloc[-1]
        gf=pd.read_csv(self.output/'evaluate_out_OptimizationReporter_0001.csv')
        gradient=gf[[f'grad_perm_{i+1}' for i in range(200)]].iloc[-1].to_numpy()
        reduced=np.array([gradient[ind].sum() for ind in self.indices])
        deviation=alpha+18.;denom=np.sqrt(deviation**2+.05**2)
        beta=self.cfg['beta_l1'];reg=beta*np.sum(denom-.05)
        reg_full=beta*deviation/denom
        reg_g=np.array([reg_full[ind].sum() for ind in self.indices])
        scale=self.cfg['objective_scale'];value=(obj+reg)*scale;g=(reduced+reg_g)*scale
        if not np.isfinite(value) or not np.all(np.isfinite(g)):raise RuntimeError('Nonfinite objective/gradient')
        row=dict(evaluation=self.count,purpose=purpose,data_objective=obj,regularization=reg,scaled_objective=value,
                 gradient_norm=float(np.linalg.norm(g)),wall_seconds=time.monotonic()-started)
        row.update({f'theta_{i+1}':v for i,v in enumerate(theta)})
        hist=self.run/'objective_history.csv'
        pd.DataFrame([row]).to_csv(hist,index=False,mode='a',header=not hist.exists())
        print(json.dumps(row),flush=True)
        if purpose in ['initial','final']:
            report=self.find_report()
            shutil.copy2(report,self.run/f'strain_{purpose}.csv')
            plot_qc(self.run,self.run/f'strain_{purpose}.csv',theta,purpose)
        if purpose in ['initial','optimize','final'] and value<self.best:
            self.best=value;np.savetxt(self.run/'best_theta.txt',theta)
        self.cache[key]=(value,g)
        return value,g
    def validate(self):
        value,g=self.evaluate(self.theta0,'initial')
        checks=[]
        # Two independent deterministic directions exercise all five zones,
        # including both fracture and SRV classes, against central differences.
        for direction in [np.array([1.,-.7,.4,-.9,.6]),np.array([-.4,.5,1.,.8,-.6])]:
            direction/=np.linalg.norm(direction);h=.02
            plus,_=self.evaluate(self.theta0+h*direction,'gradient_check')
            minus,_=self.evaluate(self.theta0-h*direction,'gradient_check')
            fd=(plus-minus)/(2*h);adj=float(g@direction)
            relative=abs(fd-adj)/max(abs(fd),abs(adj),1e-8)
            checks.append(dict(fd=fd,adjoint=adj,relative_error=relative,
                               passed=bool(abs(fd-adj)<=max(1e-7,.05*max(abs(fd),abs(adj))))))
        (self.run/'gradient_check.json').write_text(json.dumps(checks,indent=2)+'\n')
        if not all(c['passed'] for c in checks):raise RuntimeError('Gradient check failed; full optimization NOT started')
        print('GRADIENT_CHECK_PASSED',flush=True)
    def optimize(self):
        self.validate()
        def checkpoint(theta):
            np.savetxt(self.run/'checkpoint_theta.txt',theta)
        result=minimize(self.evaluate,self.theta0,method='L-BFGS-B',jac=True,
                        bounds=[tuple(self.cfg['theta_bounds'])]*5,callback=checkpoint,
                        options=dict(maxiter=60,ftol=1e-7,gtol=1e-8,maxls=20))
        (self.run/'optimizer_result.json').write_text(json.dumps(dict(success=bool(result.success),message=str(result.message),
                          iterations=int(result.nit),evaluations=int(result.nfev),theta=result.x.tolist(),objective=float(result.fun)),indent=2)+'\n')
        self.evaluate(result.x,'final',force=True)
        np.savetxt(self.run/'optimized_theta.txt',result.x)
        np.savetxt(self.run/'optimized_alpha.txt',self.expand(result.x))
        table=[]
        for i,z in enumerate(self.cfg['zones']):
            table.append(dict(zone=i+1,lo_ft=z['lo_ft'],hi_ft=z['hi_ft'],initial_alpha=self.theta0[i],final_alpha=result.x[i],permeability_m2=10**result.x[i]))
        pd.DataFrame(table).to_csv(self.run/'permeability_results.csv',index=False)
        if not result.success:raise RuntimeError(f'Optimizer did not converge: {result.message}; candidate result retained')
        (self.run/'COMPLETE').write_text('Optimization and final QC complete\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',required=True,choices=['unrectified','half_wave_positive'])
    parser.add_argument('--np',type=int,default=20);parser.add_argument('--check-input',action='store_true')
    args=parser.parse_args();run=HERE/args.case
    if not args.check_input and (run/'objective_history.csv').exists():raise RuntimeError('Existing run history; refusing to overwrite. Inspect before restart.')
    model=Inversion(run,args.np)
    if args.check_input:
        ok,stdout,stderr=model.runner.run(str(run/'optimize.i'),str(run/'input_check'),num_processors=args.np,
                        additional_args=['--check-input'],log_file_name='check.log',stream_output=False,clean_output_dir=False)
        if not ok:raise RuntimeError(stderr[-5000:])
        return
    model.optimize()

if __name__=='__main__':main()
