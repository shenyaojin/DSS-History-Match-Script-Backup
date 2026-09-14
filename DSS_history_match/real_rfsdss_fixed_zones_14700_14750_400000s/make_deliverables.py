from pathlib import Path
import importlib.util,json,re,shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/'fig_deliverable';OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('run_module',HERE/'run_inversion.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
names=['unrectified','half_wave_positive'];labels=['Unrectified','Positive half-wave']
results=[];summaries=[];table=[]
for name in names:
    p=HERE/name;cfg=json.loads((p/'config.json').read_text());h=pd.read_csv(p/'objective_history.csv')
    theta=np.loadtxt(p/'best_theta.txt');final=np.loadtxt(p/'optimized_theta.txt')
    assert np.array_equal(theta,final)
    objective=h[h.purpose.isin(['initial','optimize','final'])].scaled_objective.min()
    assert np.isclose(objective,h.iloc[-1].scaled_objective,rtol=1e-12)
    model=mod.Inversion(p,1);source=model.find_report();df=pd.read_csv(source)
    assert len(df)==cfg['n_time']*cfg['n_channels']
    assert np.allclose((df.simulation_values-df.measurement_values)[df.measurement_time>0],df.misfit_values[df.measurement_time>0],rtol=1e-8,atol=1e-18)
    old=p/'strain_final.csv'
    shutil.copy2(source,old)
    mod.plot_qc(p,old,theta,'final')
    ts=np.sort(df.measurement_time.unique());ys=np.sort(df.measurement_ycoord.unique());md=14725+(ys-50)/.3048
    grid=lambda col:df.pivot(index='measurement_ycoord',columns='measurement_time',values=col).reindex(index=ys,columns=ts).to_numpy()*1e6
    obs=grid('measurement_values');sim=grid('simulation_values')
    if name=='half_wave_positive':assert np.all(sim>=0) and np.all(obs>=0)
    results.append(dict(cfg=cfg,theta=theta,time=ts,md=md,obs=obs,sim=sim,h=h))
    for i,z in enumerate(cfg['zones']):
        table.append(dict(case=name,zone=i+1,lo_ft=z['lo_ft'],hi_ft=z['hi_ft'],initial_alpha=cfg['initial_theta'][i],best_alpha=theta[i],permeability_m2=10**theta[i]))
    baseline=h[h.purpose=='initial'].iloc[0];best=h[h.purpose=='final'].iloc[-1]
    summaries.append(dict(case=name,source=str(source.relative_to(HERE)),J_initial=float(baseline.data_objective),J_best=float(best.data_objective),
                          reduction_percent=100*(1-best.data_objective/baseline.data_objective),L1=float(best.regularization),
                          L1_percent_of_data=100*best.regularization/best.data_objective,rmse_microstrain=float(np.sqrt(np.mean((sim[:,1:]-obs[:,1:])**2))),
                          initial_frame_energy=float(.5*np.sum((sim[:,0]-obs[:,0])**2)*1e-12)))

plt.rcParams.update({'font.size':11})
fig,axs=plt.subplots(2,2,figsize=(15,12),sharex=True,sharey=True,constrained_layout=True)
for row,(r,label) in enumerate(zip(results,labels)):
    for col,(values,title) in enumerate([(r['obs'],'Observed DSS'),(r['sim'],'Simulation at best α')]):
        ax=axs[row,col];im=ax.pcolormesh(r['time'],r['md'],values,cmap='bwr',vmin=-.5,vmax=.5,shading='auto',rasterized=True)
        ax.axvspan(r['time'][-1],400000,color='0.8',alpha=.5,hatch='//')
        ax.set(title=f'{label} | {title}',xlim=(0,400000),ylim=(14750,14700),xlabel='Time [s]')
        if col==0:ax.set_ylabel('Measured depth [ft]')
fig.colorbar(im,ax=axs.ravel().tolist(),label='Strain change [με]',extend='both',shrink=.75)
fig.suptitle('POW-S RFS-DSS | Observation vs best-parameter simulation\n14700–14750 ft · 0–400,000 s · same color limits for all panels',fontsize=16)
fig.savefig(OUT/'observed_vs_best_simulation.png',dpi=160);fig.savefig(OUT/'observed_vs_best_simulation.pdf');plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(14,12),sharey=True,constrained_layout=True,gridspec_kw={'width_ratios':[3,1.25]})
for row,(r,label) in enumerate(zip(results,labels)):
    ax,ap=axs[row];im=ax.pcolormesh(r['time'],r['md'],r['obs'],cmap='bwr',vmin=-.5,vmax=.5,shading='auto',rasterized=True)
    ax.set(title=f'{label} | Observed DSS',xlabel='Time [s]',ylabel='Measured depth [ft]',xlim=(0,400000),ylim=(14750,14700))
    zones=r['cfg']['zones'];edges=sorted({14700.,14750.,*[z[k] for z in zones for k in ['lo_ft','hi_ft']]})
    for vals,style,color,leg in [(r['cfg']['initial_theta'],'--','0.5','Initial'),(r['theta'],'-','#172b4d','Best fit')]:
        xp,yp=[],[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            mid=(lo+hi)/2
            value=next((vals[i] for i,z in enumerate(zones) if z['lo_ft']<=mid<z['hi_ft']),-18.)
            xp.extend([10**value]*2);yp.extend([lo,hi])
        ap.plot(xp,yp,style,color=color,lw=1.8,label=leg)
    ap.set_xscale('log');ap.set(xlabel='Permeability k [m²]',title='Fixed-zone permeability',xlim=(5e-19,3e-14));ap.legend(fontsize=9);ap.grid(axis='x',alpha=.25)
    for z in zones:
        ap.axhspan(z['lo_ft'],z['hi_ft'],alpha=.07,color='#bc7b1f')
fig.colorbar(im,ax=axs[:,0].tolist(),label='Strain change [με]',extend='both',shrink=.7)
fig.suptitle('POW-S RFS-DSS | Observations and inferred permeability\nZone positions and widths fixed; five permeability values optimized',fontsize=16)
fig.savefig(OUT/'observed_and_inferred_permeability.png',dpi=160);fig.savefig(OUT/'observed_and_inferred_permeability.pdf');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
for r,label,color in zip(results,labels,['#2864a4','#b85c1e']):
    case='half_wave_positive' if r['cfg']['rectified'] else 'unrectified'
    track=pd.read_csv(HERE/case/'accepted_iteration_history.csv')
    track['evaluation_index']=track['iteration'];track['relative_misfit']=track.data_objective/track.data_objective.iloc[0]
    track.to_csv(OUT/f'misfit_curve_{r["cfg"]["rectified"]}.csv',index=False)
    x=track.evaluation_index.to_numpy();v=track.data_objective.to_numpy()
    axs[0].plot(x,v,'o-',ms=3,color=color,label=label)
    axs[1].plot(x,v/v[0],'o-',ms=3,color=color,label=label)
    axs[1].annotate(f'−{100*(1-v[-1]/v[0]):.2f}%',(x[-1],v[-1]/v[0]),xytext=(-50,10),textcoords='offset points',color=color)
for ax in axs:
    ax.set_xlabel('Accepted optimizer iteration (0 = initial)');ax.grid(alpha=.25);ax.legend()
axs[0].set(title='Data misfit: Jdata = ½ Σ residual²',ylabel='Jdata [strain²]')
axs[1].set(title='Reduction relative to each case’s initial misfit',ylabel='Jdata / Jdata,initial')
fig.suptitle('Misfit history | accepted L-BFGS-B iterations',fontsize=14)
fig.savefig(OUT/'misfit_history.png',dpi=180);fig.savefig(OUT/'misfit_history.pdf');plt.close(fig)
pd.DataFrame(table).to_csv(OUT/'best_permeability.csv',index=False)
(OUT/'verification.json').write_text(json.dumps(summaries,indent=2)+'\n')
print(json.dumps(summaries,indent=2))
