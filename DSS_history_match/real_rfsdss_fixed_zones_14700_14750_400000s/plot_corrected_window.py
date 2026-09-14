from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
out=HERE/'fig_deliverable';out.mkdir(exist_ok=True)
fig,ax=plt.subplots(3,1,figsize=(12,12),sharex=True,constrained_layout=True,gridspec_kw={'height_ratios':[2,2,1]})
for i,name in enumerate(['unrectified','half_wave_positive']):
 with np.load(HERE/name/'observations.npz',allow_pickle=True) as d:
  v=d['data']*1e6;t=d['taxis'];md=d['daxis']
 im=ax[i].pcolormesh(t,md,v,cmap='bwr',vmin=-.5,vmax=.5,shading='auto',rasterized=True)
 ax[i].set(ylim=(14750,14700),ylabel='Measured depth [ft]',title=['Drift corrected | signed strain','Drift corrected | positive half-wave'][i])
 ax[i].axvline(40000,color='black',ls='--',lw=1)
fig.colorbar(im,ax=ax[:2].tolist(),label='Strain change [με]',extend='both',shrink=.7)
p=pd.read_csv(HERE/'pressure_boundary.csv')
ax[2].plot(p.time_s,p.delta_pressure_Pa/1e6,lw=1,color='#184875')
ax[2].axvline(40000,color='black',ls='--',lw=1,label='Previous endpoint: 40,000 s')
ax[2].set(xlabel='Time from DSS start [s]',ylabel='Δp [MPa]',xlim=(0,400000),title='Pressure boundary | G1 pressure relative to DSS t=0')
ax[2].legend(loc='lower right',fontsize=9);ax[2].grid(alpha=.2)
for a in ax:
 a.axvspan(t[-1],400000,color='0.8',alpha=.5,hatch='//')
ax[2].annotate('No DSS samples',xy=((t[-1]+400000)/2,0.1),xycoords=('data','axes fraction'),rotation=90,fontsize=8,ha='center')
fig.suptitle('Corrected time window: 0–400,000 s | POW-S 14700–14750 ft',fontsize=16)
fig.savefig(out/'corrected_400000s_observations_and_source.png',dpi=150);plt.close(fig)
print('Saved corrected-window data/source figure')
