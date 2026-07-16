"""Analysis + deliverable for the T1=15:00 run: obs vs (s*MOOSE_tensile + DDM_shear),
everything interpolated onto a 15:00-anchored 4h grid and referenced to T1=15:00.
Produces the scale-sweep match figure and the image.png-style obs-vs-model waterfall.
"""
import glob, os
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
PROJ = "v1_srv_t1_1500"
OUT = REPO / "output" / PROJ
FIG = REPO / "figs" / "tensile_fault_qc" / PROJ; FIG.mkdir(parents=True, exist_ok=True)
FT = 0.3048; STAR = 10373.4
T1 = pd.Timestamp("2025-02-24 15:00"); T2 = pd.Timestamp("2025-02-28 00:00"); T3 = pd.Timestamp("2025-03-03 22:00")
OBS = REPO/"data_fervo/legacy/strain_4h_mean_profiles_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv"
SHEAR = REPO/"data_fervo/legacy/07152026_decomposed/v1_ddm_shear_strain_4h_20250224_1100_to_20250303_2200_10200_10500ft_4h_mean_T1_ref.csv"
PZ = OUT/"das_pressure_T1_1500_prepended.npz"
PROFILE_HOURS = 2.6


def rp(c):
    df = pd.read_csv(c); return df["measured_depth_ft"].to_numpy(float), pd.to_datetime(df.columns[1:]), df.iloc[:,1:].to_numpy(float)


def interp_time(mat, src_times, grid_times):
    ss = (src_times - T1).total_seconds().to_numpy(); gs = (grid_times - T1).total_seconds().to_numpy()
    return np.array([np.interp(gs, ss, mat[k]) for k in range(mat.shape[0])])


# MOOSE strain_yy(MD,t)
tdf = pd.read_csv(OUT/f"{PROJ}_input_csv.csv"); taxis = tdf["time"].to_numpy(float)
vpp = sorted(glob.glob(str(OUT/f"{PROJ}_input_csv_fiber_strain_sampler_*ft_*.csv")))
n = min(len(vpp), len(taxis)); taxis = taxis[:n]; vpp = vpp[:n]
yv = next(pd.read_csv(f).sort_values("y")["y"].to_numpy(float) for f in vpp if len(pd.read_csv(f)))
cols = [pd.read_csv(vpp[i]).sort_values("y")["strain_yy"].to_numpy(float) if len(pd.read_csv(vpp[i])) else np.zeros_like(yv) for i in range(n)]
m_all = (np.column_stack(cols) - np.column_stack(cols)[:,[0]]) * 1e3
m_md = STAR + yv/FT; oo = np.argsort(m_md); m_md, m_all = m_md[oo], m_all[oo]
m_times = T1 + pd.to_timedelta(taxis, unit="s")

# 15:00-anchored 4h grid
grid = pd.date_range(T1, T3, freq="4h")
o_md, o_ct, o_mat = rp(OBS)
s_md, s_ct, s_mat = rp(SHEAR)
O = interp_time(o_mat, o_ct, grid)                                  # (o_md, ng)
SH = interp_time(np.array([np.interp(o_md, s_md, s_mat[:,j]) for j in range(s_mat.shape[1])]).T, s_ct, grid)
M = interp_time(np.array([np.interp(o_md, m_md, m_all[:,j]) for j in range(m_all.shape[1])]).T, m_times, grid)
# reference everything to T1=15:00 (first grid column)
O -= O[:,[0]]; SH -= SH[:,[0]]; M -= M[:,[0]]

v = np.isfinite(M) & np.isfinite(O-SH)
s_best = float(np.dot(M[v], (O-SH)[v]) / np.dot(M[v], M[v]))
MODEL = s_best*M + SH
rms0 = np.sqrt(np.nanmean(O[v]**2)); rmsr = np.sqrt(np.nanmean((O-MODEL)[v]**2))
print(f"T1=15:00: best s={s_best:.3f}, RMS {rms0:.4f}->{rmsr:.4f} mε ({100*(1-rmsr/rms0):.0f}% var red), "
      f"obs peak {np.nanmax(O):.4f}, model peak {np.nanmax(MODEL):.4f}")

pz = np.load(PZ, allow_pickle=True); pdas = np.asarray(pz["data"],float); pic = float(pdas[0])
pt = T1 + pd.to_timedelta(np.asarray(pz["taxis"],float), unit="s"); fracp = pic + s_best*(pdas-pic)
print(f"implied fracture pressure: {pic:.0f} -> {fracp.max():.0f} psi (Δp peak {s_best*(pdas.max()-pic):.0f} psi)")
# export the pressure history CSV (deliverable)
pd.DataFrame({"time": pt, "das_pressure_psi": pdas, "fracture_pressure_psi": fracp,
             "fracture_dp_psi": fracp-pic}).to_csv(OUT/"inferred_fracture_pressure_history.csv", index=False)


def wig(ax, prof, dep, tms):
    p95 = np.nanpercentile(np.abs(prof),95) or np.nanmax(np.abs(prof)); sec = PROFILE_HOURS*3600/p95
    for j,t in enumerate(tms):
        f = np.isfinite(prof[:,j]); ax.plot(t + pd.to_timedelta(prof[f,j]*sec, unit="s"), dep[f], "k", lw=0.7, alpha=0.9, zorder=5)


def wf(ax, prof, dep, tms, title, lim=0.1):
    tf = pd.date_range(tms[0], tms[-1], periods=300); tfs=(tf-tms[0]).total_seconds().to_numpy(); tcs=(tms-tms[0]).total_seconds().to_numpy()
    W = np.array([np.interp(tfs,tcs,prof[k]) for k in range(prof.shape[0])])
    im = ax.imshow(W, aspect="auto", cmap="seismic", vmin=-lim, vmax=lim,
                   extent=[mdates.date2num(tf[0]),mdates.date2num(tf[-1]),dep[-1],dep[0]], interpolation="bilinear")
    ax.xaxis_date(); wig(ax, prof, dep, tms)
    for tt,lab in zip([tms[0],T2,tms[-1]],["T1","T2","T3"]):
        ax.axvline(tt,color="green",ls="--",lw=1.6); ax.text(tt,1.01,lab,color="green",fontweight="bold",ha="center",va="bottom",transform=ax.get_xaxis_transform())
    jl=prof.shape[1]-1; ip=int(np.nanargmax(np.abs(prof[:,jl]))); p95=np.nanpercentile(np.abs(prof),95) or 1
    xs=tms[jl]+pd.to_timedelta(prof[ip,jl]*PROFILE_HOURS*3600/p95,unit="s")
    ax.plot(xs,dep[ip],"*",color="yellow",mec="k",ms=15,zorder=10)
    ax.annotate(f"{prof[ip,jl]:+.4f} millistrain",(xs,dep[ip]),textcoords="offset points",xytext=(-8,6),fontsize=8,bbox=dict(fc="white",ec="k",alpha=0.85))
    ax.set_ylim(10500,10200); ax.set_ylabel("Gold 4-PB Measured Depth [ft]"); ax.set_title(title)
    plt.colorbar(im,ax=ax).set_label("Strain (millistrain)"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))


# deliverable waterfall (image.png style)
fig,(a1,a2)=plt.subplots(2,1,figsize=(16,10),sharex=True,constrained_layout=True)
wf(a1,O,o_md,grid,"Observed DAS Strain Waterfall with 4-hour Mean Profiles (T1=15:00 Referenced)")
wf(a2,MODEL,o_md,grid,f"Model = {s_best:.2f}×MOOSE tensile + DDM shear")
a2.set_xlabel("Time [UTC-7]"); fig.suptitle("V1 fault (T1=15:00): observed vs MOOSE+DDM strain", fontweight="bold", fontsize=13)
fig.savefig(FIG/"deliverable_obs_vs_model_waterfall.png", dpi=150); print("Saved:", FIG/"deliverable_obs_vs_model_waterfall.png")

# match / pressure figure
fig2,ax=plt.subplots(2,2,figsize=(15,10),constrained_layout=True)
sv=np.linspace(0,max(2.0,2*s_best),200); mis=[np.sqrt(np.nanmean(((O-SH)[v]-x*M[v])**2)) for x in sv]
ax[0,0].plot(sv,mis,"b-"); ax[0,0].axvline(s_best,color="r",ls="--",label=f"best s={s_best:.2f}"); ax[0,0].set(xlabel="pressure scale s",ylabel="RMS misfit (mε)",title="(a) misfit vs pressure scale"); ax[0,0].grid(alpha=.3); ax[0,0].legend()
jl=len(grid)-1
ax[0,1].plot(O[:,jl],o_md,"k-",lw=2,label=f"observed ({grid[jl]:%m-%d %H:%M})"); ax[0,1].plot(MODEL[:,jl],o_md,"r--",lw=2,label="model"); ax[0,1].plot(s_best*M[:,jl],o_md,"tab:orange",lw=1,label="MOOSE tensile×s"); ax[0,1].plot(SH[:,jl],o_md,"b",lw=1,label="DDM shear")
ax[0,1].set(xlabel="strain (mε)",ylabel="MD (ft)",title="(b) profile near T3"); ax[0,1].set_ylim(10500,10200); ax[0,1].grid(alpha=.3); ax[0,1].legend(fontsize=8)
ssnap=np.array([np.dot(M[:,j][np.isfinite(M[:,j])],(O-SH)[:,j][np.isfinite((O-SH)[:,j])])/np.dot(M[:,j][np.isfinite(M[:,j])],M[:,j][np.isfinite(M[:,j])]) if np.any(np.isfinite(M[:,j])) and np.nansum(M[:,j]**2)>0 else np.nan for j in range(len(grid))])
ax[1,0].plot(grid,ssnap,"o-",color="tab:green"); ax[1,0].axhline(s_best,color="r",ls="--",label=f"global {s_best:.2f}"); ax[1,0].axvline(T2,color="0.5",ls=":"); ax[1,0].set(xlabel="time",ylabel="per-snapshot s",title="(c) per-snapshot scale"); ax[1,0].grid(alpha=.3); ax[1,0].legend(); ax[1,0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax[1,1].plot(pt,pdas,"0.6",lw=1.2,label="DAS pressure (s=1)"); ax[1,1].plot(pt,fracp,"r-",lw=1.8,label=f"implied fracture p (s={s_best:.2f})"); ax[1,1].axhline(pic,color="k",ls=":",label=f"IC {pic:.0f} psi"); ax[1,1].set(xlabel="time",ylabel="pressure (psi)",title="(d) inferred within-fracture pressure"); ax[1,1].grid(alpha=.3); ax[1,1].legend(fontsize=8); ax[1,1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
fig2.suptitle(f"V1 (T1=15:00): best s={s_best:.2f}, RMS {rms0:.3f}→{rmsr:.3f} mε ({100*(1-rmsr/rms0):.0f}% var. red.)", fontweight="bold")
fig2.savefig(FIG/"match_and_pressure.png", dpi=140); print("Saved:", FIG/"match_and_pressure.png")
