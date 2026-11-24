"""Plot real Mariner observations for the RCP seminar; no forward model runs.

Run from any directory with the project's Python environment.
Optional --arial-font /path/to/Arial.ttf enables exact Arial raster/PDF output.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/rcp_seminar_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'fibeRIS/src'))
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.io.reader_mariner_pressureg1 import MarinerPressureG1

OUT = ROOT / 'figs/RCP_seminar_intro_20261021'
DSS = ROOT / 'data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz'
H5 = ROOT / 'data/legacy/DSSdata/Mariner 14x-36-POW-S - RFS strain change.h5'
PRESSURE = ROOT / 'data/legacy/prod/pressure_g1.csv'
PRESSURE_CACHE = ROOT / 'data/fiberis_format/prod/gauges/pressure_g1.npz'
SIGNED_CACHE = ROOT / 'scripts/DSS_history_match/real_rfsdss_fixed_zones_14373_14379_400000s_bg21/data/unrectified_native.npz'
PREPROCESSING = ROOT / 'scripts/DSS_history_match/propose_14376_initial_model.py'
# Per user's follow-up: approximate visual picks from pressure, NOT operation logs.
EVENTS = [(70000., 'Shut-in', '#865428'),
          (340000., 'Production resumes', '#765aa7')]
COLORS = ['#0072B2', '#D55E00', '#009E73']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def edges(centers):
    return np.r_[centers[0] - (centers[1]-centers[0])/2,
                 (centers[:-1]+centers[1:])/2,
                 centers[-1] + (centers[-1]-centers[-2])/2]


def broken_line(x, y, max_gap):
    """Keep every sample; insert NaNs so lines do not bridge acquisition gaps."""
    breaks = np.flatnonzero(np.diff(x) > max_gap) + 1
    return np.insert(x, breaks, np.nan), np.insert(y, breaks, np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arial-font', type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.arial_font:
        font_manager.fontManager.addfont(str(args.arial_font))
    arial = [f.fname for f in font_manager.fontManager.ttflist if f.name == 'Arial']
    actual_font = 'Arial' if arial else 'Liberation Sans'
    plt.rcParams.update({'font.family': actual_font, 'font.size': 16,
        'axes.labelsize': 17, 'xtick.labelsize': 15, 'ytick.labelsize': 14,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})

    # Reuse the exact established loader and median reference-depth correction.
    dss = DSS2D()
    dss.load_npz(str(DSS))
    assert dss.data.shape == (len(dss.daxis), len(dss.taxis))
    ref = (dss.daxis > 7500) & (dss.daxis < 15000)
    reference_values = dss.data[ref]
    reference_has_nan = bool(np.any(~np.isfinite(reference_values)))
    # NaNs remain missing. For this source the finite branch is identical to QC.
    drift = (np.nanmedian(reference_values, axis=0) if reference_has_nan
             else np.median(reference_values, axis=0))
    del reference_values
    dss.data -= drift[None, :]
    del drift
    dss.select_time(0, 400000)
    dss.select_depth(14373, 14379)
    t, md, strain = dss.taxis.copy(), dss.daxis.astype(float), dss.data.astype(float)
    start = dss.start_time
    del dss
    assert np.all(np.diff(t) > 0) and np.all(np.diff(md) > 0)
    cache_check = None
    if SIGNED_CACHE.exists():
        with np.load(SIGNED_CACHE, allow_pickle=True) as cache:
            np.testing.assert_allclose(t, cache['taxis'], rtol=0, atol=1e-8)
            np.testing.assert_array_equal(md, cache['daxis'])
            cache_check = float(np.nanmax(abs(strain - cache['data']*1e6)))
            np.testing.assert_allclose(strain, cache['data']*1e6, rtol=1e-6, atol=1e-6)

    with h5py.File(H5) as f:
        attrs = {k: (v.decode() if isinstance(v, bytes) else str(v))
                 for k, v in f['data'].attrs.items()}
        h5_start = dt.datetime.strptime(f['stamps'][0].decode(), '%m/%d/%Y %H:%M:%S.%f')
        assert h5_start == start
        assert attrs['TimeZoneFormat'] == '-06:00'
        h5_elapsed = (f['stamps_unix'][:len(t)] - f['stamps_unix'][0]) / 1e6
        np.testing.assert_allclose(h5_elapsed, t, rtol=0, atol=1e-8)
        unix_first_utc = dt.datetime.fromtimestamp(float(f['stamps_unix'][0])/1e6, dt.timezone.utc)
    start_aware = start.replace(tzinfo=dt.timezone(dt.timedelta(hours=-6)))
    unix_disagreement_s = (start_aware.astimezone(dt.timezone.utc)-unix_first_utc).total_seconds()

    pressure = MarinerPressureG1()
    pressure.read(str(PRESSURE))
    # Only recorded timestamp subtraction. NO peak alignment or inferred TZ shift.
    pt_all = pressure.taxis + (pressure.start_time-start).total_seconds()
    mask = (pt_all >= 0) & (pt_all <= 400000)
    pt, pv = pt_all[mask], pressure.data[mask].copy()
    pv[~np.isfinite(pv)] = np.nan
    assert np.all(np.diff(pt) > 0)
    with np.load(PRESSURE_CACHE, allow_pickle=True) as cache:
        np.testing.assert_array_equal(pressure.data, cache['data'])
        np.testing.assert_array_equal(pressure.taxis, cache['taxis'])
        assert pressure.start_time == cache['start_time'].item()

    chosen = [int(np.argmin(abs(md-target))) for target in [14375.609,14376.276,14376.947]]
    cadence = float(np.median(np.diff(t)))
    limit = max(.5, np.ceil(np.nanmax(abs(strain))*2)/2)
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(3, 2, width_ratios=[1,.017], height_ratios=[1,1.55,1.15],
        left=.095, right=.92, bottom=.16, top=.84, hspace=.16, wspace=.025)
    axp = fig.add_subplot(grid[0,0])
    axd = fig.add_subplot(grid[1,0], sharex=axp)
    axc = fig.add_subplot(grid[2,0], sharex=axp)
    cax = fig.add_subplot(grid[1,1])
    axes = [axp, axd, axc]
    x,y = broken_line(pt/1000,pv,.3)
    axp.plot(x,y,color='#253e54',lw=1.45)
    axp.set_ylabel('Pressure [psi]')
    axp.set_yticks([1850,2000,2150])
    axp.text(.012,.88,'Production well · pressure_g1 · gauge/absolute datum unrecorded',
             transform=axp.transAxes,fontsize=12,va='top',bbox=dict(facecolor='white',edgecolor='none',alpha=.9))

    cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#e5e7eb')
    norm = TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    splits = np.r_[0,np.flatnonzero(np.diff(t)>5*cadence)+1,len(t)]
    for i,j in zip(splits[:-1],splits[1:]):
        te=edges(t[i:j])/1000
        te[0]=max(0,te[0]);te[-1]=min(400,te[-1])
        mesh=axd.pcolormesh(te,edges(md),np.ma.masked_invalid(strain[:,i:j]),
                           cmap=cmap,norm=norm,shading='flat',rasterized=True)
    axd.set_facecolor('#e5e7eb')
    axd.set_ylim(14379,14373)
    axd.set_yticks([14373,14375,14377,14379])
    axd.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    axd.set_ylabel('Measured depth [ft]')
    cb=fig.colorbar(mesh,cax=cax);cb.set_label('Axial strain [με]',fontsize=15)
    for i,color in zip(chosen,COLORS):
        axd.axhline(md[i],color=color,ls=(0,(5,3)),lw=1.2)
        x,y=broken_line(t/1000,strain[i],5*cadence/1000)
        axc.plot(x,y,color=color,lw=1.15,label=f'{md[i]:.3f} ft')
    axc.axhline(0,color='#88919a',lw=.65,zorder=0)
    axc.set_ylabel('Axial strain [με]')
    axc.set_xlabel('Time [1000 s]')
    axc.legend(loc='upper left',ncol=3,fontsize=13,frameon=False,
               bbox_to_anchor=(0,1.03),handlelength=1.6,columnspacing=1.5)
    axc.margins(y=.19)
    # End the raster at half a native sampling interval; do not fill the gap.
    gap_start=(t[-1]+cadence/2)/1000
    if gap_start<400:
        axc.axvspan(gap_start,400,color='#e5e7eb',zorder=0)
        axd.text((gap_start+400)/2,14376,'No DSS data',rotation=90,
                 ha='center',va='center',fontsize=11,color='#59636d')
    for ax,letter in zip(axes,['a','b','c']):
        ax.set_xlim(0,400)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.tick_params(axis='both',length=4)
        ax.text(-.10,1.01,f'({letter})',transform=ax.transAxes,
                fontsize=18,fontweight='bold',va='top')
        if ax != axd:ax.grid(axis='y',color='#e6e9ed',lw=.65)
        for seconds,label,color in EVENTS:
            ax.axvline(seconds/1000,color=color,ls='--',lw=1.25,zorder=5)
        if ax != axc:ax.tick_params(labelbottom=False)
    for seconds,label,color in EVENTS:
        axp.text(seconds/1000,1.12,f'{label} ≈ {seconds/1000:.0f} ks*',
            transform=axp.get_xaxis_transform(),ha='center',color=color,fontsize=14)
    fig.text(.095,.95,'RFS-DSS and production-well pressure',fontsize=25,fontweight='bold')
    fig.text(.095,.902,'Mariner 14x-36-POW-S · signed observations · native sampling',fontsize=16,color='#465362')
    fig.text(.095,.077,'* Operation timing inferred from pressure changes; no operation log available.',fontsize=12,color='#59636d')
    fig.text(.095,.047,'t0 = 2021-08-04 17:07:09 (DSS metadata: UTC−06:00). Pressure clock timezone unrecorded; alignment provisional.',fontsize=11.5,color='#59636d')
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'RCP_intro_observations.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    # SVG retains editable text and explicitly requests Arial on a PPT machine.
    svg=OUT/'RCP_intro_observations.svg'
    text=svg.read_text()
    if actual_font != 'Arial':
        text=text.replace("'Liberation Sans'", "'Arial', 'Liberation Sans'")
    svg.write_text(text)
    sources=[DSS,H5,PRESSURE,PRESSURE_CACHE,PREPROCESSING,Path(__file__)]
    report={
        'sources':{str(p.relative_to(ROOT)):sha256(p) for p in sources},
        'dss_h5_metadata':attrs,'dss_t0_local':start_aware.isoformat(),
        'dss_t0_utc_from_text_metadata':start_aware.astimezone(dt.timezone.utc).isoformat(),
        'dss_t0_utc_from_unix_field':unix_first_utc.isoformat(),
        'unix_field_disagreement_seconds':unix_disagreement_s,
        'time_policy':'Reuse reader: textual start_time + elapsed seconds. Pressure uses recorded clock labels; pressure timezone unknown. No empirical time shift.',
        'pressure_units':'psi per existing project pressure scripts; CSV itself has no unit/datum metadata',
        'pressure_sensor':'production well pressure_g1; sensor depth and gauge/absolute datum unverified',
        'pressure_value_transform':'none; native recorded pressure, no zeroing or unit conversion',
        'strain_units':'microstrain, inherited from RFS me metadata and existing QC 1e-6 conversion to dimensionless strain',
        'depth_units':'ft, same depth axis and convention as existing QC; no depth shift',
        'preprocessing':'Subtract per-time median over 7500 < MD < 15000 ft, before depth crop. Preserve native signed values and first frame. No half-wave clipping, added baseline, smoothing, or temperature correction.',
        'thermal_status':'Upstream RFS temperature compensation not established by available metadata. No extra DTS subtraction applied. Median reference subtraction is drift removal, not a demonstrated thermal correction.',
        'reference_contains_nonfinite':reference_has_nan,
        'max_difference_from_existing_signed_cache_microstrain':cache_check,
        'shape':list(strain.shape),'depth_range_ft':[float(md[0]),float(md[-1])],
        'native_time_range_s':[float(t[0]),float(t[-1])],
        'native_cadence_median_s':cadence,'negative_fraction':float(np.mean(strain<0)),
        'strain_range_microstrain':[float(np.nanmin(strain)),float(np.nanmax(strain))],
        'color_limits_microstrain':[-float(limit),float(limit)],
        'strain_missing_samples':int(np.count_nonzero(~np.isfinite(strain))),
        'selected_depths_ft':[float(md[i]) for i in chosen],
        'pressure_samples':len(pt),'pressure_missing_samples':int(np.count_nonzero(~np.isfinite(pv))),
        'pressure_gap_policy':'Break lines across intervals >300 s, retain every measured sample',
        'pressure_gaps_over_300s':int(np.count_nonzero(np.diff(pt)>300)),
        'dss_gap_policy':'Split at intervals >5 times median cadence; no extrapolation beyond half a native interval. Missing is gray, never zero.',
        'events':[{'label':label,'approximate_time_s':s,'basis':'Visual reading of pressure rise/fall, explicitly authorized by user; not operation-log verified','recorded_clock_label':(start+dt.timedelta(seconds=s)).isoformat(),'timezone':'unverified pressure clock'} for s,label,_ in EVENTS],
        'font_requested':'Arial','font_png_pdf':actual_font,'font_svg':'Arial, Liberation Sans',
        'missing_information':['Operation log and exact event times','Pressure clock timezone','Pressure gauge/absolute datum and sensor depth','Upstream RFS temperature compensation documentation'],
        'png_pixels':[4800,2700], 'outputs':['PNG 300 dpi','PDF','SVG with editable Arial text']}
    (OUT/'provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    note=f'''RCP seminar Introduction 数据图说明

数据来源：POW-S 原始 RFS strain change.npz（由同名 HDF5 转换）；生产井 data/legacy/prod/pressure_g1.csv。CSV 与现有 pressure_g1.npz 的数值、时间轴逐项一致。测点标识仅可确认 pressure_g1，深度和表压/绝压未载明。所有来源、哈希和处理检查见 provenance.json。

时间参考：沿用项目读取器，以 DSS 文字起点 {start_aware.isoformat()} 为 t=0，横轴为秒/1000。HDF5 TimeZoneFormat=-06:00；其 Unix 字段换算的绝对时间比文字时间及其时区早 {unix_disagreement_s/3600:g} 小时，因此 Unix 字段只用于相对秒数。压力 CSV 无时区，按现有项目相同的文字时钟相减绘图，绝对对齐仍待核实；没有移动峰值、校正未知时差或把压力时区宣称为 UTC−06:00。

单位：压力保留项目既有 psi 数值，不减初值；原 CSV 无单位字段，psi 依据既有压力读取/使用流程，表压/绝压未知。DSS 沿用 HDF5 me 和现有 QC 的微应变解释（转为无量纲应变时乘 1e-6）；本图直接使用 με，不再缩放。深度沿用 QC 的 ft，不加深度校准偏移。

基线与温度：复用 DSS2D 读取，以及 propose_14376_initial_model.py 的逐时刻参考段（7500<MD<15000 ft）中位数漂移扣除。与现有有符号缓存的最大差异为 {cache_check:.3g} με。保留第一帧及全部正负值，不取正半波，不额外归零、平滑、重采样或进行温度补偿。现有材料未证明 RFS 上游是否已做温度补偿，因此不将漂移扣除称为温度校正。

操作标记：根据用户后续明确允许从压力特征估计，标注 Shut-in ≈70 ks、Production resumes ≈340 ks，三个面板对齐。它们是视觉近似位置，不是日志确认的操作时刻，图中以 * 和 pressure-inferred 说明。对应压力文字时钟约 {(start+dt.timedelta(seconds=70000)).isoformat()} 和 {(start+dt.timedelta(seconds=340000)).isoformat()}，时区未确认。

缺测与显示：18 通道、{len(t)} 个原生时刻；最后一个 DSS 时刻 {t[-1]:.6f} s，之后至400000 s留为缺测灰色，仅显示半个正常采样间隔的像素支持。时间间隔大于正常 DSS 间隔5倍时切断热图及曲线；压力间隔>300 s断线。不把缺测当作零；发散色标以0为中心，并覆盖全部有限应变值，无色标截断。

字体与文件：16:9，PNG 4800×2700，另有 PDF 和可编辑文字 SVG。本机 {'已找到 Arial，PNG/PDF 使用 Arial' if arial else '没有 Arial；PNG/PDF 如实使用 Liberation Sans，SVG 指定 Arial 并保留备用字体'}。提供 Arial.ttf 后以 --arial-font 参数重跑即可得到 Arial PNG/PDF。绘图脚本为 scripts/RCP_seminar/plot_intro_observations.py。

待补：操作日志与确切时刻、压力时区、压力基准及测点深度、RFS 上游温度补偿说明。
'''
    (OUT/'说明.txt').write_text(note)
    np.savez_compressed(OUT/'plotted_observations.npz',dss_time_s=t,depth_ft=md,
        signed_strain_microstrain=strain,pressure_time_recorded_clock_s=pt,pressure_psi=pv)
    pd.DataFrame({'time_s':t,**{f'MD_{md[i]:.6f}_ft_microstrain':strain[i] for i in chosen}}).to_csv(OUT/'selected_channels.csv',index=False)
    print(json.dumps({k:report[k] for k in ['shape','negative_fraction','strain_range_microstrain','selected_depths_ft','max_difference_from_existing_signed_cache_microstrain','font_png_pdf']},indent=2))
    print(OUT)


if __name__=='__main__':
    main()
