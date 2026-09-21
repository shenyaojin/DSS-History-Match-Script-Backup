"""QC-only scan of S-well channels against the existing timestamp-aligned g1.

Uses the existing median drift correction. Uniform-time interpolation is for
shape ranking only; native observations are retained in candidate plots.
No MOOSE inputs, observation crops, or inversion settings are modified.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'swell_pressure_channel_scan'
SOURCE = ROOT/'data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz'
GAUGE = ROOT/'data/fiberis_format/prod/gauges/pressure_g1.npz'


def correlations(values, reference):
    center = values-values.mean(axis=1, keepdims=True)
    ref = reference-reference.mean()
    denominator = np.sqrt(np.sum(center**2, axis=1)*np.sum(ref**2))
    return np.divide(center@ref, denominator, out=np.full(len(values), np.nan), where=denominator > 1e-20)


def distinct(table, n):
    indices, depths = [], []
    for row in table.itertuples():
        if all(abs(row.md_ft-d) >= 5 for d in depths):
            indices.append(int(row.channel_index)); depths.append(row.md_ft)
        if len(indices) == n:
            break
    return indices


def main():
    OUT.mkdir(exist_ok=True)
    data = DSS2D()
    data.load_npz(str(SOURCE))
    ref = (data.daxis > 7500) & (data.daxis < 15000)
    drift = np.median(data.data[ref], axis=0)
    data.data -= drift[None, :]
    data.select_time(0, 400000)
    t, md, signed = data.taxis, data.daxis.astype(float), data.data
    # Match the prepared inversion crop, including units and float32 correction.
    cache = np.load(HERE/'real_rfsdss_fixed_zones_14710_14725_400000s/data/unrectified_native.npz')
    use = (md >= 14710) & (md <= 14725)
    np.testing.assert_array_equal(md[use], cache['daxis'])
    np.testing.assert_allclose(signed[use]*1e-6, cache['data'], rtol=1e-6, atol=1e-13)
    np.testing.assert_array_equal(t, cache['taxis'])
    gauge = np.load(GAUGE, allow_pickle=True)
    gt = gauge['taxis']+(gauge['start_time'].item()-data.start_time).total_seconds()
    gp = gauge['data']*.00689476  # MPa; same conversion as the MOOSE preparation
    gp = gp-np.interp(0, gt, gp)
    assert gt.min() <= t[0] and gt.max() >= t[-1]
    grid = np.arange(0, t[-1]+.001, 1000.)
    p = np.interp(grid, gt, gp)
    left = np.searchsorted(t, grid, side='right')-1
    left = np.clip(left, 0, len(t)-2)
    fraction = (grid-t[left])/(t[left+1]-t[left])
    rows = []
    thin = np.empty((len(md), len(grid)), dtype=np.float32)
    periods = {'rise': (grid >= 70000)&(grid <= 140000),
               'late_rise': (grid >= 140000)&(grid <= 320000),
               'fall': grid >= 325000}
    for first in range(0, len(md), 512):
        native = signed[first:first+512]
        values = native[:, left]*(1-fraction)+native[:, left+1]*fraction
        thin[first:first+len(native)] = values
        positive = np.maximum(native[:, left], 0)*(1-fraction)+np.maximum(native[:, left+1], 0)*fraction
        r = correlations(values, p)
        frame = pd.DataFrame(dict(channel_index=np.arange(first, first+len(native)), md_ft=md[first:first+len(native)],
            r_signed=r, r_positive=correlations(positive, p),
            dynamic_range_microstrain=np.quantile(values, .95, axis=1)-np.quantile(values, .05, axis=1),
            peak_microstrain=np.max(values, axis=1), negative_fraction=np.mean(native < 0, axis=1),
            pre_event_std_microstrain=np.std(values[:, grid < 60000], axis=1)))
        for name, mask in periods.items():
            frame['r_'+name] = correlations(values[:, mask], p[mask])
        rows.append(frame)
    scores = pd.concat(rows, ignore_index=True)
    scores['eligible_amplitude'] = scores.dynamic_range_microstrain >= .5
    scores['legacy_depth_roi'] = scores.md_ft.between(11500, 16700)
    ranked = scores.loc[scores.eligible_amplitude & (scores.r_signed > 0)].sort_values('r_signed', ascending=False)
    roi = ranked[ranked.legacy_depth_roi]
    selected = distinct(roi, 12)
    full = distinct(ranked, 12)
    historical = [int(np.argmin(abs(md-x))) for x in [14888, 14972, 14992]]
    current = [int(np.argmin(abs(md-x))) for x in [14711, 14716.4, 14722]]
    candidates = list(dict.fromkeys(selected+full+historical+current))
    # Fixed support for every trial lag. Positive lag means strain follows pressure.
    lag_grid = np.arange(-20000, 40001, 1000.)
    support = (grid >= 40000)&(grid <= grid[-1]-20000)
    lag_references = np.array([np.interp(grid[support]-lag, gt, gp) for lag in lag_grid])
    picked = scores.iloc[candidates].copy()
    for i in candidates:
        rs = np.array([correlations(thin[i:i+1, support], pp)[0] for pp in lag_references])
        best = int(np.nanargmax(rs))
        picked.loc[i, 'best_lag_s'] = lag_grid[best]
        picked.loc[i, 'r_best_lag_common_support'] = rs[best]
        neighbors = np.arange(max(0, i-2), min(len(md), i+3))
        picked.loc[i, 'neighbor_r_median'] = np.nanmedian(scores.iloc[neighbors].r_signed)
        coef = np.polyfit(p, thin[i], 1)
        picked.loc[i, 'pressure_scale_microstrain_per_MPa'] = coef[0]
        picked.loc[i, 'pressure_fit_offset_microstrain'] = coef[1]
        picked.loc[i, 'pressure_fit_rms_microstrain'] = np.sqrt(np.mean((thin[i]-np.polyval(coef,p))**2))
    hits = []
    for path in (ROOT/'data/fiberis_format/s_well/geometry/frac_hit').glob('*.npz'):
        d = np.load(path, allow_pickle=True)
        hits.extend(np.asarray(d['data'], dtype=float).ravel())
    hits = np.asarray(hits)
    if len(hits):
        nearest = np.array([hits[np.argmin(abs(hits-x))] for x in picked.md_ft])
        picked['nearest_historical_frac_hit_ft'] = nearest
        picked['historical_frac_hit_distance_ft'] = abs(nearest-picked.md_ft)
    scores.sort_values('r_signed', ascending=False).to_csv(OUT/'all_channels.csv.gz', index=False)
    picked.sort_values('r_signed', ascending=False).to_csv(OUT/'candidate_channels.csv', index=False)
    np.savez_compressed(OUT/'candidate_native_curves.npz', data=signed[candidates], daxis=md[candidates],
                         taxis=t, start_time=data.start_time, units='microstrain',
                         pressure_taxis=gt[(gt>=0)&(gt<=t[-1])], pressure_MPa=gp[(gt>=0)&(gt<=t[-1])])

    def curves(indices, name, title):
        fig, axes = plt.subplots((len(indices)+1)//2, 2, figsize=(14, 3.2*((len(indices)+1)//2)),
                                  sharex=True, layout='constrained', squeeze=False)
        for ax, i in zip(axes.flat, indices):
            row = picked.loc[i]
            ax.plot(t/1000, signed[i], color='0.45', lw=.65, label='Signed DSS, native')
            ax.plot(grid/1000, thin[i], color='C0', lw=1, label='DSS at uniform QC times')
            mapped = row.pressure_scale_microstrain_per_MPa*p+row.pressure_fit_offset_microstrain
            ax.plot(grid/1000, mapped, color='C3', lw=1.7, label='Pressure: scale + offset only, no time shift')
            ax.set(title=f'{md[i]:.3f} ft | r={row.r_signed:.4f} | fall r={row.r_fall:.3f}',
                   ylabel='Strain [microstrain]', xlabel='Time [1000 s]', xlim=(0,400))
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(title+'\nPressure source: prod/pressure_g1; comparison only, no inversion preprocessing changes')
        fig.savefig(OUT/name, dpi=160)
        plt.close(fig)

    curves(selected[:6], 'top_candidates.png', 'Six separated candidate locations (>=5 ft apart), ranked by zero-lag signed correlation')
    curves(selected[1:3], 'recommended_candidates.png', 'Two candidates for closer QC: previously selected fracture and coherent pressure tracking')
    curves(historical+current, 'historical_and_current.png', 'Previously selected fractures (first three) and current inversion channels')
    curves(full[:6], 'full_depth_top_candidates.png', 'Best matches across the full available depth range')
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), layout='constrained')
    for axs, i in zip(axes, selected[:3]):
        local = (md >= md[i]-8)&(md <= md[i]+8)
        limit = max(float(np.quantile(abs(signed[local]), .99)), .5)
        im = axs[0].pcolormesh(t/1000, md[local], signed[local], shading='nearest', cmap='bwr', vmin=-limit, vmax=limit)
        axs[0].axhline(md[i], color='k', lw=.8)
        axs[0].set(ylim=(md[local].max(), md[local].min()), title=f'Local signed DSS: {md[i]:.3f} ft', ylabel='MD [ft]', xlabel='Time [1000 s]')
        fig.colorbar(im, ax=axs[0], label='Microstrain')
        for j in [max(0,i-2),i,min(len(md)-1,i+2)]:
            values = thin[j]
            centered = values-values.mean()
            axs[1].plot(grid/1000, centered/np.std(values), label=f'{md[j]:.3f} ft')
        axs[1].plot(grid/1000, (p-p.mean())/p.std(), 'k--', label='Pressure')
        axs[1].set(title='Neighbor coherence (display normalization only)', xlabel='Time [1000 s]', ylabel='Centered / standard deviation')
        axs[1].legend(fontsize=8); axs[1].grid(alpha=.2)
    fig.savefig(OUT/'local_waterfalls.png', dpi=160)
    plt.close(fig)
    manifest = dict(source=str(SOURCE), pressure_source=str(GAUGE), start_time=str(data.start_time),
        depth_range_ft=[float(md.min()),float(md.max())], channels=len(md), native_times=len(t), last_time_s=float(t[-1]),
        drift='Per-time median over 7500 < MD < 15000 ft; same Fiberis workflow; verified against inversion crop',
        ranking='Signed Pearson correlation, zero time shift, uniform 1000 s QC grid; threshold dynamic p95-p05 >= 0.5 microstrain',
        roi_for_primary_candidates_ft=[11500,16700], min_candidate_separation_ft=5,
        plot_pressure='Least-squares scale and offset only; not a forward model fit',
        lag_search_s=[-20000,40000], lag_step_s=1000,
        lag_note='Descriptive only, restricted common support; positive lag means strain after pressure; not hydraulic travel time',
        selected_depths_ft=md[selected].tolist(), full_depth_top_ft=md[full].tolist(), historical_depths_ft=md[historical].tolist())
    (OUT/'scan_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2), flush=True)
    print(picked.loc[selected[:6], ['md_ft','r_signed','r_positive','r_rise','r_fall','best_lag_s','neighbor_r_median','dynamic_range_microstrain']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
