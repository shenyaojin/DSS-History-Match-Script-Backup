"""Refit only the global attenuation C for the existing bg=-19/-18 forwards."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'coupling_refit'


def main():
    if OUT.exists():
        raise RuntimeError('Refusing to overwrite existing coupling refit')
    manifest = json.loads((HERE / 'manifest.json').read_text())
    old_c = float(manifest['fixed_theta'][2])
    bounds = (1e-6, 1.)
    frames = {a: pd.read_csv(HERE / f'bg_{abs(a)}/strain_prediction.csv') for a in [-21, -19, -18]}
    source_hashes = {str(a): hashlib.sha256((HERE / f'bg_{abs(a)}/strain_prediction.csv').read_bytes()).hexdigest()
                     for a in frames}
    keys = ['measurement_time', 'measurement_values', 'measurement_xcoord', 'measurement_ycoord', 'measurement_zcoord']
    canonical = pd.read_csv(HERE.parent / 'inversion/measurement_data.csv')
    rows = []
    fitted = {}
    for alpha, df in frames.items():
        np.testing.assert_allclose(df[keys], canonical[keys], rtol=1e-10, atol=1e-18)
        assert len(df) == 18 * 130 and df.measurement_time.nunique() == 130
        np.testing.assert_allclose(df.simulation_values, old_c * df.physical_simulation_values, rtol=1e-10, atol=1e-18)
        use = df.measurement_time.to_numpy() > 0
        s = df.physical_simulation_values.to_numpy()
        d = df.measurement_values.to_numpy()
        assert np.isfinite(s).all() and np.isfinite(d).all() and (s >= 0).all() and (d >= 0).all()
        ss = float(s[use] @ s[use]); sd = float(s[use] @ d[use])
        assert ss > 0
        unconstrained = sd / ss
        c = float(np.clip(unconstrained, *bounds)) if alpha != -21 else old_c
        before = .5 * float(np.sum((old_c*s[use] - d[use])**2))
        after = .5 * float(np.sum((c*s[use] - d[use])**2))
        grad = float((c*s[use] - d[use]) @ s[use])
        if alpha != -21:
            assert after <= before + 1e-24
            if bounds[0] < c < bounds[1]:
                assert abs(grad) <= 1e-12 * max(abs(c*ss), abs(sd))
        frame = df.copy()
        frame['simulation_values'] = c*s
        frame['misfit_values'] = np.where(use, c*s-d, 0.)
        fitted[alpha] = frame
        rows.append(dict(background_alpha=alpha, refitted=alpha != -21,
            previous_C=old_c, fitted_C=c, unconstrained_optimum_C=unconstrained,
            coupling_at_bound=c in bounds, derivative_dJ_dC=grad,
            data_objective_before=before, data_objective_after=after,
            rms_before_microstrain=float(np.sqrt(2*before/use.sum())*1e6),
            rms_after_microstrain=float(np.sqrt(2*after/use.sum())*1e6),
            data_error_reduction_percent=100*(1-after/before), sample_count=int(use.sum())))
    OUT.mkdir()
    for alpha, frame in fitted.items():
        frame.to_csv(OUT / f'bg_{abs(alpha)}_strain.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT / 'comparison.csv', index=False)
    result = dict(method='Exact bounded scalar least squares: C=clip(dot(s,d)/dot(s,s),1e-6,1)',
        scope='Refit one global C per bg=-19/-18; retain bg=-21 original C as comparison',
        immutable='All permeability components, mesh, physics, pressure source and forward solutions fixed',
        observations='Existing half-wave positive observations and simulations, all 18 channels; exclude t=0; retain zeros',
        regularization='No C penalty; permeability regularization is constant and does not affect this scalar optimum',
        interpretation_limit='Existing pressure overshoot has not been resolved; amplitude refit does not validate pressure numerics',
        source_prediction_sha256=source_hashes, coupling_bounds=list(bounds), results=rows)
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    plot(frames, fitted, rows, manifest)
    for a, digest in source_hashes.items():
        assert hashlib.sha256((HERE / f'bg_{abs(int(a))}/strain_prediction.csv').read_bytes()).hexdigest() == digest
    (OUT / 'COMPLETE').write_text('Exact coupling refit and QC complete; original files preserved\n')
    print(json.dumps(rows, indent=2))


def plot(original, fitted, rows, manifest, *, heading=None, core_weight=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    native = np.load(HERE.parent / 'data/half_wave_positive_native.npz')
    base = fitted[-21]
    times = np.sort(base.measurement_time.unique())
    ys = np.sort(base.measurement_ycoord.unique())
    md = 14376 + (ys - 50)/.3048
    def matrix(df, col):
        return df.pivot(index='measurement_ycoord', columns='measurement_time', values=col).reindex(index=ys, columns=times).to_numpy()*1e6
    obs = matrix(base, 'measurement_values')
    new = {a: matrix(f, 'simulation_values') for a, f in fitted.items()}
    old = {a: matrix(f, 'simulation_values') for a, f in original.items()}
    coeff = {r['background_alpha']: r['fitted_C'] for r in rows}
    zone = manifest['zone']
    core = np.flatnonzero((md >= zone['lo_ft']) & (md < zone['hi_ft']))
    shown = np.r_[core[0]-1, core, core[-1]+1]
    vmax = max(3., np.ceil(max(v.max() for v in new.values())*2)/2)
    ymax = np.ceil(max(float(obs[shown].max()), *(float(v[shown].max()) for v in list(new.values())+list(old.values())))*2+.2)/2
    fig = plt.figure(figsize=(23,11), layout='constrained')
    grid = fig.add_gridspec(7,5, width_ratios=[1.2]*4+[2.8], height_ratios=[1]*6+[.35])
    water_axes = []
    fields = [('Observed native', native['data']*1e6, native['taxis'])]
    fields += [(f'Bg {a} | C={coeff[a]:.5f}' + (' (original)' if a == -21 else ' (refit)'), new[a], times) for a in [-21,-19,-18]]
    for j, (label, values, tt) in enumerate(fields):
        ax = fig.add_subplot(grid[:6,j]);water_axes.append(ax)
        im = ax.pcolormesh(tt/1000, md, values, shading='nearest', cmap='Reds', vmin=0, vmax=vmax)
        ax.set(title=label, xlabel='Time [1000 s]', ylim=(14379,14373), xlim=(0,400))
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        if j == 0:ax.set_ylabel('Measured depth [ft]')
        else:ax.tick_params(labelleft=False)
    colors={-21:'#1f77b4',-19:'#e68613',-18:'#249443'}
    for i, ch in enumerate(shown):
        ax=fig.add_subplot(grid[i,4])
        ax.plot(times/1000, obs[ch], color='black', lw=1.2, label='Observation')
        for a in [-21,-19,-18]:
            ax.plot(times/1000,new[a][ch],color=colors[a],lw=1.3,label=f'Bg {a}'+(' original' if a==-21 else ' refit C'))
        for a in [-19,-18]:
            ax.plot(times/1000,old[a][ch],color=colors[a],lw=.9,ls='--',alpha=.7,label=f'Bg {a} old C')
        label = f'{md[ch]:.3f} ft'
        if core_weight is not None:
            label += f' | weight={core_weight:g}' if ch in core else ' | weight=1'
        ax.set(title=label,xlim=(0,400),ylim=(-.05,ymax),ylabel='microstrain')
        ax.grid(alpha=.15)
        if i<5:ax.tick_params(labelbottom=False)
        else:ax.set_xlabel('Time [1000 s]')
        if i==0:handles,labels=ax.get_legend_handles_labels()
    cax=fig.add_subplot(grid[6,:4]);fig.colorbar(im,cax=cax,orientation='horizontal',label='Positive strain [microstrain]; common linear scale')
    legend=fig.add_subplot(grid[6,4]);legend.axis('off');legend.legend(handles,labels,ncol=2,loc='center',fontsize=8)
    fig.suptitle((heading or 'Only global coupling C refitted | Permeability and physical forward fields fixed') + '\n' +
                 f'All waterfalls: 0–{vmax:g} microstrain; all channel panels: same y scale | dashed = previous C',fontsize=14)
    fig.savefig(OUT/'qc_coupling_refit.png',dpi=170)
    fig.savefig(OUT/'qc_coupling_refit.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
