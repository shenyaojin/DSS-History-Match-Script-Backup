"""Separate QC panels for the unbounded C, 3x central-channel refits."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'coupling_refit_weighted3_unbounded'


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    native = np.load(HERE.parent / 'data/half_wave_positive_native.npz')
    frames = {a: pd.read_csv(OUT / f'bg_{abs(a)}_strain.csv') for a in [-19, -18]}
    base = frames[-19]
    times = np.sort(base.measurement_time.unique())
    ys = np.sort(base.measurement_ycoord.unique())
    depths = 14376 + (ys - 50) / .3048
    core = np.flatnonzero(np.isclose(depths[:, None], summary['central_channel_depths_ft'], rtol=0, atol=1e-6).any(axis=1))
    assert len(core) == 4
    shown = np.r_[core[0]-1, core, core[-1]+1]

    def matrix(df, col):
        return df.pivot(index='measurement_ycoord', columns='measurement_time', values=col).reindex(index=ys, columns=times).to_numpy()*1e6

    obs = matrix(base, 'measurement_values')
    predictions = {a: matrix(df, 'simulation_values') for a, df in frames.items()}
    ymax = np.ceil(max(obs[shown].max(), *(v[shown].max() for v in predictions.values()))*2 + .2)/2
    vmax = max(3., np.ceil(max(v.max() for v in predictions.values())*2)/2)
    for alpha, pred in predictions.items():
        coefficient = next(r['fitted_C'] for r in summary['results'] if r['background_alpha'] == alpha)
        fig = plt.figure(figsize=(14, 10), layout='constrained')
        grid = fig.add_gridspec(7, 3, width_ratios=[1, 1, 1.8], height_ratios=[1]*6+[.25])
        for j, (title, values, tt) in enumerate([
                ('Observed RFSDSS', native['data']*1e6, native['taxis']),
                ('Fitted strain', pred, times)]):
            ax = fig.add_subplot(grid[:6, j])
            im = ax.pcolormesh(tt/1000, depths, values, shading='nearest', cmap='Reds', vmin=0, vmax=vmax)
            ax.set(title=title, xlabel='Time [1000 s]', xlim=(0,400), ylim=(14379,14373))
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            if j == 0:
                ax.set_ylabel('Measured depth [ft]')
            else:
                ax.tick_params(labelleft=False)
        for i, ch in enumerate(shown):
            ax = fig.add_subplot(grid[i,2])
            ax.plot(times/1000, obs[ch], color='black', lw=1.2, label='Observation')
            ax.plot(times/1000, pred[ch], color='#d95f02', lw=1.5, label='Fit')
            weight = 3 if ch in core else 1
            ax.set(title=f'{depths[ch]:.3f} ft | weight={weight}', xlim=(0,400), ylim=(-.05,ymax), ylabel='microstrain')
            ax.grid(alpha=.15)
            if i < len(shown)-1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Time [1000 s]')
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
        cax = fig.add_subplot(grid[6,:2])
        fig.colorbar(im, cax=cax, orientation='horizontal', label='Positive strain [microstrain]')
        legend = fig.add_subplot(grid[6,2])
        legend.axis('off')
        legend.legend(handles, labels, ncol=2, loc='center', frameon=False)
        fig.suptitle(f'Background alpha = {alpha} | C = {coefficient:.5f} (unbounded)\nCentral four channels: 3x squared-error weight', fontsize=14)
        for ext in ['png', 'pdf']:
            path = OUT / f'qc_bg_{abs(alpha)}_only.{ext}'
            fig.savefig(path, dpi=170)
            print(path)
        plt.close(fig)


if __name__ == '__main__':
    main()
