"""Read accepted results only; compare zone means and display-normalized shapes."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'comparison_xx_yy_60'


def zone_curve(frame, lo, hi, column):
    md = 14725+(frame.measurement_ycoord-50)/.3048
    select = (md >= lo) & (md < hi)
    return frame.loc[select].groupby('measurement_time')[column].mean()*1e6


def main():
    runs, snapshot = {}, {}
    for mode in ['fixed_yy', 'free_yy']:
        row = pd.read_csv(ROOT/mode/'accepted_iteration_history.csv').iloc[-1]
        iteration = int(row.iteration)
        runs[mode] = pd.read_csv(ROOT/mode/'iterations'/f'{iteration:02d}'/'strain.csv')
        snapshot[mode] = row.to_dict()
    initial = pd.read_csv(ROOT/'fixed_yy/iterations/00/strain.csv')
    cfg = json.loads((ROOT/'fixed_yy/config.json').read_text())
    zones = [z for z in cfg['zones'] if z['optimized']]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True, layout='constrained')
    table = []
    for row, label, zone in zip(axes, 'ABC', zones):
        lo, hi = zone['lo_ft'], zone['hi_ft']
        curves = [
            ('Observation', 'black', zone_curve(initial, lo, hi, 'measurement_values')),
            ('Initial model', '#7d8bb0', zone_curve(initial, lo, hi, 'simulation_values')),
            (f"Fixed yy, iter {int(snapshot['fixed_yy']['iteration'])}", '#d97706',
             zone_curve(runs['fixed_yy'], lo, hi, 'simulation_values')),
            (f"Free yy, iter {int(snapshot['free_yy']['iteration'])}", '#16806a',
             zone_curve(runs['free_yy'], lo, hi, 'simulation_values')),
        ]
        for name, color, series in curves:
            scale = float(series.max())
            row[0].plot(series.index, series.values, color=color, lw=1.7, label=name)
            if scale > 0:
                row[1].plot(series.index, series.values/scale, color=color, lw=1.7, label=name)
            table.extend(dict(zone=label, series=name, time_s=float(t), mean_positive_microstrain=float(v))
                         for t, v in series.items())
        row[0].set(title=f'{label}: {lo:g}–{hi:g} ft (visible channels only)', ylabel='Mean positive strain [microstrain]')
        row[1].set(title=f'{label}: temporal shape', ylabel='Each curve / its own peak')
        for ax in row:
            ax.set(xlim=(0, 400000))
            ax.set_xticks(np.arange(0, 400001, 100000))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}'))
            ax.grid(alpha=.2)
            ax.axvspan(series.index.max(), 400000, facecolor='.9', hatch='///', edgecolor='.6')
    axes[0, 0].legend(fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel('Time [1000 s]')
    fig.suptitle('S-well | Accepted model time curves on the unchanged 130-time schedule\n'
                 'Right: normalization for shape comparison only; inversion objective is unchanged', fontsize=13)
    fig.savefig(ROOT/'zone_time_curves.png', dpi=170)
    fig.savefig(ROOT/'zone_time_curves.pdf')
    plt.close(fig)
    pd.DataFrame(table).to_csv(ROOT/'zone_time_curves.csv', index=False)
    (ROOT/'zone_time_curves_snapshot.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    print(ROOT/'zone_time_curves.png')


if __name__ == '__main__':
    main()
