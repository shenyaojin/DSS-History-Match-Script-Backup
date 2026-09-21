"""Read-only diagnostics of completed runs; writes only moose_input_audit artifacts."""
from pathlib import Path
import hashlib
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent / 'real_rfsdss_fixed_zones_14710_14725_400000s'
RUN = HERE / 'comparison_xx_yy_60/fixed_yy'
OUT = HERE / 'moose_input_audit'


def main():
    cfg = json.loads((RUN / 'config.json').read_text())
    observations = np.load(BASE / 'data/half_wave_positive_solver.npz', allow_pickle=True)
    md, times, obs = (observations[k] for k in ('daxis', 'taxis', 'data'))
    frame = pd.read_csv(RUN / 'strain_final.csv')
    sim = frame.pivot(index='measurement_ycoord', columns='measurement_time',
                      values='simulation_values').to_numpy()
    exported_obs = frame.pivot(index='measurement_ycoord', columns='measurement_time',
                               values='measurement_values').to_numpy()
    np.testing.assert_allclose(exported_obs, obs, atol=1e-18, rtol=1e-9)
    assert np.all(np.diff(md) > 0) and len(times) == 130
    edges = np.asarray(cfg['mesh_edges_m'])
    y = cfg['y_center_m'] + (md-cfg['md_center_ft'])*.3048
    cells = np.searchsorted(edges, y, side='right')-1
    projection = np.empty_like(obs)
    groups = []
    max_sim_spread = 0.
    for cell in np.unique(cells):
        use = cells == cell
        projection[use] = obs[use].mean(axis=0)
        max_sim_spread = max(max_sim_spread, float(np.ptp(sim[use], axis=0).max()))
        groups.append(dict(layer_index_zero_based=int(cell), channels=int(use.sum()),
                           lo_ft=float(14725+(edges[cell]-50)/.3048),
                           hi_ft=float(14725+(edges[cell+1]-50)/.3048)))
    def loss(pred):
        residual = (pred-obs)[:, 1:]
        return dict(J_data=float(.5*np.sum(residual**2)),
                    rms_microstrain=float(np.sqrt(np.mean(residual**2))*1e6))
    floor, actual = loss(projection), loss(sim)
    text = (RUN / 'forward_and_adjoint.i').read_text()
    pressure_block = re.search(r'\[injection_pressure_func\](.*?)\[\]', text, re.S)[1]
    pressure = {key: np.fromstring(re.search(r"\b"+key+r"\s*=\s*'([^']*)'", pressure_block)[1], sep=' ')
                for key in ('x', 'y')}
    gauge = np.load(ROOT / 'data/fiberis_format/prod/gauges/pressure_g1.npz', allow_pickle=True)
    start = observations['start_time'].item()
    gt = gauge['taxis']+(gauge['start_time'].item()-start).total_seconds()
    gp = gauge['data']*6894.76
    reconstructed = np.interp(pressure['x'], gt, gp)-np.interp(0., gt, gp)
    Kd = 35e9/(3*(1-2*.25))
    storage = lambda compliance: (1-.8)*(.8-.01)*compliance+.01/2.2e9
    result = dict(
        input_sha256=hashlib.sha256(text.encode()).hexdigest(),
        n_channels=len(md), n_solver_times=len(times), n_depth_cells=len(groups),
        groups=groups, cell_constant_lower_bound=floor, final_fit=actual,
        lower_bound_fraction_of_final_J=floor['J_data']/actual['J_data'],
        max_within_cell_simulation_spread_strain=max_sim_spread,
        pressure_reconstruction_max_error_Pa=float(np.max(np.abs(pressure['y']-reconstructed))),
        drained_bulk_modulus_Pa=Kd, consistent_solid_bulk_compliance_per_Pa=1/Kd,
        fluid_only_storage_per_Pa=.01/2.2e9,
        declared_biot_storage_per_Pa=storage(1e-11),
        elastic_consistent_biot_storage_per_Pa=storage(1/Kd),
        note='Projection is an optimistic spatial representation bound, not a PDE solution. Initial frame excluded. No solver inputs modified.')
    OUT.mkdir(exist_ok=True)
    (OUT / 'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True, layout='constrained')
    for ax, values, title in zip(axes, [obs, projection, sim],
                                  ['Observation', 'Best cell-constant approximation\n(no flow equation enforced)', 'Final MOOSE prediction']):
        im = ax.pcolormesh(times/1000, md, values*1e6, shading='nearest',
                           vmin=0, vmax=float(obs.max()*1e6), cmap='viridis')
        ax.set(title=title, xlabel='Time [1000 s]', xlim=(0, 400), ylim=(14725, 14710))
    axes[0].set_ylabel('Measured depth [ft]')
    fig.colorbar(im, ax=axes, label='Positive strain [microstrain]')
    fig.savefig(OUT / 'spatial_resolution.png', dpi=170)
    plt.close(fig)
    native = np.load(BASE / 'data/half_wave_positive_native.npz', allow_pickle=True)
    use = (md >= 14715) & (md < 14717)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout='constrained')
    axes[0].plot(pressure['x']/1000, pressure['y']/1e6, color='tab:blue', label='Pressure used in MOOSE')
    axes[0].plot(times/1000, np.interp(times, pressure['x'], pressure['y'])/1e6,
                 '.', color='black', ms=3, label='130 solver output times')
    axes[0].set(ylabel='Pressure change [MPa]', title='Timestamp-aligned pressure and Zone B response')
    axes[1].plot(native['taxis']/1000, native['data'][use].mean(axis=0)*1e6,
                 color='.7', lw=.8, label='Native positive observation, zone mean')
    axes[1].plot(times/1000, obs[use].mean(axis=0)*1e6, color='black', label='130-time observation')
    axes[1].plot(times/1000, sim[use].mean(axis=0)*1e6, color='tab:orange', label='Final fixed-yy prediction')
    axes[1].set(ylabel='Positive strain [microstrain]', xlabel='Time [1000 s]', xlim=(0, 400))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.savefig(OUT / 'pressure_and_zone_b.png', dpi=170)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
