"""Two local DSS waterfalls with individual native channel curves to the right."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'swell_pressure_channel_scan'
SOURCE = ROOT/'data/fiberis_format/s_well/dss_data/Mariner 14x-36-POW-S - RFS strain change.npz'
GAUGE = ROOT/'data/fiberis_format/prod/gauges/pressure_g1.npz'


def main():
    data = DSS2D()
    data.load_npz(str(SOURCE))
    ref = (data.daxis > 7500) & (data.daxis < 15000)
    data.data -= np.median(data.data[ref], axis=0)[None, :]
    data.select_time(0, 400000)
    t, md, strain = data.taxis, data.daxis.astype(float), data.data
    # Ensure these plots show the same corrected native signals as the scan.
    cached = np.load(OUT/'candidate_native_curves.npz', allow_pickle=True)
    np.testing.assert_array_equal(t, cached['taxis'])
    gauge = np.load(GAUGE, allow_pickle=True)
    gt = gauge['taxis']+(gauge['start_time'].item()-data.start_time).total_seconds()
    gp = gauge['data']*.00689476
    gp -= np.interp(0, gt, gp)
    pressure_mask = (gt >= 0) & (gt <= t[-1])
    manifest = dict(time_end_s=float(t[-1]), native_times=len(t),
                    strain_units='microstrain, signed; no normalization, smoothing, or rectification',
                    pressure_units='MPa change relative to DSS start, separate right axis',
                    pressure_source=str(GAUGE), source=str(SOURCE),
                    drift='Per-time median over 7500 < MD < 15000 ft',
                    layout='Left: all native channels within +/-8 ft; right: 13 consecutive native channels within about +/-2 ft',
                    waterfall_limits_microstrain=[-5,5], cases=[])
    for proposed in [14887.6357421875, 14375.943359375]:
        center = int(np.argmin(abs(md-proposed)))
        existing = int(np.argmin(abs(cached['daxis']-md[center])))
        np.testing.assert_array_equal(strain[center], cached['data'][existing])
        local = np.where((md >= md[center]-8) & (md <= md[center]+8))[0]
        channels = np.arange(center-6, center+7)
        tag = str(round(md[center]))
        fig = plt.figure(figsize=(17, 15), layout='constrained')
        grid = fig.add_gridspec(14, 2, width_ratios=[1, 1.35], height_ratios=[1]*13+[.24])
        waterfall = fig.add_subplot(grid[:13, 0])
        im = waterfall.pcolormesh(t/1000, md[local], strain[local], shading='nearest',
                                  cmap='bwr', vmin=-5, vmax=5, rasterized=True)
        waterfall.set(xlim=(0,400), ylim=(md[local].max(),md[local].min()),
                       title=f'Local signed DSS | {len(local)} native channels',
                       xlabel='Time since DSS start [1000 s]', ylabel='Measured depth [ft]')
        waterfall.axvspan(t[-1]/1000,400,color='.88',hatch='///')
        waterfall.set_xticks(np.arange(0,401,100))
        cax = fig.add_subplot(grid[13,0])
        fig.colorbar(im,cax=cax,orientation='horizontal',label='Signed strain change [microstrain]',extend='both')
        ymin, ymax = float(strain[channels].min()), float(strain[channels].max())
        padding = .08*(ymax-ymin)
        ylim = (min(ymin-padding,-.2),ymax+padding)
        first_ax = None
        handles = None
        for row, channel in enumerate(channels):
            highlight = channel == center
            waterfall.axhline(md[channel], color='black' if highlight else '.4',
                              linewidth=1.4 if highlight else .5, linestyle='-' if highlight else '--',alpha=.8)
            waterfall.text(394,md[channel],f'{row+1:02d}',ha='right',va='center',fontsize=8,
                           bbox=dict(facecolor='#ffe8a1' if highlight else 'white',edgecolor='none',pad=.5))
            ax = fig.add_subplot(grid[row,1],sharex=first_ax)
            if first_ax is None:
                first_ax = ax
            ax.set_facecolor('#fff7dc' if highlight else 'white')
            line, = ax.plot(t/1000,strain[channel],color='#1765ad',linewidth=.75,label='Signed strain')
            ax.axhline(0,color='.55',linewidth=.45)
            ax.set(ylim=ylim,xlim=(0,400))
            ax.set_yticks([0,round(ymax*.5,1),round(ymax,1)])
            ax.tick_params(axis='both',labelsize=8)
            ax.grid(axis='x',alpha=.18)
            ax.axvspan(t[-1]/1000,400,color='.9')
            ax.text(.015,.87,f'{row+1:02d} | {md[channel]:.3f} ft'+('  [candidate]' if highlight else ''),
                    transform=ax.transAxes,fontsize=9,va='top',
                    bbox=dict(facecolor='#fff7dc' if highlight else 'white',alpha=.8,edgecolor='none',pad=.4))
            pressure_ax = ax.twinx()
            pline, = pressure_ax.plot(gt[pressure_mask]/1000,gp[pressure_mask],color='#cc3b30',
                                      linewidth=1,alpha=.8,label='Pressure change')
            pressure_ax.set(ylim=(-.15,2.45),yticks=[0,1,2])
            pressure_ax.tick_params(axis='y',labelsize=8,labelcolor='#cc3b30')
            if row == 6:
                ax.set_ylabel('Signed strain [microstrain]',color='#1765ad',labelpad=14)
                pressure_ax.set_ylabel('Pressure change [MPa]',color='#cc3b30',labelpad=10)
            if row < len(channels)-1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Time since DSS start [1000 s]')
            handles = [line,pline]
        legend_ax = fig.add_subplot(grid[13,1]);legend_ax.axis('off')
        legend_ax.legend(handles=handles,loc='center',ncol=2,frameon=False,fontsize=10)
        fig.suptitle(f'S-well RFSDSS | candidate {md[center]:.3f} ft\n'
                     'Waterfall at left; 13 individual channels at right, shallow to deep\n'
                     'Native data, existing drift correction | shared strain scale within each figure | pressure on separate axis, no time shift',
                     fontsize=13)
        stem=f'qc_region_{tag}_waterfall_channels'
        fig.savefig(OUT/(stem+'.png'),dpi=180)
        fig.savefig(OUT/(stem+'.pdf'))
        plt.close(fig)
        manifest['cases'].append(dict(candidate_ft=float(md[center]),
            waterfall_depth_ft=[float(md[local].min()),float(md[local].max())],
            individual_channels_ft=md[channels].tolist(), strain_axis_limits_microstrain=list(ylim),
            waterfall_clipped_fraction=float(np.mean(abs(strain[local])>5)),png=stem+'.png',pdf=stem+'.pdf'))
        print(OUT/(stem+'.png'),flush=True)
    (OUT/'channel_qc_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
