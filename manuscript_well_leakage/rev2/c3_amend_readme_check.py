"""Hostile re-read of the amended C3 README: every number must trace to a file.

    python3 scripts/manuscript_well_leakage/rev2/c3_amend_readme_check.py

Exits 0 when every claim below matches the file it cites. One deliberate mismatch
is expected and is footnoted in the README itself: the published amplitude-
normalised optimum is 549.4 ft^2/s, written 550 throughout because that is the
house-rules baseline and the two agree to 0.1 %.
"""
import csv, json, os, sys
import numpy as np
D='output/rev2_20260901/C3/'
R=json.load(open(D+'c3_amend_results.json'))
M=json.load(open(D+'manifest.json'))
MA=json.load(open(D+'manifest_amend.json'))
ext=list(csv.DictReader(open(D+'c3_amend_extended_window_norms_v2.csv')))
pg=list(csv.DictReader(open(D+'c3_amend_per_gauge_D_v2.csv')))
on=list(csv.DictReader(open(D+'c3_amend_onset_margin_v2.csv')))
gs=list(csv.DictReader(open(D+'c3_amend_grid_sensitivity_v2.csv')))
rs=list(csv.DictReader(open(D+'c3_refit_summary.csv')))
ce=list(csv.DictReader(open(D+'c3_cross_evaluation.csv')))
pt=list(csv.DictReader(open(D+'c3_precursor_timing.csv')))
os_=list(csv.DictReader(open(D+'c3_onset_sharpness.csv')))
pgp=list(csv.DictReader(open(D+'c3_per_gauge_D.csv')))
fails=[]
def chk(name, claim, actual, tol=0.0):
    ok = abs(claim-actual)<=tol if isinstance(claim,(int,float)) else claim==actual
    print(('PASS ' if ok else '*FAIL')+f' {name}: README={claim} file={actual}')
    if not ok: fails.append(name)

def expl(D_,g):
    return 100*float([r for r in ext if abs(float(r['D_ft2_s'])-D_)<0.6 and int(r['gauge'])==g][0]['explained_fraction_of_min'])
def dtm(D_,g):
    return float([r for r in ext if abs(float(r['D_ft2_s'])-D_)<0.6 and int(r['gauge'])==g][0]['sim_t_min_minus_obs_t_min_s'])

# --- headline / 1c table
for D_,vals in ((550.0,[142,157,107,69,34,0]),(728.8,[136,150,110,82,55,92]),
                (1150.0,[127,139,107,90,76,220]),(1603.4,[122,130,102,91,83,273])):
    for g,v in zip(range(2,8),vals):
        chk(f'explained D={D_:.0f} g{g}', v, round(expl(D_,g)), 0.51)
for g,v in zip(range(2,8),[46.5,19.9,7.0,2.9,1.4,2.7]):
    chk(f'explained window-start g{g}', v, round(expl(1126.7489,g),1), 0.051)
# dt ranges
for D_,lo,hi in ((550.0,-486,112),(728.8,-77,92),(1150.0,-126,66),(1603.4,-165,50)):
    d=[dtm(D_,g) for g in range(2,8)]
    chk(f'dt min D={D_:.0f}', lo, round(min(d)), 0.51)
    chk(f'dt max D={D_:.0f}', hi, round(max(d)), 0.51)
# rmse row of the 1c table
for D_,gm,nr in ((550.0,136.8,0.2853),(728.8,116.2,0.2582),(1150.0,89.3,0.3667),(1603.4,81.97,0.5607)):
    r=[x for x in ext if abs(float(x['D_ft2_s'])-D_)<0.6][0]
    chk(f'gm rmse D={D_:.0f}', gm, round(float(r['rmse_gauge_mean_psi_full_window']),2 if D_>1500 else 1), 0.051)
    chk(f'nrm D={D_:.0f}', nr, round(float(r['rmse_normalised_full_window']),4), 1e-4)
chk('window-start gauge-mean RMSE 82.34', 82.34, round(R['r2_uniform_reproduction']['at_published_C3_grid_point']['rmse_gauge_mean_psi'],2), 0.005)
chk('window-start normalised 0.5316', 0.5316, round(R['r2_uniform_reproduction']['at_published_C3_grid_point']['rmse_normalised'],4), 1e-4)

# --- onset table
for r in on:
    g=int(r['gauge'])
    print(f"  g{g} onset {r['onset_utc_60s'][11:19]} plateau {float(r['plateau_width_s']):.0f}s ends {r['plateau_hi_utc'][11:19]} "
          f"lead {float(r['lead_over_pump_restart_s']):.0f}s end-restart {float(r['plateau_hi_minus_pump_restart_s']):+.0f}s resolved={r['ordering_resolved']}")
chk('n resolved', 6, sum(1 for r in on if r['ordering_resolved']=='True'))
chk('g7 lead 103', 103, round(float([r for r in on if r['gauge']=='7'][0]['lead_over_pump_restart_s'])), 0.51)
chk('g7 plateau 317', 317, round(float([r for r in on if r['gauge']=='7'][0]['plateau_width_s'])), 0.51)
chk('g7 plateau end +54', 54, round(float([r for r in on if r['gauge']=='7'][0]['plateau_hi_minus_pump_restart_s'])), 0.51)
chk('g7 drop 0.42', 0.42, round(float([r for r in on if r['gauge']=='7'][0]['drop_from_max_at_restart_psi']),2), 0.005)
chk('g6 drop 49.9', 49.9, round(float([r for r in on if r['gauge']=='6'][0]['drop_from_max_at_restart_psi']),1), 0.05)
chk('g1 drop 896', 896, round(float([r for r in on if r['gauge']=='1'][0]['drop_from_max_at_restart_psi'])), 0.51)
chk('g6 lead 719', 719, round(float([r for r in on if r['gauge']=='6'][0]['lead_over_pump_restart_s'])), 0.51)
chk('g1 lead 1740', 1740, round(float([r for r in on if r['gauge']=='1'][0]['lead_over_pump_restart_s'])), 0.51)

# --- misfit partition
P=R['misfit_partition_conventions']
chk('A pooled 4.6', 4.6, round(P['A']['pooled_sum_of_squares_pct'],1), 0.005)
chk('A mean 15.3', 15.3, round(P['A']['mean_of_per_gauge_shares_pct'],1), 0.005)
chk('A gm 3.9', 3.9, round(P['A']['share_of_gauge_mean_mse_pct'],1), 0.005)
chk('B pooled 7.1', 7.1, round(P['B']['pooled_sum_of_squares_pct'],1), 0.005)
chk('B mean 19.2', 19.2, round(P['B']['mean_of_per_gauge_shares_pct'],1), 0.005)
chk('B gm 6.3', 6.3, round(P['B']['share_of_gauge_mean_mse_pct'],1), 0.005)
chk('A g4 52.0', 52.0, round(P['A']['per_gauge_pct'][2],1), 0.005)
chk('B g4 53.3', 53.3, round(P['B']['per_gauge_pct'][2],1), 0.005)

# --- spread
W=R['window_start_optima']; Q=R['quiescent_start_optima']
chk('window full span 47.57', 47.57, round(W['full']['spread_high_basin']['ratio'],2), 0.005)
chk('quiescent full span 64.8', 64.8, round(Q['full']['spread_high_basin']['ratio'],1), 0.05)
chk('quiescent A span 76.9', 76.9, round(Q['A']['spread_high_basin']['ratio'],1), 0.05)
chk('quiescent B span 81.4', 81.4, round(Q['B']['spread_high_basin']['ratio'],1), 0.05)
chk('quiescent global span 524', 524, round(Q['full']['spread_global']['ratio']), 0.51)
g7q=[r for r in Q['full']['per_gauge'] if r['gauge']==7][0]
chk('g7 low basin D 64', 64, round(g7q['D_ft2_s']), 0.51)
chk('g7 low basin rmse 13.76', 13.76, round(g7q['rmse_psi'],2), 0.005)
chk('g7 high basin D 517', 517, round(g7q['D_high_basin_ft2_s']), 0.51)
chk('g7 high basin rmse 14.77', 14.77, round(g7q['rmse_high_basin_psi'],2), 0.005)
chk('g7 basin gap 7.4', 7.4, round(g7q['second_basin_penalty_pct'],1), 0.05)
g7b=[r for r in Q['B']['per_gauge'] if r['gauge']==7][0]
chk('g7 ruleB low 80/11.95', 80, round(g7b['D_ft2_s']), 0.51)
chk('g7 ruleB low rmse 11.95', 11.95, round(g7b['rmse_psi'],2), 0.005)
chk('g7 ruleB high 412/16.86', 412, round(g7b['D_high_basin_ft2_s']), 0.51)
chk('g7 ruleB high rmse 16.86', 16.86, round(g7b['rmse_high_basin_psi'],2), 0.005)
for g,wv,qv,mv in ((2,2.00e4,3.35e4,67),(3,6920,8700,26),(4,1190,1680,41),(5,673,937,39),(6,635,855,35),(7,421,517,23)):
    a=[r for r in W['full']['per_gauge'] if r['gauge']==g][0]['D_high_basin_ft2_s']
    b=[r for r in Q['full']['per_gauge'] if r['gauge']==g][0]['D_high_basin_ft2_s']
    chk(f'g{g} window D', wv, float(f'{a:.3g}'), max(abs(wv)*0.005,0.5))
    chk(f'g{g} quiescent D', qv, float(f'{b:.3g}'), max(abs(qv)*0.005,0.5))
    chk(f'g{g} move %', mv, round(100*(b/a-1)), 0.51)
# rmse columns of 3b table
for g,a_,b_ in ((2,8.83,10.87),(3,11.88,14.52),(4,7.48,3.52),(5,15.05,5.54),(6,18.02,8.40),(7,6.08,14.77)):
    chk(f'g{g} rmse window', a_, round([r for r in W['full']['per_gauge'] if r['gauge']==g][0]['rmse_psi'],2), 0.005)
    chk(f'g{g} rmse quiescent', b_, round([r for r in Q['full']['per_gauge'] if r['gauge']==g][0]['rmse_high_basin_psi'],2), 0.005)

# --- grid sensitivity / optima
chk('amend full absD 1140', 1140, float(f"{W['full']['uniform_absolute_norm']['D_ft2_s']:.3g}"), 5)
chk('amend A absD 1000', 1000, float(f"{W['A']['uniform_absolute_norm']['D_ft2_s']:.3g}"), 5)
chk('amend B absD 1020', 1020, float(f"{W['B']['uniform_absolute_norm']['D_ft2_s']:.3g}"), 5)
chk('amend full normD 548', 548, float(f"{W['full']['uniform_normalised_norm']['D_ft2_s']:.3g}"), 0.5)
chk('amend A normD 517', 517, float(f"{W['A']['uniform_normalised_norm']['D_ft2_s']:.3g}"), 0.5)
d=[abs(float(r['difference_pct'])) for r in gs]
chk('max grid diff <=1.7', True, max(d)<=1.75)
chk('median grid diff 0.4', 0.4, round(float(np.median(d)),1), 0.005)
chk('published span full 46.97', 46.97, round(M['results']['per_gauge_spread']['full']['ratio'],2), 0.005)
chk('published span B 46.47', 46.47, round(M['results']['per_gauge_spread']['B_ten_percent']['ratio'],2), 0.005)
chk('amend span B 46.49', 46.49, round(W['B']['spread_high_basin']['ratio'],2), 0.005)
chk('published contraction -1.07', -1.07, round(100*(M['results']['per_gauge_spread']['B_ten_percent']['ratio']/M['results']['per_gauge_spread']['full']['ratio']-1),2), 0.005)
chk('amend contraction -2.26', -2.26, round(100*(W['B']['spread_high_basin']['ratio']/W['full']['spread_high_basin']['ratio']-1),2), 0.005)
# published optima 3sf
chk('published full absD 1130', 1130, float(f"{float(rs[0]['D_absolute_norm_ft2_s']):.3g}"), 5)
chk('published full normD 550', 550, float(f"{float(rs[0]['D_normalised_norm_ft2_s']):.3g}"), 0.5)
chk('published A absD 1000', 1000, float(f"{float(rs[1]['D_absolute_norm_ft2_s']):.3g}"), 5)
chk('published A normD 520', 520, float(f"{float(rs[1]['D_normalised_norm_ft2_s']):.3g}"), 0.5)
chk('published pooled D 1090', 1090, float(f"{float(rs[0]['D_pooled_norm_ft2_s']):.3g}"), 5)
chk('published pooled rmse 78.0', 78.0, round(float(rs[0]['rmse_pooled_psi_at_pooled_optimum']),1), 0.005)
chk('rmse full 82.3', 82.3, round(float(rs[0]['rmse_gauge_mean_psi']),1), 0.005)
chk('rmse A 94.1', 94.1, round(float(rs[1]['rmse_gauge_mean_psi']),1), 0.005)
chk('rmse B 97.1', 97.1, round(float(rs[2]['rmse_gauge_mean_psi']),1), 0.005)
chk('nrm full 0.230', 0.230, round(float(rs[0]['rmse_normalised']),3), 5e-4)
chk('nrm A 0.262', 0.262, round(float(rs[1]['rmse_normalised']),3), 5e-4)
chk('nrm B 0.269', 0.269, round(float(rs[2]['rmse_normalised']),3), 5e-4)
chk('n residuals full 3925', 3925, int(rs[0]['n_residuals']))
chk('n residuals A 2640', 2640, int(rs[1]['n_residuals']))
chk('n residuals B 2418', 2418, int(rs[2]['n_residuals']))
# --- R2
r2=R['r2_uniform_reproduction']
chk('R2 D 1138.944', 1138.944, round(r2['D_ft2_s'],3), 5e-4)
chk('R2 rmse exact', r2['r2_manifest_rmse_psi'], r2['rmse_gauge_mean_psi'])
chk('R2 nrm exact', r2['r2_manifest_rmse_normalised'], r2['rmse_normalised'])
chk('R2 offset -1.07', -1.07, round(r2['at_published_C3_grid_point']['D_offset_pct'],2), 0.005)
# --- cross evaluation
chk('cross A gm +1.24', 1.24, round(float([r for r in ce if r['scored_on']=='A_zero_crossing' and r['metric']=='gauge_mean_rmse_psi'][0]['penalty_using_full_window_D_pct']),2), 0.005)
chk('cross B gm +1.13', 1.13, round(float([r for r in ce if r['scored_on']=='B_ten_percent' and r['metric']=='gauge_mean_rmse_psi'][0]['penalty_using_full_window_D_pct']),2), 0.005)
chk('cross A nrm +1.29', 1.29, round(float([r for r in ce if r['scored_on']=='A_zero_crossing' and r['metric']=='normalised_rmse'][0]['penalty_using_full_window_D_pct']),2), 0.005)
chk('cross B nrm +1.26', 1.26, round(float([r for r in ce if r['scored_on']=='B_ten_percent' and r['metric']=='normalised_rmse'][0]['penalty_using_full_window_D_pct']),2), 0.005)
# --- quiescence, counts, misc
chk('g3 quiescence 0.463', 0.463, round(R['quiescence_ptp_psi']['g3'],3), 5e-4)
chk('g4 quiescence 0.224', 0.224, round(R['quiescence_ptp_psi']['g4'],3), 5e-4)
chk('g7 quiescence 0.017', 0.017, round(R['quiescence_ptp_psi']['g7'],3), 5e-4)
chk('published n_forward_solves 423', 423, M['numerics']['n_forward_solves'])
chk('amend n_forward_solves 1656', 1656, R['n_forward_solves'])
chk('total 2080', 2080, 423+1+R['n_forward_solves'])
chk('grid n 823', 823, R['grid']['n_points'])
chk('grid spacing 1.149', 1.149, round(R['grid']['spacing_pct'],3), 5e-4)
chk('pump start', '2020-03-16T11:18:57', R['pumping_start_utc'])
chk('prev shutin', '2020-03-16T10:50:34', R['preceding_cycle_shutin_utc'])
chk('prev cycle rate 22.8', 22.8, round(R['preceding_cycle']['max_rate_bpm'],1), 0.05)
# frac kept
for g,a_,b_ in zip(range(2,8),[88,86,77,63,52,39],[82,80,72,57,47,33]):
    fa=float([r for r in pgp if r['criterion']=='A_zero_crossing' and int(r['gauge'])==g][0]['frac_kept'])
    fb=float([r for r in pgp if r['criterion']=='B_ten_percent' and int(r['gauge'])==g][0]['frac_kept'])
    chk(f'frac kept A g{g}', a_, round(100*fa), 0.51)
    chk(f'frac kept B g{g}', b_, round(100*fb), 0.51)
# g4 single-fit rmse drop
chk('g4 rmse full 7.47', 7.47, round(float([r for r in pgp if r['criterion']=='full' and r['gauge']=='4'][0]['rmse_psi']),2), 0.005)
chk('g4 rmse B 1.46', 1.46, round(float([r for r in pgp if r['criterion']=='B_ten_percent' and r['gauge']=='4'][0]['rmse_psi']),2), 0.005)
# lag from arrival
chk('g1 lag arrival -116.3', -116.3, round(float([r for r in pt if r['gauge']=='1'][0]['lag_min_from_arrival_rel10pct_s']),1), 0.005)
chk('g2 lag arrival -126.3', -126.3, round(float([r for r in pt if r['gauge']=='2'][0]['lag_min_from_arrival_rel10pct_s']),1), 0.005)
chk('g7 lag arrival -357.7', -357.7, round(float([r for r in pt if r['gauge']=='7'][0]['lag_min_from_arrival_rel10pct_s']),1), 0.005)
# implied D
for g,v in ((4,1441),(5,1389),(6,1720),(7,1541),(3,3654)):
    chk(f'implied D g{g}', v, round(float([r for r in os_ if int(r['gauge'])==g][0]['implied_D_x2_over_lag_ft2_s'])), 0.51)
chk('g1 lag prev shutin -37', -37.16, round(float([r for r in os_ if r['gauge']=='1'][0]['lag_from_prev_cycle_shutin_s']),2), 0.005)
chk('g7 lag prev shutin 1600', 1600, round(float([r for r in os_ if r['gauge']=='7'][0]['lag_from_prev_cycle_shutin_s'])), 0.51)
# norm comparisons
chk('quiescent 81.97 vs window 82.33 = -0.44%', -0.44, round(100*(Q['full']['uniform_absolute_norm']['value']/W['full']['uniform_absolute_norm']['value']-1),2), 0.005)
chk('normalised 12.4% worse', 12.4, round(100*(Q['full']['uniform_normalised_norm']['value']/W['full']['uniform_normalised_norm']['value']-1),1), 0.05)
chk('quiescent normalised 0.2582', 0.2582, round(Q['full']['uniform_normalised_norm']['value'],4), 1e-4)
chk('window normalised 0.2298', 0.2298, round(W['full']['uniform_normalised_norm']['value'],4), 1e-4)
EXPECTED = ['published full normD 550']   # documented by a footnote in the README
unexpected = [f for f in fails if f not in EXPECTED]
print()
print('mismatches:', fails if fails else 'none',
      '| expected-and-footnoted:', EXPECTED)
print('UNEXPECTED FAILURES:', unexpected if unexpected else 'none')
sys.exit(1 if unexpected else 0)
