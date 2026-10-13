"""Refine, search continuous center/width locally, then conditionally invert.

Run after (or while waiting for) continuous_geometry.py controls. Each stage is
checkpointed. A failed mesh/fit/gradient check prevents automatic inversion.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

import continuous_geometry as model

OUT=model.OUT


def result(folder):
    return json.loads((folder/'summary.json').read_text())


def compare_mesh(coarse, fine):
    a=pd.read_csv(coarse/'bg_18_strain.csv');b=pd.read_csv(fine/'bg_18_strain.csv')
    np.testing.assert_allclose(a[model.KEYS],b[model.KEYS],rtol=1e-10,atol=1e-18)
    use=a.measurement_time.to_numpy()>0
    sa=a.simulation_values.to_numpy()[use];sb=b.simulation_values.to_numpy()[use]
    relative=float(np.linalg.norm(sa-sb)/np.linalg.norm(sb))
    ja=result(coarse)['results'][0]['weighted_objective_after']
    jb=result(fine)['results'][0]['weighted_objective_after']
    change=abs(ja-jb)/jb
    return dict(coarse=str(coarse),fine=str(fine),prediction_relative_difference=relative,
        weighted_objective_relative_difference=change,passed=bool(relative<=.05 and change<=.05))


def batch(specifications):
    folders=[model.prepare_case(*s) for s in specifications]
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(model.run_case,folders))
    return folders


def refine_selected():
    """Continue a failed final mesh check without repeating the geometry search."""
    selection=json.loads((OUT/'selected_geometry.json').read_text())
    assert not selection['ready_for_inversion']
    assert not (OUT/'fixed_geometry_inversion').exists()
    with (OUT/'EXTRA_REFINEMENT_STARTED').open('x') as f:f.write(f'pid={os.getpid()}\n')
    model.base.save(OUT/'selection_before_extra_refinement.json',selection)
    previous=Path(selection['folder']);previous_baseline=Path(selection['baseline_folder'])
    g=result(previous)['geometry'];bg=result(previous_baseline)['geometry']
    h=g['h_y_max_m']/2
    best=model.prepare_case('best_geometry_extra_half_h',g['center_ft'],g['width_ft'],h)
    baseline=model.prepare_case('baseline_geometry_extra_half_h',bg['center_ft'],bg['width_ft'],h)
    model.base.status(OUT/'pipeline_status.json','selected_geometry_extra_refinement',h_y_max_m=h,
        previous_prediction_difference=selection['final_mesh_check']['prediction_relative_difference'],
        best_folder=str(best),baseline_folder=str(baseline))
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(model.run_case,[best,baseline]))
    final_check=compare_mesh(previous,best);baseline_check=compare_mesh(previous_baseline,baseline)
    r=result(best);rb=result(baseline)
    gain=1-r['results'][0]['weighted_objective_after']/rb['results'][0]['weighted_objective_after']
    core_improved=r['results'][0]['core_rms_after_microstrain']<rb['results'][0]['core_rms_after_microstrain']
    ready=bool(final_check['passed'] and baseline_check['passed'] and gain>=.02 and core_improved)
    updated=dict(folder=str(best),baseline_folder=str(baseline),geometry=r['geometry'],results=r['results'],
        relative_objective_improvement=gain,final_mesh_check=final_check,baseline_mesh_check=baseline_check,
        ready_for_inversion=ready,previous_selection_file=str(OUT/'selection_before_extra_refinement.json'))
    model.base.save(OUT/'selected_geometry.json',updated)
    for ext in ['png','pdf']:shutil.copy2(best/f'qc_bg_18_only.{ext}',OUT/f'qc_selected_geometry.{ext}')
    if not ready:
        model.base.status(OUT/'pipeline_status.json','geometry_review_needed_inversion_not_started',selection=updated)
        return
    model.base.status(OUT/'pipeline_status.json','fixed_geometry_inversion_starting',selection=updated)
    with (OUT/'inversion_driver.log').open('x') as log:
        subprocess.run([str(model.base.PYTHON_BIN/'python'),'-u',str(model.HERE/'continuous_geometry_inversion.py')],
            check=True,stdout=log,stderr=subprocess.STDOUT)
    model.base.status(OUT/'pipeline_status.json','complete')


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'PIPELINE_STARTED').open('x') as f:f.write(f'pid={os.getpid()}\n')
    model.base.save(OUT/'search_protocol.json',dict(
        variables=['center_depth_ft','zone_depth_width_ft'],
        scope='Local continuous-depth geometry search; all permeability components fixed during this stage',
        search_steps_ft=[[.08,.16],[.04,.08]],
        core_channel_depths_ft=json.loads((model.REFERENCE/'summary.json').read_text())['central_channel_depths_ft'],
        core_sse_weight=3,coupling='Exact unbounded scalar least squares at each forward',
        mesh_gate='<=5% change in coupled predictions AND weighted objective on halving local dy',
        auto_inversion_gate='>=2% lower weighted objective than same-mesh initial geometry; core RMS must improve; final mesh check must pass',
        inversion='Freeze best geometry; L-BFGS-B xx/yy only, C analytically profiled, up to 60 accepted iterations after two directional FD step checks',
        note='Local search, not a claim of global geometric optimality; existing pressure overshoot recorded separately'))
    controls=[OUT/'mesh_h050',OUT/'mesh_h025']
    while not all((p/'COMPLETE').exists() for p in controls):
        for p in controls:
            if (p/'status.json').exists() and json.loads((p/'status.json').read_text())['stage']=='failed':
                raise RuntimeError('Mesh-control forward failed')
        model.base.status(OUT/'pipeline_status.json','waiting_for_mesh_controls')
        time.sleep(20)
    baseline=controls[-1];manifest=result(baseline)['geometry'];checks=[]
    check=compare_mesh(*controls);checks.append(check)
    for h in [.0125,.00625]:
        if check['passed']:break
        model.base.status(OUT/'pipeline_status.json','additional_mesh_refinement',h=h)
        finer=model.prepare_case(f'mesh_h{h:.5f}',manifest['center_ft'],manifest['width_ft'],h)
        model.run_case(finer)
        check=compare_mesh(baseline,finer);checks.append(check);baseline=finer
    model.base.save(OUT/'mesh_checks.json',checks)
    if not check['passed']:
        model.base.status(OUT/'pipeline_status.json','mesh_not_stable_inversion_not_started',checks=checks)
        return
    best=baseline;all_cases=[baseline]
    h=result(baseline)['geometry']['h_y_max_m']
    for round_index,(dc,dw) in enumerate([(.08,.16),(.04,.08)],1):
        center=result(best)['geometry']['center_ft'];width=result(best)['geometry']['width_ft']
        candidates=[(center-dc,width),(center+dc,width),(center,width-dw),(center,width+dw)]
        specs=[(f'search_r{round_index}_{i+1}',c,w,h) for i,(c,w) in enumerate(candidates)]
        model.base.status(OUT/'pipeline_status.json','geometry_search',round=round_index,
            center_ft=center,width_ft=width,candidates=candidates)
        all_cases.extend(batch(specs))
        best=min(all_cases,key=lambda p:result(p)['results'][0]['weighted_objective_after'])
        model.base.save(OUT/'best_so_far.json',dict(folder=str(best),**result(best)))
        rows=[]
        for folder in all_cases:
            r=result(folder);g=r['geometry']
            rows.append(dict(case=folder.name,center_ft=g['center_ft'],width_ft=g['width_ft'],
                lo_ft=g['lo_ft'],hi_ft=g['hi_ft'],h_y_max_m=g['h_y_max_m'],**r['results'][0]))
        pd.DataFrame(rows).to_csv(OUT/'geometry_comparison.csv',index=False)
    g=result(best)['geometry']
    model.base.status(OUT/'pipeline_status.json','best_geometry_mesh_check',best=str(best))
    final_best=model.prepare_case('best_geometry_half_h',g['center_ft'],g['width_ft'],h/2)
    if best==baseline:
        model.run_case(final_best)
        final_baseline=final_best
    else:
        bg=result(baseline)['geometry']
        final_baseline=model.prepare_case('baseline_geometry_half_h',bg['center_ft'],bg['width_ft'],h/2)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(model.run_case,[final_best,final_baseline]))
    final_check=compare_mesh(best,final_best)
    baseline_check=compare_mesh(baseline,final_baseline)
    r=result(final_best);rb=result(final_baseline)
    gain=1-r['results'][0]['weighted_objective_after']/rb['results'][0]['weighted_objective_after']
    core_improved=r['results'][0]['core_rms_after_microstrain']<rb['results'][0]['core_rms_after_microstrain']
    ready=bool(final_check['passed'] and baseline_check['passed'] and gain>=.02 and core_improved)
    selection=dict(folder=str(final_best),baseline_folder=str(final_baseline),
        geometry=r['geometry'],results=r['results'],relative_objective_improvement=gain,
        final_mesh_check=final_check,baseline_mesh_check=baseline_check,ready_for_inversion=ready)
    model.base.save(OUT/'selected_geometry.json',selection)
    for ext in ['png','pdf']:
        shutil.copy2(final_best/f'qc_bg_18_only.{ext}',OUT/f'qc_selected_geometry.{ext}')
    if not ready:
        model.base.status(OUT/'pipeline_status.json','geometry_review_needed_inversion_not_started',selection=selection)
        return
    model.base.status(OUT/'pipeline_status.json','fixed_geometry_inversion_starting',selection=selection)
    command=[str(model.base.PYTHON_BIN/'python'),'-u',str(model.HERE/'continuous_geometry_inversion.py')]
    with (OUT/'inversion_driver.log').open('w') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    model.base.status(OUT/'pipeline_status.json','complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refine-selected',action='store_true')
    args=parser.parse_args()
    try:
        if args.refine_selected:refine_selected()
        else:main()
    except Exception as exc:
        model.base.status(OUT/'pipeline_status.json','failed',reason=f'{type(exc).__name__}: {exc}')
        raise
