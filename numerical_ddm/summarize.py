"""Generate all research summary tables from actual saved solver outputs."""
import json,re,sys
from pathlib import Path
import numpy as np
from run_validation import ROOT,metrics,reference
from fiberis.moose.numerical_ddm import Element,axial_response,time_derivative
import matplotlib.pyplot as plt
OUT=ROOT/'output/numerical_ddm'


def execute():
    results={}
    for path in sorted(OUT.glob('*/metrics.json')):
        if path.parent.name=='extensions': continue
        m=json.loads(path.read_text())
        if 'displacement' not in m or 'strain' not in m:
            continue
        runpath=path.parent/'run.json'
        run=json.loads(runpath.read_text()) if runpath.exists() else {}
        if not run.get('success'):
            continue
        log=re.sub(r'\x1b\[[0-9;]*m','',(path.parent/'log.txt').read_text())
        dofs=re.search(r'Num DOFs:\s+(\d+)',log)
        version=re.search(r'MOOSE Version:\s+([^\n]+)',log)
        mem=re.findall(r'\[\s*(\d+) MB\]',log)
        m['run']=run
        m['dofs']=int(dofs.group(1)) if dofs else None
        m['moose_version']=version.group(1) if version else None
        m['reported_process_peak_memory_MB']=max(map(int,mem)) if mem else None
        results[path.parent.name]=m
    (OUT/'all_metrics.json').write_text(json.dumps(results,indent=2))
    groups={
        'mesh':(['opening_h20_L800_t2','opening_h10_L800_t2','opening_h7p5_L800_t2'],[20,10,7.5],'Core maximum cell size (m)'),
        'domain':(['opening_h10_L400_t2','opening_h10_L800_t2','opening_h10_L1600_t2'],[400,800,1600],'Domain half extent (m)'),
        'thickness':(['opening_h10_L800_t1','opening_h10_L800_t2','opening_h10_L800_t4'],[1,2,4],'Band thickness (m)'),
        'band_cells':(['opening_h10_L800_t2_b1','opening_h10_L800_t2','opening_h10_L800_t2_b4'],[1,2,4],'Elements across band'),
    }
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for ax,(kind,(names,x,label)) in zip(axs.flat,groups.items()):
        valid=[(n,v) for n,v in zip(names,x) if n in results]
        for quantity in ['displacement','strain']:
            ax.plot([v for n,v in valid],[100*results[n][quantity]['relative_l2'] for n,v in valid],'o-',label=quantity)
        ax.axhline(1,color='gray',linestyle=':',label='1% target')
        ax.set(xlabel=label,ylabel='Relative L2 error (%)');ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(OUT/'convergence.png',dpi=180);fig.savefig(OUT/'convergence.pdf');plt.close(fig)
    checks={}
    def saved(name): return np.load(OUT/name/'comparison.npz')
    if all(n in results for n in ['double_h10_L800_t2','opening_h10_L800_t2']):
        checks['opening_linearity']=metrics(saved('double_h10_L800_t2')['u'],2*saved('opening_h10_L800_t2')['u'])
    if all(n in results for n in ['mixed_h10_L800_t2','s1_h10_L800_t2','s2_h10_L800_t2','opening_h10_L800_t2']):
        expected=saved('opening_h10_L800_t2')['u']+.4*saved('s1_h10_L800_t2')['u']-.6*saved('s2_h10_L800_t2')['u']
        checks['mixed_superposition']=metrics(saved('mixed_h10_L800_t2')['u'],expected)
    if 'zero_h10_L800_t2' in results:
        checks['zero_load_max_displacement']=float(abs(saved('zero_h10_L800_t2')['u']).max())
    checks['numerical_setting_changes']={}
    base='opening_h10_L800_t2'
    for name in ['opening_h10_L800_t2_q7','opening_h10_L800_t2_b1','opening_h10_L800_t2_b4']:
        if name in results and base in results:
            checks['numerical_setting_changes'][name+'_versus_default']={
                'displacement':metrics(saved(name)['u'],saved(base)['u']),
                'strain':metrics(saved(name)['strain'],saved(base)['strain'])}
    # Spatial postprocessing: independent finer DDMPY sampling, fixed coarse
    # receiver points. Endpoints are assessed separately because they repeat.
    if 'opening_h10_L800_t2' in results:
        data=saved('opening_h10_L800_t2');p=data['points'];e=Element(100,60,.001)
        derivs=[]
        for delta in [.01,.005]:
            shift=np.array([delta,0.,0.])
            derivs.append((reference([e],p+shift)[:,0]-reference([e],p-shift)[:,0])/(2*delta))
        checks['spatial_operator_interior']=metrics(data['reference_strain'][1:-1],derivs[-1][1:-1])
        checks['spatial_operator_including_repeated_endpoints']=metrics(data['reference_strain'],derivs[-1])
        checks['derivative_reference_step_check']=metrics(derivs[0],derivs[1])
        checks['fe_plus_spatial_operator_vs_continuum_derivative']={}
        for name in ['opening_h20_L800_t2','opening_h10_L800_t2','opening_h7p5_L800_t2']:
            if name in results:
                checks['fe_plus_spatial_operator_vs_continuum_derivative'][name]=metrics(saved(name)['strain'],derivs[-1])
        # A distributed sheet integral isolates the regularization error in
        # the infinite-domain reference. It NEVER supplies FE boundary data.
        g,w=np.polynomial.legendre.leggauss(16)
        checks['band_regularization_reference']={}
        for thickness in [1,2,4]:
            integrated=np.zeros_like(p)
            for gi,wi in zip(g,w):
                sheet=Element(100,60,.001).set_global_coors(0,0,gi*thickness/2)
                integrated+=wi/2*reference([sheet],p)
            _,_,eps=axial_response(p,integrated)
            checks['band_regularization_reference'][str(thickness)]={
                'displacement':metrics(integrated,data['reference']),
                'strain':metrics(eps,data['reference_strain'])}
        # Time truncation measured on an independently defined smooth signal.
        checks['time_operator_cubic']={}
        for n in [9,17,33]:
            t=np.linspace(0,1,n)**1.2
            checks['time_operator_cubic'][str(n)]=metrics(time_derivative(t[None,:]**3,t),3*t[None,:]**2)
    geometry_history=OUT/'extensions/history_geometry/comparison.npz'
    if geometry_history.exists():
        d=np.load(geometry_history)
        strain_error=d['strain']-d['reference']
        rate_error=d['rate']-d['reference_rate']
        checks['geometry_history_rate_error_identity_max_abs']=float(abs(
            time_derivative(strain_error,d['times'])-rate_error).max())
        checks['geometry_history_relative_error_amplification']=float(
            (np.linalg.norm(rate_error)/np.linalg.norm(d['reference_rate'])) /
            (np.linalg.norm(strain_error)/np.linalg.norm(d['reference'])))
    (OUT/'physical_checks.json').write_text(json.dumps(checks,indent=2))
    lines=['# Actual numerical DD validation results','',
        'Generated from saved CSV, metrics and solver logs. Relative L2 and peak-normalized maximum errors include all 101 preselected receivers. No pointwise relative-error filtering.','',
        '| Case | u L2 (%) | axial strain L2 (%) | u peak max (%) | strain peak max (%) | DOFs | wall s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def percent(m,key):
        x=m[key]
        return 'undefined (zero reference)' if x is None else f'{x*100:.5f}'
    for name,m in results.items():
        lines.append(f"| {name} | {percent(m['displacement'],'relative_l2')} | {percent(m['strain'],'relative_l2')} | {percent(m['displacement'],'peak_normalized_max')} | {percent(m['strain'],'peak_normalized_max')} | {m['dofs']} | {m['run'].get('elapsed_seconds',float('nan')):.1f} |")
    lines+=['','Displacement L2 uses the norm of the full N×3 array. Componentwise errors are retained in each metrics.json. Strain uses the common corrected observation operator; the original DDMPY field is unchanged.','',
            '![Convergence](convergence.png)','',
            'Physical, spatial-operator, regularization and time-operator checks: [physical_checks.json](physical_checks.json).',
            'Original reference bugs and hashes: [reference_audit](reference_audit).',
            'Logs report per-process MOOSE memory; the reported process maximum is not total MPI memory. Child resource usage includes MPI launcher/children as reported by the OS; repeated-process maximum RSS is a high-water mark, not additive.']
    ext=OUT/'extensions/metrics.json'
    if ext.exists():
        lines+=['','## Extended actual FE results','',
                '| Case / well | u L2 (%) | strain L2 (%) |','|---|---:|---:|']
        for name,reports in json.loads(ext.read_text()).items():
            if isinstance(reports,list):
                for i,m in enumerate(reports):
                    lines.append(f"| {name} / {i} | {100*m['displacement']['relative_l2']:.5f} | {100*m['strain']['relative_l2']:.5f} |")
        lines+=['','Full rotation, time-history and strain-rate errors: [extensions/metrics.json](extensions/metrics.json).']
        for name in ['history_fixed','history_geometry']:
            m=json.loads(ext.read_text()).get(name)
            if m:
                lines += ['', f"{name}: strain L2 = {100*m['strain']['relative_l2']:.5f}%; strain-rate L2 = {100*m['rate']['relative_l2']:.5f}%."]
    notebook=OUT/'notebook_existing_tensile_shear'
    if notebook.exists():
        lines += ['', '## Original 0930 notebook reproduction', '',
                  'Source: `DDMpy_0930/Examples/Horizontal well existing tensile and shear.ipynb`. All 300 receivers retained. Original 30 frames independently evaluated using the intact 0930 package. Raw notebook times interpreted as minutes following its axis label; SI rates are per second.', '',
                  '| FE run | frames | u L2 (%) | strain L2 (%) | legacy rate L2 (%) | wall s |',
                  '|---|---:|---:|---:|---:|---:|']
        for path in sorted(notebook.glob('*/metrics.json')):
            m=json.loads(path.read_text())
            if 'displacement_all_times' not in m: continue
            run=json.loads((path.parent/'run.json').read_text())
            lines.append(f"| {path.parent.name} | {m['completed_notebook_frames']} | {100*m['displacement_all_times']['relative_l2']:.5f} | {100*m['strain']['relative_l2']:.5f} | {100*m['legacy_strain_rate']['relative_l2']:.5f} | {run['elapsed_seconds']:.1f} |")
        lines += ['', 'The coarse-grid crossing-well strain error is retained; refinement uses exactly the same geometry, receivers and original reference. A 3-frame pilot is not a complete notebook reproduction.', '',
                  '[Reference provenance](notebook_existing_tensile_shear/reference_provenance.json). Frozen source and hashes reside alongside it.']
        diagnostics=notebook/'diagnostics.json'
        if diagnostics.exists():
            for name,m in json.loads(diagnostics.read_text()).items():
                lines += ['', f"Notebook final-state check {name}: u L2 = {100*m['displacement']['relative_l2']:.5f}%; strain L2 = {100*m['strain']['relative_l2']:.5f}%."]
    cache=OUT/'constant_jacobian_check/metrics.json'
    if cache.exists():
        m=json.loads(cache.read_text())
        lines += ['', f"Constant-Jacobian verification: cached versus freshly assembled FE histories have relative L2 difference {m['cached_vs_uncached']['relative_l2']:.6g}; factors [0, 1, -0.7, 0.2] at times [0, 1, 2, 4] s. Both methods actually solved the three nonzero loads."]
        if 'recycling_vs_uncached' in m:
            lines += ['', f"PETSc Fischer initial-guess recycling versus freshly assembled histories: relative L2 difference {m['recycling_vs_uncached']['relative_l2']:.6g}. The original RHS and all convergence tolerances are unchanged; each requested equilibrium is checked, including steps requiring zero additional Krylov iterations."]
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:8]))
    return results

if __name__=='__main__': execute()
