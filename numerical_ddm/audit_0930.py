"""Use the intact 0930 package, including its original well postprocessor.

Run with the workspace base Python (it includes JIN_pylib notebook dependencies).
"""
import sys,json,hashlib,shutil,subprocess,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'fibeRIS/src')]
import numpy as np
from scripts.tensile_fault.data_transfer.from_pc import DDMpy_0930 as ddm
from fiberis.moose.numerical_ddm import axial_response,rotation,from_ddmpy,NumericalConfig
OUT=ROOT/'output/numerical_ddm/reference_0930'
OUT.mkdir(exist_ok=True)
source=Path(ddm.__file__).parent
frozen=OUT/'original_DDMpy_0930'
frozen.mkdir(exist_ok=True)
manifest={}
for p in sorted(source.glob('*.py')):
    target=frozen/p.name
    if target.exists() and target.read_bytes()!=p.read_bytes():
        raise RuntimeError('Reference contents changed: '+p.name)
    shutil.copy2(p,target)
    manifest[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copy2(source/'README.md',frozen/'README.md')
metadata={'commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
          'sha256':manifest,'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
          'package_path':str(source),'reference_modifications':'none'}
(OUT/'provenance.json').write_text(json.dumps(metadata,indent=2))
p=np.arange(5.)[:,None]*np.array([2.,3.,4.])
w=ddm.Well.Well().set_well_path(*p.T)
w.u1,w.u2,w.u3=(p*1e-4).T
w.calculate_strain()
dyn=ddm.DynamicFracture.RectangularFracture()
dyn.define_asymmetric_grow([0,1],30,30,50,50,width=.001)
checks={'well_uniform_dilation_expected':1e-4,'well_actual':w.strain.tolist(),
        'asymmetric_width_requested':.001,'asymmetric_width_actual':-dyn.fractures[0].elements[0].width,
        'vertical_rotation_difference_from_exact':float(abs(ddm.Fracture._rotate_matrix(np.pi/2,np.pi/2)-rotation(90,90)).max()),
        'StrainRateUtil_present':(source/'StrainRateUtil.py').exists()}
np.testing.assert_allclose(w.strain,1e-4)
assert checks['asymmetric_width_actual']==.001
# Full original package workflow, without private imports or AST edits.
e=ddm.Element.Element(100,60,.001)
e.set_global_coors(0,0,0)
w=ddm.Well.Well().set_well_path(np.linspace(-100,100,101),17.,40.)
f=ddm.Fracture.FractureZ().set_fracture_elements(e).set_monitor_wells(w)
f.calculate()
w=f.monitor_wells[0]
u=np.column_stack([w.u1,w.u2,w.u3])
np.savez(OUT/'baseline_original_package.npz',points=np.column_stack([w.x,w.y,w.z]),u=u,strain=w.strain,md=w.mds)
checks['original_vs_shared_axial_operator_max_abs']=float(abs(w.strain-axial_response(np.column_stack([w.x,w.y,w.z]),u)[2]).max())
(OUT/'audit.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))

# Import adapter preserves native nonuniform MD, sample-mean gauge and feet->SI.
s=np.linspace(-100,100,101)
w=ddm.Well.Well().set_well_path(s,17+.15*s,55+.1*s)
w.gauge_length=10/.3048
w.mds+=10000
f=ddm.Fracture.FractureZ().set_fracture_elements(e).set_monitor_wells(w)
f.calculate()
a=from_ddmpy(f,NumericalConfig('/unused',str(OUT/'unused')),length_scale=.3048)
ow=f.monitor_wells[0];nw=a.monitor_wells[0]
nw.u1,nw.u2,nw.u3=np.array([ow.u1,ow.u2,ow.u3])*.3048
nw.calculate_strain()
checks['adapter_strain_max_abs_difference']=float(abs(nw.strain-ow.strain).max())
np.testing.assert_allclose(nw.strain,ow.strain,atol=1e-16)
(OUT/'audit.json').write_text(json.dumps(checks,indent=2))
