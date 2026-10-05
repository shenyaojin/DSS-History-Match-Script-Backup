"""Freeze and audit original DDMPY without modifying any reference source."""
import ast,hashlib,importlib.util,json,shutil,sys,types,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'fibeRIS/src'))
from fiberis.moose.numerical_ddm import Well,time_derivative,RectangularFracture,rotation
SOURCE=ROOT/'scripts/tensile_fault/data_transfer/from_pc/DDMpy_log'
OUT=ROOT/'output/numerical_ddm/reference_audit'
OUT.mkdir(parents=True,exist_ok=True)
FROZEN=OUT/'original_DDMpy_log'
FROZEN.mkdir(exist_ok=True)
manifest={}
for p in sorted(SOURCE.glob('*.py')):
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    target=FROZEN/p.name
    if target.exists() and target.read_bytes()!=p.read_bytes():
        raise RuntimeError(f'Reference changed since freeze: {p}')
    shutil.copy2(p,target)
    manifest[p.name]=digest
(OUT/'sha256.json').write_text(json.dumps(manifest,indent=2))
# Private package avoids __init__ importing unavailable notebook widgets.
pkg=types.ModuleType('_frozen_ddm')
pkg.__path__=[str(FROZEN)]
sys.modules[pkg.__name__]=pkg
mods={}
for name in ['Element','Well','Fracture','StrainRateUtil']:
    spec=importlib.util.spec_from_file_location(f'_frozen_ddm.{name}',FROZEN/f'{name}.py')
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    setattr(pkg,name,mod)
    mods[name]=mod
# Only the unrelated notebook container import is excluded. All original
# class/function AST nodes (including all numerical expressions) run verbatim.
tree=ast.parse((FROZEN/'DynamicFracture.py').read_text())
tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='JIN_pylib')]
ns={'__name__':'_frozen_ddm.DynamicFracture','__package__':'_frozen_ddm'}
exec(compile(tree,str(FROZEN/'DynamicFracture.py'),'exec'),ns)

s=np.arange(5.)
p=np.column_stack([2*s,3*s,4*s])
w=mods['Well'].Well().set_well_path(*p.T)
w.u1,w.u2,w.u3=(1e-4*p).T
w.calculate_strain()
fixed=Well().set_well_path(*p.T)
fixed.u1,fixed.u2,fixed.u3=(1e-4*p).T
fixed.calculate_strain()
dyn=ns['RectangularFracture']()
dyn.define_asymmetric_grow([0,1],up=30,down=30,left=50,right=50,width=.001)
fixed_dyn=RectangularFracture().define_asymmetric_grow([0,1],30,30,50,50,width=.001)
t=np.array([0.,1.,3.,6.])
original_rate=mods['StrainRateUtil'].centered_time_diff(t[None,:]**2,t)
fixed_rate=time_derivative(t[None,:]**2,t)
Rold=mods['Fracture']._rotate_matrix(np.pi/2,np.pi/2)
Rnew=rotation(90,90)
result={
 'well_normalization':dict(expected=1e-4,original=w.strain.tolist(),corrected=fixed.strain.tolist()),
 'asymmetric_width':dict(requested=.001,original=-dyn.fractures[0].elements[0].width,corrected=-fixed_dyn.fractures[0].elements[0].width),
 'nonuniform_time':dict(times=t.tolist(),data=(t*t).tolist(),expected=(2*t).tolist(),original=original_rate.tolist(),corrected=fixed_rate.tolist()),
 'vertical_dip':dict(perturbation_rad=.001,max_matrix_change=float(abs(Rold-Rnew).max()),old=Rold.tolist(),exact=Rnew.tolist()),
 'loading_note':'Private import skips package __init__; DynamicFracture AST excludes only notebook Data2D_XT import; numerical definitions unchanged.'}
assert np.allclose(fixed.strain,1e-4)
assert not np.allclose(w.strain,1e-4)
assert -dyn.fractures[0].elements[0].width==30
assert np.allclose(fixed_rate,2*t)
assert not np.allclose(original_rate,2*t)
(OUT/'minimal_reproductions.json').write_text(json.dumps(result,indent=2))
# Near-centre jump check independently confirms constructor convention for all modes.
jumps={}
for k in range(3):
    j=np.eye(3)[k]*.001
    e=mods['Element'].Element(100,60,j[2],j[0],j[1])
    e.set_coors(np.zeros(2),np.zeros(2),np.array([-1e-5,1e-5]))
    u=np.array([e.u1,e.u2,e.u3]).T
    jumps[str(k)]=(u[1]-u[0]).tolist()
(OUT/'jump_sign.json').write_text(json.dumps(jumps,indent=2))
print(json.dumps(result,indent=2))
