"""Record exact local environment and source/binary identities without network."""
import sys,json,hashlib,subprocess,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/numerical_ddm'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def command(args):
    r=subprocess.run(args,text=True,capture_output=True)
    return dict(command=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)

record=dict(python=sys.version,platform=platform.platform(),workspace=str(ROOT),
    repos={name:command(['git','-C',str(ROOT/path),'rev-parse','HEAD']) for name,path in {
        'fiberis':'fibeRIS','moose':'moose_env/moose',
        'ddmpy0930':'scripts/tensile_fault/data_transfer/from_pc/DDMpy_0930'}.items()},
    binaries={},sources={})
for path in ['moose_env/moose/modules/combined/combined-opt',
             'moose_env/moose/modules/combined/lib/libcombined-opt.so',
             'moose_env/moose/modules/solid_mechanics/lib/libsolid_mechanics-opt.so',
             'moose_env/moose/framework/libmoose-opt.so']:
    p=ROOT/path
    if p.exists():record['binaries'][path]=dict(sha256=sha(p),bytes=p.stat().st_size)
for p in [*list((ROOT/'scripts/numerical_ddm').glob('*.py')),
          ROOT/'fibeRIS/src/fiberis/moose/numerical_ddm.py',
          ROOT/'scripts/numerical_ddm/DDMPY_0930_vs_MOOSE.ipynb',
          ROOT/'moose_env/moose/modules/combined/src/problems/ConstantJacobianProblem.C',
          ROOT/'moose_env/moose/modules/combined/include/problems/ConstantJacobianProblem.h']:
    record['sources'][str(p.relative_to(ROOT))]=sha(p)
record['conda_packages']=command([str(Path.home()/'miniforge/bin/conda'),'list','-n','moose','--json'])
record['reference_base_conda_packages']=command([str(Path.home()/'miniforge/bin/conda'),'list','-n','base','--json'])
record['hardware']=dict(cpu=command(['lscpu']),memory=command(['free','-b']))
(OUT/'environment.json').write_text(json.dumps(record,indent=2))
print('Environment recorded:',OUT/'environment.json')
