from pathlib import Path
import sys,json,re
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
sys.path.insert(0,'scripts/DSS_history_match/real_rfsdss_fixed_zones_14373_14379_400000s_bg21_coupling/background_sensitivity')
import run_experiment as e
results=[]
S=.01/2.2e9
for alpha in [-21,-19,-18]:
 t,x,y,p=e.pressure(alpha)
 src=np.argmin(abs(x-45));b=p[:,0,src]
 D=10.**alpha/(.001*S);h=np.diff(x);assert np.allclose(h,.5)
 weights=np.r_[h[0]/2,(h[:-1]+h[1:])/2,h[-1]/2]
 edge=D/h
 diag=np.r_[edge[0],edge[:-1]+edge[1:],edge[-1]]
 K=diags([-edge,diag,-edge],[-1,0,1],format='csr')
 q=np.zeros((len(t),len(x)))
 for n,dt in enumerate(np.diff(t),1):
  A=(diags(weights)+dt*K).tolil();rhs=weights*q[n-1]
  A[src,:]=0;A[src,src]=1;rhs[src]=b[n]
  q[n]=spsolve(A.tocsr(),rhs)
 actual=p[:,0,:]
 row=dict(background_alpha=alpha,diffusivity_m2_s=D,
  relative_rms_to_source_peak=float(np.sqrt(np.mean((q-actual)**2))/max(b)),
  max_absolute_difference_Pa=float(abs(q-actual).max()),
  independent_final_left_corner_Pa=float(q[-1,0]),moose_final_left_corner_Pa=float(actual[-1,0]),
  independent_final_right_corner_Pa=float(q[-1,-1]),moose_final_right_corner_Pa=float(actual[-1,-1]))
 # Parse every xx reporter and its block-specific mapping; validate element geometry
 text=(e.case_dir(alpha)/'forward.i').read_text()
 block=re.search(r'^  \[params\].*?^  \[\]',text,re.M|re.S)[0]
 names=re.search(r"real_vector_names = '([^']*)'",block)[1].split()
 values=[float(z) for z in re.search(r"real_vector_values = '([^']*)'",block)[1].split(';')]
 assert len(values)==len(names)==201
 for i in range(200):
  expected=-14.938868403094903 if i in [99,100] else alpha
  assert abs(values[i]-expected)<1e-12
  assert names[i]==f'perm_{i+1}'
  material=re.search(r'^  \[permeability_layer_'+str(i+1)+r'\].*?^  \[\]',text,re.M|re.S)[0]
  assert f"perm_xx = 'perm_xx_layer_{i+1}'" in material
  assert f"block = 'layer_{i+1}'" in material
 assert values[-1]==-21
 row['all_200_xx_material_mappings_verified']=True
 results.append(row)
 print(json.dumps(row),flush=True)
record=dict(method='Independent 1D linear diffusion: nodal finite volume, implicit Euler, same x grid/times, applied p at x45, no-flow endpoints. Compare MOOSE y=0 far from horizontal band.',storage_Pa_inv=S,results=results)
(e.HERE/'pressure_code_audit/independent_diffusion.json').write_text(json.dumps(record,indent=2)+'\n')
