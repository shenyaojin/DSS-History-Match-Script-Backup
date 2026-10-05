"""Exercise the new observation/adjoint mapping without replacing MOOSE FD checks."""
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import pandas as pd

import run_inversion as driver


class SurrogateRunner:
    def run(self, input_file, output_dir, **kwargs):
        output=Path(output_dir)
        text=Path(input_file).read_text()
        full=np.fromstring(re.search(r"initial_condition\s*=\s*'([^']*)'",text)[1],sep=';')
        a,b=full[99],full[-1]
        df=pd.read_csv(driver.RUN/'solver_measurement_data.csv')
        t=df.measurement_time.to_numpy()/df.measurement_time.max()
        y=np.tile(np.linspace(-1,1,18),130)
        # Nonlinear response with negative and positive values, and a nonzero
        # first frame: exercises the half-wave mask and explicit t=0 exclusion.
        raw=1e-6*(np.exp(a+17)*(.4+t)*(y+.3)+.2*np.exp(b+17)*(1+t*t))
        jac=np.array([1e-6*np.exp(a+17)*(.4+t)*(y+.3),
                      1e-6*.2*np.exp(b+17)*(1+t*t)])
        sim=np.maximum(raw,0)
        use=t>0
        r=np.where(use,sim-df.measurement_values,0)
        g=jac@np.where(raw>0,r,0)
        for name,values in [('fake_forward0_out_data_0130.csv',sim),
                            ('fake_forward0_out_signed_data_0130.csv',raw)]:
            frame=df.copy();frame['simulation_values']=values
            frame['misfit_values']=np.where(use,values-df.measurement_values,0)
            frame.to_csv(output/name,index=False)
        pd.DataFrame([{'OptimizationReporter/objective_value':.5*np.dot(r,r)}]).to_csv(output/'evaluate_out.csv',index=False)
        gradients={f'grad_perm_{i+1}':g[0]/2 if i in [99,100] else 0 for i in range(200)}
        gradients['grad_kyy_A']=g[1]
        pd.DataFrame([gradients]).to_csv(output/'evaluate_out_OptimizationReporter_0001.csv',index=False)
        return True,'',''


class CouplingGradientTest(unittest.TestCase):
    def test_full_gradient_and_preserved_observations(self):
        original_run=driver.RUN
        with tempfile.TemporaryDirectory() as tmp:
            try:
                driver.RUN=Path(tmp)
                cfg=json.loads((original_run/'config.json').read_text())
                (driver.RUN/'config.json').write_text(json.dumps(cfg))
                (driver.RUN/'optimize.i').write_text((original_run/'optimize.i').read_text())
                (driver.RUN/'measurement_data.csv').write_bytes((original_run/'measurement_data.csv').read_bytes())
                before=(driver.RUN/'measurement_data.csv').read_bytes()
                model=driver.Inversion();model.runner=SurrogateRunner()
                theta=np.array([-17.1,-16.9,.37])
                value,gradient=model.evaluate(theta)
                numerical=[]
                for direction in np.eye(3):
                    h=1e-5
                    numerical.append((model.evaluate(theta+h*direction)[0]-model.evaluate(theta-h*direction)[0])/(2*h))
                np.testing.assert_allclose(gradient,numerical,rtol=1e-6,atol=1e-10)
                self.assertEqual(before,(driver.RUN/'measurement_data.csv').read_bytes())
                self.assertEqual(model.regularization(theta)[1][2],0)
                folder=model.artifacts[tuple(theta)]
                predicted=pd.read_csv(folder/'strain.csv')
                physical=pd.read_csv(folder/'strain_physical.csv')
                np.testing.assert_allclose(predicted.simulation_values,theta[2]*physical.simulation_values,atol=1e-18)
                self.assertTrue((predicted.loc[predicted.measurement_time==0,'misfit_values']==0).all())
                self.assertTrue(np.isfinite(value))
            finally:
                driver.RUN=original_run


if __name__=='__main__':
    unittest.main()
