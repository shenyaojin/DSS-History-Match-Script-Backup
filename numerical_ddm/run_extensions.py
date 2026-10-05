"""Additional actual FE runs, only after the rectangular baseline has run."""
import json,sys
from pathlib import Path
import numpy as np
from run_validation import ROOT,reference,metrics
from fiberis.moose.numerical_ddm import (NumericalConfig,Element,Well,FractureZ,
    FractureGlobal,RectangularFracture,get_ellipse_fracture,axial_response,time_derivative)
import matplotlib.pyplot as plt


def execute(np_ranks=8):
    out=ROOT/'output/numerical_ddm/extensions'
    out.mkdir(exist_ok=True)
    result={}
    def config(name):
        return NumericalConfig(str(ROOT/'moose_env/moose/modules/combined/combined-opt'),
            str(out/name),mesh_size=10,processors=np_ranks,
            mpiexec=str(Path.home()/'miniforge/envs/moose/bin/mpiexec'))
    x=np.linspace(-100,100,101)
    wells=[Well().set_well_path(x,17.,40.),
           Well().set_well_path(x,17+.15*x,55+.1*x),
           Well().set_well_path(x,17+10*np.sin(x/60),60+.15*x)]
    wells[2].gauge_length=10.
    def compare(name,f,selected_wells):
        f.set_monitor_wells(selected_wells).configure(config(name)).calculate()
        R,origin=f.frame()
        reports=[]
        for i,w in enumerate(f.monitor_wells):
            p=w.points
            ref=reference(f.elements,(p-origin)@R.T)@R
            u=np.column_stack([w.u1,w.u2,w.u3])
            _,_,eps=axial_response(p,ref,w.gauge_length)
            reports.append(dict(displacement=metrics(u,ref),strain=metrics(w.strain,eps)))
            np.savez(out/name/f'well_{i}.npz',points=p,u=u,reference=ref,
                     strain=w.strain,reference_strain=eps,md=w.mds)
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for k in range(3):
                axes[0].plot(w.mds,ref[:,k]*1e6,label=f'DDM u{k+1}')
                axes[0].plot(w.mds,u[:,k]*1e6,'--')
            axes[0].set(xlabel='MD (m)',ylabel='Displacement (micrometre)')
            axes[0].legend()
            axes[1].plot(w.mds,eps*1e6,label='DDM + shared operator')
            axes[1].plot(w.mds,w.strain*1e6,'--',label='MOOSE + shared operator')
            axes[1].set(xlabel='MD (m)',ylabel='Axial microstrain')
            axes[1].legend();fig.tight_layout()
            fig.savefig(out/name/f'well_{i}.png',dpi=150);plt.close(fig)
        result[name]=reports
        (out/'metrics.json').write_text(json.dumps(result,indent=2))
        return f
    compare('multiple_wells',FractureZ().set_fracture_elements(Element(100,60,.001)),wells)
    f=FractureGlobal().set_global_coors(37,63,210,-130,80)
    f.set_fracture_elements(Element(100,60,.001))
    R,origin=f.frame()
    rotated=[Well().set_well_path(*(w.points@R+origin).T) for w in wells[:2]]
    f=compare('rotated_translated',f,rotated)
    local_data=np.load(out/'multiple_wells/well_0.npz')
    result['rotation_covariance']=metrics(np.column_stack([f.monitor_wells[0].u1,
        f.monitor_wells[0].u2,f.monitor_wells[0].u3])@R.T,local_data['u'])
    elems=[Element(50,60,.001).set_global_coors(-25,0,0),
           Element(50,60,.001).set_global_coors(25,0,0)]
    f=compare('two_rectangles',FractureZ().set_fracture_elements(elems),wells[:1])
    result['rectangle_partition_invariance']=metrics(np.column_stack([
        f.monitor_wells[0].u1,f.monitor_wells[0].u2,f.monitor_wells[0].u3]),local_data['u'])
    compare('discrete_ellipse_5',get_ellipse_fracture(100,1.2,2e-5,8e-6,-1.2e-5,5),wells[:2])
    # Genuine FE snapshots, including zero loading. No analytical fields or
    # cached amplitude scaling are used to provide any numerical answer.
    taxis=np.array([0.,1.,3.,6.])
    dynamic=RectangularFracture().define_LHW(taxis,100,60,.001*taxis**2/36)
    dynamic.set_monitor_wells(wells[:1]).configure(config('history_fixed')).calculate()
    numerical=dynamic.gather_strain_data()[0].data
    rate=dynamic.gather_strain_rate_data()[0].data
    ref_base=reference([Element(100,60,.001)],wells[0].points)
    _,_,eps_base=axial_response(wells[0].points,ref_base)
    ref_eps=eps_base[:,None]*taxis**2/36
    exact_rate=eps_base[:,None]*(2*taxis)/36
    result['history_fixed']=dict(strain=metrics(numerical,ref_eps),rate=metrics(rate,exact_rate),
                                reference_time_operator=metrics(time_derivative(ref_eps,taxis),exact_rate))
    np.savez(out/'history_fixed/comparison.npz',times=taxis,md=wells[0].mds,
             strain=numerical,reference=ref_eps,rate=rate,reference_rate=exact_rate)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i in [25,50,75]:
        axes[0].plot(taxis,ref_eps[i]*1e6,label=f'DDM MD={wells[0].mds[i]:g}')
        axes[0].plot(taxis,numerical[i]*1e6,'--')
        axes[1].plot(taxis,exact_rate[i]*1e6)
        axes[1].plot(taxis,rate[i]*1e6,'--')
    axes[0].set(xlabel='Time (s)',ylabel='Microstrain');axes[0].legend()
    axes[1].set(xlabel='Time (s)',ylabel='Microstrain / s')
    fig.tight_layout();fig.savefig(out/'history_fixed/history.png',dpi=160);plt.close(fig)
    changing=RectangularFracture().define_LHW([0,2,5],[80,100,110],[50,60,70],[.0005,.001,.0012])
    changing.set_monitor_wells(wells[:1]).configure(config('history_geometry')).calculate()
    refs=[]
    for f in changing.fractures:
        _,_,eps=axial_response(wells[0].points,reference(f.elements,wells[0].points))
        refs.append(eps)
    ref=np.column_stack(refs)
    eps=changing.gather_strain_data()[0].data
    rate=changing.gather_strain_rate_data()[0].data
    refrate=time_derivative(ref,changing.taxis)
    result['history_geometry']=dict(strain=metrics(eps,ref),rate=metrics(rate,refrate),
           note='Rates use the same discrete time operator; continuous geometry rate is not asserted.')
    np.savez(out/'history_geometry/comparison.npz',times=changing.taxis,strain=eps,
             reference=ref,rate=rate,reference_rate=refrate)
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__': execute(int(sys.argv[1]) if len(sys.argv)>1 else 8)
