"""Draw Page geometries from saved inputs, without running any FE solve."""
from pathlib import Path
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'fibeRIS/src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from fiberis.moose.numerical_ddm import rotation, NumericalConfig, _axis_nodes

DATA = ROOT / 'output/numerical_ddm'
OUT = DATA / 'page_geometries_20261001'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})
BLUE, ORANGE, GREEN, RED = '#387db3', '#d57822', '#24896c', '#b34751'
FIGURES = []
MANIFEST = []


def read(name):
    return json.loads((DATA / name / 'model.json').read_text())


def vertices(e):
    c = np.array([e['abs_x1'], e['abs_x2'], e['abs_x3']])
    l, h = e['length'] / 2, e['height'] / 2
    return np.array([[-l, -h, 0], [l, -h, 0], [l, h, 0], [-l, h, 0]]) + c


def setup2(ax, title, a, b):
    ax.set(title=title, xlabel=f'$x_{a+1}$ (m)', ylabel=f'$x_{b+1}$ (m)')
    ax.grid(alpha=.16)
    ax.set_aspect('equal', adjustable='box')


def save(fig, name, title, source):
    fig.savefig(OUT / f'{name}.png', dpi=160)
    fig.savefig(OUT / f'{name}.pdf')
    FIGURES.append((name, fig))
    MANIFEST.append(dict(name=name, title=title, source=source))
    plt.close(fig)


def geometry(name, title, model, points=None, R=None, origin=None, note='', source='', gauge=None):
    p = np.array(model['points'] if points is None else points)
    R = np.eye(3) if R is None else R
    origin = np.zeros(3) if origin is None else np.array(origin)
    polys = [vertices(e) for e in model['elements']]
    pg = p @ R + origin
    glob = [v @ R + origin for v in polys]
    fig = plt.figure(figsize=(14.4, 6.0))
    gs = fig.add_gridspec(1, 3, left=.035, right=.97, bottom=.23, top=.77, wspace=.31,
                         width_ratios=[1.08, 1, 1])
    ax = fig.add_subplot(gs[0], projection='3d')
    ax.add_collection3d(Poly3DCollection(glob, facecolors=BLUE, edgecolors=BLUE, alpha=.28))
    ax.plot(*pg.T, color=ORANGE, lw=2.2)
    ax.scatter(*pg[::max(1, len(p)//20)].T, color=ORANGE, s=8)
    ax.scatter(*pg[0], color=GREEN, s=28)
    ax.scatter(*pg[-1], color=RED, s=28)
    bound = np.vstack([pg, *glob])
    lo, hi = bound.min(0), bound.max(0)
    spans = np.maximum(hi-lo, max(hi-lo)*.18)
    center = (lo+hi)/2
    for k, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        setter(center[k]-spans[k]*.62, center[k]+spans[k]*.62)
    ax.set_box_aspect(spans)
    ax.view_init(24, -58)
    ax.set(xlabel='Global X (m)', ylabel='Global Y (m)', zlabel='Global Z (m)',
           title='3D geometry / global coordinates')
    # Local basis directions, scaled in metres only for visibility.
    length = max(np.ptp(bound, axis=0))*.16
    for k, color in enumerate([GREEN, RED, '#6b59a5']):
        vector = R[k]*length
        ax.quiver(*origin, *vector, color=color, arrow_length_ratio=.2, linewidth=1.5)
        ax.text(*(origin+vector*1.15), f'$x_{k+1}$', color=color, fontsize=9)
    ax2, ax3 = fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for v in polys:
        ax2.fill(v[:, 0], v[:, 1], color=BLUE, alpha=.22)
        closed = np.vstack([v, v[0]])
        ax2.plot(closed[:, 0], closed[:, 1], color=BLUE, lw=1.2)
        ax3.plot([v[:, 0].min(), v[:, 0].max()], [v[0, 2]]*2, color=BLUE, lw=3)
    for a, dims in [(ax2, (0, 1)), (ax3, (0, 2))]:
        i, j = dims
        a.plot(p[:, i], p[:, j], color=ORANGE, lw=2)
        a.scatter(p[::max(1, len(p)//25), i], p[::max(1, len(p)//25), j], color=ORANGE, s=9)
        a.plot(p[0, i], p[0, j], 'o', color=GREEN, ms=5)
        a.plot(p[-1, i], p[-1, j], 'o', color=RED, ms=5)
        a.plot(0, 0, '+', color=BLUE, ms=9)
        a.margins(.14)
    setup2(ax2, 'Fracture-plane projection', 0, 1)
    setup2(ax3, 'Normal section / projection', 0, 2)
    cross = np.min(p[:, 2]) < 0 < np.max(p[:, 2])
    if cross:
        j = np.flatnonzero(p[:-1, 2]*p[1:, 2] <= 0)[0]
        q = p[j] + (p[j+1]-p[j])*(-p[j, 2])/(p[j+1, 2]-p[j, 2])
        ax3.plot(q[0], 0, 'o', mfc='white', mec=RED, ms=8)
        ax3.annotate('Plane crossing', (q[0], 0), (q[0]+55, 60), fontsize=9,
                     arrowprops={'arrowstyle':'->', 'color':RED}, color=RED)
    elif np.ptp(p[:, 2]) < 1e-8:
        x = max(v[:, 0].max() for v in polys)+13
        ax3.annotate('', (x, p[0, 2]), (x, 0), arrowprops={'arrowstyle':'<->', 'color':'#555'})
        ax3.text(x+4, p[0, 2]/2, f'{p[0,2]:g} m', va='center', fontsize=9)
    if gauge:
        md = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        mid = md[len(md)//2]
        segment = np.column_stack([np.interp(np.linspace(mid-gauge/2, mid+gauge/2, 31), md, p[:,i]) for i in range(3)])
        for a, j in [(ax2, 1), (ax3, 2)]:
            a.plot(segment[:,0], segment[:,j], color=GREEN, lw=5, solid_capstyle='round')
    # Jump arrows show direction; their plotted lengths are deliberately exaggerated.
    e = model['elements'][len(model['elements'])//2]
    jump = -np.array([e['S1'], e['S2'], e['width']])*1000
    L, H = e['length'], e['height']
    for k, value in enumerate(jump):
        if abs(value)<1e-12:
            continue
        target = ax3 if k==2 else ax2
        start = np.array([0., 0.])
        end = start.copy()
        end[1 if k else 0] += np.sign(value)*min(max(L,H)*.26, 40)
        target.annotate('', end, start, arrowprops={'arrowstyle':'-|>', 'color':'#7043a0', 'lw':2})
    label = '   '.join(f'{k} = {v:g} mm' for k,v in zip(['S1','S2','W'], jump))
    cfg = model['config']
    fig.suptitle(title, x=.045, ha='left', y=.96, fontsize=16, fontweight='bold')
    size = f"{model['elements'][0]['length']:g} x {model['elements'][0]['height']:g} m" if len(polys)==1 else f'{len(polys)} actual rectangular strips'
    fig.text(.045, .875, f'{size}   |   {len(p)} receivers   |   {label}', fontsize=11)
    fig.text(.045, .155, note, fontsize=10, va='top', linespacing=1.6)
    fig.text(.045, .045, f"FE domain: local [-{cfg['half_extent']:g}, {cfg['half_extent']:g}]^3 m (outside this zoom). "
             f"Band t = {cfg['band_thickness']:g} m; h = {cfg['mesh_size']:g} m. Jump arrows are not to scale.",
             fontsize=9, color='#555')
    handles = [Line2D([],[],color=BLUE,lw=5,alpha=.5,label='Fracture'),
               Line2D([],[],color=ORANGE,marker='.',label='Well / receivers'),
               Line2D([],[],color=GREEN,marker='o',ls='',label='Start'),
               Line2D([],[],color=RED,marker='o',ls='',label='End'),
               Line2D([],[],color='#7043a0',lw=2,label='Jump direction')]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.975,.858), ncol=5, frameon=False, fontsize=9)
    save(fig, name, title, source)


base = read('doc_check_20261001_h7p5')
for name, title in [('01_migration','Migration example'), ('02_native','Native fibeRIS example'),
                    ('03_baseline','Single-rectangle opening benchmark')]:
    geometry(name, title, base, note='Fracture: z = 0, centered at (0, 0, 0). Well: (x, 17, 40) m, x = -100 ... 100 m.\nThe well is parallel to the fracture plane. Overlap in the middle projection does not mean crossing.', source='doc_check_20261001_h7p5/model.json')
for name, folder, title in [('04_s1','s1_h10_L800_t2','Pure S1 shear'),
                             ('05_s2','s2_h10_L800_t2','Pure S2 shear'),
                             ('06_mixed','mixed_h7p5_L800_t2','Mixed opening and shear')]:
    geometry(name, title, read(folder), note='Same fracture and well as the opening benchmark. Local axes equal global axes.\nS1 follows x1; S2 follows x2; W follows x3. Negative S2 points toward -x2.', source=folder+'/model.json')

# Additional consistency cases mentioned in the Page.
for name, folder, title in [('07_zero','zero_h10_L800_t2','Zero-load consistency case'),
                             ('08_double','double_h10_L800_t2','Doubled-opening consistency case')]:
    geometry(name, title, read(folder), note='Physical geometry is unchanged; only the displacement jump changes.', source=folder+'/model.json')

ext = DATA/'extensions'
for name, folder, index, title, R, origin, note, gauge in [
    ('12_curved','multiple_wells',2,'Curved well with a 10 m gauge',None,None,
     'Well: (x, 17 + 10 sin(x/60), 60 + 0.15x) m. The green segment spans 10 m along the well.\nThe normal distance changes from 45 to 75 m; the well never crosses the fracture.',10),
    ('13_rotated','rotated_translated',1,'Rotated fracture and inclined well',rotation(37,63),[210,-130,80],
     'Strike = 37 deg; dip = 63 deg; center = (210, -130, 80) m.\nLocal well: (x, 17 + 0.15x, 55 + 0.1x) m. Both fracture and well rotate and translate together.',None),
    ('14_ellipse','discrete_ellipse_5',1,'Five-strip discrete ellipse with inclined well',None,None,
     'Strip width = 20 m; heights = 36, 54.99, 60, 54.99, 36 m. Local well: (x,17+0.15x,55+0.1x).\nJump values above are for the center strip; all three jumps taper with strip height.',None)]:
    model = read('extensions/'+folder)
    pts = np.load(ext/folder/f'well_{index}.npz')['points']
    if R is not None:
        pts = (pts-origin)@R.T
    geometry(name,title,model,pts,R,origin,note,'extensions/'+folder+'/model.json',gauge)

histdir='notebook_existing_tensile_shear/full_h20_t0.2_L2400_refined_recycled'
history=read(histdir)
R=rotation(0,90,legacy_vertical=True)
origin=np.array([150,130,-30])
original_points=np.load(DATA/'notebook_existing_tensile_shear/original_0930.npz')['points']
np.testing.assert_allclose(np.array(history['points'])@R+origin,original_points,atol=1e-10)
for name,title in [('15_history','Original notebook: 30 fixed-geometry frames'),
                   ('17_history_api','DynamicFracture / sample_history with the notebook model')]:
    geometry(name,title,history,R=R,origin=origin,
      note='Well crosses at global (150, 0, -30) m: local x1 = -130 m, x2 = 0, x3 = 0.\nAll 30 frames keep this geometry. S1: 0 to 1 mm; W = 0.1 S1. The legacy 0.001 rad dip offset is retained.',
      source=histdir+'/model.json',gauge=10)

geometry('18_history_fixed','Fixed geometry with changing opening',read('extensions/history_fixed/snapshot_0003'),
         note='Times = [0, 1, 3, 6] s. W = 1 mm * (time / 6 s)^2: [0, 0.02778, 0.25, 1] mm.\nThe rectangle and the well stay fixed; only W changes. The diagram shows the final state.',
         source='extensions/history_fixed/snapshot_0003/model.json')


def mesh_panels(name,title,cases,view,notes):
    fig,axs=plt.subplots(1,len(cases),figsize=(14.4,6.2))
    fig.subplots_adjust(left=.06,right=.97,bottom=.24,top=.78,wspace=.26)
    for ax,(folder,label) in zip(axs,cases):
        m=read(folder);cfg=m['config'];e=m['elements'][0];p=np.array(m['points'])
        if view=='domain':
            A=cfg['half_extent'];ax.add_patch(Rectangle((-A,-A),2*A,2*A,fill=False,ec=BLUE,lw=2))
            ax.plot([-50,50],[0,0],color=BLUE,lw=3);ax.plot(p[:,0],p[:,2],color=ORANGE,lw=2)
            ax.set(xlim=(-1750,1750),ylim=(-1750,1750))
        elif view=='band':
            t=cfg['band_thickness'];ax.add_patch(Rectangle((-50,-t/2),100,t,fc=BLUE,alpha=.25))
            for z in np.linspace(-t/2,t/2,cfg['band_cells']+1):ax.plot([-50,50],[z,z],color=BLUE,lw=1)
            ax.axhline(0,color='#555',lw=1,ls='--');ax.set(xlim=(-60,60),ylim=(-3,3))
            ax.set_xticks([-50,0,50])
            ax.set_aspect('auto');ax.text(.04,.06,'Vertical scale enlarged\nWell at x3 = 40 m (outside inset)',transform=ax.transAxes,fontsize=9)
        else:
            h=cfg['mesh_size'];xx=np.unique(np.r_[np.linspace(-120,-50,int(np.ceil(70/h))+1),np.linspace(-50,50,int(np.ceil(100/h))+1),np.linspace(50,120,int(np.ceil(70/h))+1)])
            yy=np.unique(np.r_[np.linspace(-120,-30,int(np.ceil(90/h))+1),np.linspace(-30,30,int(np.ceil(60/h))+1),np.linspace(30,120,int(np.ceil(90/h))+1)])
            ax.vlines(xx,-65,65,colors='#ddd',lw=.7);ax.hlines(yy,-120,120,colors='#ddd',lw=.7)
            ax.add_patch(Rectangle((-50,-30),100,60,fc=BLUE,ec=BLUE,alpha=.28))
            ax.plot(p[:,0],p[:,1],color=ORANGE,lw=2);ax.set(xlim=(-115,115),ylim=(-65,65))
        ax.set_title(label);ax.set_xlabel('$x_1$ (m)');ax.set_ylabel('$x_2$ (m)' if view=='mesh' else '$x_3$ (m)')
        if view!='band':ax.set_aspect('equal',adjustable='box')
    fig.suptitle(title,x=.045,ha='left',fontsize=16,fontweight='bold',y=.95)
    fig.text(.045,.15,notes,fontsize=11,va='top',linespacing=1.6)
    save(fig,name,title,[x[0]+'/model.json' for x in cases])

mesh_panels('09_mesh','Core mesh scan: the same physical geometry',
    [(f'opening_h{h}_L800_t2',f'h = {h.replace("p", ".")} m') for h in ['20','10','7p5']], 'mesh',
    'Plan view of the core: rectangle = 100 x 60 m; orange = well projection at x2 = 17 m (actual x3 = 40 m).\nOnly the mesh changes. W = 1 mm; band t = 2 m; domain half-extent = 800 m.')
mesh_panels('10_domain','Outer-domain scan: zero displacement on all six faces',
    [(f'opening_h10_L{a}_t2',f'Half-extent A = {a} m') for a in [400,800,1600]], 'domain',
    'All panels use the same scale. Blue outline = section of the 3D cube [-A, A]^3.\nThe fracture and well keep the same positions and sizes; h = 10 m; band t = 2 m; W = 1 mm.')
mesh_panels('11_band','Band-thickness scan: numerical thickness versus opening',
    [(f'opening_h10_L800_t{t}',f'Band t = {t} m') for t in [1,2,4]],'band',
    'Local normal sections show the actual band limits x3 = +/- t/2, with two cells through the thickness.\nPhysical opening W stays 0.001 m in every panel. Band thickness is a numerical parameter, not W.')

# Actual local refinement axis nodes, rather than an invented rectangular patch mesh.
fig,axs=plt.subplots(1,2,figsize=(13,6.5));fig.subplots_adjust(left=.07,right=.97,bottom=.23,top=.78,wspace=.25)
for ax,refine in zip(axs,[False,True]):
    cfg=NumericalConfig(executable='',output_directory='',mesh_size=20,half_extent=2400,core_extent=180,band_thickness=.2)
    if refine:cfg.local_refinements=((0,-160,-100,5),(1,-30,30,10),(2,-40,40,2.5))
    xx=_axis_nodes([-150,150],cfg,0);zz=_axis_nodes([-.1,0,.1],cfg,2)
    ax.vlines(xx,-80,80,color='#cbd0d5',lw=.65);ax.hlines(zz,-180,-70,color='#cbd0d5',lw=.65)
    ax.axvspan(-160,-100,color=GREEN,alpha=.08);ax.axhspan(-40,40,color=RED,alpha=.07)
    ax.plot([-150,-70],[0,0],color=BLUE,lw=3)
    p=np.array(history['points']);ax.plot(p[:,0],p[:,2],color=ORANGE,lw=2)
    ax.plot(-130,0,'o',mec=RED,mfc='white',ms=7)
    ax.set(xlim=(-180,-70),ylim=(-80,80),xlabel='$x_1$ (m)',ylabel='$x_3$ (m)',title='Local refinement' if refine else 'Base core mesh')
    ax.set_aspect('equal')
fig.suptitle('Crossing well: same geometry, different mesh',x=.06,ha='left',fontsize=16,fontweight='bold',y=.95)
fig.text(.06,.86,'Actual axis nodes in a zoom near the well; layer t = 0.2 m, domain half-extent = 2400 m.',fontsize=11)
fig.text(.06,.15,'Well: local x1 = -130 m, x2 approximately 0; fracture tip at x1 = -150 m.\nRefinement intervals: x1 [-160,-100] / 5 m; x2 [-30,30] / 10 m; x3 [-40,40] / 2.5 m.\nEach interval refines an entire coordinate slab; colored overlap is not an isolated 3D refinement box.',fontsize=10,linespacing=1.6,va='top')
save(fig,'16_refinement','Crossing well and actual mesh',histdir+'/model.json')

fig,axs=plt.subplots(1,3,figsize=(14.4,6));fig.subplots_adjust(left=.06,right=.97,bottom=.23,top=.76,wspace=.23)
for i,(ax,t) in enumerate(zip(axs,[0,2,5])):
    m=read(f'extensions/history_geometry/snapshot_{i:04d}');e=m['elements'][0];p=np.array(m['points'])
    v=vertices(e);ax.fill(v[:,0],v[:,1],fc=BLUE,ec=BLUE,alpha=.3)
    ax.plot(p[:,0],p[:,1],color=ORANGE,lw=2);ax.scatter(p[::5,0],p[::5,1],s=8,color=ORANGE)
    setup2(ax,f"Time = {t} s\n{e['length']:g} x {e['height']:g} m; W = {-1000*e['width']:g} mm",0,1)
    ax.set(xlim=(-110,110),ylim=(-55,55))
fig.suptitle('Changing geometry: three actual snapshots',x=.045,ha='left',y=.95,fontsize=16,fontweight='bold')
fig.text(.045,.87,'Same plot scale, fixed center, fixed orientation, fixed well.',fontsize=11)
fig.text(.045,.15,'Plan views: the orange well is at x2 = 17 m and x3 = 40 m above the fracture plane.\nLength, height, and opening change together. These are separate prescribed snapshots, not a solved propagation law.',fontsize=11,linespacing=1.6,va='top')
save(fig,'19_changing','Changing geometry snapshots','extensions/history_geometry/snapshot_000*/model.json')

# One independent geometry sheet for every static verification case, including
# variants that share their physical geometry but differ in numerical settings.
catalog=[]
for file in sorted(DATA.glob('*/model.json')):
    folder=file.parent
    if not (folder/'metrics.json').exists() or not (folder/'run.json').exists():continue
    if not json.loads((folder/'run.json').read_text()).get('success'):continue
    model=json.loads(file.read_text())
    if len(model['elements'])!=1 or model['elements'][0]['length']!=100:continue
    c=model['config'];key='case_'+folder.name
    geometry(key,folder.name.replace('_',' '),model,
        note=f"Same physical rectangle and receiver positions. Band cells = {c['band_cells']}; "
             f"element order = {c['order']}; quadrature = {c['quadrature']}.\n"
             'Coordinates and jump amplitudes are read from this case model.json.',source=str(file.relative_to(DATA)))
    catalog.append(key)

with PdfPages(OUT/'all_geometries.pdf') as pdf:
    for name,fig in FIGURES:pdf.savefig(fig)
for _,fig in FIGURES:plt.close(fig)
(OUT/'manifest.json').write_text(json.dumps(MANIFEST,indent=2))
with zipfile.ZipFile(OUT/'geometry_figures.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(OUT.iterdir()):
        if file.suffix in ('.png','.pdf','.json'):archive.write(file,file.name)
    archive.write(Path(__file__),Path(__file__).name)
print(json.dumps(dict(output=str(OUT),main_figures=len(FIGURES)-len(catalog),static_cases=len(catalog),total=len(FIGURES)),indent=2))
