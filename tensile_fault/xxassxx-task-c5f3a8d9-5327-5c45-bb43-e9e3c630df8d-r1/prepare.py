from pathlib import Path
import re, json, hashlib, difflib
base = Path(__file__).resolve().parent
original = (base/'sources/original_v11.i').read_text()
common = original.replace('nx = 880', 'nx = 88').replace('ny = 110', 'ny = 80')
# Slight x grading retains the exact original loading node at index 30.
def fraction(b):
    return sum(b**i for i in range(30))/sum(b**i for i in range(88))
lo, hi = 0.99, 1.01
for _ in range(100):
    mid=(lo+hi)/2
    if fraction(mid)>755.626909/2194.56: lo=mid
    else: hi=mid
bias=(lo+hi)/2
common=common.replace('nx = 88', f'nx = 88\n    bias_x = {bias:.17g}')
common = common.replace('end_time = 635400.0', 'end_time = 61200.0')
common = re.sub(r"time_sequence = '[^']+'", "time_sequence = '0.0 18000.0 32400.0 46800.0 61200.0'", common)
# The pressure boundary must coincide with a node of the reduced mesh.
for name in ['baseline', 'shifted']:
    text = common
    if name == 'shifted':
        match = re.search(r'  \[srv_outer_bbox\].*?  \[\]', text, re.S)
        block = match.group()
        block = block.replace('-14.407896000000134', '-12.883896000000134').replace('1894.9416 14.407896000000134', '1894.9416 15.931896000000134')
        text = text[:match.start()] + block + text[match.end():]
    (base/name/'model.i').write_text(text)
    (base/name/'changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='original_v11.i',tofile=name+'/model.i')))
record = {'zone':'srv_outer','direction':'+y (cross-strike model coordinate)','translation_m':1.524,
 'before_m':{'bottom_left':[299.6184,-14.407896,0],'top_right':[1894.9416,14.407896,0],'center':[1097.28,0,0]},
 'after_m':{'bottom_left':[299.6184,-12.883896,0],'top_right':[1894.9416,15.931896,0],'center':[1097.28,1.524,0]},
 'mesh':{'nx':88,'ny_per_panel':80,'elements':28160,'old_elements':387200},'end_time_s':61200,
 'pressure_bc_x_m':{'original':755.626909,'both_tests':755.626909},'bias_x':bias,
 'note':'Common slightly graded x mesh retains exact loading location; monitoring sampler stays at x=756.8184. Pressure curve and all physical parameters unchanged.',
 'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (base/'sources').iterdir()}}
(base/'changes.json').write_text(json.dumps(record,indent=2)+'\n')
