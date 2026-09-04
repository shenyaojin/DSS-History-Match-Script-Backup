"""Read-only adapter for the mixed-layout simulation npz files.

fibeRIS commit aabffe2 (2025-02-11 22:23 MST -- the task package says 2026, but
`git -C fibeRIS log` and every file mtime say 2025) changed `pack_result` from
`data=self.snapshot` to `data=self.snapshot.T`. `snapshot` is (n_t, n_x), so files
written before that commit are TIME-major and files written after are DEPTH-major.
`output/0211_simulation_MULTIstage/` straddles the change: phase1, phase2 and the
five phase3_{ratio} files are time-major, and phase3_test.npz alone is depth-major.

Timestamps do NOT resolve it. Three of the old-layout files there were written
27-96 minutes AFTER the commit (the running interpreter still held the old module),
and six more old-layout files in `output/0218_forward_modeling_old/` were written a
week after. Layout is a property of the bytes, not of the clock. So detection here
uses ONLY len(taxis)/len(daxis) against data.shape, and RAISES whenever that is not
decisive. A wrong guess silently transposes a pressure field into a plausible but
meaningless picture, which is precisely the class of error this round exists to
remove -- no mtime tie-break, no "the longer axis is depth", no fallback.

`Panel.data` is ALWAYS canonical (n_t, n_x), matching
r1_calibration_core.solve_forward's return orientation, so rev2 analysis code and
the verified kernel share one convention with no transposes between them. The
depth-major direction is reached only at the fibeRIS/plotting boundary, through
`to_fiberis_data2d`.

Nothing in this module rewrites `output/0211_simulation_MULTIstage/`; `save_panel`
refuses to write there at runtime.
"""

import dataclasses
import datetime
import fnmatch
import glob
import hashlib
import os

import numpy as np

LAYOUT_TIME_MAJOR = "time_major"      # on-disk data is (n_t, n_x); pre-aabffe2
LAYOUT_DEPTH_MAJOR = "depth_major"    # on-disk data is (n_x, n_t); current writer
VALID_LAYOUTS = (LAYOUT_TIME_MAJOR, LAYOUT_DEPTH_MAJOR)
CANONICAL_LAYOUT = LAYOUT_TIME_MAJOR  # Panel.data is always (n_t, n_x)
REQUIRED_KEYS = ("data", "taxis", "daxis", "start_time")
LAYOUT_KEY = "layout"                 # optional 6th key; no file on disk has it yet
FROZEN_INPUT_DIRS = ("output/0211_simulation_MULTIstage",)

__all__ = ['LayoutError', 'Panel', 'detect_layout', 'load_panel', 'layout_report',
           'load_directory', 'to_fiberis_data2d', 'save_panel',
           'normalize_directory', 'trace_at_md', 'profile_at_time',
           'final_profile', 'LAYOUT_TIME_MAJOR', 'LAYOUT_DEPTH_MAJOR',
           'CANONICAL_LAYOUT', 'LAYOUT_KEY', 'FROZEN_INPUT_DIRS']


class LayoutError(ValueError):
    """Undecidable/contradictory layout, or a structurally unusable file.

    Every message names the file, data.shape, len(taxis), len(daxis) and the rule
    number that fired, so a traceback alone is enough to diagnose the file.
    """


def _sha256(path):
    try:
        from r1_calibration_core import file_sha256  # noqa: F401
        return file_sha256(path)
    except Exception:
        h = hashlib.sha256()
        with open(path, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()


def detect_layout(data_shape, n_t, n_x, declared=None):
    """Decide the on-disk layout from axis lengths alone. Returns (layout, evidence).

    The decision table is evaluated strictly top to bottom, first match wins, and
    there is no default branch:

      R0 declared not in VALID_LAYOUTS                          -> raise
      R1 declared contradicts data_shape                        -> raise
      R2 declared consistent                                    -> return declared
      R3 data not 2-D                                           -> raise
      R4 degenerate axis                                        -> raise
      R5 n_t == n_x (square)                                    -> raise
      R6 data_shape == (n_t, n_x)                               -> time_major
      R7 data_shape == (n_x, n_t)                               -> depth_major
      R8 matches neither                                        -> raise

    R5 is checked BEFORE the shape comparisons because on a square panel both
    comparisons succeed and the rule would silently become a coin flip. R2 is the
    only branch that may return on a square panel: an explicit stamp is the whole
    point of writing one. R1 comes first so a stamp can never override the bytes.
    """
    shape = tuple(int(s) for s in data_shape)
    ev = {
        'method': None, 'data_shape': list(shape),
        'n_t': int(n_t), 'n_t_source': 'len(taxis)',
        'n_x': int(n_x), 'n_x_source': 'len(daxis)',
        'matches_time_major': shape == (int(n_t), int(n_x)),
        'matches_depth_major': shape == (int(n_x), int(n_t)),
        'square': int(n_t) == int(n_x),
        'declared': declared, 'rule': None,
    }

    if declared is not None:
        if declared not in VALID_LAYOUTS:
            ev['rule'] = 0
            raise LayoutError(
                f"rule 0: declared layout {declared!r} is not one of {VALID_LAYOUTS} "
                f"(data_shape={shape}, n_t={n_t}, n_x={n_x})")
        expected = (int(n_t), int(n_x)) if declared == LAYOUT_TIME_MAJOR \
            else (int(n_x), int(n_t))
        if shape != expected:
            ev['rule'] = 1
            raise LayoutError(
                f"rule 1: declared {declared} implies shape {expected}, file has "
                f"{shape} (n_t={n_t}, n_x={n_x}). A declaration that contradicts "
                f"the array is a corrupt file, never a hint to override it.")
        ev['method'], ev['rule'] = 'declared', 2
        return declared, ev

    if len(shape) != 2:
        ev['rule'] = 3
        raise LayoutError(f"rule 3: data must be 2-D, got {len(shape)}-D "
                          f"{shape} (n_t={n_t}, n_x={n_x})")
    if int(n_t) < 1 or int(n_x) < 1:
        ev['rule'] = 4
        raise LayoutError(f"rule 4: degenerate axis: len(taxis)={n_t}, "
                          f"len(daxis)={n_x} (data_shape={shape})")
    if int(n_t) == int(n_x):
        ev['rule'] = 5
        raise LayoutError(
            f"rule 5: square panel: len(taxis) == len(daxis) == {n_t} "
            f"(data_shape={shape}); layout cannot be inferred from axis lengths. "
            f"Re-save with an explicit '{LAYOUT_KEY}' key, or pass "
            f"declared_layout= to say which it is.")
    if shape == (int(n_t), int(n_x)):
        ev['method'], ev['rule'] = 'axis_lengths', 6
        return LAYOUT_TIME_MAJOR, ev
    if shape == (int(n_x), int(n_t)):
        ev['method'], ev['rule'] = 'axis_lengths', 7
        return LAYOUT_DEPTH_MAJOR, ev
    ev['rule'] = 8
    raise LayoutError(
        f"rule 8: data shape {shape} matches neither (n_t, n_x)=({n_t}, {n_x}) "
        f"nor (n_x, n_t)=({n_x}, {n_t}); the file's axes do not describe its data.")


@dataclasses.dataclass(frozen=True)
class Panel:
    """A simulation panel in the canonical orientation.

    `detected_layout` describes the BYTES ON DISK, not `data`. `data` is always
    (n_t, n_x). Code of the form `if p.detected_layout == 'depth_major': arr =
    p.data.T` re-introduces exactly the bug this module exists to remove, and the
    result still has a plausible shape so nothing raises.
    """
    data: np.ndarray
    taxis: np.ndarray
    daxis: np.ndarray
    start_time: datetime.datetime
    detected_layout: str
    evidence: dict
    source_path: str
    sha256: object = None

    def __post_init__(self):
        if self.data.shape != (len(self.taxis), len(self.daxis)):
            raise LayoutError(
                f"{self.source_path}: adapter bug -- canonical data shape "
                f"{self.data.shape} != (len(taxis), len(daxis)) = "
                f"({len(self.taxis)}, {len(self.daxis)})")

    @property
    def data_is_always_time_major(self):
        return True

    @property
    def n_t(self):
        return int(self.data.shape[0])

    @property
    def n_x(self):
        return int(self.data.shape[1])

    def describe(self):
        return (f"{os.path.basename(self.source_path)}  on-disk "
                f"{self.detected_layout}  ->  ({self.n_t}, {self.n_x})  "
                f"rule {self.evidence.get('rule')}")


def _coerce_start_time(st, path):
    if isinstance(st, np.ndarray) and st.size == 1:
        st = st.item()
    if isinstance(st, datetime.datetime):
        # pandas.Timestamp subclasses datetime; pass it through UNCHANGED. It
        # already supports + timedelta and comparison, and downcasting would drop
        # sub-microsecond precision that right_merge and select_time key off.
        return st
    if isinstance(st, np.datetime64):
        return st.astype('datetime64[us]').astype(datetime.datetime)
    if isinstance(st, (str, np.str_)):
        s = str(st)
        for parse in (datetime.datetime.fromisoformat,
                      lambda v: datetime.datetime.strptime(v, '%Y-%m-%d %H:%M:%S.%f'),
                      lambda v: datetime.datetime.strptime(v, '%Y-%m-%d %H:%M:%S')):
            try:
                return parse(s)
            except (ValueError, TypeError):
                continue
    raise LayoutError(f"{path}: unusable start_time of type {type(st).__name__}")


def load_panel(path, *, declared_layout=None, require_monotonic=True,
               compute_sha256=True):
    """Load one npz into a canonical Panel. Read-only; never writes anything."""
    path = os.path.abspath(path)
    try:
        # allow_pickle is mandatory: start_time is a pickled pandas Timestamp in a
        # 0-d object array. Without it numpy raises "Object arrays cannot be
        # loaded when allow_pickle=False", and in a slimmed environment the
        # unpickle raises ModuleNotFoundError from inside pickle, which reads like
        # a corrupt file rather than a missing pandas.
        z = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise LayoutError(
            f"{path}: could not open as an npz ({type(exc).__name__}: {exc}). "
            f"start_time is a pickled pandas Timestamp, so pandas must be "
            f"importable and allow_pickle must be True.") from exc

    try:
        missing = [k for k in REQUIRED_KEYS if k not in z.files]
        if missing:
            raise LayoutError(f"{path}: missing key(s) {missing}; present keys "
                              f"are {list(z.files)}")
        raw = z['data']
        taxis_raw = z['taxis']
        daxis_raw = z['daxis']
        if taxis_raw.ndim != 1 or daxis_raw.ndim != 1:
            raise LayoutError(f"{path}: taxis.ndim={taxis_raw.ndim}, "
                              f"daxis.ndim={daxis_raw.ndim}; both must be 1-D")

        on_disk = None
        if LAYOUT_KEY in z.files:
            v = z[LAYOUT_KEY]
            on_disk = str(v.item() if isinstance(v, np.ndarray) and v.size == 1
                          else v)
        if declared_layout is not None and on_disk is not None \
                and declared_layout != on_disk:
            raise LayoutError(
                f"{path}: caller declared {declared_layout!r} but the file is "
                f"stamped {on_disk!r}; refusing to prefer one silently")
        declared = declared_layout if declared_layout is not None else on_disk

        layout, evidence = detect_layout(raw.shape, len(taxis_raw),
                                         len(daxis_raw), declared)

        taxis = np.ascontiguousarray(taxis_raw, dtype=float)  # phase1's is int64
        daxis = np.ascontiguousarray(daxis_raw, dtype=float)
        arr = np.asarray(raw, dtype=float)
        # ascontiguousarray, not a bare .T: a transposed view is F-ordered and
        # every later row slice data[i, :] strides badly. The copy is ~20 MB and
        # microseconds on these files.
        data = np.ascontiguousarray(arr.T) if layout == LAYOUT_DEPTH_MAJOR \
            else np.ascontiguousarray(arr)

        start_time = _coerce_start_time(z['start_time'], path)

        if require_monotonic:
            for nm, ax in (('taxis', taxis), ('daxis', daxis)):
                d = np.diff(ax)
                if ax.size > 1 and not np.all(d > 0):
                    i = int(np.argmin(d))
                    raise LayoutError(
                        f"{path}: {nm} is not strictly increasing at index {i} "
                        f"({ax[i]!r} -> {ax[i + 1]!r}); data_shape={raw.shape}, "
                        f"n_t={len(taxis_raw)}, n_x={len(daxis_raw)}")

        sha = _sha256(path) if compute_sha256 else None
        return Panel(data=data, taxis=taxis, daxis=daxis, start_time=start_time,
                     detected_layout=layout, evidence=evidence,
                     source_path=path, sha256=sha)
    finally:
        z.close()


def layout_report(paths):
    """Detection only, never raises: one bad file must not abort a directory audit."""
    rows = []
    for p in paths:
        p = os.path.abspath(p)
        row = {'path': p, 'basename': os.path.basename(p)}
        try:
            row['mtime_utc'] = datetime.datetime.utcfromtimestamp(
                os.path.getmtime(p)).isoformat() + 'Z'
            row['sha256'] = _sha256(p)
            z = np.load(p, allow_pickle=True)
            try:
                row['keys'] = list(z.files)
                shape = tuple(int(s) for s in z['data'].shape)
                n_t, n_x = len(z['taxis']), len(z['daxis'])
            finally:
                z.close()
            row.update(data_shape=list(shape), n_t=int(n_t), n_x=int(n_x))
            layout, ev = detect_layout(shape, n_t, n_x)
            row['layout'] = layout
            row['evidence'] = ev
        except Exception as exc:
            row['error'] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def load_directory(dirpath, pattern="*.npz"):
    """{basename: Panel} for every match. Raises on the first bad file."""
    files = sorted(f for f in glob.glob(os.path.join(dirpath, '*'))
                   if fnmatch.fnmatch(os.path.basename(f), pattern))
    return {os.path.basename(f): load_panel(f) for f in files}


def to_fiberis_data2d(panel, cls=None):
    """The only bridge back to fibeRIS: assign depth-major arrays onto a DSS2D.

    Assignment, not a round-trip through load_npz -- load_npz's shape validation
    (core2D.py:239-244) is exactly what rejects the old-layout files, and a temp
    file would break the read-only guarantee.

    TRAP, unrelated to layout and not fixable here: `select_time` now demands both
    arguments have the same type (core2D.py:473), so the legacy idiom
    `select_time(30, obj.get_end_time())` used at eight call sites raises
    TypeError. rev2 callers must pass two floats:
    `obj.select_time(30.0, float(obj.taxis[-1]))`.
    """
    from fiberis.analyzer.Data2D import Data2D_XT_DSS
    cls = cls or Data2D_XT_DSS.DSS2D
    obj = cls()
    obj.data = np.ascontiguousarray(panel.data.T)   # back to (n_x, n_t)
    obj.taxis = panel.taxis.copy()
    obj.daxis = panel.daxis.copy()
    obj.start_time = panel.start_time
    try:
        obj.set_name(os.path.basename(panel.source_path))
    except Exception:
        obj.name = os.path.basename(panel.source_path)
    return obj


def _inside_frozen(path):
    ap = os.path.abspath(path)
    for d in FROZEN_INPUT_DIRS:
        dd = os.path.abspath(d)
        if ap == dd or ap.startswith(dd + os.sep):
            return True
    return False


def save_panel(panel, out_path, *, layout=LAYOUT_DEPTH_MAJOR, overwrite=False,
               extra=None):
    """Write a normalised copy to a NEW location, stamped with its layout.

    The stamp is the durable fix: once `layout` is in the file, rule R2 resolves
    even a square panel and no future task has to infer anything. Every rev2 writer
    should stamp it. Zero of the 717 fiberis-format npz files in the repo are
    square today, so R5 will not be exercised by existing data and would otherwise
    rot untested until a run happens to record n_t == n_x.
    """
    out_path = os.path.abspath(out_path)
    if _inside_frozen(out_path):
        raise LayoutError(
            f"refusing to write inside a frozen input directory: {out_path}. "
            f"output/0211_simulation_MULTIstage is the only physical record of "
            f"the mixed-layout bug and is read-only for this round.")
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(f"{out_path} exists; refusing to overwrite "
                              f"(house rule 2)")
    if layout not in VALID_LAYOUTS:
        raise LayoutError(f"layout must be one of {VALID_LAYOUTS}, got {layout!r}")
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    arr = panel.data.T if layout == LAYOUT_DEPTH_MAJOR else panel.data
    np.savez(out_path, data=np.ascontiguousarray(arr), taxis=panel.taxis,
             daxis=panel.daxis, start_time=panel.start_time,
             **{LAYOUT_KEY: np.array(layout)}, **(extra or {}))
    return out_path


def normalize_directory(src_dir, dst_dir, *, layout=LAYOUT_DEPTH_MAJOR):
    """One-time conversion into a NEW directory; returns the conversion record."""
    src_dir, dst_dir = os.path.abspath(src_dir), os.path.abspath(dst_dir)
    if src_dir == dst_dir or _inside_frozen(dst_dir):
        raise LayoutError(f"dst_dir {dst_dir} must differ from src_dir and must "
                          f"not be inside {FROZEN_INPUT_DIRS}")
    rows = []
    for src in sorted(glob.glob(os.path.join(src_dir, '*.npz'))):
        p = load_panel(src)
        dst = os.path.join(dst_dir, os.path.basename(src))
        save_panel(p, dst, layout=layout)
        rows.append({
            'src': src, 'src_sha256': p.sha256,
            'src_mtime_utc': datetime.datetime.utcfromtimestamp(
                os.path.getmtime(src)).isoformat() + 'Z',
            'detected_layout': p.detected_layout, 'evidence': p.evidence,
            'dst': dst, 'dst_sha256': _sha256(dst),
            'transposed': bool(p.detected_layout != layout),
            'shape_in': list(p.evidence['data_shape']),
            'shape_out': [p.n_x, p.n_t] if layout == LAYOUT_DEPTH_MAJOR
                         else [p.n_t, p.n_x],
        })
    return {'src_dir': src_dir, 'dst_dir': dst_dir, 'layout': layout, 'files': rows}


def trace_at_md(panel, md):
    """(idx, actual MD, time series at that MD) -- length n_t."""
    i = int(np.argmin(np.abs(panel.daxis - float(md))))
    return i, float(panel.daxis[i]), panel.data[:, i]


def profile_at_time(panel, t):
    """(idx, actual time, spatial profile at that time) -- length n_x."""
    i = int(np.argmin(np.abs(panel.taxis - float(t))))
    return i, float(panel.taxis[i]), panel.data[i, :]


def final_profile(panel):
    """Last spatial profile -- the correct replacement for `data[-1, :]`.

    101:156, :199 and :233 read `prev_result['data'][-1, :]` straight off the npz.
    That is right for the old-layout files on disk and WRONG for anything current
    fibeRIS writes, where it returns the time history of the last mesh node
    instead. Going through a Panel is correct for both.
    """
    return panel.data[-1, :]
