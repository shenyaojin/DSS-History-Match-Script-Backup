"""Data access for the rev2 round: gauges, geometry, frac hits, pumping, LF-DAS.

Pure loading. No solver calls, no plotting, no file writes, no manifest logic, no
global mutable state, so every function here is safe inside a
multiprocessing.Pool initializer (which is how r2_profile_inversion drives its
sweeps).

The load path is a bit-exact superset of `r1_run_calibration.load_window_data`:
the established results (D = 1150 / 550, RMSE 82.33 / 11.87 psi) are DEFINED by
the exact arrays that function produces, so the defaults here reproduce them and
the self-test asserts `np.array_equal` gauge by gauge. Two of those defaults look
like bugs and are kept deliberately:

* `rebase='per_gauge'` rebases each gauge to its OWN first in-window sample.
  Gauge 1's first sample is at 11:24:04.791 and gauges 2-15 are at 11:24:04.807,
  so the source axis leads the target axes by 16 ms. That asymmetry is baked into
  every established RMSE. `rebase='common'` is available for work that must
  cross-reference the DAS, and changes the numbers.
* `unique=True` on frac hits. Stage 1 stores MD 16696.914 twice, so
  mean(unique) = 16683.168 while mean(raw) = 16686.6045 -- a 3.44 ft (3-node)
  shift of the frac_centroid source.

Units: MD ft, pressure psi, time s, rate bpm, D ft^2/s. LF-DAS stays in RAW
COUNTS: two mutually inconsistent counts->strain-rate scalars exist in the repo
(ratio 1.365) and resolving them is F1's deliverable, so nothing here converts
by default.
"""

import datetime
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = os.environ.get(
    'REV2_REPO_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 os.pardir, os.pardir, os.pardir)))

SWELL_GAUGE_MD_NPZ = "data/fiberis_format/s_well/geometry/gauge_md_swell.npz"
SWELL_GEOMETRY_NPZ = "data/fiberis_format/s_well/geometry/swell_geometry.npz"
SWELL_GAUGE_TEMPLATE = "data/fiberis_format/s_well/gauges/gauge{n}_data_swell.npz"
SWELL_FRAC_HIT_TEMPLATE = ("data/fiberis_format/s_well/geometry/frac_hit/"
                           "frac_hit_stage_{stage}_swell.npz")
SWELL_DAS_TEMPLATE = "data/fiberis_format/s_well/DAS/LFDASdata_stg{stage}{kind}_swell.npz"
PUMPING_DIR_TEMPLATE = "data/fiberis_format/prod/pumping_data/stage{stage}"
PROD_GAUGE_MD_NPZ = "data/fiberis_format/prod/geometry/gauge_md_prod.npz"
PROD_GAUGE_TEMPLATE = "data/fiberis_format/prod/gauges/gauge{n}_data_prod.npz"
PROD_PERF_TEMPLATE = "data/fiberis_format/prod/geometry/perf/prod_perf_stage_{stage}.npz"

N_SWELL_GAUGES = 15
PUMPING_CURVE_FILES = {"slurry_rate": "Slurry Rate.npz",
                       "treating_pressure": "Treating Pressure.npz",
                       "proppant_concentration": "Proppant Concentration.npz"}

# Frozen reference table. NOT the source of truth -- the npz is -- but asserted
# against it on load so a silent data regeneration cannot slip through.
GAUGE_MD_FT_REFERENCE = (16645, 16384, 16122, 15868, 15599, 15344, 15075, 14821,
                         14552, 14297, 14028, 13774, 13210, 12662, 12098)

DAS_CHANNEL_SPACING_FT = 3.32879

# Both scalars found in the repo, recorded, NEITHER applied by default.
DAS_COUNTS_TO_STRAINRATE = {
    "matching_104": 2.799e-8 / 10430.4,                                # 2.6835e-12
    "ratio_estimation_101": (1.55e-6 / (4 * 3.14 * 4.09 * 0.79)) / 10430.4,  # 3.6621e-12
}

# Acquisition artifact in LFDASdata_stg1_swell.npz: cross-channel RMS peaks at
# 705x the window median at t_window 604 s, and the same burst appears at 655x on
# the negative-MD downlead (fibre that is not in the well), so it is not
# formation signal. Leaving it in inflates per-channel window RMS by a median
# factor of 11.2.
STAGE1_DAS_ARTIFACT_ABS = (datetime.datetime(2020, 3, 16, 11, 33, 54, 691125),
                           datetime.datetime(2020, 3, 16, 11, 34, 14, 691125))
DEFAULT_ARTIFACT_SPANS = {(1, 'stage'): (STAGE1_DAS_ARTIFACT_ABS,)}

# The only pre-pumping quiet period in the stage-1 DAS file: 112 samples.
STAGE1_PRE_PUMPING_ABS = (datetime.datetime(2020, 3, 16, 10, 27, 39, 691125),
                          datetime.datetime(2020, 3, 16, 10, 29, 30, 691125))

# Measured reference drawdown profiles (gauges 1..15, psi) so E2/E4 can check
# they reproduce them before trusting a rebuilt Fig. 7b.
DRAWDOWN_REFERENCE = {
    ('2020-04-01', '2020-06-01'): (2415.03, 3156.46, 3258.51, 4369.47, 4072.93,
                                   4270.80, 4259.89, 4293.12, 3569.37, 4141.04,
                                   4212.83, 3395.65, 3790.93, 3915.33, 3800.13),
    ('2020-04-01', '2021-07-01'): (4127.79, 4929.46, 4865.15, 6138.02, 6020.00,
                                   6033.33, 5802.88, 6011.72, 4819.90, 5754.35,
                                   5871.52, 4620.47, 5530.17, 5499.39, 5512.80),
}


def repo_path(*parts):
    """Join onto REPO_ROOT and fail with an actionable message if absent."""
    p = os.path.join(REPO_ROOT, *parts)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found. All rev2 paths are repo-relative by design; run with "
            f"CWD = repo root (or set REV2_REPO_ROOT). CWD={os.getcwd()}")
    return p


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Window:
    md_min_ft: float
    md_max_ft: float
    t_start: datetime.datetime
    t_end: datetime.datetime

    @property
    def duration_s(self):
        return (self.t_end - self.t_start).total_seconds()


R1_WINDOW = Window(md_min_ft=15000.0, md_max_ft=16750.0,
                   t_start=datetime.datetime(2020, 3, 16, 11, 24, 0),
                   t_end=datetime.datetime(2020, 3, 16, 11, 45, 0))


@dataclass(frozen=True)
class GaugeMDTable:
    numbers: np.ndarray
    md_ft: np.ndarray

    def md_of(self, gauge):
        g = int(gauge)
        if not (1 <= g <= len(self.md_ft)):
            raise ValueError(f"gauge {g} outside 1..{len(self.md_ft)}")
        return float(self.md_ft[g - 1])

    def in_window(self, w):
        m = (self.md_ft >= w.md_min_ft) & (self.md_ft <= w.md_max_ft)
        return self.numbers[m].astype(np.int64)

    def as_dict(self):
        return {int(n): float(m) for n, m in zip(self.numbers, self.md_ft)}


@dataclass(frozen=True)
class GaugeSeries:
    gauge: int
    md_ft: float
    taxis_s: np.ndarray
    t0_abs: datetime.datetime
    raw_psi: np.ndarray
    delta_psi: np.ndarray
    baseline_psi: float
    frame: object

    @property
    def n(self):
        return int(self.taxis_s.size)

    @property
    def t_total_s(self):
        return float(self.taxis_s[-1])


@dataclass(frozen=True)
class GaugeWindow:
    window: Window
    series: dict
    numbers: np.ndarray
    md_ft: np.ndarray
    rebase: str
    baseline: str
    t_ref_abs: datetime.datetime


@dataclass(frozen=True)
class Mesh:
    x: np.ndarray
    dx_ft: float
    lo_ft: float
    hi_ft: float
    pad_low_ft: float
    pad_high_ft: float
    window_md: tuple

    @property
    def nx(self):
        return int(self.x.size)

    def index_of(self, md, *, tol_ft=None):
        md = float(md)
        if md < self.x[0] - self.dx_ft / 2.0 or md > self.x[-1] + self.dx_ft / 2.0:
            # Clamping silently is how a mis-specified source ends up sitting on
            # the Neumann boundary, which changes the physics rather than the
            # numerics.
            raise ValueError(
                f"MD {md} is outside the mesh [{self.x[0]}, {self.x[-1]}] ft "
                f"by more than half a cell")
        i = int(np.argmin(np.abs(self.x - md)))  # == mesh_utils.locate, r1:142
        if tol_ft is not None and abs(self.x[i] - md) > tol_ft:
            raise ValueError(f"MD {md} snaps to node {i} at {self.x[i]} ft, "
                             f"{abs(self.x[i] - md)} ft away (tol {tol_ft})")
        return i

    def indices_of(self, mds, *, tol_ft=None):
        return np.asarray([self.index_of(m, tol_ft=tol_ft) for m in mds],
                          dtype=np.int64)

    def snap_error_ft(self, md):
        return float(self.x[self.index_of(md)] - float(md))

    def window_mask(self):
        lo, hi = self.window_md
        return (self.x >= lo) & (self.x <= hi)


@dataclass
class DASRecord:
    stage: int
    kind: str
    daxis_ft: np.ndarray
    taxis_s: np.ndarray
    t0_abs: datetime.datetime
    data: np.ndarray
    units: str
    scale_applied: Optional[float]
    source_path: str
    default_artifact_spans: tuple = ()

    def channel_index(self, md):
        return int(np.argmin(np.abs(self.daxis_ft - float(md))))

    def abs_times(self):
        base = np.datetime64(self.t0_abs, 'us')
        return base + (self.taxis_s * 1e6).astype('timedelta64[us]')

    def artifact_mask(self, spans=(), *, auto=False, z=20.0, pad_s=2.0):
        """Boolean KEEP mask over taxis_s. True means the sample is kept."""
        keep = np.ones(self.taxis_s.size, dtype=bool)
        for lo, hi in spans:
            if isinstance(lo, datetime.datetime):
                lo = (lo - self.t0_abs).total_seconds()
            if isinstance(hi, datetime.datetime):
                hi = (hi - self.t0_abs).total_seconds()
            keep &= ~((self.taxis_s >= float(lo)) & (self.taxis_s <= float(hi)))
        if auto:
            rms_t = np.sqrt(np.mean(np.asarray(self.data, dtype=float) ** 2, axis=0))
            med = float(np.median(rms_t))
            mad = float(np.median(np.abs(rms_t - med)))
            scale = 1.4826 * mad
            if scale > 0:
                bad = (rms_t - med) / scale > float(z)
                if bad.any():
                    t_bad = self.taxis_s[bad]
                    for tb in t_bad:
                        keep &= ~((self.taxis_s >= tb - pad_s)
                                  & (self.taxis_s <= tb + pad_s))
        return keep

    def to_strain_rate(self, key):
        """Return a NEW record scaled by one of DAS_COUNTS_TO_STRAINRATE.

        Explicit by design. The two scalars in the repo differ by 1.365x, so a
        silent default would push an unresolved 36% scale error into C5's
        threshold study and into any "the model overestimates far-field dP/dt"
        claim.
        """
        s = DAS_COUNTS_TO_STRAINRATE[key]
        return DASRecord(stage=self.stage, kind=self.kind,
                         daxis_ft=self.daxis_ft, taxis_s=self.taxis_s,
                         t0_abs=self.t0_abs,
                         data=np.asarray(self.data, dtype=float) * s,
                         units='strain_rate_per_s', scale_applied=float(s),
                         source_path=self.source_path,
                         default_artifact_spans=self.default_artifact_spans)


@dataclass(frozen=True)
class PumpingStage:
    stage: int
    curves: dict
    taxis_s: np.ndarray
    t0_abs: datetime.datetime
    file_start_abs: datetime.datetime
    file_end_abs: datetime.datetime

    def rate_bpm(self):
        return np.asarray(self.curves['slurry_rate'].data, dtype=float)

    def treating_pressure_psi(self):
        return np.asarray(self.curves['treating_pressure'].data, dtype=float)

    def pumping_start(self, *, threshold_bpm=1.0, hold_s=30.0):
        """First time the rate stays above threshold for hold_s.

        `Data1DPumpingCurve.get_start_time` (Data1D_PumpingCurve.py:16-38)
        hardcodes min_index = 0 and IGNORES its threshold argument, so it returns
        the file start, not a pumping event. 101_fiberis_matching.py:42-47 uses
        it, which is why the manuscript's phase-1/2/3 boundaries are file spans.
        Use this instead when a pumping event is what is meant, and say which.
        """
        rate = self.rate_bpm()
        ok = rate > float(threshold_bpm)
        dt = float(np.median(np.diff(self.taxis_s)))
        k = max(1, int(round(float(hold_s) / dt)))
        run = np.convolve(ok.astype(int), np.ones(k, dtype=int), mode='valid')
        hit = np.where(run == k)[0]
        if hit.size == 0:
            raise ValueError(f"stage {self.stage}: rate never exceeds "
                             f"{threshold_bpm} bpm for {hold_s} s")
        return self.t0_abs + datetime.timedelta(seconds=float(self.taxis_s[hit[0]]))

    def pumping_end(self, *, threshold_bpm=1.0):
        ok = np.where(self.rate_bpm() > float(threshold_bpm))[0]
        if ok.size == 0:
            raise ValueError(f"stage {self.stage}: rate never exceeds "
                             f"{threshold_bpm} bpm")
        return self.t0_abs + datetime.timedelta(seconds=float(self.taxis_s[ok[-1]]))


@dataclass(frozen=True)
class Trajectory:
    md_ft: np.ndarray
    x_ft: np.ndarray
    y_ft: np.ndarray
    z_ft: np.ndarray

    def xyz_at(self, md):
        md = np.asarray(md, dtype=float)
        return np.stack([np.interp(md, self.md_ft, self.x_ft),
                         np.interp(md, self.md_ft, self.y_ft),
                         np.interp(md, self.md_ft, self.z_ft)], axis=-1)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_gauge_md_table(path=SWELL_GAUGE_MD_NPZ, *, check_reference=True):
    """Gauge MD table. The file stores int64; everything downstream wants float."""
    from fiberis.analyzer.Geometry3D import DataG3D_md
    p = repo_path(path)
    frame = DataG3D_md.G3DMeasuredDepth()
    frame.load_npz(p)
    md = np.asarray(frame.data, dtype=float)
    numbers = np.arange(1, md.size + 1, dtype=np.int64)
    if check_reference and os.path.normpath(path) == os.path.normpath(SWELL_GAUGE_MD_NPZ):
        ref = np.asarray(GAUGE_MD_FT_REFERENCE, dtype=float)
        if md.size != ref.size or not np.array_equal(md, ref):
            raise ValueError(
                f"gauge MD table drifted from the frozen reference.\n"
                f"  file: {list(md)}\n  ref : {list(ref)}")
    return GaugeMDTable(numbers=numbers, md_ft=md)


def load_swell_trajectory(path=SWELL_GEOMETRY_NPZ):
    """Well trajectory. The identical xyz arrays are duplicated inside
    gauge_md_swell.npz and every frac-hit file; this is the single source."""
    z = np.load(repo_path(path), allow_pickle=True)
    try:
        return Trajectory(md_ft=np.asarray(z['data'], dtype=float),
                          x_ft=np.asarray(z['xaxis'], dtype=float),
                          y_ft=np.asarray(z['yaxis'], dtype=float),
                          z_ft=np.asarray(z['zaxis'], dtype=float))
    finally:
        z.close()


def load_window_gauges(window=R1_WINDOW, *, gauges=None, md_table=None,
                       series_template=SWELL_GAUGE_TEMPLATE,
                       baseline="first_sample", rebase="per_gauge",
                       require_strictly_increasing=True):
    """Crop the S-well gauges to a window. Defaults reproduce r1 bit-exactly."""
    from fiberis.analyzer.Data1D import Data1D_Gauge

    md_table = md_table or load_gauge_md_table()
    if gauges is None:
        numbers = md_table.in_window(window)
    else:
        numbers = np.asarray(sorted(int(g) for g in gauges), dtype=np.int64)
        for g in numbers:
            if not (1 <= g <= md_table.numbers.size):
                raise ValueError(f"gauge {g} outside 1..{md_table.numbers.size}")
    mds = np.asarray([md_table.md_of(int(g)) for g in numbers], dtype=float)

    loaded = {}
    for n, md in zip(numbers, mds):
        frame = Data1D_Gauge.Data1DGauge()
        frame.load_npz(repo_path(series_template.format(n=int(n))))
        frame.crop(window.t_start, window.t_end)
        taxis = np.asarray(frame.taxis, dtype=float)
        if taxis.size == 0:
            # core1D.crop:151-157 sets empty arrays and only logs at INFO. A
            # zero-length series propagates into np.interp/np.max as NaN or an
            # IndexError far from the cause, so fail here.
            raise ValueError(
                f"gauge {int(n)}: crop to [{window.t_start}, {window.t_end}] "
                f"returned no samples")
        if require_strictly_increasing and np.any(np.diff(taxis) <= 0):
            i = int(np.argmin(np.diff(taxis)))
            raise ValueError(
                f"gauge {int(n)}: taxis is not strictly increasing at index {i} "
                f"({taxis[i]} -> {taxis[i + 1]}); np.interp with a non-increasing "
                f"xp returns silently wrong values. The full records carry 13-32 "
                f"duplicate timestamps each.")
        loaded[int(n)] = (frame, taxis, np.asarray(frame.data, dtype=float), md)

    if rebase == 'per_gauge':
        t_ref = None
    elif rebase == 'common':
        t_ref = min(f.start_time for f, _, _, _ in loaded.values())
    else:
        raise ValueError(f"rebase must be 'per_gauge' or 'common', got {rebase!r}")

    series = {}
    for n in sorted(loaded):
        frame, taxis, raw, md = loaded[n]
        t0_abs = frame.start_time  # crop already advanced it to the first sample
        if rebase == 'common':
            taxis = taxis + (t0_abs - t_ref).total_seconds()
        if baseline == 'first_sample':
            base = float(raw[0])
        elif baseline == 'none':
            base = 0.0
        elif baseline.startswith('mean_pre:'):
            s = float(baseline.split(':', 1)[1])
            m = taxis <= s
            if not m.any():
                raise ValueError(f"gauge {n}: no samples with taxis <= {s}")
            base = float(np.mean(raw[m]))
        else:
            raise ValueError(f"unknown baseline {baseline!r}")
        delta = raw - base
        # r1 reloads the file to build this frame (r1_run_calibration.py:65-70);
        # a deep copy of the already-cropped frame gives identical arrays without
        # the second read.
        dframe = frame.copy()
        dframe.taxis = taxis
        dframe.data = delta
        series[n] = GaugeSeries(gauge=n, md_ft=float(md), taxis_s=taxis,
                                t0_abs=t0_abs, raw_psi=raw, delta_psi=delta,
                                baseline_psi=base, frame=dframe)

    return GaugeWindow(window=window, series=series, numbers=numbers, md_ft=mds,
                       rebase=rebase, baseline=baseline,
                       t_ref_abs=(t_ref if rebase == 'common'
                                  else series[int(numbers[0])].t0_abs))


def load_frac_hits(stage, *, unique=True, sort=True,
                   template=SWELL_FRAC_HIT_TEMPLATE):
    """Frac-hit MDs for one stage.

    `unique=True` reproduces r1_run_calibration.py:55 (np.unique also sorts) and
    is what produced every established result. Stage 1 stores 4 values of which
    16696.914 appears twice, so mean(unique) = 16683.168 but mean(raw) =
    16686.6045 -- 3.44 ft apart, 3 nodes on a dx = 1 ft mesh.
    """
    z = np.load(repo_path(template.format(stage=int(stage))), allow_pickle=True)
    try:
        vals = np.asarray(z['data'], dtype=float)
    finally:
        z.close()
    if unique:
        return np.unique(vals)
    return np.sort(vals) if sort else vals


def frac_hit_centroid(stage, *, unique=True):
    return float(np.mean(load_frac_hits(stage, unique=unique)))


def pick_source_gauge(gw, frac_hits, rule="nearest_gauge_to_stage1_frac_hits"):
    """(gauge number, frac-hit centroid MD). Mirrors r1_run_calibration.py:83-88."""
    if rule != "nearest_gauge_to_stage1_frac_hits":
        raise ValueError(f"Unsupported source selection_rule: {rule}")
    target = float(np.mean(np.asarray(frac_hits, dtype=float)))
    best = min(gw.series.values(), key=lambda s: abs(s.md_ft - target))
    return best.gauge, target


def load_pumping(stage, *, curves=None):
    """Load a stage's pumping curves; all three share taxis and start_time."""
    from fiberis.analyzer.Data1D import Data1D_PumpingCurve
    keys = list(PUMPING_CURVE_FILES) if curves is None else list(curves)
    folder = PUMPING_DIR_TEMPLATE.format(stage=int(stage))
    out = {}
    for k in keys:
        c = Data1D_PumpingCurve.Data1DPumpingCurve()
        c.load_npz(repo_path(folder, PUMPING_CURVE_FILES[k]))
        out[k] = c
    ref = out[keys[0]]
    taxis = np.asarray(ref.taxis, dtype=float)
    for k in keys[1:]:
        if out[k].start_time != ref.start_time or \
                not np.array_equal(np.asarray(out[k].taxis, dtype=float), taxis):
            raise ValueError(f"stage {stage}: curve {k} has a different time base "
                             f"from {keys[0]}")
    t0 = ref.start_time
    return PumpingStage(stage=int(stage), curves=out, taxis_s=taxis, t0_abs=t0,
                        file_start_abs=t0 + datetime.timedelta(seconds=float(taxis[0])),
                        file_end_abs=t0 + datetime.timedelta(seconds=float(taxis[-1])))


def stage_file_span(stage):
    """The FILE bounds, which is what 101's phase boundaries actually are."""
    ps = load_pumping(int(stage), curves=('slurry_rate',))
    return ps.file_start_abs, ps.file_end_abs


def manuscript_phase_windows():
    """101's phase windows, reproduced from the file spans it actually used.

    101_fiberis_matching.py:42-47 calls get_start_time/get_end_time, which ignore
    their threshold and return file bounds -- so these are NOT pumping events.
    Returns {'phase': (source gauge, t_start, t_end)}.
    """
    s7a, s7b = stage_file_span(7)
    s8a, s8b = stage_file_span(8)
    return {'phase1': (6, s7a, s7b), 'phase2': (6, s7b, s8a),
            'phase3': (7, s8a, s8b)}


def load_das_stage(stage, *, kind="stage", md_range=None, time_range=None,
                   as_float=True):
    """Load an LF-DAS panel, slicing BEFORE any dtype cast.

    Deliberately bypasses DSS2D.load_npz: that method casts the whole (5240,
    16765) int32 stage-1 array to float64 (core2D.py:247), i.e. 2.81 GB, before
    anything is selected. Boolean-mask indexing would also copy the full array, so
    contiguous integer slices are used.

    Returns data in (n_channels, n_time) with `units='counts'` and
    `scale_applied=None`. The default artifact spans are ATTACHED, never applied.

    TIME ALIGNMENT WARNING. The first DAS sample in the R1 window is at
    11:24:00.691125, while the gauges' first in-window sample is at 11:24:04.807.
    The often-quoted "aligned to within 0.7 s" is the offset from the NOMINAL
    window boundary; after each loader rebases to its own first sample the two
    axes differ by 4.115875 s (4.099875 s against gauge 1). Cross-DAS/gauge work
    must use absolute times, or `rebase='common'` plus this record's `t0_abs`.
    """
    suffix = '' if kind == 'stage' else '_interval'
    path = repo_path(SWELL_DAS_TEMPLATE.format(stage=int(stage), kind=suffix))
    z = np.load(path, allow_pickle=True)
    try:
        daxis = np.asarray(z['daxis'], dtype=float)
        taxis = np.asarray(z['taxis'], dtype=float)
        st = z['start_time']
        if isinstance(st, np.ndarray) and st.size == 1:
            st = st.item()
        if not isinstance(st, datetime.datetime):
            st = datetime.datetime.fromisoformat(str(st))

        if md_range is None:
            i0, i1 = 0, daxis.size
        else:
            sel = np.where((daxis >= md_range[0]) & (daxis <= md_range[1]))[0]
            if sel.size == 0:
                raise ValueError(f"no DAS channel in MD {md_range}")
            i0, i1 = int(sel[0]), int(sel[-1]) + 1

        if time_range is None:
            j0, j1 = 0, taxis.size
        else:
            if isinstance(time_range, (int, float)):
                lo, hi = 0.0, float(time_range)
            else:
                lo, hi = time_range
                if isinstance(lo, datetime.datetime):
                    lo = (lo - st).total_seconds()
                if isinstance(hi, datetime.datetime):
                    hi = (hi - st).total_seconds()
            sel = np.where((taxis >= float(lo)) & (taxis <= float(hi)))[0]
            if sel.size == 0:
                raise ValueError(f"no DAS sample in time range {time_range}")
            j0, j1 = int(sel[0]), int(sel[-1]) + 1

        block = z['data'][i0:i1, j0:j1]
        data = block.astype(np.float64) if as_float else block
        t_off = float(taxis[j0])
        return DASRecord(
            stage=int(stage), kind=kind, daxis_ft=daxis[i0:i1].copy(),
            taxis_s=taxis[j0:j1] - t_off,
            t0_abs=st + datetime.timedelta(seconds=t_off),
            data=data, units='counts', scale_applied=None, source_path=path,
            default_artifact_spans=DEFAULT_ARTIFACT_SPANS.get(
                (int(stage), kind), ()))
    finally:
        z.close()


def das_channel_noise_rms(rec, reference):
    """Per-channel RMS over an absolute-time reference interval, shape (n_ch,).

    `STAGE1_PRE_PUMPING_ABS` is the documented quiet reference: it is the only
    pre-pumping period in the stage-1 file and is just 112 s long, so it is a
    thin noise estimate and should be reported as such.
    """
    lo, hi = reference
    if isinstance(lo, datetime.datetime):
        lo = (lo - rec.t0_abs).total_seconds()
    if isinstance(hi, datetime.datetime):
        hi = (hi - rec.t0_abs).total_seconds()
    m = (rec.taxis_s >= float(lo)) & (rec.taxis_s <= float(hi))
    if not m.any():
        raise ValueError(f"reference interval {reference} selects no sample")
    blk = np.asarray(rec.data[:, m], dtype=float)
    return np.sqrt(np.mean(blk ** 2, axis=1))


def production_drawdown(t_start, t_end, *, gauges=None, md_table=None,
                        well='s_well'):
    """(gauge numbers, MD, drawdown psi) sorted by ASCENDING GAUGE NUMBER.

    drawdown = raw[0] - raw[-1] per gauge, reproducing
    103p_forward_modeling_viz_without_scalar.py:46-52.

    The file list is built by formatting the template over range(1, 16). Neither
    `os.listdir` (filesystem order here: gauge10, 3, 12, 15, 7, 2, 14, 6, 9, 13,
    1, 11, 5, 8, 4 -- the order the manuscript figure actually used, plotted
    against "Gauge Number" 1..15) nor `sorted()` (lexicographic, gauge10-15 before
    gauge1) gives 1..15, and both produce plausible-looking wrong profiles because
    every value is the same order of magnitude.
    """
    from fiberis.analyzer.Data1D import Data1D_Gauge
    if well == 's_well':
        md_table = md_table or load_gauge_md_table()
        template = SWELL_GAUGE_TEMPLATE
        n_max = N_SWELL_GAUGES
        mask_zeros = False
    elif well == 'prod':
        md_table = md_table or load_gauge_md_table(PROD_GAUGE_MD_NPZ,
                                                   check_reference=False)
        template = PROD_GAUGE_TEMPLATE
        n_max = md_table.numbers.size
        mask_zeros = True  # each producer gauge has one exact-zero dropout
    else:
        raise ValueError(f"well must be 's_well' or 'prod', got {well!r}")

    numbers = list(range(1, n_max + 1)) if gauges is None \
        else sorted(int(g) for g in gauges)
    out_n, out_md, out_dd = [], [], []
    for n in numbers:
        f = Data1D_Gauge.Data1DGauge()
        f.load_npz(repo_path(template.format(n=n)))
        f.crop(t_start, t_end)
        raw = np.asarray(f.data, dtype=float)
        if raw.size == 0:
            raise ValueError(f"gauge {n}: crop returned no samples")
        if mask_zeros:
            raw = raw[raw != 0.0]
            if raw.size == 0:
                raise ValueError(f"gauge {n}: all samples are exactly zero")
        out_n.append(n)
        out_md.append(md_table.md_of(n))
        out_dd.append(float(raw[0] - raw[-1]))
    return (np.asarray(out_n, dtype=np.int64), np.asarray(out_md, dtype=float),
            np.asarray(out_dd, dtype=float))


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

def build_mesh(window, pad_low_ft, pad_high_ft, dx_ft, *, hi_override_ft=None):
    """Uniform padded mesh.

    The count-based form `lo + dx*arange(n)` is used instead of
    `np.arange(lo, hi + dx/2, dx)`: they agree bit-for-bit at dx in
    {1.0, 0.5, 0.25, 2.0, 10.0} and give the same node count everywhere tested,
    but arange accumulates rounding at dx in {0.2, 0.1, 0.3} (up to 2.5e-8 ft),
    always in the count-based form's favour.

    `hi_override_ft` exists for the frac_centroid variant, where r2 uses
    hi = max(win_hi, src_md + pad_above_source_ft) (r2_profile_inversion.py:59).
    `x[-1]` can exceed `hi_ft` by up to dx; both are exposed so a manifest records
    what was actually built rather than what was asked for.
    """
    if isinstance(window, Window):
        win_lo, win_hi = window.md_min_ft, window.md_max_ft
    else:
        win_lo, win_hi = float(window[0]), float(window[1])
    dx_ft = float(dx_ft)
    lo = win_lo - float(pad_low_ft)
    hi = float(hi_override_ft) if hi_override_ft is not None \
        else win_hi + float(pad_high_ft)
    if dx_ft <= 0:
        raise ValueError(f"dx_ft must be > 0, got {dx_ft}")
    if hi <= lo:
        raise ValueError(f"mesh hi ({hi}) must exceed lo ({lo})")
    n = int(round((hi - lo) / dx_ft)) + 1
    x = lo + dx_ft * np.arange(n, dtype=float)
    return Mesh(x=x, dx_ft=dx_ft, lo_ft=lo, hi_ft=hi,
                pad_low_ft=float(pad_low_ft), pad_high_ft=float(pad_high_ft),
                window_md=(win_lo, win_hi))


def make_targets(gw, mesh, source_md_ft, *, exclude_gauges=()):
    """Target dicts with exactly the keys the verified core consumes.

    r1_calibration_core.evaluate_profile:257-264 reads gauge/md_ft/distance_ft/
    idx/taxis/data; misfit_for_profile:553-560 reads idx/taxis/data. `source_md_ft`
    is a float, not a gauge number, so the frac_centroid variant needs no special
    case: pass exclude_gauges=(src_gauge,) for 'gauge' mode and () for
    'frac_centroid'.
    """
    excl = {int(g) for g in exclude_gauges}
    out = []
    for n in sorted(gw.series):
        if n in excl:
            continue
        s = gw.series[n]
        out.append({'gauge': int(n), 'md_ft': float(s.md_ft),
                    'distance_ft': abs(float(s.md_ft) - float(source_md_ft)),
                    'idx': mesh.index_of(s.md_ft),
                    'taxis': s.taxis_s, 'data': s.delta_psi})
    return out


def setup_r1(window=R1_WINDOW, *, source_mode="gauge", pad_low_ft=5000.0,
             pad_high_ft=0.0, dx_ft=1.0, pad_above_source_ft=150.0,
             stage=1, **gauge_kwargs):
    """Drop-in replacement for r2_profile_inversion._setup, both source modes."""
    gw = load_window_gauges(window, **gauge_kwargs)
    fh = load_frac_hits(stage)
    src_gauge, centroid = pick_source_gauge(gw, fh)
    src = gw.series[src_gauge]

    if source_mode == 'gauge':
        src_md = float(src.md_ft)
        hi_override = None
        exclude = (src_gauge,)
    elif source_mode == 'frac_centroid':
        # Prescribing pressure at the frac-hit centroid turns the source gauge
        # into an independent target, which is the test of whether the near-source
        # high D is an artifact of pinning the boundary condition 38 ft away.
        src_md = float(centroid)
        hi_override = max(window.md_max_ft, src_md + float(pad_above_source_ft))
        exclude = ()
    else:
        raise ValueError(f"source_mode must be 'gauge' or 'frac_centroid', "
                         f"got {source_mode!r}")

    mesh = build_mesh(window, pad_low_ft, pad_high_ft, dx_ft,
                      hi_override_ft=hi_override)
    return dict(gauge_window=gw, src_gauge=src_gauge, src_series=src,
                src_md=src_md, fh_centroid=centroid, frac_hits=fh, mesh=mesh,
                source_idx=mesh.index_of(src_md),
                targets=make_targets(gw, mesh, src_md, exclude_gauges=exclude),
                t_total_s=src.t_total_s, source_mode=source_mode)


# ---------------------------------------------------------------------------
# Self-check (writes nothing)
# ---------------------------------------------------------------------------

def _selfcheck():
    import json
    import sys
    ok = True

    def report(name, passed, detail=''):
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
              + (f" -- {detail}" if detail else ''))

    sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts',
                                    'manuscript_well_leakage',
                                    'baseline_calibration'))
    from r1_run_calibration import load_config, load_window_data
    from r1_run_calibration import pick_source_gauge as r1_pick

    cfg, _ = load_config(os.path.join(REPO_ROOT, 'configs',
                                      'r1_baseline_calibration.json'))
    r1_series, r1_nums, r1_mds, r1_fh, _, _ = load_window_data(cfg)
    gw = load_window_gauges(R1_WINDOW)

    same = (list(gw.numbers) == [int(n) for n in r1_nums]
            and np.array_equal(gw.md_ft, r1_mds))
    for n in sorted(r1_series):
        s, r = gw.series[n], r1_series[n]
        same = same and np.array_equal(s.taxis_s, r['taxis']) \
            and np.array_equal(s.raw_psi, r['raw_psi']) \
            and np.array_equal(s.delta_psi, r['delta_psi']) \
            and s.md_ft == r['md_ft']
    report('1 load_window_gauges == r1 load_window_data (bitwise)', same)

    sg, cent = pick_source_gauge(gw, load_frac_hits(1))
    r1sg, r1cent = r1_pick(cfg, r1_series, r1_fh)
    report('2 source gauge / centroid', sg == r1sg == 1 and abs(cent - 16683.168) < 1e-3
           and abs(cent - r1cent) < 1e-12, f"gauge {sg}, centroid {cent:.4f}")

    mesh = build_mesh(R1_WINDOW, 5000.0, 0.0, 1.0)
    ref = np.arange(10000, 16750.5, 1.0)
    S = setup_r1()
    report('3 build_mesh == r1 arange mesh', np.array_equal(mesh.x, ref)
           and mesh.nx == 6751 and S['source_idx'] == 6645,
           f"nx={mesh.nx}, source_idx={S['source_idx']}")

    tg = S['targets']
    report('4 make_targets matches r1',
           [t['gauge'] for t in tg] == [2, 3, 4, 5, 6, 7]
           and [t['idx'] for t in tg] == [6384, 6122, 5868, 5599, 5344, 5075]
           and [int(t['distance_ft']) for t in tg] == [261, 523, 777, 1046, 1301, 1570])

    S2 = setup_r1(source_mode='frac_centroid')
    report('5 frac_centroid variant', S2['mesh'].nx == 6834
           and S2['source_idx'] == 6683 and len(S2['targets']) == 7,
           f"nx={S2['mesh'].nx}, idx={S2['source_idx']}, "
           f"snap={S2['mesh'].snap_error_ft(S2['src_md']):.3f} ft")

    rec = load_das_stage(1, md_range=(15000, 16750),
                         time_range=(R1_WINDOW.t_start, R1_WINDOW.t_end))
    report('6 DAS stage-1 window', rec.data.shape == (525, 1260)
           and abs(rec.daxis_ft[0] - 15003.1836) < 1e-3
           and abs(rec.daxis_ft[-1] - 16747.4748) < 1e-3
           and np.all(np.diff(rec.taxis_s) == 1.0)
           and int(np.isnan(rec.data).sum()) == 0,
           f"shape {rec.data.shape}, t0 {rec.t0_abs.isoformat()}")
    return ok


if __name__ == '__main__':
    import sys
    print("rev2_data self-check (writes nothing)")
    sys.exit(0 if _selfcheck() else 1)
