"""The rev2 run manifest: write it, and check it later.

House rule 3: a solver run without a manifest is invalid. This module generalises
the manifest block already implemented in `r1_run_calibration.py:706-835` and
closes the three gaps the audit found, plus one regression:

* r1 hashes exactly TWO code files (its runner and its core). Nothing pins
  fibeRIS, the loaders or the figure code, and `.git/` is empty so there is no
  commit to fall back on. `code_closure()` discovers every repo `.py` currently in
  `sys.modules` -- which captures fibeRIS for free, since it lives inside the repo
  at `fibeRIS/src/fiberis` (measured: 12 files, 112 KB, 1.8 ms to hash).
* r2's emitter (`r2_profile_inversion.py:199-215`) dropped config_path, the
  matplotlib version, fiberis_path and EVERY input-data hash, so that manifest
  carries no data provenance at all. `write_manifest` makes inputs a required
  argument and additionally sweeps the config for file paths.
* The source protocol, the numerical parameters (theta, interface_avg, w,
  adaptive tolerances, realised step counts) and the output inventory were
  recorded nowhere. They are required keyword-only arguments here, so omitting one
  is a TypeError at the call site, before any work happens.

Design principle, applied everywhere: RECORD REALISED, NOT REQUESTED -- and record
both. Every builder takes the array the run actually used (mesh, taxis, boolean
barrier mask) and derives the realised numbers itself, so the manifest cannot
disagree with what the solver did. A requested/realised pair that disagrees beyond
tolerance lands in the top-level `discrepancies` list rather than being silently
reconciled: recording `w_requested` instead of the applied mask is exactly how the
0.076-69 ft effective-width spread went unnoticed for a year.

Stdlib + numpy only. fibeRIS is never imported, so `verify()` runs on a machine
where the solver is broken.
"""

import datetime
import glob as _glob
import hashlib
import json
import os
import platform
import socket
import sys
import time as _time
import warnings

import numpy as np

# 1.1 (2026-09-01 repair batch): source_protocol gained per-phase groups, so
# `duplicate_mesh_idx` now means "repeated WITHIN one solve" -- in a /1
# document from a multi-phase chain it may also mean "re-driven in a later
# phase". `outputs.config_paths_excluded_from_input_scan` is new, and a /1
# document may carry a declared output hashed as an `auto_from_config` input.
# Readers must not compare this string for equality; both are valid.
# 1.2 (2026-09-02 A4_repair2, acting on the independent adversarial review):
# `verify()` now lets manifest_self drift and a MISSING sidecar reach `status`
# (they were computed and printed but never scored, so an edited manifest audited
# clean); an inline config's sha256 is taken from the JSONIFIED payload, which is
# what verify() re-hashes, so np.int64 / NaN / ndarray / long-list / NONE_DECLARED
# configs round-trip; a config-named file whose mtime postdates `started_utc` is
# recorded under `outputs.config_paths_excluded_from_input_scan` with reason
# 'written_during_run' instead of being hashed as an input; and a run that used
# worker processes without declaring `worker_modules=` / `require_modules=` /
# `extra_code_files=` carries a `worker_code_closure_undeclared` discrepancy that
# verify() scores as `incomplete`. A /1.1 document stays readable and is scored by
# the same rules; only /1.2 documents can carry the new fields. Readers must not
# compare this string for equality.
SCHEMA_VERSION = "rev2-manifest/1.2"
ROUND_TAG = "rev2_20260901"

UNITS = {
    "D": "ft^2/s", "md": "ft", "pressure": "psi", "time": "s",
    "Gamma_psi_inv": 8.94e-9, "strain": "dimensionless",
    "lambda_leak": "s^-1", "barrier_half_width_w": "ft",
    "ratio": "D_barrier/D_baseline, dimensionless",
}

_ALLOWED_ROLES = ("config", "gauge_series", "geometry", "pumping", "das",
                  "prior_run_output", "mesh", "auto_from_config", "other")
_OUTPUT_ROLES = ("arrays_npz", "figure_png", "figure_pdf", "csv", "summary_txt",
                 "report_md", "log", "manifest_json", "json", "other")
_EXTERNAL_PACKAGES = ("numpy", "scipy", "matplotlib", "pandas", "h5py",
                      "sklearn", "numba", "netCDF4")


class _Sentinel:
    """Explicit 'this run has none', distinguishable from 'I forgot'."""

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return self._name

    def __bool__(self):
        return False


NONE_DECLARED = _Sentinel("NONE_DECLARED")


class ManifestError(Exception):
    pass


class ManifestIncomplete(ManifestError):
    pass


class OutputMissing(ManifestError):
    pass


class ManifestDrift(ManifestError):
    pass


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def repo_root():
    """Directory containing both `scripts` and `fibeRIS`."""
    env = os.environ.get('BAKKEN_REPO_ROOT')
    if env:
        return os.path.realpath(env).rstrip(os.sep)
    here = os.path.dirname(os.path.realpath(__file__))
    while True:
        if os.path.isdir(os.path.join(here, 'scripts')) and \
                os.path.isdir(os.path.join(here, 'fibeRIS')):
            return here.rstrip(os.sep)
        parent = os.path.dirname(here)
        if parent == here:
            raise ManifestError(
                "could not locate the repo root (a directory holding both "
                "'scripts' and 'fibeRIS'); set BAKKEN_REPO_ROOT")
        here = parent


def _cache_path():
    return os.path.join(repo_root(), 'output', ROUND_TAG, '.manifest_hash_cache.json')


def _load_cache():
    try:
        with open(_cache_path()) as fh:
            c = json.load(fh)
        return c if isinstance(c, dict) else {}
    except Exception:
        return {}  # a corrupt or absent cache is simply empty


def _store_cache(cache):
    p = _cache_path()
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + f'.tmp{os.getpid()}'
        with open(tmp, 'w') as fh:
            json.dump(cache, fh)
        os.replace(tmp, p)  # atomic: a dozen rev2 tasks share this file
    except Exception:
        pass


def sha256_file(path, *, use_cache=True):
    """1 MiB-chunked sha256, byte-identical to r1_calibration_core.file_sha256.

    Cached on (realpath, size, mtime_ns) because hashing is not free at this
    scale: data/ is 49 GB and one 805 MB npz takes 4.5 s. Without the cache
    someone eventually trims the input list "for speed", which is how r2 ended up
    with zero data provenance. verify() always passes use_cache=False -- a file
    edited inside one mtime tick would otherwise verify clean.
    """
    rp = os.path.realpath(path)
    st = os.stat(rp)
    key = f"{rp}|{st.st_size}|{st.st_mtime_ns}"
    cache = _load_cache() if use_cache else {}
    if use_cache and key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(rp, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    digest = h.hexdigest()
    if use_cache:
        cache[key] = digest
        _store_cache(cache)
    return digest


def sha256_array(a):
    """Byte-order-pinned hash of an array, so mesh/D/taxis identity is portable."""
    return hashlib.sha256(
        np.ascontiguousarray(a, dtype='<f8').tobytes()).hexdigest()


def _rel(path, root=None):
    root = root or repo_root()
    ap = os.path.realpath(path)
    if ap.startswith(root + os.sep):
        return os.path.relpath(ap, root).replace(os.sep, '/')
    return ap.replace(os.sep, '/')


# A file the run wrote can carry an mtime a hair BEFORE the recorded started_utc
# -- a log opened during start-up, a coarse filesystem timestamp, NFS clock skew.
# Measured locally: os.path.getmtime lagged datetime.now(utc) by 1.2 ms on a file
# created immediately after the stamp. 2 s is far above that and far below the
# gap in any real run (D2: 86 minutes).
_RUN_PRODUCT_MTIME_TOL_S = 2.0

# Fallback basis when the caller did not pass started_utc. This module is imported
# at the top of every task script, so its import time is a lower bound on the run's
# start; a file whose mtime postdates it was written after the run began. Weaker
# evidence than an explicit started_utc, so the exclusion record says which basis
# fired.
_IMPORT_TIME = _time.time()


def _is_output_key(dotted):
    """Does this dotted config path sit under an 'output'/'outputs' subtree?"""
    return any(seg.split('[')[0].lower() in ('output', 'outputs')
               for seg in str(dotted).split('.'))


def _parse_utc(ts):
    """Epoch seconds from an ISO-8601 stamp, or None. A naive stamp is UTC."""
    if not ts:
        return None
    try:
        dt = datetime.datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp()


def _worker_hint(parallel):
    """(largest worker count, lower-cased free text) sniffed from `parallel`.

    The block is free-form across the round -- 'processes', 'n_workers',
    'processes_used', 'mode' -- so both signals are collected and either one can
    indicate that a child process ran code.
    """
    n, txt = 0, ''
    if parallel is NONE_DECLARED or parallel is None:
        return 0, ''
    if isinstance(parallel, dict):
        for k, v in parallel.items():
            kl = str(k).lower()
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool) and \
                    any(w in kl for w in ('process', 'worker', 'nproc', 'njob')):
                n = max(n, int(v))
            elif isinstance(v, str):
                txt += ' ' + v.lower()
    else:
        txt = ' ' + str(parallel).lower()
    return n, txt


_WORKER_WORDS = ('multiprocessing', 'pool(', 'pool ', '.pool', 'joblib',
                 'concurrent.futures', 'processpool', 'mpi')


def _uses_workers(parallel):
    """Did this run execute code in a child process?"""
    n, txt = _worker_hint(parallel)
    return bool(n > 1 or any(w in txt for w in _WORKER_WORDS))


def _mtime_utc(path):
    return datetime.datetime.fromtimestamp(
        os.path.getmtime(path), datetime.timezone.utc).isoformat()


def file_record(path, *, role, logical_name=None, use_cache=True):
    if role not in _ALLOWED_ROLES:
        raise ValueError(f"role must be one of {_ALLOWED_ROLES}, got {role!r}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"declared input not found: {os.path.abspath(path)}")
    return {
        'logical_name': logical_name or os.path.basename(path),
        'path': _rel(path), 'abspath': os.path.realpath(path), 'role': role,
        'sha256': sha256_file(path, use_cache=use_cache),
        'size_bytes': int(os.path.getsize(path)),
        'mtime_utc': _mtime_utc(path),  # informational; verify never compares it
    }


def assert_absent(paths):
    """Mechanise house rule 2 instead of relying on discipline."""
    exists = [p for p in paths if os.path.exists(p)]
    if exists:
        raise FileExistsError(
            "refusing to overwrite existing output(s) (house rule 2): "
            + ", ".join(os.path.abspath(p) for p in exists))
    return None


def _jsonify(obj, *, max_array_len=64, _path='', _none=None, _nonfinite=None):
    """JSON-safe conversion that never invents or drops data.

    Deliberately NOT r1's `clean()` (r1_run_calibration.py:707-718), which deletes
    any key literally named 'grid'/'curve'/'rows'/'per_gauge_mse'/
    'bootstrap_argmins' ANYWHERE in the tree -- including inside config_resolved,
    so a config key called "rows" would silently vanish from the provenance
    record. Long float arrays are summarised rather than dropped, with a hash so
    identity survives. Unknown types raise, naming the dotted path, because a
    str()'d DataFrame in a manifest is worse than a crash.
    """
    _none = [] if _none is None else _none
    _nonfinite = [] if _nonfinite is None else _nonfinite

    def rec(o, p):
        if o is NONE_DECLARED:
            _none.append(p)
            return None
        if o is None or isinstance(o, (bool, str)):
            return o
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (int, np.integer)):
            return int(o)
        if isinstance(o, (float, np.floating)):
            v = float(o)
            if not np.isfinite(v):
                _nonfinite.append(p)
                return ("NaN" if np.isnan(v)
                        else ("Infinity" if v > 0 else "-Infinity"))
            return v
        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()
        if isinstance(o, _Sentinel):
            _none.append(p)
            return None
        if isinstance(o, dict):
            return {str(k): rec(v, f"{p}.{k}" if p else str(k))
                    for k, v in o.items()}
        if isinstance(o, np.ndarray):
            if o.ndim == 1 and o.size > max_array_len and \
                    np.issubdtype(o.dtype, np.number):
                return {'__summarised__': True, 'n': int(o.size),
                        'dtype': str(o.dtype), 'min': float(np.min(o)),
                        'max': float(np.max(o)), 'first': float(o[0]),
                        'last': float(o[-1]), 'sha256': sha256_array(o)}
            return [rec(v, f"{p}[{i}]") for i, v in enumerate(o.tolist())]
        if isinstance(o, (list, tuple, set)):
            seq = list(o)
            if len(seq) > max_array_len and all(
                    isinstance(v, (int, float, np.integer, np.floating))
                    and not isinstance(v, bool) for v in seq):
                a = np.asarray(seq, dtype=float)
                return {'__summarised__': True, 'n': int(a.size),
                        'dtype': str(a.dtype), 'min': float(np.min(a)),
                        'max': float(np.max(a)), 'first': float(a[0]),
                        'last': float(a[-1]), 'sha256': sha256_array(a)}
            return [rec(v, f"{p}[{i}]") for i, v in enumerate(seq)]
        raise TypeError(f"cannot serialise {type(o).__name__} at manifest path "
                        f"'{p or '<root>'}'")

    out = rec(obj, _path)
    return out, _none, _nonfinite


# ---------------------------------------------------------------------------
# Code closure
# ---------------------------------------------------------------------------

def code_closure(*, require=(), extra=(), root=None):
    """Hash every repo .py currently imported, plus whatever the caller declares.

    Snapshot this at manifest-write time, not at import time: with only
    `import r1_run_calibration` the repo closure is 2 files, and after the lazy
    imports inside `load_window_data` have run it is 13 (Data1D_Gauge, core1D,
    DataG3D_md, coreG3D, history_utils, signal_utils). A closure taken too early
    silently records a smaller set and drift in the loaders or in matbuilder goes
    undetected -- the exact failure this function exists to prevent.

    Automatic discovery is the FLOOR; `require` is the caller's assertion that the
    modules it believes it used are actually in there. A module imported only
    inside a multiprocessing worker never enters the parent's sys.modules, so such
    files must be passed via `extra`.
    """
    root = root or repo_root()
    files = {}

    def add(path, module=None):
        rp = os.path.realpath(path)
        if not rp.startswith(root + os.sep) or not rp.endswith('.py'):
            return False
        if '/site-packages/' in rp or '/.venv/' in rp:
            return False
        if not os.path.exists(rp):
            return False
        rel = _rel(rp, root)
        if rel not in files:
            files[rel] = {'sha256': sha256_file(rp),
                          'size_bytes': int(os.path.getsize(rp)),
                          'mtime_utc': _mtime_utc(rp), 'module': module}
        elif files[rel]['module'] is None and module:
            files[rel]['module'] = module
        return True

    main_script = None
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if f:
            add(f, name)
    main = sys.modules.get('__main__')
    if main is not None and getattr(main, '__file__', None):
        if add(main.__file__, '__main__'):
            main_script = _rel(main.__file__, root)
    for e in extra:
        if not add(e, None):
            raise ManifestIncomplete(
                f"extra_code_files entry is not a repo .py file: {e}")

    for name in require:
        mod = sys.modules.get(name)
        f = getattr(mod, '__file__', None) if mod is not None else None
        if f is None or not os.path.realpath(f).startswith(root + os.sep):
            raise ManifestIncomplete(
                f"required module not in code closure: {name} "
                f"(import it before calling write_manifest, or pass its file via "
                f"extra_code_files=)")

    fingerprint = "\n".join(f"{p}:{files[p]['sha256']}" for p in sorted(files))
    packages = {}
    for pkg in _EXTERNAL_PACKAGES:
        m = sys.modules.get(pkg)
        if m is not None and getattr(m, '__version__', None):
            packages[pkg] = str(m.__version__)
    return {
        'closure_mode': ('sys.modules scan restricted to repo root, snapshotted '
                         'at manifest-write time'),
        'root': root, 'n_files': len(files), 'files': files,
        'closure_sha256': hashlib.sha256(fingerprint.encode()).hexdigest(),
        'required_modules': list(require),
        'extra_declared': [_rel(e, root) for e in extra],
        'main_script': main_script,
        'external_packages': packages,
        'caveat': ('modules imported lazily inside functions appear only if that '
                   'function has already run in THIS process; snapshot at write '
                   'time and use require_modules='),
    }


# ---------------------------------------------------------------------------
# Group 1: source protocol
# ---------------------------------------------------------------------------

_BUILDER = '__builder__'

_APPLICATIONS = ('dirichlet_node', 'volumetric_source_term', 'neumann_flux',
                 'initial_condition_only')
_TIME_LEVELS = ('n', 'n+1', 'theta-weighted')
_DRIVER_KINDS = ('gauge_series', 'pumping_curve', 'synthetic', 'constant')
_BASELINE_REMOVALS = ('subtract_first_sample', 'none_absolute_psi',
                      'subtract_constant', 'other')


def driver_record(*, kind, baseline_removal, value_units, series_path=None,
                  gauge_number=None, gauge_md_ft=None, taxis=None, values=None,
                  time_start=None, time_end=None):
    """What actually drove a source node.

    `baseline_removal` is required because 101 seeds absolute psi
    (101:132-133) while r1/r2 subtract the first sample; without it two runs'
    numbers are not comparable and nothing in the manifest says so.
    """
    if kind not in _DRIVER_KINDS:
        raise ValueError(f"kind must be one of {_DRIVER_KINDS}, got {kind!r}")
    if baseline_removal not in _BASELINE_REMOVALS:
        raise ValueError(f"baseline_removal must be one of {_BASELINE_REMOVALS}")
    if value_units not in ('psi', 'delta_psi'):
        raise ValueError("value_units must be 'psi' or 'delta_psi'")
    rec = {_BUILDER: 'driver_record', 'kind': kind,
           'baseline_removal': baseline_removal, 'value_units': value_units,
           'series_path': None if series_path is None else _rel(series_path),
           'gauge_number': None if gauge_number is None else int(gauge_number),
           'gauge_md_ft': None if gauge_md_ft is None else float(gauge_md_ft),
           'time_start': time_start, 'time_end': time_end}
    if taxis is not None:
        t = np.asarray(taxis, dtype=float)
        rec.update(n_samples=int(t.size), t_first_s=float(t[0]),
                   t_last_s=float(t[-1]),
                   sample_dt_s=float(np.median(np.diff(t))) if t.size > 1 else None,
                   taxis_sha256=sha256_array(t))
    if values is not None:
        v = np.asarray(values, dtype=float)
        rec.update(value_min=float(np.min(v)), value_max=float(np.max(v)),
                   values_sha256=sha256_array(v))
    return rec


def source_record(mesh, *, md_requested_ft, mesh_idx, driver, label=None,
                  excluded_from_misfit=True, index_in_source_list=None):
    """One applied source node, with its snap error made visible.

    Node snapping is up to 0.067 ft in the real 101 case; recording the requested
    MD alone would hide it.
    """
    m = np.asarray(mesh, dtype=float)
    i = int(mesh_idx)
    if not (0 <= i < m.size):
        raise ValueError(f"mesh_idx {i} outside mesh of length {m.size}")
    if not isinstance(driver, dict) or driver.get(_BUILDER) != 'driver_record':
        raise ManifestIncomplete("driver must be built by driver_record()")
    return {_BUILDER: 'source_record', 'label': label,
            'index_in_source_list': index_in_source_list,
            'md_requested_ft': float(md_requested_ft), 'mesh_idx': i,
            'mesh_md_ft': float(m[i]),
            'snap_error_ft': float(m[i] - float(md_requested_ft)),
            'driver': driver,
            'excluded_from_misfit': bool(excluded_from_misfit)}


def _is_source_record(o):
    return isinstance(o, dict) and o.get(_BUILDER) == 'source_record'


def _normalise_source_phases(sources, phase_labels):
    """Accept EITHER a flat list of source_record()s or one list per phase.

    A flat list is one solve, exactly as before. A list of lists is a multi-phase
    chain (A5 drives phases 1 and 2 from the SAME six stage-7 nodes with different
    series), which is the case the single-solve duplicate check misfired on. The
    two forms may not be mixed: a mixture is far more likely to be a bug in the
    caller than an intent.
    """
    raw = list(sources)
    if not raw:
        raise ManifestIncomplete("sources must list at least one applied node")
    if all(_is_source_record(s) for s in raw):
        phases, nested = [raw], False
    elif all(not isinstance(s, dict) and isinstance(s, (list, tuple))
             for s in raw):
        phases, nested = [list(g) for g in raw], True
        for k, g in enumerate(phases):
            if not g:
                raise ManifestIncomplete(
                    f"phase {k} declares no source node; a phase that applies no "
                    f"source must be left out, not declared empty")
            for s in g:
                if not _is_source_record(s):
                    raise ManifestIncomplete(
                        "every entry of every phase must come from source_record()")
    else:
        raise ManifestIncomplete(
            "sources must be EITHER a flat list of source_record()s (one solve) "
            "OR a list of per-phase lists of them (a multi-phase chain); the two "
            "forms cannot be mixed")
    if phase_labels is None:
        labels = [None] if not nested else [f"phase{k + 1}"
                                            for k in range(len(phases))]
    else:
        labels = [str(x) for x in phase_labels]
        if len(labels) != len(phases):
            raise ManifestIncomplete(
                f"phase_labels has {len(labels)} entries for {len(phases)} "
                f"phase(s)")
    return phases, labels, nested


def source_protocol(*, application, solver_class, placement_rule, sources,
                    targets, time_level, phase_chaining, boundary_conditions,
                    phase_labels=None):
    """How the drive was applied. Every argument except phase_labels is required.

    `application='dirichlet_node'` is the fact that today survives only as prose
    in `configs/r1_baseline_calibration.json:source.comment`; fibeRIS imposes it
    that way at matbuilder.py:73-76 (single) and :150-161 (multi).

    `sources` takes two forms, mirroring `numerics(time=...)`, which already
    accepts one time_record() per solve:

      * a FLAT list of source_record()s -- one solve, the original behaviour;
      * a list of PER-PHASE lists of source_record()s -- a multi-phase chain.

    The duplicate-mesh-index check is per phase, because that is the failure it
    was written to catch: two frac hits snapping to one node on a dx = 10 ft mesh
    (102r/103r) means a run declaring 6 sources applies 5, and within ONE solve
    the second Dirichlet row silently overwrites the first. ACROSS phases the same
    node recurring is normal -- A5's phases 1 and 2 legitimately drive the same six
    stage-7 nodes from different gauge series -- so it is recorded as
    `shared_mesh_idx_across_phases`, not flagged. Declaring such a chain as one
    flat list of 18 raised a false `duplicate_source_mesh_idx` error before this;
    pass the per-phase form instead.

    `sources` in the returned document stays the FLAT list in declaration order,
    so existing readers are unaffected; per-phase detail is added under `phases`.
    """
    if application not in _APPLICATIONS:
        raise ValueError(f"application must be one of {_APPLICATIONS}")
    if time_level not in _TIME_LEVELS:
        raise ValueError(f"time_level must be one of {_TIME_LEVELS}")
    if not isinstance(solver_class, str) or not solver_class:
        raise ManifestIncomplete("solver_class must be a non-empty string")
    if not isinstance(placement_rule, str) or not placement_rule:
        raise ManifestIncomplete("placement_rule must be a non-empty string")
    phases, labels, nested = _normalise_source_phases(sources, phase_labels)

    disc = []
    flat, phase_recs, dup_all = [], [], []
    for k, (grp, label) in enumerate(zip(phases, labels)):
        idxs_k = [s['mesh_idx'] for s in grp]
        dup_k = sorted({i for i in idxs_k if idxs_k.count(i) > 1})
        dup_all.extend(dup_k)
        if dup_k:
            # Two frac hits can snap to one node on a coarse mesh (102r/103r use
            # dx = 10 ft), so a run declaring 6 sources may apply 5. This is a
            # WITHIN-SOLVE error; sharing a node between phases is not.
            where = f"phase {k}" + (f" ('{label}')" if label else "")
            disc.append({'kind': 'duplicate_source_mesh_idx',
                         'phase_index': k, 'phase_label': label,
                         'detail': f"{where} carries more than one declared "
                                   f"source on mesh indices {dup_k}; within one "
                                   f"solve the run applies fewer independent "
                                   f"sources than it declares",
                         'severity': 'error'})
        for s in grp:
            if nested:
                s = dict(s)
                s['phase_index'], s['phase_label'] = k, label
            flat.append(s)
        phase_recs.append({'phase_index': k, 'label': label,
                           'n_sources': len(grp), 'mesh_idx': idxs_k,
                           'n_unique_mesh_idx': len(set(idxs_k)),
                           'duplicate_mesh_idx': dup_k,
                           'source_labels': [s.get('label') for s in grp]})
    all_idx = [s['mesh_idx'] for s in flat]
    shared = sorted({i for i in set(all_idx)
                     if sum(1 for p in phase_recs if i in p['mesh_idx']) > 1})
    return {
        _BUILDER: 'source_protocol', 'application': application,
        'solver_class': solver_class, 'placement_rule': placement_rule,
        'time_level': time_level, 'sources': flat,
        'n_sources': len(flat), 'n_unique_mesh_idx': len(set(all_idx)),
        'duplicate_mesh_idx': sorted(set(dup_all)),
        'multi_phase': nested, 'n_phases': len(phases), 'phases': phase_recs,
        'shared_mesh_idx_across_phases': shared,
        'targets': targets, 'phase_chaining': phase_chaining,
        'boundary_conditions': boundary_conditions,
        '_discrepancies': disc,
    }


# ---------------------------------------------------------------------------
# Group 2: numerics
# ---------------------------------------------------------------------------

def count_rejected(history_lines):
    """Rejected adaptive steps, parsed from fibeRIS's log (tso.py:32).

    Returns None, never 0, for an empty/unavailable history: "unknown" recorded as
    "zero" would make a manifest claim a perfectly efficient adaptive run.
    """
    if not history_lines:
        return None
    n = 0
    for line in history_lines:
        if 'Dynamic time sampling rejected' in str(line):
            n += 1
    return int(n)


def time_record(taxis, *, mode, theta, t_total_requested_s, dt_requested_s=None,
                dt_init_s=None, tol=None, controller_tol=None, max_dt_s=None,
                min_dt_s=None, safety_factor=None, order_p=None,
                n_steps_rejected=None, source_time_level=None,
                theta_startup_steps=0, zero_field_policy=None,
                flip_margin=None, label=None):
    """Realised stepping for one solve, derived from the taxis it produced."""
    t = np.asarray(taxis, dtype=float)
    if t.size < 2:
        raise ManifestIncomplete("time_record needs a taxis with >= 2 entries")
    if mode not in ('fixed', 'adaptive'):
        raise ValueError("mode must be 'fixed' or 'adaptive'")
    dts = np.diff(t)
    disc = []
    if mode == 'fixed':
        if dt_requested_s is None:
            raise ManifestIncomplete("mode='fixed' requires dt_requested_s")
        if abs(float(dts.max()) - float(dt_requested_s)) > 1e-9:
            disc.append({'kind': 'dt_requested_vs_realised',
                         'detail': f"dt_requested_s={dt_requested_s} but realised "
                                   f"max dt is {float(dts.max())}",
                         'severity': 'warn'})
    else:
        missing = [k for k, v in (('tol', tol), ('max_dt_s', max_dt_s),
                                  ('min_dt_s', min_dt_s), ('dt_init_s', dt_init_s),
                                  ('safety_factor', safety_factor),
                                  ('order_p', order_p),
                                  ('n_steps_rejected', n_steps_rejected))
                   if v is None]
        if missing:
            raise ManifestIncomplete(
                "mode='adaptive' requires " + ", ".join(missing)
                + " (a reviewer will not accept 'the optimizer guaranteed "
                  "convergence'; pass count_rejected(pds.history) for a "
                  "fibeRIS-driven run)")
    return {
        _BUILDER: 'time_record', 'label': label, 'mode': mode,
        'theta': float(theta), 'source_time_level': source_time_level,
        'theta_startup_steps': int(theta_startup_steps),
        't_total_requested_s': float(t_total_requested_s),
        'dt_requested_s': None if dt_requested_s is None else float(dt_requested_s),
        'dt_init_s': None if dt_init_s is None else float(dt_init_s),
        'tol': None if tol is None else float(tol),
        # controller_tol is separate on purpose: fibeRIS's tso.py never forwards
        # `tol` to adjust_dt (tso.py:4, :21, :40), so the controller always runs
        # at its own 1e-3 no matter what the caller asked for.
        'controller_tol': None if controller_tol is None else float(controller_tol),
        'max_dt_s': None if max_dt_s is None else float(max_dt_s),
        'min_dt_s': None if min_dt_s is None else float(min_dt_s),
        'safety_factor': None if safety_factor is None else float(safety_factor),
        'order_p': None if order_p is None else int(order_p),
        'zero_field_policy': zero_field_policy, 'flip_margin': flip_margin,
        'n_steps_accepted': int(t.size - 1),
        'n_steps_rejected': (None if n_steps_rejected is None
                             else int(n_steps_rejected)),
        'n_steps_total': (None if n_steps_rejected is None
                          else int(t.size - 1 + int(n_steps_rejected))),
        'dt_realised_s': {'min': float(dts.min()), 'max': float(dts.max()),
                          'mean': float(dts.mean()),
                          'median': float(np.median(dts))},
        'uniform_dt': bool(float(np.ptp(dts)) < 1e-12),
        't0_s': float(t[0]), 't_end_realised_s': float(t[-1]),
        # non-zero by construction: `while taxis[-1] < t_total` (pds.py:287)
        'overshoot_s': float(t[-1] - float(t_total_requested_s)),
        'taxis_sha256': sha256_array(t),
        'error_norm': ('relative_l2_full_vs_two_half_steps' if mode == 'adaptive'
                       else None),
        '_discrepancies': disc,
    }


def mesh_record(mesh, *, dx_requested_ft, window_md_ft, pad_low_ft, pad_high_ft,
                refinement=None):
    """The mesh, pinned by hash.

    nx plus [md_min, md_max] does NOT determine a refined mesh -- 101 turns 5500
    uniform nodes into 5656 with dx spanning 0.133-1.0 ft -- so mesh_sha256 is
    mandatory rather than decorative.
    """
    m = np.asarray(mesh, dtype=float)
    dx = np.diff(m)
    lo, hi = float(window_md_ft[0]), float(window_md_ft[1])
    pad_lo_real = lo - float(m[0])
    disc = []
    if pad_lo_real < 5000.0:
        disc.append({
            'kind': 'low_end_padding',
            'detail': f"realised low-end pad is {pad_lo_real:.1f} ft; the house "
                      f"rules require >= 5000 ft because g7 (MD 15075) sits 75 ft "
                      f"from the no-flux boundary and reflection nearly doubles "
                      f"its simulated peak",
            'severity': 'warn'})
    return {
        _BUILDER: 'mesh_record', 'nx': int(m.size),
        'md_min_ft': float(m[0]), 'md_max_ft': float(m[-1]),
        'dx_requested_ft': float(dx_requested_ft),
        'dx_min_ft': float(dx.min()), 'dx_max_ft': float(dx.max()),
        'dx_median_ft': float(np.median(dx)),
        'uniform': bool(float(np.ptp(dx)) < 1e-9),
        'mesh_sha256': sha256_array(m),
        'window_md_ft': [lo, hi],
        'pad_low_ft': float(pad_low_ft), 'pad_high_ft': float(pad_high_ft),
        'pad_low_realised_ft': float(pad_lo_real),
        'pad_high_realised_ft': float(float(m[-1]) - hi),
        'refinement': refinement if refinement is not None else NONE_DECLARED,
        '_discrepancies': disc,
    }


def barrier_record(mesh, mask, *, label, centre_md_ft, w_requested_ft, ratio,
                   d_baseline, report=None):
    """One barrier, derived from the boolean node mask the run ACTUALLY applied.

    Two widths are reported and both are named, because reporting one without
    saying which convention it uses is how the 900x effective-width spread arose:

    * `w_realised_node_ft` -- half the span of the reduced nodes themselves;
    * `w_realised_control_volume_ft` -- half of
      (mesh[i1]-mesh[i0]) + dx_left(i0)/2 + dx_right(i1)/2, the sum of the
      captured nodes' control volumes. This is the PHYSICAL width: with harmonic
      face averaging a single reduced node is exactly a slab spanning its own
      control volume, so this is the number that is comparable across meshes and
      the number the paper should quote.

    Pass rev2_core.build_barrier_profile's report as `report` to store its realised
    extent verbatim.
    """
    m = np.asarray(mesh, dtype=float)
    msk = np.asarray(mask, dtype=bool)
    if msk.shape != m.shape:
        raise ValueError(f"mask shape {msk.shape} != mesh shape {m.shape}")
    idx = np.where(msk)[0]
    if idx.size == 0:
        # Verified failure mode: w_requested = 0.0 on a refined mesh silently
        # yields an empty mask and the barrier disappears with no error.
        raise ManifestIncomplete(f"barrier '{label}' selected no nodes")
    i0, i1 = int(idx[0]), int(idx[-1])
    nx = m.size
    dxl = float(m[i0] - m[i0 - 1]) if i0 > 0 else 0.0
    dxr = float(m[i1 + 1] - m[i1]) if i1 < nx - 1 else 0.0
    full_cv = float(m[i1] - m[i0]) + dxl / 2.0 + dxr / 2.0
    dx_local = float(np.median(np.diff(m[i0:i1 + 1]))) if i1 > i0 \
        else float(max(dxl, dxr))
    w_node = float(m[i1] - m[i0]) / 2.0
    disc = []
    if abs(w_node - float(w_requested_ft)) > 0.5 * dx_local:
        disc.append({'kind': 'barrier_width_requested_vs_realised',
                     'detail': f"barrier '{label}': w_requested={w_requested_ft} ft, "
                               f"w_realised_node={w_node:.4f} ft, local dx "
                               f"{dx_local:.4f} ft",
                     'severity': 'warn'})
    return {
        _BUILDER: 'barrier_record', 'label': label,
        'centre_md_ft': float(centre_md_ft),
        'w_requested_ft': float(w_requested_ft),
        'full_width_requested_ft': 2.0 * float(w_requested_ft),
        'ratio': float(ratio), 'd_baseline_ft2_s': float(d_baseline),
        'D_barrier_ft2_s': float(ratio) * float(d_baseline),
        'n_nodes': int(idx.size), 'i0': i0, 'i1': i1,
        'span_md_ft': [float(m[i0]), float(m[i1])],
        'w_realised_node_ft': w_node,
        'w_realised_control_volume_ft': full_cv / 2.0,
        'full_width_realised_control_volume_ft': full_cv,
        'control_volume_span_md_ft': [float(m[i0]) - dxl / 2.0,
                                      float(m[i1]) + dxr / 2.0],
        'dx_local_ft': dx_local, 'dx_left_ft': dxl, 'dx_right_ft': dxr,
        'barrier_report': report,
        '_discrepancies': disc,
    }


def numerics(*, time, mesh, interface_avg, boundary, diffusivity, barriers,
             leakage, kernel, rng, parallel, units=UNITS, amplification=None):
    """Everything about how the numbers were produced. All arguments required."""
    times = time if isinstance(time, (list, tuple)) else [time]
    for t in times:
        if not isinstance(t, dict) or t.get(_BUILDER) != 'time_record':
            raise ManifestIncomplete("time must be a time_record (or a list of them)")
    if not isinstance(mesh, dict) or mesh.get(_BUILDER) != 'mesh_record':
        raise ManifestIncomplete("mesh must be built by mesh_record()")
    if interface_avg not in ('harmonic', 'arithmetic'):
        raise ValueError("interface_avg must be 'harmonic' or 'arithmetic'")
    for name, val in (('boundary', boundary), ('diffusivity', diffusivity),
                      ('kernel', kernel)):
        if not isinstance(val, dict):
            raise ManifestIncomplete(f"{name} must be a dict")
    for key in ('lbc', 'rbc', 'pml_thickness', 'sigma_max'):
        if key not in boundary:
            # matbuilder adds dt*sigma to the diagonal AFTER writing the Dirichlet
            # row (:76-78, :164-165), so a non-zero PML scales the prescribed
            # source value; recording only "Dirichlet" would make a corrupted run
            # look identical to a clean one.
            raise ManifestIncomplete(f"boundary is missing '{key}'")
    if barriers is not NONE_DECLARED:
        blist = list(barriers)
        if not blist:
            raise ManifestIncomplete(
                "barriers=[] is not accepted: a run with no barrier must say so "
                "with NONE_DECLARED, so 'there is none' and 'I forgot' stay "
                "distinguishable")
        for b in blist:
            if not isinstance(b, dict) or b.get(_BUILDER) != 'barrier_record':
                raise ManifestIncomplete("every barrier must come from "
                                         "barrier_record()")
    else:
        blist = NONE_DECLARED

    disc = []
    for t in times:
        disc.extend(t.pop('_discrepancies', []))
    disc.extend(mesh.pop('_discrepancies', []))
    if blist is not NONE_DECLARED:
        for b in blist:
            disc.extend(b.pop('_discrepancies', []))

    return {
        _BUILDER: 'numerics', 'time': times, 'mesh': mesh,
        'interface_avg': interface_avg, 'boundary': boundary,
        'diffusivity': diffusivity, 'barriers': blist, 'leakage': leakage,
        'kernel': kernel, 'rng': rng, 'parallel': parallel, 'units': units,
        'amplification': amplification if amplification is not None
                         else NONE_DECLARED,
        '_discrepancies': disc,
    }


# ---------------------------------------------------------------------------
# Group 3: outputs
# ---------------------------------------------------------------------------

def output_decl(path, *, role, dpi=None, note=None):
    """DECLARE an output. Hashing happens inside write_manifest, after it exists."""
    if role not in _OUTPUT_ROLES:
        raise ValueError(f"role must be one of {_OUTPUT_ROLES}, got {role!r}")
    return {_BUILDER: 'output_decl', 'path': path, 'role': role,
            'dpi': dpi, 'note': note}


# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------

_REQUIRED_TOP = ('schema_version', 'round_tag', 'study_id', 'task_id', 'run_utc',
                 'config', 'environment', 'code', 'inputs', 'source_protocol',
                 'numerics', 'outputs')
_REQUIRED_NUMERICS = ('time', 'mesh', 'interface_avg', 'boundary', 'diffusivity',
                      'barriers', 'leakage', 'kernel', 'rng', 'parallel')
_REQUIRED_SOURCE = ('application', 'solver_class', 'placement_rule', 'sources',
                    'targets', 'time_level', 'phase_chaining',
                    'boundary_conditions')


def _require_fields(doc):
    """Collect EVERY missing dotted path at once, not just the first."""
    missing = []
    for k in _REQUIRED_TOP:
        if k not in doc or doc[k] is None:
            missing.append(k)
    sp = doc.get('source_protocol') or {}
    for k in _REQUIRED_SOURCE:
        if k not in sp:
            missing.append(f"source_protocol.{k}")
    nm = doc.get('numerics') or {}
    for k in _REQUIRED_NUMERICS:
        if k not in nm:
            missing.append(f"numerics.{k}")
    if missing:
        raise ManifestIncomplete("manifest is missing required field(s): "
                                 + ", ".join(missing))


def _sweep_config_for_files(cfg, declared_abs):
    """Any config string that resolves to an existing file becomes an input.

    Direct fix for the r2 regression, where zero input files were hashed.

    `declared_abs` is the realpath set to skip. The caller must put the run's own
    DECLARED OUTPUTS in it (write_manifest does): a config almost always names the
    files the run writes, and hashing one of those as an input records the
    post-run bytes of a product as though they were pre-run data.
    """
    found = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, str) and o and len(o) < 4096 and '{' not in o:
            if os.path.isfile(o) and os.path.realpath(o) not in declared_abs:
                found.append((path, o))

    walk(cfg, '')
    return found


def write_manifest(manifest_path, *, study_id, task_id, config, config_path,
                   inputs, source, numerics, outputs,
                   results=None, notes=None, started_utc=None, run_label=None,
                   require_modules=(), extra_code_files=(), worker_modules=None,
                   overwrite=False, allow_undeclared_outputs=False,
                   recursive_output_scan=False):
    """Assemble and write the manifest. Every group is a required argument.

    Enforcement in layers, because required arguments alone are not enough:
      L1 the eight group arguments are keyword-only with no defaults, so omitting
         one is a TypeError at the call site before any hashing;
      L2 source/numerics/outputs must come from their builders, each of which has
         all-required arguments of its own, so a half-filled group cannot be
         constructed; a hand-rolled bare dict is rejected by the builder tag;
      L3 `_require_fields` walks the assembled document and lists every missing
         dotted path at once;
      L4 NONE_DECLARED is the only way to say "not applicable" -- None and
         omission both fail -- and every such assertion is listed under
         `explicitly_none` so an auditor sees the author claimed absence.

    Output files are hashed HERE, after they exist, and the manifest's directory is
    scanned for undeclared files, so a run that crashed before writing its figure
    cannot emit a clean manifest.

    `worker_modules` closes the code-closure escape hatch that `code_closure`'s
    docstring only warned about: the closure is a scan of the PARENT's sys.modules,
    so a repo module first imported inside a `multiprocessing` child is never
    hashed, and rewriting a constant in it (an independent reviewer used exactly
    the 8.94e-9 -> 2.298e-7 psi->strain error the house rules warn about, a factor
    26) leaves `verify()` reporting clean. Pass the names of every repo module the
    workers import -- they are appended to `require_modules`, so an import the
    parent never made is a ManifestIncomplete rather than a silent gap -- or pass
    `worker_modules=()` as the explicit assertion "no repo module is imported only
    inside a worker". If a run's `numerics['parallel']` says child processes were
    used and NONE of `worker_modules` / `require_modules` / `extra_code_files` was
    given, the manifest records a `worker_code_closure_undeclared` discrepancy and
    `verify()` scores it `incomplete`. Backward compatibility is the reason this
    is a discrepancy and not a raise: a completed task must stay re-runnable.
    """
    root = repo_root()
    manifest_path = os.path.abspath(manifest_path)
    if not overwrite:
        assert_absent([manifest_path])

    if not isinstance(source, dict) or source.get(_BUILDER) != 'source_protocol':
        raise ManifestIncomplete("source= must be built by source_protocol()")
    if not isinstance(numerics, dict) or numerics.get(_BUILDER) != 'numerics':
        raise ManifestIncomplete("numerics= must be built by numerics()")
    out_decls = list(outputs)
    for o in out_decls:
        if not isinstance(o, dict) or o.get(_BUILDER) != 'output_decl':
            raise ManifestIncomplete("every outputs= entry must come from "
                                     "output_decl()")

    # ---- inputs -----------------------------------------------------------
    # A config almost always names the files the run WRITES. Those realpaths are
    # collected FIRST so the auto-scan below can never hash a declared output as
    # an input: doing so records the post-run bytes of a product as pre-run data
    # and makes the provenance record circular (found in A1's manifests).
    self_abs = {os.path.realpath(manifest_path),
                os.path.realpath(manifest_path + '.sha256')}
    output_abs = {os.path.realpath(o['path']) for o in out_decls}
    in_recs = {}
    declared_abs = set()
    for item in inputs:
        if isinstance(item, dict):
            rec = item
        else:
            path, role = item[0], item[1]
            logical = item[2] if len(item) > 2 else None
            rec = file_record(path, role=role, logical_name=logical)
        in_recs[rec['logical_name']] = rec
        declared_abs.add(os.path.realpath(rec['abspath']))
    if config_path and os.path.isfile(config_path):
        declared_abs.add(os.path.realpath(config_path))
    input_is_output = sorted(
        k for k, v in in_recs.items()
        if os.path.realpath(v['abspath']) in output_abs)
    auto_excluded = []
    post_run_inputs = []
    t_started = _parse_utc(started_utc)
    t_basis = 'started_utc'
    if t_started is None:
        t_started, t_basis = _IMPORT_TIME, 'rev2_manifest import time'
    for dotted, path in _sweep_config_for_files(config, declared_abs):
        rp = os.path.realpath(path)
        if rp in output_abs or rp in self_abs:
            auto_excluded.append(
                {'config_path': dotted, 'path': _rel(path, root),
                 'reason': ('declared_output' if rp in output_abs
                            else 'manifest_itself')})
            continue
        post_run = (t_started is not None and os.path.getmtime(path)
                    >= t_started - _RUN_PRODUCT_MTIME_TOL_S)
        if post_run and _is_output_key(dotted):
            # An UNDECLARED product the run wrote is not input provenance: its
            # bytes are post-run, so it drifts on the next append and shows up in
            # an audit as data drift (this was the whole of D2's 8 drift rows,
            # every one of them a log file). TWO signals are required before a
            # file is dropped from the scan -- the mtime evidence AND a dotted key
            # under an 'output'/'outputs' subtree -- because a run that DERIVES an
            # input in-process still deserves that input's hash. The file is
            # recorded here, with its reason, never silently dropped.
            auto_excluded.append(
                {'config_path': dotted, 'path': _rel(path, root),
                 'reason': 'written_during_run', 'basis': t_basis,
                 'mtime_utc': _mtime_utc(path), 'started_utc': str(started_utc)})
            continue
        if post_run:
            post_run_inputs.append((dotted, _rel(path, root), _mtime_utc(path)))
        rec = file_record(path, role='auto_from_config',
                          logical_name=f"auto:{dotted}")
        in_recs[rec['logical_name']] = rec
        declared_abs.add(os.path.realpath(rec['abspath']))

    # ---- config hash ------------------------------------------------------
    if config_path:
        # Raw FILE BYTES, exactly as load_config does (r1_run_calibration.py:28-31),
        # so r1/r2 hashes stay comparable. It therefore changes with whitespace.
        cfg_hash = sha256_file(config_path)
        cfg_mode = 'file_bytes'
    else:
        # PLACEHOLDER. The real hash is taken further down from the JSONIFIED
        # payload, because that is what verify() re-hashes. Hashing the raw object
        # here (as this did until 2026-09-02) made a manifest report config DRIFT
        # with nothing changed on disk for np.int64, NaN, any ndarray, a numeric
        # list longer than max_array_len, and the NONE_DECLARED sentinel -- all
        # five of which _jsonify rewrites on the way out.
        cfg_hash = None
        cfg_mode = 'canonical_json'

    now = datetime.datetime.now(datetime.timezone.utc)
    disc = list(source.pop('_discrepancies', [])) + \
        list(numerics.pop('_discrepancies', []))
    for k in input_is_output:
        # Not silently dropped: an explicit declaration is the author's word, and
        # the contradiction has to be visible rather than resolved by the writer.
        disc.append({'kind': 'input_is_declared_output',
                     'detail': f"'{k}' is declared BOTH as an input and as an "
                               f"output of this run; its recorded hash is the "
                               f"post-run content, so it is not input provenance",
                     'severity': 'error'})

    # ---- outputs: they must exist NOW -------------------------------------
    out_files = []
    for o in out_decls:
        p = o['path']
        if not os.path.exists(p):
            raise OutputMissing(
                f"declared output does not exist: {os.path.abspath(p)}. Hashes "
                f"are computed after writing, so a run that died before emitting "
                f"this file cannot produce a clean manifest.")
        rec = {'path': _rel(p, root), 'abspath': os.path.realpath(p),
               'role': o['role'], 'size_bytes': int(os.path.getsize(p)),
               'sha256': sha256_file(p, use_cache=False),
               'mtime_utc': _mtime_utc(p), 'dpi': o['dpi'], 'note': o['note']}
        if o['role'].startswith('figure') and (o['dpi'] is None or o['dpi'] < 300):
            disc.append({'kind': 'figure_dpi',
                         'detail': f"{rec['path']} declared dpi={o['dpi']}; house "
                                   f"rule 9 requires >= 300",
                         'severity': 'warn'})
        out_files.append(rec)

    out_dir = os.path.dirname(manifest_path)
    known = {r['abspath'] for r in out_files} | {manifest_path,
                                                 manifest_path + '.sha256'}
    pattern = os.path.join(out_dir, '**', '*') if recursive_output_scan \
        else os.path.join(out_dir, '*')
    undeclared = sorted(
        _rel(p, root) for p in _glob.glob(pattern, recursive=recursive_output_scan)
        if os.path.isfile(p) and os.path.realpath(p) not in known
        and not p.endswith('.FAILED.json'))
    if undeclared and not allow_undeclared_outputs:
        raise ManifestIncomplete(
            "undeclared files in the output directory: " + ", ".join(undeclared)
            + ". Declare them with output_decl(), or pass "
              "allow_undeclared_outputs=True to record them as an exception.")
    if undeclared:
        disc.append({'kind': 'undeclared_outputs',
                     'detail': f"{len(undeclared)} undeclared file(s) in "
                               f"{_rel(out_dir, root)}",
                     'severity': 'warn'})

    for _dotted, _p, _mt in post_run_inputs:
        disc.append({
            'kind': 'config_input_written_during_run',
            'detail': (f"config path '{_dotted}' names {_p}, whose mtime {_mt} "
                       f"postdates this run's start (basis: {t_basis}). It IS "
                       f"hashed as an input, but the recorded bytes are post-run, "
                       f"so they are not pre-run provenance and will drift if the "
                       f"file is written again."),
            'severity': 'warn'})
    for _e in auto_excluded:
        if _e['reason'] == 'written_during_run':
            disc.append({
                'kind': 'undeclared_product_in_config',
                'detail': (f"config path '{_e['config_path']}' names "
                           f"{_e['path']}, whose mtime {_e['mtime_utc']} "
                           f"postdates this run's start (basis: {_e['basis']}, "
                           f"started_utc {_e['started_utc']}), so the run WROTE "
                           f"it. It was NOT hashed as an input, because its bytes "
                           f"are post-run. Declare it with output_decl() if it is "
                           f"a product of this run."),
                'severity': 'warn'})

    worker_names = tuple(worker_modules) if worker_modules is not None else ()
    if _uses_workers(numerics.get('parallel')) and worker_modules is None \
            and not require_modules and not extra_code_files:
        msg = ("numerics['parallel'] says this run used worker processes, but the "
               "code closure is a scan of the PARENT's sys.modules only. A repo "
               "module first imported inside a worker is NOT hashed, so editing it "
               "leaves verify() clean. Declare worker_modules=(...) (or "
               "worker_modules=() to assert there are none), require_modules=, or "
               "extra_code_files=.")
        disc.append({'kind': 'worker_code_closure_undeclared', 'detail': msg,
                     'severity': 'error'})
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    fiberis_path = None
    fib = sys.modules.get('fiberis')
    if fib is not None:
        fiberis_path = getattr(fib, '__file__', None)

    doc = {
        'schema_version': SCHEMA_VERSION, 'round_tag': ROUND_TAG,
        'study_id': study_id, 'task_id': task_id, 'run_label': run_label,
        'started_utc': started_utc, 'run_utc': now.isoformat(),
        'wall_seconds': None, 'hostname': socket.gethostname(),
        'config': {'path': (os.path.abspath(config_path) if config_path else None),
                   'sha256': cfg_hash, 'sha256_mode': cfg_mode,
                   'resolved': config},
        'environment': {
            'python': sys.version.split()[0],
            'python_executable': sys.executable,
            'platform': platform.platform(), 'hostname': socket.gethostname(),
            'cwd': os.getcwd(), 'repo_root': root,
            'numpy': np.__version__,
            'scipy': getattr(sys.modules.get('scipy'), '__version__', None),
            'matplotlib': getattr(sys.modules.get('matplotlib'), '__version__',
                                  None),
            'fiberis_path': fiberis_path,
            'fiberis_inside_repo': bool(fiberis_path
                                        and os.path.realpath(fiberis_path)
                                        .startswith(root + os.sep)),
            'env': {k: os.environ.get(k) for k in
                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'PYTHONHASHSEED')},
            'note': ('bakken_mariner/.git is empty; no commit hash is recoverable, '
                     'so code identity is pinned by the sha256 values in `code`.'),
        },
        'code': dict(code_closure(require=tuple(require_modules) + worker_names,
                                  extra=extra_code_files, root=root),
                     worker_modules_declared=(None if worker_modules is None
                                              else list(worker_names))),
        'inputs': in_recs,
        'source_protocol': source,
        'numerics': numerics,
        'outputs': {'files': out_files,
                    'undeclared_files_in_output_dir': undeclared,
                    'config_paths_excluded_from_input_scan': auto_excluded,
                    'manifest_path': _rel(manifest_path, root),
                    'self_sha256_sidecar': _rel(manifest_path + '.sha256', root)},
        'results': results if results is not None else {},
        'notes': list(notes or []),
        'discrepancies': disc,
    }
    if started_utc:
        try:
            t0 = datetime.datetime.fromisoformat(str(started_utc))
            doc['wall_seconds'] = (now - t0).total_seconds()
        except Exception:
            pass

    _require_fields(doc)
    payload, none_paths, nonfinite = _jsonify(doc)
    if cfg_mode == 'canonical_json':
        # Writer and verifier now hash the SAME BYTES by construction: this is the
        # object that is about to be written to disk and that verify() reads back.
        blob = json.dumps(payload['config']['resolved'], sort_keys=True,
                          separators=(',', ':'), default=str).encode('utf-8')
        payload['config']['sha256'] = hashlib.sha256(blob).hexdigest()
    payload['explicitly_none'] = sorted(none_paths)
    payload['nonfinite_fields'] = sorted(nonfinite)
    for k in list(payload.get('source_protocol', {})):
        if k == _BUILDER:
            payload['source_protocol'].pop(k)
    for k in list(payload.get('numerics', {})):
        if k == _BUILDER:
            payload['numerics'].pop(k)

    os.makedirs(out_dir or '.', exist_ok=True)
    tmp = manifest_path + f'.tmp{os.getpid()}'
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp, manifest_path)
    # The manifest cannot hash itself; the sidecar closes that loop.
    with open(manifest_path + '.sha256', 'w') as fh:
        fh.write(f"{sha256_file(manifest_path, use_cache=False)}  "
                 f"{os.path.basename(manifest_path)}\n")
    return payload


class RunRecorder:
    """Context manager that guarantees a manifest is written at all.

    Required arguments stop a caller forgetting a group, but not a run that writes
    its outputs and then never calls the writer. On an exception this dumps
    `<manifest>.FAILED.json` with everything collected so far plus the traceback,
    and re-raises.
    """

    def __init__(self, manifest_path, *, study_id, task_id, config,
                 config_path=None, **kw):
        self.manifest_path = os.path.abspath(manifest_path)
        self.kw = dict(study_id=study_id, task_id=task_id, config=config,
                       config_path=config_path, **kw)
        self.inputs = []
        self.outputs = []
        self.source = None
        self.numerics = None
        self.results = None
        self.notes = []
        self.started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.manifest = None

    def declare_inputs(self, items):
        self.inputs.extend(items)
        return self

    def declare_output(self, path, *, role, dpi=None, note=None):
        self.outputs.append(output_decl(path, role=role, dpi=dpi, note=note))
        return self

    def set_source(self, group):
        self.source = group
        return self

    def set_numerics(self, group):
        self.numerics = group
        return self

    def set_results(self, results):
        self.results = results
        return self

    def note(self, text):
        self.notes.append(text)
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            import traceback
            failed = {'schema_version': SCHEMA_VERSION, 'status': 'FAILED',
                      'study_id': self.kw.get('study_id'),
                      'task_id': self.kw.get('task_id'),
                      'started_utc': self.started_utc,
                      'failed_utc': datetime.datetime.now(
                          datetime.timezone.utc).isoformat(),
                      'traceback': ''.join(
                          traceback.format_exception(exc_type, exc, tb)),
                      'notes': self.notes}
            try:
                os.makedirs(os.path.dirname(self.manifest_path) or '.',
                            exist_ok=True)
                with open(self.manifest_path + '.FAILED.json', 'w') as fh:
                    json.dump(_jsonify(failed)[0], fh, indent=2, sort_keys=True)
            except Exception:
                pass
            return False
        for name, val in (('source', self.source), ('numerics', self.numerics)):
            if val is None:
                raise ManifestIncomplete(f"RunRecorder: {name} group was never set")
        self.manifest = write_manifest(
            self.manifest_path, inputs=self.inputs, source=self.source,
            numerics=self.numerics, outputs=self.outputs, results=self.results,
            notes=self.notes, started_utc=self.started_utc, **self.kw)
        return False


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------

def _legacy_adapt(doc, path):
    """Map an r1/r2 manifest onto the rev2 shape so verify() can read it."""
    env = doc.get('environment', {}) or {}
    code = {}
    for name, sha in (env.get('code_sha256') or {}).items():
        code[name] = sha
    inputs = {}
    for k, v in (env.get('input_data_sha256') or {}).items():
        p = ((doc.get('config_resolved', {}) or {}).get('data', {}) or {}).get(k)
        inputs[k] = {'path': p, 'sha256': v, 'role': 'geometry'}
    tmpl = ((doc.get('config_resolved', {}) or {}).get('data', {}) or {}).get(
        'gauge_series_template')
    for n, v in (env.get('gauge_series_sha256') or {}).items():
        inputs[f"gauge_series_{n}"] = {
            'path': tmpl.format(n=n) if tmpl else None,
            'sha256': v, 'role': 'gauge_series'}
    # r1 hashes only the runner and the core; resolve them by convention.
    base = os.path.join(repo_root(), 'scripts', 'manuscript_well_leakage',
                        'baseline_calibration')
    guess = {'core': os.path.join(base, 'r1_calibration_core.py')}
    sid = doc.get('study_id', '')
    guess['runner'] = os.path.join(
        base, 'r2_profile_inversion.py' if sid.startswith('r2')
        else 'r1_run_calibration.py')
    files = {}
    for name, sha in code.items():
        p = guess.get(name)
        files[name] = {'sha256': sha, 'path': p}
    return {
        'schema_version': 'legacy/r1r2', 'study_id': sid,
        'task_id': None, 'run_utc': doc.get('run_utc'),
        'config': {'path': doc.get('config_path'),
                   'sha256': doc.get('config_sha256'),
                   'sha256_mode': 'file_bytes',
                   'resolved': doc.get('config_resolved')},
        'environment': env, '_legacy_code': files, '_legacy_inputs': inputs,
        'inputs_recorded': bool(inputs),
    }


def _cmp_file(path, recorded_sha, memo=None):
    """`memo` is a per-AUDIT dict, not the on-disk cache.

    A round-level audit re-checks the same fibeRIS/core files from hundreds of
    manifests; hashing each once per audit run costs nothing in fidelity (the memo
    dies with the call and is keyed on size and mtime_ns, so a file rewritten
    mid-audit is re-hashed), while the on-disk cache is still never used here.
    """
    if not path:
        return {'status': 'unresolved', 'recorded': recorded_sha, 'actual': None}
    ap = path if os.path.isabs(path) else os.path.join(repo_root(), path)
    if not os.path.exists(ap):
        return {'status': 'missing', 'path': ap, 'recorded': recorded_sha}
    try:
        if memo is None:
            actual = sha256_file(ap, use_cache=False)
        else:
            rp = os.path.realpath(ap)
            st = os.stat(rp)
            key = f"{rp}|{st.st_size}|{st.st_mtime_ns}"
            actual = memo.get(key)
            if actual is None:
                actual = memo[key] = sha256_file(ap, use_cache=False)
    except OSError as exc:
        return {'status': 'unreadable', 'path': ap, 'detail': str(exc)}
    return {'status': 'ok' if actual == recorded_sha else 'drift', 'path': ap,
            'recorded': recorded_sha, 'actual': actual,
            'size_actual': int(os.path.getsize(ap))}


def verify(manifest_path, *, repo_root=None, check_outputs=True,
           check_inputs=True, strict=False, hash_memo=None):
    """Re-hash everything a manifest names and report drift.

    Always re-hashes with use_cache=False. Distinguishes "not recorded" from
    "recorded and matching": r2's manifest has no inputs section at all, and a
    checker that reported "inputs: 0 files, all ok" would present a manifest with
    zero data provenance as verified. That guard applies to BOTH schema branches
    since 2026-09-02; it used to live inside `if legacy:` only, and a rev2-schema
    manifest with an empty inputs dict was scored `clean` (reachable in practice
    because `_sweep_config_for_files` skips any string containing '{', so an
    r1-style `gauge_series_template` records nothing).

    THE MANIFEST'S OWN BYTES ARE PART OF THE VERDICT. `<manifest>.sha256` is the
    only thing that can catch a manifest edited after the fact -- the manifest
    cannot hash itself -- and until 2026-09-02 `manifest_self` was computed,
    printed, and then ignored by the status block: editing results.RMSE_psi,
    numerics.diffusivity.D_ft2_s and source_protocol.sources[].mesh_idx in a
    written manifest left status='clean', --strict silent and the CLI exit code 0,
    and deleting the sidecar hid even the printed line. A drifting sidecar is now
    `drift`; a MISSING sidecar on a rev2-schema document is at least `incomplete`.
    The whole round rests on "every number can be re-run from its manifest", which
    an undetectably editable manifest does not support.

    Statuses, worst first: unreadable > missing > drift > incomplete > clean.
    """
    with open(manifest_path) as fh:
        doc = json.load(fh)
    legacy = 'schema_version' not in doc
    adapted = _legacy_adapt(doc, manifest_path) if legacy else doc

    root_source = 'argument'
    if repo_root is not None:
        root = repo_root
    else:
        env = adapted.get('environment', {}) or {}
        root = env.get('repo_root')
        root_source = 'manifest.environment.repo_root'
        if not root or not os.path.isdir(root):
            root = env.get('cwd')
            root_source = 'manifest.environment.cwd'
        if not root or not os.path.isdir(root):
            root = os.getcwd()
            root_source = 'os.getcwd()'

    lines = []
    sections = {}

    def run_section(name, items):
        res = {'n': len(items), 'ok': [], 'drift': [], 'missing': [],
               'unresolved': [], 'unreadable': []}
        for label, path, sha in items:
            r = _cmp_file(path, sha, memo=hash_memo)
            r['logical_name'] = label
            key = {'ok': 'ok', 'drift': 'drift', 'missing': 'missing',
                   'unresolved': 'unresolved', 'unreadable': 'unreadable'}[r['status']]
            res[key].append(r if key != 'ok' else label)
            if key != 'ok':
                lines.append(f"{name}: {label} -> {key.upper()} ({path})")
        sections[name] = res
        return res

    if legacy:
        run_section('code', [(k, v.get('path'), v.get('sha256'))
                             for k, v in adapted['_legacy_code'].items()])
        if check_inputs:
            items = [(k, v.get('path'), v.get('sha256'))
                     for k, v in adapted['_legacy_inputs'].items()]
            if not items:
                sections['inputs'] = {'n': 0, 'not_recorded': True, 'ok': [],
                                      'drift': [], 'missing': []}
                lines.append("inputs: NOT RECORDED in this manifest "
                             "(no data provenance)")
            else:
                run_section('inputs', items)
        sections['outputs'] = {'n': 0, 'not_recorded': True, 'ok': [], 'drift': [],
                               'missing': []}
        missing_groups = [g for g in ('source_protocol', 'numerics', 'outputs')
                          if g not in doc or g == 'outputs']
        completeness = {'missing_groups': missing_groups,
                        'missing_fields': ['code closure (only 2 files hashed)']}
    else:
        run_section('code', [(p, os.path.join(root, p), v['sha256'])
                             for p, v in (doc.get('code', {}).get('files') or {}).items()])
        if check_inputs:
            items = [(k, v.get('abspath') or os.path.join(root, v['path']),
                      v['sha256'])
                     for k, v in (doc.get('inputs') or {}).items()]
            if not items:
                # Same guard as the legacy branch: "0 files, all ok" is exactly
                # the presentation this function exists to prevent.
                sections['inputs'] = {'n': 0, 'not_recorded': True, 'ok': [],
                                      'drift': [], 'missing': []}
                lines.append("inputs: NOT RECORDED in this manifest "
                             "(no data provenance)")
            else:
                run_section('inputs', items)
        if check_outputs:
            run_section('outputs',
                        [(o['path'], o.get('abspath') or os.path.join(root, o['path']),
                          o['sha256'])
                         for o in (doc.get('outputs', {}).get('files') or [])])
        completeness = {'missing_groups': [g for g in ('source_protocol', 'numerics',
                                                       'outputs') if g not in doc],
                        'missing_fields': []}

    cfg = adapted.get('config', {}) or {}
    if cfg.get('sha256_mode') == 'canonical_json':
        blob = json.dumps(cfg.get('resolved'), sort_keys=True,
                          separators=(',', ':'), default=str).encode('utf-8')
        actual = hashlib.sha256(blob).hexdigest()
        cfg_res = {'status': 'inline',
                   'match': actual == cfg.get('sha256'),
                   'recorded': cfg.get('sha256'), 'actual': actual}
    elif cfg.get('path'):
        cfg_res = _cmp_file(cfg['path'], cfg.get('sha256'), memo=hash_memo)
    else:
        cfg_res = {'status': 'not_recorded'}
    if cfg_res.get('status') == 'drift' or cfg_res.get('match') is False:
        lines.append(f"config: DRIFT ({cfg.get('path')})")

    env = adapted.get('environment', {}) or {}
    live = {'python': sys.version.split()[0], 'numpy': np.__version__,
            'scipy': getattr(sys.modules.get('scipy'), '__version__', None),
            'matplotlib': getattr(sys.modules.get('matplotlib'), '__version__',
                                  None)}
    env_cmp = {}
    for k, v in live.items():
        rec_v = env.get(k)
        env_cmp[k] = {'recorded': rec_v, 'actual': v,
                      'match': (rec_v is None or v is None or rec_v == v)}
        if not env_cmp[k]['match']:
            lines.append(f"environment: {k} recorded {rec_v}, running {v}")

    self_res = None
    sidecar = manifest_path + '.sha256'
    if os.path.exists(sidecar):
        with open(sidecar) as fh:
            recorded = fh.read().split()[0]
        actual = sha256_file(manifest_path, use_cache=False)
        self_res = {'status': 'ok' if actual == recorded else 'drift',
                    'recorded': recorded, 'actual': actual,
                    'sidecar': os.path.abspath(sidecar)}
        if self_res['status'] != 'ok':
            lines.append("manifest_self: DRIFT (the manifest file itself changed "
                         "after it was written; its numbers are not the ones the "
                         "run produced)")
    elif not legacy:
        # Deleting the sidecar must not be a way to silence the check. Legacy
        # (r1/r2 hand-rolled) documents never had one and are already scored
        # incomplete for their missing groups, so the rule is limited to
        # rev2-schema documents, every one of which is written with a sidecar.
        self_res = {'status': 'sidecar_missing', 'recorded': None,
                    'actual': sha256_file(manifest_path, use_cache=False),
                    'sidecar': os.path.abspath(sidecar)}
        lines.append("manifest_self: SIDECAR MISSING (this manifest cannot be "
                     "checked against its own bytes)")

    worker_gap = [d for d in (adapted.get('discrepancies') or [])
                  if isinstance(d, dict)
                  and d.get('kind') == 'worker_code_closure_undeclared']
    if worker_gap:
        lines.append("code: worker code closure UNDECLARED (a repo module "
                     "imported only inside a worker process is not hashed)")

    status = 'clean'
    if any(sections[s].get('missing') for s in sections):
        status = 'missing'
    if any(sections[s].get('drift') for s in sections) or \
            cfg_res.get('status') == 'drift' or cfg_res.get('match') is False or \
            (self_res is not None and self_res['status'] == 'drift'):
        status = 'drift'
    if completeness['missing_groups'] and status == 'clean':
        status = 'incomplete'
    if status == 'clean' and (
            sections.get('inputs', {}).get('not_recorded')
            or (self_res is not None
                and self_res['status'] == 'sidecar_missing')
            or worker_gap):
        status = 'incomplete'

    report = {
        'manifest': os.path.abspath(manifest_path),
        'schema_version': adapted.get('schema_version'),
        'study_id': adapted.get('study_id'), 'task_id': adapted.get('task_id'),
        'run_utc': adapted.get('run_utc'), 'repo_root_used': root,
        'root_source': root_source, 'config': cfg_res, 'environment': env_cmp,
        'code': sections.get('code', {}), 'inputs': sections.get('inputs', {}),
        'outputs': sections.get('outputs', {}), 'manifest_self': self_res,
        'worker_closure_undeclared': bool(worker_gap),
        'completeness': completeness, 'status': status, 'lines': lines,
    }
    if strict:
        if env.get('cwd') and env['cwd'] != os.getcwd():
            report['lines'].append(f"cwd recorded {env['cwd']}, running in "
                                   f"{os.getcwd()}")
        if report['status'] != 'clean' or \
                any(not v['match'] for v in env_cmp.values()):
            raise ManifestDrift(f"{manifest_path}: status={report['status']}; "
                                + "; ".join(report['lines'][:10]))
    return report


MANIFEST_NAME_GLOBS = ('manifest*.json', '*_manifest.json', '*.manifest.json')
_SNIFF_MAX_BYTES = 64 << 20


def _looks_like_manifest(path):
    """Content sniff for a manifest that no naming rule would catch.

    Keys are dumped sort_keys=True, so `schema_version` sits near the END of the
    file and a prefix read would miss it; the whole file is read, but only as
    bytes, with no JSON parse.
    """
    try:
        if os.path.getsize(path) > _SNIFF_MAX_BYTES:
            return False
        with open(path, 'rb') as fh:
            blob = fh.read()
    except OSError:
        return False
    if b'"schema_version"' in blob and b'"round_tag"' in blob:
        return True
    # r1/r2 legacy shape, which _legacy_adapt can still read.
    return b'"config_sha256"' in blob and b'"config_resolved"' in blob


def find_manifests(root=None, *, patterns=MANIFEST_NAME_GLOBS, recursive=True,
                   sniff=True):
    """Every manifest under `root`, however it is named and however deep.

    The original two globs (`<root>/*/manifest.json`, `<root>/manifest.json`)
    found 5 of the 293 manifests actually present in output/rev2_20260901 (272 of
    rev2 schema, 21 of the legacy r1/r2 shape): this round writes
    `manifest_v2.json`, `manifest_addendum.json`, `manifest_dx0p1.json` and, far
    more often, `<task>/<run>/manifest.json` one level deeper than the old glob
    reaches -- so a "round-level audit" silently audited 1.7% of the round.

    Discovery is by NAME first (patterns, recursively) and then, unless
    sniff=False, by CONTENT over every other .json in the tree, so a manifest
    called something else entirely is still audited. `.FAILED.json` sidecars are
    excluded here and reported separately by verify_all: they are the record of a
    run that never produced a manifest.
    """
    root = root or os.path.join(repo_root(), 'output', ROUND_TAG)
    hits = set()
    for pat in patterns:
        hits.update(_glob.glob(os.path.join(root, '**', pat), recursive=True)
                    if recursive else
                    _glob.glob(os.path.join(root, pat)))
    if sniff:
        for p in (_glob.glob(os.path.join(root, '**', '*.json'), recursive=True)
                  if recursive else _glob.glob(os.path.join(root, '*.json'))):
            if p not in hits and _looks_like_manifest(p):
                hits.add(p)
    keep = []
    for p in sorted(hits):
        base = os.path.basename(p)
        if not os.path.isfile(p):
            continue
        if base.endswith('.FAILED.json') or '.tmp' in base \
                or os.sep + '__pycache__' + os.sep in p:
            continue
        keep.append(p)
    return keep


def verify_all(root=None, *, patterns=MANIFEST_NAME_GLOBS, recursive=True,
               sniff=True, share_hashes=True, **kw):
    """Verify every manifest under `root` and roll the statuses up.

    Returns {manifest_path: report, '_rollup': {...}}. The rollup carries
    `n_found` -- how many manifests the discovery actually saw -- because the
    failure this replaces was an audit that reported on four manifests out of a
    round that held thirty-nine (D1), and would today report on five out of 293.
    A rollup without a denominator cannot be read as coverage.
    """
    root = root or os.path.join(repo_root(), 'output', ROUND_TAG)
    paths = find_manifests(root, patterns=patterns, recursive=recursive,
                           sniff=sniff)
    memo = {} if share_hashes else None
    out = {}
    roll = {'root': root, 'n_found': len(paths), 'n_clean': 0, 'n_drift': 0,
            'n_missing': 0, 'n_incomplete': 0, 'n_unreadable': 0, 'errors': [],
            'failed_runs': [], 'recursive': bool(recursive),
            'sniff': bool(sniff)}
    for p in paths:
        try:
            rep = verify(p, hash_memo=memo, **kw)
        except Exception as exc:
            # A manifest that cannot even be READ is its own status: folding it
            # into 'incomplete' would hide a corrupt file among honest ones.
            rep = {'status': 'unreadable', 'error': f"{type(exc).__name__}: {exc}"}
            roll['errors'].append({'manifest': p,
                                   'error': f"{type(exc).__name__}: {exc}"})
        out[p] = rep
        roll[f"n_{rep['status']}"] = roll.get(f"n_{rep['status']}", 0) + 1
    roll['failed_runs'] = sorted(
        _glob.glob(os.path.join(root, '**', '*.FAILED.json'), recursive=True)
        if recursive else _glob.glob(os.path.join(root, '*', '*.FAILED.json')))
    roll['n_failed_runs'] = len(roll['failed_runs'])
    roll['n_verified'] = sum(v for k, v in roll.items()
                             if k.startswith('n_') and k not in
                             ('n_found', 'n_failed_runs', 'n_verified'))
    out['_rollup'] = roll
    return out


def _main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ('verify', 'verify-all'):
        print(__doc__.strip().splitlines()[0])
        print("usage: rev2_manifest.py verify <manifest.json> [--strict] "
              "[--json]")
        print("       rev2_manifest.py verify-all [--root DIR] [--json] "
              "[--no-sniff]")
        return 2
    as_json = '--json' in argv
    strict = '--strict' in argv
    if argv[0] == 'verify':
        rest = [a for a in argv[1:] if not a.startswith('--')]
        if not rest:
            print("verify needs a manifest path")
            return 3
        rep = verify(rest[0], strict=strict)
        reps = {rest[0]: rep}
    else:
        root = None
        if '--root' in argv:
            root = argv[argv.index('--root') + 1]
        reps = verify_all(root=root, sniff='--no-sniff' not in argv)
        rep = None
        if not as_json:
            r = reps['_rollup']
            print(f"discovered {r['n_found']} manifest(s) under {r['root']} "
                  f"(recursive={r['recursive']}, sniff={r['sniff']})")
    if as_json:
        print(json.dumps(reps, indent=2, default=str))
    else:
        for k, v in reps.items():
            if k == '_rollup':
                print(f"rollup: {v}")
                continue
            print(f"{k}: {v.get('status')} "
                  f"(schema {v.get('schema_version')}, root {v.get('root_source')})")
            for line in v.get('lines', []):
                print(f"  {line}")
            c = v.get('completeness', {})
            if c.get('missing_groups'):
                print(f"  missing groups: {c['missing_groups']}")
    statuses = [v.get('status') for k, v in reps.items() if k != '_rollup']
    if 'unreadable' in statuses:
        return 4
    if 'missing' in statuses:
        return 2
    if 'drift' in statuses:
        return 1
    if 'incomplete' in statuses:
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(_main())
