"""Pressure-history inversion with the V11 MOOSE SRV model (fixed geometry), two approaches.

Approach A  (handoff request, Pengchao 2026-10-05)
    Invert a time-dependent injection-BC pressure dP(t) (2-hour knots, piecewise linear) so that
    MOOSE axial strain on the 44 DAS channels reproduces the DDM tensile strain profiles.
    Targets: inverted tensile (strain-fit, primary), baseline tensile, handoff rate-fit tensile.

Approach B  (joint)
    observed ~ a * MOOSE(dP_baseline) + b * DDM_shear_baseline
    two global scalars, linear least squares; fitted to observed strain, and separately to rate.

Forward operator (built from a single MOOSE ramp-response run, run_step_response.py):
    - model fiber = straight line perpendicular to the fracture at the monitoring well;
      MD = 10365.107 + y / 0.3048  (y = fault-normal model coordinate, m)
    - axial strain = strain_yy, 10-m gauge average (41-point boxcar over y_c +/- 5 m)
    - superposition of hat bases, evaluated at every valid native sample time and averaged over
      each 2-hour window (valid_observation_sample support from the handoff)
    - strain referenced to the first-window mean per channel; rate not referenced

    python invert_pressure.py            # uses every finished run under output/v11_pinv/
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
# small dense problems; unbounded BLAS threads fight the MOOSE runs (20 min -> 30 s)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "tensile_fault" / "inversion"))
import fault_geometry as fg  # noqa: E402

FT, PSI_MPA = 0.3048, 0.00689476
STAR = fg.STAR_1200
DP_RAMP = 1000.0                  # psi, ramp height used in run_step_response.py
P0_PSI = 4056.0                   # V11 prior initial pressure (IC), psi
WIN = 7200.0
T1 = pd.Timestamp("2025-02-24 12:00")
K_T2, K_T3 = 42, 114              # knot indices of T2 (+84 h) and T3 (+228 h)
GAUGE_M = 10.0

RUNS = REPO / "output" / "v11_pinv"
LEG = REPO / "data_fervo" / "legacy"
HAND = LEG / "handoff" / "Shenyao_MOOSE_Tensile_2026-10-05"
RES = RUNS / "results"
RES.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- data
def pivot(csv, s_col="strain_millistrain", r_col="strain_rate_nanostrain_s"):
    d = pd.read_csv(csv, parse_dates=["center_time"])
    s = d.pivot(index="md_ft", columns="center_time", values=s_col)
    r = d.pivot(index="md_ft", columns="center_time", values=r_col)
    return s.to_numpy(float), r.to_numpy(float), s.index.to_numpy(float), s.columns


Z = np.load(HAND / "data" / "tensile_only.npz")
MD = Z["depth_ft"]
t_native = (pd.to_datetime(Z["time_ns_fixed_utc_minus_7"]) - T1).total_seconds().to_numpy()
TV = t_native[Z["valid_observation_sample"]]
STARTS = np.searchsorted(TV, np.arange(114) * WIN)
assert np.all(np.diff(STARTS) > 0)
COUNTS = np.diff(np.append(STARTS, len(TV)))

TARGETS = {}
for key, stem in [("inverted", "inverted"), ("baseline", "baseline")]:
    s, r, md, cols = pivot(LEG / f"{stem}_tensile_profiles_2h.csv")
    ss, sr, _, _ = pivot(LEG / f"{stem}_shear_profiles_2h.csv")
    assert np.allclose(md, MD)
    TARGETS[key] = dict(strain=s, rate=r, shear_strain=ss, shear_rate=sr)
CENTERS = cols
TARGETS["handoff_ratefit"] = dict(strain=Z["tensile_strain_millistrain"], rate=Z["tensile_strain_rate_nanostrain_s"])
OBS_S, OBS_R, md, _ = pivot(LEG / "csv from DDM modeling" / "observed_profiles_2h.csv")
assert np.allclose(md, MD)
OPEN_KNOTS = pd.read_csv(HAND / "data" / "tensile_opening_knots_2h.csv")["tensile_opening_mm"].to_numpy()


# ----------------------------------------------------------------------------- MOOSE
def ramp_response(name):
    """Gauge-averaged axial strain (millistrain/psi) on the MD channels, at the MOOSE output times.

    Rates are taken as the derivative of this piecewise-linear history (= MOOSE's backward-difference
    rate, assigned to its own step interval), not from strain_rate_yy point values, which would lag
    by half a step."""
    d, proj = RUNS / name, f"v11_pinv_{name}"
    t = pd.read_csv(d / f"{proj}_input_csv.csv")["time"].to_numpy(float)
    yc = (MD - STAR) * FT
    pts = yc[:, None] + np.linspace(-GAUGE_M / 2, GAUGE_M / 2, 41)[None, :]

    def read(kind, col):
        files = sorted(d.glob(f"{proj}_input_csv_fiber_{kind}_sampler_*ft_*.csv"))
        n = min(len(files), len(t))
        out = []
        for f in files[:n]:
            dd = pd.read_csv(f).sort_values("y")
            if dd.empty:                          # t = 0 sampler is written empty: zero strain
                out.append(np.zeros(len(MD)))
                continue
            v = np.nan_to_num(dd[col].to_numpy(float))
            out.append(np.interp(pts, dd["y"].to_numpy(float), v).mean(axis=1))
        return np.column_stack(out), n

    strain, n = read("strain", "strain_yy")
    strain = (strain - strain[:, [0]]) / DP_RAMP * 1e3
    return t[:n], strain


def pressure_response(name):
    """Pore-pressure ramp response (per psi of BC) on the fiber line: at the fracture crossing (y = 0,
    ~1.2 m from the BC node) and averaged over the SRV outer zone (|y| <= 47.27 ft)."""
    d, proj = RUNS / name, f"v11_pinv_{name}"
    t = pd.read_csv(d / f"{proj}_input_csv.csv")["time"].to_numpy(float)
    files = sorted(d.glob(f"{proj}_input_csv_fiber_pressure_sampler_*ft_*.csv"))[:len(t)]
    p_ic = float(np.load(d / "ramp_pressure.npz")["data"][0])                # psi (IC of this run)
    loc, mean = [], []
    for f in files:
        dd = pd.read_csv(f).sort_values("y")
        if dd.empty:
            loc.append(0.0)
            mean.append(0.0)
            continue
        y, pp = dd["y"].to_numpy(float), dd["pp"].to_numpy(float) / 6894.757   # Pa -> psi
        m = np.abs(y) <= fg.HALF_1200 * FT
        loc.append(np.interp(0.0, y, pp) - p_ic)
        mean.append(pp[m].mean() - p_ic)
    return t[:len(files)], np.array(loc) / DP_RAMP, np.array(mean) / DP_RAMP


def superpose_at_knots(tg, r, x):
    """Response r(tau) (per psi, ramp) superposed for knot values x (1..114) at the knot times."""
    tk = np.arange(115) * WIN
    ramp = np.array([np.interp(tk - j * WIN, tg, r, left=0.0) for j in range(115)])   # [ramp start, knot time]
    hat = ramp[:-1] - ramp[1:]
    return x @ hat


def window_means(tg, R, shift, deriv=False):
    """Window means over valid native samples of R(t - shift) (or its time derivative, per s); 0 before the shift."""
    tau = TV - shift
    idx = np.clip(np.searchsorted(tg, tau, side="right") - 1, 0, len(tg) - 2)
    if deriv:
        val = (R[:, idx + 1] - R[:, idx]) / (tg[idx + 1] - tg[idx])
    else:
        w = np.clip((tau - tg[idx]) / (tg[idx + 1] - tg[idx]), 0.0, 1.0)
        val = R[:, idx] * (1 - w) + R[:, idx + 1] * w
    val[:, tau < 0] = 0.0
    return np.add.reduceat(val, STARTS, axis=1) / COUNTS          # [channel, window]


def basis(name):
    """G_strain, G_rate: [window, channel, knot] responses to unit (1 psi) hat functions, knots 1..114."""
    tg, Rs = ramp_response(name)
    assert tg[-1] >= TV[-1] - 1, f"{name}: run stops at {tg[-1]:.0f} s"
    As = np.stack([window_means(tg, Rs, j * WIN) for j in range(115)])   # [ramp start, c, w]
    Ar = np.stack([window_means(tg, Rs, j * WIN, deriv=True) for j in range(115)]) * 1e6  # nanostrain/s
    Hs, Hr = As[:-1] - As[1:], Ar[:-1] - Ar[1:]                          # [knot-1, c, w]
    Gs = np.transpose(Hs - Hs[:, :, [0]], (2, 1, 0))                      # first-window reference
    Gr = np.transpose(Hr, (2, 1, 0))
    return Gs, Gr, (tg, Rs)


# ----------------------------------------------------------------------------- inversion
NK = 114
L = np.zeros((NK, NK))                       # second differences of [0, x1..x114]
for i in range(NK):
    for j, c in zip((i - 1, i, i + 1), (1.0, -2.0, 1.0)):
        if 0 <= j < NK:
            L[i, j] = c
L = L[:-1]


def solve(G, d, lam):
    A = G.T @ G + lam ** 2 * L.T @ L
    x = np.linalg.solve(A, G.T @ d)
    Ainv = np.linalg.inv(A)
    edf = np.trace(Ainv @ G.T @ G)
    return x, edf, Ainv


def gcv_scan(G, d, lams):
    n, out = len(d), []
    for lam in lams:
        x, edf, _ = solve(G, d, lam)
        r = G @ x - d
        out.append((lam, n * (r @ r) / (n - edf) ** 2, np.sqrt(np.mean(r ** 2)), np.linalg.norm(L @ x), edf))
    return np.array(out)


def invert(Gs, Gr, target):
    G = Gs.reshape(-1, NK)
    d = target["strain"].T.reshape(-1)
    lams = np.logspace(-9, 1, 81)
    scan = gcv_scan(G, d, lams)
    lam = scan[np.argmin(scan[:, 1]), 0]
    x, edf, Ainv = solve(G, d, lam)
    r = G @ x - d
    sigma2 = (r @ r) / (len(d) - edf)
    cov = sigma2 * Ainv @ G.T @ G @ Ainv
    x0 = np.linalg.lstsq(G, d, rcond=None)[0]
    dp = np.r_[0.0, x]
    model_s = (Gs @ x).T                                            # [channel, window]
    model_r = (Gr @ x).T
    return dict(lam=lam, scan=scan, dp=dp, dp_sigma=np.r_[0.0, np.sqrt(np.diag(cov))],
                dp_unreg=np.r_[0.0, x0], model_strain=model_s, model_rate=model_r,
                edf=edf, cond=np.linalg.cond(G))


def rmse(a, b, skip_first=False):
    a, b = np.asarray(a), np.asarray(b)
    if skip_first:
        a, b = a[:, 1:], b[:, 1:]
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rel(a, b, skip_first=False):
    bb = b[:, 1:] if skip_first else b
    return rmse(a, b, skip_first) / float(np.sqrt(np.mean(bb ** 2)))


def approach_b(Gs, Gr, dp_base, base):
    xs = dp_base[1:]
    Ms, Mr = (Gs @ xs).T, (Gr @ xs).T
    out = {}
    for name, obs, M, S in [("strain", OBS_S, Ms, base["shear_strain"]), ("rate", OBS_R, Mr, base["shear_rate"])]:
        A = np.column_stack([M.ravel(), S.ravel()])
        (a, b), *_ = np.linalg.lstsq(A, obs.ravel(), rcond=None)
        out[name] = dict(a=float(a), b=float(b))
    for name, c in out.items():
        a, b = c["a"], c["b"]
        c["model_strain"] = a * Ms + b * base["shear_strain"]
        c["model_rate"] = a * Mr + b * base["shear_rate"]
        c["rmse_strain"] = rmse(c["model_strain"], OBS_S)
        c["rmse_rate"] = rmse(c["model_rate"], OBS_R)
        c["pressure_psi"] = P0_PSI + a * dp_base
    return out


def pressure_vs_opening(dp):
    o = OPEN_KNOTS
    alpha, beta = np.polyfit(o, dp, 1)
    pred = alpha * o + beta
    r2 = 1 - np.sum((dp - pred) ** 2) / np.sum((dp - dp.mean()) ** 2)
    n_dp, n_o = dp / dp[K_T3], o / o[K_T3]
    di, do = np.diff(dp), np.diff(o)
    lags = np.arange(-12, 13)
    cc = [np.corrcoef(di[max(0, l):len(di) + min(0, l)], do[max(0, -l):len(do) - max(0, l)])[0, 1] for l in lags]
    r2_12 = np.polyfit(o[:K_T2 + 1], dp[:K_T2 + 1], 1)
    return dict(alpha_psi_per_mm=float(alpha), beta_psi=float(beta), r2=float(r2),
                shape_rmse_normalized=float(np.sqrt(np.mean((n_dp - n_o) ** 2))),
                shape_rmse_normalized_T1T2=float(np.sqrt(np.mean((dp[:K_T2 + 1] / dp[K_T2] - o[:K_T2 + 1] / o[K_T2]) ** 2))),
                alpha_T1T2_psi_per_mm=float(r2_12[0]),
                best_lag_hours=int(2 * lags[int(np.nanargmax(cc))]), lag_corr=[float(v) for v in cc])


def knot_table(dp, sigma, dp_loc=None, dp_mean=None):
    """Imposed BC pressure on the monitoring well; optionally the resulting pore pressure on the fiber
    at the fracture crossing and averaged over the SRV outer zone (both lag the BC by diffusion)."""
    times = T1 + pd.to_timedelta(np.arange(115) * WIN, unit="s")
    p = P0_PSI + dp
    df = pd.DataFrame({
        "time_utc_minus_7": times.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s_from_T1": np.arange(115) * WIN,
        "pressure_psi": p, "pressure_MPa": p * PSI_MPA,
        "dP_from_T1_psi": dp, "dP_from_T1_MPa": dp * PSI_MPA,
        "dP_formal_1sigma_psi": sigma,
    })
    if dp_loc is not None:
        df["p_fiber_crossing_psi"], df["p_srv_outer_mean_psi"] = P0_PSI + dp_loc, P0_PSI + dp_mean
    df["marker"] = ["T1" if k == 0 else "T2" if k == K_T2 else "T3" if k == K_T3 else "" for k in range(115)]
    return df


def profile_table(target, inv):
    n_md, n_w = target["strain"].shape
    return pd.DataFrame({
        "center_time": np.repeat(CENTERS.strftime("%Y-%m-%d %H:%M:%S"), n_md),
        "md_ft": np.tile(MD, n_w),
        "target_strain_millistrain": target["strain"].T.ravel(),
        "model_strain_millistrain": inv["model_strain"].T.ravel(),
        "residual_strain_millistrain": (inv["model_strain"] - target["strain"]).T.ravel(),
        "target_rate_nanostrain_s": target["rate"].T.ravel(),
        "model_rate_nanostrain_s": inv["model_rate"].T.ravel(),
        "residual_rate_nanostrain_s": (inv["model_rate"] - target["rate"]).T.ravel(),
    })


def main():
    runs = [n for n in ["base", "fine", "srvx3", "srvd3", "hfcore16", "ic3000"]
            if (RUNS / n / f"v11_pinv_{n}_input_csv.csv").exists()
            and pd.read_csv(RUNS / n / f"v11_pinv_{n}_input_csv.csv")["time"].iloc[-1] >= 820800 - 1]
    print("finished runs:", runs)
    summary, store = {"runs": runs, "P0_psi": P0_PSI, "md_ft": MD.tolist()}, {}
    for run in runs:
        Gs, Gr, (tg, Rs) = basis(run)
        store[f"{run}/tg"], store[f"{run}/Rs"] = tg, Rs
        tp, r_loc, r_mean = pressure_response(run)
        res = {}
        for key, target in TARGETS.items():
            inv = invert(Gs, Gr, target)
            res[key] = inv
            for f in ["dp", "dp_sigma", "dp_unreg", "model_strain", "model_rate", "scan"]:
                store[f"{run}/{key}/{f}"] = inv[f]
            m = dict(lam=inv["lam"], edf=inv["edf"], cond=inv["cond"],
                     rmse_strain=rmse(inv["model_strain"], target["strain"]),
                     rel_rmse_strain=rel(inv["model_strain"], target["strain"], True),
                     rmse_rate=rmse(inv["model_rate"], target["rate"]),
                     rel_rmse_rate=rel(inv["model_rate"], target["rate"]),
                     dP_T2_psi=inv["dp"][K_T2], dP_T3_psi=inv["dp"][K_T3],
                     dP_T2_sigma_psi=inv["dp_sigma"][K_T2],
                     dP_T2_unreg_psi=inv["dp_unreg"][K_T2])
            m["vs_opening"] = pressure_vs_opening(inv["dp"])
            # early spike check: replace knots 1-5 by a straight line from 0 to knot 6, refit nothing
            xs = inv["dp"][1:].copy()
            xs[:5] = np.linspace(0.0, xs[5], 7)[1:6]
            store[f"{run}/{key}/dp_nospike"] = np.r_[0.0, xs]
            m["rel_rmse_strain_nospike"] = rel((Gs @ xs).T, target["strain"], True)
            loc = superpose_at_knots(tp, r_loc, inv["dp"][1:])
            mean = superpose_at_knots(tp, r_mean, inv["dp"][1:])
            store[f"{run}/{key}/dp_fiber_crossing"], store[f"{run}/{key}/dp_srv_mean"] = loc, mean
            m["dP_T2_fiber_crossing_psi"], m["dP_T2_srv_mean_psi"] = float(loc[K_T2]), float(mean[K_T2])
            m["dP_T3_fiber_crossing_psi"], m["dP_T3_srv_mean_psi"] = float(loc[K_T3]), float(mean[K_T3])
            G, d = Gs.reshape(-1, NK), target["strain"].T.reshape(-1)
            for f in (10.0, 100.0):
                x = solve(G, d, inv["lam"] * f)[0]
                m[f"dP_T2_lam_x{f:g}_psi"] = float(x[K_T2 - 1])
                m[f"dP_T3_lam_x{f:g}_psi"] = float(x[K_T3 - 1])
                m[f"rmse_strain_lam_x{f:g}"] = rmse((Gs @ x).T, target["strain"])
                store[f"{run}/{key}/dp_lam_x{f:g}"] = np.r_[0.0, x]
            summary[f"A/{run}/{key}"] = m
            print(f"[{run:6s}] A {key:16s} lam={m['lam']:.1e} edf={m['edf']:.1f}  "
                  f"rmse strain {m['rmse_strain']:.5f} ({100*m['rel_rmse_strain']:.1f}%)  "
                  f"rate {m['rmse_rate']:.4f} ({100*m['rel_rmse_rate']:.1f}%)  "
                  f"dP(T2)={m['dP_T2_psi']:.0f}+-{m['dP_T2_sigma_psi']:.0f}  dP(T3)={m['dP_T3_psi']:.0f} psi")
            if run == "base":
                knot_table(inv["dp"], inv["dp_sigma"], loc, mean).to_csv(RES / f"A_{key}_pressure_knots.csv", index=False)
                profile_table(target, inv).to_csv(RES / f"A_{key}_profiles_2h.csv", index=False)

        b = approach_b(Gs, Gr, res["baseline"]["dp"], TARGETS["baseline"])
        for fit, c in b.items():
            for f in ["model_strain", "model_rate", "pressure_psi"]:
                store[f"{run}/B_{fit}/{f}"] = c[f]
            summary[f"B/{run}/{fit}"] = {k: c[k] for k in ["a", "b", "rmse_strain", "rmse_rate"]}
            print(f"[{run:6s}] B fit-{fit:6s} a={c['a']:.3f} b={c['b']:.3f}  "
                  f"rmse strain {c['rmse_strain']:.5f}  rate {c['rmse_rate']:.4f}")
        if run == "base":
            for fit, c in b.items():
                knot_table(c["pressure_psi"] - P0_PSI, np.zeros(115)).to_csv(RES / f"B_fit{fit}_pressure_knots.csv", index=False)

    # reference decompositions (DDM only), and A-inverted + DDM inverted shear
    inv_t, base_t = TARGETS["inverted"], TARGETS["baseline"]
    refs = {
        "DDM baseline total (nb 03)": (base_t["strain"] + base_t["shear_strain"], base_t["rate"] + base_t["shear_rate"]),
        "DDM inverted total (nb 04)": (inv_t["strain"] + inv_t["shear_strain"], inv_t["rate"] + inv_t["shear_rate"]),
    }
    if "base" in runs:
        refs["A: MOOSE(inverted) + DDM inverted shear"] = (store["base/inverted/model_strain"] + inv_t["shear_strain"],
                                                           store["base/inverted/model_rate"] + inv_t["shear_rate"])
    summary["reference_fits"] = {k: dict(rmse_strain=rmse(s, OBS_S), rmse_rate=rmse(r, OBS_R)) for k, (s, r) in refs.items()}
    for k, v in summary["reference_fits"].items():
        print(f"  vs observed: {k:42s} strain {v['rmse_strain']:.5f}  rate {v['rmse_rate']:.4f}")

    np.savez_compressed(RES / "inversion_results.npz", **store, md=MD, centers=np.array(CENTERS.strftime("%Y-%m-%d %H:%M:%S"), dtype="U19"),
                        obs_strain=OBS_S, obs_rate=OBS_R, open_knots=OPEN_KNOTS,
                        **{f"target/{k}/{q}": v[q] for k, v in TARGETS.items() for q in v})
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("saved", RES)


if __name__ == "__main__":
    main()
