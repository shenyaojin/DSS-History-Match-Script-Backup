import numpy as np


def centered_time_diff(data, taxis):
    """
    Estimate strain rate from a strain array using a centered time difference.

    A centered difference approximates the instantaneous derivative AT each
    sample time (2nd-order accurate), unlike a backward difference which
    approximates the average rate over the PRECEDING interval. This matters
    because downstream reintegration (trapezoidal rule) assumes its input is
    a point sample of the rate at each time, not an interval-average - using
    a backward difference there introduces a systematic half-step lag.

    data: array, shape (n_channels, n_time)
    taxis: array, shape (n_time,), strictly increasing

    returns: rate array, same shape as data
    """
    taxis = np.asarray(taxis, dtype=float)
    data = np.asarray(data, dtype=float)
    rate = np.zeros_like(data)

    # interior points: centered difference
    rate[:, 1:-1] = (data[:, 2:] - data[:, :-2]) / (taxis[2:] - taxis[:-2])

    # boundaries: one-sided difference (best available without extra data)
    rate[:, 0] = (data[:, 1] - data[:, 0]) / (taxis[1] - taxis[0])
    rate[:, -1] = (data[:, -1] - data[:, -2]) / (taxis[-1] - taxis[-2])

    return rate


def integrate_rate_trapezoidal(rate, taxis):
    """
    Reconstruct strain (indefinite integral, referenced to 0 at the first
    sample) from a rate array using the composite trapezoidal rule - the
    same scheme DASCore's Patch.integrate(dim="time") uses on real DAS
    strain-rate data, so simulated and observed strain can be produced by
    a matching operator.

    rate: array, shape (n_channels, n_time)
    taxis: array, shape (n_time,)

    returns: strain array, same shape as rate, strain[:, 0] == 0
    """
    taxis = np.asarray(taxis, dtype=float)
    rate = np.asarray(rate, dtype=float)
    dt = np.diff(taxis)
    strain = np.zeros_like(rate)
    if rate.shape[1] > 1:
        strain[:, 1:] = np.cumsum(
            0.5 * (rate[:, :-1] + rate[:, 1:]) * dt[None, :], axis=1
        )
    return strain


def build_dense_taxis(t_start, t_end, coarse_dt, refine_times, fine_dt, fine_half_width):
    """
    Build a time axis that is coarsely sampled everywhere, but refined
    (densified) only in windows around specified transition times. This
    keeps the total sample count - and therefore the DDM evaluation cost,
    which scales ~linearly with sample count - much lower than uniformly
    refining the whole time range.

    t_start, t_end: bounds of the time axis
    coarse_dt: base spacing used away from transitions
    refine_times: iterable of times (e.g. T1, T2, T3) to densify around
    fine_dt: spacing used inside each refinement window
    fine_half_width: half-width of each refinement window around a refine_time

    returns: sorted, de-duplicated 1D array of time samples
    """
    coarse = np.arange(t_start, t_end + coarse_dt, coarse_dt)
    pieces = [coarse]
    for rt in refine_times:
        lo = max(t_start, rt - fine_half_width)
        hi = min(t_end, rt + fine_half_width)
        if hi > lo:
            pieces.append(np.arange(lo, hi + fine_dt, fine_dt))
    taxis = np.unique(np.concatenate(pieces))
    taxis = taxis[(taxis >= t_start) & (taxis <= t_end)]
    return taxis
