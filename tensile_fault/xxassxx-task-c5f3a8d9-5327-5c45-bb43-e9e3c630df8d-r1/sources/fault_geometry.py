"""Canonical fault / SRV geometry constants, so the scripts cannot drift apart.

The MD values below are the projections of the DDM fault planes onto the Gold 4-PB monitoring
well, supplied by Shenyao from the modelling notebook. They must come from the notebook rather
than from this repo's own projection: the notebook resamples and SMOOTHS the well path
(`Well.set_well_by_points(control_points, N=len*10, smooth=31)`) before projecting, which
shifts the MD <-> position mapping by ~10 ft against a raw linear interpolation of the
168-point survey. Since the DDM and observed profiles are indexed on the notebook's MD axis,
using anything else misaligns the MOOSE basis against them by that same ~10 ft.

    parallel-fault update (T1 = 2025-02-24 12:00)
        SRV centre / fault1 tensile plane  ->  MD 10365.107
        fault2 shear plane                 ->  MD 10317.837
        separation                             47.270 ft   <- sets the SRV half-width

    previous vintage (T1 = 2025-02-24 11:00), kept so the old runs stay reproducible
        fracture MD 10373.4, shear plane MD 10319.0
"""

# --- parallel-fault vintage ---------------------------------------------------
STAR_1200 = 10365.107          # SRV centre = tensile fracture, MD ft
SHEAR_MD_1200 = 10317.837      # DDM shear plane projected on the well, MD ft
HALF_1200 = STAR_1200 - SHEAR_MD_1200          # 47.270 ft

# --- previous vintage ---------------------------------------------------------
STAR_1100 = 10373.4
SHEAR_MD_1100 = 10319.0

T1_1200 = "2025-02-24 12:00"
T2_1200 = "2025-02-28 00:00"
T3_1200 = "2025-03-04 00:00"

SUFFIX_1200 = "20250224_1200_to_20250304_0000_10200_10500ft_4h_mean_T1_ref"

# Zone half-heights follow the V2 grading ratios, scaled so srv_outer's edge lands exactly on
# the shear plane while the stack stays symmetric about the fracture.
SRV_ZONES_1200 = [
    ("srv_outer", HALF_1200, 1e-17, 0.08),
    ("srv_wide", HALF_1200 * 90.0 / 130.0, 3e-17, 0.10),
    ("srv_narrow", HALF_1200 * 45.0 / 130.0, 1e-16, 0.12),
]


# Projects built BEFORE the parallel-fault update use the old MDs. Everything else uses the
# canonical 1200 values. Listing the old ones explicitly avoids the prefix-matching bug that
# silently mislabelled v11 with STAR_1100 (the SRV then looked 8 ft short of the shear plane).
LEGACY_PROJECTS = ("v3_srv_asym", "v4_srv_centred", "v4_srv_boundary", "v5_full_length",
                   "v6_symmetric_srv", "v6b_bc_at_fiber", "v7_past_injector",
                   "v7b_bc_at_fiber")


def star(vintage):
    return STAR_1200 if str(vintage) == "1200" else STAR_1100


def shear_md(vintage):
    return SHEAR_MD_1200 if str(vintage) == "1200" else SHEAR_MD_1100


def star_for_project(proj):
    return STAR_1100 if proj in LEGACY_PROJECTS else STAR_1200


def shear_md_for_project(proj):
    return SHEAR_MD_1100 if proj in LEGACY_PROJECTS else SHEAR_MD_1200


if __name__ == "__main__":
    print(f"SRV centre / fracture   MD {STAR_1200:.3f}")
    print(f"shear plane             MD {SHEAR_MD_1200:.3f}")
    print(f"separation (SRV half)      {HALF_1200:.3f} ft")
    for n, h, k, por in SRV_ZONES_1200:
        print(f"  {n:11s} MD {STAR_1200 - h:9.3f} .. {STAR_1200 + h:9.3f}  "
              f"+/-{h:6.3f} ft  k={k:.0e}  phi={por}")
