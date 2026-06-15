"""
Plot all-data Bearskin pressure gauges from fiberis-format NPZ files.

This reads the full pressure traces in:
    data_fervo/fiberis_format/pressure_data/*_Pressure.npz

and saves one clean figure plus a small summary table under figs/.

Run from the repository root:
    python scripts/tensile_fault/107_plot_all_pressure_data.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FIBERIS_SRC = REPO_ROOT / "fibeRIS" / "src"
if str(FIBERIS_SRC) not in sys.path:
    sys.path.insert(0, str(FIBERIS_SRC))

from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge


PRESSURE_DIR = REPO_ROOT / "data_fervo" / "fiberis_format" / "pressure_data"
FIG_DIR = REPO_ROOT / "figs" / "tensile_fault_pressure_data"


def pretty_well_name(path: Path) -> str:
    name = path.stem
    name = name.replace("Bearskin_", "Bearskin ")
    name = name.replace("_Pressure", "")
    return name


def load_gauge(path: Path, name: str | None = None) -> Data1DGauge:
    gauge = Data1DGauge()
    gauge.load_npz(str(path))
    gauge.name = name or pretty_well_name(path)
    return gauge


def load_all_pressure_gauges() -> list[Data1DGauge]:
    paths = sorted(PRESSURE_DIR.glob("Bearskin_*_Pressure.npz"))
    if not paths:
        raise FileNotFoundError(f"No all-data pressure files found in {PRESSURE_DIR}")
    return [load_gauge(path) for path in paths]


def load_stage_starts() -> list[tuple[int, np.datetime64]]:
    stage_paths = sorted(PRESSURE_DIR.glob("*_Stage_*.npz"))
    starts_by_stage: dict[int, np.datetime64] = {}

    for path in stage_paths:
        match = re.search(r"_Stage_(\d+)", path.stem)
        if not match:
            continue
        stage = int(match.group(1))
        gauge = load_gauge(path, name=path.stem)
        start = np.datetime64(gauge.start_time)
        if stage in starts_by_stage:
            starts_by_stage[stage] = min(starts_by_stage[stage], start)
        else:
            starts_by_stage[stage] = start

    return [(stage, start) for stage, start in sorted(starts_by_stage.items())]


def write_summary(gauges: list[Data1DGauge], output_path: Path) -> None:
    lines = [
        "well,n_points,start_time,end_time,min_psi,max_psi,mean_psi",
    ]
    for gauge in gauges:
        times = gauge.calculate_time()
        lines.append(
            ",".join(
                [
                    str(gauge.name),
                    str(len(gauge.data)),
                    str(times[0]),
                    str(times[-1]),
                    f"{np.nanmin(gauge.data):.3f}",
                    f"{np.nanmax(gauge.data):.3f}",
                    f"{np.nanmean(gauge.data):.3f}",
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_pressure_data(
    gauges: list[Data1DGauge],
    stage_starts: list[tuple[int, np.datetime64]],
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        len(gauges),
        1,
        figsize=(13.5, 8.4),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    if len(gauges) == 1:
        axes = [axes]

    colors = {
        "Bearskin 1-IA": "#1f77b4",
        "Bearskin 3-PA": "#d62728",
        "Bearskin 4-PB": "#2ca02c",
    }

    for ax, gauge in zip(axes, gauges):
        for stage, start in stage_starts:
            ax.axvline(start, color="#737982", alpha=0.2, linewidth=0.9)

        time_axis = gauge.calculate_time()
        ax.plot(
            time_axis,
            gauge.data,
            color=colors.get(gauge.name),
            linewidth=1.15,
            alpha=0.95,
        )
        ax.text(
            0.012,
            0.86,
            gauge.name,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color=colors.get(gauge.name),
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        ax.set_ylabel("Pressure (psi)")
        ax.set_ylim(bottom=0)
        ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.16)
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x:,.0f}")

    top_ax = axes[0]
    for stage, start in stage_starts:
        top_ax.text(
            start,
            1.02,
            f"S{stage}",
            transform=top_ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
            color="#5b6067",
        )

    axes[-1].set_xlabel("Date/time")
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))

    start_times = [g.calculate_time()[0] for g in gauges]
    end_times = [g.calculate_time()[-1] for g in gauges]
    axes[-1].set_xlim(min(start_times), max(end_times))

    fig.suptitle("Bearskin Pressure Data - Full Traces", fontsize=16, y=0.985)
    subtitle = (
        f"{len(gauges)} full pressure records from {PRESSURE_DIR.relative_to(REPO_ROOT)}; "
        "vertical markers show stage starts 22-28"
    )
    fig.text(0.125, 0.952, subtitle, fontsize=9.5, color="#4d535a")

    fig.autofmt_xdate(rotation=0, ha="center")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.91, hspace=0.08)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    gauges = load_all_pressure_gauges()
    stage_starts = load_stage_starts()

    figure_path = FIG_DIR / "bearskin_all_pressure_data.png"
    summary_path = FIG_DIR / "bearskin_all_pressure_data_summary.csv"

    plot_pressure_data(gauges, stage_starts, figure_path)
    write_summary(gauges, summary_path)

    print(f"Saved pressure figure: {figure_path}")
    print(f"Saved pressure summary: {summary_path}")


if __name__ == "__main__":
    main()
