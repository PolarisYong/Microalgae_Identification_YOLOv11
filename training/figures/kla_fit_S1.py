from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

@dataclass
class Row:
    time_s: float
    do_mg_l: float
    cstar_mg_l: float
    ln_val: float | None
    used_for_fit: bool


def load_rows(csv_path: Path) -> list[Row]:
    rows: list[Row] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            ln_raw = (record.get("ln_Cstar_minus_CL") or "").strip()
            rows.append(
                Row(
                    time_s=float(record["time_s"]),
                    do_mg_l=float(record["DO_mg_L"]),
                    cstar_mg_l=float(record["Cstar_mg_L"]),
                    ln_val=None if not ln_raw else float(ln_raw),
                    used_for_fit=(record.get("used_for_fit") or "").strip().lower() == "yes",
                )
            )
    return sorted(rows, key=lambda row: row.time_s)


def linear_regression(rows: list[Row]) -> tuple[float, float, float, float, float]:
    fit_rows = [row for row in rows if row.used_for_fit and row.ln_val is not None]
    if len(fit_rows) < 2:
        raise ValueError("Not enough fit points were found in kLa_BG11_DO_fit_table.csv.")

    n = float(len(fit_rows))
    sum_x = sum(row.time_s for row in fit_rows)
    sum_y = sum(row.ln_val for row in fit_rows if row.ln_val is not None)
    sum_xy = sum(row.time_s * row.ln_val for row in fit_rows if row.ln_val is not None)
    sum_x2 = sum(row.time_s * row.time_s for row in fit_rows)
    sum_y2 = sum(row.ln_val * row.ln_val for row in fit_rows if row.ln_val is not None)

    slope = ((n * sum_xy) - (sum_x * sum_y)) / ((n * sum_x2) - (sum_x * sum_x))
    intercept = (sum_y - (slope * sum_x)) / n
    r2_num = (n * sum_xy) - (sum_x * sum_y)
    r2_den = ((n * sum_x2) - (sum_x * sum_x)) * ((n * sum_y2) - (sum_y * sum_y))
    r2 = (r2_num * r2_num) / r2_den
    kla_s = -1.0 * slope
    kla_h = kla_s * 3600.0
    return slope, intercept, r2, kla_s, kla_h


def plot_figure(rows: list[Row], output_path: Path) -> None:
    fit_rows = [row for row in rows if row.used_for_fit and row.ln_val is not None]
    slope, intercept, r2, _kla_s, kla_h = linear_regression(rows)
    cstar = rows[0].cstar_mg_l

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=200)


    ax1.plot(
        [row.time_s for row in rows],
        [row.do_mg_l for row in rows],
        color="#27548A",
        linewidth=2.5,
        marker="o",
        markersize=4.5,
        label="[DO] profile",
    )
    ax1.scatter(
        [row.time_s for row in fit_rows],
        [row.do_mg_l for row in fit_rows],
        color="#C63840",
        s=30,
        label="Fit window",
        zorder=3,
    )
    ax1.plot(
        [rows[0].time_s, rows[-1].time_s],
        [cstar, cstar],
        color="forestgreen",
        linewidth=2,
        linestyle="--",
        label="[DO]*",
    )
    ax1.text(
        -0.07, 1.02, "(a)",
        transform=ax1.transAxes,
        fontsize=24,
        fontweight="bold",
        fontname="Times New Roman"
    )
    ax1.set_xlabel("Time (s)", fontsize=28, fontweight="bold")
    ax1.set_ylabel("[DO] (mg L$^{-1}$)", fontsize=28, fontweight="bold")
    ax1.tick_params(axis="both", direction="in", labelsize=16, length=6, width=1.5)
    ax1.set_xlim(0, 2600)
    ax1.set_ylim(0, 6.5)
    ax1.set_xticks(range(0, 2601, 300))
    ax1.set_yticks(range(0, 7, 1))
    ax1.grid(True, color="gainsboro")
    ax1.legend(loc="lower center", fontsize=18)

    ax2.scatter(
        [row.time_s for row in fit_rows],
        [row.ln_val for row in fit_rows],
        color="#27548A",
        s=30,
        label="Linear-fit points",
        zorder=3,
    )
    x1 = fit_rows[0].time_s
    x2 = fit_rows[-1].time_s
    ax2.plot(
        [x1, x2],
        [(slope * x1) + intercept, (slope * x2) + intercept],
        color="#C63840",
        linewidth=2.5,
        label="Linear fit",
    )
    ax2.text(
        1.06, 1.02, "(b)",
        transform=ax1.transAxes,
        fontsize=24,
        fontweight="bold",
        fontname="Times New Roman"
    )
    ax2.set_xlabel("Time (s)", fontsize=28, fontweight="bold")
    ax2.set_ylabel("ln([DO]* - [DO])", fontsize=28, fontweight="bold")
    ax2.tick_params(axis="both", direction="in", labelsize=16, length=6, width=1.5)
    ax2.set_xlim(60, 1050)
    ax2.grid(True, color="gainsboro")
    ax2.legend(loc="lower center", fontsize=18)

    annotation = (
        f"$k_L$a = {kla_h:.2f} h$^{{-1}}$\n"
        f"[DO]* = {cstar:.3f} mg L$^{{-1}}$\n"
        f"R$^2$ = {r2:.4f}\n"
        f"ln([DO]*  - [DO]) = {slope:.6f}t + {intercept:.4f}"
    )
    ax2.text(
        0.50,
        0.82,
        annotation,
        transform=ax2.transAxes,
        fontsize=14,
        bbox={"facecolor": "#EBF5FF", "edgecolor": "lightgray"},
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(r"E:\pythonProject\Microalgae_Identification_YOLOv11\generate_figures")
    fit_table_path = root / "kLa_BG11_DO_fit_table.csv"
    output_path = root / "Fig_S1_kLa_reactor_characterization.png"

    rows = load_rows(fit_table_path)
    plot_figure(rows, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
