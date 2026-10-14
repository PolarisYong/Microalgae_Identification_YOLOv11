import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})
# -----------------------------
# Parameters
# -----------------------------
n_assets = np.arange(1, 51)          # Number of assets: 1 to 50
sigma_systematic = 0.0126            # Non-diversifiable risk floor
sigma_idiosyncratic = 0.0380         # Diversifiable risk when N=1

# Total portfolio risk:
# Total risk = sqrt(systematic^2 + idiosyncratic^2 / N)
total_risk = np.sqrt(sigma_systematic**2 + (sigma_idiosyncratic**2 / n_assets))

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 6))

# Total portfolio risk curve
plt.plot(
    n_assets, total_risk,
    color="#1f77b4", linewidth=2.5,
    label="Total Portfolio Risk"
)

# Systematic risk floor
plt.axhline(
    y=sigma_systematic,
    color="#d62728", linestyle="--", linewidth=2,
    label="Systematic Risk Floor"
)

# Fill diversifiable risk area
plt.fill_between(
    n_assets, sigma_systematic, total_risk,
    color="#87CEEB", alpha=0.35,
    label="Diversifiable Risk"
)

# Optional: highlight early rapid decline region
plt.axvspan(1, 10, color="gray", alpha=0.08)

# Annotations
plt.annotate(
    "Rapid risk reduction\nthrough diversification",
    xy=(4, total_risk[3]),
    xytext=(10, 0.028),
    arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
    fontsize=18
)

plt.annotate(
    "Diminishing marginal benefit",
    xy=(25, total_risk[24]),
    xytext=(30, 0.018),
    arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
    fontsize=18
)

plt.annotate(
    "Systematic risk cannot\nbe diversified away",
    xy=(42, sigma_systematic),
    xytext=(37, 0.0148),
    arrowprops=dict(arrowstyle="->", lw=1.2, color="#d62728"),
    fontsize=18,
    color="#b22222"
)

# Labels and title
plt.xlabel("Number of Assets in Portfolio", fontsize=18)
plt.ylabel("Portfolio Risk (Standard Deviation)", fontsize=18)

# Grid and legend
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend(frameon=True, fontsize=18, loc="upper right")

# Axis limits
plt.xlim(1, 50)
plt.ylim(0.010, 0.042)

plt.tight_layout()
plt.savefig("diversification_effect.png", dpi=300, bbox_inches="tight")
plt.close()