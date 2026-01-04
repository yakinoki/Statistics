# =========================================
# SEM (covariance structure analysis) demo
# Theme: Sales and Advertising/CM effects
# Library: semopy
# =========================================

import numpy as np
import pandas as pd

# ---- (A) Install notes ----
# pip install semopy graphviz pandas numpy
#
# If the semplot output fails, install Graphviz:
# - Windows: choco install graphviz  (or installer)
# - macOS:   brew install graphviz
# - Linux:   sudo apt-get install graphviz
#
# semopy uses Graphviz to render the path diagram.

# ---- (B) Create example data (replace with your actual dataset) ----
rng = np.random.default_rng(42)
n = 400  # e.g., weekly observations

# Exogenous observed variables
tv_spend = rng.normal(0, 1, n)
online_spend = rng.normal(0, 1, n)
influencer_spend = rng.normal(0, 1, n)
price_index = rng.normal(0, 1, n)            # higher => more expensive
seasonality = rng.normal(0, 1, n)
competitor_promo = rng.normal(0, 1, n)

# Latent: Marketing Pressure (MKT)
mkt_latent = (
    0.6 * tv_spend
    + 0.7 * online_spend
    + 0.5 * influencer_spend
    + rng.normal(0, 0.7, n)
)

# Indicators for MKT
ad_reach = 0.9 * mkt_latent + rng.normal(0, 0.6, n)
ad_freq = 0.8 * mkt_latent + rng.normal(0, 0.7, n)
share_of_voice = 0.7 * mkt_latent + rng.normal(0, 0.7, n)

# Latent: Awareness (AW) driven by MKT + controls
aw_latent = (
    0.7 * mkt_latent
    + 0.2 * seasonality
    - 0.1 * competitor_promo
    + rng.normal(0, 0.8, n)
)

# Indicators for AW
search_index = 0.9 * aw_latent + rng.normal(0, 0.6, n)
survey_awareness = 0.8 * aw_latent + rng.normal(0, 0.7, n)
sns_mentions = 0.7 * aw_latent + rng.normal(0, 0.7, n)

# Outcome: Sales (observed)
sales = (
    0.55 * aw_latent          # mediated effect via awareness
    + 0.25 * mkt_latent       # direct effect (immediate response)
    - 0.35 * price_index
    + 0.25 * seasonality
    - 0.20 * competitor_promo
    + rng.normal(0, 1.0, n)
)

df = pd.DataFrame({
    "ad_reach": ad_reach,
    "ad_freq": ad_freq,
    "share_of_voice": share_of_voice,
    "search_index": search_index,
    "survey_awareness": survey_awareness,
    "sns_mentions": sns_mentions,
    "sales": sales,
    "price_index": price_index,
    "seasonality": seasonality,
    "competitor_promo": competitor_promo
})
# ---- (B2) Quick visualization of the generated "input data" ----
import os
import matplotlib.pyplot as plt

out_dir = "sem_viz"
os.makedirs(out_dir, exist_ok=True)

# (1) Basic summary (table)
print("\n=== Descriptive stats (selected) ===")
print(df.describe().T[["mean", "std", "min", "max"]])

# (2) Histograms of key variables
cols_hist = ["sales", "ad_reach", "search_index", "price_index", "seasonality", "competitor_promo"]
df[cols_hist].hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "01_histograms.png"), dpi=200)
plt.close()

# (3) Correlation matrix heatmap (input structure)
corr = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
plt.imshow(corr, aspect="auto")  # default colormap is fine
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation matrix (observed variables)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "02_corr_heatmap.png"), dpi=200)
plt.close()

# (4) Create simple proxy indices for MKT and AW (for intuitive scatter plots)
#     (SEMでは潜在変数ですが、説明用に「観測指標の平均Z」を作ると通りやすいです)
def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)

mkt_proxy = (
    zscore(df["ad_reach"]) + zscore(df["ad_freq"]) + zscore(df["share_of_voice"])
) / 3.0
aw_proxy = (
    zscore(df["search_index"]) + zscore(df["survey_awareness"]) + zscore(df["sns_mentions"])
) / 3.0

df["mkt_proxy"] = mkt_proxy
df["aw_proxy"] = aw_proxy

# (5) Scatter: MKT proxy -> AW proxy (this supports the path MKT -> AW)
plt.figure(figsize=(6, 5))
plt.scatter(df["mkt_proxy"], df["aw_proxy"], s=12, alpha=0.6)
plt.xlabel("Marketing proxy (mean z of indicators)")
plt.ylabel("Awareness proxy (mean z of indicators)")
plt.title("Input data: MKT proxy vs AW proxy")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "03_scatter_mkt_aw.png"), dpi=200)
plt.close()

# (6) Scatter: AW proxy -> Sales (supports AW -> sales)
plt.figure(figsize=(6, 5))
plt.scatter(df["aw_proxy"], df["sales"], s=12, alpha=0.6)
plt.xlabel("Awareness proxy (mean z of indicators)")
plt.ylabel("Sales")
plt.title("Input data: Awareness proxy vs Sales")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "04_scatter_aw_sales.png"), dpi=200)
plt.close()

# (7) Scatter: MKT proxy -> Sales (supports direct path MKT -> sales)
plt.figure(figsize=(6, 5))
plt.scatter(df["mkt_proxy"], df["sales"], s=12, alpha=0.6)
plt.xlabel("Marketing proxy (mean z of indicators)")
plt.ylabel("Sales")
plt.title("Input data: Marketing proxy vs Sales")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "05_scatter_mkt_sales.png"), dpi=200)
plt.close()

print(f"\nSaved visualizations to ./{out_dir}/")
print(" - 01_histograms.png")
print(" - 02_corr_heatmap.png")
print(" - 03_scatter_mkt_aw.png")
print(" - 04_scatter_aw_sales.png")
print(" - 05_scatter_mkt_sales.png")

# ---- (C) Fit SEM with semopy ----
from semopy import Model, calc_stats, semplot

model_desc = """
# Measurement model (latent -> observed indicators)
MKT =~ ad_reach + ad_freq + share_of_voice
AW  =~ search_index + survey_awareness + sns_mentions

# Structural model
AW ~ MKT + seasonality + competitor_promo
sales ~ AW + MKT + price_index + seasonality + competitor_promo

# Correlated exogenous variables (optional but realistic)
price_index ~~ competitor_promo
seasonality ~~ competitor_promo
"""

mod = Model(model_desc)
mod.fit(df)

# ---- (D) Inspect results ----
# Standardized estimates are great for slides
est = mod.inspect(std_est=True)

# Fit indices (CFI/TLI/RMSEA etc.)
fit = calc_stats(mod)

print("=== Fit indices (selected) ===")
# calc_stats returns a dict-like object; keys can vary by version
# so we print it as a whole for safety:
print(fit)

print("\n=== Standardized estimates (top rows) ===")
print(est.head(20))

# ---- (E) Plot path diagram (PNG) ----
# Use the inspection table so coefficients appear on the diagram.
# std_ests=True prints standardized estimates on edges.
g = semplot(
    mod,
    filename="sem_path.png",
    inspection=est,
    plot_covs=True,
    std_ests=True
)

# If you want to see/modify the Graphviz source:
# print(g.source)

print("\nSaved: sem_path.png")
