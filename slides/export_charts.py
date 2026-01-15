#!/usr/bin/env python3
"""Export presentation charts as PNG files."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import Ridge

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 14
plt.rcParams['figure.dpi'] = 150

# Paths
ROOT = Path(__file__).parent.parent.resolve()
OUT = ROOT / 'slides' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# Load data
ad = pd.read_csv(ROOT / 'data' / 'processed' / 'ad_data_clean.csv', parse_dates=['date'])
app = pd.read_csv(ROOT / 'data' / 'processed' / 'app_data_clean.csv', parse_dates=['date'])
df = app.merge(ad, on='date', how='left')

CAMPAIGN_START = pd.Timestamp('2014-10-07')
df['is_campaign'] = df['date'] >= CAMPAIGN_START
df['t'] = (df['date'] - df['date'].min()).dt.days.astype(float)
df['dow'] = df['date'].dt.dayofweek.astype(int)

# Prepare subsets
pre_period = df[~df['is_campaign']].copy()
camp = df[df['is_campaign']].copy()
camp['cpi_paid'] = camp['spend'] / camp['paid_downloads']
camp['cpi_all'] = camp['spend'] / camp['total_downloads']
CPI_WORKING = camp['spend'].sum() / camp['total_downloads'].sum()

pre_avg = pre_period['non_attributed_downloads'].mean()
camp_non_attr_avg = camp['non_attributed_downloads'].mean()

print(f"Exporting charts to {OUT}/")

# =============================================================================
# CHART 1: Halo Effect (3-panel)
# =============================================================================
print("  Chart 1: Halo effect...")

plot_df = df.copy()
plot_df['paid_downloads'] = plot_df['paid_downloads'].fillna(0)
plot_df['spend'] = plot_df['spend'].fillna(0)
pre_df = plot_df[~plot_df['is_campaign']]
camp_df = plot_df[plot_df['is_campaign']]

fig, (ax_pre, ax_mid, ax_full) = plt.subplots(3, 1, figsize=(14, 10), 
                                               gridspec_kw={'height_ratios': [1, 1.2, 1.5]})

# Top: Pre-campaign
ax_pre.bar(pre_df['date'], pre_df['non_attributed_downloads'], 
           label='Non-attributed', color='#94a3b8', alpha=0.9, width=0.8)
ax_pre.set_ylabel('Downloads', fontsize=12)
ax_pre.set_ylim(0, 35)
ax_pre.set_title(f'Before Campaign: Non-Attributed Downloads (avg {pre_avg:.0f}/day)', 
                 fontsize=13, fontweight='bold', color='#64748b')
ax_pre.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax_pre.axhline(pre_avg, color='#dc2626', linestyle='--', linewidth=2, label=f'Avg: {pre_avg:.0f}/day')
ax_pre.legend(loc='upper right', fontsize=11)

# Middle: During-campaign non-attributed
ax_mid.bar(camp_df['date'], camp_df['non_attributed_downloads'], 
           label='Non-attributed', color='#94a3b8', alpha=0.9, width=0.8)
ax_mid.set_ylabel('Downloads', fontsize=12)
ax_mid.set_title(f'During Campaign: Non-Attributed Downloads (avg {camp_non_attr_avg:.0f}/day) — {camp_non_attr_avg/pre_avg:.0f}× increase', 
                 fontsize=13, fontweight='bold', color='#22c55e')
ax_mid.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax_mid.axhline(camp_non_attr_avg, color='#22c55e', linestyle='--', linewidth=2, label=f'Avg: {camp_non_attr_avg:.0f}/day')
ax_mid.axhline(pre_avg, color='#dc2626', linestyle=':', linewidth=2, alpha=0.7, label=f'Pre-campaign avg: {pre_avg:.0f}/day')
ax_mid.legend(loc='upper right', fontsize=11)

# Bottom: Full stacked
ax_full.bar(camp_df['date'], camp_df['paid_downloads'], 
            label='Paid (attributed)', color='#2563eb', alpha=0.9, width=0.8)
ax_full.bar(camp_df['date'], camp_df['non_attributed_downloads'], 
            bottom=camp_df['paid_downloads'], label='Non-attributed', color='#94a3b8', alpha=0.8, width=0.8)
ax_full.set_ylabel('Downloads', fontsize=12)
ax_full.set_title('Full Picture: Paid + Non-Attributed', fontsize=13, fontweight='bold', color='#2563eb')
ax_full2 = ax_full.twinx()
ax_full2.plot(camp_df['date'], camp_df['spend'], color='#f97316', linewidth=2.5)
ax_full2.set_ylabel('Spend ($)', color='#f97316', fontsize=12)
ax_full2.tick_params(axis='y', labelcolor='#f97316')
ax_full2.set_ylim(0, 120)
ax_full.legend(loc='upper right', fontsize=11)
ax_full.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.suptitle('The Halo Effect: Non-Attributed Downloads Explode After Campaign Start', 
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT / 'chart1_halo.png', bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# CHART 2: CPI Comparison
# =============================================================================
print("  Chart 2: CPI comparison...")

cpi_paid_avg = camp['cpi_paid'].mean()
cpi_all_avg = camp['cpi_all'].mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(camp['date'], camp['cpi_paid'], label='CPI_paid (attributed only)', color='#dc2626', linewidth=2.5)
ax.plot(camp['date'], camp['cpi_all'], label='CPI_all (includes halo)', color='#2563eb', linewidth=3)
ax.fill_between(camp['date'], camp['cpi_all'], camp['cpi_paid'],
                where=(camp['cpi_paid'] >= camp['cpi_all']), color='#fca5a5', alpha=0.25,
                label='Gap (non-attributed installs)')
ax.axhline(cpi_paid_avg, color='#dc2626', linestyle='--', linewidth=1.5, alpha=0.5)
ax.axhline(cpi_all_avg, color='#2563eb', linestyle='--', linewidth=1.5, alpha=0.5)
ax.set_ylabel('Cost per Install ($)', fontsize=14)
ax.legend(loc='upper left', fontsize=12)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=45)
ax.set_ylim(0, 0.12)
plt.title('CPI_paid vs CPI_all: The True Acquisition Cost', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'chart2_cpi.png', bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# CHART 3: LTV Curve (Deconvolution)
# =============================================================================
print("  Chart 3: LTV curve...")

MAX_LAG = 30
df['incremental_installs'] = np.where(df['is_campaign'], df['total_downloads'].astype(float), 0.0)
df['incremental_revenue'] = np.where(df['is_campaign'], df['total_revenue'].astype(float), 0.0)

lag_cols = []
for k in range(MAX_LAG + 1):
    lag_cols.append(df['incremental_installs'].shift(k).fillna(0.0).to_numpy())
X = np.column_stack(lag_cols)
y = df['incremental_revenue'].to_numpy()
mask = df['is_campaign'].to_numpy()
Xc, yc = X[mask], y[mask]

ridge = Ridge(alpha=1.0, positive=True, fit_intercept=False)
ridge.fit(Xc, yc)
h_coeffs = ridge.coef_
ltv_cumulative = np.cumsum(h_coeffs)
LTV_30 = float(ltv_cumulative[29])

fig, ax = plt.subplots(figsize=(10, 5))
days = np.arange(MAX_LAG + 1)
ax.plot(days, ltv_cumulative, color='#2563eb', linewidth=3, label='Cumulative LTV')
ax.axhline(CPI_WORKING, color='#dc2626', linestyle='--', linewidth=2.5, 
           label=f'CPI_all (${CPI_WORKING:.3f})')
payback_day = np.where(ltv_cumulative >= CPI_WORKING)[0]
if len(payback_day) > 0:
    pb = int(payback_day[0])
    ax.axvline(pb, color='#22c55e', linestyle=':', linewidth=2, alpha=0.7)
    ax.scatter([pb], [ltv_cumulative[pb]], color='#22c55e', s=150, zorder=5)
    ax.annotate(f'Payback: Day {pb}', xy=(pb, ltv_cumulative[pb]), 
                xytext=(pb+3, ltv_cumulative[pb]+0.012), fontsize=14, fontweight='bold')
ax.set_xlabel('Days Since Install', fontsize=14)
ax.set_ylabel('Cumulative Revenue per Install ($)', fontsize=14)
ax.legend(loc='lower right', fontsize=12)
ax.set_xlim(0, MAX_LAG)
ax.set_ylim(0, max(ltv_cumulative.max(), CPI_WORKING) * 1.25)
plt.title('LTV Curve vs Acquisition Cost', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'chart3_ltv.png', bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# CHART 4: Daily Profitability
# =============================================================================
print("  Chart 4: Daily profitability...")

camp_ux = camp.copy()
camp_ux['margin_per_install_30d'] = LTV_30 - camp_ux['cpi_all']
camp_ux['net_profit_daily_30d'] = camp_ux['total_downloads'] * camp_ux['margin_per_install_30d']
DEC1 = pd.Timestamp('2014-12-01')

fig, ax1 = plt.subplots(figsize=(12, 5))
bar_colors = np.where(camp_ux['net_profit_daily_30d'] >= 0, '#22c55e', '#ef4444')
ax1.bar(camp_ux['date'], camp_ux['net_profit_daily_30d'], color=bar_colors, alpha=0.65, width=0.8, 
        label='Estimated net profit ($/day)')
ax1.axhline(0, color='#0f172a', linewidth=1)
ax1.set_ylabel('Estimated net profit ($/day)', fontsize=14)
ax1.axvline(DEC1, color='#0f172a', linestyle=':', linewidth=2, alpha=0.7)
ax1.text(DEC1, ax1.get_ylim()[1] * 0.95, 'Dec 1', rotation=90, va='top', ha='right', fontsize=12)

ax2 = ax1.twinx()
ax2.plot(camp_ux['date'], camp_ux['margin_per_install_30d'], color='#2563eb', linewidth=3,
         label='Estimated margin/install (LTV_30 - CPI)')
ax2.plot(camp_ux['date'], camp_ux['appd_paid_d0'], color='#64748b', linestyle='--', linewidth=2.5,
         label='Reported APPD (day-0)')
ax2.set_ylabel('$/install', fontsize=14)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=11)

ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=45)
plt.title('Daily Profitability: LTV-Implied vs Reported APPD', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(OUT / 'chart4_profit.png', bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# CHART 5: Weekday Seasonality
# =============================================================================
print("  Chart 5: Weekday seasonality...")

camp_seas = camp.copy()
camp_seas['value_per_dollar_30'] = LTV_30 / camp_seas['cpi_all']
camp_seas['dow_name'] = camp_seas['date'].dt.day_name()
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
camp_seas['week_start'] = camp_seas['date'].dt.to_period('W-MON').apply(lambda p: p.start_time)
week_mean = camp_seas.groupby('week_start')['value_per_dollar_30'].transform('mean')
camp_seas['rel_value_per_dollar'] = camp_seas['value_per_dollar_30'] / week_mean

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=camp_seas, x='dow_name', y='rel_value_per_dollar', hue='dow_name',
            order=order, ax=ax, palette='RdYlGn', errorbar=('ci', 95), capsize=0.1, legend=False)
ax.axhline(1.0, color='#0f172a', linestyle='--', linewidth=2)
ax.set_xlabel('')
ax.set_ylabel('Relative value per $ (vs same-week avg)', fontsize=14)
ax.set_title('Weekday Seasonality: ROI by Day of Week', fontsize=16, fontweight='bold')

# Add percentage labels
summary = camp_seas.groupby('dow_name')['rel_value_per_dollar'].mean().reindex(order)
for i, (day, val) in enumerate(summary.items()):
    pct = (val - 1) * 100
    color = '#22c55e' if pct > 0 else '#ef4444'
    ax.text(i, val + 0.02, f'{pct:+.0f}%', ha='center', fontsize=12, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig(OUT / 'chart5_weekday.png', bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n✓ All charts exported to {OUT}/")
