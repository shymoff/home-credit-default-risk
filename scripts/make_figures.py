"""Generate the README figures (English labels) from artifacts saved by modeling.ipynb.

Usage (from the repository root, after running the notebooks):
    python scripts/make_figures.py
"""
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

DATA = Path('data/processed')
OUT = Path('reports/figures')
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ['Ensemble', 'LightGBM', 'XGBoost', 'RandomForest']  # legend order
DRAW_ORDER = MODELS[::-1]  # draw the ensemble last so it is not hidden under LightGBM
COLORS = {'LightGBM': '#2a78d6', 'XGBoost': '#eb6834', 'RandomForest': '#1baf7a', 'Ensemble': '#e87ba4'}
INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'
plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 150, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
    'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e', 'axes.titlecolor': INK,
    'axes.titleweight': 'bold', 'axes.titlesize': 11, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'legend.frameon': False, 'font.size': 9,
})

df = pd.read_parquet(DATA / 'holdout_predictions.parquet')
y = df['TARGET'].values
pred = {m: df[f'pred_{m}'].values for m in MODELS}


def legend(ax, **kw):
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for l in sorted(labels, key=lambda l: next(
        (i for i, m in enumerate(MODELS) if l.startswith(m)), len(MODELS)))]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], **kw)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches='tight')
    plt.close(fig)
    print(OUT / name)


# 1. ROC and Precision-Recall curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for m in DRAW_ORDER:
    fpr, tpr, _ = roc_curve(y, pred[m])
    axes[0].plot(fpr, tpr, color=COLORS[m], lw=2, label=f'{m}  (AUC {roc_auc_score(y, pred[m]):.4f})')
    prec, rec, _ = precision_recall_curve(y, pred[m])
    axes[1].plot(rec, prec, color=COLORS[m], lw=2, label=f'{m}  (AP {average_precision_score(y, pred[m]):.4f})')
axes[0].plot([0, 1], [0, 1], color=MUTED, ls='--', lw=1, label='random (0.5)')
axes[0].set(xlabel='False positive rate', ylabel='True positive rate (recall)', title='ROC curve (holdout)')
legend(axes[0], loc='lower right')
axes[1].axhline(y.mean(), color=MUTED, ls='--', lw=1, label=f'random ({y.mean():.3f})')
axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–Recall curve (holdout)')
legend(axes[1], loc='upper right')
save(fig, 'roc_pr.png')

# 2. Cumulative gains and calibration
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
frac = np.linspace(0, 1, 201)
for m in DRAW_ORDER:
    order = np.argsort(-pred[m])
    cum = np.concatenate([[0], np.cumsum(y[order])]) / y.sum()
    axes[0].plot(frac, np.interp(frac, np.linspace(0, 1, len(cum)), cum), color=COLORS[m], lw=2, label=m)
    pt, pp = calibration_curve(y, pred[m], n_bins=15, strategy='quantile')
    axes[1].plot(pp, pt, color=COLORS[m], lw=2, marker='o', ms=4, label=m)
axes[0].plot([0, 1], [0, 1], color=MUTED, ls='--', lw=1, label='random')
axes[0].set(xlabel='Share of applications reviewed (highest risk first)', ylabel='Share of defaults captured',
            title='Cumulative gains (holdout)')
axes[0].xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
legend(axes[0], loc='lower right')
lim = 0.4
axes[1].plot([0, lim], [0, lim], color=MUTED, ls='--', lw=1, label='perfect calibration')
axes[1].set(xlim=(0, lim), ylim=(0, lim), xlabel='Mean predicted probability', ylabel='Observed default rate',
            title='Calibration (holdout, 15 quantile bins)')
legend(axes[1], loc='upper left')
save(fig, 'gains_calibration.png')

# 3. LightGBM feature importance (share of total gain)
booster = lgb.Booster(model_file=str(DATA / 'model_lightgbm.txt'))
imp = pd.Series(booster.feature_importance('gain'), index=booster.feature_name())
imp = (imp / imp.sum()).sort_values(ascending=False).head(20)[::-1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp.index, imp.values, color=COLORS['LightGBM'], height=0.7)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
ax.set(xlabel='Share of total gain', title='LightGBM – top 20 features')
ax.grid(axis='y', visible=False); ax.set_axisbelow(True)
save(fig, 'feature_importance.png')
