"""Analyze phonon mode softness from too-low forces compared to DFT training data."""

# %%
import os

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from pymatviz.io import save_fig

from ffonons import PAPER_DIR, PDF_FIGS, SOFT_PES_DIR
from ffonons.enums import DB, Key, Model
from ffonons.io import get_df_summary

__author__ = "Janosh Riebesell"
__date__ = "2024-02-22"


# %% load summary data
df_summary = get_df_summary(which_db := DB.phonon_db, refresh_cache=False)

os.makedirs(FIG_DIR := f"{PDF_FIGS}/{which_db}", exist_ok=True)
os.makedirs(FIG_DIR := SOFT_PES_DIR, exist_ok=True)

df_ph_freq = df_summary[Key.max_freq].unstack().dropna()

# compute diff or ratio of ML and DFT max phonon frequencies
df_ph_freq_ml_vs_pbe = df_ph_freq.div(df_ph_freq[Key.pbe], axis=0)

df_ph_freq_ml_vs_pbe = df_ph_freq_ml_vs_pbe.drop(columns=Key.pbe)
df_ph_freq_ml_vs_pbe = df_ph_freq_ml_vs_pbe.rename(columns=Model.val_label_dict())

models_in_asc_mean = df_ph_freq_ml_vs_pbe.mean().sort_values().index


# %% matplotlib version for visual consistency with PES softening paper 2024-02-22
ax = sns.violinplot(
    df_ph_freq_ml_vs_pbe[models_in_asc_mean],
    inner="box",
    palette=Model.label_desc_dict(),
    saturation=0.65,
    linewidth=0,
)
ax.axhline(1, linestyle=":", color="gray", zorder=0)

for anno, y_pos in (
    ("hardening\n(overstiff phonons)", 0.93),
    ("softening\n(understiff phonons)", 0.06),
):
    ax.text(
        0.5, y_pos, anno, ha="center", va="center", fontsize=16, transform=ax.transAxes
    )

# annotate fraction of materials with softening and hardening for each model
for idx, col in enumerate(models_in_asc_mean):
    n_soft = (df_ph_freq_ml_vs_pbe[col] < 1).sum()
    n_hard = (df_ph_freq_ml_vs_pbe[col] > 1).sum()
    n_total = len(df_ph_freq_ml_vs_pbe)
    if n_total != n_soft + n_hard:
        raise ValueError(f"{n_total=} != {n_soft=} + {n_hard=}")
    color = Model.label_desc_dict()[col]
    anno = f"{n_hard / n_total:.0%}\n\n\n\n\n\n\n\n{n_soft / n_total:.0%}"
    ax.text(idx - 0.2, 0.85, anno, ha="center", va="center", fontsize=18, color=color)

    # mean = df_max_freq_rel[col_name].mean()
    # ax.text(idx, 1.4, f"mean\n{mean:.2f}", ha="center", va="center", fontsize=14)

ax.set_xticklabels(
    [tick.get_text().split(" ")[0].split("-")[0] for tick in ax.get_xticklabels()]
)

y_range = 0.7
y_label = r"$\Omega_{\text{max}}^{\text{ML}} \;/\; \Omega_{\text{max}}^{\text{DFT}}$"
ax.set(xlabel="", ylabel=y_label, ylim=[1 - y_range, 1 + y_range])

save_fig(ax, f"{FIG_DIR}/max-freq-rel-violin.pdf")
plt.show()


# %% violin plot of max_freq_ml / max_freq_dft and max_freq_ml - max_freq_dft
for key, op in ((Key.max_freq, "div"), (Key.max_freq, "sub"), (Key.min_freq, "sub")):
    print(f"{key=!s} {op=}")
    df_ph_freq = df_summary[key].unstack().dropna()
    # compute diff or ratio of ML and DFT max phonon frequencies
    df_ph_freq_ml_vs_pbe = getattr(df_ph_freq, op)(df_ph_freq[Key.pbe], axis=0)

    df_ph_freq_ml_vs_pbe = df_ph_freq_ml_vs_pbe.drop(columns=Key.pbe)
    df_ph_freq_ml_vs_pbe = df_ph_freq_ml_vs_pbe.rename(columns=Model.val_label_dict())

    fig = go.Figure()

    soft_boundary = {"div": 1, "sub": 0}[op]

    fig.add_hline(y=soft_boundary, line=dict(dash="dot", width=2))
    if op == "div":
        # fig.add_hrect(y0=1.2, y1=10, fillcolor="red", opacity=0.1, layer="below")
        # fig.add_hrect(y0=0, y1=0.8, fillcolor="red", opacity=0.1, layer="below")

        # shade region of acceptably low error green
        y_low_err = 0.1
        y0, y1 = soft_boundary - y_low_err, soft_boundary + y_low_err
        fig.add_hrect(y0=y0, y1=y1, fillcolor="green", opacity=0.1, layer="below")

        y_range = 0.8
        title = f"(ML / DFT) {key.label.replace('(THz)', '')}"
        fig.layout.yaxis.update(title=title, range=[1 - y_range, 1 + y_range])
    elif op == "sub":
        title = f"(ML - DFT) {key.label}"
        fig.layout.yaxis.update(title=title, title_standoff=5)
        if key == Key.min_freq:
            fig.layout.yaxis.update(range=[-5, 5])
        # shade region of acceptably low error green
        y_low_err = 1.5 if key == Key.max_freq else 0.5
        y0, y1 = soft_boundary - y_low_err, soft_boundary + y_low_err
        fig.add_hrect(y0=y0, y1=y1, fillcolor="green", opacity=0.1, layer="below")
    else:
        raise ValueError(f"Unknown {op=}")

    for anno, y_pos in (("phonons<br>too soft", y0), ("phonons<br>too stiff", y1)):
        yshift = (-1 if "soft" in anno else 1) * 10
        fig.add_annotation(
            x=1,
            y=y_pos,
            text=anno,
            showarrow=False,
            yshift=yshift,
            xanchor="right",
            xref="paper",
            yanchor="top" if "soft" in anno else "bottom",
        )

    # main loop to add violins
    for idx, (col, ys) in enumerate(df_ph_freq_ml_vs_pbe.items()):
        fig.add_violin(
            y=ys,
            name=col.split(" ")[0].split("-")[0],
            box=dict(visible=True),
            showlegend=False,
            marker_color=Model.label_desc_dict()[col],
            width=1.3,
            side="positive",
            points=False,
        )
        # mean = ys.mean()
        # fig.add_annotation(  # add mean as annotation
        #     x=col_name.split(" ")[0].split("-")[0],
        #     y=1.4,
        #     text=f"mean<br>{mean:.2f}",
        #     showarrow=False,
        #     font=dict(size=10),
        # )
        # annotate fraction of materials with softening and hardening for each model
        n_soft = (df_ph_freq_ml_vs_pbe[col] < y0).sum()
        n_hard = (df_ph_freq_ml_vs_pbe[col] > y1).sum()
        n_low_err = ((ys > y0) & (ys < y1)).sum()

        n_total = len(df_ph_freq_ml_vs_pbe)
        color = Model.label_desc_dict()[col]
        for val, y_pos in ((n_soft, y0), (n_hard, y1), (n_low_err, (y0 + y1) / 2)):
            yshift = {y0: -1, y1: 1}.get(y_pos, 0) * 20
            fig.add_annotation(
                x=idx - 0.01,
                y=y_pos,
                text=f"{val / n_total * 100:.0f}",
                showarrow=False,
                font=dict(size=15, color=color),
                yshift=yshift,
                xanchor="right",
                textangle=-90,
            )

    # reduce y-axis range by 10% top and bottom
    ymin, ymax = fig.full_figure_for_development(warn=False).layout.yaxis.range
    fig.layout.yaxis.update(range=(0.9 * ymin, 0.9 * ymax))

    fig.layout.update(width=500, height=350)
    fig.layout.margin = dict(l=5, r=5, t=5, b=5)
    fig.layout.update(font_size=16)
    fig.show()
    img_name = f"violin-ph-{key.replace('_', '-').replace('-thz', '')}-{op}"
    save_fig(fig, f"{FIG_DIR}/{img_name}.pdf")
    save_fig(fig, f"{PAPER_DIR}/{img_name}.svg")
