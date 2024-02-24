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

os.makedirs(figs_out_dir := f"{PDF_FIGS}/{which_db}", exist_ok=True)
os.makedirs(figs_out_dir := SOFT_PES_DIR, exist_ok=True)

df_max_freq = df_summary[Key.max_freq].unstack().dropna()
df_max_freq_rel = df_max_freq.div(df_max_freq[Key.pbe], axis=0).drop(columns=Key.pbe)
df_max_freq_rel = df_max_freq_rel.rename(columns=Model.val_label_dict())

models_in_asc_mean = df_max_freq_rel.mean().sort_values().index


# %% violin plot of max_freq_ml / max_freq_dft
fig = go.Figure()

for idx, (col, ys) in enumerate(df_max_freq_rel.items()):
    fig.add_violin(
        y=ys,
        name=col.split(" ")[0].split("-")[0],
        box=dict(visible=True),
        showlegend=False,
        marker_color=Model.label_desc_dict()[col],
        width=0.8,
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
    n_soft = (df_max_freq_rel[col] < 1).sum()
    n_hard = (df_max_freq_rel[col] > 1).sum()
    n_total = len(df_max_freq_rel)
    assert n_total == n_soft + n_hard
    color = Model.label_desc_dict()[col]
    anno = f"{n_hard/n_total:.0%}{n_soft/n_total:.0%}"
    # get max of values to place annotation
    for val, y_pos in ((n_hard, 1.1), (n_soft, 0.5)):
        fig.add_annotation(
            x=idx + 0.15,
            y=y_pos,
            text=f"<b>{val/n_total:.0%}</b>",
            showarrow=False,
            font=dict(size=14, color=color),
        )

for anno, y_pos in (
    ("hardening<br>(overstiff phonons)", 1),
    ("softening<br>(understiff phonons)", 0),
):
    fig.add_annotation(
        x=0.5, y=y_pos, text=anno, showarrow=False, xref="paper", yref="paper"
    )

# fig.add_hrect(y0=1.2, y1=10, fillcolor="red", opacity=0.1, layer="below")
# fig.add_hrect(y0=0, y1=0.8, fillcolor="red", opacity=0.1, layer="below")
fig.add_hrect(y0=0.9, y1=1.1, fillcolor="green", opacity=0.1, layer="below")


fig.add_hline(y=1, line_dash="dot", line_color="gray", layer="below")


# fig.layout.update(width=350, height=350)
fig.layout.margin = dict(l=5, r=5, t=5, b=5)
y_range = 0.8
fig.layout.yaxis.update(
    title=Key.max_freq_rel.label, range=[0.8 - y_range, 0.8 + y_range]
)
fig.layout.update(font_size=16)
fig.show()
save_fig(fig, f"{figs_out_dir}/max-freq-rel-violin.pdf")
save_fig(fig, f"{PAPER_DIR}/max-freq-rel-violin.svg")


# %% matplotlib version for consistency with PES softening paper 2024-02-22
ax = sns.violinplot(
    df_max_freq_rel[models_in_asc_mean],
    inner="box",
    palette=Model.label_desc_dict(),
    saturation=0.65,
    linewidth=0,
)
ax.axhline(1, linestyle=":", color="gray", zorder=0)

for anno, y_pos in (
    ("hardening\n(overstiff phonons)", 0.95),
    ("softening\n(understiff phonons)", 0.05),
):
    ax.text(
        0.5, y_pos, anno, ha="center", va="center", fontsize=14, transform=ax.transAxes
    )

# annotate fraction of materials with softening and hardening for each model
for idx, col in enumerate(models_in_asc_mean):
    n_soft = (df_max_freq_rel[col] < 1).sum()
    n_hard = (df_max_freq_rel[col] > 1).sum()
    n_total = len(df_max_freq_rel)
    assert n_total == n_soft + n_hard
    color = Model.label_desc_dict()[col]
    anno = f"{n_hard/n_total:.0%}\n\n\n\n\n\n\n\n{n_soft/n_total:.0%}"
    ax.text(idx - 0.2, 0.8, anno, ha="center", va="center", fontsize=14, color=color)

    # mean = df_max_freq_rel[col_name].mean()
    # ax.text(idx, 1.4, f"mean\n{mean:.2f}", ha="center", va="center", fontsize=10)

ax.set_xticklabels(
    [tick.get_text().split(" ")[0].split("-")[0] for tick in ax.get_xticklabels()]
)

y_range = 0.7
y_label = r"$\Omega_{\text{max}}^{\text{ML}} \;/\; \Omega_{\text{max}}^{\text{DFT}}$"
ax.set(xlabel="", ylabel=y_label, ylim=[1 - y_range, 1 + y_range])

# plt.savefig(f"{figs_out_dir}/max-freq-rel-violin.pdf")
plt.show()
