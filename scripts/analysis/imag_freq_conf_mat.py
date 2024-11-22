"""Calculate confusion matrix for whether PBE and MACE both predict imaginary modes
for each material at the Gamma point.
"""

# %%
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import pymatviz as pmv
from pymatviz.enums import Key
from sklearn.metrics import confusion_matrix

import ffonons
from ffonons.enums import DB, Model

__author__ = "Janosh Riebesell"
__date__ = "2023-12-15"


# %% compute last phonon DOS peak for each model and MP
imaginary_freq_tol = 0.01
df_summary = ffonons.io.get_df_summary(
    which_db := DB.phonon_db, imaginary_freq_tol=imaginary_freq_tol
)

avail_models = {*df_summary.index.get_level_values(1)}

for models in (
    [*{*Model} - {Model.pbe}],
    (models_incl_pbe := [Model.pbe, Model.chgnet_030, Model.mace_mp0, Model.m3gnet_ms]),
):
    avail_keys = avail_models & set(models)  # df.loc doesn't handle missing keys
    df_subset = df_summary.loc[(slice(None), list(avail_keys)), :]
    print(f"{len(df_subset):,} rows for {', '.join(map(repr, avail_keys))}")


# %%
unstable_rate = (df_summary[Key.has_imag_ph_modes]).mean()
print(f"PBE unstable rate: {unstable_rate:.0%} of {which_db.label}")

unstable_rate = (df_summary[Key.has_imag_ph_gamma_modes]).mean()
print(f"PBE Γ-unstable rate: {unstable_rate:.0%} of {which_db.label}")


# %% plot confusion matrix

# get material IDs where all models have results
idx_n_avail = df_summary[Key.max_ph_freq].unstack().dropna(thresh=4).index

# whether to only use materials with predictions from all models or every material with
# predictions from the current model and PBE
enforce_same_mat_ids_across_models = True

for model in (Model.chgnet_030, Model.mace_mp0, Model.m3gnet_ms):
    for col in (Key.has_imag_ph_gamma_modes, Key.has_imag_ph_modes):
        col_names = (
            models_incl_pbe if enforce_same_mat_ids_across_models else [Key.pbe, model]
        )
        df_clean = df_summary[col].unstack(level=1)[col_names].dropna()
        y_true, y_pred = (df_clean[key] for key in (Key.pbe, model))
        conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize="all")

        label1, label2 = (
            ("Γ-Stable", "Γ-Unstable") if "gamma" in col else ("Stable", "Unstable")
        )
        annos = (
            (f"True<br>{label1}", f"False<br>{label2}"),
            (f"False<br>{label1}", f"True<br>{label2}"),
        )
        conf_mat_pct = (100 * conf_mat).astype(int).astype(str)
        annotated_vals = np.array(annos, dtype=object) + "<br>" + conf_mat_pct + "%"
        fig = ff.create_annotated_heatmap(
            z=np.rot90(conf_mat.T),
            x=(label1, label2),
            y=(label2, label1),
            annotation_text=np.rot90(annotated_vals.T),
            colorscale="blues",
            xgap=7,
            ygap=7,
        )
        # annotate accuracy, n_materials, imaginary freq. tolerance
        acc = conf_mat.diagonal().sum() / conf_mat.sum()
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=(x_anno := 0.45),
            y=(y_anno := -0.12),
            text=f"Acc={acc:.0%}, N={len(y_true)}",
            showarrow=False,
            font=dict(size=(font_size := 26)),
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=x_anno,
            y=y_anno,
            text=f"Tol={imaginary_freq_tol:.2}",
            showarrow=False,
            font=dict(size=4),
        )

        fig.layout.xaxis.title = model.label
        fig.layout.yaxis.update(title=Key.pbe.label, tickangle=-90)
        fig.layout.font.size = font_size
        fig.layout.height = fig.layout.width = 425  # force same width and height
        fig.layout.margin = dict(l=10, r=10, t=10, b=40)

        fig.layout.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        # remove border line
        fig.layout.xaxis.update(showline=False, showgrid=False)
        fig.layout.yaxis.update(showline=False, showgrid=False)

        fig.show()

        img_name = f"{col.replace('_', '-')}-{model}-confusion-matrix"
        pmv.save_fig(fig, f"{ffonons.PAPER_DIR}/{img_name}.pdf")
        pmv.save_fig(fig, f"{ffonons.PAPER_DIR}/{img_name}.png")
        pmv.save_fig(fig, f"{ffonons.PDF_FIGS}/{which_db}/{img_name}.pdf")
        pmv.save_fig(fig, f"{ffonons.PAPER_DIR}/{img_name}.svg")


# %% plot imaginary modes confusion matrix as parity plot using min. freq. across all
# k-points (with shaded regions for TP, FP, FN, TN)
y_col, color_col = Key.min_ph_freq, Key.model

df_melt = (
    df_summary.set_index(Key.formula, append=True)
    .unstack(level=1)[y_col]
    .reset_index(names=[Key.mat_id, Key.formula])
    .melt(
        id_vars=[Key.mat_id, Key.formula, Key.pbe],
        value_vars=list({*Model.val_label_dict()} & avail_models),
        var_name=color_col,
        value_name=y_col,
    )
).dropna()

fig = px.scatter(
    df_melt,
    x=Key.pbe,
    y=y_col,
    hover_data=[Key.mat_id, Key.formula],
    color=color_col,
)

imaginary_tol = 0.1
# add dashed dynamic instability decision boundary separators
fig.add_vline(x=-imaginary_tol, line=dict(width=1, dash="dash"))
fig.add_hline(y=-imaginary_tol, line=dict(width=1, dash="dash"))
pmv.powerups.add_identity_line(fig)


# add transparent rectangle with TN, TP, FN, FP labels in each quadrant
for sign_x, sign_y, label, color in (
    (-1, -1, "TP", "green"),
    (-1, 1, "FN", "orange"),
    (1, -1, "FP", "red"),
    (1, 1, "TN", "blue"),
):
    common = dict(row="all", col="all")
    fig.add_shape(
        type="rect",
        x0=-imaginary_tol,
        y0=-imaginary_tol,
        x1=10 * sign_x,
        y1=10 * sign_y,
        fillcolor=color,
        line=dict(width=0),  # remove edge line
        layer="below",
        opacity=0.1,
        **common,
    )
    fig.add_annotation(
        x=int(sign_x > 0),
        y=int(sign_y > 0),
        xshift=-15 * sign_x,
        yshift=-5 * sign_y,
        text=label,
        showarrow=False,
        font=dict(size=16, color=color),
        xref="x domain",
        yref="y domain",
        **common,
    )
min_xy = df_melt[[y_col, Key.pbe]].min().min() - 0.4
max_xy = df_melt[[y_col, Key.pbe]].max().max() + 0.4
fig.update_xaxes(range=[min_xy, max_xy], title="PBE min. freq. (THz)")
fig.update_yaxes(range=[min_xy, max_xy], title="ML min. freq. (THz)")
fig.layout.legend.update(x=0.02, y=0.9, title="")
fig.layout.margin = dict(l=10, r=10, t=10, b=10)

fig.show()
