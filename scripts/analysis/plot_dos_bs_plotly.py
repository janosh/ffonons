"""Plot phonon DOS and band structure comparing DFT with different ML models.

Uses pymatviz DOS and band potting functions powered by Plotly.
"""

# %%
import os

import pandas as pd
import pymatviz as pmv
from pymatgen.phonon import PhononBandStructureSymmLine, PhononDos
from pymatviz.enums import Key
from tqdm import tqdm

import ffonons
from ffonons import PDF_FIGS, SITE_FIGS, SOFT_PES_DIR
from ffonons.enums import DB, Model
from ffonons.plots import plotly_title, pretty_labels

__author__ = "Janosh Riebesell"
__date__ = "2024-02-22"


# %% load summary data
df_summary = ffonons.io.get_df_summary(which_db := DB.phonon_db)

os.makedirs(FIGS_DIR := SOFT_PES_DIR, exist_ok=True)
os.makedirs(FIGS_DIR := f"{PDF_FIGS}/{which_db}", exist_ok=True)

idx_n_avail: dict[int, pd.Index] = {}

for idx in range(1, 6):
    idx_n_avail[idx] = df_summary[Key.max_ph_freq].unstack().dropna(thresh=idx).index
    n_avail = len(idx_n_avail[idx])
    print(f"{n_avail:,} materials with results from at least {idx} models (incl. DFT)")


# %% get material with n_sites < 10 and most underpredicted max freq

# get index sorted by most underpredicted max freq according to CHGNet compared to DFT
most_underpred = (
    df_summary[Key.max_ph_freq].xs(Model.chgnet_030, level=1)
    - df_summary[Key.max_ph_freq].xs(Key.pbe, level=1)
).sort_values()

# get intersection with materials with less than 10 sites
most_underpred = most_underpred.index.intersection(idx_n_avail[3])

df_summary.loc[most_underpred].loc[most_underpred].query(f"{Key.n_sites} < 6")


material_ids_to_load = {
    *df_summary.loc[idx_n_avail[4]]
    .query(f"{Key.n_sites} <= 4")
    .index.get_level_values(0)
}


# %% load docs (takes a minute)
load_all_docs = False
ph_docs = ffonons.io.load_pymatgen_phonon_docs(
    which_db, materials_ids=() if load_all_docs else material_ids_to_load
)


# %%
model = Model.m3gnet_ms

for mp_id in tqdm(idx_n_avail[2]):
    bs_pbe = getattr(ph_docs[mp_id][Key.pbe], Key.ph_band_structure)
    bs_ml = getattr(ph_docs[mp_id][model], Key.ph_band_structure)

    band_structs = {Key.pbe.label: bs_pbe, model.label: bs_ml}
    out_path = f"{FIGS_DIR}/{mp_id}-bands-pbe-vs-{model}.pdf"
    if os.path.isfile(out_path):
        continue
    try:
        fig_bs = pmv.phonon_bands(band_structs, line_kwds=dict(width=1.5))
    except ValueError as exc:
        print(f"{mp_id=} {exc=}")
        continue

    formula = ph_docs[mp_id][Key.pbe].structure.formula
    title = plotly_title(formula, mp_id)
    fig_bs.layout.title = dict(text=title, x=0.5, y=0.96)
    fig_bs.layout.margin = dict(t=65, b=0, l=5, r=5)

    fig_bs.show()
    pmv.save_fig(fig_bs, out_path, prec=5)


# %% plotly bands+DOS and similarity heatmaps
for mp_id in tqdm(idx_n_avail[4]):  # Use materials with all 4 models available
    ph_doc = ffonons.io.load_pymatgen_phonon_docs(which_db, materials_ids=[mp_id])[
        mp_id
    ]

    keys = sorted(ph_doc, reverse=True)
    bands_dict: dict[str, PhononBandStructureSymmLine] = {
        pretty_labels.get(key, key): getattr(ph_doc[key], Key.ph_band_structure)
        for key in keys
    }
    dos_dict: dict[str, PhononDos] = {
        pretty_labels.get(key, key): getattr(ph_doc[key], Key.ph_dos) for key in keys
    }
    img_name = f"{mp_id}-bs-dos-{'-vs-'.join(keys)}"
    out_path = f"{FIGS_DIR}/{img_name}.pdf"
    # if os.path.isfile(out_path):
    #     continue
    color_map = {
        model.label: {"line_color": clr}
        for model, clr in (
            (Key.pbe, "red"),
            (Model.mace_mp0, "green"),
            (Model.chgnet_030, "orange"),
            (Model.m3gnet_ms, "blue"),
        )
    }
    try:
        fig_bs_dos = pmv.phonon_bands_and_dos(
            bands_dict,
            dos_dict,
            all_line_kwargs=dict(line_width=2),
            per_line_kwargs=color_map,
            bands_kwargs={
                # "branches": ("GAMMA-Z", "Z-D", "D-B", "B-GAMMA"),
                # "branch_mode": "intersect",
            },
        )
    except ValueError as exc:
        print(f"{mp_id=} {exc=}")
        continue

    # Remap legend labels
    for trace in fig_bs_dos.data:
        trace.name = {
            "PBE": "DFT",
            Model.mace_mp0: Model.mace_mp0.label,
            Model.chgnet_030: Model.chgnet_030.label,
            Model.m3gnet_ms: Model.m3gnet_ms.label,
        }.get(trace.name, trace.name)

    formula = next(iter(ph_doc.values())).structure.formula
    # fig_bs_dos.layout.title = dict(text=plotly_title(formula, mp_id), x=0.5, y=0.97)
    # fig_bs_dos.layout.margin = dict(t=40, b=0, l=5, r=5)
    fig_bs_dos.layout.margin = dict(t=5, b=0, l=5, r=5)
    legend = dict(
        x=0.5, y=1.1, xanchor="center", itemsizing="constant", bgcolor="rgba(0,0,0,0)"
    )
    # legend = dict(x=1, y=1, xanchor="right", orientation="v", itemsizing="constant")
    fig_bs_dos.layout.legend.update(**legend)
    # fig_bs_dos.layout.xaxis.update(title_standoff=0)
    # fig_bs_dos.layout.xaxis2.update(title_standoff=0)

    fig_bs_dos.show()
    height = 400
    pmv.save_fig(fig_bs_dos, out_path, prec=4, height=height, width=1.3 * height)
    fig_bs_dos.layout.update(template="pymatviz_dark", paper_bgcolor="rgba(0,0,0,0)")
    pmv.save_fig(fig_bs_dos, f"{SITE_FIGS}/{img_name}.svelte", prec=4)
