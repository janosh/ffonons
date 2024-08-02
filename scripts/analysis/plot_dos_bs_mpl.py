"""Plot phonon DOS and band structure comparing DFT with different ML models.

Uses pymatgen DOS and band potting methods powered by matplotlib.
"""

# %%
import os

import pandas as pd
import pymatviz as pmv
from pymatgen.phonon import PhononBSPlotter
from pymatgen.util.string import latexify
from pymatviz.enums import Key
from tqdm import tqdm

from ffonons import PDF_FIGS
from ffonons.enums import DB, Model
from ffonons.io import get_df_summary, load_pymatgen_phonon_docs
from ffonons.plots import plot_phonon_dos_mpl

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"

model1 = Model.mace_mp
model2 = Model.chgnet_030


# %% load summary data
df_summary = get_df_summary(which_db := DB.one_off)

os.makedirs(FIGS_DIR := f"{PDF_FIGS}/{which_db}", exist_ok=True)

idx_n_avail: dict[int, pd.Index] = {}

for idx in range(1, 5):
    idx_n_avail[idx] = df_summary[Key.max_ph_freq].unstack().dropna(thresh=idx).index
    n_avail = len(idx_n_avail[idx])
    print(f"{n_avail:,} materials with results from at least {idx} models (incl. DFT)")


# %% load docs (takes a minute)
ph_docs = load_pymatgen_phonon_docs(which_db)


# %% matplotlib DOS
model1 = Model.mace_mp
model2 = Model.chgnet_030

for mp_id in idx_n_avail[2]:
    model1 = ph_docs[mp_id][model1]
    # model2 = ph_docs[mp_id][model2]
    formula = model1[Key.formula]

    # now the same for DOS
    doses = {
        model1.label: model1[Key.ph_dos],
        # pretty_label_map[model2]: model2[Key.ph_dos],
        Key.pbe.label: ph_docs[mp_id][Key.pbe][Key.ph_dos],
    }
    ax_dos = plot_phonon_dos_mpl(doses, last_peak_anno=r"${key}={last_peak:.1f}$")
    ax_dos.set_title(
        f"{mp_id} {latexify(formula, bold=True)}", fontsize=22, fontweight="bold"
    )
    pmv.save_fig(ax_dos, f"{PDF_FIGS}/{mp_id}-{formula.replace(' ', '')}/dos-all.pdf")


# %% matplotlib bands
for mp_id in tqdm(idx_n_avail[1]):
    # pbe_bands = ph_docs[mp_id][Key.pbe][Key.ph_band_structure]
    ml1_bands = getattr(ph_docs[mp_id][model1], Key.ph_band_structure)
    # ml2_bands = ph_docs[mp_id][model2][Key.ph_band_structure]

    bands_fig_path = f"{FIGS_DIR}/{mp_id}-bands-pbe-vs-{model1}.pdf"

    formula = ph_docs[mp_id][model1].structure.formula
    pbe_bs_plotter = PhononBSPlotter(ml1_bands, label=Key.pbe.label)
    ml_bs_plotter = PhononBSPlotter(ml1_bands, label=model1.label)

    ax_bands = pbe_bs_plotter.plot_compare(
        ml_bs_plotter,
        linewidth=2,
        on_incompatible="warn",
        other_kwargs=dict(linestyle="dashed"),
    )
    if not ax_bands:
        continue
    ax_bands.set_title(f"{latexify(formula)} {mp_id}", fontsize=24)
    ax_bands.figure.subplots_adjust(top=0.95)  # make room for title
    # show
    ax_bands.figure.show()
