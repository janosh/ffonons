"""Plot phonon DOS and band structure DFT vs ML comparison plots for all materials with
at least 2 models (where DFT also counts as a model).
"""

# %%
import os

from pymatgen.phonon.plotter import PhononBSPlotter
from pymatgen.util.string import latexify
from pymatviz import plot_phonon_bands, plot_phonon_bands_and_dos
from pymatviz.io import save_fig
from tqdm import tqdm

from ffonons import FIGS_DIR, SITE_FIGS, bs_key, dft_key, dos_key, formula_key
from ffonons.io import load_pymatgen_phonon_docs
from ffonons.plots import plot_phonon_dos_mpl, plotly_title, pretty_label_map

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"

model1_key = "mace-y7uhwpje"
model2_key = "chgnet-v0.3.0"


# %% load docs
ph_docs, df_summary = load_pymatgen_phonon_docs(which_db := "phonon-db")
os.makedirs(figs_out_dir := f"{FIGS_DIR}/{which_db}", exist_ok=True)

materials_with_2 = [key for key, val in ph_docs.items() if len(val) >= 2]
print(f"{len(materials_with_2)=}")
materials_with_3 = [key for key, val in ph_docs.items() if len(val) >= 3]
print(f"{len(materials_with_3)=}")


# %% matplotlib DOS
model1_key = "mace-y7uhwpje"
model2_key = "chgnet-v0.3.0"

# for mp_id in materials_with_3:
for mp_id in materials_with_2:
    mp_id = "mp-21836"
    model1 = ph_docs[mp_id][model1_key]
    # model2 = ph_docs[mp_id][model2_key]
    formula = model1[formula_key]

    # now the same for DOS
    doses = {
        pretty_label_map[model1_key]: model1[dos_key],
        # pretty_label_map[model2_key]: model2[dos_key],
        pretty_label_map[dft_key]: ph_docs[mp_id][dft_key][dos_key],
    }
    ax_dos = plot_phonon_dos_mpl(doses, last_peak_anno=r"${key}={last_peak:.1f}$")
    ax_dos.set_title(
        f"{mp_id} {latexify(formula, bold=True)}", fontsize=22, fontweight="bold"
    )
    save_fig(ax_dos, f"{FIGS_DIR}/{mp_id}-{formula.replace(' ', '')}/dos-all.pdf")


# %% matplotlib bands
for mp_id in tqdm(materials_with_2):
    pbe_bands = ph_docs[mp_id][dft_key][bs_key]
    ml1_bands = ph_docs[mp_id][model1_key][bs_key]
    # ml2_bands = ph_docs[mp_id][model2_key][bs_key]

    bands_fig_path = f"{figs_out_dir}/{mp_id}-bands-pbe-vs-{model1_key}.pdf"

    formula = ph_docs[mp_id][dft_key][formula_key]
    pbe_bs_plotter = PhononBSPlotter(pbe_bands, label=pretty_label_map[dft_key])
    ml_bs_plotter = PhononBSPlotter(ml1_bands, label=pretty_label_map[model1_key])

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
    save_fig(ax_bands, bands_fig_path)


# %% plotly bands
for mp_id in tqdm(materials_with_2):
    bs_pbe = ph_docs[mp_id][dft_key][bs_key]
    dft_label = pretty_label_map[dft_key]

    bs_ml = ph_docs[mp_id][model1_key][bs_key]
    ml_label = pretty_label_map[model1_key]

    band_structs = {dft_label: bs_pbe, ml_label: bs_ml}
    out_path = f"{figs_out_dir}/{mp_id}-bands-pbe-vs-{model1_key}.pdf"
    if os.path.isfile(out_path):
        continue
    try:
        fig_bs = plot_phonon_bands(band_structs, line_kwds=dict(width=1.5))
    except ValueError as exc:
        print(f"{mp_id=} {exc=}")
        continue

    formula = ph_docs[mp_id][dft_key][formula_key]
    title = plotly_title(formula, mp_id)
    fig_bs.layout.title = dict(text=title, x=0.5, y=0.96)
    fig_bs.layout.margin = dict(t=65, b=0, l=5, r=5)

    # fig_bs.show()

    save_fig(fig_bs, out_path, prec=5)


# %% plotly bands+DOS
for mp_id in tqdm(materials_with_3):
    keys = sorted(ph_docs[mp_id], reverse=True)
    bands_dict = {
        pretty_label_map.get(key, key): getattr(ph_docs[mp_id], bs_key) for key in keys
    }
    dos_dict = {
        pretty_label_map.get(key, key): getattr(ph_docs[mp_id], dos_key) for key in keys
    }
    img_name = f"{mp_id}-bs-dos-{'-vs-'.join(keys)}"
    out_path = f"{figs_out_dir}/{img_name}.pdf"
    # if os.path.isfile(out_path):
    #     continue
    try:
        fig_bs_dos = plot_phonon_bands_and_dos(bands_dict, dos_dict)
    except ValueError as exc:
        print(f"{mp_id=} {exc=}")
        continue

    formula = next(iter(ph_docs[mp_id].values())).structure.formula
    fig_bs_dos.layout.title = dict(text=plotly_title(formula, mp_id), x=0.5, y=0.97)
    fig_bs_dos.layout.margin = dict(t=40, b=0, l=5, r=5)
    fig_bs_dos.layout.legend.update(x=1, y=1.07, xanchor="right")

    fig_bs_dos.show()
    save_fig(fig_bs_dos, out_path, prec=4)
    fig_bs_dos.layout.update(template="pymatviz_black", paper_bgcolor="rgba(0,0,0,0)")
    save_fig(fig_bs_dos, f"{SITE_FIGS}/{img_name}.svelte", prec=4)
