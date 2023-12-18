"""Plot phonon DOS and band structure DFT vs ML comparison plots for all materials with
at least 2 models (where DFT also counts as a model).
"""

# %%
import os

from pymatgen.phonon.plotter import PhononBSPlotter
from pymatgen.util.string import htmlify, latexify
from pymatviz import plot_phonon_bands
from pymatviz import plot_phonon_dos as plot_phonon_dos_plotly
from pymatviz.io import save_fig
from tqdm import tqdm

from ffonons import (
    FIGS_DIR,
    SITE_FIGS,
    bs_key,
    dft_key,
    dos_key,
    formula_key,
    pretty_label_map,
)
from ffonons.io import load_pymatgen_phonon_docs
from ffonons.plots import plot_phonon_dos

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"

model1_key = "mace-y7uhwpje"
model2_key = "chgnet-v0.3.0"


# %% load docs
ph_docs, df_summary = load_pymatgen_phonon_docs(which_db := "phonon_db")
figs_out_dir = f"{FIGS_DIR}/{which_db}"

materials_with_2 = [key for key, val in ph_docs.items() if len(val) >= 2]
print(f"{len(materials_with_2)=}")
materials_with_3 = [key for key, val in ph_docs.items() if len(val) >= 3]
print(f"{len(materials_with_3)=}")


# %% redraw multi-model DOS comparison plots
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
    ax_dos = plot_phonon_dos(doses, last_peak_anno=r"${key}={last_peak:.1f}$")
    ax_dos.set_title(
        f"{mp_id} {latexify(formula, bold=True)}", fontsize=22, fontweight="bold"
    )
    save_fig(ax_dos, f"{FIGS_DIR}/{mp_id}-{formula.replace(' ', '')}/dos-all.pdf")


# %% MATPLOTLIB band structure comparison plots
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


# %% PLOTLY band structure comparison plots
for mp_id in tqdm(materials_with_2):
    band_structs = {
        pretty_label_map[key]: ph_docs[mp_id][key][bs_key]
        for key in (dft_key, model1_key)  # , model2_key)
    }
    out_path = f"{figs_out_dir}/{mp_id}-bands-pbe-vs-{model1_key}.pdf"
    if os.path.isfile(out_path):
        continue
    try:
        fig_bs = plot_phonon_bands(band_structs, line_kwds=dict(width=1.5))
    except ValueError as exc:
        print(f"{mp_id=} {exc=}")
        continue

    # turn title into link to materials project page
    formula = ph_docs[mp_id][dft_key][formula_key]
    href = f"https://legacy.materialsproject.org/materials/{mp_id}"
    title = f"{htmlify(formula)}  <a {href=}>{mp_id}</a>"
    fig_bs.layout.title = dict(text=title, x=0.5, y=0.99)
    fig_bs.layout.margin = dict(t=30, b=0, l=5, r=5)

    save_fig(fig_bs, out_path, prec=5)


# %% PLOTLY DOS comparison plots
for mp_id in tqdm(materials_with_2):
    ml_dos = ph_docs[mp_id][model1_key][dos_key]
    pbe_dos = ph_docs[mp_id][dft_key][dos_key]
    doses = {pretty_label_map[model1_key]: ml_dos, pretty_label_map[dft_key]: pbe_dos}
    img_name = f"{mp_id}-dos-pbe-vs-{model1_key}"
    out_path = f"{figs_out_dir}/{img_name}.pdf"
    # if os.path.isfile(out_path):
    #     continue
    fig_dos = plot_phonon_dos_plotly(
        doses, normalize="integral", sigma=0.03, fill="tozeroy"
    )

    # turn title into link to materials project page
    formula = ph_docs[mp_id][dft_key][formula_key]
    href = f"https://legacy.materialsproject.org/materials/{mp_id}"
    dos_mae = pbe_dos.mae(ml_dos)
    mae_text = f"<span style='font-size: 16px;'>MAE={dos_mae:.3} THz</span>"
    title = f"{htmlify(formula)}  <a {href=}>{mp_id}</a>  {mae_text}"
    fig_dos.layout.title = dict(text=title, x=0.5, y=0.97)
    fig_dos.layout.margin = dict(t=40, b=0, l=5, r=5)

    fig_dos.show()
    save_fig(fig_dos, out_path, prec=4)
    fig_dos.layout.template = "pymatviz_black"
    fig_dos.layout.paper_bgcolor = "rgba(0,0,0,0)"
    save_fig(fig_dos, f"{SITE_FIGS}/{img_name}.svelte", prec=4)
