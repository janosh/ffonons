# %%
from pymatgen.phonon.plotter import PhononBSPlotter
from pymatgen.util.string import htmlify, latexify
from pymatviz import plot_band_structure
from pymatviz.io import save_fig
from tqdm import tqdm

from ffonons import FIGS_DIR, bs_key, dft_key, dos_key, formula_key, pretty_label_map
from ffonons.io import load_pymatgen_phonon_docs
from ffonons.plots import plot_phonon_dos

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"

model1_key = "mace-y7uhwpje"
model2_key = "chgnet-v0.3.0"


# %% load docs
phonon_docs = load_pymatgen_phonon_docs(which_db := "phonon_db")
figs_out_dir = f"{FIGS_DIR}/{which_db}"

materials_with_2 = [key for key, val in phonon_docs.items() if len(val) >= 2]
print(f"{len(materials_with_2)=}")
materials_with_3 = [key for key, val in phonon_docs.items() if len(val) >= 3]
print(f"{len(materials_with_3)=}")


# %% redraw multi-model DOS comparison plots
model1_key = "mace-y7uhwpje"
model2_key = "chgnet-v0.3.0"

# for mp_id in materials_with_3:
for mp_id in materials_with_2:
    model1 = phonon_docs[mp_id][model1_key]
    # model2 = phonon_docs[mp_id][model2_key]
    formula = model1[formula_key]

    # now the same for DOS
    doses = {
        pretty_label_map[model1_key]: model1[dos_key],
        # pretty_label_map[model2_key]: model2[dos_key],
        pretty_label_map[dft_key]: phonon_docs[mp_id][dft_key][dos_key],
    }
    ax_dos = plot_phonon_dos(doses, last_peak_anno=r"${key}={last_peak:.1f}$")
    ax_dos.set_title(
        f"{mp_id} {latexify(formula, bold=True)}", fontsize=22, fontweight="bold"
    )
    # save_fig(ax_dos, f"{FIGS_DIR}/{mp_id}-{formula.replace(' ', '')}/dos-all.pdf")

    # fig_dos = plot_dos(doses)
    # fig_dos.layout.title = f"{mp_id} {htmlify(formula)}"
    # fig_dos.show()


# %% MATPLOTLIB band structure comparison plots
for mp_id in tqdm(materials_with_2):
    pbe_bands = phonon_docs[mp_id][dft_key][bs_key]
    ml1_bands = phonon_docs[mp_id][model1_key][bs_key]
    # ml2_bands = phonon_docs[mp_id][model2_key][bs_key]

    bands_fig_path = f"{figs_out_dir}/{mp_id}-bands-pbe-vs-{model1_key}.pdf"

    formula = phonon_docs[mp_id][dft_key][formula_key]
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
        pretty_label_map[key]: phonon_docs[mp_id][key][bs_key]
        for key in (dft_key, model1_key)  # , model2_key)
    }
    fig = plot_band_structure(band_structs)
    formula = phonon_docs[mp_id][dft_key][formula_key]

    # turn title into link to materials project page
    href = f"https://legacy.materialsproject.org/materials/{mp_id}"
    title = f"<a {href=}>{mp_id}</a>  {htmlify(formula)}"
    fig.layout.title = dict(text=title, x=0.5)
    fig.show()
