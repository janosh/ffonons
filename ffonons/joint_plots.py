# %%
from pymatgen.phonon.plotter import PhononBSPlotter
from pymatviz import plot_band_structure
from pymatviz.io import save_fig

from ffonons import FIGS_DIR, bs_key, dos_key, formula_key
from ffonons.io import load_pymatgen_phonon_docs
from ffonons.plots import plot_phonon_dos

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"


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
dft_key = "pbe"

# for mp_id in materials_with_3:
for mp_id in materials_with_2:
    mace = phonon_docs[mp_id][model1_key]
    # chgnet = phonon_docs[mp_id][model2_key]
    formula = mace[formula_key]

    # now the same for DOS
    doses = {
        # "CHGnet": chgnet[dos_key],
        "MACE": mace[dos_key],
        dft_key.upper(): phonon_docs[mp_id][dft_key][dos_key],
    }
    ax = plot_phonon_dos(doses, last_peak_anno=r"${key}={last_peak:.1f}$")
    ax.set_title(f"{mp_id} {formula}", fontsize=22, fontweight="bold")
    # save_fig(ax, f"{FIGS_DIR}/{mp_id}-{formula.replace(' ', '')}/dos-all.pdf")

    # fig_dos = plot_dos(doses)
    # fig_dos.layout.title = f"{mp_id} {formula}"
    # fig_dos.show()

    break


# %% redraw matplotlib band structure comparison plots
model1_key = "mace-y7uhwpje"
model2_key = "chgnet-v0.3.0"
dft_key = "pbe"

for mp_id in materials_with_2:
    pbe_bands = phonon_docs[mp_id][dft_key][bs_key]
    ml1_bands = phonon_docs[mp_id][model1_key][bs_key]
    # ml2_bands = phonon_docs[mp_id][model2_key][bs_key]

    bands_fig_path = f"{figs_out_dir}/{mp_id}-bands-pbe-vs-{model1_key}.pdf"

    formula = phonon_docs[mp_id][dft_key][formula_key]
    pbe_bs_plotter = PhononBSPlotter(pbe_bands, label="PBE")
    ml_bs_plotter = PhononBSPlotter(ml1_bands, label=model1_key)
    ax_bands_compare = pbe_bs_plotter.plot_compare(ml_bs_plotter, linewidth=2)
    ax_bands_compare.set_title(f"{formula} {mp_id}", fontsize=24)
    ax_bands_compare.figure.subplots_adjust(top=0.95)  # make room for title
    save_fig(ax_bands_compare, bands_fig_path)

    band_structs = {
        dft_key: pbe_bands,
        model1_key: ml1_bands,
        # model2_key: ml2_bands,
    }
    fig = plot_band_structure(band_structs)

    # turn title into link to materials project page
    href = f"https://legacy.materialsproject.org/materials/{mp_id}"
    title = f"<a {href=}>{mp_id}</a> {formula}"
    fig.layout.title = dict(text=title, x=0.5)
    fig.show()
