# %%
from pymatviz import plot_band_structure, plot_dos
from pymatviz.io import save_fig

from ffonons import FIGS_DIR, bs_key, dos_key, formula_key
from ffonons.load_all_docs import all_docs
from ffonons.plots import plot_phonon_dos

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"


materials_with_2 = [key for key, val in all_docs.items() if len(val) >= 2]
print(f"{len(materials_with_2)=}")
materials_with_3 = [key for key, val in all_docs.items() if len(val) >= 3]
print(f"{len(materials_with_3)=}")


# %%
for mp_id in materials_with_3:
    mace, chgnet = all_docs[mp_id]["mace"], all_docs[mp_id]["chgnet"]
    formula = mace[formula_key]

    band_structs = {"CHGnet": chgnet[bs_key], "MACE": mace[bs_key]}
    fig = plot_band_structure(band_structs)
    formula = chgnet[formula_key]
    # turn title into link to materials project page
    href = f"https://legacy.materialsproject.org/materials/{mp_id}"
    title = f"<a {href=}>{mp_id}</a> {formula}"
    fig.layout.title = dict(text=title, x=0.5)
    fig.show()

    # now the same for DOS
    doses = {
        "CHGnet": chgnet[dos_key],
        "MACE": mace[dos_key],
        "MP": all_docs[mp_id]["mp"][dos_key],
    }
    ax = plot_phonon_dos(doses, last_peak_anno=r"${key}={last_peak:.1f}$")
    ax.set_title(f"{mp_id} {formula}", fontsize=22, fontweight="bold")
    save_fig(ax, f"{FIGS_DIR}/{mp_id}-{formula.replace(' ', '')}/dos-all.pdf")

    fig_dos = plot_dos(doses)
    fig_dos.layout.title = f"{mp_id} {formula}"
    fig_dos.show()
