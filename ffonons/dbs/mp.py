import json
import os
from typing import TYPE_CHECKING

from monty.io import zopen
from mp_api.client import MPRester
from pymatviz.io import save_fig

from ffonons import DATA_DIR, FIGS_DIR, bs_key, dos_key
from ffonons.plots import plot_phonon_bs_mpl, plot_phonon_dos_mpl

if TYPE_CHECKING:
    from emmet.core.phonon import PhononBSDOSDoc
    from pymatgen.core import Structure

__author__ = "Janosh Riebesell"
__date__ = "2023-12-07"

mpr = MPRester(mute_progress_bars=True)


def get_mp_ph_data(mp_id: str, docs_dir: str = DATA_DIR) -> None:
    """Get phonon data from MP and save it to disk. Also plot and save phonon band
    structure and DOS. Returns the structure and the phonon data as dict.
    """
    struct: Structure = mpr.get_structure_by_material_id(mp_id)
    struct.properties["id"] = mp_id

    id_formula = f"{mp_id}-{struct.formula.replace(' ', '')}"
    figs_out_dir = f"{FIGS_DIR}/{id_formula}"
    mp_dos_fig_path = f"{figs_out_dir}/dos-mp.pdf"
    mp_bands_fig_path = f"{figs_out_dir}/bands-mp.pdf"
    # struct.to(filename=f"{out_dir}/struct.cif")

    if not os.path.isfile(mp_dos_fig_path) or not os.path.isfile(mp_bands_fig_path):
        mp_phonon_doc: PhononBSDOSDoc = mpr.materials.phonon.get_data_by_id(mp_id)

        mp_doc_dict = {
            dos_key: mp_phonon_doc.ph_dos.as_dict(),
            bs_key: mp_phonon_doc.ph_bs.as_dict(),
        }
        # create dir only if MP has phonon data
        os.makedirs(figs_out_dir, exist_ok=True)
        with zopen(f"{docs_dir}/{id_formula}-mp.json.lzma", "wt") as file:
            file.write(json.dumps(mp_doc_dict))

        ax_bs_mp = plot_phonon_bs_mpl(mp_phonon_doc.ph_bs, "MP", struct)
        # always save figure right away, plotting another figure will clear the axis!
        save_fig(ax_bs_mp, mp_bands_fig_path)

        ax_dos_mp = plot_phonon_dos_mpl({"MP": mp_phonon_doc.ph_dos}, "MP", struct)

        save_fig(ax_dos_mp, mp_dos_fig_path)

    return struct, mp_doc_dict, figs_out_dir
