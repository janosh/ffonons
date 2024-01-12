import json
import os
from typing import TYPE_CHECKING

from emmet.core.phonon import PhononBSDOSDoc
from monty.io import zopen
from monty.json import MontyEncoder
from mp_api.client import MPRester

from ffonons import DATA_DIR

if TYPE_CHECKING:
    from pymatgen.core import Structure

__author__ = "Janosh Riebesell"
__date__ = "2023-12-07"

mpr = MPRester(mute_progress_bars=True)


def get_mp_ph_docs(mp_id: str, docs_dir: str = DATA_DIR) -> tuple[PhononBSDOSDoc, str]:
    """Get phonon data from MP and save it to disk.

    Args:
        mp_id (str): Material ID.
        docs_dir (str = DATA_DIR): Directory to save the MP phonon doc. Set to None or
            "" to not save.

    Returns:
        tuple[PhononBSDOSDoc, str]: Phonon doc and path to saved doc.
    """
    struct: Structure = mpr.get_structure_by_material_id(mp_id)

    id_formula = f"{mp_id}-{struct.formula.replace(' ', '')}"
    mp_ph_doc_path = f"{docs_dir}/{id_formula}-mp.json.lzma" if docs_dir else ""

    if mp_ph_doc_path and not os.path.isfile(mp_ph_doc_path):
        mp_phonon_doc: PhononBSDOSDoc = mpr.materials.phonon.get_data_by_id(mp_id)

        with zopen(mp_ph_doc_path, "wt") as file:
            json.dump(mp_phonon_doc, file, cls=MontyEncoder)

    return mp_phonon_doc, mp_ph_doc_path
