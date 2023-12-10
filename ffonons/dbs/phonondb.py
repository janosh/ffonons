import copy
from typing import Any

from pymatgen.core import Structure
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.symmetry.kpath import KPathSeek

__author__ = "Janine George, Aakash Naik, Janosh Riebesell"
__date__ = "2023-12-07"


def get_phonopy_kpath(
    structure: Structure, kpath_scheme: str, symprec: float, **kwargs: Any
) -> tuple:
    """Get high-symmetry points in k-space in phonopy format.

    Args:
        structure (Structure): pymatgen structure object
        kpath_scheme (str): kpath scheme
        symprec (float): precision for symmetry determination
        **kwargs: additional params passed to HighSymmKpath or KPathSeek

    Returns:
        tuple: kpoints and path
    """
    if kpath_scheme in ("setyawan_curtarolo", "latimer_munro", "hinuma"):
        high_symm_kpath = HighSymmKpath(
            structure, path_type=kpath_scheme, symprec=symprec, **kwargs
        )
        kpath = high_symm_kpath.kpath
    elif kpath_scheme == "seekpath":
        high_symm_kpath = KPathSeek(structure, symprec=symprec, **kwargs)
        kpath = high_symm_kpath._kpath  # noqa: SLF001

    path = copy.deepcopy(kpath["path"])

    for path_idx, label_set in enumerate(kpath["path"]):
        for label_idx, label in enumerate(label_set):
            path[path_idx][label_idx] = kpath["kpoints"][label]
    return kpath["kpoints"], path
