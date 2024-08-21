from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from pymatgen.core import Lattice, Structure
from pymatgen.phonon import PhononBandStructureSymmLine, PhononDos

import ffonons


def test_load_pymatgen_phonon_docs(mock_data_dir: Path) -> None:
    mp_dir = mock_data_dir / "mp"
    mp_dir.mkdir()
    (mp_dir / "mp-1-NaCl-pbe.json.gz").touch()
    (mp_dir / "mp-2-MgO-ml_model.json.lzma").touch()

    mock_ph_doc = MagicMock(spec=PhononBSDOSDoc)
    mock_ph_doc.structure = Structure(
        Lattice.cubic(5.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    mock_ph_doc.supercell = np.eye(3) * 2
    mock_ph_doc.phonon_bandstructure = MagicMock(spec=PhononBandStructureSymmLine)
    mock_ph_doc.phonon_dos = MagicMock(spec=PhononDos)

    with (
        patch("ffonons.io.zopen"),
        patch("ffonons.io.json.load", return_value=mock_ph_doc),
        patch("ffonons.io.re.search") as mock_search,
    ):
        # Mock the regex search to return the expected groups
        mock_search.return_value.groups.side_effect = [
            ("mp-1", "NaCl", "pbe"),
            ("mp-2", "MgO", "ml_model"),
        ]
        result = ffonons.io.load_pymatgen_phonon_docs(docs_to_load="mp")

    assert len(result) == 2
    assert "mp-1" in result
    assert "mp-2" in result
    assert "pbe" in result["mp-1"]
    assert "ml_model" in result["mp-2"]
    assert hasattr(result["mp-1"]["pbe"], "file_path")
