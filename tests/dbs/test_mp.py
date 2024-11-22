import json
import os
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from emmet.core.phonon import PhononBSDOSDoc
from monty.io import zopen
from pymatgen.core import Structure

from ffonons import TEST_FILES
from ffonons.dbs.mp import get_mp_ph_docs


@pytest.fixture
def mock_mp_rester() -> Generator[MagicMock, None, None]:
    with patch("ffonons.dbs.mp.MPRester") as mock:
        yield mock.return_value


@pytest.fixture
def mock_structure() -> MagicMock:
    structure = MagicMock(spec=Structure)
    structure.formula = "Si2"
    return structure


@pytest.fixture
def mock_phonon_doc() -> PhononBSDOSDoc:
    return PhononBSDOSDoc(
        material_id="mp-149", last_updated=datetime(2023, 1, 1, tzinfo=UTC)
    )


def test_get_mp_ph_docs_new_file(
    mock_mp_rester: MagicMock,
    mock_structure: MagicMock,
    mock_phonon_doc: PhononBSDOSDoc,
    tmp_path: Path,
) -> None:
    mock_mp_rester.get_structure_by_material_id.return_value = mock_structure
    mock_mp_rester.materials.phonon.get_data_by_id.return_value = mock_phonon_doc

    ph_doc, file_path = get_mp_ph_docs("mp-149", docs_dir=str(tmp_path))

    assert isinstance(ph_doc, PhononBSDOSDoc)
    assert ph_doc.material_id == mock_phonon_doc.material_id
    assert ph_doc.last_updated.replace(tzinfo=UTC) <= datetime.now(UTC)
    assert file_path == f"{tmp_path}/mp-149-Si2.json.xz"
    assert os.path.isfile(file_path)

    with zopen(file_path, mode="rt") as file:
        ph_doc_from_disk = json.load(file)
    assert ph_doc_from_disk["material_id"] == mock_phonon_doc.material_id
    last_updated = ph_doc_from_disk["last_updated"]["string"].rstrip("Z")
    saved_date = datetime.fromisoformat(last_updated).replace(tzinfo=UTC)
    assert saved_date <= datetime.now(UTC)


def test_get_mp_ph_docs_existing_file(
    mock_mp_rester: MagicMock,
    mock_phonon_doc: PhononBSDOSDoc,
    mock_structure: MagicMock,
) -> None:
    mock_mp_rester.get_structure_by_material_id.return_value = mock_structure

    file_path = f"{TEST_FILES}/mp/mp-149-Si2.json.xz"
    ph_doc, returned_path = get_mp_ph_docs("mp-149", docs_dir=f"{TEST_FILES}/mp")

    assert isinstance(ph_doc, dict)
    assert ph_doc["material_id"] == mock_phonon_doc.material_id
    print(f'{ph_doc["last_updated"]=}')
    last_updated = ph_doc["last_updated"].rstrip("Z")
    saved_date = datetime.fromisoformat(last_updated).replace(tzinfo=UTC)
    assert saved_date <= datetime.now(UTC)
    assert returned_path == file_path
    mock_mp_rester.materials.phonon.get_data_by_id.assert_not_called()


def test_get_mp_ph_docs_no_save(
    mock_mp_rester: MagicMock,
    mock_structure: MagicMock,
    mock_phonon_doc: PhononBSDOSDoc,
) -> None:
    get_ph_data_by_id = mock_mp_rester.materials.phonon.get_data_by_id
    mock_mp_rester.get_structure_by_material_id.return_value = mock_structure
    get_ph_data_by_id.return_value = mock_phonon_doc

    ph_doc, file_path = get_mp_ph_docs("mp-149", docs_dir="")

    assert isinstance(ph_doc, PhononBSDOSDoc)
    assert ph_doc.material_id == mock_phonon_doc.material_id
    assert ph_doc.last_updated.replace(tzinfo=UTC) <= datetime.now(UTC)
    assert file_path == ""
    get_ph_data_by_id.assert_called_once_with("mp-149")
