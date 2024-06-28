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

from ffonons import DATA_DIR
from ffonons.dbs.mp import get_mp_ph_docs


@pytest.fixture()
def mock_mpr() -> Generator[MagicMock, None, None]:
    with patch("ffonons.dbs.mp.MPRester") as mock:
        yield mock.return_value


@pytest.fixture()
def mock_structure() -> MagicMock:
    structure = MagicMock(spec=Structure)
    structure.formula = "Si2"
    return structure


@pytest.fixture()
def mock_phonon_doc() -> PhononBSDOSDoc:
    return PhononBSDOSDoc(
        material_id="mp-149", last_updated=datetime(2023, 1, 1, tzinfo=UTC)
    )


def test_get_mp_ph_docs_new_file(
    mock_mpr: MagicMock,
    mock_structure: MagicMock,
    mock_phonon_doc: PhononBSDOSDoc,
    tmp_path: Path,
) -> None:
    mock_mpr.get_structure_by_material_id.return_value = mock_structure
    mock_mpr.materials.phonon.get_data_by_id.return_value = mock_phonon_doc

    result, file_path = get_mp_ph_docs("mp-149", docs_dir=str(tmp_path))

    assert isinstance(result, PhononBSDOSDoc)
    assert result.material_id == mock_phonon_doc.material_id
    assert result.last_updated.replace(tzinfo=UTC) <= datetime.now(UTC)
    assert file_path == str(tmp_path / "mp-149-Si2.json.lzma")
    assert os.path.isfile(file_path)

    with zopen(file_path, "rt") as f:
        saved_data = json.load(f)
    assert saved_data["material_id"] == mock_phonon_doc.material_id
    saved_date = datetime.fromisoformat(saved_data["last_updated"].rstrip("Z")).replace(
        tzinfo=UTC
    )
    assert saved_date <= datetime.now(UTC)


def test_get_mp_ph_docs_existing_file(
    mock_mpr: MagicMock,
    mock_phonon_doc: PhononBSDOSDoc,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "mp-149-Si2.json.lzma"
    with zopen(file_path, "wt") as f:
        json.dump(mock_phonon_doc.model_dump(mode="json"), f)

    result, returned_path = get_mp_ph_docs("mp-149", docs_dir=str(tmp_path))

    assert isinstance(result, dict)
    assert result["material_id"] == mock_phonon_doc.material_id
    saved_date = datetime.fromisoformat(result["last_updated"].rstrip("Z")).replace(
        tzinfo=UTC
    )
    assert saved_date <= datetime.now(UTC)
    assert returned_path == str(file_path)
    mock_mpr.materials.phonon.get_data_by_id.assert_not_called()


def test_get_mp_ph_docs_no_save(
    mock_mpr: MagicMock, mock_structure: MagicMock, mock_phonon_doc: PhononBSDOSDoc
) -> None:
    get_ph_data_by_id = mock_mpr.materials.phonon.get_data_by_id
    mock_mpr.get_structure_by_material_id.return_value = mock_structure
    get_ph_data_by_id.return_value = mock_phonon_doc

    result, file_path = get_mp_ph_docs("mp-149", docs_dir="")

    assert isinstance(result, PhononBSDOSDoc)
    assert result.material_id == mock_phonon_doc.material_id
    assert result.last_updated.replace(tzinfo=UTC) <= datetime.now(UTC)
    assert file_path == ""
    get_ph_data_by_id.assert_not_called()


def test_get_mp_ph_docs_default_dir(
    mock_mpr: MagicMock, mock_structure: MagicMock, mock_phonon_doc: PhononBSDOSDoc
) -> None:
    get_ph_data_by_id = mock_mpr.materials.phonon.get_data_by_id
    mock_mpr.get_structure_by_material_id.return_value = mock_structure
    get_ph_data_by_id.return_value = mock_phonon_doc

    with patch("ffonons.dbs.mp.os.path.isfile", return_value=False):
        result, file_path = get_mp_ph_docs("mp-149")

    assert isinstance(result, PhononBSDOSDoc)
    assert result.material_id == mock_phonon_doc.material_id
    assert result.last_updated.replace(tzinfo=UTC) <= datetime.now(UTC)
    assert file_path == f"{DATA_DIR}/mp/mp-149-Si2.json.lzma"
    if os.path.isfile(file_path):
        get_ph_data_by_id.assert_not_called()
    else:
        get_ph_data_by_id.assert_called_once_with("mp-149")
