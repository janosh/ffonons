from pathlib import Path
from unittest.mock import patch

import ffonons


def test_update_key_name(mock_data_dir: Path) -> None:
    test_dir = mock_data_dir / "test_update"
    test_dir.mkdir()
    (test_dir / "test_file.json.gz").touch()

    with (
        patch("ffonons.io.json.load") as mock_json_load,
        patch("ffonons.io.json.dump") as mock_json_dump,
    ):
        mock_json_load.return_value = {"old_key": "value"}
        ffonons.io.update_key_name(str(test_dir), {"old_key": "new_key"})

    mock_json_dump.assert_called_once()
    args, _ = mock_json_dump.call_args
    assert "new_key" in args[0]
    assert "old_key" not in args[0]
