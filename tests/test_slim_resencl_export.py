import json
from pathlib import Path

import pytest

from slim_resencl_export import make_slim_export, sha256


def make_source(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    (source / "plans.json").write_text("{}")
    (source / "dataset.json").write_text("{}")
    members = []
    for virtual_fold in range(10):
        folder = source / f"fold_{virtual_fold}"
        folder.mkdir()
        (folder / "checkpoint_final.pth").write_bytes(bytes([virtual_fold]) * 101)
        members.append(
            {
                "virtual_fold": virtual_fold,
                "source_fold": virtual_fold // 2,
                "source_checkpoint": (
                    "checkpoint_final.pth" if virtual_fold % 2 == 0 else "checkpoint_best.pth"
                ),
            }
        )
    (source / "export_manifest.json").write_text(
        json.dumps({"folds": list(range(10)), "member_sources": members})
    )
    return source


def test_slim_export_keeps_only_five_finals_and_hashes(tmp_path):
    source = make_source(tmp_path)
    target = tmp_path / "slim"
    result = make_slim_export(source, target)
    assert result["folds"] == [0, 2, 4, 6, 8]
    manifest = json.loads((target / "export_manifest.json").read_text())
    assert manifest["folds"] == [0, 2, 4, 6, 8]
    assert len(manifest["member_sources"]) == 5
    assert all(row["source_checkpoint"] == "checkpoint_final.pth" for row in manifest["member_sources"])
    for fold in manifest["folds"]:
        relative = Path(f"fold_{fold}/checkpoint_final.pth")
        assert sha256(source / relative) == sha256(target / relative)
    assert not (target / "fold_1").exists()
    with pytest.raises(FileExistsError):
        make_slim_export(source, target)


def test_slim_export_rejects_incomplete_source(tmp_path):
    source = make_source(tmp_path)
    (source / "fold_8/checkpoint_final.pth").unlink()
    target = tmp_path / "slim"
    with pytest.raises(FileNotFoundError):
        make_slim_export(source, target)
    assert not target.exists()
