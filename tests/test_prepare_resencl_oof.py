import json

import pytest

from prepare_resencl_oof import prepare


def test_prepare_requires_82_unique_excluded_fold_cases(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    identifiers = [f"case_{index:03d}" for index in range(82)]
    for identifier in identifiers:
        (images / f"{identifier}_0000.nii.gz").write_bytes(b"image")
    splits = [{"val": identifiers[fold::5]} for fold in range(5)]
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits))
    output = tmp_path / "oof"
    plan = prepare(images, splits_path, output)
    assert sum(len(fold["validation_ids"]) for fold in plan["folds"]) == 82
    assert [fold["virtual_fold"] for fold in plan["folds"]] == [0, 2, 4, 6, 8]
    for fold in plan["folds"]:
        for identifier in fold["validation_ids"]:
            path = output / f"fold_{fold['source_fold']}" / "input" / f"{identifier}_0000.nii.gz"
            assert path.is_symlink() and path.resolve().is_file()
    with pytest.raises(FileExistsError):
        prepare(images, splits_path, output)


def test_prepare_rejects_duplicate_validation_case(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    identifiers = [f"case_{index:03d}" for index in range(81)]
    for identifier in identifiers:
        (images / f"{identifier}_0000.nii.gz").write_bytes(b"image")
    splits = [{"val": identifiers[fold::5]} for fold in range(5)]
    splits[4]["val"].append(identifiers[0])
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits))
    with pytest.raises(ValueError, match="82 distinct"):
        prepare(images, splits_path, tmp_path / "oof")
