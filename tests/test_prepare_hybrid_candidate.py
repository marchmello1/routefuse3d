import json

import pytest

from prepare_hybrid_candidate import assemble, merge_runtime


def make_family(path, *, isolated=0.55, contiguous=0.65):
    path.mkdir()
    (path / "fold0_member0.pt").write_bytes(b"unique weight file")
    (path / "ensemble_manifest.json").write_text(json.dumps([
        {"fold": 0, "file": "fold0_member0.pt"}
    ]))
    (path / "runtime.json").write_text(json.dumps({
        "roi": 128, "overlap": 0.25, "sw_batch_size": 2,
        "density_threshold": 0.5,
        "isolated_threshold": isolated,
        "contiguous_threshold": contiguous,
        "isolated_sparse_minimum_size": 17,
        "contiguous_sparse_minimum_size": 1,
    }))


def test_assembler_keeps_reference_isolated_and_candidate_contiguous(tmp_path):
    reference, candidate, output = (
        tmp_path / "reference", tmp_path / "candidate", tmp_path / "output"
    )
    make_family(reference, isolated=0.55, contiguous=0.65)
    make_family(candidate, isolated=0.25, contiguous=0.45)
    result = assemble(reference, candidate, output, members=1)
    runtime = result["runtime"]
    assert runtime["isolated_threshold"] == 0.55
    assert runtime["contiguous_threshold"] == 0.45
    assert runtime["hybrid_reference_members"] == 1
    assert sorted(path.name for path in output.glob("*.pt")) == [
        "a_reference_fold0.pt", "b_candidate_fold0.pt"
    ]
    assert len(json.loads((output / "hybrid_manifest.json").read_text())) == 2
    with pytest.raises(FileExistsError):
        assemble(reference, candidate, output, members=1)


def test_reject_incompatible_preprocessing():
    with pytest.raises(ValueError, match="roi"):
        merge_runtime({"roi": 128}, {"roi": 96}, 1)
    with pytest.raises(ValueError, match="at least one"):
        merge_runtime({}, {}, 0)
