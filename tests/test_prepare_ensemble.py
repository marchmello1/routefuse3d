from prepare_ensemble import checkpoint_candidates


def test_best_checkpoint_precedes_last_for_single_member_package(tmp_path):
    (tmp_path / "fold2_last.pt").touch()
    (tmp_path / "fold2_best.pt").touch()
    (tmp_path / "fold2_epoch009.pt").touch()
    candidates = checkpoint_candidates(tmp_path, 2)
    assert [path.name for path in candidates[:2]] == [
        "fold2_best.pt", "fold2_last.pt",
    ]


def test_ema_requires_explicit_preference(tmp_path):
    for suffix in ("best", "last", "ema"):
        (tmp_path / f"fold1_{suffix}.pt").touch()
    assert checkpoint_candidates(tmp_path, 1)[0].name == "fold1_best.pt"
    assert checkpoint_candidates(tmp_path, 1, prefer_ema=True)[0].name == "fold1_ema.pt"
