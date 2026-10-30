"""Lightweight paths for fold-separated nnU-Net validation outputs."""

from pathlib import Path


def prediction_path(
    results_root: Path | None,
    template: str | None,
    fold: int,
    identifier: str,
) -> Path:
    if template is not None:
        root = Path(template.format(fold=fold))
    elif results_root is not None:
        root = results_root
    else:
        raise ValueError("nnU-Net results root or fold template is required")
    return root / f"fold_{fold}" / "validation" / f"{identifier}.nii.gz"
