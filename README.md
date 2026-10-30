# RouteFuse-3D

This is code for **RouteFuse-3D**, team Marchmello01's entry to the
[SELMA3D 2026](https://selma3d.grand-challenge.org/) challenge on segmenting
light-sheet microscopy volumes.

Final ranking: joint 1st in Task 1 and joint 2nd in Task 2.

## Method

RouteFuse-3D combines two independently trained segmentation families with a
morphology router.

1. **nnU-Net ResEnc-L.** The encoder is initialised from our multi-scale,
   SparK-style masked-image-modelling pretraining (`MSCResEncLTrainer.py`,
   `msc_*.py`, `train_msc_spark.py`). Fine-tuning uses patch size 160³ for
   1,000 epochs, over five folds.
2. **Routed residual U-Net** (`src/`). It has a shared segmentation head,
   per-category expert heads, and image-level morphology (isolated/contiguous)
   and density heads. Self-supervised pretraining is in `train_ssl.py` and
   supervised fine-tuning in `train_supervised.py`.
3. **Router.** A case counts as contiguous if either:
   - the morphology score is ≥ 0.5, or
   - the morphology score is ≥ 0.1 and the mean connected-object size of the
     ResEnc-L foreground is ≥ 581 voxels.
4. **Fusion.**
   - Isolated cases use the ResEnc-L ensemble, then remove components smaller
     than 9 voxels.
   - Contiguous cases use a 50/50 probability average of both families, then
     remove components smaller than 20 voxels.
   - A never-empty fallback lowers the threshold step by step until the mask
     is no longer empty.

Pretraining data:
- **Task 1:** the official unannotated Task 1 patches only.
- **Task 2:** the routed U-Net's pretraining is also continued on crops of the
  official Task 2 raw stacks. `cut_tiff_stacks.py` makes these crops.

Fine-tuning uses only the 82 official annotated patches, with the official
five-fold split.

## Layout

| Path | Contents |
|---|---|
| `src/` | Routed U-Net model, data, losses and hybrid routing policy |
| `container_h200_hybrid/` | Grand Challenge inference container for the hybrid |
| `container_resencl/` | ResEnc-L-only inference container |
| `container/` | Routed U-Net-only inference container |
| `*.py` (top level) | Self-supervised and supervised training, nnU-Net trainers and plans, out-of-fold evaluation, export and ensembling |
| `tests/` | Unit tests |

## Usage

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q tests
```

Model weights and challenge data are not included. You can get the data from
the SELMA3D organisers.

See `THIRD_PARTY_NOTICES.md` for third-party code (nnU-Net, nnssl).
