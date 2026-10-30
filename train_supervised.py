from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import LabeledPatchDataset, discover_labeled_cases, stratified_fold
from src.losses import contiguous_loss, isolated_loss
from src.model import RoutedResidualUNet, strip_distributed_prefix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ssl", type=Path)
    parser.add_argument(
        "--ssl-source",
        choices=("model", "teacher"),
        default="model",
        help=(
            "State to transfer from an SSL checkpoint. 'teacher' selects the "
            "momentum encoder; the default preserves earlier online-encoder runs."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Continue a supervised fold checkpoint and preserve Adam moments.",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--samples-per-epoch", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pretrained-backbone-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="Exponential moving average used for a separately evaluable checkpoint.",
    )
    args = parser.parse_args()
    if args.ssl and args.resume:
        parser.error("--ssl and --resume are mutually exclusive")
    random.seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    torch.manual_seed(args.seed + args.fold)

    cases = discover_labeled_cases(args.data)
    train_cases, validation_cases = stratified_fold(cases, args.fold)
    if len(cases) != 82:
        raise RuntimeError(f"Expected 82 paired cases, found {len(cases)}")
    print(f"fold={args.fold} train={len(train_cases)} validation={len(validation_cases)}", flush=True)
    dataset = LabeledPatchDataset(train_cases, args.samples_per_epoch)
    print(f"cached={len(dataset.cache)} labeled volumes", flush=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    device = torch.device("cuda")
    # Density routing is learned during SSL transfer fine-tuning. Keep the
    # no-SSL diagnostic architecture/checkpoint format identical across folds.
    with_density_head = args.ssl is not None
    model = RoutedResidualUNet(
        args.base_channels, with_density_head=with_density_head
    ).to(device)
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    start_epoch = 0
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        resume_state = strip_distributed_prefix(resume_checkpoint["model"])
        has_density_head = any(
            key.startswith("density_head.") for key in resume_state
        )
        if has_density_head != with_density_head:
            # A resumed SSL-transfer checkpoint carries the density/strata
            # heads even though --ssl is intentionally not supplied again.
            with_density_head = has_density_head
            model = RoutedResidualUNet(
                args.base_channels, with_density_head=with_density_head
            ).to(device)
        model.load_state_dict(resume_state)
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        if int(resume_checkpoint.get("fold", args.fold)) != args.fold:
            raise RuntimeError("Resume checkpoint fold does not match --fold")
        print(
            f"resumed={args.resume} start_epoch={start_epoch} ",
            f"previous_loss={resume_checkpoint.get('train_loss')}",
            flush=True,
        )
    elif args.ssl:
        checkpoint = torch.load(args.ssl, map_location="cpu", weights_only=False)
        if args.ssl_source not in checkpoint:
            raise RuntimeError(
                f"SSL checkpoint {args.ssl} has no {args.ssl_source!r} state"
            )
        ssl_state = strip_distributed_prefix(checkpoint[args.ssl_source])
        missing, unexpected = model.load_state_dict(ssl_state, strict=False)
        print(
            f"loaded SSL source={args.ssl_source} "
            f"missing={missing} unexpected={unexpected}",
            flush=True,
        )
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    if resume_checkpoint is not None and "ema_model" in resume_checkpoint:
        ema_model.load_state_dict(resume_checkpoint["ema_model"])
        print("restored supervised EMA", flush=True)
    if with_density_head:
        head_prefixes = (
            "shared_head",
            "isolated_head",
            "contiguous_head",
            "morphology_head",
            "density_head",
            "strata_head",
        )
        backbone = [p for name, p in model.named_parameters() if not name.startswith(head_prefixes)]
        heads = [p for name, p in model.named_parameters() if name.startswith(head_prefixes)]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone, "lr": args.pretrained_backbone_lr},
                {"params": heads, "lr": args.lr},
            ],
            weight_decay=1e-4,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if resume_checkpoint is not None and "optimizer" in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        # Start a fresh, lower-amplitude cosine cycle for the continuation.
        target_lrs = (
            [args.pretrained_backbone_lr, args.lr]
            if with_density_head
            else [args.lr]
        )
        for group, learning_rate in zip(
            optimizer.param_groups, target_lrs, strict=True
        ):
            group["lr"] = learning_rate
            group["initial_lr"] = learning_rate
    remaining_epochs = args.epochs - start_epoch
    if remaining_epochs <= 0:
        raise RuntimeError(
            f"Checkpoint is already at epoch {start_epoch - 1}; "
            f"--epochs must exceed {start_epoch}"
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, remaining_epochs, eta_min=2e-6
    )
    scaler = torch.amp.GradScaler("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    best_loss = (
        float(resume_checkpoint.get("train_loss", float("inf")))
        if resume_checkpoint is not None
        else float("inf")
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        running = 0.0
        for step, batch in enumerate(loader):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            contiguous = batch["contiguous"].to(device, non_blocking=True)
            dense = batch["dense"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(image)
                iso_mask = contiguous < 0.5
                con_mask = ~iso_mask
                shared_parts = []
                expert_parts = []
                if iso_mask.any():
                    shared_parts.append(isolated_loss(output.shared[iso_mask], target[iso_mask]))
                    expert_parts.append(isolated_loss(output.isolated[iso_mask], target[iso_mask]))
                if con_mask.any():
                    shared_parts.append(contiguous_loss(output.shared[con_mask], target[con_mask]))
                    expert_parts.append(contiguous_loss(output.contiguous[con_mask], target[con_mask]))
                shared_loss = torch.stack(shared_parts).mean()
                expert_loss = torch.stack(expert_parts).mean()
                if output.strata is not None:
                    # 0/1 are isolated sparse/dense; 2/3 contiguous sparse/dense.
                    stratum_index = contiguous.long() * 2 + dense.long()
                    gather_index = stratum_index[:, None, None, None, None].expand(
                        -1, 1, *output.strata.shape[2:]
                    )
                    stratum_logits = output.strata.gather(1, gather_index)
                    stratum_parts = []
                    if iso_mask.any():
                        stratum_parts.append(
                            isolated_loss(stratum_logits[iso_mask], target[iso_mask])
                        )
                    if con_mask.any():
                        stratum_parts.append(
                            contiguous_loss(stratum_logits[con_mask], target[con_mask])
                        )
                    stratum_loss = torch.stack(stratum_parts).mean()
                    seg_loss = (
                        0.45 * shared_loss
                        + 0.20 * expert_loss
                        + 0.35 * stratum_loss
                    )
                else:
                    seg_loss = 0.65 * shared_loss + 0.35 * expert_loss
                route_loss = F.binary_cross_entropy_with_logits(output.morphology, contiguous)
                if output.density is not None:
                    # Uniform subtype sampling yields four sparse versus two
                    # dense structures. Balance the router so unseen dense
                    # structures are not systematically sent to sparse heads.
                    density_loss = F.binary_cross_entropy_with_logits(
                        output.density,
                        dense,
                        pos_weight=torch.as_tensor(2.0, device=dense.device),
                    )
                    loss = seg_loss + 0.15 * route_loss + 0.15 * density_loss
                else:
                    loss = seg_loss + 0.20 * route_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                for ema_parameter, online_parameter in zip(
                    ema_model.parameters(), model.parameters(), strict=True
                ):
                    ema_parameter.lerp_(
                        online_parameter.detach(), 1.0 - args.ema_decay
                    )
            running += float(loss.detach())
        scheduler.step()
        mean_loss = running / len(loader)
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(
            f"epoch={epoch} train_loss={mean_loss:.5f} "
            f"lr={scheduler.get_last_lr()[0]:.3g} peak_memory_gib={peak_gib:.2f}",
            flush=True,
        )
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": mean_loss,
            "fold": args.fold,
            "validation_cases": [str(x.image) for x in validation_cases],
            "args": vars(args),
        }
        torch.save(state, args.output / f"fold{args.fold}_last.pt")
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            torch.save(
                {
                    "model": state["ema_model"],
                    "source": "supervised_ema",
                    "epoch": epoch,
                    "train_loss": mean_loss,
                    "fold": args.fold,
                    "validation_cases": state["validation_cases"],
                    "args": state["args"],
                },
                args.output / f"fold{args.fold}_ema.pt",
            )
        if (epoch + 1) % 10 == 0:
            torch.save(state, args.output / f"fold{args.fold}_epoch{epoch + 1:03d}.pt")
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                {
                    "model": state["model"],
                    "epoch": epoch,
                    "train_loss": mean_loss,
                    "fold": args.fold,
                    "validation_cases": state["validation_cases"],
                    "args": state["args"],
                },
                args.output / f"fold{args.fold}_best.pt",
            )


if __name__ == "__main__":
    main()
