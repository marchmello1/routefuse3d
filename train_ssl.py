from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from src.data import UnlabeledPatchDataset
from src.model import RoutedResidualUNet


def block_mask(batch: torch.Tensor, block: int, ratio: float) -> torch.Tensor:
    shape = batch.shape[2:]
    grid = tuple((n + block - 1) // block for n in shape)
    mask = torch.rand((batch.shape[0], 1, *grid), device=batch.device) < ratio
    mask = mask.repeat_interleave(block, 2).repeat_interleave(block, 3).repeat_interleave(block, 4)
    return mask[:, :, : shape[0], : shape[1], : shape[2]]


def gradient_reconstruction_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for dimension in (2, 3, 4):
        losses.append(
            F.smooth_l1_loss(
                torch.diff(reconstruction, dim=dimension),
                torch.diff(target, dim=dimension),
            )
        )
    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--samples-per-epoch", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--multiscale", action="store_true")
    parser.add_argument("--teacher-momentum", type=float, default=0.996)
    parser.add_argument("--mask-min", type=float, default=0.50)
    parser.add_argument("--mask-max", type=float, default=0.85)
    parser.add_argument("--focus-prefix", type=str)
    parser.add_argument("--focus-probability", type=float, default=0.0)
    args = parser.parse_args()

    distributed = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank() if distributed else 0

    # The initial tmpfs subset establishes local completion features cheaply.
    # Manifest-backed continuation sees the complete corpus and deliberately
    # broadens physical context (128/179/240 source voxels) before resampling.
    # Task 2 already requests this explicitly; enabling it for a complete
    # Task-1 manifest makes the two-stage curriculum scale-aware as well.
    use_multiscale = args.multiscale or args.manifest is not None
    dataset = UnlabeledPatchDataset(
        args.data,
        args.samples_per_epoch,
        manifest=args.manifest,
        multiscale=use_multiscale,
        focus_prefix=args.focus_prefix,
        focus_probability=args.focus_probability,
    )
    if rank == 0:
        print(
            f"ssl_paths={len(dataset.paths)} groups={len(dataset.groups)} "
            f"multiscale={use_multiscale}",
            flush=True,
        )
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    model = RoutedResidualUNet(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        start_epoch = checkpoint["epoch"] + 1
    teacher = RoutedResidualUNet(args.base_channels).to(device)
    if args.resume and "teacher" in checkpoint:
        teacher.load_state_dict(checkpoint["teacher"])
    else:
        teacher.load_state_dict(model.state_dict())
    teacher.eval()
    teacher.requires_grad_(False)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    remaining_epochs = max(1, args.epochs - start_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, remaining_epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda")
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        for step, batch in enumerate(loader):
            target = batch["image"].to(device, non_blocking=True)
            structure = batch["structure"].to(device, non_blocking=True)
            ratio1 = float(torch.empty(1).uniform_(args.mask_min, args.mask_max))
            ratio2 = float(torch.empty(1).uniform_(args.mask_min, args.mask_max))
            mask1, mask2 = block_mask(target, 16, ratio1), block_mask(target, 16, ratio2)
            view1 = target.masked_fill(mask1, 0)
            view2 = target.masked_fill(mask2, 0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                recon1, embed1, latent1 = model(view1, ssl=True)
                recon2, embed2, latent2 = model(view2, ssl=True)
                with torch.no_grad():
                    target_latent = teacher.forward_ssl_target(target).detach()
                intensity1, fields1 = recon1[:, :1], recon1[:, 1:]
                intensity2, fields2 = recon2[:, :1], recon2[:, 1:]
                masked1 = 0.25 * F.smooth_l1_loss(intensity1[mask1], target[mask1])
                masked2 = 0.25 * F.smooth_l1_loss(intensity2[mask2], target[mask2])
                structural_mask1 = mask1.expand_as(fields1)
                structural_mask2 = mask2.expand_as(fields2)
                structural = F.smooth_l1_loss(fields1[structural_mask1], structure[structural_mask1])
                structural += F.smooth_l1_loss(fields2[structural_mask2], structure[structural_mask2])
                global_recon = 0.025 * (
                    F.l1_loss(intensity1, target) + F.l1_loss(intensity2, target)
                )
                edge_recon = 0.5 * (
                    gradient_reconstruction_loss(intensity1, target)
                    + gradient_reconstruction_loss(intensity2, target)
                )
                consistency = 1.0 - F.cosine_similarity(embed1, embed2.detach(), dim=1).mean()
                consistency += 1.0 - F.cosine_similarity(embed2, embed1.detach(), dim=1).mean()
                latent_mask1 = F.interpolate(
                    mask1.float(), size=latent1.shape[2:], mode="nearest"
                ).squeeze(1).bool()
                latent_mask2 = F.interpolate(
                    mask2.float(), size=latent2.shape[2:], mode="nearest"
                ).squeeze(1).bool()
                latent_error1 = 1.0 - F.cosine_similarity(latent1, target_latent, dim=1)
                latent_error2 = 1.0 - F.cosine_similarity(latent2, target_latent, dim=1)
                predictive = latent_error1[latent_mask1].mean() + latent_error2[latent_mask2].mean()
                loss = (
                    masked1
                    + masked2
                    + structural
                    + global_recon
                    + 0.05 * edge_recon
                    + 0.15 * consistency
                    + 0.25 * predictive
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            raw_model = model.module if distributed else model
            with torch.no_grad():
                momentum = args.teacher_momentum
                for target_parameter, online_parameter in zip(
                    teacher.parameters(), raw_model.parameters(), strict=True
                ):
                    target_parameter.mul_(momentum).add_(
                        online_parameter.detach(), alpha=1.0 - momentum
                    )
            running += float(loss.detach())
            if rank == 0 and step % 50 == 0:
                print(f"epoch={epoch} step={step}/{len(loader)} loss={running/(step+1):.5f}", flush=True)
        scheduler.step()
        if rank == 0:
            raw = model.module if distributed else model
            state = {
                "epoch": epoch,
                "model": raw.state_dict(),
                "teacher": teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            torch.save(state, args.output / "ssl_last.pt")
            if (epoch + 1) % 10 == 0:
                torch.save(state, args.output / f"ssl_epoch_{epoch+1:03d}.pt")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
