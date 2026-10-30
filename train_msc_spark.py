"""Standalone sparse ResEnc-L structural-completion pretraining experiment.

Uses upstream nnSSL's SparK architecture and this project's on-demand patch
loader. Do not use a checkpoint for submission until downstream validation and
nnU-Net transfer have been verified. The nnSSL upstream license is CC-BY-SA-4.0.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from msc_dataset import MSCPatchDataset
from msc_loss import MaskedStructuralCompletionLoss
from msc_schedule import polynomial_lr
from nnssl.architectures import spark_utils
from nnssl.architectures.architecture_registry import get_res_enc_l
from nnssl.architectures.spark_model import SparK3D
from nnssl.architectures.spark_utils import convert_to_spark_cnn
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer


def build_model() -> SparK3D:
    architecture = get_res_enc_l(1, 4, False)
    architecture.encoder = convert_to_spark_cnn(architecture.encoder)
    return SparK3D(architecture, use_mask_token=True)


def save_checkpoint(
    path: Path, step: int, model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, crop_size: int,
) -> None:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    payload = {
        "step": step,
        "spark_weights": module.state_dict(),
        "network_weights": module.architecture.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "architecture": "nnssl SparK3D ResEncL",
        "input_channels": 1,
        "output_channels": 4,
        "crop_size": crop_size,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=62_500)
    parser.add_argument("--batch-size", type=int, default=1, help="Per GPU")
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--targets-in-workers", action="store_true",
                        help="Compute identical structural targets in DataLoader workers")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--multiscale", action="store_true")
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--memory-fraction", type=float, default=None,
                        help="Optional CUDA allocator cap for coexistence checks")
    args = parser.parse_args()

    distributed = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if args.memory_fraction is not None:
        if not 0 < args.memory_fraction <= 1:
            raise ValueError("memory fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=device)
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1

    dataset = MSCPatchDataset(
        args.data,
        samples_per_epoch=args.steps * args.batch_size * world,
        crop_size=args.crop_size,
        manifest=args.manifest,
        multiscale=args.multiscale,
        targets_in_workers=args.targets_in_workers,
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
    model = build_model().to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=3e-5)
    start_step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["spark_weights"])
        optimizer.load_state_dict(saved["optimizer_state"])
        start_step = int(saved["step"])
    if not 0 <= start_step <= args.steps:
        raise ValueError(f"Invalid resume step {start_step} for {args.steps} total steps")
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    loss_function = MaskedStructuralCompletionLoss(workers=2)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        print(f"MSC_START paths={len(dataset.paths)} world={world} batch={args.batch_size} "
              f"start={start_step} target={args.steps}", flush=True)
    if distributed:
        dist.barrier()

    iterator = iter(loader)
    for step in range(start_step, args.steps):
        started = time.monotonic()
        current_lr = polynomial_lr(args.lr, step, args.steps, args.poly_power)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        try:
            batch = next(iterator)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(step)
            iterator = iter(loader)
            batch = next(iterator)
        data = batch["data"].to(device, non_blocking=True).clamp(-5.0, 20.0)
        ratio = float(torch.empty(()).uniform_(0.6, 0.9))
        mask = BaseMAETrainer.mask_creation(args.batch_size, data.shape[2:], ratio).to(device)
        spark_utils._cur_active = mask
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(data)
            loss = loss_function(output, data, mask, batch.get("targets"))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
        optimizer.step()
        if rank == 0 and (step == start_step or (step + 1) % args.log_every == 0):
            peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(f"step={step + 1} loss={loss.item():.6f} "
                  f"lr={current_lr:.8g} elapsed={time.monotonic() - started:.2f}s "
                  f"peak_gib={peak_gib:.2f}", flush=True)
        if rank == 0 and ((step + 1) % args.save_every == 0 or step + 1 == args.steps):
            save_checkpoint(args.output / "msc_last.pt", step + 1, model, optimizer, args.crop_size)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
