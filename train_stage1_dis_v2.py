#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage‑1 video pre‑training script (refactored).

Usage: torchrun --nproc_per_node=4 train_stage1_dis.py  [--your_args ...]
"""
from __future__ import annotations

import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from timm.loss import LabelSmoothingCrossEntropy

# --------------------------------------------------------------------------- #
# Project‑local imports (after setting sys.path)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config_video import get_args_parser          # noqa: E402
from models.video.stage1 import MatVisionTransformer  # noqa: E402
from utils.initial import init_video                  # noqa: E402
from utils.lr_sched import adjust_learning_rate       # noqa: E402
from utils.set_wandb import set_wandb                 # noqa: E402
from utils.eval_flag import (                         # noqa: E402
    eval_mat_combined_dis,
)
from video_datasets.video_datasets import build_dataset  # noqa: E402
import wandb

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
FLAGS_LIST = ["l", "m", "s", "ss", "sss"]
MLP_RATIO_LIST = [4, 4, 3, 3, 2, 1, 0.5]
MHA_HEAD_LIST = [12, 11, 10, 9, 8, 7, 6]
EVAL_MLP_RATIO_LIST = [4, 3, 2, 1, 0.5]
EVAL_MHA_HEAD_LIST = [12, 11, 10, 8, 6]


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Set RNG seeds (torch / numpy / random) across ranks."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model (whether in DDP 或单卡模式)."""
    return model.module if isinstance(model, DDP) else model


def create_dataloaders(
    dataset_train: Dataset,
    dataset_val: Dataset,
    batch_size: int,
    num_workers: int = 8,
) -> Tuple[DataLoader, DataLoader]:
    """Build distributed dataloaders."""
    sampler_train = DistributedSampler(dataset_train, shuffle=True, drop_last=True)
    sampler_val = DistributedSampler(dataset_val, shuffle=False, drop_last=False)

    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=sampler_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        sampler=sampler_val,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader_train, loader_val


def build_model(args) -> torch.nn.Module:
    """Instantiate backbone and optionally load ImageNet‑pretrained weights."""
    model = MatVisionTransformer(
        embed_dim=args.initial_embed_dim,
        depth=args.initial_depth,
        num_heads=args.initial_embed_dim // 64,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=True,
    )

    if args.pretrained:
        ckpt_path = Path(args.pretrained_path)  # add to your parser
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        init_video(model, checkpoint, init_width=768, depth=12, width=768)

    return model


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train(args) -> None:
    # --- distributed init -------------------------------------------------- #
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()
    is_main = local_rank == 0

    # --- reproducibility --------------------------------------------------- #
    seed = 42
    set_seed(seed)
    if world_size > 1:  # broadcast seed to ensure identical shuffles
        dist.broadcast_object_list([seed])

    # --- datasets ---------------------------------------------------------- #
    ds_train, ds_val, args.metric = build_dataset(args=args)
    dl_train, dl_val = create_dataloaders(ds_train, ds_val, args.batch_size)

    # --- model / criterion / optimizer ------------------------------------ #
    device = torch.device("cuda", local_rank)
    model = build_model(args).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    criterion = (
        LabelSmoothingCrossEntropy(args.smoothing)
        if args.smoothing > 0
        else torch.nn.CrossEntropyLoss()
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- logging ----------------------------------------------------------- #
    if is_main:
        log_root = (
            Path("logs_weight")
            / "video_stage_1"
            / f"{args.model}_{args.dataset}_{args.epochs}"
        )
        time_tag = datetime.now().strftime("%b%d_%H-%M-%S")
        (log_root / time_tag / "weight").mkdir(parents=True, exist_ok=True)
        set_wandb(args, name="nips_video_stage1")

    # --- training ---------------------------------------------------------- #
    current_stage = 0
    try:
        for epoch in range(args.epochs):
            dl_train.sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch + 1, args)

            if epoch in args.stage_epochs:
                current_stage += 1
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.5
                if is_main:
                    wandb.log({"stage": current_stage, "epoch": epoch + 1})

            # -------------------------- train one epoch -------------------- #
            model.train()
            total_loss = 0.0
            with tqdm(
                total=len(dl_train), ncols=100, disable=not is_main
            ) as pbar:
                pbar.set_description(f"Train {epoch+1}/{args.epochs}")
                for img, label in dl_train:
                    img, label = img.to(device), label.to(device)
                    optimizer.zero_grad()

                    # ---- sub‑network sampling -------------------------------- #
                    r = random.randint(0, current_stage)
                    sub_dim = 64 * MHA_HEAD_LIST[r]
                    mha_head = MHA_HEAD_LIST[r]

                    depth_list = list(range(12))
                    r_depth = random.randint(0, 5)
                    if r_depth > 2:
                        r_depth = 0
                    if r_depth > 0:
                        k = random.choice(range(r_depth))
                        drop_idx = random.sample(range(len(depth_list)), k)
                        depth_list = [
                            d for i, d in enumerate(depth_list) if i not in drop_idx
                        ]

                    mlp_ratio = MLP_RATIO_LIST[random.randint(0, current_stage)]

                    unwrap_ddp(model).configure_subnetwork(
                        sub_dim=sub_dim,
                        depth_list=depth_list,
                        mlp_ratio=mlp_ratio,
                        mha_head=mha_head,
                    )

                    # ---- forward / backward ---------------------------------- #
                    logits = model(img)
                    loss = criterion(logits, label)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    if is_main and pbar.n % 10 == 0:
                        wandb.log({"batch_loss": loss.item()})
                    if is_main:
                        pbar.set_postfix(loss=f"{loss.item():.4f}")
                        pbar.update()

            epoch_loss = total_loss / len(dl_train)
            if is_main:
                wandb.log({"train_loss": epoch_loss, "epoch": epoch + 1})
                wandb.log({"learning_rate": optimizer.param_groups[0]['lr'], "epoch": epoch + 1})

            # -------------------------- evaluation ------------------------- #
            if epoch % 20 == 0:
                for idx, flag in enumerate(FLAGS_LIST):
                    sub_dim = 64 * EVAL_MHA_HEAD_LIST[idx]
                    mha_head = EVAL_MHA_HEAD_LIST[idx]
                    depth_list = list(range(12))
                    mlp_ratio = EVAL_MLP_RATIO_LIST[idx]
                    eval_mat_combined_dis(
                        model,
                        dl_val,
                        criterion,
                        epoch,
                        optimizer,
                        args,
                        flag=flag,
                        device=device,
                        local_rank=local_rank,
                        sub_dim=sub_dim,
                        depth_list=depth_list,
                        mlp_ratio=mlp_ratio,
                        mha_head=mha_head,
                    )

            # -------------------------- checkpoint ------------------------- #
            if is_main:
                ckpt_path = log_root / time_tag / "weight" / "video_stage1.pth"
                torch.save(unwrap_ddp(model).state_dict(), ckpt_path)

    except KeyboardInterrupt:
        if is_main:
            print("Interrupted — saving last checkpoint.")
            ckpt_path = log_root / f"INTERRUPTED_{datetime.now():%H%M%S}.pth"
            torch.save(unwrap_ddp(model).state_dict(), ckpt_path)


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ARGS = get_args_parser()  # assume it returns parsed args
    train(ARGS)
