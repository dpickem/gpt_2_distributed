#!/usr/bin/env python
"""
Training script for the token shard dataset.
"""

# train_gpt2_distributed.py
# ------------------------------------------------------------
# Usage (2 nodes × 4 GPUs each, FSDP):
#
#   torchrun --nnodes 2 --nproc_per_node 4 \
#            --master_addr $MASTER --master_port 29000 \
#            train_gpt2_distributed.py \
#            --metadata ~/data/fineweb/metadata.json \
#            --mode fsdp --batch 4 --epochs 3 --save_dir ckpt
#
# DDP is identical except:  --mode ddp
# ------------------------------------------------------------

import argparse
import dataclasses
from functools import partial
import pathlib
import time
import os
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import fsdp
from torch.distributed.fsdp import wrap as fsdp_wrap
from torch.utils.tensorboard import SummaryWriter

from model import GPT2, GPT2Config, GPT2Block
import dataloader as gpt2_dataloader
from stats_tracker import StatsTracker

# Define a manual seed.
SEED = 42

# Set default model config.
DEFAULT_CONFIG = GPT2Config(
    n_layer=12, n_head=12, n_embd=768, n_positions=1024, vocab_size=50257
)


# ---------------------------------------------------------------------------
#                              Helper functions
# ---------------------------------------------------------------------------
def init_distributed() -> None:
    """Initialize distributed training.

    This function should only be called if this script is run via torchrun.
    """
    if dist.is_initialized():
        return

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def is_primary() -> bool:
    """Check if the current process is the primary process."""
    return dist.get_rank() == 0 if dist.is_initialized() else True


def save_checkpoint(
    model: Union[fsdp.FullyShardedDataParallel, DDP, GPT2],
    opt: torch.optim.Optimizer,
    step: int,
    out_dir: str,
):
    """Save a checkpoint.

    Args:
        model: The model to save.
        opt: The optimizer to save.
        step: The step number to save.
        out_dir: The directory to save the checkpoint.
    """
    if not is_primary():
        return
    out_dir = os.path.join(out_dir, f"step_{step:07d}")
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(model, fsdp.FullyShardedDataParallel):
        policy = fsdp.FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        )
        with fsdp.FullyShardedDataParallel.state_dict_type(
            model, fsdp.StateDictType.FULL_STATE_DICT, policy
        ):
            torch.save(model.state_dict(), f"{out_dir}/model.pt")
    elif isinstance(model, DDP):
        torch.save(model.module.state_dict(), f"{out_dir}/model.pt")
    else:
        # Local mode - save model directly
        torch.save(model.state_dict(), f"{out_dir}/model.pt")

    torch.save(opt.state_dict(), f"{out_dir}/optim.pt")


def load_checkpoint(
    model: Union[fsdp.FullyShardedDataParallel, DDP, GPT2],
    opt: torch.optim.Optimizer,
    step: int,
    out_dir: str,
):
    """Load a checkpoint."""
    pass


def get_tensorboard_writer(log_dir: pathlib.Path) -> Optional[SummaryWriter]:
    """Create a TensorBoard writer.

    Args:
        root: The root directory to save the runs.

    Returns:
    """
    if is_primary():
        t = time.strftime("%Y%m%d-%H%M%S")
        return SummaryWriter(str(log_dir / t))

    return None


def setup_model(
    config: GPT2Config, device: torch.device, training_mode: str
) -> Union[fsdp.FullyShardedDataParallel, DDP, GPT2]:
    """Setup the model.

    Args:
        config: The model configuration.
        device: The device to use for training.
        training_mode: The training mode.

    Returns:
        The model wrapped in FSDP or DDP (or not wrapped at all), depending on the training mode.
    """
    # Set up base model.
    model: Union[fsdp.FullyShardedDataParallel, DDP, GPT2] = GPT2(config).to(device)

    # Wrap the model in FSDP or DDP.
    if training_mode == "fsdp":
        wrap_policy = partial(
            fsdp_wrap.transformer_auto_wrap_policy,
            transformer_layer_cls={GPT2Block},
        )
        mp = fsdp.MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.float32,
        )
        model = fsdp.FullyShardedDataParallel(
            model,
            auto_wrap_policy=wrap_policy,
            mixed_precision=mp,
            sharding_strategy=fsdp.ShardingStrategy.FULL_SHARD,
        )
    elif training_mode == "ddp":
        model = DDP(model, device_ids=[device])

    return model


def print_device_info():
    """Print device information."""
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA current device: {torch.cuda.current_device()}")
    print(
        f"CUDA current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}"
    )


def get_memory_info() -> Tuple[float, float]:
    """Get memory information.

    Returns:
        memory_allocated: Memory allocated in GB.
        memory_reserved: Memory reserved in GB.
    """
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return memory_allocated, memory_reserved

    return 0.0, 0.0


def main():
    """Main function.

    Example usage (local, not distributed):

    ```
    # --------------------------------------------------------------------------------------------
    # Local training with a single GPU.
    # See scripts/run_training_local_single_gpu.sh
    # --------------------------------------------------------------------------------------------

    python train_gpt2_distributed.py \
            --data_dir ~/data/datasets/fineweb/sample/10BT_tokenized/ \
            --device cuda \
            --training_mode local --batch 4 --epochs 3 \
            --save_dir ./checkpoints \
            --log_dir ./logs
    ```

    # --------------------------------------------------------------------------------------------
    # Local training with a single GPU via torchrun and DDP
    # See scripts/run_training_local_single_gpu.sh
    # --------------------------------------------------------------------------------------------
    torchrun --nnodes 1 --nproc_per_node 1 \
            --master_addr localhost --master_port 29000 \
            train_gpt2_distributed.py \
            --data_dir ~/data/datasets/fineweb/sample/10BT_tokenized/ \
            --training_mode ddp \
            --batch 4 \
            --grad_accum_steps 4 \
            --epochs 3 \
            --save_dir ./checkpoints \
            --log_dir ./logs

    # --------------------------------------------------------------------------------------------
    # Local training with a single GPU via torchrun and FSDP
    # See scripts/run_training_local_single_gpu_fsdp.sh
    # --------------------------------------------------------------------------------------------
    torchrun --nnodes 1 --nproc_per_node 1 \
            --master_addr localhost --master_port 29000 \
            train_gpt2_distributed.py \
            --data_dir ~/data/datasets/fineweb/sample/10BT_tokenized/ \
            --training_mode fsdp \
            --batch 4 \
            --grad_accum_steps 4 \
            --epochs 3 \
            --save_dir ./checkpoints \
            --log_dir ./logs


    # --------------------------------------------------------------------------------------------
    # Distributed training with (2 nodes x 4 GPUs each) via torchrun and FSDP
    # See scripts/run_training_distributed_fsdp_main.sh
    # See scripts/run_training_distributed_fsdp_worker.sh
    # --------------------------------------------------------------------------------------------
    # MAIN:
    torchrun --nnodes 2 --nproc_per_node 4 --node_rank 0 \
            --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
            train_gpt2_distributed.py \
            --data_dir ~/data/datasets/fineweb/sample/10BT_tokenized/ \
            --training_mode fsdp \
            --batch 4 \
            --grad_accum_steps 4 \
            --epochs 3 \
            --save_dir ./checkpoints \
            --log_dir ./logs

    # WORKER
    torchrun --nnodes 2 --nproc_per_node 4 --node_rank 1 \
            --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
            train_gpt2_distributed.py \
            --data_dir ~/data/datasets/fineweb/sample/10BT_tokenized/ \
            --training_mode fsdp \
            --batch 4 \
            --grad_accum_steps 4 \
            --epochs 3 \
    ```

    """
    # 0. Set seed and environment variables.
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Reduce memory fragmentation.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 1. Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to data directory containing .bin files.",
    )
    parser.add_argument(
        "--training_mode", choices=["fsdp", "ddp", "local"], default="local"
    )
    parser.add_argument(
        "--device", required=False, help="Device to use for training.", default="cuda"
    )
    parser.add_argument(
        "--seq_len", type=int, default=gpt2_dataloader.DEFAULT_CONTEXT_LENGTH
    )
    parser.add_argument("--batch", type=int, default=gpt2_dataloader.DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps to simulate larger batch size",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=1000)  # steps
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--workers", type=int, default=gpt2_dataloader.DEFAULT_N_PROCS)
    args = parser.parse_args()

    # 2. Distributed init (only for distributed modes).
    if args.training_mode in ["fsdp", "ddp"]:
        init_distributed()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 3. Device setup.
    print_device_info()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available.")

    device = torch.device(args.device, local_rank)

    if is_primary():
        print(
            f"==> started rank {rank}/{world_size} on GPU {local_rank} in "
            f"{args.training_mode.upper()} mode"
        )
        print(f"==> Micro batch: {args.batch}, Gradient accum: {args.grad_accum_steps}")
        print(f"==> Sequence length: {args.seq_len}")

    # 4. Dataset and loader setup.
    ds = gpt2_dataloader.TokenShardDataset(
        shard_paths=gpt2_dataloader.get_shard_paths(
            pathlib.Path(args.data_dir), split="train"
        ),
        seq_len=args.seq_len,
        shuffle=True,
    )
    dl = gpt2_dataloader.create_dataloader(
        ds=ds,
        batch_size=args.batch,
        num_workers=args.workers,
    )

    # 5. Model setup.
    config = dataclasses.replace(DEFAULT_CONFIG, n_positions=args.seq_len)
    model = setup_model(config=config, device=device, training_mode=args.training_mode)

    # 6. Optimizer setup.
    # TODO: Use a cosine annealing scheduler for the learning rate.
    # NOTE: Use fused AdamW in bfloat16 to save –20 % optimiser memory + speed.
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),  # bf16-safe defaults
        fused=True,  # CUDA 12.x, large speed win
    )

    # 7. Tracker setup.
    tracker = StatsTracker(
        tb_dir=args.log_dir,
        batch_size=args.batch * args.grad_accum_steps,  # Effective batch size
        seq_len=args.seq_len,
        world_size=world_size,
        tb_every=1,  # log every step to TensorBoard
        cli_every=20,  # print each 20th step
    )

    # 8. Training loop.
    global_step = 0

    # Initialize gradients
    optim.zero_grad()

    # Print initial memory state only on primary process
    if is_primary():
        print(
            f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated"
        )

    for epoch in range(args.epochs):
        ds.set_epoch(epoch)  # ensures new shard order
        tracker.start_epoch(epoch)  # Start timing the epoch

        if is_primary():
            print(f"\n==== Epoch {epoch} ====")

        epoch_batch_count = 0
        accum_step = 0

        for _, (x, y) in enumerate(dl):
            # Move data to device.
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward pass (model returns logits and loss).
            # NOTE: autocast is used to cast the model to a lower precision and memory-intensive
            #       data type.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Compute loss.
                _, loss = model(x, labels=y)

                # Scale loss for gradient accumulation
                loss = loss / args.grad_accum_steps

            # Backward pass.
            loss.backward()

            # Only step optimizer after accumulating gradients
            accum_step += 1
            if accum_step % args.grad_accum_steps == 0:
                # Calculate gradient norm BEFORE optimizer step (while gradients still exist and
                # are not zeroed out yet).
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf")
                )

                # Step optimizer and reset gradients.
                optim.step()
                optim.zero_grad()

                # Update global step and counters only after optimizer step
                global_step += 1
                epoch_batch_count += 1

                # Log detailed progress statistics.
                tracker.update(
                    step=global_step,
                    loss=(
                        loss.item() * args.grad_accum_steps
                    ),  # Unscale loss for logging
                    lr=optim.param_groups[0]["lr"],
                    grad_norm=grad_norm,
                    epoch=epoch,
                    batch=epoch_batch_count,
                )

                # Clear CUDA cache more frequently for distributed training
                if args.training_mode in ["fsdp", "ddp"] and global_step % 50 == 0:
                    torch.cuda.empty_cache()
                elif global_step % 100 == 0:
                    torch.cuda.empty_cache()

                # Save checkpoint.
                if global_step % args.save_every == 0:
                    if dist.is_initialized():
                        dist.barrier()

                    save_checkpoint(model, optim, global_step, args.save_dir)

                    if dist.is_initialized():
                        dist.barrier()

    # 8. Final checkpoint.
    if is_primary():
        save_checkpoint(model, optim, global_step, args.save_dir)

    # 9. Clean up distributed resources (if set up) and TensorBoard writer.
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    # 10. Close tracker.
    tracker.close()


if __name__ == "__main__":
    main()
