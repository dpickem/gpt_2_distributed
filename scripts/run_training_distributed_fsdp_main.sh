# Activate the right conda env.
# conda activate gpt_2_distributed

# Tell the allocator to defragment.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"

# NOTE: This runs with a fairly small batch size of 4 since a batch
#       size of 16 led to OOM errors on a NVIDIA RTX 5000 with 32GB.
# TODO: Depending on which machines this runs on, the batch size can be
#       increased beyond 4.

# -----------------------------------------------------------------------
# Distributed training with (2 nodes x 4 GPUs each) via torchrun and FSDP
# -----------------------------------------------------------------------
# MAIN:
torchrun \
  --nnodes 2 \
  --nproc_per_node 4 \
  --node_rank 0 \
  --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
  train_gpt2_distributed.py \
  --data_dir ~/data/datasets/fineweb/sample/10BT_tokenized/ \
  --training_mode fsdp \
  --batch 4 \
  --grad_accum_steps 4 \
  --epochs 3 \
  --save_dir ./checkpoints \
  --log_dir ./logs
