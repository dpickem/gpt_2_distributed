# Activate the right conda env.
# conda activate gpt_2_distributed

# Tell the allocator to defragment.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"

# NOTE: This runs with a fairly small batch size of 4 since a batch size of 16
#       led to OOM errors on a NVIDIA RTX 5000 with 32GB.
python ../train_gpt2_distributed.py \
  --data_dir ~/data/datasets/fineweb/sample/10BT_tokenized/ \
  --device cuda \
  --training_mode local \
  --batch 4 \
  --grad_accum_steps 4 \
  --epochs 3 \
  --save_dir ../checkpoints \
  --log_dir ../logs
