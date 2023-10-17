export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

OUTPUT_DIR="$(dirname $0)"

srun -p video3 -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=128 \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=10066 models/train_eeg_mae.py \
    --batch_size 256 \
    --lr 5e-5 \
    --mask_ratio 0.75 \
    --warmup_epochs 0 \
    --num_epoch 100 \
    --hop_size 100 \
    --close_wandb \
    --smooth \
    --add_cor_loss \
    --resume /mnt/petrelfs/likunchang/eeg/BISS/exp/stage1_b256_lr1e-4_w20e200_hop100_smooth_cor_m75/checkpoints/checkpoint.pth \
    --output_path ${OUTPUT_DIR}