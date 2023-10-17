export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

OUTPUT_DIR="$(dirname $0)"
LOG_DIR="$(dirname $0)/log"

srun -p Gvlab-S1 -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=128 --quotatype=spot --async \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=10066 models/train_eeg_mae.py \
    --batch_size 256 \
    --lr 1e-4 \
    --mask_ratio 0.75 \
    --warmup_epochs 40 \
    --num_epoch 500 \
    --hop_size 100 \
    --close_wandb \
    --smooth \
    --add_cor_loss \
    --add_seg_cor_loss \
    --output_path ${OUTPUT_DIR}