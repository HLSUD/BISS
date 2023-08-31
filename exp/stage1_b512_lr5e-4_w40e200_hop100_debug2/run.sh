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
    --batch_size 512 \
    --lr 5e-4 \
    --warmup_epochs 40 \
    --num_epoch 200 \
    --hop_size 100 \
    --close_wandb \
    --output_path ${OUTPUT_DIR}