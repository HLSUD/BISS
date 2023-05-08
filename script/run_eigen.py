import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:20480"
from models.train import train, eval

def main(model_config = None):
    modelConfig = {
        "type": "train", # train or eval
        "model_name": "LNR", # CNN or LNR
        "num_channels": 64,
        "num_times": 500,
        "output_size": 34480,
        "output_type": 0,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 0.001,
        "device": "cpu",
        "train_data_dir": "./data/train/",
        "val_data_dir": "./data/val/",
        "test_data_dir": "./data/test/",
        "eeg_data_dir": "./data/",
        "eeg_data_name": "eeg_data.npy",
        "coch_img_name": "spec_idx.csv",
        "save_weight_dir": "/mnt/nvme-ssd/hliuco/Documents/data/BISS/checkpoints/multigpu_lnr/",
        "load_weights": False,
        "ckpt_path": "/mnt/nvme-ssd/hliuco/Documents/data/BISS/checkpoints/multigpu_lnr/ckps/LNR_corr_ckpt_19.pth.tar",
        "image_data_dir": '/mnt/nvme-ssd/hliuco/Documents/data/BISS/images/spectrogram/'
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["type"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()