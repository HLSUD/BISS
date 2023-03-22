import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from models.LNR.train import train, eval

def main(model_config = None):
    modelConfig = {
        "type": "eval", # train or eval
        "num_channels": 64,
        "num_times": 30,
        "output_size": 257,
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 0,
        "device": "cuda:1",
        "train_data_dir": "./data/train/",
        "test_data_dir": "./data/test/",
        "eeg_data_name": "eeg_data.npy",
        "eigen_data_name": "coch_imags.npy",
        "save_weight_dir": "./checkpoints/",
        "load_weights": True,
        "ckpt_path": "./checkpoints/lnr_egn_ckpt_0.pt"
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["type"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()