import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:20480"
from models.train_latent_encoders import train, eval

def main(model_config = None):
    modelConfig = {
        "type": "train", # train or eval
        "model_name": "NMTrans", # CNN or LNR
        "num_subj": 50,
        "num_channels": 64,
        "num_times": 60000,
        "merge_size": 100,
        "output_size": 6960,
        "output_type": 0,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 0.001,
        "device": "cuda",
        "train_data_dir": "./data/train/",
        "val_data_dir": "./data/val/",
        "test_data_dir": "./data/test/",
        "eeg_data_dir": "/scratch/ResearchGroups/lt_jixingli/honghaoliu/eeg/50/",
        "spec_img_name": "spec_idx_all.csv",
        "save_weight_dir": "/scratch/ResearchGroups/lt_jixingli/honghaoliu/BISS/checkpoints/multigpu_nmtrans/",
        "load_weights": False,
        "ckpt_path": "/scratch/ResearchGroups/lt_jixingli/honghaoliu/BISS/checkpoints/multigpu_cnn/subj1/keep/CNN_corr_ckpt_86.pth.tar",
        "image_data_dir": '/scratch/ResearchGroups/lt_jixingli/honghaoliu/BISS/images/spec/',
        "beta": 1,
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["type"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()