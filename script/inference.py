### wandb amp

import os
from typing import Dict
import logging

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
# import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import wandb
from models.LNR.lnr_eigen import LNR_eigen
from models.egg_dataset import coch_set
from models.NN.dnn_basic import dnn_basic

def eval(modelConfig: Dict, model = None, dataloader = None):
    ## eeg start index
    eeg_indices = []
    # get args
    type = modelConfig["type"]
    model_name = modelConfig["model_name"]
    subj_name = modelConfig["subj_name"]
    num_channels = modelConfig["num_channels"]
    num_times = modelConfig["num_times"]
    output_size = modelConfig["output_size"]
    load_weights = modelConfig["load_weights"]
    ckpt_path = modelConfig["ckpt_path"]
    batch_size = modelConfig["batch_size"]
    train_data_dir =  modelConfig["train_data_dir"]
    val_data_dir =  modelConfig["val_data_dir"]
    test_data_dir =  modelConfig["test_data_dir"]
    eeg_data_dir =  modelConfig["eeg_data_dir"]
    eeg_data_name =  modelConfig["eeg_data_name"]
    coch_img_name =  modelConfig["coch_img_name"]
    eeg_indices = modelConfig["eeg_indices"]
    output_type = modelConfig["output_type"]
    image_data_dir = modelConfig["image_data_dir"]
    device = modelConfig["device"]

    eeg_data_name = subj_name + '_' + eeg_data_name
    input_size = num_channels * num_times

    infer_data_dir = None
    res_save_dir = None
    if type == 'train':
        infer_data_dir = train_data_dir
        res_save_dir = '/train/'
    elif type == 'val':
        infer_data_dir = val_data_dir
        res_save_dir = '/val/'
    elif type == 'test':
        infer_data_dir = test_data_dir
        res_save_dir = '/test/'
    # device
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    # load the weights
    ckpt = None
    if model is None:
        if model_name == 'LNR':
            model = LNR_eigen(input_size=input_size, output_size=output_size)
        elif model_name == 'CNN':
            model = dnn_basic(input_size=input_size, output_size=output_size)
        if device == 'cuda':
            model= nn.DataParallel(model)
            print('Using GPUs for the inference...')
        model.to(device)
        if load_weights:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            # model.load_state_dict(ckpt)
            state_dict =ckpt['state_dict']
            
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                if 'module' not in k:
                    k = 'module.'+k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict[k]=v

            model.load_state_dict(new_state_dict)
            logging.info(f"Load weight from " + ckpt_path)
        else :
            logging.info(f"Please give the ckp path...")
            return
    

    dataset = coch_set(eeg_file = eeg_data_dir + eeg_data_name,
                                coch_img_file = infer_data_dir + coch_img_name,
                                data_dir= image_data_dir,
                                eeg_merge_size = num_times,
                                eeg_hop_size= 10,
                                output_type = output_type
                                )
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    similarity_score = []
    model.eval()
    with torch.no_grad():
        for idx in eeg_indices:
            x, y = dataset[idx]
            x = x[None, :]
            # print(x)
            # print(x.shape,y.shape)
            x = x.to(device)
            y = y.to(device)
            pred_spec = model(x)
            
            
            # difference between scores and y NOT IMPLEMENTED YET

            # save spectrogram as numpy array
            # spectrogram = 20 * torch.log10(torch.clamp(torch.reshape(pred_spec, (80, 431)), min=1e-5)) - 20
            # spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
            # print(spectrogram.shape)
            # np.save('../diffwave/train_spec_'+str(idx)+'.npy',spectrogram.cpu().numpy())
            image_name = dataset.get_image_name(idx).split('.')[0]
            image = np.squeeze(pred_spec.cpu().numpy(), axis=0).reshape(80,431)
            plt.imsave('data/outputs/'+ subj_name + res_save_dir +image_name + model_name +'.png', image)
            img = Image.fromarray(image)
            img.save('data/outputs/'+subj_name + res_save_dir + image_name + model_name + '.tiff')
            true_img = Image.fromarray(y.cpu().numpy().reshape(80,431))
            true_img.save('data/outputs/'+subj_name +'/true/'+image_name+'.tiff')
            

    model.train()
    return similarity_score

def main(model_config = None):
    modelConfig = {
        "type": "test",
        "model_name": "CNN", # CNN or LNR
        "subj_name": "subj1",
        "num_channels": 64,
        "num_times": 500,
        "output_size": 34480,
        "output_type": 0,
        "epochs": 100,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "weight_decay": 0.001,
        "device": "cuda",
        "train_data_dir": "./data/train/",
        "val_data_dir": "./data/val/",
        "test_data_dir": "./data/test/",
        "eeg_data_dir": "./data/eeg_data/",
        "eeg_data_name": "eeg_data.npy",
        "coch_img_name": "spec_idx.csv",
        "save_weight_dir": "/mnt/nvme-ssd/hliuco/Documents/data/BISS/checkpoints/multigpu_lnr/",
        "load_weights": True,
        "ckpt_path": "/mnt/nvme-ssd/hliuco/Documents/data/BISS/checkpoints/multicpu_cnn/subj1/CNN_corr_ckpt_19.pth.tar",
        "image_data_dir": '/mnt/nvme-ssd/hliuco/Documents/data/BISS/images/spectrogram/',
        "eeg_indices": [0,1,3]
        }
    if model_config is not None:
        modelConfig = model_config
    
    eval(modelConfig)

if __name__ == '__main__':
    main()