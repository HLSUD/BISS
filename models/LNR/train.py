from tkinter.tix import Tree
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from models.LNR.lnr_eigen import LNR_eigen
from models.egg_dataset import coch_set
def train(modelConfig: Dict):

    # get args
    num_channels = modelConfig["num_channels"]
    num_times = modelConfig["num_times"]
    output_size = modelConfig["output_size"]
    num_epochs = modelConfig["epochs"]
    learning_rate = modelConfig["learning_rate"]
    batch_size = modelConfig["batch_size"]
    data_dir =  modelConfig["train_data_dir"]
    eeg_data_name =  modelConfig["eeg_data_name"]
    eigen_data_name =  modelConfig["eigen_data_name"]
    weight_dir = modelConfig["save_weight_dir"]
    load_weights = modelConfig["load_weights"]
    ckpt_path = modelConfig["ckpt_path"]
    weight_decay = modelConfig["weight_decay"]

    input_size = num_channels * num_times
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Load data from "+ data_dir +"...")
    # load data
    dataset = coch_eigen_set(eeg_file = data_dir + eeg_data_name,
                                coch_eigen_np_file = data_dir + eigen_data_name,
                                eeg_merge_size = num_times,
                                )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initlize network
    model = LNR_eigen(input_size=input_size, output_size=output_size).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss() ## l2
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose = True)

    if load_weights:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
        print("Load weight from " + ckpt_path)

    # add tqdm + lr scheduler + save checkpoint
    # Train
    for epoch in range(num_epochs):
        losses = []

        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for data, targets in tqdmDataLoader:
                data = data.to(device)
                data = data.reshape(data.shape[0],-1)
                targets = targets.to(device)

                # forward
                scores = model(data)
                loss = criterion(scores, targets)
                losses.append(loss.item())

                # backward
                optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                optimizer.step()

                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": epoch,
                    "loss": loss.item(),
                    "data shape: ": data.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        # lr_scheduler
        mean_loss = sum(losses) / len(losses)
        lr_scheduler.step(mean_loss)
        torch.save(model.state_dict(), os.path.join(
            weight_dir, 'lnr_egn_ckpt_' + str(epoch) + ".pt"))
    return

def eval(modelConfig: Dict):

    # get args
    num_channels = modelConfig["num_channels"]
    num_times = modelConfig["num_times"]
    output_size = modelConfig["output_size"]
    load_weights = modelConfig["load_weights"]
    ckpt_path = modelConfig["ckpt_path"]
    batch_size = modelConfig["batch_size"]
    data_dir =  modelConfig["test_data_dir"]
    eeg_data_name =  modelConfig["eeg_data_name"]
    eigen_data_name =  modelConfig["eigen_data_name"]

    input_size = num_channels * num_times

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load weights and evaluate
    
    model = LNR_eigen(input_size=input_size, output_size=output_size).to(device)
    criterion = nn.MSELoss() ## l2

    # load the weights
    ckpt = None
    if load_weights:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
        print("Load weight from " + ckpt_path)
    else :
        print("Please give the ckp path...")
        return
    # load data
    dataset = coch_eigen_set(eeg_file = data_dir + eeg_data_name,
                                coch_eigen_np_file = data_dir + eigen_data_name,
                                eeg_merge_size = num_times,
                                )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.eval()
    with torch.no_grad():
        losses = []
        for x,y in dataloader:
            x = x.to(device)
            y = y.to(device)
            eigen_vs = model(x)
            loss = criterion(eigen_vs, y)
            losses.append(loss.item())
            # torch.save(eigen_vs, 'eigen_vs.pt')
        mean_loss = sum(losses) / len(losses)
        print("The average loss for test set is %.8lf" % (mean_loss))
    return
