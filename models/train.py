### wandb amp

import os
from typing import Dict
import logging


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

import wandb
from models.LNR.lnr_eigen import LNR_eigen
from models.egg_dataset import coch_eigen_set
from models.NN.dnn_basic import dnn_basic

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    logging.info(f"=> Saving checkpoint %s" % (filename))
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    logging.info(f"=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def train(modelConfig: Dict):

    logging.basicConfig(filename="train.log",level=logging.INFO, format='%(levelname)s: %(message)s')

    # get args
    model_name = modelConfig["model_name"]
    num_channels = modelConfig["num_channels"]
    num_times = modelConfig["num_times"]
    output_size = modelConfig["output_size"]
    num_epochs = modelConfig["epochs"]
    learning_rate = modelConfig["learning_rate"]
    batch_size = modelConfig["batch_size"]
    train_data_dir =  modelConfig["train_data_dir"]
    eeg_data_name =  modelConfig["eeg_data_name"]
    eigen_data_name =  modelConfig["eigen_data_name"]
    weight_dir = modelConfig["save_weight_dir"]
    load_weights = modelConfig["load_weights"]
    ckpt_path = modelConfig["ckpt_path"]
    weight_decay = modelConfig["weight_decay"]

    input_size = num_channels * num_times
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f'Using device {device}')

    logging.info(f"Loading data from {train_data_dir} ...")
    
    # load data
    dataset = coch_eigen_set(eeg_file = train_data_dir + eeg_data_name,
                                coch_eigen_np_file = train_data_dir + eigen_data_name,
                                eeg_merge_size = num_times,
                                )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = None
    # Initlize network
    if model_name == 'LNR_eigen':
        model = LNR_eigen(input_size=input_size, output_size=output_size).to(device)
    elif model_name == 'CNN_eigen':
        model = dnn_basic(input_size=input_size, output_size=output_size).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss() ## l2 + pearson??
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose = True)

    if load_weights:
        load_checkpoint(torch.load(ckpt_path, map_location=device), model, optimizer)
        logging.info(f"Loading weights from {ckpt_path} ...")
        
    # if load_weights:
    #     ckpt = torch.load(ckpt_path, map_location=device)
    #     model.load_state_dict(ckpt)
    #     print("Load weight from " + ckpt_path)


    # (Initialize logging) check
    ##  grad_scaler = torch.cuda.amp.GradScaler(enabled=amp) support multiple presicion
    experiment = wandb.init(project='BISS_'+model_name, resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)
    )

    logging.info(f'''Starting training:
        Epochs:          {num_epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(dataset)}
        Device:          {device}
    ''')
    global_step = 0
    # Train
    for epoch in range(num_epochs):
        losses = []
        model.train()

        with tqdm(dataloader) as tqdmDataLoader:
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

                # print(data[0])
                # print(scores[0])
                # logging.info(f"The loss for train set is %.8lf" % (loss.item()))

                global_step += 1
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })

                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": epoch,
                    "loss": loss.item(),
                    "data shape: ": data.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })

                # Evaluation round
                eval_record_times = 5
                division_step = (len(dataset) // (eval_record_times * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                        
                        score = eval(modelConfig, model, dataloader)
                        # lr_scheduler.step(val_score)

                        logging.info('Epoch: %d, Validation score: %.8lf' % (epoch, score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation score': score,
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass
        # lr_scheduler
        mean_loss = sum(losses) / len(losses)
        # logging.info(f"The average loss for train set is %.8lf" % (mean_loss))
        # print("The average loss for train set is %.8lf" % (mean_loss))
        lr_scheduler.step(mean_loss)



        
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        # save checkpoint
        save_checkpoint(checkpoint, os.path.join(
            weight_dir,model_name + '_ckpt_' + str(epoch) + ".pth.tar"))
        # torch.save(model.state_dict(), os.path.join(
        #     weight_dir, 'lnr_egn_ckpt_' + str(epoch) + ".pt"))
    return

def eval(modelConfig: Dict, model = None, dataloader = None):

    # get args
    model_name = modelConfig["model_name"]
    num_channels = modelConfig["num_channels"]
    num_times = modelConfig["num_times"]
    output_size = modelConfig["output_size"]
    load_weights = modelConfig["load_weights"]
    ckpt_path = modelConfig["ckpt_path"]
    batch_size = modelConfig["batch_size"]
    test_data_dir =  modelConfig["test_data_dir"]
    eeg_data_name =  modelConfig["eeg_data_name"]
    eigen_data_name =  modelConfig["eigen_data_name"]

    input_size = num_channels * num_times

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load weights and evaluate
    criterion = nn.MSELoss() ## l2

    # load the weights
    ckpt = None
    if model is None:
        if model_name == 'LNR_eigen':
            model = LNR_eigen(input_size=input_size, output_size=output_size).to(device)
        elif model_name == 'CNN_eigen':
            model = dnn_basic(input_size=input_size, output_size=output_size).to(device)
        if load_weights:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt)
            logging.info(f"Load weight from " + ckpt_path)
        else :
            logging.info(f"Please give the ckp path...")
            return
    
    if dataloader is None:
        # load data
        dataset = coch_eigen_set(eeg_file = test_data_dir + eeg_data_name,
                                    coch_eigen_np_file = test_data_dir + eigen_data_name,
                                    eeg_merge_size = num_times,
                                    )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    losses = []
    with torch.no_grad():
        # step = 0
        for x,y in dataloader:
            x = x.to(device)
            y = y.to(device)
            eigen_vs = model(x)
            loss = criterion(eigen_vs, y)
            losses.append(loss.item())
            cos_v2 = torch.sum(eigen_vs[:,:256] * y[:,:256],1)
            ratio = (-torch.log(eigen_vs[:,-1]) - 1) / (-torch.log(y[:,-1]) - 1)
            len_cond = torch.logical_and(ratio<2,ratio>0.5)
            dice_score = dice_score + torch.sum(torch.logical_and(cos_v2>0.7,len_cond)).item()

            # logging.info(f"The loss for test set is %.8lf" % (loss.item()))
            # print(x[0])
            # print(eigen_vs[0])
            # step = step + 1
            # if step == 2:
            #     model.train()
            #     return

            # torch.save(eigen_vs, 'eigen_vs.pt')
    mean_loss = sum(losses) / len(losses)
    logging.info(f"The average eval loss is %.8lf" % (mean_loss))

    model.train()
    return dice_score/num_val_batches