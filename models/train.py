### wandb amp

import os
from typing import Dict
from collections import OrderedDict
import logging


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

import wandb
from models.LNR.lnr_eigen import LNR_eigen
from models.egg_dataset import coch_set
from models.NN.dnn_basic import dnn_basic

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    logging.info(f"=> Saving checkpoint %s" % (filename))
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, num_gpu = 1):
    logging.info(f"=> Loading checkpoint")
    state_dict =checkpoint['state_dict']   
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k]=v
    if num_gpu > 1:
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint["optimizer"])


def corr_loss(output, target):
    output_sub = output - torch.mean(output)
    target_sub = target - torch.mean(target)


    output_var = torch.sum((output - torch.mean(output))**2)
    target_var = torch.sum((target - torch.mean(target))**2)
    
    return 1-torch.sum(output_sub*target_sub)/torch.sqrt(output_var*target_var)


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
    val_data_dir = modelConfig["val_data_dir"]
    eeg_data_dir =  modelConfig["eeg_data_dir"]
    eeg_data_name =  modelConfig["eeg_data_name"]
    coch_img_name =  modelConfig["coch_img_name"]
    weight_dir = modelConfig["save_weight_dir"]
    load_weights = modelConfig["load_weights"]
    ckpt_path = modelConfig["ckpt_path"]
    weight_decay = modelConfig["weight_decay"]
    output_type = modelConfig["output_type"]
    image_data_dir = modelConfig["image_data_dir"]

    input_size = num_channels * num_times
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    logging.info(f'Using device {device}')

    logging.info(f"Loading data from {train_data_dir} ...")
    
    # load data
    dataset = coch_set(eeg_file = eeg_data_dir + eeg_data_name,
                                coch_img_file = train_data_dir + coch_img_name,
                                data_dir= image_data_dir,
                                eeg_merge_size = num_times,
                                eeg_hop_size= 10,
                                output_type = output_type
                                )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # print(len(dataset),len(dataloader))
    val_dataset = coch_set(eeg_file = eeg_data_dir + eeg_data_name,
                                coch_img_file = val_data_dir + coch_img_name,
                                data_dir= image_data_dir,
                                eeg_merge_size = num_times,
                                eeg_hop_size= 10,
                                output_type = output_type
                                )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    
    model = None
    # Initlize network
    if model_name == 'LNR':
        model = LNR_eigen(input_size=input_size, output_size=output_size).to(device)
    elif model_name == 'CNN':
        model = dnn_basic(input_size=input_size, output_size=output_size).to(device)
    model= nn.DataParallel(model)
    model.to(device)
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
    # eval(modelConfig, model, dataloader)
    # Train
    for epoch in range(num_epochs):
        losses = []
        model.train()

        with tqdm(dataloader) as tqdmDataLoader:
            for data, targets in tqdmDataLoader:
                data = data.to(device)
                # print(data.shape)
                data = data.reshape(data.shape[0],-1)
                targets = targets.to(device)

                # forward
                scores = model(data)
                loss = criterion(scores, targets) + corr_loss(scores, targets)
                # loss = criterion(scores, targets)
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
                eval_record_times = 10
                division_step = (len(dataset) // (eval_record_times * batch_size))
                # division_step = 0
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                        
                        val_score = eval(modelConfig, model, val_dataloader)
                        lr_scheduler.step(val_score)

                        logging.info('Epoch: %d, step: %d Validation score: %.8lf' % (epoch, global_step, val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation score': val_score,
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass
        # lr_scheduler
        mean_loss = sum(losses) / len(losses)
        logging.info(f"The average loss for train set is %.8lf" % (mean_loss))
        # print("The average loss for train set is %.8lf" % (mean_loss))
        # lr_scheduler.step(mean_loss)



        # model.module.state_dict()
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        # save checkpoint
        save_checkpoint(checkpoint, os.path.join(
            weight_dir,model_name + '_corr_ckpt_' + str(epoch) + ".pth.tar"))
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
    eeg_data_dir =  modelConfig["eeg_data_dir"]
    eeg_data_name =  modelConfig["eeg_data_name"]
    coch_img_name =  modelConfig["coch_img_name"]

    input_size = num_channels * num_times

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load weights and evaluate
    criterion = nn.MSELoss() ## l2

    # load the weights
    ckpt = None
    if model is None:
        if model_name == 'LNR':
            model = LNR_eigen(input_size=input_size, output_size=output_size).to(device)
        elif model_name == 'CNN':
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
        dataset = coch_set(eeg_file = eeg_data_dir + eeg_data_name,
                                    coch_img_file = test_data_dir + coch_img_name,
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
            scores = model(x)
            # loss = criterion(eigen_vs, y)
            # add pearson loss
            loss = criterion(scores, y) + corr_loss(scores, y)

            losses.append(loss.item())
            # cos_v2 = torch.sum(eigen_vs[:,:256] * y[:,:256],1)
            # ratio = (-torch.log(eigen_vs[:,-1]) - 1) / (-torch.log(y[:,-1]) - 1)
            # len_cond = torch.logical_and(ratio<2,ratio>0.5)
            # dice_score = dice_score + torch.sum(torch.logical_and(cos_v2>0.7,len_cond)).item()

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
    # dice_score = dice_score/num_val_batches
    dice_score = mean_loss
    model.train()
    return dice_score