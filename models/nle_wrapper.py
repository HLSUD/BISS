# https://github.com/moein-shariatnia/OpenAI-CLIP
# https://github.com/microsoft/CLAP/blob/main/src/CLAPWrapper.py

### AvgMeter ???? get_lr???

import datetime
import time
from models.NN.nle import NLE
import re
import numpy as np
from models.utils import read_config_as_args
import os
import torch
from importlib_resources import files
import pandas as pd
from tqdm import tqdm
from models.eeg_dataset import NLEDataset
from models.NN.nle import NLE
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from models.utils import AvgMeter, get_lr
import timm.optim.optim_factory as optim_factory

class NLEWrapper():
    """
    A class for interfacing NLE model.  
    """

    def __init__(self, model_fp = None, use_cuda=False):
        self.np_str_obj_array_pattern = re.compile(r'[SaUO]')
        self.file_path = os.path.realpath(__file__)
        self.default_collate_err_msg_format = (
            "default_collate: batch must contain tensors, numpy arrays, numbers, "
            "dicts or lists; found {}")
        self.config_as_str = files('configs').joinpath('config.yml').read_text()
        self.model_fp = model_fp # file path
        self.use_cuda = use_cuda
        self.nle, self.args = self.load_nle()

    def load_nle(self):
        r"""Load NLE model with args from config file"""

        args = read_config_as_args(self.config_as_str, is_config_str=True)

        if args.text_cl:
            if 'bert' in args.text_model:
                self.token_keys = ['input_ids', 'token_type_ids', 'attention_mask']
            else:
                self.token_keys = ['input_ids', 'attention_mask']


        nle = NLE(
            args.channels,
            args.timepoints,
            args.embed_dim,
            args.depth,
            args.heads,
            args.audio_model,
            args.processor_model,
            args.transformer_embed_dim,
            args.out_dims,
            args.temperature
            )

        # Load pretrained weights for model
        if self.model_fp is not None:
            model_state_dict = torch.load(self.model_fp, map_location=torch.device('cpu'))['model']
            nle.load_state_dict(model_state_dict)

        nle.eval()  # set nle in eval mode
    

        if self.use_cuda and torch.cuda.is_available():
            nle = nle.cuda()

        return nle, args


    def make_train_valid_dfs(self):
        ### revise
        args = read_config_as_args(self.config_as_str, is_config_str=True)
        dataframe = pd.read_csv(f"{args.captions_path}/captions.csv")
        max_id = dataframe["id"].max() + 1 if not args.debug else 100
        image_ids = np.arange(0, max_id)
        np.random.seed(42)
        valid_ids = np.random.choice(
            image_ids, size=int(0.2 * len(image_ids)), replace=False
        )
        train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
        train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
        valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
        return train_dataframe, valid_dataframe


    def build_loaders(self, eeg_path, audio_path):
        args = read_config_as_args(self.config_as_str, is_config_str=True)
        # transforms = get_transforms(mode=mode)
        ### data sample revise
        
        # eeg_path = './data/eeg_data/', 
        # audio_path = './data/audio_data/'
        dataset = NLEDataset(eeg_path, audio_path)
    
        print(f'Dataset size: {len(dataset)}\n Time len: {dataset.data_len}')
        sampler = torch.utils.data.DistributedSampler(dataset, rank=args.local_rank) if torch.cuda.device_count() > 1 else None 

        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, 
                    shuffle=(sampler is None), pin_memory=True)
        
        return dataloader


    def train_epoch(self, model, train_loader, optimizer, lr_scheduler, step):
        # args = read_config_as_args(self.config_as_str, is_config_str=True)
        ## revise
        loss_meter = AvgMeter() 
        tqdm_object = tqdm(train_loader, total=len(train_loader))
        for batch in tqdm_object:
            # batch = {k: v.to(args.device) for k, v in batch.items() if k != "caption"}
            loss = model(batch['audio'],batch['eeg'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step == "batch":
                lr_scheduler.step()

            count = batch["eeg"].size(0)
            loss_meter.update(loss.item(), count)

            tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
        return loss_meter


    def valid_epoch(self, model, valid_loader):

        loss_meter = AvgMeter()

        tqdm_object = tqdm(valid_loader, total=len(valid_loader))
        for batch in tqdm_object:
            batch = {k: v.to(args.device) for k, v in batch.items() if k != "caption"}
            loss = model(batch)

            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)

            tqdm_object.set_postfix(valid_loss=loss_meter.avg)
        return loss_meter

    

    def train_nle(self):
        args = read_config_as_args(self.config_as_str, is_config_str=True)
        if torch.cuda.device_count() > 1:
            torch.cuda.set_device(args.local_rank) 
            torch.distributed.init_process_group(backend='nccl')
        output_path = os.path.join(args.root_path, 'results', 'nle',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
        args.output_path = output_path
        # logger = wandb_logger(config) if config.local_rank == 0 else None
        logger = None
        
        if args.local_rank == 0:
            os.makedirs(output_path, exist_ok=True)
            print(args.__dict__)
            with open(os.path.join(output_path, 'README.md'), 'w+') as f:
                print(args.__dict__, file=f)
            
        
        device = torch.device(f'cuda:{args.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # create dataset and dataloader
        eeg_path = './data/eeg_data/', 
        audio_path = './data/audio_data/'
        dataset = NLEDataset(eeg_path, audio_path)
    
        print(f'Dataset size: {len(dataset)}\n Time len: {dataset.data_len}')
        sampler = torch.utils.data.DistributedSampler(dataset, rank=args.local_rank) if torch.cuda.device_count() > 1 else None 

        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, 
                    shuffle=(sampler is None), pin_memory=True)

        # valid_loader = build_loaders(valid_df, tokenizer, mode="valid") check
        # create model
        args.time_len=dataset.data_len
        model = self.nle
        model.to(device)
        model_without_ddp = model
        if torch.cuda.device_count() > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=args.use_nature_img_loss)

        param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        print(optimizer)
        # loss_scaler = NativeScaler()

        if logger is not None:
            logger.watch_model(model,log='all', log_freq=1000)

        loss_list = []
        start_time = time.time()
        print('Start Training the NLE ...')
        
        for ep in range(args.num_epoch):
            
            if torch.cuda.device_count() > 1: 
                sampler.set_epoch(ep) # to shuffle the data at every epoch
            model.train()
            ### revise
            loss = self.train_epoch(model, dataloader, optimizer, lr_scheduler, step)
            model.eval()
            loss_list.append(loss)

            # save models
            if (ep % 20 == 0 or ep + 1 == args.num_epoch) and args.local_rank == 0: #and ep != 0
            # if True:
                save_model(args, ep, model_without_ddp, optimizer, loss_scaler, os.path.join(output_path,'checkpoints'))
            # with torch.no_grad():
            #     valid_loss = valid_epoch(model, valid_loader)
        
            # if valid_loss.avg < best_loss:
            #     best_loss = valid_loss.avg
            #     torch.save(model.state_dict(), "best.pt")
            #     print("Saved Best Model!")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        if logger is not None:
            logger.log('max cor', np.max(loss_list), step=args.num_epoch-1)
            logger.finish()


if __name__ == "__main__":
    nle_trainer = NLEWrapper()
    nle_trainer.train_nle()