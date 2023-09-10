# https://github.com/moein-shariatnia/OpenAI-CLIP
# https://github.com/microsoft/CLAP/blob/main/src/CLAPWrapper.py

### AvgMeter ???? get_lr???
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import datetime
import time
from models.NN.nle import NLE
import re
import numpy as np
from models.utils import read_config_as_args, save_model
import torch
from importlib_resources import files
import pandas as pd
from tqdm import tqdm
from models.eeg_dataset import NLEDataset
from models.eeg_dataset import split_dataset
from models.NN.nle import NLE
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from models.utils import AvgMeter, get_lr
import timm.optim.optim_factory as optim_factory

class NLEWrapper():
    """
    A class for interfacing NLE model.  
    """

    def __init__(self, eeg_pretrain_model = None, use_cuda=False):
        self.np_str_obj_array_pattern = re.compile(r'[SaUO]')
        self.file_path = os.path.realpath(__file__)
        self.default_collate_err_msg_format = (
            "default_collate: batch must contain tensors, numpy arrays, numbers, "
            "dicts or lists; found {}")
        self.config_as_str = files('config').joinpath('config.yml').read_text()
        self.eeg_pretrain_model = eeg_pretrain_model # file path
        self.use_cuda = use_cuda
        self.nle, self.args = self.load_nle()
        if eeg_pretrain_model is None:
            self.eeg_pretrain_model = self.args.eeg_pretrain_path

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
            args.audioenc_name,
            args.audio_model,
            args.processor_model,
            args.transformer_embed_dim,
            args.out_dims,
            args.trainable,
            args.temperature
            )

        # Load eeg pretrained weights for model
        if self.eeg_pretrain_model is not None:
            model_state_dict = torch.load(self.eeg_pretrain_model, map_location=torch.device('cpu'))['model']
            nle.neuro_encoder.load_state_dict(model_state_dict)

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


    def build_loaders(self, eeg_path, audio_path, type = 'train'):
        args = read_config_as_args(self.config_as_str, is_config_str=True)
        # transforms = get_transforms(mode=mode)
        ### data sample revise
        split_dataset(eeg_path=eeg_path, hop_size=self.args.hop_size)
        csv_file = None
        if type == 'train':
            csv_file = 'data/train_idx_name.csv'
        elif type == 'val':
            csv_file = 'data/val_idx_name.csv'
        elif type == 'test':
            csv_file = 'test/train_idx_name.csv'

        dataset = NLEDataset(eeg_path, audio_path, csv_file, self.args.hop_size, self.args.smooth)
        print(dataset[0]['eeg'])
        print(dataset[0]['audio'])
        print(f'Dataset size: {len(dataset)}')
        sampler = torch.utils.data.DistributedSampler(dataset, rank=args.local_rank) if torch.cuda.device_count() > 1 else None 

        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, 
                    shuffle=(sampler is None), pin_memory=True)
        
        return dataloader, sampler


    def train_epoch(self, model, train_loader, optimizer, device):
        # args = read_config_as_args(self.config_as_str, is_config_str=True)
        ## revise
        loss_meter = AvgMeter()
        tqdm_object = tqdm(train_loader, total=len(train_loader))
        for batch in tqdm_object:
            #batch = batch.to(self.args.device)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(batch['audio'],batch['eeg'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = batch["eeg"].size(0)
            loss_meter.update(loss.item(), count)

            tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
        return loss_meter


    def valid_epoch(self, model, valid_loader, device):

        loss_meter = AvgMeter()

        tqdm_object = tqdm(valid_loader, total=len(valid_loader))
        for batch in tqdm_object:
            batch = {k: v.to(device) for k, v in batch.items()}

            loss = model(batch['audio'],batch['eeg'])


            count = batch["eeg"].size(0)
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
        eeg_path = './data/eeg_data/'
        audio_path = './data/audio_data/'
        # dataset = NLEDataset(eeg_path, audio_path)
    
        # print(f'Dataset size: {len(dataset)}\n Time len: {dataset.data_len}')
        # sampler = torch.utils.data.DistributedSampler(dataset, rank=args.local_rank) if torch.cuda.device_count() > 1 else None 

        # dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, 
                    # shuffle=(sampler is None), pin_memory=True)
        dataloader, sampler = self.build_loaders(eeg_path,audio_path)
        valid_loader, _ = self.build_loaders(eeg_path,audio_path, type='val')
        # create model
        # args.time_len=dataset.data_len
        model = self.nle
        model.to(device)
        model_without_ddp = model
        if torch.cuda.device_count() > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
        param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        print(optimizer)
        # loss_scaler = NativeScaler()
        ## load eeg model
        start_epoch = 0
        model_path = os.path.join(output_path, 'checkpoints', 'checkpoint.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % model_path)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch'] + 1
                # if 'scaler' in checkpoint:
                #     loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")


        if logger is not None:
            logger.watch_model(model,log='all', log_freq=1000)
        best_loss = -1
        loss_list = []
        val_loss_list = []
        start_time = time.time()
        print('Start Training the NLE ...')
        for ep in range(start_epoch, args.num_epoch):
            
            if torch.cuda.device_count() > 1: 
                sampler.set_epoch(ep) # to shuffle the data at every epoch
            model.train()
            ### revise
            loss = self.train_epoch(model, dataloader, optimizer, device)
            model.eval()
            loss_list.append(loss.avg)

            with torch.no_grad():
                valid_loss = self.valid_epoch(model, valid_loader, device)
                val_loss_list.append(valid_loss.avg)
                
            if best_loss == -1:
                best_loss = valid_loss.max
            
            if args.local_rank == 0 and valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                # torch.save(model.state_dict(), "best.pt")
                save_model(args, ep, model_without_ddp, optimizer, os.path.join(output_path, 'checkpoints'))
                print("NLE saved...")


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        if logger is not None:
            logger.log('Max train loss', np.max(loss_list), 'Max val loss', np.max(val_loss_list), step=args.num_epoch-1)
            logger.finish()


if __name__ == "__main__":
    nle_trainer = NLEWrapper()
    nle_trainer.train_nle()
#    split_dataset(eeg_path='./data/eeg_data/', hop_size=100)
