import os
import numpy as np

class Config_MAE_fMRI: # back compatibility
    pass
class Config_MBM_finetune: # back compatibility
    pass 

class Config_MBM_EEG(Config_MAE_fMRI):
    # configs for fmri_pretrain.py
    def __init__(self):
    # --------------------------------------------
    # MAE for fMRI
        # Training Parameters
        self.lr = 2.5e-4
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 500
        self.warmup_epochs = 40
        self.batch_size = 64
        self.clip_grad = 0.8
        self.hop_size = 10
        self.smooth = False
        self.add_cor_loss = False
        self.add_seg_cor_loss = False
        
        # Model Parameters
        self.mask_ratio = 0.1
        self.patch_size = 4 #  1
        self.embed_dim = 1024 #256 # has to be a multiple of num_heads
        self.decoder_embed_dim = 512 #128
        self.depth = 24
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0
        self.resume = ''

        # Project setting
        self.close_wandb = False
        self.output_path = ''
        self.seed = 2022
        self.roi = 'VC'
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = True
        self.accum_iter = 1

        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0