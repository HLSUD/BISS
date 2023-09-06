# utilities for models
import numpy as np
import math
import torch
import os
import scipy.fftpack
from scipy.signal import savgol_filter
import argparse
import yaml

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def read_config_as_args(config_path,args=None,is_config_str=False):
    return_dict = {}

    if config_path is not None:
        if is_config_str:
            yml_config = yaml.load(config_path, Loader=yaml.FullLoader)
        else:
            with open(config_path, "r") as f:
                yml_config = yaml.load(f, Loader=yaml.FullLoader)

        if args != None:
            for k, v in yml_config.items():
                if k in args.__dict__:
                    args.__dict__[k] = v
                else:
                    sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))
        else:
            for k, v in yml_config.items():
                return_dict[k] = v

    args = args if args != None else return_dict
    return argparse.Namespace(**args)

def smooth_signal(signal, weight_threshold=100, keep_ratio=0.05, savgol=True, win=7, poly=1):
    x_axis = np.arange(0, signal.shape[-1])
    # to frequent domain
    w = scipy.fftpack.rfft(signal)
    f = scipy.fftpack.rfftfreq(signal.shape[-1], x_axis[1]-x_axis[0])
    spectrum = w**2
    # remove f with small value
    cutoff_idx = spectrum < (spectrum.max()/weight_threshold)
    if (cutoff_idx.sum() / len(cutoff_idx)) > (1-keep_ratio):
        idx = int((1-keep_ratio)*signal.shape[-1])
        cutoff_idx = spectrum < np.sort(spectrum)[idx]
    w2 = w.copy()
    w2[cutoff_idx] = 0
    s_smooth = scipy.fftpack.irfft(w2)

    # Savitzky-Golay filter
    if savgol:
        s_smooth = savgol_filter(s_smooth, win, poly, mode='nearest')
    
    return s_smooth

def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_l = np.arange(length, dtype=np.float32)

    grid_l = grid_l.reshape([1, length])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_l)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # cls token
        # height (== width) for the checkpoint position embedding
        orig_size = int(pos_embed_checkpoint.shape[-2] - num_extra_tokens)
        # height (== width) for the new position embedding
        new_size = int(num_patches)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %d to %d" % (orig_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, embedding_size).permute(0, 2, 1)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size))
            pos_tokens = pos_tokens.permute(0, 2, 1)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed



def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config.warmup_epochs:
        lr = config.lr * epoch / config.warmup_epochs 
    else:
        lr = config.min_lr + (config.lr - config.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config.warmup_epochs) / (config.num_epoch - config.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def save_model(config, epoch, model, optimizer, loss_scaler, checkpoint_paths):
    os.makedirs(checkpoint_paths, exist_ok=True)
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'config': config,
    }
    torch.save(to_save, os.path.join(checkpoint_paths, 'checkpoint.pth'))
    

def load_model(config, model, checkpoint_path ):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print(f'Model loaded with {checkpoint_path}')

def patchify(imgs, patch_size):
    """
    imgs: (N, 1, num_voxels)
    x: (N, L, patch_size)
    """
    p = patch_size
    assert imgs.ndim == 3 and imgs.shape[2] % p == 0

    h = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], h, p))
    return x

def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size)
    imgs: (N, 1, num_voxels)
    """
    p = patch_size
    h = x.shape[1]
    
    imgs = x.reshape(shape=(x.shape[0], 1, h * p))
    return imgs
