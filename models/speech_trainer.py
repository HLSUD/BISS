#### https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/learner.py
# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import spawn

from tqdm import tqdm

# from diffwave.dataset import from_path, from_gtzan
from models.NN.latent2speech import DiffWave
from models.NN.latent2speech import whis2diffpooling
from models.eeg_dataset import Whisper_Collator, ConditionalDataset
from models.whisper.__init__ import load_model
from config.wave_config import Config_Wave
from argparse import ArgumentParser
import random

def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)

def from_path(data_dirs, params, is_distributed=False):
  #with condition
    # print(type(data_dirs),type(params.dataset_type))
    dataset = ConditionalDataset(data_dirs, params.dataset_type)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=not is_distributed,
        # collate_fn=Whisper_Collator(params).collate,
        num_workers=os.cpu_count(),
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=True)

class WhisperDiffWaveLearner:
  def __init__(self, model_dir, diff_model, whis_model, dataset, optimizer, params, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.diff_model = diff_model
    self.whis_model = whis_model
    
    self.dataset = dataset
    self.optimizer = optimizer
    self.params = params
    if params.audio_model == 'base':
        in_channel = 512
    self.pooling = whis2diffpooling(in_channel, self.params.num_channel)

    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True

    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = nn.L1Loss()
    self.summary_writer = None

  def state_dict(self):
    if hasattr(self.diff_model, 'module') and isinstance(self.diff_model.module, nn.Module):
      model_state = self.diff_model.module.state_dict()
    else:
      model_state = self.diff_model.state_dict()
    if hasattr(self.pooling, 'module') and isinstance(self.pooling.module, nn.Module):
      pool_state = self.pooling.module.state_dict()
    else:
      pool_state = self.pooling.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': {key:value for key, value in self.params.__dict__.items() if not key.startswith('__') and not callable(key)},
        'scaler': self.scaler.state_dict(),
        'pool_layer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in pool_state.items() },
    }

  def load_state_dict(self, state_dict):
    if hasattr(self.diff_model, 'module') and isinstance(self.diff_model.module, nn.Module):
      self.diff_model.module.load_state_dict(state_dict['model'])
    else:
      self.diff_model.load_state_dict(state_dict['model'])
    if hasattr(self.pooling, 'module') and isinstance(self.pooling.module, nn.Module):
      self.pooling.module.load_state_dict(state_dict['pool_layer'])
    else:
      self.pooling.load_state_dict(state_dict['pool_layer'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']

  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)

  def restore_from_checkpoint(self, filename='weights'):
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      print(f"Loading weights from {self.model_dir}...")
      return True
    except FileNotFoundError:
      return False

  def train(self, max_steps=None):
    print(f"Begin training on {self.params.device}...")
    print(f"Dataset size: {len(self.dataset)}")
    print(f"Batch size: {self.params.batch_size}")
    print(f"Learning rate: {self.params.learning_rate}, number of steps: {max_steps}")
    device = next(self.diff_model.parameters()).device

    while True:
      for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
        if max_steps is not None and self.step >= max_steps:
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        loss = self.train_step(features)
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.step % 50 == 0:
            self._write_summary(self.step, features, loss)
          if self.step % len(self.dataset) == 0:
            self.save_to_checkpoint()
        self.step += 1

  def train_step(self, features):
    for param in self.diff_model.parameters():
      param.grad = None

    audio = features['audio']
    spectrogram = features['spectrogram']
    ### preprocess
    # print('getting audio features...')
    audio_features = self.whis_model.embed_audio(spectrogram)

    ### random start position of audio feature
    start = random.randint(0, audio_features.shape[-1] - self.params.whisper_len)
    end = start+self.params.whisper_len
    audio_features = audio_features[:,start:end]
    samples_per_frame = self.params.sample_rate // self.params.token_num_per_sec
    start *= samples_per_frame
    end *= samples_per_frame
    audio = audio[:,start:end]
    audio = F.pad(audio, (0,(end-start) - audio.shape[-1]), "constant", 0)

    # print('projecting...')
    audio_features = self.pooling(audio_features)

    # print('audio genrating...')
    N, T = audio.shape
    device = audio.device
    self.noise_level = self.noise_level.to(device)

    with self.autocast:
      t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
      noise_scale = self.noise_level[t].unsqueeze(1)
      noise_scale_sqrt = noise_scale**0.5
      noise = torch.randn_like(audio)
      noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise
      
      predicted = self.diff_model(noisy_audio, t, audio_features)
      loss = self.loss_fn(noise, predicted.squeeze(1))

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.diff_model.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss

  def _write_summary(self, step, features, loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)
    # if not self.params.unconditional:
    #   writer.add_image('feature/spectrogram', torch.flip(features['spectrogram'][:1], [1]), step)
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer


def _train_impl(replica_id, diff_model, whisper_model, dataset, args, params):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(diff_model.parameters(), lr=params.learning_rate)

  learner = WhisperDiffWaveLearner(args.model_dir, diff_model, whisper_model, dataset, opt, params, fp16=args.fp16)
  learner.is_master = (replica_id == 0)
  learner.restore_from_checkpoint()
  learner.train(max_steps=args.max_steps)




def train(args, params):
  device = params.device
#   if args.data_dirs[0] == 'gtzan':
#     dataset = from_gtzan(params)
#   else:
  print(f'Loading the dataset from {params.data_dirs}')
  dataset = from_path(params.data_dirs, params)
  print(f'Loading the whisper and diffwave models...')
  whisper_model = load_model(params.audio_model).to(device)
  for p in whisper_model.parameters():
    p.requires_grad = params.audio_trainable

  diff_model = DiffWave(params).to(device)
  _train_impl(0, diff_model, whisper_model, dataset, args, params)


def train_distributed(replica_id, replica_count, port, args, params):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
#   if args.data_dirs[0] == 'gtzan':
#     dataset = from_gtzan(params, is_distributed=True)
#   else:
  dataset = from_path(params.data_dirs, params, is_distributed=True)
  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  model = DiffWave(params).to(device)
  model = DistributedDataParallel(model, device_ids=[replica_id])
  _train_impl(replica_id, model, dataset, args, params)



def _get_free_port():
  import socketserver
  with socketserver.TCPServer(('localhost', 0), None) as s:
    return s.server_address[1]


def main(args):
  params = Config_Wave()
  replica_count = torch.cuda.device_count()
  if replica_count > 1:
    if params.batch_size % replica_count != 0:
      raise ValueError(f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
    params.batch_size = params.batch_size // replica_count
    port = _get_free_port()
    spawn(train_distributed, args=(replica_count, port, args, params), nprocs=replica_count, join=True)
  else:
    train(args, params)


if __name__ == '__main__':
  parser = ArgumentParser(description='train (or resume training) a DiffWave model')
  parser.add_argument('model_dir',
      help='directory in which to store model checkpoints and training logs')
#   parser.add_argument('data_dirs', nargs='+',
#       help='space separated list of directories from which to read .wav files for training')
  parser.add_argument('--max_steps', default=None, type=int,
      help='maximum number of training steps')
  parser.add_argument('--fp16', action='store_true', default=False,
      help='use 16-bit floating point operations for training')
  main(parser.parse_args())