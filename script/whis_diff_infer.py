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

from models.eeg_dataset import ConditionalDataset
import numpy as np
import os
import torch
import torchaudio

from models.whisper.__init__ import load_model
from models.speech_trainer import from_path
from config.wave_config import Config_Wave
from argparse import ArgumentParser

# from models.params import AttrDict, params as base_params
from models.NN.latent2speech import DiffWave, whis2diffpooling


models = {}

def predict(spectrogram=None, model_dir=None, params=None, device=torch.device('cuda'), fast_sampling=False):
  # Lazy load model.
  
  if not model_dir in models:
    if os.path.exists(f'{model_dir}/whis_2_diff.pt'):
    
      checkpoint = torch.load(f'{model_dir}/whis_2_diff.pt',map_location=torch.device('cpu'))
      print('Load diff model from path...')
    else:
      print(model_dir)
      checkpoint = torch.load(model_dir)
    model = DiffWave(params).to(device)
    model.load_state_dict(checkpoint['model'])
    if params.audio_model == 'base':
        in_channel = 512
    pooling = whis2diffpooling(in_channel, params.num_channel).to(device)
    pooling.load_state_dict(checkpoint['pool_layer'])
    model.eval()
    pooling.eval()
    models[model_dir] = model

  model = models[model_dir]
#   model.params.override(params)
  with torch.no_grad():
    # Change in notation from the DiffWave paper for fast sampling.
    # DiffWave paper -> Implementation below
    # --------------------------------------
    # alpha -> talpha
    # beta -> training_noise_schedule
    # gamma -> alpha
    # eta -> beta
    training_noise_schedule = np.array(params.noise_schedule)
    inference_noise_schedule = np.array(params.inference_noise_schedule) if fast_sampling else training_noise_schedule

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)


    if not params.unconditional:
      if len(spectrogram.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
        spectrogram = spectrogram.unsqueeze(0)
      spectrogram = spectrogram.to(device)
      spectrogram = spectrogram[:,:params.time *params.token_num_per_sec]
      spectrogram = pooling(spectrogram)
      print(spectrogram.shape)
      audio = torch.randn(spectrogram.shape[0], params.time * params.sample_rate, device=device)
      print(spectrogram.shape[-1],params.hop_samples)
    else:
      audio = torch.randn(1, params.audio_len, device=device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

    for n in range(len(alpha) - 1, -1, -1):
      c1 = 1 / alpha[n]**0.5
      c2 = beta[n] / (1 - alpha_cum[n])**0.5
      audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=audio.device), spectrogram).squeeze(1))
      if n > 0:
        noise = torch.randn_like(audio)
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        audio += sigma * noise
      audio = torch.clamp(audio, -1.0, 1.0)
  return audio, params.sample_rate


def transform(filename):
  audio, sr = torchaudio.load(filename)
  audio = torch.clamp(audio[0], -1.0, 1.0)

  if sr != 16000:
      audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
  mel_args = {
      'sample_rate': sr,
      'win_length': 256 * 4,
      'hop_length': 256,
      'n_fft': 1024,
      'f_min': 20.0,
      'f_max': sr / 2.0,
      'n_mels': 80,
      'power': 1.0,
      'normalized': True,
  }
  mel_spec_transform = torchaudio.transforms.MelSpectrogram(**mel_args)

  with torch.no_grad():
    spectrogram = mel_spec_transform(audio)
    print(spectrogram.shape)

    # spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    # spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    # np.save(f'{filename}.spec.npy', spectrogram.cpu().numpy())

# transform('data/mozila/cv-corpus-11.0-2022-09-21/zh-CN/clips/common_voice_zh-CN_33211332.mp3')

def main(args):
  
  params = Config_Wave()
  device = params.device
#   dataset = from_path(params.data_dirs, params)
  dataset = ConditionalDataset(params.data_dirs, params.dataset_type)
  whisper_model = load_model(params.audio_model).to(device)
  for p in whisper_model.parameters():
    p.requires_grad = params.audio_trainable
  
  k = 1
  audio = dataset[k]['audio']
  params.time = 5
  print(audio.shape)
  spectrogram = dataset[k]['spectrogram']
  params.time = dataset[k]['time']
  if len(spectrogram.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
    spectrogram = spectrogram.unsqueeze(0).to(device)
    ### preprocess
    # print('getting audio features...')
  audio_features = whisper_model.embed_audio(spectrogram)
  print(audio_features.shape)
  
  audio, sr = predict(audio_features, model_dir=args.model_dir, fast_sampling=args.fast, params=params,device=params.device)
  torchaudio.save(args.output, audio.cpu(), sample_rate=sr)


if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
#   parser.add_argument('--spectrogram_path', '-s',
#       help='path to a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('--output', '-o', default='output.wav',
      help='output file name')
  parser.add_argument('--fast', '-f', action='store_true',
      help='fast sampling procedure')
  main(parser.parse_args())