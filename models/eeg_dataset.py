from __future__ import print_function, division
from models.whisper.audio import log_mel_spectrogram, pad_or_trim
import os
import torch
import pandas as pd
import numpy as np
from skimage import io
from PIL import Image
from torch.utils.data import Dataset
import random
from pathlib import Path
from scipy.fftpack import rfft, irfft
from scipy.interpolate import interp1d
import torchaudio
from scipy.signal import savgol_filter
from typing import Callable, Optional, Tuple, Union
# Ignore warnings
import warnings
from models.utils import smooth_signal
warnings.filterwarnings("ignore")


class coch_set(Dataset):
    """eeg coch dataset."""

    def __init__(self, eeg_file, coch_img_file, data_dir='./data/images/', eeg_merge_size=30, eeg_hop_size = 1, output_type = 0, transform=None):
        """
        Args:
            eeg_file (string): Paht to eeg data - list of np array
            coch_eigen_np_file (string): Path to coch_eigen_vector data - numpy array
            eeg_merge_size (int): merge eeg_merge_size eeg data vectors to a vector
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        # print(coch_eigen_np_file)
        self.data_dir = data_dir
        self.coch_imgs = pd.read_csv(coch_img_file)
        self.eeg_data = np.load(eeg_file)
        self.eeg_merge_size = eeg_merge_size
        self.hop_size = eeg_hop_size
        self.output_type = output_type
        # self.eeg_data =  self.eeg_data.reshape(-1,self.eeg_data.shape[-1]*eeg_merge_size)
        # self.coch_eigen_data = np.load(coch_eigen_np_file)
        # self.coch_eigen_data = self.generate_eigen_dataset(self.coch_eigen_data)
        self.transform = transform

    def __len__(self):
        return len(self.coch_imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # data
        start_pos = self.coch_imgs.iloc[idx, 0]
        end_pos = start_pos + self.eeg_merge_size
        eeg = self.eeg_data[start_pos:end_pos].reshape(-1,self.eeg_data.shape[-1]*self.eeg_merge_size)
        eeg = np.squeeze(eeg)
        eeg = torch.tensor(eeg,dtype=torch.float32)
        # print(self.coch_imgs.iloc[idx, 2])
        # target
        img_name = os.path.join(self.data_dir, self.coch_imgs.iloc[idx, 1],
                                self.coch_imgs.iloc[idx, 2])
        image = np.array(Image.open(img_name))
        if len(image.shape) != 2:
            print("Image size is wrong...")
        # image = io.imread(img_name,as_gray=True)

        target = None
        if self.output_type == 0:
            target = image.reshape(-1,image.shape[0]*image.shape[1])
            target = np.squeeze(target)
            target = torch.tensor(target, dtype=torch.float32)
        # eigen
        elif self.output_type == 1:
            eigen_values, eigen_vectors = np.linalg.eig(image)
            sum_v = np.multiply(eigen_values,eigen_vectors).sum(axis=1)
            nor_factor =  np.linalg.norm(sum_v)
            unit_v = sum_v / nor_factor
            coch_eigen_vector = np.append(unit_v,np.exp(-nor_factor-1))
            coch_eigen_vector = torch.tensor(coch_eigen_vector,dtype=torch.float32)
            target = coch_eigen_vector

        if self.transform:
            eeg = self.transform(eeg)
            target = self.transform(target)
        
        sample = [eeg, target]
        # sample = {'eeg_data': eeg, 'coch_eigen': coch_eigen_vector}


        return sample

    def get_image_name(self, idx):
        print(self.coch_imgs.iloc[idx, 2])
        return self.coch_imgs.iloc[idx, 2]

    def generate_eigen_dataset(self, coch_images):
        """
        coch_images (numpy): coch images .npy
        """
        
        eigen_values, eigen_vectors = np.linalg.eig(coch_images)
        y = np.empty((0,coch_images.shape[1]+1))
        for i in range(len(eigen_values)):
            sum_v = np.multiply(eigen_values[i],eigen_vectors[i]).sum(axis=1)
            nor_factor =  np.linalg.norm(sum_v)
            unit_v = sum_v / nor_factor
            y = np.append(y,[np.append(unit_v,np.exp(-nor_factor-1))], axis=0)
            # y = np.append(y,[sum_v], axis=0)
        return y


def eeg_interp_repeat(data, data_chan, data_len):
    if data.shape[-1] > data_len: 
        idx = np.random.randint(0, int(data.shape[-1] - data_len)+1)

        data = data[:, idx: idx+data_len]
    else: # interp1d
        x = np.linspace(0, 1, data.shape[-1])
        x2 = np.linspace(0, 1, data_len)
        f = interp1d(x, data)
        data = f(x2)
    ret = np.zeros((data_chan, data_len))
    if (data_chan > data.shape[-2]): # replicate
        for i in range((data_chan//data.shape[-2])):

            ret[i * data.shape[-2]: (i+1) * data.shape[-2], :] = data
        if data_chan % data.shape[-2] != 0:

            ret[ -(data_chan%data.shape[-2]):, :] = data[: (data_chan%data.shape[-2]), :]
    elif(data_chan < data.shape[-2]):
        idx2 = np.random.randint(0, int(data.shape[-2] - data_chan)+1)
        ret = data[idx2: idx2+data_chan, :]
    # print(ret.shape)
    elif(data_chan == data.shape[-2]):
        ret = data
    ret = ret/10 # reduce an order
    return ret

abnormal_data_idx = [2,10,13,24,25,34,39,52,54,60,63,67,68,70,71,75,76,80,86,87,28,79,84]

def remove_bad_data_paths(indices, root_path, input_paths):
    for i in indices:
        bad_paths = [root_path + 'subj' + str(i) +'_mixed_f.npy',
                     root_path + 'subj' + str(i) +'_mixed_m.npy',
                     root_path + 'subj' + str(i) +'_single_f.npy',
                     root_path + 'subj' + str(i) +'_single_m.npy']
        for bp in bad_paths:
            if bp in input_paths:
                input_paths.remove(bp)
    return input_paths

# ------------------------------------------------------------------------------
def split_dataset(eeg_path, hop_size = 50, win_size = 500, data_len=512, data_chan=128, time_pts=60000, freq = 100):
    """
    split data into train, val, test datasets 8:1:1
    """
    train_file = 'data/train_idx_name.csv'
    val_file = 'data/val_idx_name.csv'
    test_file = 'data/test_idx_name.csv'
    if Path(train_file).is_file() and Path(val_file).is_file() and Path(test_file).is_file():
        print('Train, val, test datasets have been created...')
        return
    

    eeg_paths = [str(f) for f in sorted(Path(eeg_path).rglob('*')) if is_npy_ext(f) and os.path.isfile(f)]
    eeg_paths = remove_bad_data_paths(abnormal_data_idx,eeg_path,eeg_paths)
    assert len(eeg_paths) != 0, 'No data found'

    num_pitchs = (time_pts - win_size) // hop_size + 1
    num_items = num_pitchs * len(eeg_paths)

    num_train = int(num_items * 0.8)
    num_val = int(num_items * 0.1)
    num_test = int(num_items * 0.1)
    idx = np.arange(num_items)
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    val_idx = idx[num_train:num_val+num_train]
    test_idx = idx[num_val+num_train:]

    df_train = pd.DataFrame(columns=['eeg_name','inner_idx'])
    df_val = pd.DataFrame(columns=['eeg_name','inner_idx'])
    df_test = pd.DataFrame(columns=['eeg_name','inner_idx'])

    for idx in train_idx:
        eeg_idx = idx // num_pitchs
        inner_idx = idx - eeg_idx * num_pitchs
        path = eeg_paths[eeg_idx]
        df_train.loc[len(df_train.index)] = [path, inner_idx]
    for idx in val_idx:
        eeg_idx = idx // num_pitchs
        inner_idx = idx - eeg_idx * num_pitchs
        path = eeg_paths[eeg_idx]
        df_val.loc[len(df_val.index)] = [path, inner_idx]
    for idx in test_idx:
        eeg_idx = idx // num_pitchs
        inner_idx = idx - eeg_idx * num_pitchs
        path = eeg_paths[eeg_idx]
        df_test.loc[len(df_test.index)] = [path, inner_idx]

    df_train.to_csv(train_file, index=False)
    df_val.to_csv(val_file, index=False)
    df_test.to_csv(test_file, index=False)
    
    return

#### smooth signals
def smooth_signal(signal, weight_threshold=100, keep_ratio=0.05, savgol=True, win=7, poly=1):
        # to frequent domain
        w = rfft(signal)
        spectrum = w**2
        # remove f with small value
        cutoff_idx = spectrum < (spectrum.max()/weight_threshold)
        if (cutoff_idx.sum() / len(cutoff_idx)) > (1-keep_ratio):
            idx = int((1-keep_ratio) * signal.shape[-1])
            cutoff_idx = spectrum < np.sort(spectrum)[idx]
        w2 = w.copy()
        w2[cutoff_idx] = 0

        s_smooth = irfft(w2)
        # Savitzky-Golay filter
        if savgol:
            s_smooth = savgol_filter(s_smooth, win, poly, mode='nearest')
        return s_smooth

#### Contrastive
class NLEDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_path = './data/eeg_data/', audio_path = './data/audio_data/', csv_file = './data/train_idx_path.csv', hop_size = 50, smooth = True, win_size = 500, data_len=512, data_chan=128, time_pts=60000, freq = 100, sr = 16000, transforms = None):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        # self.eeg_paths = [str(f) for f in sorted(Path(eeg_path).rglob('*')) if is_npy_ext(f) and os.path.isfile(f)]
        # self.audio_paths = [str(f) for f in sorted(Path(audio_path).rglob('*')) if is_npy_ext(f) and os.path.isfile(f)]
        ## remove bad data path from input_path
        # self.eeg_paths = remove_bad_data_paths(abnormal_data_idx,eeg_path,self.eeg_paths)
        # assert len(self.eeg_paths) != 0, 'No data found'

        self.audio_path = audio_path
        self.eeg_path = eeg_path
        self.data_info = pd.read_csv(csv_file)
        
        ### length and channels
        self.data_len  = data_len
        self.data_chan = data_chan

        self.win_size = win_size
        self.hop_size = hop_size
        self.freq = freq
        self.sr = sr # audio sample rate
        self.smooth = smooth

        self.num_pitchs = (time_pts - self.win_size) // self.hop_size + 1

        print(f"Window size: {self.win_size}")
        print(f"Hop size: {self.hop_size}")
        print(f"Num of pitches: {self.num_pitchs}")
        print(f"Smooth: {self.smooth}")
        self.transforms = transforms

    def __getitem__(self, index):
        # eeg_idx = index // self.num_pitchs
        # inner_idx = index - eeg_idx * self.num_pitchs
        # data_path = self.input_paths[eeg_idx]
        eeg_path = self.data_info.iloc[index, 0]
        inner_idx = self.data_info.iloc[index, 1]
        eeg_data = np.load(eeg_path)
        
        # load audio
        audio_array = None
        a_path = ''
        if 'single_m' in eeg_path:
            a_path = self.audio_path + 'male_s4.wav'
        elif 'single_f' in eeg_path:
            a_path = self.audio_path + 'female_s1.wav'
        elif 'mix' in eeg_path:
            a_path = self.audio_path + 'mix_s1s4.wav'
        else:
            print(f"EEG name error: {eeg_path}")
        audio, sr = torchaudio.load(a_path)
        if sr != self.sr:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sr)
        start_loc = int(inner_idx * self.hop_size / self.freq * self.sr)
        end_loc = int(start_loc + self.win_size / self.freq * self.sr)
        audio_array = audio[0,start_loc:end_loc] #  to mono audio
        audio_array = pad_or_trim(audio_array)
        mel = log_mel_spectrogram(audio_array) ## check 

        start_loc = inner_idx * self.hop_size
        data = eeg_data[:,start_loc:(start_loc+self.win_size)]
        
        ret = eeg_interp_repeat(data, self.data_chan, self.data_len)
        for i in range(data.shape[-2]):
            ret[i] = smooth_signal(ret[i])
        
        ret = torch.from_numpy(ret).float()
        return {'eeg': ret, 'audio': mel}


    def __len__(self):
        return len(self.data_info)

# -----------------------------------------------------------
# MAE

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_npy_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'{ext}' == 'npy'# type: ignore

class eeg_pretrain_dataset(Dataset):
    def __init__(self, path='./data/eeg_data/',hop_size = 50, smooth=False, win_size = 500, data_len=512, data_chan=128, time_pts=60000):
        super(eeg_pretrain_dataset, self).__init__()
        data = []
        images = []
        ## get input path/ data arrays
        self.input_paths = [str(f) for f in sorted(Path(path).rglob('*')) if is_npy_ext(f) and os.path.isfile(f)]
        # print(self.input_paths)
        ## remove bad data path from input_path
        self.input_paths = remove_bad_data_paths(abnormal_data_idx,path,self.input_paths)
        assert len(self.input_paths) != 0, 'No data found'
        ### length and channels
        self.data_len  = data_len
        self.data_chan = data_chan

        self.win_size = win_size
        self.hop_size = hop_size
        self.num_pitchs = (time_pts - self.win_size) // self.hop_size + 1
        # print(len(self.input_paths))
        self.smooth = smooth
        print(f"Num of input: {len(self.input_paths)}")
        print(f"Window size: {self.win_size}")
        print(f"Hop size: {self.hop_size}")
        print(f"Num of pitches: {self.num_pitchs}")
        print(f"Smooth: {self.smooth}")

    def __len__(self):
        return len(self.input_paths) * self.num_pitchs
    
    def smooth_signal(self, signal, weight_threshold=100, keep_ratio=0.05, savgol=True, win=7, poly=1):
        # to frequent domain
        w = rfft(signal)
        spectrum = w**2
        # remove f with small value
        cutoff_idx = spectrum < (spectrum.max()/weight_threshold)
        if (cutoff_idx.sum() / len(cutoff_idx)) > (1-keep_ratio):
            idx = int((1-keep_ratio) * signal.shape[-1])
            cutoff_idx = spectrum < np.sort(spectrum)[idx]
        w2 = w.copy()
        w2[cutoff_idx] = 0
        
        s_smooth = irfft(w2)
        # Savitzky-Golay filter
        if savgol:
            s_smooth = savgol_filter(s_smooth, win, poly, mode='nearest')
        return s_smooth
    
    def __getitem__(self, index):
        eeg_idx = index // self.num_pitchs
        inner_idx = index - eeg_idx * self.num_pitchs
        data_path = self.input_paths[eeg_idx]

        # print(index,eeg_idx)
        eeg_data = np.load(data_path)
        # print(inner_idx)
        # print(eeg_data.shape)
        start_loc = inner_idx * self.hop_size
        data = eeg_data[:,start_loc:(start_loc+self.win_size)]
        # print(data.shape)
        ret = eeg_interp_repeat(data, self.data_chan, self.data_len)
        for i in range(data.shape[-2]):
            ret[i] = smooth_signal(ret[i])
        # torch.tensor()
        if self.smooth:
            for idx in range(len(ret)):
                ret[idx] = self.smooth_signal(ret[idx])
        ret = torch.from_numpy(ret).float()
        return {'eeg': ret } #,

#### speech diffwave dataset
### audio hz - 16000
### crop_mel_frames ??? check
## 1. read tsv
## 2. load clip 
## 3. resample to 16000
class ConditionalDataset(Dataset):
  def __init__(self, path, set_type = 'train', sr = 16000):
    super().__init__()
    tsv_filename = path + set_type + '.tsv'
    
    self.sr = sr
    self.path = path
    self.data=pd.read_csv(tsv_filename,sep='\t')
    self.filenames = self.data['path']
    

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.path + 'clips/' + self.filenames[idx]
    # spec_filename = f'{audio_filename}.spec.npy'
    signal, sr = torchaudio.load(audio_filename)
    if sr != self.sr:
        signal = torchaudio.functional.resample(signal, orig_freq=sr, new_freq=self.sr)
    # spectrogram = np.load(spec_filename)
    # mel = log_mel_spectrogram(signal[0])
    return {
        'audio': signal[0],
        'spectrogram': None
    }

class Whisper_Collator:
    def __init__(self, params):
        self.params = params

    def collate(self, minibatch):
        
        for record in minibatch:
        
            # Filter out records that aren't long enough.
            if len(record['audio']) < self.params.audio_len:
                # del record['spectrogram']
                del record['audio']
                continue

            start = random.randint(0, record['audio'].shape[-1] - self.params.audio_len)
            end = start + self.params.audio_len
            record['audio'] = record['audio'][start:end]
            record['audio'] = np.pad(record['audio'], (0, (end - start) - len(record['audio'])), mode='constant')
      
        audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
        # audio = pad_or_trim(audio)
        mel = log_mel_spectrogram(pad_or_trim(audio))
    
        return {
            'audio': torch.from_numpy(audio),
            'spectrogram': mel
        }

class Collator:
  def __init__(self, params):
    self.params = params

  def collate(self, minibatch):
    samples_per_frame = self.params.hop_samples
    for record in minibatch:
      if self.params.unconditional:
          # Filter out records that aren't long enough.
          if len(record['audio']) < self.params.audio_len:
            # del record['spectrogram']
            del record['audio']
            continue

          start = random.randint(0, record['audio'].shape[-1] - self.params.audio_len)
          end = start + self.params.audio_len
          record['audio'] = record['audio'][start:end]
          record['audio'] = np.pad(record['audio'], (0, (end - start) - len(record['audio'])), mode='constant')
      else:
          # Filter out records that aren't long enough.
          record['spectrogram'] = record['spectrogram'].T
          if len(record['spectrogram']) < self.params.crop_mel_frames:
            del record['spectrogram']
            del record['audio']
            continue

          start = random.randint(0, record['spectrogram'].shape[0] - self.params.crop_mel_frames)
          end = start + self.params.crop_mel_frames
          record['spectrogram'] = record['spectrogram'][start:end].T

          start *= samples_per_frame
          end *= samples_per_frame
          
          record['audio'] = record['audio'][start:end]
          record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant')

    audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
    if self.params.unconditional:
        return {
            'audio': torch.from_numpy(audio),
            'spectrogram': None,
        }
    spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
    return {
        'audio': torch.from_numpy(audio),
        'spectrogram': torch.from_numpy(spectrogram),
    }

if __name__ == '__main__':
    eeg_pre = eeg_pretrain_dataset()
