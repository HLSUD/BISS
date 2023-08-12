from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from skimage import io
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from scipy.interpolate import interp1d
from typing import Callable, Optional, Tuple, Union
# Ignore warnings
import warnings
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

# ------------------------------------------------------------------------------
#### Contrastive
class NLEDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_file, captions, tokenizer, eeg_merge_size=30, eeg_hop_size = 1, output_type = 0, transforms = None):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.eeg_data = np.load(eeg_file)
        self.eeg_merge_size = eeg_merge_size
        self.hop_size = eeg_hop_size
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)

# -----------------------------------------------------------
# MAE
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

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_npy_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'{ext}' == 'npy'# type: ignore

class eeg_pretrain_dataset(Dataset):
    def __init__(self, path='./data/eeg_data/',hop_size = 50, win_size = 500, data_len=512, data_chan=128, time_pts=60000):
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
        print(len(self.input_paths))
        print(self.num_pitchs)

    def __len__(self):
        return len(self.input_paths)*self.num_pitchs
    
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
        if data.shape[-1] > self.data_len: 
            idx = np.random.randint(0, int(data.shape[-1] - self.data_len)+1)

            data = data[:, idx: idx+self.data_len]
        else: # interp1d
            x = np.linspace(0, 1, data.shape[-1])
            x2 = np.linspace(0, 1, self.data_len)
            f = interp1d(x, data)
            data = f(x2)
        ret = np.zeros((self.data_chan, self.data_len))
        if (self.data_chan > data.shape[-2]): # replicate
            for i in range((self.data_chan//data.shape[-2])):

                ret[i * data.shape[-2]: (i+1) * data.shape[-2], :] = data
            if self.data_chan % data.shape[-2] != 0:

                ret[ -(self.data_chan%data.shape[-2]):, :] = data[: (self.data_chan%data.shape[-2]), :]
        elif(self.data_chan < data.shape[-2]):
            idx2 = np.random.randint(0, int(data.shape[-2] - self.data_chan)+1)
            ret = data[idx2: idx2+self.data_chan, :]
        # print(ret.shape)
        elif(self.data_chan == data.shape[-2]):
            ret = data
        ret = ret/10 # reduce an order
        # torch.tensor()
        ret = torch.from_numpy(ret).float()
        return {'eeg': ret } #,

if __name__ == '__main__':
    eeg_pre = eeg_pretrain_dataset()