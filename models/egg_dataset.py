from __future__ import print_function, division
import os
import torch
# import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class coch_eigen_set(Dataset):
    """eeg coch eigen vector dataset."""

    def __init__(self, eeg_file, coch_eigen_np_file, eeg_merge_size, transform=None):
        """
        Args:
            eeg_file (string): Paht to eeg data - list of np array
            coch_eigen_np_file (string): Path to coch_eigen_vector data - numpy array
            eeg_merge_size (int): merge eeg_merge_size eeg data vectors to a vector
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        # print(coch_eigen_np_file)
        self.eeg_data = np.load(eeg_file)
        self.eeg_data =  self.eeg_data.reshape(-1,self.eeg_data.shape[-1]*eeg_merge_size)
        self.coch_eigen_data = np.load(coch_eigen_np_file)
        self.coch_eigen_data = self.generate_eigen_dataset(self.coch_eigen_data)
        self.transform = transform

    def __len__(self):
        return len(self.coch_eigen_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        eeg = torch.tensor(self.eeg_data[idx],dtype=torch.float32)
        coch_eigen_vector = torch.tensor(self.coch_eigen_data[idx],dtype=torch.float32)

        if self.transform:
            eeg = self.transform(eeg)
            coch_eigen_vector = self.transform(coch_eigen_vector)
        
        sample = [eeg, coch_eigen_vector]
        # sample = {'eeg_data': eeg, 'coch_eigen': coch_eigen_vector}


        return sample

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
        return y

