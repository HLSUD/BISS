from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from skimage import io
from PIL import Image
from torch.utils.data import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# class coch_eigen_set(Dataset):
#     """eeg coch eigen vector dataset."""

#     def __init__(self, eeg_file, coch_img_file, data_dir='./data/images/', eeg_merge_size=30, eeg_hop_size = 1, transform=None):
#         """
#         Args:
#             eeg_file (string): Paht to eeg data - list of np array
#             coch_eigen_np_file (string): Path to coch_eigen_vector data - numpy array
#             eeg_merge_size (int): merge eeg_merge_size eeg data vectors to a vector
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """

#         # print(coch_eigen_np_file)
#         self.data_dir = data_dir
#         self.coch_imgs = pd.read_csv(coch_img_file)
#         self.eeg_data = np.load(eeg_file)
#         self.eeg_merge_size = eeg_merge_size
#         self.hop_size = eeg_hop_size
#         # self.eeg_data =  self.eeg_data.reshape(-1,self.eeg_data.shape[-1]*eeg_merge_size)
#         # self.coch_eigen_data = np.load(coch_eigen_np_file)
#         # self.coch_eigen_data = self.generate_eigen_dataset(self.coch_eigen_data)
#         self.transform = transform

#     def __len__(self):
#         return len(self.coch_imgs)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        
#         # data
#         start_pos = self.coch_imgs.iloc[idx, 0]
#         end_pos = start_pos + self.eeg_merge_size
#         eeg = self.eeg_data[start_pos:end_pos].reshape(-1,self.eeg_data.shape[-1]*self.eeg_merge_size)
#         eeg = np.squeeze(eeg)
#         eeg = torch.tensor(eeg,dtype=torch.float32)

#         # target
#         img_name = os.path.join(self.data_dir, self.coch_imgs.iloc[idx, 1],
#                                 self.coch_imgs.iloc[idx, 2])
#         image = io.imread(img_name,as_gray=True)

#         # eigen
#         eigen_values, eigen_vectors = np.linalg.eig(image)
#         sum_v = np.multiply(eigen_values,eigen_vectors).sum(axis=1)
#         nor_factor =  np.linalg.norm(sum_v)
#         unit_v = sum_v / nor_factor
#         coch_eigen_vector = np.append(unit_v,np.exp(-nor_factor-1))
#         coch_eigen_vector = torch.tensor(coch_eigen_vector,dtype=torch.float32)

#         if self.transform:
#             eeg = self.transform(eeg)
#             coch_eigen_vector = self.transform(coch_eigen_vector)
        
#         sample = [eeg, coch_eigen_vector]
#         # sample = {'eeg_data': eeg, 'coch_eigen': coch_eigen_vector}


#         return sample

#     def generate_eigen_dataset(self, coch_images):
#         """
#         coch_images (numpy): coch images .npy
#         """
        
#         eigen_values, eigen_vectors = np.linalg.eig(coch_images)
#         y = np.empty((0,coch_images.shape[1]+1))
#         for i in range(len(eigen_values)):
#             sum_v = np.multiply(eigen_values[i],eigen_vectors[i]).sum(axis=1)
#             nor_factor =  np.linalg.norm(sum_v)
#             unit_v = sum_v / nor_factor
#             y = np.append(y,[np.append(unit_v,np.exp(-nor_factor-1))], axis=0)
#             # y = np.append(y,[sum_v], axis=0)
#         return y

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
        print(self.coch_imgs.iloc[idx, 2])
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

