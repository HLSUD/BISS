import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import torch
import torchaudio
# import IPython

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.image import imsave
from PIL import Image
import librosa
import librosa.display
# import IPython.display as ipd
import numpy as np
# from pycochleagram import cochleagram as cgram
# from pycochleagram import erbfilter as erb
# from pycochleagram import utils
from scipy.signal import welch, decimate, resample
import argparse
import os
from random import choice
from time import sleep, time

# np_spec = spectrogram.numpy()
# print(np_spec.shape)
# plot_spectrogram(np_spec)


def mask_spec(height, width, num_seg, ratio, spec, method=0):
    #     if method == 1:
    #         # based on row
    #     if method == 2:
    #         # based on column
    #     if method == 0:
    #         # default
    spec = np.copy(spec)
    num_mask = int(ratio * num_seg * num_seg)
    mask_ind = np.random.choice(range(num_seg*num_seg), num_mask, replace=False)
    #     col_ind = np.random.choice(range(num_seg), num_mask, replace=False)
    r_mask_size = height / num_seg
    c_mask_size = width / num_seg
    row_start = np.empty((num_mask,),dtype=np.int32)
    row_end = np.empty((num_mask,),dtype=np.int32)
    col_start = np.empty((num_mask,),dtype=np.int32)
    col_end = np.empty((num_mask,),dtype=np.int32)
    #     row_start = row_ind * r_mask_size
    row_start = (mask_ind / num_seg * r_mask_size).astype(np.int32)
    row_end = (row_start + r_mask_size).astype(np.int32)
    col_start = (mask_ind%num_seg * c_mask_size).astype(np.int32)
    col_end = (col_start + c_mask_size).astype(np.int32)

    for i in range(num_mask):
        spec[row_start[i]:row_end[i],col_start[i]:col_end[i]] = 0
    
    return spec

def plot_waveform(waveform, sr, title="Waveform"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    figure, axes = plt.subplots(num_channels, 1)
    axes.plot(time_axis, waveform[0], linewidth=1)
    axes.grid(True)
    figure.suptitle(title)
    plt.show(block=False)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    #     im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    im = axs.imshow(specgram, origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)
    plt.close(fig)

def wav_2_spec(filename, audio_name, mel_args, win_len, hop_size, audio_dur = 600, sample_rate=22050):

    audio, sr = torchaudio.load(filename)
    audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=sample_rate)
    audio = torch.clamp(audio[0], -1.0, 1.0)
    audio = audio[:audio_dur*sample_rate]
    num_spec = int((audio_dur*1000 - win_len) // hop_size + 1)
    print(num_spec)
    seg_sample_size = sample_rate * win_len // 1000
    start_loc = 0
    # spectrogram = None
    for i in range(start_loc, num_spec):
        s_audio = audio[i*hop_size*sample_rate//1000:i*hop_size*sample_rate//1000+seg_sample_size]
        mel_spec_transform = torchaudio.transforms.MelSpectrogram(**mel_args)
        spectrogram = None
        # n_stft = n_stft = int((1024//2) + 1)
        spectrogram = mel_spec_transform(s_audio)
        imag_name = audio_name + '_' + str(i) + '.tiff'
        imag_path = '/mnt/nvme-ssd/hliuco/Documents/data/BISS/images/spectrogram/' + 'male_s4' +'/'+ imag_name
        # imsave(imag_path, spectrogram)
        img = Image.fromarray(spectrogram.numpy())
        img.save(imag_path)
        print(imag_name + ' saved ...')
    return

def spec_2_wav(spectrogram, inv_mel_args, g_mel_args):
    invers_transform = torchaudio.transforms.InverseMelScale(**inv_mel_args)
    grifflim_transform = torchaudio.transforms.GriffinLim(**g_mel_args)
    inverse_waveform = invers_transform(spectrogram)
    waveform = grifflim_transform(inverse_waveform)
    return waveform

def w2s_demo():
    filename = '/home/hliuco/Documents/BISS/data/audio/male_s4_600.wav'
    audio_name = 'male_s4'
    spec_win_len = 5000
    hop_size = 100
    audio_dur = 600
    mel_args = {
        'sample_rate': 22050,
        'win_length': 256 * 4,
        'hop_length': 256,
        'n_fft': 1024,
        'f_min': 20.0,
        'f_max': 22050 / 2.0,
        'n_mels': 80,
        'power': 1.0,
        'normalized': True,
    }
    wav_2_spec(filename, audio_name, mel_args, spec_win_len, hop_size, audio_dur)
    # mel_spec = 20 * torch.log10(torch.clamp(mel_spec, min=1e-5)) - 20
    # mel_spec = torch.clamp((mel_spec + 100) / 100, 0.0, 1.0)
    return

def s2w_demo(mel_spec):
    # n_stft = n_stft = int((1024//2) + 1)

    inv_mel_args = {
        'sample_rate': 22050,
        'n_stft': 513,
    }
    g_mel_args = {
        'n_fft': 1024,
    }
    wave = spec_2_wav(mel_spec, inv_mel_args, g_mel_args)
    return wave

def wave_2_coch(filename, coch_len, hop_size, imag_size, audio_name, fold_name, duration=600):
    audio, sr = librosa.load(filename,duration=duration)
    audio_dur = librosa.get_duration(y=audio,sr=sr)
    num_cochs = int((audio_dur*1000 - coch_len) // hop_size + 1)
    seg_sample_size = sr * coch_len // 1000
#     imag_arr = np.empty((0,imag_size,imag_size))
    print(audio_dur)
    start_loc = 0
    for i in range(start_loc, num_cochs):
        s_audio = audio[i*hop_size:i*hop_size+seg_sample_size]
#         print(len(s_audio))
        ### coch arguments
        sample_factor=4
        downsample=1000 # dowsample to 1000 HZ
        nonlinearity='power' # db, power
        n = 50  
        # default filter for low_lim=50 hi_lim=20000
        low_lim = 30  # this is the default for cochleagram.human_cochleagram ### modified
        hi_lim = 7860  # this is the default for cochleagram.human_cochleagram !!!!
        # # 3/10 power compression
        coch_pow = cgram.human_cochleagram(s_audio, sr, n, sample_factor=sample_factor, \
            low_lim=low_lim, hi_lim=hi_lim, downsample=downsample, nonlinearity=nonlinearity, strict=False)
        tmp = resample(coch_pow,imag_size)
        tmp = resample(tmp.T,imag_size).T
#         imag_arr = np.append(imag_arr, np.array([tmp]), axis=0)
        imag_name = audio_name + '_' + str(i) + '.png'
        imag_path = '../data/images/' + fold_name + '/' + imag_name
        imsave(imag_path, tmp)
        print(imag_name + ' saved ...')
#     np.save('../data/'+fold_name+'_imags.npy',imag_arr)

def w2c_demo():
    filename = '../data/audio/male_s4_600.wav'
    wave_2_coch(filename, 300, 10, 256, "male_s4", 'male_s4_hop10')


def split_dataset():
    ### seperate to train and test
    female_imags = np.load("../data/female_s1_imags.npy")
    male_images = np.load("../data/male_s4_imags.npy")

    num_train = int(len(female_imags) * 0.7)
    idx = np.arange(len(female_imags))
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    train_female_imags = female_imags[train_idx]
    train_male_imags = male_images[train_idx]
    train_imags = np.concatenate((train_female_imags, train_male_imags), axis=0)

    test_female_imags = female_imags[test_idx]
    test_male_imags = male_images[test_idx]
    test_imags = np.concatenate((test_female_imags, test_male_imags), axis=0)
    np.save("../data/train/coch_imags.npy",train_imags)
    np.save("../data/train/indices.npy",train_idx)
    np.save("../data/test/coch_imags.npy",test_imags)
    np.save("../data/test/indices.npy",test_idx)

    female_eeg_data = np.load("../data/subj1_single_f.npy").T[:60000]
    male_eeg_data = np.load("../data/subj1_single_m.npy").T[:60000]
    print(female_eeg_data.shape,male_eeg_data.shape)
    num_times = 30
    num_channels = 64
    train_female_eeg_data = female_eeg_data.reshape(-1,num_channels*num_times)[train_idx].reshape(-1,num_channels)
    train_male_eeg_data = male_eeg_data.reshape(-1,num_channels*num_times)[train_idx].reshape(-1,num_channels)
    train_eeg_data = np.concatenate((train_female_eeg_data, train_male_eeg_data), axis=0)

    test_female_eeg_data = female_eeg_data.reshape(-1,num_channels*num_times)[test_idx].reshape(-1,num_channels)
    test_male_eeg_data = male_eeg_data.reshape(-1,num_channels*num_times)[test_idx].reshape(-1,num_channels)
    test_eeg_data = np.concatenate((test_female_eeg_data, test_male_eeg_data), axis=0)

    np.save("../data/train/eeg_data.npy",train_eeg_data)
    np.save("../data/test/eeg_data.npy",test_eeg_data)

    print(test_eeg_data.shape)
    print(train_eeg_data.shape)

    ####
    female_eeg_data = np.load("../data/subj1_single_f.npy").T[:60000]
    male_eeg_data = np.load("../data/subj1_single_m.npy").T[:60000]
    eeg_data = np.concatenate((female_eeg_data, male_eeg_data), axis=0)
    np.save("../data/eeg_data.npy",eeg_data)
    print(male_eeg_data.shape,female_eeg_data.shape,eeg_data.shape)

    coch_num = 5951
    num_train = int(coch_num * 0.8)
    num_val = int(coch_num * 0.1)
    num_test = int(coch_num * 0.1)
    idx = np.arange(coch_num)
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    val_idx = idx[num_train:num_val+num_train]
    test_idx = idx[num_val+num_train:]

    df_train = pd.DataFrame(columns=['eeg_idx','spec_dir','spec_name'])
    df_val = pd.DataFrame(columns=['eeg_idx','spec_dir','spec_name'])
    df_test = pd.DataFrame(columns=['eeg_idx','spec_dir','spec_name'])

    coch_dir_2 = 'male_s4'
    coch_dir_1 = 'female_s1'
    for i,idx in enumerate(train_idx):
        coch_name_1 = 'female_s1_' + str(idx) + '.png'
        coch_name_2 = 'male_s4_' + str(idx) + '.png'
        df_train.loc[len(df_train.index)] = [idx * 10, coch_dir_1, coch_name_1]     
        df_train.loc[len(df_train.index)] = [idx * 10 + 60000, coch_dir_2, coch_name_2] 
    for i,idx in enumerate(val_idx):
        coch_name_1 = 'female_s1_' + str(idx) + '.png'
        coch_name_2 = 'male_s4_' + str(idx) + '.png'
        df_val.loc[len(df_val.index)] = [idx*10, coch_dir_1, coch_name_1]     
        df_val.loc[len(df_val.index)] = [idx*10+60000, coch_dir_2, coch_name_2] 
    for i,idx in enumerate(test_idx):
        coch_name_1 = 'female_s1_' + str(idx) + '.png'
        coch_name_2 = 'male_s4_' + str(idx) + '.png'
        df_test.loc[len(df_test.index)] = [idx*10, coch_dir_1, coch_name_1]     
        df_test.loc[len(df_test.index)] = [idx*10+60000, coch_dir_2, coch_name_2]
        
    df_train.to_csv('data/train_spec_idx.csv', index=False)
    df_val.to_csv('../data/val_spec_idx.csv', index=False)
    df_test.to_csv('../data/test_spec_idx.csv', index=False)

def split_spec_dataset(spec_num, hop_size, win_size=50):
    # spec_num = 5951
    num_zero_look = 551
    remain = spec_num - num_zero_look
    num_train = int(remain * 0.8)
    num_val = int(remain * 0.1)
    num_test = int(remain * 0.1)
    idx = np.arange(remain)
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    val_idx = idx[num_train:num_val+num_train]
    test_idx = idx[num_val+num_train:]

    df_train = pd.DataFrame(columns=['eeg_idx','spec_dir','spec_name'])
    df_val = pd.DataFrame(columns=['eeg_idx','spec_dir','spec_name'])
    df_test = pd.DataFrame(columns=['eeg_idx','spec_dir','spec_name'])
    df_zero_look = pd.DataFrame(columns=['eeg_idx','spec_dir','spec_name'])

    coch_dir_2 = 'male_s4'
    coch_dir_1 = 'female_s1'
    for i,idx in enumerate(train_idx):
        coch_name_1 = 'female_s1_' + str(idx) + '.tiff'
        coch_name_2 = 'male_s4_' + str(idx) + '.tiff'
        df_train.loc[len(df_train.index)] = [idx * hop_size, coch_dir_1, coch_name_1]     
        df_train.loc[len(df_train.index)] = [idx * hop_size + 60000, coch_dir_2, coch_name_2] 
    for i,idx in enumerate(val_idx):
        coch_name_1 = 'female_s1_' + str(idx) + '.tiff'
        coch_name_2 = 'male_s4_' + str(idx) + '.tiff'
        df_val.loc[len(df_val.index)] = [idx*hop_size, coch_dir_1, coch_name_1]     
        df_val.loc[len(df_val.index)] = [idx*hop_size+60000, coch_dir_2, coch_name_2] 
    for i,idx in enumerate(test_idx):
        coch_name_1 = 'female_s1_' + str(idx) + '.tiff'
        coch_name_2 = 'male_s4_' + str(idx) + '.tiff'
        df_test.loc[len(df_test.index)] = [idx*hop_size, coch_dir_1, coch_name_1]     
        df_test.loc[len(df_test.index)] = [idx*hop_size + 60000, coch_dir_2, coch_name_2]
    start_idx = remain + win_size
    for idx in range(start_idx, spec_num):
        coch_name_1 = 'female_s1_' + str(idx) + '.tiff'
        coch_name_2 = 'male_s4_' + str(idx) + '.tiff'
        df_zero_look.loc[len(df_zero_look.index)] = [idx*hop_size, coch_dir_1, coch_name_1]     
        df_zero_look.loc[len(df_zero_look.index)] = [idx*hop_size + 60000, coch_dir_2, coch_name_2]

    df_train.to_csv('data/train_spec_idx_tiff.csv', index=False)
    df_val.to_csv('data/val_spec_idx_tiff.csv', index=False)
    df_test.to_csv('data/test_spec_idx_tiff.csv', index=False)
    df_zero_look.to_csv('data/zerolook_spec_idx_tiff.csv', index=False)

def merge_fm(subj_id):
    female_eeg_data = np.load("data/eeg_data/subj"+str(subj_id)+"_single_f.npy").T[:60000]
    male_eeg_data = np.load("data/eeg_data/subj"+str(subj_id)+"_single_m.npy").T[:60000]
    eeg_data = np.concatenate((female_eeg_data, male_eeg_data), axis=0)
    np.save("data/eeg_data/subj"+str(subj_id)+"_eeg_data.npy",eeg_data)

if __name__ == '__main__':
    w2s_demo()
    # split_spec_dataset(5951,10)
    # merge_fm(2)
    # merge_fm(4)