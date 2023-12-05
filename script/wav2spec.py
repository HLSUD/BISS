from argparse import ArgumentParser
import os
import torchaudio
from PIL import Image
from models.whisper.audio import  pad_or_trim, log_mel_spectrogram
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch

def wave2spec(audio_filename, time, sample_rate):
    audio_filename
    signal, sr = torchaudio.load(audio_filename)
    if sr != sample_rate:
        signal = torchaudio.functional.resample(signal, orig_freq=sr, new_freq=sample_rate)
    # signal = signal[:time*sample_rate]
    audio = pad_or_trim(signal[0])
    mel = log_mel_spectrogram(audio)
    mel_time = int(time*sample_rate / len(audio) * mel.shape[1])
    
    mel = mel[:,:mel_time]
    return mel

def main(args):
    mel = wave2spec(args.file_path, args.time, args.sr)
    img = Image.fromarray(mel.numpy())
    file_name = args.file_path.split('/')[-1].split('.')[0] + '.tiff'
    imag_path = os.path.join(args.save_dir,file_name)
    print(imag_path)
    img.save(imag_path)
    return

def evaluate_spec(args):
    spec1 = np.array(Image.open(args.spec1_path))
    spec2 = np.array(Image.open(args.spec2_path))
    cor = np.corrcoef([spec1.flatten(), spec2.flatten()])[1,0]
    mse = ((spec1 - spec2)**2).mean()
    ssim_noise = ssim(spec1, spec2)
    print(cor,mse,ssim_noise)

def evaluate_audio(args):
    eps = 1e-8
    target = torchaudio.load(args.spec1_path)[0]
    input = torchaudio.load(args.spec2_path)[0][:,:195840]

    input_mean = torch.mean(input, dim=-1, keepdim=True)
    target_mean = torch.mean(target, dim=-1, keepdim=True)
    input = input - input_mean
    target = target - target_mean

    res = input - target
    print((res ** 2).sum(-1))
    losses = 10 * torch.log10(
        (target ** 2).sum(-1) / ((res ** 2).sum(-1) + eps) + eps
    )
    print(losses.item())

# if __name__ == '__main__':
#     parser = ArgumentParser(description='audio to spectrogram')
#     parser.add_argument('--file_path', help='the audio file path',  required=True)
#     parser.add_argument('--save_dir', required=True)
#     parser.add_argument('--time', default=30, help='audio duration, unit - second', type=int)
#     parser.add_argument('--sr', default=16000, help='sampling rate', type=int)

#     main(parser.parse_args())

if __name__ == '__main__':
    parser = ArgumentParser(description='audio to spectrogram')
    parser.add_argument('--spec1_path', help='the audio file path',  required=True)
    parser.add_argument('--spec2_path', help='the audio file path',  required=True)

    evaluate_spec(parser.parse_args())
    evaluate_audio(parser.parse_args())


