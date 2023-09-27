import numpy as np
import torch

class Config_Wave:
    # configs for fmri_pretrain.py
    def __init__(self):
        # Training params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_type = 'train'
        self.data_dirs = 'data/mozila/cv-corpus-11.0-2022-09-21/zh-CN/'
        self.batch_size=16
        self.learning_rate=2e-4
        self.max_grad_norm=None

        # Data params
        # self.sample_rate=22050, ### to 16,000
        self.sample_rate=16000
        # self.n_mels=80,
        self.n_mels=128 ### not channel num of mel but shape of whisper encoder output
        self.n_fft=400 # 1024
        self.hop_samples=160 # 256

        self.time = 2 ## audio time second, <30
        self.crop_mel_frames=62,  # Probably an error in paper.
        
        # Model params
        self.residual_layers=30
        self.residual_channels=64
        self.dilation_cycle_length=10
        self.unconditional = False
        self.noise_schedule=np.linspace(1e-4, 0.05, 50).tolist()
        self.inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5]
        self.audio_model = 'large'
        self.audio_trainable = False

        # unconditional sample len
        self.audio_len = self.sample_rate*self.time # unconditional_synthesis_samples

        self.whisper_len = self.time * 50
        self.num_channel = 128

if __name__ == '__main__':
    params = Config_Wave()
    dic = {key:value for key, value in params.__dict__.items() if not key.startswith('__') and not callable(key)}
    print(dic)