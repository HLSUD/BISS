import numpy as np
import torch
from models.NN.eeg_mae import eeg_encoder
torch.set_default_tensor_type(torch.FloatTensor)

class Neuro_Encoder():
    """class for computing the likelihood of observing brain recordings given a word sequence
    """
    def __init__(self, ckpt_path, device = "cpu"):
        self.device = device
        # self.weights = torch.from_numpy(weights[:, voxels]).float().to(self.device)
        # load model
        self.base = eeg_encoder(512, patch_size=4, embed_dim=1024, in_chans=128, depth=24, num_heads=16).eval().to(device)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        print(type(state_dict))
        # print(state_dict.keys())
        print('----------------------------------------')
        
        self.base.load_checkpoint(state_dict)
        # self.resp = torch.from_numpy(resp).float().to(self.device)
        
        
    def make_resp(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        N = x.shape[0]
        res = np.empty((N,128,1024))
        for i in range(N):
            res[i] = self.base(x[i]).detach().cpu().numpy()[0]
        return res