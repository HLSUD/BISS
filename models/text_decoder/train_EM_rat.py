### save std and mean of stimuli
import os
import sys
 
# setting path
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import numpy as np
from pathlib import Path
from typing import Union
import argparse
import config
from mGPT import GPT
from neuro_encoder import Neuro_Encoder
from stimuli_model import LMFeatures
from data_proc import get_stim, get_resp_word
from ridge import ridge, bootstrap_ridge
import pandas as pd
np.random.seed(42)

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_npy_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'{ext}' == 'npy'# type: ignore

def get_resp_stim(gpt, neural_encoder, features,word_info_df, save_location=None):
    rat_neural_path = os.path.join('data/rat_eeg/' ,str(args.subject))
    rat_neural_paths = [str(f) for f in sorted(Path(rat_neural_path).rglob('*')) if is_npy_ext(f) and os.path.isfile(f)]
    resp = [np.load(rat_neural_paths[i]) for i in range(len(rat_neural_paths))]
    resp = np.vstack(resp)

    print('Geting stim...')
    rstim, r_mean, r_std = get_stim(word_info_df, features, config.GPT_WORDS)
    if save_location is not None:
        np.save(save_location+'/r_mean.npy',r_mean)
        np.save(save_location+'/r_std.npy',r_std)
    print('Geting resp...')
    rresp = get_resp_word(resp, word_info_df, config.GPT_WORDS)
    print('Passing resp to encoder...')
    rresp = neural_encoder.make_resp(rresp)

    N, chan, remb_len = rresp.shape
    N, word_len, semb_len =  rstim.shape
    rresp = np.reshape(rresp,(N,chan*remb_len))
    rstim = np.reshape(rstim,(N,word_len*semb_len))
    
    if save_location is not None:
        np.save(rstim,save_location+'/rstim.npy')
        np.save(rresp,save_location+'/rresp.npy')
    return rresp, rstim, r_mean, r_std

def regression(rresp, rstim, r_mean, r_std, save_location, save_name):
    nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))
    print('Start regression...')
    weights, alphas, bscorrs = bootstrap_ridge(rstim, rresp, use_corr = False, alphas = config.ALPHAS,
        nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks)        
    bscorrs = bscorrs.mean(2).max(0)
    vox = np.sort(np.argsort(bscorrs)[-config.VOXELS:])
    del rstim, rresp
   
    np.savez(os.path.join(save_location, "encoding_model_%s" % save_name), 
        weights = weights, alphas = alphas, voxels = vox, stories = stories,r_mean = r_mean, r_std = r_std
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--gpt", type = str, default = "perceived")
    # parser.add_argument("--session", type = str, required = True) # single female _single_f
    #     default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    args = parser.parse_args()
    save_location = os.path.join(config.RESULT_DIR, args.subject)
    save_location = '/Volumes/Westside/lhh/'+args.subject+'/rat'
    os.makedirs(save_location, exist_ok = True)

    # without noise estimation
    # training stories
    fs = 100
    stories = []

    # load gpt
    # with open(os.path.join(config.DATA_LM_DIR, args.gpt, "vocab.json"), "r") as f:
    #     gpt_vocab = json.load(f)
    gpt = GPT(config.GPT_name, None, device = config.GPT_DEVICE)
    path = config.NEURO_ENCODER_PATH
    neural_encoder = Neuro_Encoder(path, device = config.NEURO_DEVICE)
    features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)
    
    # load word info
    word_info_path = 'data/text_decoding/word_whole.csv'
    word_info_df = pd.read_csv(word_info_path)
    ## second to 10 milisecond
    word_info_df['onset'] = np.array(word_info_df['offset'] * fs, dtype=np.int32)
    word_info_df['offset'] = np.array(word_info_df['onset'] * fs, dtype=np.int32)

    print(f'Subject id: {args.subject}, word info {word_info_path}')
    # rresp, rstim, r_mean, r_std = get_resp_stim(gpt, neural_encoder, features,word_info_df, save_location=None)
    
    print(save_location)
    load_resp_stim = True
    if load_resp_stim:
        rresp = np.load(save_location+'/rresp.npy')
        rstim = np.load(save_location+'/rstim.npy')
        r_mean = np.load(save_location+'/r_mean.npy')
        r_std = np.load(save_location+'/r_std.npy')
    stim_resp_len = 2000
    rresp = rresp[:stim_resp_len,:]
    rstim = rstim[:stim_resp_len,:]
    print(rstim.shape,rresp.shape)
    save_name = args.gpt
    regression(rresp, rstim, r_mean, r_std, save_location, save_name)
    # estimate noise model
    # stim_dict = {story : get_stim([story], features, tr_stats = tr_stats) for story in stories}
    # resp_dict = get_resp_word(args.subject, stories, stack = False, vox = vox)
    # noise_model = np.zeros([len(vox), len(vox)])
    # for hstory in stories:
    #     tstim, hstim = np.vstack([stim_dict[tstory] for tstory in stories if tstory != hstory]), stim_dict[hstory]
    #     tresp, hresp = np.vstack([resp_dict[tstory] for tstory in stories if tstory != hstory]), resp_dict[hstory]
    #     bs_weights = ridge(tstim, tresp, alphas[vox])
    #     resids = hresp - hstim.dot(bs_weights)
    #     bs_noise_model = resids.T.dot(resids)
    #     noise_model += bs_noise_model / np.diag(bs_noise_model).mean() / len(stories)
    # del stim_dict, resp_dict
    
    # # save
    # save_location = os.path.join(config.MODEL_DIR, args.subject)
    # os.makedirs(save_location, exist_ok = True)
    # np.savez(os.path.join(save_location, "encoding_model_%s" % args.gpt), 
        # weights = weights, noise_model = noise_model, alphas = alphas, voxels = vox, stories = stories,
        # tr_stats = np.array(tr_stats), word_stats = np.array(word_stats))