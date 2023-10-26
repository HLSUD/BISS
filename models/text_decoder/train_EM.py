### save std and mean of stimuli
import os
import sys
 
# setting path
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import numpy as np
import json
import argparse
import config
from mGPT import GPT
from neuro_encoder import Neuro_Encoder
from stimuli_model import LMFeatures
from data_proc import get_stim, get_resp_word
from ridge import ridge, bootstrap_ridge
import pandas as pd
np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--gpt", type = str, default = "perceived")
    # parser.add_argument("--sessions", nargs = "+", type = int, 
    #     default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    args = parser.parse_args()

    # training stories
    stories = []
    # with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
    #     sess_to_story = json.load(f) 
    # for sess in args.sessions:
    #     stories.extend(sess_to_story[str(sess)])

    # load gpt
    # with open(os.path.join(config.DATA_LM_DIR, args.gpt, "vocab.json"), "r") as f:
    #     gpt_vocab = json.load(f)
    # gpt = GPT(config.GPT_name, None, device = config.GPT_DEVICE)
    # path = '/Users/honghaoliu/Documents/GitHub/BISR/code/results/checkpoints/stage1_b256_lr1e-4_w20e200_hop100_smooth_cor_m75.pth'
    # neural_encoder = Neuro_Encoder(path, device = config.NEURO_DEVICE)
    # features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)
    
    # estimate encoding model
    word_info_path = 'data/text_decoding/word_s1.csv'
    neural_path = 'data/eeg_data/subj'+str(args.subject)+'_single_f.npy'
    word_info_df = pd.read_csv(word_info_path)
    print('Geting stim...')
    # rstim = get_stim(word_info_df, features, config.GPT_WORDS)
    # print('Geting resp...')
    # rresp = get_resp_word(neural_path, word_info_df, config.GPT_WORDS)
    # print('Passing resp to encoder...')
    # rresp = neural_encoder.make_resp(rresp)
    
    # N, chan, remb_len = rresp.shape
    # N, word_len, semb_len =  rstim.shape
    # rresp = np.reshape(rresp,(N,chan*remb_len))
    # rstim = np.reshape(rstim,(N,word_len*semb_len))
    
    # rresp = neural_encoder(rresp)
    rresp = np.load('/Users/honghaoliu/Documents/GitHub/BISR/code/results/regression_arr/resp.npy')
    rstim = np.load('/Users/honghaoliu/Documents/GitHub/BISR/code/results/regression_arr/rstim.npy')

    print(rstim.shape,rresp.shape)

    nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))
    print('Start regression...')
    weights, alphas, bscorrs = bootstrap_ridge(rstim, rresp, use_corr = False, alphas = config.ALPHAS,
        nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks)        
    bscorrs = bscorrs.mean(2).max(0)
    vox = np.sort(np.argsort(bscorrs)[-config.VOXELS:])
    del rstim, rresp
    # save_location = os.path.join(config.MODEL_DIR, args.subject)
    save_location = config.RESULT_DIR
    os.makedirs(save_location, exist_ok = True)
    # without noise estimation
    np.savez(os.path.join(save_location, "encoding_model_%s" % args.gpt), 
        weights = weights, alphas = alphas, voxels = vox, stories = stories,
        )
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