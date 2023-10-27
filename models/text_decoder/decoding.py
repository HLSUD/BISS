### likelihood
import os
import numpy as np
import json
import argparse
# import h5py
from pathlib import Path
import pandas as pd
import config
import logging
from mGPT import GPT
from decoder import Decoder, Hypothesis
from LanguageModel import LanguageModel
from encoder_model import EncodingModel
from neuro_encoder import Neuro_Encoder
from stimuli_model import StimulusModel, affected_trs, LMFeatures
from data_proc import get_stim, get_resp_word
# from utils_stim import predict_word_rate, predict_word_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--experiment", type = str, required = True)
    # parser.add_argument("--task", type = str, required = True)
    args = parser.parse_args()
    sent_words = 3
    # determine GPT checkpoint based on experiment
    if args.experiment in ["imagined_speech"]: gpt_checkpoint = "imagined"
    else: gpt_checkpoint = "perceived"

    # determine word rate model voxels based on experiment
    if args.experiment in ["imagined_speech", "perceived_movies"]: word_rate_voxels = "speech"
    else: word_rate_voxels = "auditory"

    save_location = os.path.join(config.RESULT_DIR, args.subject, args.experiment)
    os.makedirs(save_location, exist_ok = True)
    logging.basicConfig(filename=save_location+'/text.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

    # load responses
    # hf = h5py.File(os.path.join(config.DATA_TEST_DIR, "test_response", args.subject, args.experiment, args.task + ".hf5"), "r")
    # resp = np.nan_to_num(hf["data"][:])
    resp = np.load('/Users/honghaoliu/Documents/GitHub/BISR/code/results/regression_arr/resp.npy')
    path = '/Users/honghaoliu/Documents/GitHub/BISR/code/results/checkpoints/stage1_b256_lr1e-4_w20e200_hop100_smooth_cor_m75.pth'
    neural_path = 'data/eeg_data/subj'+str(args.subject)+'_single_f.npy'
    neural_encoder = Neuro_Encoder(path, device = config.NEURO_DEVICE)
    word_info_path = 'data/text_decoding/word_s1.csv'
    word_info_df = pd.read_csv(word_info_path)
    

    # hf.close()
    
    # load gpt
    # with open(os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "vocab.json"), "r") as f:
    #     gpt_vocab = json.load(f)
    # with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
    #     decoder_vocab = json.load(f)
    gpt = GPT(config.GPT_name, None, device = config.GPT_DEVICE)
    features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)
    lm = LanguageModel(gpt, vocab=None, nuc_mass = config.LM_MASS, nuc_ratio = config.LM_RATIO)

    rstim, r_mean, r_std = get_stim(word_info_df[:sent_words], features, config.GPT_WORDS) ## mean and std
    # rresp = get_resp_word(neural_path, word_info_df, config.GPT_WORDS)
    # rresp = neural_encoder.make_resp(rresp)
    # N, chan, remb_len = rresp.shape
    N, word_len, semb_len =  rstim.shape
    print(f'stim shape {rstim.shape}')
    # rresp = np.reshape(rresp,(N,chan*remb_len))
    rresp = np.load('/Users/honghaoliu/Documents/GitHub/BISR/code/results/regression_arr/resp.npy')
    rstim = np.reshape(rstim,(N,word_len*semb_len))
    init_stim = rstim[0]
    # load models
    load_location = os.path.join(config.RESULT_DIR, args.subject)
    # word_rate_model = np.load(os.path.join(load_location, "word_rate_model_%s.npz" % word_rate_voxels), allow_pickle = True)
    encoding_model = np.load(os.path.join(config.RESULT_DIR, "encoding_model_%s.npz" % gpt_checkpoint))
    weights = encoding_model["weights"]
    vox = encoding_model["voxels"]
    # noise_model = encoding_model["noise_model"]
    # tr_stats = encoding_model["tr_stats"]
    # word_stats = encoding_model["word_stats"]

    noise = np.zeros([len(vox), len(vox)])
    em = EncodingModel(rresp, weights, vox, noise, device = config.EM_DEVICE)
    em.set_shrinkage(config.NM_ALPHA) #noise related
    # assert args.task not in encoding_model["stories"] # test related
    
    # predict word times
    # word_rate = predict_word_rate(resp, word_rate_model["weights"], word_rate_model["voxels"], word_rate_model["mean_rate"])
    # if args.experiment == "perceived_speech": word_times, tr_times = predict_word_times(word_rate, resp, starttime = -10)
    # else: word_times, tr_times = predict_word_times(word_rate, resp, starttime = 0)
    # lanczos_mat = get_lanczos_mat(word_times, tr_times)

    word_times = (word_info_df['onset']*10/1000)[:sent_words] # second
    # decode responses
    decoder = Decoder(word_times, config.WIDTH)
    hyp = decoder.beam[0]
    hyp.words = ['当','我','还','只有','六']
    hyp.logprobs = [0,0,0,0,0]
    
    hyp.embs = [hyp.embs+[init_stim[i*semb_len:(i+1)*semb_len]] for i in range(5)]
    logging.info(hyp)
    
    # local_extensions = [Hypothesis(parent = hyp, extension = x) for x in zip(nuc, logprobs, extend_embs)]
    # decoder.add_extensions(local_extensions, likelihoods, nextensions)

    # sm = StimulusModel(device = config.SM_DEVICE)
    sm = None
    logging.info(f'mean std {r_mean.shape},{r_std.shape}')
    total_stim_emb = np.expand_dims(init_stim.flatten(), axis=0)
    # stim = np.expand_dims(init_stim.flatten(), axis=0)
    
    for sample_index in range(len(word_times)):
        # trs = affected_trs(decoder.first_difference(), sample_index, lanczos_mat)
        # ncontext = decoder.time_window(sample_index, config.LM_TIME, floor = 5)
        ncontext = config.GPT_WORDS
        logging.info('-----------------------------------------------------')
        logging.info(f'sample id {sample_index} len beam {len(decoder.beam)}, width {decoder.beam_width}')
        beam_nucs = lm.beam_propose(decoder.beam, ncontext)
        
        logging.info(f'decoding beam {len(beam_nucs)}, beam len {len(decoder.beam)}')
        for i in range(5):
            if i < len(decoder.beam):
                logging.info(decoder.beam[i].words)
        
       
        
        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[c]
            
            if len(nuc) < 1: continue
            extend_words = [hyp.words + [x] for x in nuc]
            
            extend_embs = list(features.extend(extend_words))
            extend_embs = np.nan_to_num(np.dot((extend_embs - r_mean), np.linalg.inv(np.diag(r_std))))
            N,num_word,_ = extend_embs.shape
            stim = total_stim_emb
            stim = np.tile(stim[:,num_word*semb_len:], (extend_embs.shape[0], 1))
            stim = np.concatenate((stim, extend_embs.reshape(N,-1)),axis=1)
            # stim = sm.make_variants(sample_index, hyp.embs, extend_embs, trs)
            likelihoods = em.prs(stim, sample_index+1)
            local_extensions = [Hypothesis(parent = hyp, extension = x) for x in zip(nuc, logprobs, extend_embs)]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)
        decoder.extend(verbose = False)
        
    if args.experiment in ["perceived_movie", "perceived_multispeaker"]: decoder.word_times += 10
    decoder.save(os.path.join(save_location))