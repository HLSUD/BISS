### Rat semantic text decoding
from models.text_decoder.data_proc import get_stim_mean_std
import os
import numpy as np
import json
import argparse
# import h5py
from pathlib import Path
from typing import Union
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
from train_EM_rat import get_resp_stim
# from utils_stim import predict_word_rate, predict_word_times

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_npy_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'{ext}' == 'npy'# type: ignore

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    # parser.add_argument("--experiment", type = str, required = True)
    parser.add_argument("--word_info_path", type = str, required = True)
    parser.add_argument("--logname", type = str, default= 'text_gpt.log')
    # parser.add_argument("--task", type = str, required = True)
    args = parser.parse_args()
    # merge bands
    
    sent_words = 17
    beam_log_len = 15
    fs = 100
    gpt_checkpoint = "perceived"
    # determine GPT checkpoint based on experiment
    # if args.experiment in ["imagined_speech"]: gpt_checkpoint = "imagined"
    # else: gpt_checkpoint = "perceived"

    # determine word rate model voxels based on experiment
    # if args.experiment in ["imagined_speech", "perceived_movies"]: word_rate_voxels = "speech"
    # else: word_rate_voxels = "auditory"

    save_location = os.path.join(config.RESULT_DIR, args.subject, 'rat')
    os.makedirs(save_location, exist_ok = True)
    save_location = '/Volumes/Westside/lhh/'+args.subject+'/rat'
    logging.basicConfig(filename=save_location+'/'+args.logname,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
    logging.info(f'beam width {config.WIDTH}, number of context {config.GPT_WORDS}')
    # load responses
    # hf = h5py.File(os.path.join(config.DATA_TEST_DIR, "test_response", args.subject, args.experiment, args.task + ".hf5"), "r")
    # resp = np.nan_to_num(hf["data"][:])
    # resp = np.load('/Users/honghaoliu/Documents/GitHub/BISR/code/results/regression_arr/resp.npy')
    path = config.NEURO_ENCODER_PATH
    neural_encoder = Neuro_Encoder(path, device = config.NEURO_DEVICE)
    word_info_path = args.word_info_path
    word_info_df = pd.read_csv(word_info_path)
    ## second to 10 milisecond
    word_info_df['onset'] = np.array(word_info_df['offset'] * fs, dtype=np.int32)
    word_info_df['offset'] = np.array(word_info_df['onset'] * fs, dtype=np.int32)
    
    # load gpt
    
    gpt = GPT(config.GPT_name, None, device = config.GPT_DEVICE)
    features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)
    lm = LanguageModel(gpt, vocab=None, nuc_mass = config.LM_MASS, nuc_ratio = config.LM_RATIO)

    # rstim, _, _ = get_stim(word_info_df, features, config.GPT_WORDS) ## mean and std ### the shape is not as what expected
    
    # load models
    # load_location = os.path.join(config.RESULT_DIR, args.subject)
    # word_rate_model = np.load(os.path.join(load_location, "word_rate_model_%s.npz" % word_rate_voxels), allow_pickle = True)
    encoding_model = np.load(os.path.join(config.RESULT_DIR,args.subject,'rat', "encoding_model_%s.npz" % gpt_checkpoint))
    
    print(save_location)
    load_resp_stim = True
    
    rat_neural_path = os.path.join('data/rat_eeg/' ,str(args.subject))
    rat_neural_paths = [str(f) for f in sorted(Path(rat_neural_path).rglob('*')) if is_npy_ext(f) and os.path.isfile(f)]
    resp = [np.load(rat_neural_paths[i]) for i in range(len(rat_neural_paths))]
    resp = np.vstack(resp)
    if load_resp_stim:
        # rresp = np.load(save_location+'/rresp.npy')
    #     rstim = np.load(save_location+'/rstim.npy')
        r_mean = np.load(save_location+'/r_mean.npy')
        r_std = np.load(save_location+'/r_std.npy')
    # else:
    #     rresp, rstim, r_mean, r_std = get_resp_stim(gpt, neural_encoder, features,word_info_df, save_location=None)
    
    weights = encoding_model["weights"]
    vox = encoding_model["voxels"]
    # stim_stat = np.load(os.path.join(config.RESULT_DIR, "stim_mean_std_%s.npz" % gpt_checkpoint))
    # r_mean = stim_stat["r_mean"]
    # r_std = stim_stat["r_std"]

    rstim,_,_ = get_stim_mean_std(word_info_df, features, config.GPT_WORDS, r_mean, r_std)
    
    
    rresp = get_resp_word(resp, word_info_df, config.GPT_WORDS)
    # print(rresp.shape)
    rresp = neural_encoder.make_resp(rresp)
    
    N, chan, remb_len = rresp.shape
    N, word_len, semb_len =  rstim.shape
    rresp = np.reshape(rresp,(N,chan*remb_len))
    
    # N, chan, remb_len = rresp.shape
    N, word_len, semb_len =  rstim.shape
    print(f'stim shape {rstim.shape}')
    rstim = np.reshape(rstim,(N,word_len*semb_len))
    init_stim = rstim[0]
    
    # noise model
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
    hyp.words = ['在','一','本','描写','原始']
    hyp.logprobs = [0,0,0,0,0]

    
    hyp.embs = [hyp.embs+[init_stim[i*semb_len:(i+1)*semb_len]] for i in range(5)]
    logging.info(hyp)
    
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
        for i in range(beam_log_len):
            if i < len(decoder.beam):
                logging.info(decoder.beam[i].words)
        
       
        
        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[c]
            
            if len(nuc) < 1: continue
            extend_words = [hyp.words + [x] for x in nuc]
            
            extend_embs = features.extend(extend_words)
            
            extend_embs = np.nan_to_num(np.dot((extend_embs - r_mean), np.linalg.inv(np.diag(r_std))))
            
            N,num_word,_ = extend_embs.shape
            # stim = total_stim_emb
            # stim = np.tile(stim[:,num_word*semb_len:], (extend_embs.shape[0], 1))
            # stim = np.concatenate((stim, extend_embs.reshape(N,-1)),axis=1)
            stim = extend_embs.reshape(N,-1)
            # stim = sm.make_variants(sample_index, hyp.embs, extend_embs, trs)
            likelihoods = em.prs(stim, sample_index+1)
            # local_extensions = [Hypothesis(parent = hyp, extension = x[:3], likelihood=x[3], previous_likelihood = pre_likeli) for x in zip(nuc, logprobs, extend_embs, likelihoods)]
            local_extensions = [Hypothesis(parent = hyp, extension = x) for x in zip(nuc, logprobs, extend_embs)]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)
        decoder.extend(verbose = False)
        
    # if args.experiment in ["perceived_movie", "perceived_multispeaker"]: decoder.word_times += 10
    decoder.save(os.path.join(save_location))