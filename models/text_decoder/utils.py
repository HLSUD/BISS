import os
import pandas as pd
import numpy as np
from typing import Union
from pathlib import Path
from jiwer import wer,cer
from datasets import load_metric
from bert_score import BERTScorer

def word_rate_info(path):
    """ a csv file with word onset offset information
    """
    df = pd.read_csv(path)
    onset = np.array(df['onset'])
    offset = np.array(df['offset'])

    max_ind = np.argmax(offset - onset)
    max_val = np.max(offset - onset)
    min_ind = np.argmin(offset - onset)
    min_val = np.min(offset - onset)
    mean_val = np.mean(offset - onset)
    std_val = np.std(offset - onset)
    print(f"max time {max_val}, min time {min_val}, mean {mean_val}, std {std_val}")

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_csv_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'{ext}' == 'csv'

def generate_datasets(sections_path, hop_dur, fs, train_test_ratio=0.9):
    word_paths = [str(f) for f in sorted(Path(sections_path).rglob('*')) if is_csv_ext(f) and os.path.isfile(f)]
    num_sections = len(word_paths)

    df_list = [pd.read_csv(path) for path in word_paths]
    for i in range(num_sections):
        seg_ind_list = []
        
        onset = np.array(df_list[i]['onset'])
        offset = np.array(df_list[i]['offset'])
        # seg_num = audio_dur // hop_dur
        seg_i = 0
        for oni,offi in zip(onset, offset):
            ctl = seg_i*hop_dur*fs
            if offi<ctl:
                seg_ind_list.append(seg_i)
            elif oni < ctl:
                seg_ind_list.append(seg_i)
            else:
                seg_i = seg_i + 1
                seg_ind_list.append(seg_i)
        section_list = [i+1]*len(onset)
        df_list[i]['seg_id'] = seg_ind_list
        df_list[i]['section'] = section_list
        df_list[i].to_csv(word_paths[i],index=False)
    return 

def zscore(mat, return_unzvals=False):
    """Z-scores the rows of [mat] by subtracting off the mean and dividing
    by the standard deviation.
    If [return_unzvals] is True, a matrix will be returned that can be used
    to return the z-scored values to their original state.
    """
    zmat = np.empty(mat.shape, mat.dtype)
    unzvals = np.zeros((zmat.shape[0], 2), mat.dtype)
    for ri in range(mat.shape[0]):
        unzvals[ri,0] = np.std(mat[ri,:])
        unzvals[ri,1] = np.mean(mat[ri,:])
        zmat[ri,:] = (mat[ri,:]-unzvals[ri,1]) / (1e-10+unzvals[ri,0])
    
    if return_unzvals:
        return zmat, unzvals
    
    return zmat

def center(mat, return_uncvals=False):
    """Centers the rows of [mat] by subtracting off the mean, but doesn't 
    divide by the SD.
    Can be undone like zscore.
    """
    cmat = np.empty(mat.shape)
    uncvals = np.ones((mat.shape[0], 2))
    for ri in range(mat.shape[0]):
        uncvals[ri,1] = np.mean(mat[ri,:])
        cmat[ri,:] = mat[ri,:]-uncvals[ri,1]
    
    if return_uncvals:
        return cmat, uncvals
    
    return cmat

def unzscore(mat, unzvals):
    """Un-Z-scores the rows of [mat] by multiplying by unzvals[:,0] (the standard deviations)
    and then adding unzvals[:,1] (the row means).
    """
    unzmat = np.empty(mat.shape)
    for ri in range(mat.shape[0]):
        unzmat[ri,:] = mat[ri,:]*(1e-10+unzvals[ri,0])+unzvals[ri,1]
    return unzmat

def gaussianize(vec):
    """Uses a look-up table to force the values in [vec] to be gaussian."""
    ranks = np.argsort(np.argsort(vec))
    cranks = (ranks+1).astype(float)/(ranks.max()+2)
    vals = scipy.stats.norm.isf(1-cranks)
    zvals = vals/vals.std()
    return zvals

def gaussianize_mat(mat):
    """Gaussianizes each column of [mat]."""
    gmat = np.empty(mat.shape)
    for ri in range(mat.shape[1]):
        gmat[:,ri] = gaussianize(mat[:,ri])
    return gmat

def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays] 
    (in samples).
    
    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt,ndim = stim.shape
    dstims = []
    for di,d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d<0: ## negative delay
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else: ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)

def mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.
    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array
    Output:
      mult_diag(d, mts, left=True) == dot(diag(d), mtx)
      mult_diag(d, mts, left=False) == dot(mtx, diag(d))
    
    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        return (d*mtx.T).T
    else:
        return d*mtx

import time
import logging
def counter(iterable, countevery=100, total=None, logger=logging.getLogger("counter")):
    """Logs a status and timing update to [logger] every [countevery] draws from [iterable].
    If [total] is given, log messages will include the estimated time remaining.
    """
    start_time = time.time()

    ## Check if the iterable has a __len__ function, use it if no total length is supplied
    if total is None:
        if hasattr(iterable, "__len__"):
            total = len(iterable)
    
    for count, thing in enumerate(iterable):
        yield thing
        
        if not count%countevery:
            current_time = time.time()
            rate = float(count+1)/(current_time-start_time)

            if rate>1: ## more than 1 item/second
                ratestr = "%0.2f items/second"%rate
            else: ## less than 1 item/second
                ratestr = "%0.2f seconds/item"%(rate**-1)
            
            if total is not None:
                remitems = total-(count+1)
                remtime = remitems/rate
                timestr = ", %s remaining" % time.strftime('%H:%M:%S', time.gmtime(remtime))
                itemstr = "%d/%d"%(count+1, total)
            else:
                timestr = ""
                itemstr = "%d"%(count+1)

            formatted_str = "%s items complete (%s%s)"%(itemstr,ratestr,timestr)
            if logger is None:
                print(formatted_str)
            else:
                logger.info(formatted_str)

"""
WER
"""
class WER(object):
    def __init__(self, use_score = True):
        self.use_score = use_score
    
    def score(self, ref, pred):
        scores = []
        for ref_seg, pred_seg in zip(ref, pred):
            if len(ref_seg) == 0 : error = 1.0
            else: error = wer(ref_seg, pred_seg)
            if self.use_score: scores.append(1 - error)
            else: scores.append(error)
        return np.array(scores)

"""
CER
"""
class CER(object):
    def __init__(self, use_score = True):
        self.use_score = use_score
    
    def score(self, ref, pred):
        scores = []
        for ref_seg, pred_seg in zip(ref, pred):
            if len(ref_seg) == 0 : error = 1.0
            else: error = cer(ref_seg, pred_seg)
            if self.use_score: scores.append(1 - error)
            else: scores.append(error)
        return np.array(scores)
    
"""
BLEU (https://aclanthology.org/P02-1040.pdf)
"""
class BLEU(object):
    def __init__(self, n = 4):
        self.metric = load_metric("bleu", keep_in_memory=True)
        self.n = n
        
    def score(self, ref, pred):
        results = []
        for r, p in zip(ref, pred):
            self.metric.add_batch(predictions=[p], references=[[r]])
            results.append(self.metric.compute(max_order = self.n)["bleu"])
        return np.array(results)
    
"""
METEOR (https://aclanthology.org/W05-0909.pdf)
"""
class METEOR(object):
    def __init__(self):
        self.metric = load_metric("meteor", keep_in_memory=True)

    def score(self, ref, pred):
        results = []
        ref_strings = [" ".join(x) for x in ref]
        pred_strings = [" ".join(x) for x in pred]
        for r, p in zip(ref_strings, pred_strings):
            self.metric.add_batch(predictions=[p], references=[r])
            results.append(self.metric.compute()["meteor"])
        return np.array(results)
        
"""
BERTScore (https://arxiv.org/abs/1904.09675)
"""
class BERTSCORE(object):
    def __init__(self, idf_sents=None, rescale = True, score = "f"):
        self.metric = BERTScorer(lang = "en", rescale_with_baseline = rescale, idf = (idf_sents is not None), idf_sents = idf_sents)
        if score == "precision": self.score_id = 0
        elif score == "recall": self.score_id = 1
        else: self.score_id = 2

    def score(self, ref, pred):
        ref_strings = [" ".join(x) for x in ref]
        pred_strings = [" ".join(x) for x in pred]
        return self.metric.score(cands = pred_strings, refs = ref_strings)[self.score_id].numpy()

if __name__ == '__main__':
    word_rate_info('data/little_prince_word_info/word_s2.csv')
    # generate_datasets('data/segs/', 1, 100, train_test_ratio=0.9)
    