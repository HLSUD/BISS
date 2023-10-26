import numpy as np
from models.eeg_dataset import eeg_interp_repeat
from models.utils import smooth_signal

def get_stim(word_info_df, features, word_len):
    # one story
    word_seq = list(word_info_df['word'])
    word_vecs = features.make_stim(word_seq)
    # word_mean, word_std = word_vecs.mean(0), word_vecs.std(0) ## check
    r_mean, r_std = word_vecs.mean(0), word_vecs.std(0)
    r_std[r_std == 0] = 1
    ds_mat = np.nan_to_num(np.dot((word_vecs - r_mean), np.linalg.inv(np.diag(r_std))))
    ## win resp
    N, emb_len = ds_mat.shape
    win_mat = np.empty((N-word_len,word_len,emb_len))
    for i in range(N-word_len):
        win_mat[i] = ds_mat[i:i+word_len,:]
    return win_mat, r_mean, r_std


def get_resp_word(data_path, word_info_df, word_len,out_chan=128, timepts=512, stack = True):
    # one story
    resp = np.load(data_path)
    
    # sec_2_resp = np.load(data_path+data_name_m)

    onsets = np.array(word_info_df['onset'])
    offsets = np.array(word_info_df['offset'])
    
    num_word = len(word_info_df)
    resp_arr = np.empty((num_word-word_len,out_chan,timepts))
    print(num_word - word_len)
    for i in range(num_word - word_len):
        seg_resp = resp[:,onsets[i]:onsets[i+word_len]]
        seg_resp = eeg_interp_repeat(seg_resp,out_chan,timepts)
        resp_arr[i] = np.vstack([smooth_signal(seg_resp[ci]) for ci in range(out_chan)])
    return resp_arr

def get_resp_time(data_path, subject, word_info_df, fs, hop_dur, word_len, stack = True):
    data_name_f = 'subj'+str(subject) + '_single_f.npy'
    data_name_m = 'subj'+str(subject) + '_single_m.npy'
    sec_1_resp = np.load(data_path+data_name_f)
    sec_2_resp = np.load(data_path+data_name_m)
    seg_ids = word_info_df['seg_id']
    sec_ids = word_info_df['section']
    seg_resp_len = hop_dur * fs
    pre_seg_id = 0
    counter = 0
    resp_arr = []
    for i in range(len(seg_ids)):
        if seg_ids[i] != pre_seg_id:
            if counter < word_len:
                # save check shape
                if sec_ids == 1:
                    cur_resp = sec_1_resp[(pre_seg_id-1)*seg_resp_len:pre_seg_id*seg_resp_len]
                    resp_arr = np.vstack(resp_arr,cur_resp) #check
                elif sec_ids == 2:
                    cur_resp = sec_2_resp[(pre_seg_id-1)*seg_resp_len:pre_seg_id*seg_resp_len]
                    resp_arr = np.vstack(resp_arr,cur_resp)
            else:
                counter = 1
        else:
            counter = counter + 1
            if counter >=word_len:
                cur_resp = sec_2_resp[(pre_seg_id-1)*seg_resp_len:pre_seg_id*seg_resp_len]
                resp_arr = np.vstack(resp_arr,cur_resp)

    return resp_arr