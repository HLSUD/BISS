import pandas as pd
import numpy as np
import librosa
import librosa.display
import IPython

# max min and mean duration of a word/char
def eeg_word_duration_stat():
    # 0.06 1.21 0.29759269823160295 0.15984597761034613
    word_time_df = pd.read_csv('../word_s1.csv')
    start_time = np.array(word_time_df.iloc[:,1])
    end_time = np.array(word_time_df.iloc[:,2])
    words = word_time_df.iloc[:,0]
    dur = (end_time - start_time) /100
    return (dur.min(),dur.max(),dur.mean(),dur.std())
