import pandas as pd
import numpy as np

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


if __name__ == '__main__':
    word_rate_info('data/little_prince_word_info/word_s1.csv')
    