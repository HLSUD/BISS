# https://github.com/moein-shariatnia/OpenAI-CLIP
# https://github.com/microsoft/CLAP/blob/main/src/CLAPWrapper.py

import random
import torchaudio
from torch._six import string_classes
import collections
import re
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from models.utils import read_config_as_args
from models.clap import CLAP
import math
import torchaudio.transforms as T
import os
import torch
from importlib_resources import files


class encoderWrapper():
    """
    A class for interfacing CLAP model.  
    """

    def __init__(self, model_fp, use_cuda=False):
        self.np_str_obj_array_pattern = re.compile(r'[SaUO]')
        self.file_path = os.path.realpath(__file__)
        self.default_collate_err_msg_format = (
            "default_collate: batch must contain tensors, numpy arrays, numbers, "
            "dicts or lists; found {}")
        self.config_as_str = files('configs').joinpath('config.yml').read_text()
        self.model_fp = model_fp
        self.use_cuda = use_cuda
        self.clap, self.tokenizer, self.args = self.load_clap()

    def load_clap(self):
        r"""Load CLAP model with args from config file"""

        args = read_config_as_args(self.config_as_str, is_config_str=True)

        if 'bert' in args.text_model:
            self.token_keys = ['input_ids', 'token_type_ids', 'attention_mask']
        else:
            self.token_keys = ['input_ids', 'attention_mask']

### REVISE !!!

        clap = CLAP(
            audioenc_name=args.audioenc_name,
            sample_rate=args.sampling_rate,
            window_size=args.window_size,
            hop_size=args.hop_size,
            mel_bins=args.mel_bins,
            fmin=args.fmin,
            fmax=args.fmax,
            classes_num=args.num_classes,
            out_emb=args.out_emb,
            text_model=args.text_model,
            transformer_embed_dim=args.transformer_embed_dim,
            d_proj=args.d_proj
        )

        # Load pretrained weights for model
        model_state_dict = torch.load(self.model_fp, map_location=torch.device('cpu'))['model']
        clap.load_state_dict(model_state_dict)

        clap.eval()  # set clap in eval mode
        tokenizer = AutoTokenizer.from_pretrained(args.text_model)

        if self.use_cuda and torch.cuda.is_available():
            clap = clap.cuda()

        return clap, tokenizer, args

    def default_collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if self.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(
                        self.default_collate_err_msg_format.format(elem.dtype))

                return self.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            return {key: self.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError(
                    'each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [self.default_collate(samples) for samples in transposed]

        raise TypeError(self.default_collate_err_msg_format.format(elem_type))

    # ********************
    # --------------------
    # --------------------
    # --------------------
    # --------------------
    def load_eeg_to_tensor(eeg_data, start_position, timepoints, freq, duration):
        neuro_data = eeg_data[start_position:start_position+timepoints].reshape(-1,eeg_data.shape[-1]*timepoints)
        neuro_data = np.squeeze(neuro_data)
        if timepoints < duration * freq:
            repeat_factor = int(np.ceil(duration * freq / timepoints))
            # Repeat neuro data by repeat_factor to match duration
            neuro_data = neuro_data.repeat(repeat_factor)
            neuro_data = neuro_data[0:duration*freq]
        else:
            start_index = random.randrange(timepoints - duration*freq)
            neuro_data = neuro_data[start_index:start_index + duration*freq]
        neuro_data = torch.tensor(neuro_data,dtype=torch.float32)

    def preprocess_neuro(self, neuro_indices, resample):
        
        neuro_tensors = []
        for idx in neuro_indices:
            neuro_tensor = load_eeg_to_tensor(eeg_data, idx, timepoints, freq, duration)
            #### reshape???
            neuro_tensor = neuro_tensor.reshape(
                1, -1).cuda() if self.use_cuda and torch.cuda.is_available() else neuro_tensor.reshape(1, -1)
            neuro_tensors.append(neuro_tensor)
        batch = None
        #batch = self.default_collate(neuro_tensors)
        return batch
    
    def preprocess_text(self, text_queries):
        r"""Load list of class labels and return tokenized text"""
        tokenized_texts = []
        for ttext in text_queries:
            tok = self.tokenizer.encode_plus(
                text=ttext, add_special_tokens=True, max_length=self.args.text_len, pad_to_max_length=True, return_tensors="pt")
            for key in self.token_keys:
                tok[key] = tok[key].reshape(-1).cuda() if self.use_cuda and torch.cuda.is_available() else tok[key].reshape(-1)
            tokenized_texts.append(tok)
        return self.default_collate(tokenized_texts)


