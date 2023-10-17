import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from models.whisper.__init__ import load_model,pad_or_trim,log_mel_spectrogram,load_audio
from models.whisper.decoding import PyTorchInference, DecodingOptions, DecodingResult, decode, GreedyDecoder
from models.whisper.tokenizer import Tokenizer, get_tokenizer
from models.whisper.decoding import SuppressBlank, SuppressTokens, ApplyTimestampRules, MaximumLikelihoodRanker
from models.whisper.audio import CHUNK_LENGTH
import torch, torchaudio

def _get_suppress_tokens(tokenizer, options):
    suppress_tokens = options.suppress_tokens

    if isinstance(suppress_tokens, str):
        suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

    if -1 in suppress_tokens:
        suppress_tokens = [t for t in suppress_tokens if t >= 0]
        suppress_tokens.extend(tokenizer.non_speech_tokens)
    elif suppress_tokens is None or len(suppress_tokens) == 0:
        suppress_tokens = []  # interpret empty string as an empty list
    else:
        assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

    suppress_tokens.extend(
        [
            tokenizer.transcribe,
            tokenizer.translate,
            tokenizer.sot,
            tokenizer.sot_prev,
            tokenizer.sot_lm,
        ]
    )
    if tokenizer.no_speech is not None:
        # no-speech probability is collected separately
        suppress_tokens.append(tokenizer.no_speech)

    return tuple(sorted(set(suppress_tokens)))


device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model('tiny').to(device)
print(f"Device: {device}...")

##### load audio and to mel spectrogram
audio_path = os.path.join(os.path.dirname(__file__), "../data/beast_ch.wav")
# audio = load_audio(audio_path)
audio, sr = torchaudio.load(audio_path)
if sr != 16000:
    audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)[0,:]

audio = pad_or_trim(audio)
    
mel = log_mel_spectrogram(audio).to(model.device)

_, probs = model.detect_language(mel)

## Tokenize
language = "zh"
task = "transcribe"
tokenizer = get_tokenizer(
    model.is_multilingual, language=language, task=task
)

### decoding options
all_tokens = []

#initial_prompt = "以下是普通话的句子"
#print(len(initial_prompt))
#if initial_prompt is not None:
#        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
#        all_tokens.extend(initial_prompt_tokens)
#print(all_tokens)

options = DecodingOptions(fp16 = False,prompt = all_tokens[:])

#result = model.decode( mel, options)
#print(result.text)
#exit()

## decoder 
# decoder: implements how to select the next tokens, given the autoregressive distribution
decoder = GreedyDecoder(options.temperature, tokenizer.eot)

# get_initial_tokens
n_ctx: int = model.dims.n_text_ctx # input text context size (text win size), default 448
### prompt used to set simplied chinese
prompt = options.prompt
tokens = list(tokenizer.sot_sequence)
if prompt:
    prompt_tokens = (
        tokenizer.encode(" " + prompt.strip())
        if isinstance(prompt, str)
        else prompt
    )
    tokens = (
        [tokenizer.sot_prev]
        + prompt_tokens[-(n_ctx // 2 - 1) :]
        + tokens
    )
initial_tokens = tuple(tokens)
sot_index = initial_tokens.index(tokenizer.sot)
sample_begin: int = len(initial_tokens)

# logit filters: applies various rules to suppress or penalize certain tokens
logit_filters = []
if options.suppress_blank:
    logit_filters.append(SuppressBlank(tokenizer, sample_begin))
if options.suppress_tokens:
    logit_filters.append(SuppressTokens(_get_suppress_tokens(tokenizer,options)))
if not options.without_timestamps:
    precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
    max_initial_timestamp_index = None
    if options.max_initial_timestamp:
        max_initial_timestamp_index = round(
            options.max_initial_timestamp / precision
        )
    logit_filters.append(
        ApplyTimestampRules(
            tokenizer, sample_begin, max_initial_timestamp_index
        )
    )

### this class could release some GPU memory
inference = PyTorchInference(model, len(initial_tokens))

## mel spectrogram
single = (mel.ndim == 2)
if single:
    mel = mel.unsqueeze(0)

### inference encoder
audio_features = model.embed_audio(mel)

n_audio = 1
tokens = torch.tensor([initial_tokens]).repeat(n_audio, 1).to(device)

tokens = tokens.repeat_interleave(1, dim=0)

## sampleing loop
n_batch = tokens.shape[0]
sum_logprobs = torch.zeros((n_batch)).to(device)
no_speech_probs = [np.nan] * n_batch
sample_len: int = options.sample_len or model.dims.n_text_ctx // 2
#print(f"Sample length: {sample_len}...")
n_ctx = 50 # set smaller, cause memory may be insufficient
#initial_token_length = tokens.shape[-1]
try:
    for i in range(sample_len):
        # print(type(tokens),type(audio_features))
        #if tokens.shape[-1] > initial_token_length:
            # only need to use the last token except in the first forward pass
        # consider the logits at the last token only
        logits = inference.logits(tokens, audio_features)
        #logits = model.decoder(tokens, audio_features)[:, -1]
        if (
            i == 0 and tokenizer.no_speech is not None
        ):  # save no_speech_probs
            probs_at_sot = logits[:, sot_index].float().softmax(dim=-1)
            no_speech_probs = probs_at_sot[:, tokenizer.no_speech].tolist()
        logits = logits[:, -1] 

        # apply the logit filters, e.g. for suppressing or applying penalty to
        for logit_filter in logit_filters:
            logit_filter.apply(logits, tokens)
        # expand the tokens tensor with the selected next tokens
        tokens, completed = decoder.update(tokens, logits, sum_logprobs)
        #print("torch.cuda.memory_reserved: %fGB %d"%(i,torch.cuda.memory_reserved(0)/1024/1024/1024))

        if completed or tokens.shape[-1] > n_ctx:
            break
finally:
    inference.cleanup_caching()
    print("Clean up cache...")

# reshape the tensors to have (n_audio, n_group) as the first two dimensions
n_group = 1
audio_features = audio_features[:: n_group]
no_speech_probs = no_speech_probs[:: n_group]
assert audio_features.shape[0] == len(no_speech_probs) == n_audio

tokens = tokens.reshape(n_audio, n_group, -1)
sum_logprobs = sum_logprobs.reshape(n_audio, n_group)

# get the final candidates for each group, and slice between the first sampled token and EOT
tokens, sum_logprobs = decoder.finalize(tokens, sum_logprobs)
tokens = [
    [t[sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s]
    for s in tokens
]

# select the top-ranked sample in each group
sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)
selected = sequence_ranker.rank(tokens, sum_logprobs)
tokens = [t[i].tolist() for i, t in zip(selected, tokens)]
for t in tokens:
    all_tokens.extend(t)

# texts = [tokenizer.decode(t).strip() for t in tokens]
texts = tokenizer.decode(all_tokens).strip()
# result = decode(model, mel, options)

print(texts)
### 大野手 叶头上就是那副话的模本 这本书中写到 tiny 也受 葉頭上就是那副話的模本這本書中寫到
### 大野獸,叶头上就是那副话的模本,这本书中写到 base 大野獸,葉頭上就是納附化的魔本,這本書中寫到
### 大野兽》,液头上就是那幅画的《魔本》,这本书中写道 small
### medium 大野獸。《夜頭上就是那幅畫》的魔本,這本書中寫道
### 大野兽,叶头上就是那幅画的模本。这本书中写到的large 大野獸葉頭上就是那幅畫的模本這本書中寫道
