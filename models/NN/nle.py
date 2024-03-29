import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel
from transformers import MT5Tokenizer, GPT2LMHeadModel

import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from .neuro_transformer import NeuroTransformer
from .eeg_mae import eeg_encoder
"""
1. implememnt eeg_encoder as neuro_encoder for contrastive learning 
"""

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class NeuroEncoder(nn.Module):
    def __init__(self, channels: int, timepoints: int, width: int, layers: int,
            heads: int, out_dims: int) -> None:
        super().__init__()

        # audio_encoder = get_audio_encoder(audioenc_name)

        self.base = NeuroTransformer(
            channels, timepoints, width, layers, heads)

        self.projection = Projection(width, out_dims)

    def forward(self, x):
        neuro_features = self.base(x)
        neuro_embeddings = self.projection(neuro_features)
        return neuro_embeddings

class NeuroMAE(nn.Module):
    def __init__(self, channels: int, timepoints: int, embed_dim: int, depth: int,
            heads: int, out_dims: int) -> None:
        super().__init__()

        self.base = eeg_encoder(timepoints, 4, embed_dim, channels, depth, heads)


        self.projection = Projection(embed_dim, out_dims) ### may have errors

    def forward(self, x):
        neuro_features = self.base(x)
        neuro_embeddings = self.projection(neuro_features)
        return neuro_embeddings

class TextEncoder(nn.Module):
    def __init__(self, text_model: str, transformer_embed_dim: int, out_dims: int, lang='eng') -> None:
        super().__init__()
        if lang == 'eng':
            self.base = AutoModel.from_pretrained(text_model)
        elif lang == 'chi':
            self.base = GPT2LMHeadModel.from_pretrained(text_model)
        
        self.projection = Projection(transformer_embed_dim, out_dims)
        self.target_token_idx = 0

    def forward(self, x):
        text_features = self.base(**x)[0]
        text_features = text_features[:, self.target_token_idx, :]  # get CLS token output
        text_embeddings = self.projection(text_features)
        return text_embeddings

class AudioEncoder(nn.model):
    def __init__(self, audio_model: str, processor_model: str, transformer_embed_dim: int, out_dims: int) -> None:
        """
            Wav2Vec_2 zh as audioencoder
            audio_model default - "ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"
            process_model default - "ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"
        """
        super().__init__()
        
        self.processor = Wav2Vec2Processor.from_pretrained(processor_model)
        self.base = Wav2Vec2ForCTC.from_pretrained(audio_model)
        
        self.projection = Projection(transformer_embed_dim, out_dims)
        # self.target_token_idx = 0

    def speech_file_to_array_fn(batch):
        resampler = torchaudio.transforms.Resample(48_000, 16_000)
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        batch["speech"] = resampler(speech_array).squeeze().numpy()
        return batch

    def forward(self, x):
        inputs = self.processor(x, sampling_rate=16_000, return_tensors="pt", padding=True)

        audio_features = self.base(inputs.input_values, attention_mask=inputs.attention_mask)[0]
       
        # text_features = text_features[:, self.target_token_idx, :]  # get CLS token output
        audio_embeddings = self.projection(audio_features)
        return audio_embeddings

    def speech_recognition(self):
        test_dataset = load_dataset("common_voice", "zh-CN", split="test")
        test_dataset = test_dataset.map(self.speech_file_to_array_fn)

        inputs = self.processor(test_dataset[:2]["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.base(inputs.input_values, attention_mask=inputs.attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        print("Prediction:", self.processor.batch_decode(predicted_ids))
        print("Reference:", test_dataset[:2]["sentence"])

class NLE(nn.Module):
    def __init__(self,
                # neuro
                channels: int,
                timepoints: int, 
                embed_dim: int, 
                depth: int,
                heads: int,
                # audio
                audio_model: str,
                processor_model: str,
                # text_model: str,
                transformer_embed_dim: int,
                # common
                out_dims: int,
                temperature,
                ):
        super().__init__()

        
        self.neuro_encoder = NeuroMAE(
            channels, timepoints, embed_dim, depth, heads, out_dims
        )

        self.audio_encoder = AudioEncoder(
            audio_model, processor_model, transformer_embed_dim,out_dims
        )
        self.temperature = temperature
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, audio, neuro):
        audio_embeddings = self.audio_encoder(audio)
        neuro_embeddings = self.neuro_encoder(neuro)

        
        # Calculating the Loss
        logits = (audio_embeddings @ neuro_embeddings.T) / self.temperature
        neuro_similarity = neuro_embeddings @ neuro_embeddings.T
        audios_similarity = audio_embeddings @ audio_embeddings.T
        targets = F.softmax(
            (neuro_similarity + audios_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()