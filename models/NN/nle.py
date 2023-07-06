import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel
from transformers import MT5Tokenizer, GPT2LMHeadModel
from .neuro_transformer import NeuroTransformer

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

class NLE(nn.Module):
    def __init__(self,
                # audio
                channels: int,
                timepoints: int, 
                width: int, 
                layers: int,
                heads: int,
                # text
                text_model: str,
                transformer_embed_dim: int,
                # common
                out_dims: int,
                temperature,
                ):
        super().__init__()

        
        self.neuro_encoder = NeuroEncoder(
            channels, timepoints, width, layers, heads, out_dims
        )

        self.caption_encoder = TextEncoder(
            text_model, transformer_embed_dim,out_dims
        )
        self.temperature = temperature
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, audio, text):
        neuro_embeddings = self.audio_encoder(audio)
        text_embeddings = self.caption_encoder(text)

        
        # Calculating the Loss
        logits = (text_embeddings @ neuro_embeddings.T) / self.temperature
        neuro_similarity = neuro_embeddings @ neuro_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (neuro_similarity + texts_similarity) / 2 * self.temperature, dim=-1
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