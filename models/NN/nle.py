import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
# from transformers import AutoModel
# from transformers import MT5Tokenizer, GPT2LMHeadModel

import torchaudio
# from datasets import load_dataset
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperModel
from models.whisper.__init__ import load_model
from .neuro_transformer import NeuroTransformer
from .eeg_mae import MAEforEEG, eeg_encoder
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

        # self.base = MAEforEEG(timepoints, 4, embed_dim, channels, depth, heads)
        #  embed_dim=1024, in_chans=128, depth=24, num_heads=16
        # encoder forward_encoder is different from MAE forward_encoder
        self.base = eeg_encoder(timepoints, patch_size=4, embed_dim=embed_dim, in_chans=channels, depth=depth, num_heads=heads)
        ### load MAE !!!!!

        self.projection = Projection(embed_dim, out_dims) ### may have errors

    def forward(self, x):
        neuro_features = self.base(x)
        print(neuro_features.shape)
        neuro_embeddings = self.projection(neuro_features)
        print(neuro_embeddings.shape)
        return neuro_embeddings

# class TextEncoder(nn.Module):
#     def __init__(self, text_model: str, transformer_embed_dim: int, out_dims: int, lang='eng') -> None:
#         super().__init__()
#         if lang == 'eng':
#             self.base = AutoModel.from_pretrained(text_model)
#         elif lang == 'chi':
#             self.base = GPT2LMHeadModel.from_pretrained(text_model)
        
#         self.projection = Projection(transformer_embed_dim, out_dims)
#         self.target_token_idx = 0

#     def forward(self, x):
#         text_features = self.base(**x)[0]
#         text_features = text_features[:, self.target_token_idx, :]  # get CLS token output
#         text_embeddings = self.projection(text_features)
#         return text_embeddings

class AudioEncoder(nn.Module):
    def __init__(self, model_name:str, audio_model: str, processor_model: str, transformer_embed_dim: int, out_dims: int, trainable: bool) -> None:
        """
            Wav2Vec_2 zh as audioencoder
            audio_model default - "ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"/ "openai/whisper-base.en"
            process_model default - "ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"/ "openai/whisper-base.en"

            # https://huggingface.co/models?search=openai/whisper
            large
        """
        super().__init__()
        self.model_name = model_name
        # if model_name == 'wav2vec':
        #     self.processor = Wav2Vec2Processor.from_pretrained(processor_model)
        #     self.base = Wav2Vec2ForCTC.from_pretrained(audio_model)
        if model_name == 'whisper':
            self.base = load_model(audio_model)
            # self.processor = WhisperProcessor.from_pretrained(processor_model)
            # self.base = WhisperModel.from_pretrained(audio_model)
        for p in self.base.parameters():
            p.requires_grad = trainable
        self.projection = Projection(transformer_embed_dim, out_dims)
        # self.target_token_idx = 0

    def speech_file_to_array_fn(batch):
        resampler = torchaudio.transforms.Resample(48000, 16000)
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        batch["speech"] = resampler(speech_array).squeeze().numpy()
        return batch

    def forward(self, x):

        # if self.model_name == 'wav2vec':
            # inputs = self.processor(x, sampling_rate=16000, return_tensors="pt", padding=True)
            ### encoder_last_hidden_state or last_hidden_state
            # audio_features = self.base(inputs.input_values, attention_mask=inputs.attention_mask).last_hidden_state
        
        if self.model_name == 'whisper':
            audio_features = self.base.embed_audio(x) ### audio encoder features
        
        audio_embeddings = self.projection(audio_features)
        print(audio_embeddings.shape)
        return audio_embeddings

    # def speech_recognition(self):
    #     # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    #     test_dataset = load_dataset("common_voice", "zh-CN", split="test")
    #     test_dataset = test_dataset.map(self.speech_file_to_array_fn)

    #     inputs = self.processor(test_dataset[:2]["speech"], sampling_rate=16000, return_tensors="pt", padding=True)

    #     with torch.no_grad():
    #         logits = self.base(inputs.input_values, attention_mask=inputs.attention_mask).logits

    #     predicted_ids = torch.argmax(logits, dim=-1)
    #     print("Prediction:", self.processor.batch_decode(predicted_ids))
    #     print("Reference:", test_dataset[:2]["sentence"])

class NLE(nn.Module):
    def __init__(self,
                # neuro
                channels: int,
                timepoints: int, 
                embed_dim: int, 
                depth: int,
                heads: int,
                # audio
                audioenc_name: str,
                audio_model: str,
                processor_model: str,
                # text_model: str,
                transformer_embed_dim: int,
                # common
                out_dims: int,
                trainable: bool,
                temperature,
                ):
        super().__init__()

        
        self.neuro_encoder = NeuroMAE(
            channels, timepoints, embed_dim, depth, heads, out_dims
        )

        self.audio_encoder = AudioEncoder(
            audioenc_name, audio_model, processor_model, transformer_embed_dim,out_dims, trainable
        )
        self.temperature = temperature
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, audio, neuro):
        audio_embeddings = self.audio_encoder(audio)
        neuro_embeddings = self.neuro_encoder(neuro)

        # audio torch.Size([8, 1500, 256])
        ## eeg torch.Size([8, 128, 256]) 
        # Calculating the Loss
        logits = (audio_embeddings @ neuro_embeddings.T) / self.temperature
        print(logits.shape)
        neuro_similarity = neuro_embeddings @ neuro_embeddings.T
        audios_similarity = audio_embeddings @ audio_embeddings.T
        targets = F.softmax(
            (neuro_similarity + audios_similarity) / 2 * self.temperature, dim=-1
        )
        audio_loss = cross_entropy(logits, targets, reduction='none')
        neuro_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (audio_loss + neuro_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
