import torch
import numpy as np
# from transformers import AutoModelForCausalLM
from transformers import MT5Tokenizer, GPT2LMHeadModel, TextGenerationPipeline
from torch.nn.functional import softmax

class GPT():    
    """wrapper for https://huggingface.co/openai-gpt
    """
    def __init__(self, path, vocab, device = 'cpu'): 
        self.device = device
        # self.model = AutoModelForCausalLM.from_pretrained(path).eval().to(self.device)

        self.tokenizer = MT5Tokenizer.from_pretrained(path)
        self.model = GPT2LMHeadModel.from_pretrained(path).eval().to(device)
        self.start_id = np.array([259])
        self.vocab = vocab
        # self.word2id = {w : i for i, w in enumerate(self.vocab)}
        # self.UNK_ID = self.word2id['<unk>']

    def encode(self, words):
        """map from words to ids
        """
        return [self.word2id[x] if x in self.word2id else self.UNK_ID for x in words]
        
    def get_story_array(self, words, context_words):
        """get word ids for each phrase in a stimulus story
        """
        nctx = context_words + 1
        story_ids = np.array([self.tokenizer.encode(w)[1] for w in words])
        story_array = np.zeros([len(story_ids), nctx])
        print(story_ids[:context_words].shape)
        for i in range(len(story_array)):
            segment = np.concatenate((self.start_id,story_ids[i:i+context_words]))
            story_array[i, :len(segment)] = segment
        return torch.tensor(story_array).long()

    def get_context_array(self, contexts):
        """get word ids for each context
        """
        context_array = np.array([self.tokenizer.encode(words) for words in contexts])
        return torch.tensor(context_array).long()

    def get_hidden(self, ids, layer):
        """get hidden layer representations
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids = ids.to(self.device), 
                                 attention_mask = mask.to(self.device), output_hidden_states = True)
        return outputs.hidden_states[layer].detach().cpu().numpy()

    def get_probs(self, ids):
        """get next word probability distributions
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids = ids.to(self.device), attention_mask = mask.to(self.device))
        probs = softmax(outputs.logits, dim = 2).detach().cpu().numpy()
        return probs