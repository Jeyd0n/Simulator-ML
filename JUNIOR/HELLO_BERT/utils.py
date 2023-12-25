import torch
import numpy as np
from dataclasses import dataclass
from transformers import DistilBertTokenizer, PreTrainedTokenizer
from typing import List, Generator, Tuple


MODEL_NAME = 'distilbert-base-uncased'


@dataclass
class DataLoader:
    """
    Custom DataLoader for sentiment analysis task
    """
    path: str
    tokenizer: PreTrainedTokenizer.from_pretrained(MODEL_NAME)
    batch_size: int = 512
    max_length: int = 128
    padding: str = None

    def __iter__(self) -> Generator[List[List[int]], None, None]:
        """Iterate over batches"""
        for i in range(len(self)):
            yield self.batch_tokenized(i)

    def __len__(self):
        """Number of batches"""
        with open(self.path, 'r') as f:
            next(f) 

            len_ = len(f.readlines())

        return int(np.ceil(len_ / self.batch_size))

    def tokenize(self, batch: List[str]) -> List[List[int]]:
        """Tokenize list of texts"""
        if self.padding is None:
            tokenized_batch = self.tokenizer(
                text=batch,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True
            )
        elif self.padding == 'max_length':
            tokenized_batch = self.tokenizer(
                text=batch,
                add_special_tokens=True,
                padding='max_length',
                max_length=self.max_length,
                truncation=True
            )
        elif self.padding == 'batch':
            tokenized_batch = self.tokenizer(
                text=batch,
                add_special_tokens=True,
                padding='longest',
                max_length=self.max_length,
                truncation=True
            )

        return tokenized_batch['input_ids']

    def batch_loaded(self, i: int) -> Tuple[List[str], List[int]]:
        """Return loaded i-th batch of data (text, label)"""
        batch_start = self.batch_size * i
        batch_end = batch_start + self.batch_size
        texts, labels = [], []

        f = open(self.path, 'r')
        next(f)

        for idx, line in enumerate(f):
            line = line.rstrip().split(",", 4)

            if batch_start <= idx < batch_end:
                texts.append(line[-1])

                if line[-2] == 'positive':
                    labels.append(1)
                elif line[-2] == 'negative':
                    labels.append(-1)
                else:
                    labels.append(0)

        return (texts, labels)
        

    def batch_tokenized(self, i: int) -> Tuple[List[List[int]], List[int]]:
        """Return tokenized i-th batch of data"""
        texts, labels = self.batch_loaded(i)
        tokens = self.tokenize(texts)

        return tokens, labels


def attention_mask(padded: List[List[int]]) -> List[List[int]]:
    masks = []

    for tokens in padded:
        mask = []

        for value in tokens:
            if value != 0:
                mask.append(1)
            else:
                mask.append(0)

        masks.append(mask)

    return masks


def review_embedding(tokens: List[List[int]], model) -> List[List[float]]:
    """Return embedding for batch of tokenized texts"""
    mask = attention_mask(tokens)
    mask = mask + [0] * (128 - len(mask[0]))
        
    tokens = torch.tensor(tokens) 
    mask = torch.tensor(mask)
        
    with torch.no_grad():
        last_hidden_states = model(tokens, attention_mask=mask)

    features = last_hidden_states[0][:,0,:].tolist()

    return features
