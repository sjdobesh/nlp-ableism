#!/usr/bin/env python3
'''
@author : samantha dobesh
@desc : get context embeddings from bert
'''
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer, pipeline

# load tokenizer, model, and pipeline
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained(
    'bert-base-uncased',
    output_hidden_states=True
)
model.eval()
unmask = pipeline('fill-mask', model='bert-base-uncased')
sentiment = pipeline('sentiment-analysis')

class AbleistDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: pd.DataFrame) -> None:
        super(self)
        # convert data frame to simple list for speed
        self.length: int = len(dataset)
        self.data: list = []
        for i in range(len(dataset)):
            self.data.append(dataset['Sentence'].iloc[i])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> str:
        return self.data[i]


def embed(sentence: str) -> torch.Tensor:
    '''embed a single sentence and return the hidden states of the model'''
    tagged: str = '[CLS] ' + sentence + ' [SEP]'
    tokenized: list = tokenizer.tokenize(tagged)
    indexed_tokens: list = tokenizer.convert_tokens_to_ids(tokenized)
    segment_ids: list = [1] * len(tokenized)
    tokens: torch.Tensor = torch.tensor([indexed_tokens])
    segments: torch.Tensor = torch.tensor([segment_ids])
    with torch.no_grad():
        outputs = model(tokens, segments)
        hidden_states = outputs[2]
    token_vectors = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vectors, dim=0)
    return sentence_embedding


def cosine(a: torch.Tensor, b: torch.Tensor) -> list:
    '''cosine similarity of two tensors'''
    return cosine_similarity(
        a.reshape(1, -1),
        b.reshape(1, -1)
    ).tolist()


def compare_sentences(a: str, b: str) -> float:
    '''streamline comparing two sentences embeddings.'''
    return cosine(embed(a), embed(b))[0][0]


