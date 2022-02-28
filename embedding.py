#!/usr/bin/env python3
'''
@author : samantha dobesh
@desc : get context embeddings from bert
'''

import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

from dataset import *


# load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained(
    'bert-base-uncased',
    output_hidden_states=True
)
model.eval()


def embed(sentence: str) -> torch.Tensor:
    '''embed a single sentence and return the hidden states of the model'''
    tagged = '[CLS] ' + sentence + ' [SEP]'
    tokenized = tokenizer.tokenize(tagged)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized)
    segment_ids = [1] * len(tokenized)
    tokens = torch.tensor([indexed_tokens])
    segments = torch.tensor([segment_ids])
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

