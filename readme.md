# Measuring Ableism in BERT Context Embeddings

Author: Samantha Dobesh

Date: Feb 10th. 2022

Based on: [Unpacking the Interdependent Systems of Discrimination: Ableist Bias in NLP Systems through an Intersectional Lens](https://arxiv.org/pdf/2110.00521.pdf)

[GitHub](https://github.com/saadhassan96/ableist-bias)

***NOTE:*** Repo does not contain our reddit dataset as it is unpublished.

## Outline

### Learning objectives

- Students will measure ableist biases in BERT context embeddings.
- Students will measure how intersectionality of disability, race, and gender effect the biases in embeddings.
- Students will fine tune the model to help address these biases and learn the limitations of techniques like counterfactual data substitution (CDS).
- Students will consider the impact on intersectional groups and how to currate and augment datasets to reduce possible harm.

### Activities

- Supplementary reading on ableism, intersectionality, and transformer embeddings.
- Measure ableist biases in a pre-trained BERT model.
- Fine tune the model to reduce ableist biases and measure this reduction.
- Reflection questions.

### Outcomes

- Students will have a basic level of knowledge on the many ways that BERT internalizes different biases.
- Students will learn to consider the external sources of these biases.
- Students will learn how to fine tune BERT.

## Activities

### Supplementary reading
#### Ableism & intersectionality
#### Biases in context embeddings

### Measuring bias
#### Measurment techniques
##### Cosine Method
##### Masked Language Modeling Method
```python

```
## Assignment
### Interactive Portion
#### Dependencies
- [Python 3+](https://www.python.org/downloads/)
- [Pytorch](https://pytorch.org/)
- [SKLearn](https://scikit-learn.org/stable/)
```
pip install torch sklearn`
```

In a Python terminal, do the following...

#### Make an Embedding
[Embeddings](https://medium.com/analytics-vidhya/contextual-word-embeddings-part1-20d84787c65).

Get the required imports. These include the model and tokenizer architecture.
```python
>>> from transformer import BertModel, BertTokenizer
```

Initialize the tokenizer and model and set them to evaluation mode. 
This saves us a little time by not calculating back propogation paths.
```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained(
    'bert-base-uncased',
    output_hidden_states=True
)
model.eval()
```

To use these we need to...
1. Tag the sentence with a start and seperator token.
  - `[CLS]` and `[SEP]`
  - One `[CLS]` at the beginning of the input.
  - We always need at least one `[SEP]` after the first sentence, but not after the following ones.
```python
>>> sentence = 'Embed this sentence!'
>>> tagged_sentence = '[CLS] ' + tagged_sentence + ' [SEP]'
```
2. Use the tokenizer on the tagged sentence to create a token vector.
```python
>>> tokenized = tokenizer.tokenize(tagged_sentence)
```
3. Create an indexed set of these tokens.
```python
>>> indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized)
```
4. Create segment IDs.
For single sentences, segments are easy, all ones!
```python
>>> segment_ids = [1] * len(tokenized)
```
5. Convert indexed tokens and segment IDs into tensors.
```python
>>> tokens = torch.tensor([indexed_tokens])
>>> segments = torch.tensor([segment_ids])
```
6. Feed tokens and segments into the model to get a prediction.
```python
>>> with torch.no_grad():
>>>     outputs = model(tokens, segments)
>>>     hidden_states = outputs[2]
```
7. Extract the hidden states.
```python
>>> token_vectors = hidden_states[-2][0]
```
8. The mean of the hidden_states token vectors is our sentence embedding.
```python
>>> sentence_embedding = torch.mean(token_vectors, dim=0)
```


We can do all of these things in an interactive Python session, but it may be helpful for you to create a script, perhaps called `embedding.py`.
This way you can write this is a whole function and just import it into the live environment with `from embedding import *`

```python
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
```

#### Cosine Difference
We will use the [cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) function in the sklearn library.
You should also define this in your `embedding.py` file.
```python
from sklearn.metrics.pairwise import cosine_similarity
def cosine(a: torch.Tensor, b: torch.Tensor) -> list:
    '''cosine similarity of two tensors'''
    return cosine_similarity(
        a.reshape(1, -1),
        b.reshape(1, -1)
    ).tolist()
```
Now we can take the embeddings of two sentences and compare their distance.
```python
def compare_sentences(a: str, b: str) -> float:
    '''streamline comparing two sentences embeddings.'''
    return cosine(embed(a), embed(b))[0][0]
```
#### BERT MLM

```python
# import a bert pipeline with the fill mask task
>>> from transformer import pipeline 
>>> bert = pipeline('fill-mask', model='bert-base-uncased')
```

We can now supply masked sentences to BERT, which will return the context embedding showing
the relative liklihood of different words occupying the mask.

```python
>>> prediction = bert('I went to the [MASK].')
```

The structure of the prediction is a list of dictionaries. If more than a single
mask is supplied, they will be masked one by one, causing this to return a list of 
lists of dictionaries. The dictionaries are as follows...

```
{
  'score': float,    # the probability of the guess (how sure the model is)
  'token': int,      # the guessed words token index
  'token_str': str,  # the guessed word
  'sequence': str    # the guessed word inserted into the original sentence
}
```

##### Dataset
We will now load the dataset used to evaluate biases into a dataframe and make some measurements.
```python
>>> import pandas as pd
>>> df = pd.read_csv('./path', delimiter='\t')
```

#### Script

### Fine tuning
#### Process overview
1. Import sentence data
2. Tokenize the data
3. Create mask array
4. Create a PyTorch dataset and dataloader class
5. for epoch in range(epochs): train!
6. save a checkpoint
7. load checkpoint and measure BEC-Pro again
#### Data preparation
#### Training
#### Evaluation

### Reflection
**q1:**
**q2:**
**q3:**
