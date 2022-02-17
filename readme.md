# Measuring Ableism in BERT Context Embeddings

Author: Samantha Dobesh

Date: Feb 10th. 2022

Based on: [Unpacking the Interdependent Systems of Discrimination: Ableist Bias in NLP Systems through an Intersectional Lens](https://arxiv.org/pdf/2110.00521.pdf)

[GitHub](https://github.com/saadhassan96/ableist-bias)

## Outline

### Learning objectives

- Students will measure ableist biases in BERT context embeddings.
- Students will measure how intersectionality of disability, race, and gender effect the biases in embeddings.
- Students will fine tune the model to help address these biases and learn the limitations of techniques like counterfactual data substitution (CDS).
- Students will consider the impact on intersectional groups and how to currate datasets to reduce possible harm.

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

#### Interactive

##### BERT MLM
In a python terminal, do the following...

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
#### Data preparation
#### Training
#### Evaluation

### Reflection
**q1:**
**q2:**
**q3:**
