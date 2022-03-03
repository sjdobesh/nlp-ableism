#!/usr/bin/env python3
'''
@author : samantha dobesh
@desc : measure ableist bias in BERT context embeddings
'''

import csv
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from embedding import *

# debug prints
DBUG = True
# csv file data
CSV_FILES = ['A', 'B', 'C', 'D', 'E']
CSV_PATH = './csv/'
# reddit data
REDDIT_DATASET_PATH = './reddit_dataset'
# load unmasking pipeline
unmask = pipeline('fill-mask', model='bert-base-uncased', device=0)
sentiment = pipeline('sentiment-analysis', device=0)


def debug(string: str, obj=None) -> None:
    '''debug print that can be disabled w macro'''
    if DBUG:
        print("**** [DEBUG] ****")
        print(string)
        if obj is not None:
            pprint(obj)


def dataset_help() -> None:
    '''print helpful info'''
    print("******** A11y Dataset ********")
    print('Useful functions:')
    print('- load_synthetic_dataset() -> list:\n    \
          load synthetic disability dataset.')
    print('- load_reddit_dataset() -> list:\n    \
          load reddit dataset.')
    print('- load_csv_dataset() -> list:\n    \
          load csv of masks you generate.')
    print('- save_dataset(data: list, name: str) -> None:\n    \
          save a list as a csv.')
    print('- mask(data: list) -> list:\n    \
          mask and save to a masks.csv file.')


def load_synthetic_dataset() -> List[pd.DataFrame]:
    '''
    load csvs from synthetic dataset into a list
    of dataframes seperated by file
    '''
    data = []
    for csv_file in CSV_FILES:
        data.append(pd.read_csv(CSV_PATH+csv_file+'.csv'))
    return data


def load_reddit_dataset() -> pd.DataFrame:
    '''load all csvs from reddit data set into a single dataframe'''
    # glob data frames
    dataframes = []
    for fp in Path(REDDIT_DATASET_PATH).rglob('*.csv'):
        with open(fp) as fd:
            dataframes.append(pd.read_csv(fd))
    # append dataframes into a single frame
    df = dataframes[0]
    for i in range(1, len(dataframes)):
        df.append(dataframes[i])
    return df


def save_dataset(data: List[pd.DataFrame], name: str) -> None:
    '''
    save a generated dataset as a csv

    data argument structure
    {
        masked sentence,
        filled sentence,
        guess word,
        score
    }
    '''
    with open(CSV_PATH + name + '.csv', 'w') as csv_file:
        writer = csv.DictWriter(csv_file, data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def load_csv_dataset(name: str) -> List[pd.DataFrame]:
    '''load a csv data set into a single dataframe'''
    # glob data frames
    dataframes = []
    with open(CSV_PATH + name + '.csv') as f:
        dataframes.append(pd.read_csv(f))
    return dataframes


def mask(data: List[pd.DataFrame]) -> List[dict]:
    '''get all masked sentences'''
    pred = []
    for datum in data:
        for disability, sentence in tqdm(zip(
            datum['Disability'],
            datum['Sentence']
        )):
            guess = unmask(sentence+'.')[0]
            guess_dict: dict = {
                'masked':     sentence,
                'unmasked':   guess['sequence'],
                'word':       guess['token_str'],
                'disability': disability,
                'score':      guess['score']
            }
            pred.append(guess_dict)
    return pred


def get_sentiment(data: List[pd.DataFrame]) -> List[tuple]:
    '''get all masked sentences w sentiment'''
    pred = []
    disabilities = []
    for datum in data:
        for disability in datum['Disability']:
            disabilities.append(disability)
        for i, sentence in enumerate(datum['Sentence']):
            guess = unmask(sentence+'.')[0]['sequence']
            score = sentiment(guess)
            pred.append((sentence, str(disabilities[i]), guess, score))

    return pred


def generate_data() -> List[dict]:
    '''
    generate mask data set and save it.
    also returns the masks
    '''
    data = load_synthetic_dataset()
    masks = mask(data)
    save_dataset(masks, 'masks')
    return masks


def connector_set(
    connector_word: str,
    data: List[pd.DataFrame]
) -> List[pd.DataFrame]:
    '''get the subset of the synthetic data for a given connector word'''
    # remember each thing in the data list is one of the csvs
    word = connector_word  # u gotta make this a local variable for query
    querried: List[pd.DataFrame] = []
    for datum in data:
        querried.append(datum.query('connector == @word'))
    return querried


def analyze(
    data: List[pd.DataFrame]
) -> List[Dict[str, str]]:
    '''run cosine similarity of all others against the first'''
    # basically data[0] is always one element with no descriptors
    # use this as the baseline
    result = []
    sentence = str(data[0]['Sentence'].iloc[0] + '.')
    pred = unmask(sentence)[0]
    baseline = pred['sequence']
    result.append((baseline))
    baseline = '[CLS] ' + baseline + ' [SEP]'
    for i in range(1, len(data)):  # 1 less
        for j in range(len(data[i])):
            compare = str(data[i]['Sentence'].iloc[j])
            disability = str(data[i]['Disability'].iloc[j])
            race = str(data[i]['Race'].iloc[j])
            gender = str(data[i]['Gender'].iloc[j])
            pred = unmask(compare + '.')
            compare = pred[0]['sequence']
            # tag sentences
            compare = '[CLS] ' + compare + ' [SEP]'
            result.append({
                'cosine_similarity': compare_sentences(baseline, compare),
                'disability': disability,
                'race': race,
                'gender': gender,
                'prediction': str(pred[0]['token_str'])
            })
    return result


def process() -> List[List[pd.DataFrame]]:
    '''process all connector sets'''
    results = []
    data = load_synthetic_dataset()
    for word in tqdm(data[0]['connector']):
        word_set = connector_set(word, data)
        word_results = analyze(word_set)
        results.append(word_results)
    for csv_file, result in zip(CSV_FILES[1:], results):
        with open(CSV_PATH + csv_file + '_processed.csv', 'w') as f:
            writer: csv.DictWriter = csv.DictWriter(f, result[3].keys())
            writer.writeheader()
            writer.writerows(result)
    return results


def main() -> None:
    '''u know, main?'''
    # do nothing for rn
    # we are using this file to load functions into an environment


if __name__ == "__main__":
    main()
