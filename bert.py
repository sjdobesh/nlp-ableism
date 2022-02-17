#!/usr/bin/env python3
'''
@author : samantha dobesh
@desc : measure ableist bias in BERT context embeddings
'''


from pathlib import Path
import pandas as pd
from pprint import pprint
from transformers import pipeline


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


def debug(string, obj=None):
    '''debug print that can be disabled w macro'''
    if DBUG:
        print("**** [DEBUG] ****")
        print(string)
        if obj is not None:
            pprint(obj)


def load_synthetic_dataset():
    '''
    load csvs from synthetic dataset into a list
    of dataframes seperated by file
    '''
    data = []
    for csv in CSV_FILES:
        data.append(pd.read_csv(CSV_PATH+csv+'.csv'))
    return data


def load_reddit_dataset():
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


def run(data):
    '''get all masked sentences'''
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


def main():
    '''u know, main?'''
    # do nothing for rn


if __name__ == "__main__":
    main()
