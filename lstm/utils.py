import numpy as np
import pandas as pd
import json
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences



def convert(x):
    """
    Coverting JSON to pandas dataframe

    """    
    ob = json.loads(x)
    for k, v in ob.items():
        if isinstance(v, list):
            ob[k] = ','.join(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                ob['%s_%s' % (k, kk)] = vv
            del ob[k]
    return ob


def filter_data(data):
    """
    Converting into pandas dataframe and filtering only text and ratings given by the users
    """

    df = pd.DataFrame([convert(line) for line in data])
    df.drop(columns=df.columns.difference(['text','stars']),inplace=True)
    df.loc[:, ("sentiment")] = 0

    #I have considered a rating above 3 as positive and less than or equal to 3 as negative.
    df.loc[:,'sentiment']=['pos' if (x>3) else 'neg' for x in df.loc[:, 'stars']]
    df.loc[:,'text'] = df.loc[:,'text'].apply(lambda x: x.lower())
    df.loc[:,'text'] = df.loc[:,'text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    for idx,row in df.iterrows():
        df.loc[:,'text']= [x for x in df.loc[:,'text']]
    return df


def min_max_normalize(tokens):
    min_val = min(tokens)
    max_val = max(tokens)
    normalized_tokens = [(token - min_val) / (max_val - min_val) for token in tokens]
    return normalized_tokens




def get_dataset():
    json_filename = 'review_mockup_500.json'
    with open(json_filename,'rb') as f:
        data = f.readlines()
    data = filter_data(data)
    tokenizer = Tokenizer(num_words = 2500, split=' ')
    tokenizer.fit_on_texts(data=data.loc[:,'text'].values)

    X = tokenizer.texts_to_sequences(data=data.loc[:,'text'].values)
    #X = [min_max_normalize(i) for i in X]
    X = pad_sequences(X)
    Y = pd.get_dummies(data['sentiment'],dtype=int).values 

    return X, Y
