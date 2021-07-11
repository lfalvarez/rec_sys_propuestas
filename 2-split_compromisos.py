import numpy as np
import pandas as pd
import json

def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

def to_json(df, filename):
    data = df.groupby('candidate')['proposal'].apply(list).to_dict()
    data = json.dumps(data)
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)




if __name__ == '__main__':
    df_proposals = pd.read_csv('propuestas.csv')
    df = pd.read_csv('compromisos.csv')
    df = df.loc[df['id'].isin(df_proposals.id)].reset_index()
    train, validate, test = train_validate_test_split(df, train_percent=.7, validate_percent=.15)
    to_json(train, "data/compromisos_train.json")
    to_json(validate, "data/compromisos_validate.json")
    to_json(test, "data/compromisos_test.json")
