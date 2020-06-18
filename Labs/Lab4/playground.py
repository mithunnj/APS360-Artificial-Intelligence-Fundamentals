import csv
import numpy as np
import random
import torch
import torch.utils.data
import pandas as pd

header = ['age', 'work', 'fnlwgt', 'edu', 'yredu', 'marriage', 'occupation',
 'relationship', 'race', 'sex', 'capgain', 'caploss', 'workhr', 'country']
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    names=header,
    index_col=False)

metric_headers = ["age", "yredu", "capgain", "caploss", "workhr"]

print('Original dataframe: {}\n'.format(df))

for item in metric_headers: 
    
    # Normalize the data
    total = sum(df[item])
    df[item] = df[item] / total

contcols = ["age", "yredu", "capgain", "caploss", "workhr"]
catcols = ["work", "marriage", "occupation", "edu", "relationship", "sex"]
features = contcols + catcols
df = df[features]

missing = pd.concat([df[c] == " ?" for c in catcols], axis=1).any(axis=1)
df_with_missing = df[missing]
df_not_missing = df[~missing]

data = pd.get_dummies(df_not_missing)
print(data.shape)