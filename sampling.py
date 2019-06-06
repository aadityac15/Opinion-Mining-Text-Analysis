# Dataset Sampling 

import numpy as np
import pandas as pd
import csv

if __name__ == "__main__":
    dataset = pd.read_csv('onemilliontweets.csv', header=None, encoding='latin-1', low_memory=False)
    dataset.columns = ['sentiment', 'id', 'date', 'query', 'user', 'tweets']
    #Dropping unnecessary columns
    dataset = dataset.drop(['id','date','query','user'],axis = 1)
    datasetk = dataset.head(20000).append(dataset.tail(20000))
    print(len(datasetk))
    datasetk.to_csv('DatasetF.csv',encoding='UTF-8')
