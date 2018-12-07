import numpy as np
import pandas as pd
from apyori import apriori

dataset = pd.read_csv('Data/GroceryStoreDataSet.csv', header=None, skiprows=1)
print(dataset)

transactions = []
for i in range (0, dataset.shape[0]):
    transactions.append([str(dataset.values[i,j]) for j in range (0, dataset.shape[1])])
transactions
 
rules = apriori(transactions, min_support=0.1, min_confidence=0.9,  min_lift=3, min_length=2)

results = list(rules)
for r in results:
    print(str(r) +'\n')
