import csv
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import prepare as prep
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

REAPER_RESULTS = 'reaper_results.csv'
CICLASSIFIER_RESULTS = 'cinder_bigdataset.csv'

combined = {}

# READ CICLASSIFIER
with open(CICLASSIFIER_RESULTS) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    first = True
    for row in csv_reader:
        if first:
            first = False
            continue
        name = row[0].replace('/', '_')
        combined[name] = {
            'name': name,
            'ciclassifier': int(row[13])
        }

# READ REAPER
with open(REAPER_RESULTS) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    first = True
    for row in csv_reader:
        if first:
            first = False
            continue
        name = row[1].replace('/', '_')
        if name in combined:
            combined[name]['reaper'] = row[2]
        else:
            print("Not found: " + name)

matrix = np.array([[0, 0], [0, 0]])

reaper_scores = []
ciclassifier_scores = []

with open('.reaper_combined.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['name', 'reaper', 'ciclassifier'])
    writer.writeheader()
    for itm in combined.items():
        name = itm[0]
        comb = itm[1]
        if 'reaper' in comb and 'ciclassifier' in comb:
            reaper_score = int(comb['reaper'])
            ciclassifier_score = int(comb['ciclassifier'])
            reaper_scores.append(reaper_score)
            ciclassifier_scores.append(ciclassifier_score)
            matrix[ciclassifier_score, reaper_score] = matrix[ciclassifier_score, reaper_score] + 1
            writer.writerow({
                'name': name,
                'reaper': reaper_score,
                'ciclassifier': ciclassifier_score
            })

    np_reaper = np.array(reaper_scores)
    np_ciclassifier = np.array(ciclassifier_scores)

print(np.corrcoef(np_reaper, np_ciclassifier))
print(matrix)

# plt.scatter(np_reaper, np_ciclassifier)
# plt.show()
