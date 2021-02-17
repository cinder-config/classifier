import csv

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import prepare as prep
from sklearn.model_selection import KFold, cross_val_score, RepeatedKFold
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score

from parameters import get_rf_parameters, get_features
import random

FEATURES_FILE = 'data/truth.csv'

raw_data = pd.read_csv(FEATURES_FILE)
labels = raw_data['actual']
features = raw_data[get_features()]
np_features = np.array(features)
np_labels = np.array(labels)

NUMBER_OF_RUNS = 10
SIZES = [30,50,70,90,110,130,150]

results = []

for size in SIZES:
    size_accuracies = []
    size_std = []

    for run in range(NUMBER_OF_RUNS):
        print(str(size) + "/" + str(run))
        subset = random.sample(range(0, len(np_labels)), size)
        subset_features = np_features[subset]
        subset_labels = np_labels[subset]

        rf = RandomForestClassifier(**get_rf_parameters())
        scores = cross_val_score(rf, subset_features, subset_labels, scoring='accuracy', cv=10, n_jobs=-1)

        size_accuracies.append(scores.mean())
        size_std.append(scores.std())

    results.append({
        'size': size,
        'accuracy': np.mean(size_accuracies),
        'std': np.mean(size_std)
    })

with open('results/results.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['size', 'accuracy', 'std'])
    writer.writeheader()
    for result in results:
        writer.writerow(result)
