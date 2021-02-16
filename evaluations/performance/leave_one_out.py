import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut

from parameters import get_rf_parameters, beautify_feature_names, get_features

FEATURES_FILE = '../data/export-new.csv'

raw_data = pd.read_csv(FEATURES_FILE)

# Labels are the values we want to predict
labels = raw_data['actual']
features = raw_data[get_features()]
np_features = np.array(features)
np_labels = np.array(labels)

loo = LeaveOneOut()
accuracies = []

for train, test in loo.split(features):
    rf = RandomForestClassifier(**get_rf_parameters())
    rf.fit(np_features[train], np_labels[train])

    # Use the forest's predict method on the test data
    predictions = rf.predict(np_features[test])

    accuracy = accuracy_score(np_labels[test], predictions)

    accuracies.append(accuracy)

arr = np.array(accuracies)

print('Model Performance')
print('Accuracies')
print(arr)
print('Accuracy = {:0.2f}%.'.format(np.average(arr) * 100))
print('Variance = {:0.2f}%.'.format(np.var(arr) * 100))
print('Std = {:0.2f}%.'.format(np.std(arr) * 100))