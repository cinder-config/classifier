import csv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, LeaveOneOut, RepeatedKFold
from sklearn.metrics import accuracy_score, precision_recall_curve, confusion_matrix, classification_report, \
    precision_recall_fscore_support
from sklearn.model_selection import cross_val_score

import helper
from parameters import get_rf_parameters, get_features
import itertools

AVAILABLE_FEATURES = [
    'ch.uzh.ciclassifier.features.configuration.EnvSize',
    'ch.uzh.ciclassifier.features.configuration.LintScore',
    'ch.uzh.ciclassifier.features.configuration.NumberOfComments',
    'ch.uzh.ciclassifier.features.configuration.NumberOfConfigurationChars',
    'ch.uzh.ciclassifier.features.configuration.NumberOfConfigurationLines',
    'ch.uzh.ciclassifier.features.configuration.NumberOfJobs',
    'ch.uzh.ciclassifier.features.configuration.NumberOfNotifications',
    'ch.uzh.ciclassifier.features.configuration.NumberOfStages',
    'ch.uzh.ciclassifier.features.configuration.ParseScore',
    'ch.uzh.ciclassifier.features.configuration.TemplateSimilarity',
    'ch.uzh.ciclassifier.features.configuration.UniqueInstructions',
    'ch.uzh.ciclassifier.features.configuration.UseBranches',
    'ch.uzh.ciclassifier.features.configuration.UseCache',
    'ch.uzh.ciclassifier.features.configuration.UseDeploy',
    'ch.uzh.ciclassifier.features.github.OwnerType',
    'ch.uzh.ciclassifier.features.github.PrimaryLanguage',
    'ch.uzh.ciclassifier.features.repository.CommitsUntilConfigAdded',
    'ch.uzh.ciclassifier.features.repository.ConfigChangeFrequency',
    'ch.uzh.ciclassifier.features.repository.DaysUntilConfigAdded',
    'ch.uzh.ciclassifier.features.repository.NumberOfConfigurationFileChanges',
    'ch.uzh.ciclassifier.features.repository.NumberOfContributors',
    'ch.uzh.ciclassifier.features.repository.NumberOfContributorsOnConfigurationFile',
    'ch.uzh.ciclassifier.features.repository.ProjectName',
    'ch.uzh.ciclassifier.features.travisci.BuildTimeAverage',
    'ch.uzh.ciclassifier.features.travisci.BuildSuccessRatio',
    'ch.uzh.ciclassifier.features.travisci.BuildTimeLatestAverage',
    'ch.uzh.ciclassifier.features.travisci.ManualInteractionRatio',
    'ch.uzh.ciclassifier.features.travisci.PullRequestRatio',
    'ch.uzh.ciclassifier.features.travisci.TimeToFixAverage',
    'ch.uzh.ciclassifier.features.travisci.TimeToFixLatestAverage',
]

FEATURES_FILE = 'data/truth.csv'
LANGAUGES = ['Ruby', 'JavaScript', 'Python', 'Java', 'C++', 'PHP']

results = []

NUMBER_OF_RUNS = 10

for language in LANGAUGES:
    raw_data = pd.read_csv(FEATURES_FILE)
    subset = raw_data.loc[raw_data['language'] == language]
    features = subset[get_features()]
    labels = np.array(subset['actual'])
    features = np.array(features)

    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
    rf = RandomForestClassifier(**get_rf_parameters())
    scores = cross_val_score(rf, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)

    accuracies = []
    precisions = []
    recalls = []
    for run in range(NUMBER_OF_RUNS):
        raw_data = pd.read_csv(FEATURES_FILE)
        subset_train = raw_data.loc[raw_data['language'] != language]
        subset_test = raw_data.loc[raw_data['language'] == language]

        features_train = subset_train[get_features()]
        labels_train = np.array(subset_train['actual'])

        attributes = get_rf_parameters()
        #attributes['random_state'] = None
        rf = RandomForestClassifier(**attributes)
        rf.fit(features_train, labels_train)

        features_test = subset_test[get_features()]
        labels_test = np.array(subset_test['actual'])

        predictions = rf.predict(features_test)
        p, r, f1, s = precision_recall_fscore_support(labels_test, predictions)

        accuracies.append(accuracy_score(labels_test, predictions))
        precisions.append(np.mean(p))
        recalls.append(np.mean(r))

    results.append({
        'language': language,
        'accuracy_self': scores.mean(),
        'std_self': scores.std(),
        'samples': len(subset.index),
        'accuracy_other': np.mean(accuracies),
        'std_other': np.std(accuracies),
        'samples_train_other': len(features_train.index),
        'samples_test_other': len(features_test.index),
        'precision_other': np.mean(precisions),
        'recall_other': np.mean(recalls)
    })

with open('results.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['language', 'accuracy_other', 'std_other', 'samples_train_other',
                                                 'samples_test_other', 'precision_other', 'recall_other',
                                                 'accuracy_self', 'std_self', 'samples'])
    writer.writeheader()
    for result in results:
        writer.writerow(result)
