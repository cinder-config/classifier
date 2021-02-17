import csv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, LeaveOneOut, RepeatedKFold
from sklearn.metrics import accuracy_score, precision_recall_curve, confusion_matrix, classification_report
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
TYPES = ['configuration','repository','travisci']
results = []
for i in range(len(TYPES)):
    for permutation in list(itertools.combinations(TYPES, i+1)):
        raw_data = pd.read_csv(FEATURES_FILE)
        features = raw_data[get_features()]

        for feature in AVAILABLE_FEATURES:
            featureType = helper.type_from_feature_name(feature)
            if featureType not in permutation and feature in features:
                features = features.drop(feature, axis=1)

        labels = np.array(raw_data['actual'])
        features = np.array(features)

        cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        rf = RandomForestClassifier(**get_rf_parameters())
        scores = cross_val_score(rf, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
        results.append({
            'permutation': ','.join(permutation),
            'accuracy': scores.mean(),
            'std' : scores.std()
        })

with open('results.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['permutation','accuracy','std'])
    writer.writeheader()
    for result in results:
        writer.writerow(result)