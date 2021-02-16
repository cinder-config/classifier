import csv
import pickle
import time
import numpy as np
import pandas as pd

TARGET = 'data/replica/replica_dataset.csv'
EXPORT = 'results/ciclassifier.csv'
MODEL_PATH = '../../models/classifier_configuration.sav'

RESOLVABLE_CSV = 'data/resolvable.csv'
FEATURES_CSV = 'data/features.csv'
RESULTS_CSV = 'data/results.csv'

FEATURES = [
    'ch.uzh.ciclassifier.features.configuration.EnvSize',
    # 'ch.uzh.ciclassifier.features.configuration.LintScore',
    'ch.uzh.ciclassifier.features.configuration.NumberOfComments',
    # 'ch.uzh.ciclassifier.features.configuration.NumberOfConfigurationChars',
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
    # 'ch.uzh.ciclassifier.features.github.OwnerType',
    # 'ch.uzh.ciclassifier.features.github.PrimaryLanguage',
    # 'ch.uzh.ciclassifier.features.repository.CommitsUntilConfigAdded',
    # 'ch.uzh.ciclassifier.features.repository.ConfigChangeFrequency',
    # 'ch.uzh.ciclassifier.features.repository.DaysUntilConfigAdded',
    # 'ch.uzh.ciclassifier.features.repository.NumberOfConfigurationFileChanges',
    # 'ch.uzh.ciclassifier.features.repository.NumberOfContributorsOnConfigurationFile',
    # 'ch.uzh.ciclassifier.features.repository.ProjectName',
    # 'ch.uzh.ciclassifier.features.travisci.BuildSuccessRatio',
    # 'ch.uzh.ciclassifier.features.travisci.BuildTimeAverage',
    # 'ch.uzh.ciclassifier.features.travisci.BuildTimeLatestAverage',
    # 'ch.uzh.ciclassifier.features.travisci.ManualInteractionRatio',
    # 'ch.uzh.ciclassifier.features.travisci.PullRequestRatio',
    # 'ch.uzh.ciclassifier.features.travisci.TimeToFixAverage',
    # 'ch.uzh.ciclassifier.features.travisci.TimeToFixLatestAverage',
]

model = pickle.load(open(MODEL_PATH, 'rb'))
features = pd.read_csv(FEATURES_CSV)
projects = pd.read_csv(RESOLVABLE_CSV)

with open(RESULTS_CSV, 'w', newline='') as results_csv:
    projects_writer = csv.DictWriter(results_csv, fieldnames=projects.columns)
    projects_writer.writeheader()
    for index, row in projects.iterrows():
        project_features = features.loc[features['name'] == row.slug]
        project_features = project_features[FEATURES]

        prediction = model.predict(project_features)

        project_dict = row.to_dict()
        project_dict['status'] = prediction[0]
        projects_writer.writerow(project_dict)
