import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from parameters import get_rf_parameters

### Trains a model based on the given features

MODEL_PATH = 'models/classifier_configuration.sav'

FEATURES_FILE = 'data/truth.csv'

raw_data = pd.read_csv(FEATURES_FILE)

used_features = [
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

# Labels are the values we want to predict
labels = np.array(raw_data['actual'])
features = raw_data[used_features]

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(**get_rf_parameters())

# Train the model on training data
rf.fit(features, labels)

pickle.dump(rf, open(MODEL_PATH, 'wb'))
