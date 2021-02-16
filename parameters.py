import numpy as np

AVAILABLE_FEATURES = [
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
    'ch.uzh.ciclassifier.features.repository.CommitsUntilConfigAdded',
    'ch.uzh.ciclassifier.features.repository.ConfigChangeFrequency',
    'ch.uzh.ciclassifier.features.repository.DaysUntilConfigAdded',
    'ch.uzh.ciclassifier.features.repository.NumberOfConfigurationFileChanges',
    'ch.uzh.ciclassifier.features.repository.NumberOfContributorsOnConfigurationFile',
    # 'ch.uzh.ciclassifier.features.repository.ProjectName',
    'ch.uzh.ciclassifier.features.travisci.BuildSuccessRatio',
    'ch.uzh.ciclassifier.features.travisci.BuildTimeAverage',
    # 'ch.uzh.ciclassifier.features.travisci.BuildTimeLatestAverage',
    'ch.uzh.ciclassifier.features.travisci.ManualInteractionRatio',
    'ch.uzh.ciclassifier.features.travisci.PullRequestRatio',
    'ch.uzh.ciclassifier.features.travisci.TimeToFixAverage',
    # 'ch.uzh.ciclassifier.features.travisci.TimeToFixLatestAverage',
]

HYPERPARAMETERS = {'bootstrap': False,
                   'max_depth': 30,
                   'max_features': 'log2',
                   'min_samples_leaf': 4,
                   'min_samples_split': 2,
                   'n_estimators': 32,
                   'random_state': 42}

ALTERNATIVE = {'bootstrap': True,
               'max_depth': 10,
               'max_features': 'sqrt',
               'min_samples_leaf': 4,
               'min_samples_split': 2,
               'n_estimators': 400}

HYPERPARAMETERS_CLASSIFIER = {'bootstrap': False,
                              'max_depth': None,
                              'max_features': 'auto',
                              'min_samples_leaf': 1,
                              'min_samples_split': 5,
                              'n_estimators': 1000,
                              'random_state': 43}


def get_threshold():
    return 0.5


def get_features():
    return AVAILABLE_FEATURES


def get_rf_parameters():
    return HYPERPARAMETERS


THRESHOLD = 1.5


def evaluate(el):
    return 2 if el > THRESHOLD else 1


def evaluateMany(predictions):
    evaluate_vectorized = np.vectorize(evaluate)
    return evaluate_vectorized(predictions)


BEAUTIFUL_FEATURE_NAMES = {
    'ch.uzh.ciclassifier.features.configuration.EnvSize': 'envs',
    'ch.uzh.ciclassifier.features.configuration.LintScore': 'lints',
    'ch.uzh.ciclassifier.features.configuration.NumberOfComments': 'comments',
    'ch.uzh.ciclassifier.features.configuration.NumberOfConfigurationChars': 'chars',
    'ch.uzh.ciclassifier.features.configuration.NumberOfConfigurationLines': 'lines',
    'ch.uzh.ciclassifier.features.configuration.NumberOfJobs': 'jobs',
    'ch.uzh.ciclassifier.features.configuration.NumberOfNotifications': 'notifications',
    'ch.uzh.ciclassifier.features.configuration.NumberOfStages': 'stages',
    'ch.uzh.ciclassifier.features.configuration.ParseScore': 'parse',
    'ch.uzh.ciclassifier.features.configuration.TemplateSimilarity': 'template_similarity',
    'ch.uzh.ciclassifier.features.configuration.UniqueInstructions': 'unique_instructions',
    'ch.uzh.ciclassifier.features.configuration.UseBranches': 'use_branches',
    'ch.uzh.ciclassifier.features.configuration.UseCache': 'use_cache',
    'ch.uzh.ciclassifier.features.configuration.UseDeploy': 'use_deploy',
    'ch.uzh.ciclassifier.features.github.OwnerType': 'owner_type',
    'ch.uzh.ciclassifier.features.github.PrimaryLanguage': 'primary_language',
    'ch.uzh.ciclassifier.features.repository.CommitsUntilConfigAdded': 'commits_until_config',
    'ch.uzh.ciclassifier.features.repository.ConfigChangeFrequency': 'config_change_frequency',
    'ch.uzh.ciclassifier.features.repository.DaysUntilConfigAdded': 'days_until_config_added',
    'ch.uzh.ciclassifier.features.repository.NumberOfConfigurationFileChanges': 'config_changes',
    'ch.uzh.ciclassifier.features.repository.NumberOfContributorsOnConfigurationFile': 'config_change_contributors',
    'ch.uzh.ciclassifier.features.repository.ProjectName': 'project_name',
    'ch.uzh.ciclassifier.features.travisci.BuildSuccessRatio': 'build_success_ratio',
    'ch.uzh.ciclassifier.features.travisci.BuildTimeAverage': 'build_time',
    'ch.uzh.ciclassifier.features.travisci.BuildTimeLatestAverage': 'build_time_latest',
    'ch.uzh.ciclassifier.features.travisci.ManualInteractionRatio': 'manual_interaction_ratio',
    'ch.uzh.ciclassifier.features.travisci.PullRequestRatio': 'pull_request_ratio',
    'ch.uzh.ciclassifier.features.travisci.TimeToFixAverage': 'time_to_fix',
    'ch.uzh.ciclassifier.features.travisci.TimeToFixLatestAverage': 'time_to_fix_latest'
}


def beautify_feature_names(labels):
    return [BEAUTIFUL_FEATURE_NAMES[l] for l in labels]


def beautify_feature_name(label):
    return BEAUTIFUL_FEATURE_NAMES[label]
