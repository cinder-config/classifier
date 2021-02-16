import csv
import pickle
import time
import numpy as np
import requests

GITHUB_API_TOKEN = "changeme"  # GitHub API TOKEN (PRIVATE!!!)
TARGET = 'data/replica/replica_dataset.csv'
EXPORT = 'results/ciclassifier.csv'

MODEL_PATH = '../../models/classifier_configuration.sav'

NOT_RESOLVABLE = 'data/not_resolvable.csv'
RESOLVABLE = 'data/resolvable.csv'
FEATURES_CSV = 'data/features.csv'

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


def block_until_github_limit_resetted():
    print('Checking GitHub Quota..')
    resp = requests.get("https://api.github.com/rate_limit",
                        headers={'Authorization': "token " + GITHUB_API_TOKEN})
    data = resp.json()
    ts = int(round(time.time(), 0))
    reset = data.get('resources').get('core').get('reset')
    remaining = data.get('resources').get('core').get('remaining')
    if remaining > 0:
        print('We have some quota remaining, yay!')
        return
    if ts > reset:
        print('Yay, time is up, new quota!')
        return
    else:
        print('Sleeping for 10 Seconds...')
        time.sleep(10)
        block_until_github_limit_resetted()


csv_headers = [
    'name',
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
]
model = pickle.load(open(MODEL_PATH, 'rb'))
with open(FEATURES_CSV, 'a', newline='') as csv_features:
    features_writer = csv.DictWriter(csv_features, fieldnames=csv_headers)
    features_writer.writeheader()
    with open(RESOLVABLE, 'a', newline='') as csv_resolvable:
        writer_resolvable = csv.writer(csv_resolvable, quoting=csv.QUOTE_ALL)
        with open(NOT_RESOLVABLE, 'a', newline='') as csv_not_resolvable:
            writer_not_resolvable = csv.writer(csv_not_resolvable, quoting=csv.QUOTE_ALL)
            with open(TARGET) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=';')
                first = True
                for row in csv_reader:
                    if first:
                        first = False
                        continue
                        row.append("status")
                        writer_resolvable.writerow(row)
                        writer_not_resolvable.writerow(row)

                    block_until_github_limit_resetted()
                    name = row[3]
                    print(name)
                    github_response = requests.get("https://api.github.com/repos/" + name,
                                                   headers={'Authorization': "token " + GITHUB_API_TOKEN})

                    if github_response.status_code != 200:
                        row.append("no github")
                        writer_not_resolvable.writerow(row)
                        continue

                    data = github_response.json()
                    travis_file_resp = requests.get(
                        "https://raw.githubusercontent.com/" + data.get('full_name') + "/" + data.get(
                            'default_branch') + "/.travis.yml")

                    if travis_file_resp.status_code != 200:
                        row.append("no travis")
                        writer_not_resolvable.writerow(row)
                        continue
                    else:
                        config = travis_file_resp.text
                        feature_response = requests.post("http://localhost:8080/extract", json={"target": config})
                        feature_data = feature_response.json()
                        if feature_data.get('status') != "success":
                            row.append("no features")
                            writer_not_resolvable.writerow(row)
                            continue

                        features = feature_data.get('features')

                        csv_write_dict = features
                        csv_write_dict['name'] = name
                        features_writer.writerow(csv_write_dict)

                        sanitized = []
                        for feature in FEATURES:
                            if feature in features:
                                sanitized.append(features[feature])

                        np_sanitized = [np.array(sanitized)]
                        predictions = model.predict(np_sanitized)
                        row.append(predictions[0])
                        writer_resolvable.writerow(row)
