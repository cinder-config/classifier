import csv
import pickle
import pandas as pd
from parameters import get_features

### Predict based on a given model and save to file

TARGET = 'projects_to_predict.csv'
EXPORT = 'to_save.csv'
MODEL_PATH = 'model_path.sav'

raw_data = pd.read_csv(TARGET)
to_predict = raw_data.drop('project', axis=1)
features = get_features()
to_predict = to_predict[features]

model = pickle.load(open(MODEL_PATH, 'rb'))

predictions = model.predict(to_predict)
projects = raw_data.to_dict('records')

with open(EXPORT, 'w', newline='') as csvfile:
    fieldnames = ['project']
    fieldnames.extend(features)
    fieldnames.append('score')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    index = 0
    for project in projects:
        prediction = predictions[index]
        project['score'] = prediction
        writer.writerow(project)
        index = index + 1
