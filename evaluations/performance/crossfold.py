import pandas as pd
import numpy as np
import shap
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import prepare as prep
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, classification_report, \
    precision_recall_fscore_support
from parameters import get_rf_parameters, get_features, beautify_feature_names
import matplotlib.pyplot as plt

FEATURES_FILE = '../data/truth.csv'

raw_data = pd.read_csv(FEATURES_FILE)
feature_labels = get_features()
#feature_labels = list(filter(lambda s: ".configuration" in s, feature_labels))
#feature_labels = list(filter(lambda s: "Template" not in s, feature_labels))

# Labels are the values we want to predict
labels = raw_data['actual']
# Convert to numpy array
features = raw_data[feature_labels]
# Saving feature names for later use
feature_list = list(features.columns)
short_feature_list = beautify_feature_names(feature_list)
features.columns = short_feature_list

names = raw_data['name']

np_features = np.array(features)
np_labels = np.array(labels)

kfold = KFold(n_splits=10, shuffle=True, random_state=43)

accuracys = []
precisions = []
precision_0 = []
precision_1 = []
recall_0 = []
recall_1 = []
f1_0 = []
f1_1 = []

for train, test in kfold.split(features):
    rf = RandomForestClassifier(**get_rf_parameters())
    rf.fit(np_features[train], np_labels[train])
    # Use the forest's predict method on the test data
    predictions = rf.predict(np_features[test])
    exact_prob = rf.predict_proba(np_features[test])

    explainer = shap.TreeExplainer(rf)
    i = 0
    for index in test:
        if predictions[i] != labels[index]:
            print(names[index], labels[index], predictions[i], exact_prob[i])
            #choosen_instance = features.loc[[index]]
            #shap_values = explainer.shap_values(choosen_instance)
            #shap.decision_plot(explainer.expected_value[1], shap_values[1], choosen_instance, show=False)
            #shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance, matplotlib=True, show=False)
            #underline_name = names[index].replace('/','_')
            #plt.tight_layout()
            #plt.savefig('plots/shap_single/' + underline_name + '.pdf',  format='pdf', dpi=1200, bbox_inches='tight')
            #plt.close()
        i = i + 1

    accuracy = accuracy_score(labels[test], predictions)
    p, r, f1, s = precision_recall_fscore_support(labels[test], predictions)

    precision_0.append(p[0])
    precision_1.append(p[1])
    recall_0.append(r[0])
    recall_1.append(r[1])
    f1_0.append(f1[0])
    f1_1.append(f1[1])

    accuracys.append(accuracy)

print('Model Performance')
print('Accuracies')
print(accuracys)
print('Accuracy = {:0.2f}%.'.format(np.mean(accuracys) * 100))
print('Std = {:0.2f}%.'.format(np.std(accuracys) * 100))

print('Precision_0 = {:0.2f}%.'.format(np.mean(precision_0) * 100))
print('Precision_1 = {:0.2f}%.'.format(np.mean(precision_1) * 100))

print('Recall_0 = {:0.2f}%.'.format(np.mean(recall_0) * 100))
print('Recall_1 = {:0.2f}%.'.format(np.mean(recall_1) * 100))

print('F1_0 = {:0.2f}%.'.format(np.mean(f1_0) * 100))
print('F1_1 = {:0.2f}%.'.format(np.mean(f1_1) * 100))


cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
rf = RandomForestClassifier(**get_rf_parameters())
scores = cross_val_score(rf, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))