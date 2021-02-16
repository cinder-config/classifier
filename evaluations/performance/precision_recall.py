import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, plot_precision_recall_curve, roc_auc_score, \
    roc_curve, classification_report, precision_recall_curve, f1_score, auc
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import prepare as prep
from parameters import get_rf_parameters, get_threshold, get_features
import matplotlib.pyplot as plt

FEATURES_FILE = '../data/export-new.csv'

raw_data = pd.read_csv(FEATURES_FILE)
labels = raw_data['actual']
features = raw_data[get_features()]
np_features = np.array(features)
np_labels = np.array(labels)


# Convert to numpy array
features = np.array(features)

X = features
y = labels

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=42)
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(testy))]
# fit a model
model = RandomForestClassifier()
model.fit(trainX, trainy)
# predict probabilities
lr_probs = model.predict_proba(testX)
predictions = model.predict(testX)

# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Classifier')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot

ax = plt.gca()
ax.set_title('Receiver operating characteristic')

plt.savefig('plots/ruc4.pdf')

print(classification_report(testy, predictions))

cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
rf = RandomForestClassifier()
scores = cross_val_score(rf, features, labels, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


average_precision = average_precision_score(testy, predictions)
print('Average precision-recall score: {0:0.2f}'.format(
    average_precision))

plt.cla()

lr_probs = model.predict_proba(testX)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = model.predict(testX)
lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(testy[testy==1]) / len(testy)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, marker='.', label='Classifier')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# plt.show()
# show the plot

ax = plt.gca()
ax.set_title('Precision recall curve')
#disp = plot_precision_recall_curve(model, testX, testy)
#disp.ax_.

plt.savefig('plots/prc4.pdf')
