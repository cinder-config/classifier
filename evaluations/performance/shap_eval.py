import pandas as pd
import numpy as np
import shap
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import prepare as prep
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve
import matplotlib.pyplot as plt

from helper import short_name_with_type
from parameters import get_rf_parameters, beautify_feature_names, get_features

FEATURES_FILE = '../data/export-new.csv'

features = get_features()
#features = list(filter(lambda s: "configuration." in s,features))
raw_data = pd.read_csv(FEATURES_FILE)

# Labels are the values we want to predict
labels = np.array(raw_data['actual'])
features = raw_data[features]
feature_list = list(features.columns)

# Saving feature names for later use
short_feature_list = beautify_feature_names(feature_list)
features.columns = short_feature_list

# Convert to numpy array
# features = np.array(features)

shap.initjs()

rf = RandomForestClassifier(**get_rf_parameters())
rf.fit(features, labels)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(features)
shap.summary_plot(shap_values[1], features, plot_type="dot", show=False, max_display=22)
plt.tight_layout()
plt.savefig('plots/shap_summary_plot.pdf')

plt.clf()

## Waterfall Plot

def make_shap_waterfall_plot(shap_values, features, num_display=22):
    column_list = features.columns
    feature_ratio = (np.abs(shap_values).sum(0) / np.abs(shap_values).sum()) * 100
    column_list = column_list[np.argsort(feature_ratio)[::-1]]
    feature_ratio_order = np.sort(feature_ratio)[::-1]
    cum_sum = np.cumsum(feature_ratio_order)
    column_list = column_list[:num_display]
    feature_ratio_order = feature_ratio_order[:num_display]
    cum_sum = cum_sum[:num_display]

    num_height = 0
    if (num_display >= 22) & (len(column_list) >= 22):
        num_height = (len(column_list) - 22) * 1

    fig, ax1 = plt.subplots(figsize=(8, 10.3 + num_height))
    ax1.plot(cum_sum[::-1], column_list[::-1], c='blue', marker='o')
    ax2 = ax1.twiny()
    ax2.barh(column_list[::-1], feature_ratio_order[::-1], alpha=0.6)

    ax1.grid(True)
    ax2.grid(False)
    ax1.set_xticks(np.arange(0, round(cum_sum.max(), -1) + 1, 10))
    ax2.set_xticks(np.arange(0, round(feature_ratio_order.max(), -1) + 1, 10))
    ax1.set_xlabel('Cumulative Ratio')
    ax2.set_xlabel('Composition Ratio')
    ax1.tick_params(axis="y", labelsize=13)
    plt.ylim(-1, len(column_list))

make_shap_waterfall_plot(shap_values[1], features, 22)
plt.tight_layout()
plt.savefig('plots/shap_waterfall_plot.pdf')


# Visualizing top features
for feature in short_feature_list:
    plt.clf()

    shap.dependence_plot(feature, shap_values[1], features, interaction_index=None, show=False)
    plt.tight_layout()
    plt.savefig('plots/feature/'+feature+'.pdf')

