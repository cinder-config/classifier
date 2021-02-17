import pandas as pd
import numpy as np
import shap
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import prepare as prep
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve
import matplotlib.pyplot as plt

from helper import short_name_with_type
from parameters import get_rf_parameters, beautify_feature_names, get_features

FEATURES_FILE = 'data/truth.csv'

raw_data = pd.read_csv(FEATURES_FILE)

# Labels are the values we want to predict
labels = np.array(raw_data['actual'])
features = raw_data[get_features()]

# Saving feature names for later use
feature_list = list(features.columns)
short_feature_list = beautify_feature_names(feature_list)
features.columns = short_feature_list

# Convert to numpy array
# features = np.array(features)

shap.initjs()

rf = RandomForestClassifier(**get_rf_parameters())
rf.fit(features, labels)

explainer = shap.TreeExplainer(rf)


index = 3 #CHOSE INSTANCE
choosen_instance = features.loc[[index]]
shap_values = explainer.shap_values(choosen_instance)
shap.decision_plot(explainer.expected_value[1], shap_values[1], choosen_instance, show=False)
plt.tight_layout()
plt.savefig('plots/shap_observation_1.pdf')