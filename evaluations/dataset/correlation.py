import matplotlib
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from helper import short_name
from parameters import beautify_feature_names, get_features

FEATURES_FILE = '../data/truth.csv'

raw_data = pd.read_csv(FEATURES_FILE)
features = raw_data[get_features()]

feature_list = list(features.columns)
short_feature_list = beautify_feature_names(feature_list)
features.columns = short_feature_list

corrMatrix = features.corr()
sn.set(font_scale=0.8)
sn.heatmap(corrMatrix, cmap="RdBu", vmin=-1, vmax=1, center=0, square=True)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(12, 8)
fig.autofmt_xdate()
plt.tight_layout()
fig.savefig('../plots/feature_correlation.pdf', dpi=300)
plt.show()
