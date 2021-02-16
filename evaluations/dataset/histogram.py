import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from scipy.stats import stats

from helper import short_name
from parameters import get_features, beautify_feature_name

DATA = '../data/export-new.csv'
raw_data = pd.read_csv(DATA)
good = raw_data.loc[raw_data['actual'] == 1]
bad = raw_data.loc[raw_data['actual'] == 0]

fig, axs = plt.subplots(6,4, figsize=(10,12))

index = 0
row = 0

for feature in get_features():
    beautiful_name = beautify_feature_name(feature)
    col = index % 4
    print(row,col)

    tmp_good = good[feature]
    #tmp_good = tmp_good[tmp_good.between(tmp_good.quantile(.00), tmp_good.quantile(.95))]
    # tmp_good = tmp_good[(np.abs(stats.zscore(tmp_good)) < 3)]

    tmp_bad = bad[feature]
    # tmp_bad = tmp_bad[tmp_bad.between(tmp_bad.quantile(.00), tmp_bad.quantile(.95))]
    # tmp_bad = tmp_bad[(np.abs(stats.zscore(tmp_bad)) < 3)]

    #axs[row, col].hist([good[feature],bad[feature]], 20, alpha=0.5)
    #axs[row, col].hist(, 20, facecolor='r', alpha=0.5)

    ax = sns.violinplot(data=[tmp_good, tmp_bad], ax=axs[row, col], cut=0)
    ax.set_xticklabels(['Genuine','Pseudo'])
    # axs[row, col].violinplot([good[feature],bad[feature]], showmeans=True, showextrema=False)
    # axs[row, col].boxplot([good[feature],bad[feature]])
    # axs[row, col].set_yscale('log')
    axs[row, col].set_title(beautiful_name)
    if index % 4 == 3:
        row = row + 1
    index = index + 1

for label in ['commits','stars']:
    col = index % 4
    tmp_good = good[label]
    tmp_bad = bad[label]
    ax = sns.violinplot(data=[tmp_good, tmp_bad], ax=axs[row, col], cut=0)
    ax.set_xticklabels(['Genuine','Pseudo'])
    axs[row, col].set_title(label)
    if index % 4 == 3:
        row = row + 1
    index = index + 1

plt.tight_layout()
# plt.show()
plt.savefig('histogram_dataset.pdf')