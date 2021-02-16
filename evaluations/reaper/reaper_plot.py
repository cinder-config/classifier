import matplotlib
import pandas as pd
import scipy
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from helper import short_name

REAPER_FILE = 'reaper_combined.csv'

data = pd.read_csv(REAPER_FILE)
correlation_matrix = data.drop('name', axis=1).corr()
confusion_matrix = pd.crosstab(data['ciclassifier'], data['reaper'], rownames=['ciclassifier'], colnames=['reaper'])

corr = pearsonr(data['ciclassifier'],data['reaper'])
print(corr)
print(correlation_matrix)
print(confusion_matrix)

# Build the plot
plt.figure(figsize=(16, 7))
sn.set(font_scale=1.4)
sn.heatmap(confusion_matrix, annot=True, cmap=plt.cm.Blues, linewidths=0.2, fmt="d")

# Add labels to the plot
class_names = ['Not engineered', 'Engineered']
class_names2 = ['Pseudo', 'Genuine']
tick_marks = np.arange(len(class_names)) + 0.5
tick_marks2 = tick_marks
plt.xticks(tick_marks, class_names, rotation=0)
plt.yticks(tick_marks2, class_names2, rotation=0)
plt.xlabel('Reaper')
plt.ylabel('Model')
plt.title('Confusion Matrix')
plt.savefig('reaper.pdf')