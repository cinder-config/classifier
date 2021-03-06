import csv
import matplotlib.pyplot as plt
import numpy as np

CTEs = list()
error = list()
labels = list()

with open('../results/dataset.csv', 'r') as file:
    reader = csv.DictReader(file)
    for line in reader:
        labels.append(line['size'])
        CTEs.append(float(line['accuracy']))
        error.append(float(line['std']))

x_pos = np.arange(len(labels))

print(CTEs,error)

# Build the plot
fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
ax.bar(x_pos, CTEs,
       yerr=error,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=10)
ax.set_ylabel('Accuracy')
ax.set_xlabel('Size of data set')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('Model performance by data set size')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('../plots/subsize.pdf')
plt.show()