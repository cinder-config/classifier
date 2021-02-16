import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines

CTEs = list()
error = list()
labels = list()

with open('results.csv', 'r') as file:
    reader = csv.DictReader(file)
    for line in reader:
        labels.append(line['language'] + "\n" + "$\\it{n_{train}=" + line['samples_train_other'] + "}$" + "\n" + "$\\it{n_{test}=" + line['samples_test_other'] + "}$")
        CTEs.append(float(line['accuracy_other']))
        error.append(float(line['std_other']))

x_pos = np.arange(len(labels))

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
ax.set_xlabel('Programming Language')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title("Classifier score by programming language")
ax.yaxis.grid(True)

plt.xlim([-0.75, 5.75])

# averages = np.empty(6,dtype=float)
# averages.fill(np.mean(CTEs))
# x_pos[0] = -1
ax.plot([-1,6], [np.mean(CTEs),np.mean(CTEs)])

print(np.mean(CTEs))

# Save the figure and show
plt.tight_layout()
plt.savefig('language.pdf')
plt.show()