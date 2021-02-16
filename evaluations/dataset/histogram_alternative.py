from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

DATA = [['big', 'Ruby', 1, 4],
        ['big', 'Ruby', 0, 0],
        ['medium', 'Ruby', 1, 2],
        ['medium', 'Ruby', 0, 1],
        ['small', 'Ruby', 1, 2],
        ['small', 'Ruby', 0, 16],
        ['popular', 'Ruby', 1, 2],
        ['popular', 'Ruby', 0, 3],
        ['big', 'JavaScript', 1, 1],
        ['big', 'JavaScript', 0, 0],
        ['medium', 'JavaScript', 1, 3],
        ['medium', 'JavaScript', 0, 4],
        ['small', 'JavaScript', 1, 0],
        ['small', 'JavaScript', 0, 11],
        ['popular', 'JavaScript', 1, 1],
        ['popular', 'JavaScript', 0, 5],
        ['big', 'Python', 1, 4],
        ['big', 'Python', 0, 2],
        ['medium', 'Python', 1, 2],
        ['medium', 'Python', 0, 1],
        ['small', 'Python', 1, 6],
        ['small', 'Python', 0, 11],
        ['popular', 'Python', 1, 0],
        ['popular', 'Python', 0, 1],
        ['big', 'Java', 1, 6],
        ['big', 'Java', 0, 0],
        ['medium', 'Java', 1, 1],
        ['medium', 'Java', 0, 3],
        ['small', 'Java', 1, 0],
        ['small', 'Java', 0, 8],
        ['popular', 'Java', 1, 2],
        ['popular', 'Java', 0, 2],
        ['big', 'C++', 1, 4],
        ['big', 'C++', 0, 0],
        ['medium', 'C++', 1, 3],
        ['medium', 'C++', 0, 1],
        ['small', 'C++', 1, 7],
        ['small', 'C++', 0, 7],
        ['popular', 'C++', 1, 1],
        ['popular', 'C++', 0, 0],
        ['big', 'PHP', 1, 3],
        ['big', 'PHP', 0, 0],
        ['medium', 'PHP', 1, 4],
        ['medium', 'PHP', 0, 0],
        ['small', 'PHP', 1, 4],
        ['small', 'PHP', 0, 9],
        ['popular', 'PHP', 1, 2],
        ['popular', 'PHP', 0, 2]]

DATA.reverse()
data = np.transpose(np.array(DATA))
X = data[1]
Y = data[0]
size = data[3].astype('int32')
COLOR_MAPPING = {
    '1': 'green',
    '0': 'red'
}
color = np.array(list(map(lambda c: COLOR_MAPPING[c], data[2])))

plt.scatter(X, Y, s=100 * size, c=color, alpha=0.5)
plt.xlabel(['Ruby', 'JavaScript', 'Python', 'Java', 'C++', 'PHP'])
plt.ylabel(['big','medium','small','popular'])
plt.tight_layout()
plt.show()
