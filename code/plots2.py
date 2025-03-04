import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from Utils import *

# Load the data

LD_data = np.load('./t-SNE/LD_data.npy')
HD_data = np.load('./t-SNE/HD_data.npy')

results = np.load('./t-SNE/results.npy', allow_pickle=True)

results = results.tolist()
# print(results.keys())
# print(results[3])
print(levenshteinDistanceDP(
    [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]))

freq = {}
total = 0
for key in results:
    freq[key] = len(results[key])
    total += len(results[key])
# sort the dictionary
freq = dict(sorted(freq.items()))
plt.plot(np.array(sorted(freq.keys())), freq.values())
plt.show()
for key in freq:
    freq[key] = freq[key]/total

sum = 0
for key in freq:
    sum += freq[key]
plt.plot(freq.keys(), freq.values())
# key where the cumulative sum is greater than 0.95
key_92 = 0
for key in freq:
    sum -= freq[key]
    if sum < 0.02:
        key_92 = key
        break
# plot a line at key 95
plt.axvline(x=key_92, color='r', linestyle='--')
plt.axhline(y=freq[key_92], color='r', linestyle='--')
# plot_data = np.array(sorted(freq.items()))
# print(freq)
# plt.plot(plot_data[:, 0], plot_data[:, 1])
plt.show()

# remove the keys that are more than 95% of the data
for key in list(results.keys()):
    if key > key_92:
        results.pop(key)


average = {}
for key in results:
    results[key] = np.array(results[key])
    # onmy keep first element of tuple
    results[key] = results[key][:, 0]
    average[key] = np.mean(results[key])

average = dict(sorted(average.items()))
# print(average)
# plt.plot(average.keys(), average.values())
# plt.show()

distances = {}
worst_case_distances = {}
best_case_distances = {}
for i in range(10000):
    for j in range(100):

        array_1 = np.random.rand(j+2)
        # random permutation
        array_2 = np.random.permutation(array_1)

        # compute the distance between the two arrays
        distance = levenshteinDistanceDP(array_1, array_2)
        if j+2 not in worst_case_distances:
            worst_case_distances[j+2] = distance
            best_case_distances[j+2] = distance
        else:
            if distance > worst_case_distances[j+2]:
                worst_case_distances[j+2] = distance
            if distance < best_case_distances[j+2]:
                best_case_distances[j+2] = distance

        if j+2 not in distances:
            distances[j+2] = distance/10000
        else:
            distances[j+2] += distance/10000

# print("distances")
# print(distances)
# print(average)
# plt.plot(distances.keys(), distances.values())
x = np.array(list(distances.keys()))
y = np.array(list(distances.values()))
# fit a line to the data
m, b = np.polyfit(x, y, 1)
print(m, b)
m_worst, b_worst = np.polyfit(x, np.array(
    list(worst_case_distances.values())), 1)
print(m_worst, b_worst)
m_best, b_best = np.polyfit(x, np.array(list(best_case_distances.values())), 1)
print(m_best, b_best)
# generate subplots
# plt.plot(x, m*x + b)

average_x = np.array(list(average.keys()))
average_y = np.array(list(average.values()))
diago = (m*average_x+b)/average_x
worst_diago = (m_worst*average_x+b_worst)
best_diago = (m_best*average_x+b_best)

plt.plot(average_x, diago)
plt.plot(average_x, worst_diago)
plt.plot(average_x, average_y)
plt.plot(average_x, best_diago)
plt.legend(["diago", "worst_diago", "average", "best_diago"])
plt.show()
plt.plot(best_case_distances.keys(), best_case_distances.values())

# rescale diagonal to 0-1
average_y = (average_y - np.min(diago)) / (np.max(diago) - np.min(diago))
diago = (diago - np.min(diago)) / (np.max(diago) - np.min(diago))
pathrnx = (diago - average_y)/diago

# plt.legend(["average", "diago"])
plt.show()
plt.plot(average_x, pathrnx)
plt.show()
