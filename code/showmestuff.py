from Utils import *
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn import linear_model
# 0.0: 1, 2.0: 92, 3.0: 400, 4.0: 3662, 5.0: 15514, 6.0: 55967, 7.0: 116077, 8.0: 130811, 9.0: 40356
data = {}

for size in range(10, 500):
    init = np.arange(size)
    compare = init.copy()
    sum = 0
    for i in range(100000):
        data[size] = 0
        np.random.shuffle(compare)
        distance = levenshteinDistanceDP(init, compare)
        sum += distance
    data[size] = sum / 100000
linear_regressor = linear_model.LinearRegression()
regression = linear_regressor.fit(np.array(
    list(data.keys())).reshape(-1, 1), np.array(list(data.values())).reshape(-1, 1))
newy = regression.predict(np.array(list(data.keys())).reshape(-1, 1))
print(regression.coef_, regression.intercept_)
plt.plot(np.array(list(data.keys())).reshape(-1, 1), newy, color='red')
plt.plot(data.keys(), data.values())
plt.show()
