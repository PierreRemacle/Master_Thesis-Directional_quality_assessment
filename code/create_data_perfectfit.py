import pandas as pd
import numpy as np


dataHD = np.zeros((100, 3))
data = np.random.rand(100, 2)
dataHD[:, 1:] = data
df = pd.DataFrame(data)
df.to_csv(f"../DATA/perfect_fit_LD.csv")
df = pd.DataFrame(dataHD)
df.to_csv(f"../DATA/perfect_fit_HD.csv")
