# vectorize the coil-20 dataset of images

import numpy as np
import pandas as pd
import os
from PIL import Image
path = "../DATASETS/coil-20/"
folders = os.listdir(path)
folders.remove(".DS_Store")
out = []
for i, folder in enumerate(folders):
    files = os.listdir(path + folder)
    for image in files:
        img = Image.open(path + folder + "/" + image)
        img = np.asarray(img)
        img = img.flatten()
        out.append(img)


# save data to csv
# random embeding of the data
X_embedded = np.random.rand(len(out), 2)
df = pd.DataFrame(X_embedded)
df.to_csv("../DATA/coil_20/random.csv")
