import os
import numpy as np
import h5py
import pandas as pd
import config as cfg
import math

import pickle
import matplotlib.pyplot as plt

print("[INFO] reading dataset....")
with h5py.File(cfg.dataset_file, "r") as hdf:
    metadata = hdf.get("/barefoot/metadata")[:]


index = ["Sample_" + str(i) for i in np.arange(1744)]
col = [str(i) for i in np.arange(8)]
a = pd.DataFrame(metadata[0:1744, 0:8], index=index, columns=col)


index = ["Sample_" + str(i) for i in np.arange(1744)]
w = pd.DataFrame(np.zeros([1744, 2]), index=index, columns=["ID", "distance"])
subjects = set(a["0"])

for subject in subjects:
    b = a[a["0"] == subject]
    steps = set(b["4"])
    # print(b)

    diss = list()
    for step in range(b.shape[0] - 1):
        c = b.iloc[:, 4:8].values
        aa = step + 1
        if c[step, 1] == c[step + 1, 1]:
            x = c[step, 2] - c[step + 1, 2]
            y = c[step, 3] - c[step + 1, 3]
            a.loc[(a["4"] == step) & (a["0"] == subject), "2"] = math.sqrt(
                x ** 2 + y ** 2
            )
        else:
            a.loc[(a["4"] == step) & (a["0"] == subject), "2"] = a.loc[
                (a["4"] == step - 1) & (a["0"] == subject), "2"
            ].values
    a.loc[(a["4"] == step + 1) & (a["0"] == subject), "2"] = a.loc[
        (a["4"] == step) & (a["0"] == subject), "2"
    ].values
    # diss.append(diss[-1])


print(a["2"].head(50))
# 1/0
with open(os.path.join(cfg.pickle_dir, "df_inter_stride.pickle"), "wb") as handle:
    pickle.dump(a["2"], handle)

with open(os.path.join(cfg.pickle_dir, "df_inter_stride.txt"), "w") as handle:
    handle.write(a["2"].to_string())