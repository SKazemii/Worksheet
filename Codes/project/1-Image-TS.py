print("[INFO] importing libraries....")


import os, sys
import numpy as np
import h5py
import pandas as pd
import config as cfg
import pickle
import matplotlib.pyplot as plt

print("[INFO] defining functions....")


def interpolate(inp, fi):
    i, f = (
        int(fi // 1),
        fi % 1,
    )  # Split floating-point index into whole & fractional parts.
    j = i + 1 if round(f, 4) > 0 else i  # Avoid index error.
    return (1 - f) * inp[i] + f * inp[j]


def centriod(img):
    wx = list()
    wy = list()
    for j in range(90):
        non_zero_elements = np.nonzero(img[j, :, :])
        x = 0
        y = 0
        for i in range(len(non_zero_elements[0])):
            x += non_zero_elements[1][i]
            y += non_zero_elements[0][i]

        if x == 0:
            # x = 0
            pass
        else:
            x /= len(non_zero_elements[0])

        if y == 0:
            # y = 0
            pass
        else:
            y /= len(non_zero_elements[0])

        wx.append(x)
        wy.append(y)

    return wx, wy


print("[INFO] reading dataset....")
with h5py.File(cfg.dataset_file, "r") as hdf:
    barefoots = hdf.get("/barefoot/data")[:]
    metadata = hdf.get("/barefoot/metadata")[:]

print("[INFO] saving 3 samples....")
img = np.squeeze(barefoots[1, :, :, :])

index = ["Sample_" + str(i) for i in np.arange(barefoots.shape[0] - 1)]


df_inter_stride = pd.DataFrame(metadata[0:1744, 6:8], index=index, columns=["x", "y"])
with open(os.path.join(cfg.pickle_dir, "df_inter_stride.pickle"), "wb") as handle:
    pickle.dump(df_inter_stride, handle)
with open(os.path.join(cfg.pickle_dir, "df_inter_stride.txt"), "w") as handle:
    handle.write(df_inter_stride.to_string())


ss = np.concatenate(
    (img[18, 21:65, 17:42], img[25, 21:65, 17:42], img[32, 21:65, 17:42]), axis=1
)
i = plt.imshow(ss)
plt.savefig(os.path.join(cfg.fig_dir, "frame.png"), bbox_inches="tight")

print("[INFO] storing in pandas dataframe ....")
Data_sum = list()
Data_max = list()
Data_xCe = list()
Data_yCe = list()
new_len = 100


for sample in range(barefoots.shape[0] - 1):
    img = np.squeeze(barefoots[sample, :, :, :])
    aa = np.sum(img, axis=2)
    bb = np.sum(aa, axis=1)
    # print(bb)
    # plt.plot(bb)
    # plt.show()
    # 1 / 0
    bb = np.trim_zeros(bb)
    delta = (len(bb) - 1) / (new_len - 1)
    outp = [interpolate(bb, i * delta) for i in range(new_len)]
    Data_sum.append(outp)

    img = np.squeeze(barefoots[sample, :, :, :])
    aa = np.max(img, axis=2)
    bb = np.max(aa, axis=1)
    bb = np.trim_zeros(bb)
    delta = (len(bb) - 1) / (new_len - 1)
    outp = [interpolate(bb, i * delta) for i in range(new_len)]
    Data_max.append(outp)

    wx, wy = centriod(img)
    wx = np.trim_zeros(wx)
    delta = (len(wx) - 1) / (new_len - 1)
    outp = [interpolate(wx, i * delta) for i in range(new_len)]
    Data_xCe.append(outp)
    wy = np.trim_zeros(wy)
    delta = (len(wy) - 1) / (new_len - 1)
    outp = [interpolate(wy, i * delta) for i in range(new_len)]
    Data_yCe.append(outp)


index = ["Sample_" + str(i) for i in np.arange(barefoots.shape[0] - 1)]
column = [i for i in np.arange(new_len)]

df_sum = pd.DataFrame(Data_sum, columns=column, index=index).T
with open(os.path.join(cfg.pickle_dir, "df_sum.pickle"), "wb") as handle:
    pickle.dump(df_sum, handle)

df_max = pd.DataFrame(Data_max, columns=column, index=index).T
with open(os.path.join(cfg.pickle_dir, "df_max.pickle"), "wb") as handle:
    pickle.dump(df_max, handle)

df_xCe = pd.DataFrame(Data_xCe, columns=column, index=index).T
with open(os.path.join(cfg.pickle_dir, "df_xCe.pickle"), "wb") as handle:
    pickle.dump(df_xCe, handle)

df_yCe = pd.DataFrame(Data_yCe, columns=column, index=index).T
with open(os.path.join(cfg.pickle_dir, "df_yCe.pickle"), "wb") as handle:
    pickle.dump(df_yCe, handle)

df_label = pd.DataFrame(metadata[0:1744, 0], index=index, columns=["ID"])
with open(os.path.join(cfg.pickle_dir, "df_label.pickle"), "wb") as handle:
    pickle.dump(df_label, handle)

sys.exit()
print("[INFO] Saving and showing the plot of the some samples...")
plt.figure()
axes = plt.axes()
df_sum["Sample_1"].plot(ax=axes)
df_sum["Sample_100"].plot(ax=axes)
df_sum["Sample_200"].plot(ax=axes)
df_sum["Sample_150"].plot(ax=axes)
df_sum["Sample_400"].plot(ax=axes)

plt.legend(["Sample 1", "Sample 100", "Sample 200", "Sample 300", "Sample 400"])
plt.savefig(os.path.join(cfg.fig_dir, "df_sum.png"), bbox_inches="tight")


print("[INFO] Saving and showing the plot of the some samples...")
plt.figure()
axes = plt.axes()
df_max["Sample_1"].plot(ax=axes)
df_max["Sample_100"].plot(ax=axes)
df_max["Sample_200"].plot(ax=axes)
df_max["Sample_150"].plot(ax=axes)
df_max["Sample_400"].plot(ax=axes)

plt.legend(["Sample 1", "Sample 100", "Sample 200", "Sample 300", "Sample 400"])
plt.savefig(os.path.join(cfg.fig_dir, "df_max.png"), bbox_inches="tight")

print("[INFO] Saving and showing the plot of the some samples...")
plt.figure()
axes = plt.axes()
df_xCe["Sample_1"].plot(ax=axes)
df_xCe["Sample_100"].plot(ax=axes)
df_xCe["Sample_200"].plot(ax=axes)
df_xCe["Sample_150"].plot(ax=axes)
df_xCe["Sample_400"].plot(ax=axes)

plt.legend(["Sample 1", "Sample 100", "Sample 200", "Sample 300", "Sample 400"])
plt.savefig(os.path.join(cfg.fig_dir, "df_xCe.png"), bbox_inches="tight")

print("[INFO] Saving and showing the plot of the some samples...")
plt.figure()
axes = plt.axes()
df_yCe["Sample_1"].plot(ax=axes)
df_yCe["Sample_100"].plot(ax=axes)
df_yCe["Sample_200"].plot(ax=axes)
df_yCe["Sample_150"].plot(ax=axes)
df_yCe["Sample_400"].plot(ax=axes)

plt.legend(["Sample 1", "Sample 100", "Sample 200", "Sample 300", "Sample 400"])
plt.savefig(os.path.join(cfg.fig_dir, "df_yCe.png"), bbox_inches="tight")
