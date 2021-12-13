import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import sys, os
from scipy.spatial import distance


working_path = os.getcwd()

print(sys.platform)
print(working_path)

meta_path = os.path.join(working_path, 'Datasets', 'Casia-D', 'perFootDataBarefoot', 'PerFootMetaDataBarefoot.npy')


print(meta_path)
print('[INFO] datalist and metadatalist\n Header of metsdata:\t\n1-"subject ID",\t\n2-"left(0)/right(1) foot classification",\t\n3-"foot index in gait cycle",\t\n4-"partial",\t\n5-" y center-offset",\t\n6-"x center-offset",\t\n7-"time center-offset"')
PerFootMetaDataBarefoot = np.load(
    meta_path
)

PerFootMetaDataBarefoot = pd.DataFrame(
    PerFootMetaDataBarefoot,
    columns=[
        "subject ID",
        "left(0)/right(1) foot classification",
        "foot index in gait cycle",
        "partial",
        " y center-offset",
        "x center-offset",
        "time center-offset",
    ],
).reset_index()

CompleteMetaDataBarefoot = PerFootMetaDataBarefoot[
    PerFootMetaDataBarefoot["partial"] == 0
]
print("[INFO] shape of Meta Data", CompleteMetaDataBarefoot.shape)
CompleteMetaDataBarefoot = CompleteMetaDataBarefoot.reset_index()

# Dataset_path = r"C:\Users\skazemi1\Documents\Projects\Worksheet\Datasets/RSScanData//AlignedFootDataBarefoot.npz"
dataset_path = os.path.join(working_path, 'Datasets', 'Casia-D', 'alignedPerFootDataBarefoot', 'AlignedFootDataBarefoot.npz')

AlignedFootDataBarefoot = np.load(
    dataset_path
)
files = AlignedFootDataBarefoot.files

datalist = list()
metadatalist = list()
for i in CompleteMetaDataBarefoot.index:
    datalist.append(AlignedFootDataBarefoot[files[i]])
    metadatalist.append(CompleteMetaDataBarefoot.iloc[i,:].values[2:])
print("[INFO] length of data", len(datalist))

saving_path = os.path.join(working_path, 'Datasets', 'Casia-D', 'Data-barefoot.npy')
np.save(saving_path, datalist)

saving_path = os.path.join(working_path, 'Datasets', 'Casia-D', 'Metadata-barefoot.npy')
np.save(saving_path, metadatalist)



print("[INFO] Done!!!")