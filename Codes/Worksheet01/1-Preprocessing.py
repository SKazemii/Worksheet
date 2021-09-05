import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import sys
from scipy.spatial import distance


print('[INFO] datalist and metadatalist\n Header of metsdata:\t\n1-"subject ID",\t\n2-"left(0)/right(1) foot classification",\t\n3-"foot index in gait cycle",\t\n4-"partial",\t\n5-" y center-offset",\t\n6-"x center-offset",\t\n7-"time center-offset"')
PerFootMetaDataBarefoot = np.load(
    "./Datasets/RSScanData/perFootDataBarefoot/PerFootMetaDataBarefoot.npy"
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
)
PerFootMetaDataBarefoot = PerFootMetaDataBarefoot.reset_index()

CompleteMetaDataBarefoot = PerFootMetaDataBarefoot[
    PerFootMetaDataBarefoot["partial"] == 0
]
print("[INFO] shape of Meta Data", CompleteMetaDataBarefoot.shape)
CompleteMetaDataBarefoot = CompleteMetaDataBarefoot.reset_index()


AlignedFootDataBarefoot = np.load(
    "Datasets/RSScanData/alignedPerFootDataBarefoot/AlignedFootDataBarefoot.npz"
)
files = AlignedFootDataBarefoot.files

datalist = list()
metadatalist = list()
for i in CompleteMetaDataBarefoot.index:
    datalist.append(AlignedFootDataBarefoot[files[i]])
    metadatalist.append(CompleteMetaDataBarefoot.iloc[i,:].values[2:])
print("[INFO] length of data", len(datalist))


np.save("./Datasets/datalist.npy", datalist)
np.save("./Datasets/metadatalist.npy", metadatalist)