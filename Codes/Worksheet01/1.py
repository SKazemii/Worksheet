import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import sys
from scipy.spatial import distance


# print("*** topic 1 ***")
# PerFootMetaDataBarefoot = np.load(
#     "./Datasets/RSScanData/perFootDataBarefoot/PerFootMetaDataBarefoot.npy"
# )
# PerFootMetaDataBarefoot = pd.DataFrame(
#     PerFootMetaDataBarefoot,
#     columns=[
#         "subject ID",
#         "left(0)/right(1) foot classification",
#         "foot index in gait cycle",
#         "partial",
#         " y center-offset",
#         "x center-offset",
#         "time center-offset",
#     ],
# )
# PerFootMetaDataBarefoot = PerFootMetaDataBarefoot.reset_index()

# CompleteMetaDataBarefoot = PerFootMetaDataBarefoot[
#     PerFootMetaDataBarefoot["partial"] == 0
# ]
# print("[INFO] shape of Meta Data", CompleteMetaDataBarefoot.shape)
# CompleteMetaDataBarefoot = CompleteMetaDataBarefoot.reset_index()


# AlignedFootDataBarefoot = np.load(
#     "Datasets/RSScanData/alignedPerFootDataBarefoot/AlignedFootDataBarefoot.npz"
# )
# files = AlignedFootDataBarefoot.files

# datalist = list()
# for i in CompleteMetaDataBarefoot.index:
#     datalist.append(AlignedFootDataBarefoot[files[i]])
# print("[INFO] length of data", len(datalist))

# print("*** topic 2 ***")

# # print(data[0].shape)
# # np.save("./Datasets/a.npy", data[0])


# # data = np.load("./Datasets/a.npy")
# # print(data.shape[2])

# features = list()

# for j in range(len(datalist)):
#     ML = list()
#     AP = list()

#     aML = list()
#     aAP = list()
#     data = datalist[j]
#     for i in range(data.shape[2]):
#         temp = data[:, :, i]
#         # imgplot = plt.imshow(temp)
#         temp2 = ndimage.measurements.center_of_mass(temp)
#         ML.append(temp2[0])
#         AP.append(temp2[1])

#         temp[temp == 0] = 0
#         temp[temp > 0] = 1
#         temp3 = ndimage.measurements.center_of_mass(temp)
#         aML.append(temp2[0])
#         aAP.append(temp2[1])

#     ML_f = ML - np.mean(ML)
#     aML_f = aML - np.mean(aML)
#     # plt.figure()
#     # plt.plot(range(100), aML)
#     # plt.figure()
#     # plt.plot(range(100), aML_f)

#     AP_f = AP - np.mean(AP)
#     aAP_f = aAP - np.mean(aAP)
#     # plt.figure()
#     # plt.plot(range(100), aAP)
#     # plt.figure()
#     # plt.plot(range(100), aAP_f)

#     # plt.figure()
#     # plt.plot(aAP_f, aML_f)

#     a = ML_f ** 2
#     b = AP_f ** 2
#     RD_f = np.sqrt(a + b)

#     a = aML_f ** 2
#     b = aAP_f ** 2
#     aRD_f = np.sqrt(a + b)
#     # plt.figure()
#     # plt.plot(range(100), aRD_f)

#     MDIST = np.mean(RD_f)
#     aMDIST = np.mean(aRD_f)

#     MDIST1 = np.mean(np.abs(ML_f))
#     aMDIST1 = np.mean(aML_f)

#     MDIST2 = np.mean(np.abs(AP_f))
#     aMDIST2 = np.mean(aAP_f)
#     # print(MDIST1)
#     # print(MDIST2)
#     # print("finished")
#     # plt.show()
#     # sys.exit()
#     MDISTAP = np.mean(np.abs(AP_f))
#     aMDISTAP = np.mean(np.abs(aAP_f))

#     RDIST = np.sqrt(np.mean(RD_f ** 2))
#     aRDIST = np.sqrt(np.mean(aRD_f ** 2))

#     RDISTAP = np.sqrt(np.mean(AP_f ** 2))
#     aRDISTAP = np.sqrt(np.mean(aAP_f ** 2))
#     features.append(
#         [
#             MDIST,
#             aMDIST,
#             MDIST1,
#             aMDIST1,
#             MDIST2,
#             aMDIST2,
#             MDISTAP,
#             aMDISTAP,
#             RDIST,
#             aRDIST,
#             RDISTAP,
#             aRDISTAP,
#             CompleteMetaDataBarefoot["left(0)/right(1) foot classification"][j],
#             CompleteMetaDataBarefoot["subject ID"][j],
#         ]
#     )


# npfeatures = np.array(features)
# np.save("./Datasets/npfeatures.npy", npfeatures)
npfeatures = np.load("./Datasets/npfeatures.npy")


print("*** topic 3 ***")
print(npfeatures.shape)
npfeatures = pd.DataFrame(
    npfeatures,
    columns=[
        "MDIST",
        "aMDIST",
        "MDIST1",
        "aMDIST1",
        "MDIST2",
        "aMDIST2",
        "MDISTAP",
        "aMDISTAP",
        "RDIST",
        "aRDIST",
        "RDISTAP",
        "aRDISTAP",
        "left(0)/right(1)",
        "subject ID",
    ],
)

npfeaturesL = npfeatures[npfeatures["left(0)/right(1)"] == 0]
npfeaturesL4 = npfeaturesL[npfeaturesL["subject ID"] == 4]
npfeaturesLImposter = npfeaturesL[npfeaturesL["subject ID"] != 4]

# print(npfeaturesL.head())
# print(npfeaturesL4.iloc[:, 0:5:2])

model1 = np.zeros((npfeaturesL4.shape[0], npfeaturesL4.shape[0]))
model11 = np.zeros((npfeaturesL4.shape[0], npfeaturesL4.shape[0]))

for i in range(npfeaturesL4.shape[0]):
    for j in range(npfeaturesL4.shape[0]):
        model1[i, j] = distance.euclidean(
            npfeaturesL4.iloc[i, 0:5:2], npfeaturesL4.iloc[j, 0:5:2]
        )
        model11[i, j] = np.corrcoef(
            npfeaturesL4.iloc[i, 0:5:2], npfeaturesL4.iloc[j, 0:5:2]
        )[0,1]

# print(model1)
np.save("./Datasets/model1.npy", model1)
np.save("./Datasets/model11.npy", model11)

model2 = np.zeros((npfeaturesL4.shape[0], npfeaturesLImposter.shape[0]))
model22 = np.zeros((npfeaturesL4.shape[0], npfeaturesLImposter.shape[0]))
for i in range(npfeaturesL4.shape[0]):
    for j in range(npfeaturesLImposter.shape[0]):
        model2[i, j] = distance.euclidean(
            npfeaturesL4.iloc[i, 0:5:2], npfeaturesLImposter.iloc[j, 0:5:2]
        )
        model22[i, j] = np.corrcoef(
            npfeaturesL4.iloc[i, 0:5:2], npfeaturesLImposter.iloc[j, 0:5:2]
        )[0,1]

np.save("./Datasets/model2.npy", model2)
np.save("./Datasets/model22.npy", model22)




print("finished")
sys.exit()
