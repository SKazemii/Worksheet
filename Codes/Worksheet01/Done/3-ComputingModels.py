import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import sys, os
from scipy.spatial import distance


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLPackage import Features as fe



features = np.load("./Datasets/features.npy")


print(features.shape)
columnsName = ["feature_" + str(i) for i in range(50)] + [ "subject ID", "left(0)/right(1)"]

DF_features = pd.DataFrame(
    features,
    columns = columnsName 
)

DF_featuresL = DF_features[DF_features["left(0)/right(1)"] == 0]
DF_featuresL4 = DF_features[DF_features["subject ID"] == 4]
DF_featuresLImposter = DF_features[DF_features["subject ID"] != 4]

# print(DF_featuresL4.head())
# print(DF_featuresL4.iloc[:, :-2])
# print(DF_featuresL4.iloc[:, :-4])


distModel1 = np.zeros((DF_featuresL4.shape[0], DF_featuresL4.shape[0]))
# corrModel1 = np.zeros((DF_featuresL4.shape[0], DF_featuresL4.shape[0]))

for i in range(DF_featuresL4.shape[0]):
    for j in range(DF_featuresL4.shape[0]):
        distModel1[i, j] = distance.euclidean(
            DF_featuresL4.iloc[i, :-4], DF_featuresL4.iloc[j, :-4]
        )
        # corrModel1[i, j] = np.corrcoef(
        #     DF_featuresL4.iloc[i, :-4], DF_featuresL4.iloc[j, :-4]
        # )[0,1]

# print(model1)
np.save("./Datasets/distModel1.npy", distModel1)
# np.save("./Datasets/corrModel1.npy", corrModel1)

distModel2 = np.zeros((DF_featuresL4.shape[0], DF_featuresLImposter.shape[0]))
# corrModel2 = np.zeros((DF_featuresL4.shape[0], DF_featuresLImposter.shape[0]))
for i in range(DF_featuresL4.shape[0]):
    for j in range(DF_featuresLImposter.shape[0]):
        distModel2[i, j] = distance.euclidean(
            DF_featuresL4.iloc[i, :-4], DF_featuresLImposter.iloc[j, :-4]
        )
        # corrModel2[i, j] = np.corrcoef(
        #     DF_featuresL4.iloc[i, :-4], DF_featuresLImposter.iloc[j, :-4]
        # )[0,1]

np.save("./Datasets/distModel2.npy", distModel2)
# np.save("./Datasets/corrModel2.npy", corrModel2)




print("[INFO] Done!!!")
sys.exit()