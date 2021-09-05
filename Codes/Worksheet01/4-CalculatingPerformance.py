import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import sys, os
from scipy.spatial import distance


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLPackage import ROC_plot as perf



features = np.load("./Datasets/features.npy")


print("[INFO] feature shape: ", features.shape)
columnsName = ["feature_" + str(i) for i in range(50)] + [ "subject ID", "left(0)/right(1)"]

DF_features = pd.DataFrame(
    features,
    columns = columnsName 
)
EER = list(); FPR = list(); FNR = list()
subjects = (DF_features["subject ID"].unique())
DF_featuresL = DF_features[DF_features["left(0)/right(1)"] == 0]
for subject in subjects:
    
    if (subject % 5) == 0:
        print("[INFO] subject number: ", subject)
    
    
    path = "/Users/saeedkazemi/Documents/Python/Worksheet/Datasets/results/subject_" + str(int(subject)) + "/"
    if not os.path.exists(path):
        os.chdir("/Users/saeedkazemi/Documents/Python/Worksheet/Datasets/results/")
        os.system("mkdir " + "subject_" + str(int(subject)))
        os.system("touch subject_" + str(int(subject)) + "/file.txt")


    
    
    DF_featuresL4 = DF_featuresL[DF_featuresL["subject ID"] == subject]
    DF_featuresLImposter = DF_featuresL[DF_featuresL["subject ID"] != subject]



    distModel1 = np.zeros((DF_featuresL4.shape[0], DF_featuresL4.shape[0]))
    for i in range(DF_featuresL4.shape[0]):
        for j in range(DF_featuresL4.shape[0]):
            distModel1[i, j] = distance.euclidean(
                DF_featuresL4.iloc[i, :-2], DF_featuresL4.iloc[j, :-2]
            )


    distModel2 = np.zeros((DF_featuresL4.shape[0], DF_featuresLImposter.shape[0]))
    for i in range(DF_featuresL4.shape[0]):
        for j in range(DF_featuresLImposter.shape[0]):
            distModel2[i, j] = distance.euclidean(
                DF_featuresL4.iloc[i, :-2], DF_featuresLImposter.iloc[j, :-2]
            )


    np.seterr('raise')
    EER1, FPR1, FNR1 = perf.performance(distModel1, distModel2, path)
    EER.append(EER1)
    FPR.append(FPR1)
    FNR.append(FNR1)



np.save("./Datasets/EER.npy", EER)
np.save("./Datasets/FPR.npy", FPR)
np.save("./Datasets/FNR.npy", FNR)


print("[INFO] Done!!!")

