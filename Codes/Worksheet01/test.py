from re import A
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import sys, os
from scipy.spatial import distance

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLPackage import util as perf
from MLPackage import Features as fe
####################################################
# COPTS = np.load("./Codes/Worksheet01/ToonCodes/COP.npy")
# f3D = np.load("./Codes/Worksheet01/ToonCodes/3D.npy")
# np.seterr('raise')

# print(fe.computeFDCE(COPTS))
data = np.load("./Datasets/datalist.npy")
metadata = np.load("./Datasets/metadatalist.npy")
print("[INFO] data shape: ", data.shape)
print("[INFO] metadata shape: ",metadata.shape)
first = list()
for j in range(metadata.shape[0]):
    # print(metadata[j,0:2])
    if metadata[j,0] == 4 and metadata[j,1] == 0:
        first.append(data[j])

print(len(first))
print(first[2].shape)
# np.save("./Datasets/first.npy", first)



feature_matrix = list()
for j in range(len(first)):
    COPTS = fe.computeCOPTimeSeries(first[j])
    # COATS = fe.computeCOATimeSeries(data[j], Binarize = "simple", Threshold = 1)

    pMDIST = fe.computeFDCE(COPTS)
    print(pMDIST)
sys.exit()

feature_matrix.append(pMDIST)
# afeatures.append(np.concatenate((aMDIST, aRDIST, aTOTEX, aMVELO, aRANGE, [aAREACC], [aAREACE], aMFREQ, aFDPD, [aFDCC], [aFDCE], metadata[j,0:2]), axis = 0) )
    


np.save("./Datasets/feature_matrix.npy", feature_matrix)
# np.save("./Datasets/afeatures.npy", afeatures)
print(len(feature_matrix))






dist_model = np.zeros((16,16))
for i in range(16):
    for j in range(16):
        # print(positive_samples.shape)
        # print(positive_samples.iloc[i, :])
        # print(positive_samples.iloc[i, :].values)
        dist_model[i, j] = distance.euclidean(
            feature_matrix[i], feature_matrix[j]
        )

print(dist_model.shape)
np.save("./Datasets/dist_model.npy", dist_model)



Model_client = np.min(np.ma.masked_where(dist_model==0,dist_model), axis = 0)
Model_client = np.expand_dims(Model_client,-1)
print(Model_client.shape)
print(Model_client)


FRR_temp = []
THRESHOLDs = np.linspace(0, 2, 50)
for tx in THRESHOLDs:
    E1 = np.zeros((Model_client.shape))
    E1[Model_client > tx] = 1
    FRR_temp.append(np.sum(E1)/dist_model.shape[1])

print(FRR_temp)   
print(THRESHOLDs.shape)   
plt.plot(THRESHOLDs,FRR_temp) 
plt.show()
print("[INFO] Done!!!")

####################################################
# EER = np.load("./Datasets/EER.npy")
# FPR = np.load("./Datasets/FPR.npy")
# FNR = np.load("./Datasets/FNR.npy")

# far = (np.mean(FPR, axis=0))
# print(np.mean(EER))
# frr = (np.mean(FNR, axis=0))
# THRESHOLDs = np.linspace(0, 300, 10)

# perf.ROC_plot_v2(far, frr, THRESHOLDs, "./")
# plt.show()



# %computeCFREQ
# %compute95FREQ
# %computeFREQD
# %computeMEDFREQ
# %computePOWER
####################################################

# a = np.load("./Codes/Worksheet01/ToonCodes/a.npy")

# a = (data[0:20,0:3])
# positive_model = np.zeros((a.shape[0], a.shape[0]))

# for i in range(a.shape[0]):
#     for j in range(a.shape[0]):
#         # print(positive_samples.shape)
#         # print(positive_samples.iloc[i, :])
#         # print(positive_samples.iloc[i, :].values)
#         positive_model[i, j] = distance.euclidean(
#             a[i, :], a[j, :]
#         )
# Model = np.min(np.ma.masked_where(positive_model==0,positive_model), axis = 0)
# Model = np.expand_dims(Model, -1)

# for tx in THRESHOLDs:
#     E1 = np.zeros((Model.shape))
#     E1[Model > tx] = 1
#     FRR_temp.append(np.sum(E1)/distModel1.shape[1])
# # for j in range(data.shape[0]):
# COPTS = fe.computeCOPTimeSeries(img)
# print(COPTS.shape)
# MDIST = fe.computeMDIST(COPTS)
# print(positive_model) 



# np.save("./Codes/Worksheet01/ToonCodes/positive_model.npy", positive_model)