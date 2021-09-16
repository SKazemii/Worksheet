import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from scipy import ndimage
import matplotlib.pyplot as plt
import sys, os
from scipy.spatial import distance

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLPackage import util as perf

persentage = 0.95
model_type = "average" #min median average
THRESHOLDs = np.linspace(0, 10, 50)

pfeatures = np.load("./Datasets/pfeatures.npy")
# afeatures = np.load("./Datasets/afeatures.npy")

# features = np.concatenate((pfeatures[:, :-2],afeatures), axis = -1)
features = pfeatures

print("[INFO] feature shape: ", features.shape)
columnsName = ["feature_" + str(i) for i in range(features.shape[1]-2)] + [ "subject ID", "left(0)/right(1)"]

DF_features = pd.DataFrame(
    features,
    columns = columnsName 
)


# pMDIST, pRDIST, pTOTEX, pMVELO, pRANGE, [pAREACC], [pAREACE], pMFREQ, pFDPD, [pFDCC], [pFDCE], [pAREASW]

# DF_features.drop(DF_features.columns[[range(15,features.shape[1]-2)]], axis = 1, inplace = True)
# DF_features.drop(DF_features.columns[[range(0,12)]], axis = 1, inplace = True)
print(DF_features.head())
# sys.exit()
DF_features = DF_features.fillna(0)

subjects = (DF_features["subject ID"].unique())


labels = DF_features["subject ID"].values
labels = (np.expand_dims(labels, axis = -1))

###############################
# Scale data befor applying PCA
scaling = StandardScaler()
Scaled_data = scaling.fit_transform(DF_features.iloc[:, :-2])

principal = PCA()
PCA_out = principal.fit_transform(Scaled_data)

variance_ratio = (np.cumsum(principal.explained_variance_ratio_))
high_var_PC = np.zeros(variance_ratio.shape)
high_var_PC[variance_ratio <= persentage] = 1

loadings = principal.components_
num_pc = int(np.sum(high_var_PC))
print(num_pc)
columnsName = ["PC"+str(i) for i in list(range(1, num_pc+1))] + ["subject ID", "left(0)/right(1)"]
DF_features_PCA = (pd.DataFrame(np.concatenate((PCA_out[:,:num_pc],DF_features.iloc[:, -2:].values), axis = 1), columns = columnsName))
###############################

DF_features_PCA = DF_features.copy()

EER_L = list(); FAR_L = list(); FRR_L = list()
EER_R = list(); FAR_R = list(); FRR_R = list()

EER_L_1 = list(); FAR_L_1 = list(); FRR_L_1 = list()
EER_R_1 = list(); FAR_R_1 = list(); FRR_R_1 = list()

print(DF_features_PCA.head())
# sys.exit()
for subject in subjects:
    
    if (subject % 10) == 0:
        print("[INFO] subject number: ", subject)
    
    for idx, direction in enumerate(["left_0", "right_1"]):
        DF_featuresL = DF_features_PCA[DF_features_PCA["left(0)/right(1)"] == idx]

        path = "/Users/saeedkazemi/Documents/Python/Worksheet/Datasets/results/subject_" + str(int(subject)) + "/" + direction + "/"
        if not os.path.exists(path):
            os.chdir("/Users/saeedkazemi/Documents/Python/Worksheet/Datasets/results/")
            os.system("mkdir " + "subject_" + str(int(subject)))
            os.system("mkdir " + "subject_" + str(int(subject)) + "/" + direction)
            os.system("touch subject_" + str(int(subject)) + "/" + direction + "/file.txt")

    
        DF_featuresL4 = DF_featuresL[DF_featuresL["subject ID"] == subject]
        DF_featuresLImposter = DF_featuresL[DF_featuresL["subject ID"] != subject]


        distModel1, distModel2 = perf.compute_model(DF_featuresL4.iloc[:, :-2].values, DF_featuresLImposter.iloc[:, :-2].values, mode = "dist")

        np.save(path + "distModel1.npy", distModel1)
        np.save(path + "distModel2.npy", distModel2)

        
        if model_type == "average":
            Model_client = (np.sum(distModel1, axis = 0))/(distModel1.shape[1]-1)
            Model_client = np.expand_dims(Model_client,-1)
            
            Model_imposter = (np.sum(distModel2, axis = 0))/(distModel1.shape[1])
            Model_imposter = np.expand_dims(Model_imposter, -1)
              
        elif model_type == "min":

            Model_client = np.min(np.ma.masked_where(distModel1==0,distModel1), axis = 0)
            Model_client = np.expand_dims(Model_client,-1)
            
            Model_imposter = np.min(np.ma.masked_where(distModel2==0,distModel2), axis = 0)
            Model_imposter = np.expand_dims(Model_imposter, -1)
                 
        elif model_type == "median":
            temp = np.ma.masked_where(distModel1 == 0, distModel1)
            Model_client = np.ma.median(temp, axis = 0).filled(0)
            Model_client = np.expand_dims(Model_client,-1)
            # print(Model_client.shape)
            

            Model_imposter = np.median(distModel2, axis = 0)
            Model_imposter = np.expand_dims(Model_imposter, -1)
            # print(Model_imposter.shape)
            # sys.exit()
        

        FRR_temp = list()
        FAR_temp = list()

        FRR_temp_1 = list()
        FAR_temp_1 = list()

        for tx in THRESHOLDs:
            E1 = np.zeros((Model_client.shape))
            E1[Model_client > tx] = 1
            FRR_temp.append(np.sum(E1)/(distModel1.shape[1]))


            E2 = np.zeros((Model_imposter.shape))
            E2[Model_imposter < tx] = 1
            FAR_temp.append(np.sum(E2)/distModel2.shape[1])

        
        perf.ROC_plot_v2(FAR_temp, FRR_temp, THRESHOLDs, path + model_type)
        EER_temp = (perf.compute_eer(FAR_temp, FRR_temp))

        if direction == "left_0":
            EER_L.append(EER_temp)
            FAR_L.append(FAR_temp)
            FRR_L.append(FRR_temp)
            
        elif direction == "right_1":
            EER_R.append(EER_temp)
            FAR_R.append(FAR_temp)
            FRR_R.append(FRR_temp)

        






plt.close()
os.chdir("/Users/saeedkazemi/Documents/Python/Worksheet/")


far = (np.mean(FAR_L, axis=0))
frr = (np.mean(FRR_L, axis=0))
perf.ROC_plot_v2(far, frr, THRESHOLDs, "./L_" + model_type)

print(np.mean(EER_L, axis=0))
print(np.min(EER_L, axis=0))
print(np.max(EER_L, axis=0))

far = (np.mean(FAR_R, axis=0))
frr = (np.mean(FRR_R, axis=0))
perf.ROC_plot_v2(far, frr, THRESHOLDs, "./R_" + model_type)

print(np.mean(EER_R, axis=0))
print(np.min(EER_R, axis=0))
print(np.max(EER_R, axis=0))

np.save("./Datasets/NPY/EER_R_B.npy", EER_R)
np.save("./Datasets/NPY/FAR_R_B.npy", FAR_R)
np.save("./Datasets/NPY/FRR_R_B.npy", FRR_R)


np.save("./Datasets/NPY/EER_L_B.npy", EER_L)
np.save("./Datasets/NPY/FAR_L_B.npy", FAR_L)
np.save("./Datasets/NPY/FRR_L_B.npy", FRR_L)


print("[INFO] Done!!!")

