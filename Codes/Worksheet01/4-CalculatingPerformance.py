import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from scipy import ndimage
import matplotlib.pyplot as plt
import sys, os
from scipy.spatial import distance

from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLPackage import util as perf

test_ratio = 0.2
persentage = 0.95
model_type = "average" #min median average
THRESHOLDs = np.linspace(0, 1, 100)
score = None#"B"
score = "B"#"B"

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
DF_features_PCA = perf.PCA_func(DF_features, persentage = persentage )
# DF_features_PCA = DF_features.copy()

EER_L = list(); FAR_L = list(); FRR_L = list()
EER_R = list(); FAR_R = list(); FRR_R = list()

EER_L_1 = list(); FAR_L_1 = list(); FRR_L_1 = list()
EER_R_1 = list(); FAR_R_1 = list(); FRR_R_1 = list()
ACC_L = list(); ACC_R = list()


# sys.exit()
for subject in subjects:
    
    if (subject % 10) == 0:
        print("[INFO] subject number: ", subject)
        # break
    
    for idx, direction in enumerate(["left_0", "right_1"]):
        DF_side = DF_features_PCA[DF_features_PCA["left(0)/right(1)"] == idx]
 
        path = "/Users/saeedkazemi/Documents/Python/Worksheet/Datasets/results/subject_" + str(int(subject)) + "/" + direction + "/"
        if not os.path.exists(path):
            os.chdir("/Users/saeedkazemi/Documents/Python/Worksheet/Datasets/results/")
            os.system("mkdir " + "subject_" + str(int(subject)))
            os.system("mkdir " + "subject_" + str(int(subject)) + "/" + direction)
            os.system("touch subject_" + str(int(subject)) + "/" + direction + "/file.txt")

    
        DF_positive_samples = DF_side[DF_side["subject ID"] == subject]
        DF_negative_samples = DF_side[DF_side["subject ID"] != subject]

        DF_positive_samples_test = DF_positive_samples.sample(frac = test_ratio, replace = False, random_state = 2)#frac =0.50
        DF_positive_samples_train = DF_positive_samples.drop(DF_positive_samples_test.index)

        DF_negative_samples_test = DF_negative_samples.sample(frac = test_ratio, replace = False, random_state = 2)#frac =0.50
        DF_negative_samples_train = DF_negative_samples.drop(DF_negative_samples_test.index)



        distModel1, distModel2 = perf.compute_model(DF_positive_samples_train.iloc[:, :-2].values, DF_negative_samples_train.iloc[:, :-2].values, mode = "dist", score = score)
        Model_client, Model_imposter = perf.model(distModel1, distModel2, model_type = model_type, score = score )
        
        np.save(path + "distModel1.npy", distModel1)
        np.save(path + "distModel2.npy", distModel2)

        

        

        FRR_temp = list()
        FAR_temp = list()

        FRR_temp_1 = list()
        FAR_temp_1 = list()

        if score != None:
            for tx in THRESHOLDs:
                E1 = np.zeros((Model_client.shape))
                E1[Model_client < tx] = 1
                FRR_temp.append(np.sum(E1)/(distModel1.shape[1]))


                E2 = np.zeros((Model_imposter.shape))
                E2[Model_imposter > tx] = 1
                FAR_temp.append(np.sum(E2)/distModel2.shape[1])

        elif score == None:
            for tx in THRESHOLDs:
                E1 = np.zeros((Model_client.shape))
                E1[Model_client > tx] = 1
                FRR_temp.append(np.sum(E1)/(distModel1.shape[1]))


                E2 = np.zeros((Model_imposter.shape))
                E2[Model_imposter < tx] = 1
                FAR_temp.append(np.sum(E2)/distModel2.shape[1])



        # print(distModel1)
        # print(Model_client)
        # print(FRR_temp)
        # print(THRESHOLDs)

        # sys.exit()


        perf.ROC_plot_v2(FAR_temp, FRR_temp, THRESHOLDs, path + model_type)
        EER_temp = (perf.compute_eer(FAR_temp, FRR_temp))


        samples_test = np.concatenate((DF_positive_samples_test.iloc[:, :-2].values, DF_negative_samples_test.iloc[:, :-2].values),axis = 0)
        one = (np.ones((DF_positive_samples_test.iloc[:, -2:-1].values.shape)))
        zero = (np.zeros((DF_negative_samples_test.iloc[:, -2:-1].values.shape)))
        label_test = np.concatenate((one, zero),axis = 0)
        distModel1 , distModel2 = perf.compute_model(DF_positive_samples_train.iloc[:, :-2].values, samples_test, mode = "dist", score = score)
        Model_client, Model_test = perf.model(distModel1, distModel2, model_type = model_type )

    
        t_idx = EER_temp[1]
        
        y_pred = np.zeros((Model_test.shape))
        y_pred[Model_test > THRESHOLDs[t_idx]] = 1
        temp = (accuracy_score(label_test, y_pred)*100)
        # print(THRESHOLDs[t_idx])


        if direction == "left_0":
            EER_L.append(EER_temp)
            FAR_L.append(FAR_temp)
            FRR_L.append(FRR_temp)
            ACC_L.append([subject, idx, temp, DF_positive_samples_test.shape[0], DF_negative_samples_test.shape[0], test_ratio])

            
        elif direction == "right_1":
            EER_R.append(EER_temp)
            FAR_R.append(FAR_temp)
            FRR_R.append(FRR_temp)
            ACC_R.append([subject, idx, temp, DF_positive_samples_test.shape[0], DF_negative_samples_test.shape[0], test_ratio])






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


np.save("./Datasets/NPY/ACC_L.npy", ACC_L)
print(np.mean(ACC_L, axis=0))
print(np.min(ACC_L, axis=0))
print(np.max(ACC_L, axis=0))

np.save("./Datasets/NPY/ACC_R.npy", ACC_R)
print(np.mean(ACC_R, axis=0))
print(np.min(ACC_R, axis=0))
print(np.max(ACC_R, axis=0))

print("[INFO] Done!!!")

