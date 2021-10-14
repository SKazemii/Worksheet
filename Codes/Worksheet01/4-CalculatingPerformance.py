import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from pandas.core.frame import DataFrame
from scipy import ndimage
import matplotlib.pyplot as plt
import sys, os, timeit
from pathlib import Path as Pathlb

from scipy.spatial import distance

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLPackage import util as perf

test_ratios = [0.2, 0.35, 0.5]
persentages = [1.0]#[0.95, 1.0]
modes = ["corr", "dist"]
model_types = ["min", "median", "average"]
THRESHOLDs = np.linspace(0, 1, 100)
score = "A"#"B"
normilizings = ["z-score"]#["None", "z-score", "minmax"]

classifier = "knn"#"matching"

feature_names = ["MDIST", "RDIST", "TOTEX", "MVELO", "RANGE", "AREAXX", "MFREQ", "FDPD", "FDCX"]


cols = ["Mode", "Model_Type", "Test_Size", "Normalizition", "Features_Set", "PCA",
"Mean_Accuracy_Left", "Mean_EER_Left", "Mean_Accuracy_Right", "Mean_EER_Right",
"Min_Accuracy_Left", "Min_EER_Left", "Min_Accuracy_Right", "Min_EER_Right",
"Max_Accuracy_Left", "Max_EER_Left", "Max_Accuracy_Right", "Max_EER_Right",
"Median_Accuracy_Left", "Median_EER_Left", "Median_Accuracy_Right", "Median_EER_Right"] + ["FAR_L_" + str(i) for i in range(100)] + ["FRR_L_" + str(i) for i in range(100)] + ["FAR_R_" + str(i) for i in range(100)] + ["FRR_R_" + str(i) for i in range(100)]

Results_DF = pd.DataFrame(columns=cols)
working_path = os.getcwd()


print(sys.platform)

feature_path = os.path.join(working_path, 'Datasets', 'pfeatures.npy')
pfeatures = np.load(feature_path)
# afeatures = np.load("./Datasets/afeatures.npy")


# features = np.concatenate((pfeatures[:, :-2],afeatures), axis = -1)
features = pfeatures

index =0
tic=timeit.default_timer()
for persentage in persentages:
    for normilizing in normilizings:
        if normilizing == "minmax":
            scaling = preprocessing.MinMaxScaler()
            Scaled_data = scaling.fit_transform(features[:, :-2])
        elif normilizing == "z-score":
            scaling = preprocessing.StandardScaler()
            Scaled_data = scaling.fit_transform(features[:, :-2])
        elif normilizing == "None":
            Scaled_data = features[:, :-2]

        print("[INFO] feature shape: ", features.shape)
        columnsName = ["feature_" + str(i) for i in range(features.shape[1]-2)] + [ "subject ID", "left(0)/right(1)"]

        DF_features_all = pd.DataFrame(
            np.concatenate((Scaled_data,features[:, -2:]), axis = 1),
            columns = columnsName 
        )



        print(DF_features_all.head())
        DF_features_all = DF_features_all.fillna(0)

        subjects = (DF_features_all["subject ID"].unique())


        labels = DF_features_all["subject ID"].values
        labels = (np.expand_dims(labels, axis = -1))

        ###############################
        # pMDIST, pRDIST, pTOTEX, pMVELO, pRANGE, [pAREACC], [pAREACE], pMFREQ, pFDPD, [pFDCC], [pFDCE], [pAREASW]
        for x in range(-3,features.shape[1]-2,3):
        # for x in range(-3,5,3):
            if x == -3:
                DF_features = DF_features_all.copy()
                feat_name = "All"
            else:
                break
                DF_features = DF_features_all.copy()
                DF_features.drop(DF_features.columns[[range(x+3,features.shape[1]-2)]], axis = 1, inplace = True)
                DF_features.drop(DF_features.columns[[range(0,x)]], axis = 1, inplace = True)
                feat_name = feature_names[int(x/3)]


            



            for mode in modes:

                if mode == "corr" and feat_name != "All" and persentage != 1.0:
                    break
                
                for model_type in model_types:
                    for test_ratio in test_ratios:

                        

                        EER_L = list(); FAR_L = list(); FRR_L = list()
                        EER_R = list(); FAR_R = list(); FRR_R = list()

                        EER_L_1 = list(); FAR_L_1 = list(); FRR_L_1 = list()
                        EER_R_1 = list(); FAR_R_1 = list(); FRR_R_1 = list()
                        ACC_L = list(); ACC_R = list()


                        folder = str(persentage) + "_" + normilizing + "_" + feat_name  + "_" + mode + "_" + model_type + "_" +  str(test_ratio) 

                        folder_path = os.path.join(working_path, 'results', folder)

                        Pathlb(folder_path).mkdir(parents=True, exist_ok=True)


                        print("[INFO] Working Directory:  ", folder)

                        for subject in subjects:
                            if (subject % 86) == 0:
                                continue
                            
                            if (subject % 20) == 0:
                                print("[INFO] --------------- Subject Number: ", subject)
                                # break
                            
                            for idx, direction in enumerate(["left_0", "right_1"]):

                                DF_side = DF_features[DF_features["left(0)/right(1)"] == idx]


                            
                                DF_positive_samples = DF_side[DF_side["subject ID"] == subject]
                                DF_negative_samples = DF_side[DF_side["subject ID"] != subject]

                                    
                                    
                                DF_positive_samples_test = DF_positive_samples.sample(frac = test_ratio, 
                                                                                      replace = False, 
                                                                                      random_state = 2)
                                DF_positive_samples_train = DF_positive_samples.drop(DF_positive_samples_test.index)

                                DF_negative_samples_test = DF_negative_samples.sample(frac = test_ratio,
                                                                                      replace = False, 
                                                                                      random_state = 2)
                                DF_negative_samples_train = DF_negative_samples.drop(DF_negative_samples_test.index)

                                if persentage != 1.0:
                                    df_train = pd.concat([DF_positive_samples_train, DF_negative_samples_train])
                                    df_test = pd.concat([DF_positive_samples_test, DF_negative_samples_test])

                                    scaling = StandardScaler()
                                    Scaled_train = scaling.fit_transform(df_train.iloc[:, :-2])
                                    Scaled_test = scaling.transform(df_test.iloc[:, :-2])

                                    principal = PCA()
                                    PCA_out_train = principal.fit_transform(Scaled_train)
                                    PCA_out_test = principal.transform(Scaled_test)

                                    variance_ratio = np.cumsum(principal.explained_variance_ratio_)
                                    high_var_PC = np.zeros(variance_ratio.shape)
                                    high_var_PC[variance_ratio <= persentage] = 1

                                    loadings = principal.components_
                                    num_pc = int(np.sum(high_var_PC))


                                    columnsName = ["PC"+str(i) for i in list(range(1, num_pc+1))] + ["subject ID", "left(0)/right(1)"]
                                    DF_features_PCA_train = (pd.DataFrame(np.concatenate((PCA_out_train[:,:num_pc],df_train.iloc[:, -2:].values), axis = 1), columns = columnsName))
                                    DF_features_PCA_test = (pd.DataFrame(np.concatenate((PCA_out_test[:,:num_pc],df_test.iloc[:, -2:].values), axis = 1), columns = columnsName))

                                    DF_positive_samples_train = DF_features_PCA_train[DF_features_PCA_train["subject ID"] == subject]
                                    DF_negative_samples_train = DF_features_PCA_train[DF_features_PCA_train["subject ID"] != subject]
                                    
                                    
                                    DF_positive_samples_test = DF_features_PCA_test[DF_features_PCA_test["subject ID"] == subject]   
                                    DF_negative_samples_test = DF_features_PCA_test[DF_features_PCA_test["subject ID"] != subject]



                                distModel1, distModel2 = perf.compute_model(DF_positive_samples_train.iloc[:, :-2].values,
                                                                            DF_negative_samples_train.iloc[:, :-2].values,
                                                                            mode = mode, score = score)


                                Model_client, Model_imposter = perf.model(distModel1,
                                                                            distModel2, 
                                                                            model_type = model_type, 
                                                                            score = score )


                                FRR_temp = list()
                                FAR_temp = list()

                                FRR_temp_1 = list()
                                FAR_temp_1 = list()

                                if score is not None:
                                    for tx in THRESHOLDs:
                                        E1 = np.zeros((Model_client.shape))
                                        E1[Model_client < tx] = 1
                                        FRR_temp.append(np.sum(E1)/(distModel1.shape[1]))


                                        E2 = np.zeros((Model_imposter.shape))
                                        E2[Model_imposter > tx] = 1
                                        FAR_temp.append(np.sum(E2)/distModel2.shape[1])

                                elif score is None:
                                    for tx in THRESHOLDs:
                                        E1 = np.zeros((Model_client.shape))
                                        E1[Model_client > tx] = 1
                                        FRR_temp.append(np.sum(E1)/(distModel1.shape[1]))


                                        E2 = np.zeros((Model_imposter.shape))
                                        E2[Model_imposter < tx] = 1
                                        FAR_temp.append(np.sum(E2)/distModel2.shape[1])


                                EER_temp = (perf.compute_eer(FAR_temp, FRR_temp))


                                acc = list()
                                f1 = list()
                                for _ in range(50):
                                    pos_samples = DF_positive_samples_test.shape
                                    temp = DF_negative_samples_test.sample(n = pos_samples[0])

                                    samples_test = np.concatenate((DF_positive_samples_test.iloc[:, :-2].values, temp.iloc[:, :-2].values),axis = 0)


                                    one = (np.ones((DF_positive_samples_test.iloc[:, -2:-1].values.shape)))
                                    zero = (np.zeros((temp.iloc[:, -2:-1].values.shape)))
                                    label_test = np.concatenate((one, zero),axis = 0)
                                    distModel1 , distModel2 = perf.compute_model(DF_positive_samples_train.iloc[:, :-2].values, samples_test, mode = mode, score = score)
                                    Model_client, Model_test = perf.model(distModel1, distModel2, model_type = model_type, score = score)

                                
                                    t_idx = EER_temp[1]
                                    
                                    y_pred = np.zeros((Model_test.shape))
                                    y_pred[Model_test > THRESHOLDs[t_idx]] = 1
                                    acc.append( accuracy_score(label_test, y_pred)*100 )
                                    f1.append(  f1_score(label_test, y_pred)*100 )
                                

                                if direction == "left_0":
                                    EER_L.append(EER_temp)
                                    FAR_L.append(FAR_temp)
                                    FRR_L.append(FRR_temp)
                                    ACC_L.append([subject, idx, np.mean(acc), np.mean(f1), DF_positive_samples_test.shape[0], temp.shape[0], DF_negative_samples_test.shape[0], test_ratio])

                                    
                                elif direction == "right_1":
                                    EER_R.append(EER_temp)
                                    FAR_R.append(FAR_temp)
                                    FRR_R.append(FRR_temp)
                                    ACC_R.append([subject, idx, np.mean(acc), np.mean(f1), DF_positive_samples_test.shape[0], temp.shape[0], DF_negative_samples_test.shape[0], test_ratio])


                        np.save(os.path.join(folder_path,   'EER_R.npy'), EER_R)
                        np.save(os.path.join(folder_path,   'FAR_R.npy'), FAR_R)
                        np.save(os.path.join(folder_path,   'FRR_R.npy'), FRR_R)


                        np.save(os.path.join(folder_path,   'EER_L.npy'), EER_L)
                        np.save(os.path.join(folder_path,   'FAR_L.npy'), FAR_L)
                        np.save(os.path.join(folder_path,   'FRR_L.npy'), FRR_L)


                        np.save(os.path.join(folder_path,   'ACC_L.npy'), ACC_L)
                        np.save(os.path.join(folder_path,   'ACC_R.npy'), ACC_R)



                        
                        A = [[mode, model_type, test_ratio, normilizing, feat_name, persentage, 
                        np.mean( np.array(ACC_L)[:,2] ), 
                        np.mean( np.array(EER_L)[:,0] ),
                        np.mean( np.array(ACC_R)[:,2] ),
                        np.mean( np.array(EER_R)[:,0] ),

                        np.min( np.array(ACC_L)[:,2] ),
                        np.min( np.array(EER_L)[:,0] ),
                        np.min( np.array(ACC_R)[:,2] ),
                        np.min( np.array(EER_R)[:,0] ),

                        np.max( np.array(ACC_L)[:,2] ),
                        np.max( np.array(EER_L)[:,0] ),
                        np.max( np.array(ACC_R)[:,2] ),
                        np.max( np.array(EER_R)[:,0] ),

                        np.median( np.array(ACC_L)[:,2] ),
                        np.median( np.array(EER_L)[:,0] ),
                        np.median( np.array(ACC_R)[:,2] ),
                        np.median( np.array(EER_R)[:,0] )] +
                        np.concatenate((np.mean(np.array(FAR_L), axis=0), np.mean(np.array(FRR_L), axis=0)), axis=0).tolist()+
                        np.concatenate((np.mean(np.array(FAR_R), axis=0), np.mean(np.array(FRR_R), axis=0)), axis=0).tolist()]



                        z = pd.DataFrame(A, columns = cols )

                        Results_DF = Results_DF.append(z)

                        index = index + 1
                        print("[INFO] ------ stage {:} of 1080".format(index))

                        Results_DF.to_excel(os.path.join(working_path, 'results', 'Results_DF.xlsx'))
                        toc=timeit.default_timer()
                        print("[INFO] Process time: ", toc - tic) #elapsed time in seconds
                        sys.exit()
toc=timeit.default_timer()
print("[INFO] Process time: ", toc - tic) #elapsed time in seconds

print(Results_DF.head(  ))                       
print("[INFO] Done!!!")

