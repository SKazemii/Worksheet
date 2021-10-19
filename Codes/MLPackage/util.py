import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys, os, timeit

from scipy.spatial import distance

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score

from pathlib import Path as Pathlb

THRESHOLDs = np.linspace(0, 1, 100)
test_ratios = [0.2]
persentages = [1.0, 0.95]
modes = ["corr", "dist"]
model_types = ["min", "median", "average"]
normilizings = ["z-score", "minmax"]
verbose = False

features_types = ["afeatures-simple", "afeatures-otsu", "pfeatures", "COAs-otsu", "COAs-simple", "COPs"]
color = ['darkorange', 'navy', 'red', 'greenyellow', 'lightsteelblue', 'lightcoral', 'olive', 'mediumpurple', 'khaki', 'hotpink', 'blueviolet']


working_path = os.getcwd()
score = "A"
feature_names = ["MDIST", "RDIST", "TOTEX", "MVELO", "RANGE", "AREAXX", "MFREQ", "FDPD", "FDCX"]
cols = ["Feature_Type", "Mode", "Model_Type", "Test_Size", "Normalizition", "Features_Set", "PCA", "Time", "Number_of_PCs",
        "Mean_sample_test_L", "Mean_Acc_L", "Mean_f1_L", "Mean_EER_L", 
        "Mean_sample_test_R","Mean_Acc_R", "Mean_f1_R", "Mean_EER_R",

        "std_sample_test_L", "std_Acc_L", "std_f1_L", "std_EER_L", 
        "std_sample_test_R","std_Acc_R", "std_f1_R", "std_EER_R",

        "Min_sample_test_L", "Min_Acc_L", "Min_f1_L", "Min_EER_L", 
        "Min_sample_test_R","Min_Acc_R", "Min_f1_R", "Min_EER_R",

        "Max_sample_test_L", "Max_Acc_L", "Max_f1_L", "Max_EER_L", 
        "Max_sample_test_R", "Max_Acc_R", "Max_f1_R", "Max_EER_R"] + ["FAR_L_" + str(i) for i in range(100)] + ["FRR_L_" + str(i) for i in range(100)] + ["FAR_R_" + str(i) for i in range(100)] + ["FRR_R_" + str(i) for i in range(100)]



def fcn(DF_features_all, foldername, features_excel):
    
    subjects = (DF_features_all["subject ID"].unique())
    
    persentage = float(foldername.split('_')[0])
    normilizing = foldername.split('_')[1]
    x = int(foldername.split('_')[2])
    mode = foldername.split('_')[3]  
    model_type = foldername.split('_')[4]
    test_ratio = float(foldername.split('_')[5])

    if x == -3:
        DF_features = DF_features_all.copy()
        feat_name = "All"
    else:
        DF_features = DF_features_all.copy()
        DF_features.drop(DF_features.columns[[range(x+3,DF_features_all.shape[1]-2)]], axis = 1, inplace = True)
        DF_features.drop(DF_features.columns[[range(0,x)]], axis = 1, inplace = True)
        feat_name = feature_names[int(x/3)]


    tic=timeit.default_timer()
    folder = str(persentage) + "_" + normilizing + "_" + feat_name + "_" + mode + "_" + model_type + "_" +  str(test_ratio) 
    folder_path = os.path.join(working_path, 'results', features_excel, folder)
    Pathlb(folder_path).mkdir(parents=True, exist_ok=True)

    EER_L = list(); FAR_L = list(); FRR_L = list()
    EER_R = list(); FAR_R = list(); FRR_R = list()


    ACC_L = list(); ACC_R = list()
    


    print("[INFO] start:   +++   ", folder)

    for subject in subjects:
        if (subject % 86) == 0:
            continue
        
        if (subject % 30) == 0 and verbose is True:
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
            
            
            df_train = pd.concat([DF_positive_samples_train, DF_negative_samples_train])
            df_test = pd.concat([DF_positive_samples_test, DF_negative_samples_test])

            if normilizing == "minmax":
                scaling = preprocessing.MinMaxScaler()
                Scaled_train = scaling.fit_transform(df_train.iloc[:, :-2])
                Scaled_test = scaling.transform(df_test.iloc[:, :-2])


            elif normilizing == "z-score":
                scaling = preprocessing.StandardScaler()
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



            distModel1, distModel2 = compute_model(DF_positive_samples_train.iloc[:, :-2].values,
                                                        DF_negative_samples_train.iloc[:, :-2].values,
                                                        mode = mode, score = score)


            Model_client, Model_imposter = model(distModel1,
                                                        distModel2, 
                                                        model_type = model_type, 
                                                        score = score )


            FRR_temp = list()
            FAR_temp = list()

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


            EER_temp = (compute_eer(FAR_temp, FRR_temp))


            acc = list()
            f1 = list()
            for _ in range(50):
                pos_samples = DF_positive_samples_test.shape
                temp = DF_negative_samples_test.sample(n = pos_samples[0])

                samples_test = np.concatenate((DF_positive_samples_test.iloc[:, :-2].values, temp.iloc[:, :-2].values),axis = 0)


                one = (np.ones((DF_positive_samples_test.iloc[:, -2:-1].values.shape)))
                zero = (np.zeros((temp.iloc[:, -2:-1].values.shape)))
                label_test = np.concatenate((one, zero),axis = 0)
                distModel1 , distModel2 = compute_model(DF_positive_samples_train.iloc[:, :-2].values, samples_test, mode = mode, score = score)
                Model_client, Model_test = model(distModel1, distModel2, model_type = model_type, score = score)

            
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


    toc=timeit.default_timer()
    print("[INFO] End:     ---    {}, \t\t Process time: {:.2f}  seconds".format(folder, toc - tic)) 



    A = [[features_excel, mode, model_type, test_ratio, normilizing, feat_name, persentage, (toc - tic), num_pc,

        np.mean( np.array(ACC_L)[:,4] ), 
        np.mean( np.array(ACC_L)[:,2] ), 
        np.mean( np.array(ACC_L)[:,3] ), 
        np.mean( np.array(EER_L)[:,0] ),
        np.mean( np.array(ACC_R)[:,4] ), 
        np.mean( np.array(ACC_R)[:,2] ),
        np.mean( np.array(ACC_R)[:,3] ), 
        np.mean( np.array(EER_R)[:,0] ),

        np.std( np.array(ACC_L)[:,4] ), 
        np.std( np.array(ACC_L)[:,2] ),
        np.std( np.array(ACC_L)[:,3] ), 
        np.std( np.array(EER_L)[:,0] ),
        np.std( np.array(ACC_R)[:,4] ), 
        np.std( np.array(ACC_R)[:,2] ),
        np.std( np.array(ACC_R)[:,3] ), 
        np.std( np.array(EER_R)[:,0] ),

        np.min( np.array(ACC_L)[:,4] ), 
        np.min( np.array(ACC_L)[:,2] ),
        np.min( np.array(ACC_L)[:,3] ), 
        np.min( np.array(EER_L)[:,0] ),
        np.min( np.array(ACC_R)[:,4] ), 
        np.min( np.array(ACC_R)[:,2] ),
        np.min( np.array(ACC_R)[:,3] ), 
        np.min( np.array(EER_R)[:,0] ),


        np.max( np.array(ACC_L)[:,4] ), 
        np.max( np.array(ACC_L)[:,2] ),
        np.max( np.array(ACC_L)[:,3] ), 
        np.max( np.array(EER_L)[:,0] ),
        np.max( np.array(ACC_R)[:,4] ), 
        np.max( np.array(ACC_R)[:,2] ),
        np.max( np.array(ACC_R)[:,3] ), 
        np.max( np.array(EER_R)[:,0] )] +

        np.concatenate((np.mean(np.array(FAR_L), axis=0), np.mean(np.array(FRR_L), axis=0)), axis=0).tolist()+
        np.concatenate((np.mean(np.array(FAR_R), axis=0), np.mean(np.array(FRR_R), axis=0)), axis=0).tolist()]



    z = pd.DataFrame(A, columns = cols )
    return z


def plot(FAR_L, FRR_L, FAR_R, FRR_R, labels):
    for idx in range(len(FAR_L)):
        plt.subplot(1,2,1)
        auc = (1 + np.trapz( FRR_L[idx], FAR_L[idx]))
        # label=a[idx] #+ ' AUC = ' + str(round(auc, 2))

        plt.plot(FAR_L[idx], FRR_L[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=labels[idx] + str(auc), clip_on=False)

        plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Acceptance Rate')
        plt.ylabel('False Rejection Rate')
        plt.title('ROC curve, left side')
        plt.gca().set_aspect('equal')
        plt.legend(loc="best")

        plt.subplot(1,2,2)
        auc = (1 + np.trapz( FRR_R[idx], FAR_R[idx]))
        plt.plot(FAR_R[idx], FRR_R[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=labels[idx] + str(auc), clip_on=False)

        plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Acceptance Rate')
        plt.ylabel('False Rejection Rate')
        plt.title('ROC curve, Right side')
        plt.gca().set_aspect('equal')
        plt.legend(loc="best")


def model(distModel1, distModel2, model_type = "average", score = None ):
    if score is None:
        if model_type == "average":
            # model_client = (np.sum(distModel1, axis = 0))/(distModel1.shape[1]-1)
            model_client = np.mean(np.ma.masked_where(distModel1==0,distModel1), axis = 0)
            model_client = np.expand_dims(model_client,-1)
            
            model_imposter = (np.sum(distModel2, axis = 0))/(distModel1.shape[1])
            model_imposter = np.expand_dims(model_imposter, -1)
                
        elif model_type == "min":

            model_client = np.min(np.ma.masked_where(distModel1==0,distModel1), axis = 0)
            model_client = np.expand_dims(model_client,-1)
            
            model_imposter = np.min(np.ma.masked_where(distModel2==0,distModel2), axis = 0)
            model_imposter = np.expand_dims(model_imposter, -1)
                    
        elif model_type == "median":
            model_client = np.median(distModel1, axis = 0)
            model_client = np.expand_dims(model_client,-1)
            

            model_imposter = np.median(distModel2, axis = 0)
            model_imposter = np.expand_dims(model_imposter, -1)

    if score is not None:
        if model_type == "average":
            model_client = np.mean(np.ma.masked_where(distModel1==1,distModel1), axis = 0)
            model_client = np.expand_dims(model_client,-1)
            
            model_imposter = (np.sum(distModel2, axis = 0))/(distModel1.shape[1])
            model_imposter = np.expand_dims(model_imposter, -1)
                
        elif model_type == "min":

            model_client = np.max(np.ma.masked_where(distModel1==1,distModel1), axis = 0)
            model_client = np.expand_dims(model_client,-1)
            
            model_imposter = np.max(np.ma.masked_where(distModel2==1,distModel2), axis = 0)
            model_imposter = np.expand_dims(model_imposter, -1)
                    
        elif model_type == "median":
            model_client = np.median(distModel1, axis = 0)
            model_client = np.expand_dims(model_client,-1)            

            model_imposter = np.median(distModel2, axis = 0)
            model_imposter = np.expand_dims(model_imposter, -1)
    

    return model_client, model_imposter


def compute_score(distance, mode = "A"):
    distance = np.array(distance)

    if mode == "A":
        return np.power(distance+1, -1) 
    elif mode =="B":
        return 1/np.exp(distance)


def compute_eer(fpr, fnr):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    abs_diffs = np.abs(np.subtract(fpr, fnr)) 
    mmin = min(abs_diffs)   
    idxs = np.where(abs_diffs == mmin)

    min_index = np.max(idxs)#np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))

    return eer, min_index


def ROC_plot(TPR, FPR):
    """plot ROC curve"""
    plt.figure()
    auc = 1 * np.trapz(TPR, FPR)

    plt.plot(FPR, TPR, linestyle='--', marker='o', color='darkorange', lw = 2, label='ROC curve', clip_on=False)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve, AUC = %.2f'%auc)
    plt.legend(loc="lower right")
    # plt.savefig(path + 'AUC.png')


def ROC_plot_v2(FPR, FNR,THRESHOLDs, path):
    """plot ROC curve"""
    # fig = plt.figure()
    color = ['darkorange', 'orange']
    auc = 1/(1 + np.trapz( FPR,FNR))
    plt.plot(FPR, FNR, linestyle='--', marker='o', color=color[path], lw = 2, label='ROC curve', clip_on=False)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, AUC = %.2f'%auc)
    plt.legend(loc="best")
    # path1 = path + "_ROC.png"

    # plt.savefig(path1)

    # plt.figure()
    # plt.plot(THRESHOLDs, FPR, linestyle='--', marker='o', color='darkorange', lw = 2, label='FAR curve', clip_on=False)
    # plt.plot(THRESHOLDs, FNR, linestyle='--', marker='o', color='navy', lw = 2, label='FRR curve', clip_on=False)

    # EER,_ = compute_eer(FPR, FNR)
    # path2 = path + "_ACC.png"
    # plt.title('FPR and FNR curve, EER = %.2f'%EER)
    # plt.legend(loc="upper right")
    # plt.xlabel('Threshold')
    # plt.savefig(path2)
    # plt.close('all')


# def performance(model1, model2, path):

#     # THRESHOLDs = np.linspace(0, 2*np.max(model1), 10)
#     THRESHOLDs = np.linspace(0, 300, 1000)
#     FN = list();   TP = list();  TN = list();  FP = list()
#     ACC = list(); FDR = list(); FNR = list(); FPR = list()
#     NPV = list(); PPV = list(); TNR = list(); TPR = list()

#     for idx, thresh in enumerate(THRESHOLDs):
#         TPM = np.zeros((model1.shape))
#         TPM[model1 < thresh] = 1
#         TP.append(TPM.sum()/16)
        

#         FNM = np.zeros((model1.shape))
#         FNM[model1 >= thresh] = 1
#         FN.append(FNM.sum()/16)

#         FPM = np.zeros((model2.shape))
#         FPM[model2 < thresh] = 1
#         FP.append(FPM.sum()/16)

#         TNM = np.zeros((model2.shape))
#         TNM[model2 >= thresh] = 1
#         TN.append(TNM.sum()/16)

#         # Sensitivity, hit rate, recall, or true positive rate
#         # reflects the classifier’s ability to detect members of the positive class (pathological state)
#         TPR.append(TP[idx] / (TP[idx]  + FN[idx] ))
#         # Specificity or true negative rate
#         # reflects the classifier’s ability to detect members of the negative class (normal state)
#         TNR.append(TN[idx]  / (TN[idx]  + FP[idx] ))
#         # Precision or positive predictive value
#         # PPV.append(TP[idx]  / (TP[idx]  + FP[idx] ))
#         # Negative predictive value
#         # NPV.append(TN[idx]  / (TN[idx]  + FN[idx] ))
#         # Fall out or false positive rate
#         # reflects the frequency with which the classifier makes a mistake by classifying normal state as pathological
#         FPR.append(FP[idx]  / (FP[idx]  + TN[idx] ))
#         # False negative rate
#         # reflects the frequency with which the classifier makes a mistake by classifying pathological state as normal
#         FNR.append(FN[idx]  / (TP[idx]  + FN[idx] ))
#         # False discovery rate
#         # FDR.append(FP[idx]  / (TP[idx]  + FP[idx] ))
#         # Overall accuracy
#         ACC.append((TP[idx]  + TN[idx] ) / (TP[idx]  + FP[idx]  + FN[idx]  + TN[idx] ))

#     EER, minindex = compute_eer(FPR, FNR)



#     if False:
#         # print("\n#################################################################################################")
#         # print("#################################################################################################\n")
#         # print("THRESHOLDs:                                                                          {}".format(THRESHOLDs))
#         # print("EER:                                                                                 {}".format(EER))
#         # print("False Positive (FP):                                                                 {}".format(FP))
#         # print("False Negative (FN):                                                                 {}".format(FN))
#         # print("True Positive (TP):                                                                  {}".format(TP))
#         # print("True Negative (TN):                                                                  {}".format(TN))
#         # print("True Positive Rate (TPR)(Recall):                                                    {}".format(TPR))
#         # print("True Negative Rate (TNR)(Specificity):                                               {}".format(TNR))
#         # print("Positive Predictive Value (PPV)(Precision):                                          {}".format(PPV))
#         # print("Negative Predictive Value (NPV):                                                     {}".format(NPV))
#         # print(
#         #      "False Positive Rate (FPR)(False Match Rate (FMR))(False Acceptance Rate (FAR)):      {}".format(
#         #         FPR
#         #     )
#         # )
#         # print(
#         #      "False Negative Rate (FNR)(False Non-Match Rate (FNMR))(False Rejection Rate (FRR)):  {}".format(
#         #         FNR
#         #     )
#         # )
#         # print("False Discovery Rate (FDR):                                                          {}".format(FDR))
#         # print("Overall accuracy (ACC):                                                              {}".format(ACC))
#         pass
#     if False:
#         print("\n#################################################################################################")
#         print("\n#################################################################################################")
#         print("#################################################################################################\n")
#         print("THRESHOLDs:                                                                          {}".format(THRESHOLDs[minindex]))
#         print("EER:                                                                                 {}".format(EER))
#         print("False Positive (FP):                                                                 {}".format(FP[minindex]))
#         print("False Negative (FN):                                                                 {}".format(FN[minindex]))
#         print("True Positive (TP):                                                                  {}".format(TP[minindex]))
#         print("True Negative (TN):                                                                  {}".format(TN[minindex]))
#         print("True Positive Rate (TPR)(Recall):                                                    {}".format(TPR[minindex]))
#         print("True Negative Rate (TNR)(Specificity):                                               {}".format(TNR[minindex]))
#         print("Positive Predictive Value (PPV)(Precision):                                          {}".format(PPV[minindex]))
#         print("Negative Predictive Value (NPV):                                                     {}".format(NPV[minindex]))
#         print(
#             "False Positive Rate (FPR)(False Match Rate (FMR))(False Acceptance Rate (FAR)):      {}".format(
#                 FPR[minindex]
#             )
#         )
#         print(
#             "False Negative Rate (FNR)(False Non-Match Rate (FNMR))(False Rejection Rate (FRR)):  {}".format(
#                 FNR[minindex]
#             )
#         )
#         print("False Discovery Rate (FDR):                                                          {}".format(FDR[minindex]))
#         print("Overall accuracy (ACC):                                                              {}".format(ACC[minindex]))
#         print("\n#################################################################################################")

#     with open(os.path.join(path, "file.txt"), "a") as f:
#         f.write("\n#################################################################################################")
#         f.write("\n#################################################################################################")
#         f.write("\n#################################################################################################\n")
#         f.write("\nTHRESHOLDs:                                                                          {}".format(THRESHOLDs[minindex]))
#         f.write("\nEER:                                                                                 {}".format(EER))
#         f.write("\nFalse Positive (FP):                                                                 {}".format(FP[minindex]))
#         f.write("\nFalse Negative (FN):                                                                 {}".format(FN[minindex]))
#         f.write("\nTrue Positive (TP):                                                                  {}".format(TP[minindex]))
#         f.write("\nTrue Negative (TN):                                                                  {}".format(TN[minindex]))
#         f.write("\nTrue Positive Rate (TPR)(Recall):                                                    {}".format(TPR[minindex]))
#         f.write("\nTrue Negative Rate (TNR)(Specificity):                                               {}".format(TNR[minindex]))
#         # f.write("\nPositive Predictive Value (PPV)(Precision):                                          {}".format(PPV[minindex]))
#         # f.write("\nNegative Predictive Value (NPV):                                                     {}".format(NPV[minindex]))
#         f.write(
#             "\nFalse Positive Rate (FPR)(False Match Rate (FMR))(False Acceptance Rate (FAR)):      {}".format(
#                 FPR[minindex]
#             )
#         )
#         f.write(
#             "\nFalse Negative Rate (FNR)(False Non-Match Rate (FNMR))(False Rejection Rate (FRR)):  {}".format(
#                 FNR[minindex]
#             )
#         )
#         # f.write("\nFalse Discovery Rate (FDR):                                                          {}".format(FDR[minindex]))
#         f.write("\nOverall accuracy (ACC):                                                              {}".format(ACC[minindex]))
#         f.write("\n#################################################################################################")
#     ROC_plot(TPR, FPR, path)
#     ROC_plot_v2(FPR, FNR, THRESHOLDs, path)
#     return EER, FPR, FNR


def compute_model(positive_samples, negative_samples, mode = "dist", score = None):

    positive_model = np.zeros((positive_samples.shape[0], positive_samples.shape[0]))
    negative_model = np.zeros((positive_samples.shape[0], negative_samples.shape[0]))

    if mode == "dist":

        for i in range(positive_samples.shape[0]):
            for j in range(positive_samples.shape[0]):
                # print(positive_samples.shape)
                # print(positive_samples.iloc[i, :])
                # print(positive_samples.iloc[i, :].values)
                positive_model[i, j] = distance.euclidean(
                    positive_samples[i, :], positive_samples[j, :]
                )
            for j in range(negative_samples.shape[0]):
                negative_model[i, j] = distance.euclidean(
                    positive_samples[i, :], negative_samples[j, :]
                )
        if score != None:
            return compute_score(positive_model, score), compute_score(negative_model, score)
        elif score == None:
            return positive_model, negative_model



    elif mode == "corr":

        for i in range(positive_samples.shape[0]):
            for j in range(positive_samples.shape[0]):
                positive_model[i, j] = abs(np.corrcoef(
                    positive_samples[i, :], positive_samples[j, :]
                )[0,1])

            for j in range(negative_samples.shape[0]):
                negative_model[i, j] = abs(np.corrcoef(
                    positive_samples[i, :], negative_samples[j, :]
                )[0,1])
        return positive_model, negative_model



    # np.save("./Datasets/distModel1.npy", distModel1)
    # np.save("./Datasets/distModel2.npy", distModel2)






def main():
    features = [[4.48283092],[3.26954198],
            [4.38550358],
            [3.63498196],
            [3.38680787],
            [3.38417218],
            [2.66482688],
            [2.49328531],
            [2.36398827],
            [2.83137372],
            [2.63417024],
            [2.99511643]]


    features = np.array(features)
    print(features)
    print(compute_score(features, mode = "B"))
    
    print("[INFO] Done!!!")



if __name__ == "__main__":
    main()