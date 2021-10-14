import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os

from scipy.spatial import distance

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def PCA_func(DF_features, persentage ):
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


    columnsName = ["PC"+str(i) for i in list(range(1, num_pc+1))] + ["subject ID", "left(0)/right(1)"]
    DF_features_PCA = (pd.DataFrame(np.concatenate((PCA_out[:,:num_pc],DF_features.iloc[:, -2:].values), axis = 1), columns = columnsName))

    return DF_features_PCA





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


def performance(model1, model2, path):

    # THRESHOLDs = np.linspace(0, 2*np.max(model1), 10)
    THRESHOLDs = np.linspace(0, 300, 1000)
    FN = list();   TP = list();  TN = list();  FP = list()
    ACC = list(); FDR = list(); FNR = list(); FPR = list()
    NPV = list(); PPV = list(); TNR = list(); TPR = list()

    for idx, thresh in enumerate(THRESHOLDs):
        TPM = np.zeros((model1.shape))
        TPM[model1 < thresh] = 1
        TP.append(TPM.sum()/16)
        

        FNM = np.zeros((model1.shape))
        FNM[model1 >= thresh] = 1
        FN.append(FNM.sum()/16)

        FPM = np.zeros((model2.shape))
        FPM[model2 < thresh] = 1
        FP.append(FPM.sum()/16)

        TNM = np.zeros((model2.shape))
        TNM[model2 >= thresh] = 1
        TN.append(TNM.sum()/16)

        # Sensitivity, hit rate, recall, or true positive rate
        # reflects the classifier’s ability to detect members of the positive class (pathological state)
        TPR.append(TP[idx] / (TP[idx]  + FN[idx] ))
        # Specificity or true negative rate
        # reflects the classifier’s ability to detect members of the negative class (normal state)
        TNR.append(TN[idx]  / (TN[idx]  + FP[idx] ))
        # Precision or positive predictive value
        # PPV.append(TP[idx]  / (TP[idx]  + FP[idx] ))
        # Negative predictive value
        # NPV.append(TN[idx]  / (TN[idx]  + FN[idx] ))
        # Fall out or false positive rate
        # reflects the frequency with which the classifier makes a mistake by classifying normal state as pathological
        FPR.append(FP[idx]  / (FP[idx]  + TN[idx] ))
        # False negative rate
        # reflects the frequency with which the classifier makes a mistake by classifying pathological state as normal
        FNR.append(FN[idx]  / (TP[idx]  + FN[idx] ))
        # False discovery rate
        # FDR.append(FP[idx]  / (TP[idx]  + FP[idx] ))
        # Overall accuracy
        ACC.append((TP[idx]  + TN[idx] ) / (TP[idx]  + FP[idx]  + FN[idx]  + TN[idx] ))

    EER, minindex = compute_eer(FPR, FNR)



    if False:
        # print("\n#################################################################################################")
        # print("#################################################################################################\n")
        # print("THRESHOLDs:                                                                          {}".format(THRESHOLDs))
        # print("EER:                                                                                 {}".format(EER))
        # print("False Positive (FP):                                                                 {}".format(FP))
        # print("False Negative (FN):                                                                 {}".format(FN))
        # print("True Positive (TP):                                                                  {}".format(TP))
        # print("True Negative (TN):                                                                  {}".format(TN))
        # print("True Positive Rate (TPR)(Recall):                                                    {}".format(TPR))
        # print("True Negative Rate (TNR)(Specificity):                                               {}".format(TNR))
        # print("Positive Predictive Value (PPV)(Precision):                                          {}".format(PPV))
        # print("Negative Predictive Value (NPV):                                                     {}".format(NPV))
        # print(
        #      "False Positive Rate (FPR)(False Match Rate (FMR))(False Acceptance Rate (FAR)):      {}".format(
        #         FPR
        #     )
        # )
        # print(
        #      "False Negative Rate (FNR)(False Non-Match Rate (FNMR))(False Rejection Rate (FRR)):  {}".format(
        #         FNR
        #     )
        # )
        # print("False Discovery Rate (FDR):                                                          {}".format(FDR))
        # print("Overall accuracy (ACC):                                                              {}".format(ACC))
        pass
    if False:
        print("\n#################################################################################################")
        print("\n#################################################################################################")
        print("#################################################################################################\n")
        print("THRESHOLDs:                                                                          {}".format(THRESHOLDs[minindex]))
        print("EER:                                                                                 {}".format(EER))
        print("False Positive (FP):                                                                 {}".format(FP[minindex]))
        print("False Negative (FN):                                                                 {}".format(FN[minindex]))
        print("True Positive (TP):                                                                  {}".format(TP[minindex]))
        print("True Negative (TN):                                                                  {}".format(TN[minindex]))
        print("True Positive Rate (TPR)(Recall):                                                    {}".format(TPR[minindex]))
        print("True Negative Rate (TNR)(Specificity):                                               {}".format(TNR[minindex]))
        print("Positive Predictive Value (PPV)(Precision):                                          {}".format(PPV[minindex]))
        print("Negative Predictive Value (NPV):                                                     {}".format(NPV[minindex]))
        print(
            "False Positive Rate (FPR)(False Match Rate (FMR))(False Acceptance Rate (FAR)):      {}".format(
                FPR[minindex]
            )
        )
        print(
            "False Negative Rate (FNR)(False Non-Match Rate (FNMR))(False Rejection Rate (FRR)):  {}".format(
                FNR[minindex]
            )
        )
        print("False Discovery Rate (FDR):                                                          {}".format(FDR[minindex]))
        print("Overall accuracy (ACC):                                                              {}".format(ACC[minindex]))
        print("\n#################################################################################################")

    with open(os.path.join(path, "file.txt"), "a") as f:
        f.write("\n#################################################################################################")
        f.write("\n#################################################################################################")
        f.write("\n#################################################################################################\n")
        f.write("\nTHRESHOLDs:                                                                          {}".format(THRESHOLDs[minindex]))
        f.write("\nEER:                                                                                 {}".format(EER))
        f.write("\nFalse Positive (FP):                                                                 {}".format(FP[minindex]))
        f.write("\nFalse Negative (FN):                                                                 {}".format(FN[minindex]))
        f.write("\nTrue Positive (TP):                                                                  {}".format(TP[minindex]))
        f.write("\nTrue Negative (TN):                                                                  {}".format(TN[minindex]))
        f.write("\nTrue Positive Rate (TPR)(Recall):                                                    {}".format(TPR[minindex]))
        f.write("\nTrue Negative Rate (TNR)(Specificity):                                               {}".format(TNR[minindex]))
        # f.write("\nPositive Predictive Value (PPV)(Precision):                                          {}".format(PPV[minindex]))
        # f.write("\nNegative Predictive Value (NPV):                                                     {}".format(NPV[minindex]))
        f.write(
            "\nFalse Positive Rate (FPR)(False Match Rate (FMR))(False Acceptance Rate (FAR)):      {}".format(
                FPR[minindex]
            )
        )
        f.write(
            "\nFalse Negative Rate (FNR)(False Non-Match Rate (FNMR))(False Rejection Rate (FRR)):  {}".format(
                FNR[minindex]
            )
        )
        # f.write("\nFalse Discovery Rate (FDR):                                                          {}".format(FDR[minindex]))
        f.write("\nOverall accuracy (ACC):                                                              {}".format(ACC[minindex]))
        f.write("\n#################################################################################################")
    ROC_plot(TPR, FPR, path)
    ROC_plot_v2(FPR, FNR, THRESHOLDs, path)
    return EER, FPR, FNR


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