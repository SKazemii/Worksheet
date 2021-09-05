import matplotlib.pyplot as plt
import numpy as np
import os

def compute_eer(fpr, fnr):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    abs_diffs = np.abs(np.subtract(fpr, fnr))    
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, min_index


def ROC_plot(TPR, FPR, path):
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
    plt.savefig(path + 'AUC.png')


def ROC_plot_v2(FPR, FNR,THRESHOLDs, path):
    """plot ROC curve"""
    plt.figure()
    auc = 1 + np.trapz( FPR,FNR)
    plt.plot(FPR, FNR, linestyle='--', marker='o', color='darkorange', lw = 2, label='ROC curve', clip_on=False)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, AUC = %.2f'%auc)
    plt.legend(loc="lower right")
    plt.savefig(path + 'AUC-1.png')

    plt.figure()
    plt.plot(THRESHOLDs, FPR, linestyle='--', marker='o', color='darkorange', lw = 2, label='FAR curve', clip_on=False)
    plt.plot(THRESHOLDs, FNR, linestyle='--', marker='o', color='navy', lw = 2, label='FRR curve', clip_on=False)

    EER,_ = compute_eer(FPR, FNR)
    plt.title('FPR and FNR curve, EER = %.2f'%EER)
    plt.legend(loc="upper right")
    plt.xlabel('Threshold')
    plt.savefig(path +'AUC-2.png')


def performance(model1, model2, path):

    # THRESHOLDs = np.linspace(0, 2*np.max(model1), 10)
    THRESHOLDs = np.linspace(0, 300, 10)
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
        NPV.append(TN[idx]  / (TN[idx]  + FN[idx] ))
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
        f.write("\nNegative Predictive Value (NPV):                                                     {}".format(NPV[minindex]))
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