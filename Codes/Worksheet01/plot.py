import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, os
from pathlib import Path as Pathlb


# from scipy.spatial import distance
# from sklearn import preprocessing
# from sklearn.metrics import accuracy_score
# from itertools import combinations, product


from scipy.stats import shapiro, ttest_ind, mannwhitneyu


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MLPackage import util as perf
from MLPackage import FS 


plt.rcParams["font.size"] = 13
plt.tight_layout()

print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "Manuscripts", "src", "figures")
tbl_dir = os.path.join(project_dir, "Manuscripts", "src", "tables")
data_dir = os.path.join(project_dir, "results")

Pathlb(fig_dir).mkdir(parents=True, exist_ok=True)
Pathlb(tbl_dir).mkdir(parents=True, exist_ok=True)





THRESHOLDs = perf.THRESHOLDs
feature_names = perf.feature_names

color = ['darkorange', 'navy', 'red', 'greenyellow', 'lightsteelblue', 'lightcoral', 'olive', 'mediumpurple', 'khaki', 'hotpink', 'blueviolet']






Results_DF = pd.read_excel(os.path.join(data_dir, 'Results_DF.xlsx'), index_col = 0)
Results_DF["Mode"] = Results_DF["Mode"].map(lambda x: "Correlation" if x == "corr" else "Euclidean distance")
Results_DF["Normalizition"] = Results_DF["Normalizition"].map(lambda x: "Z-score algorithm" if x == "z-score" else "Minmax algorithm")
Results_DF["PCA"] = Results_DF["PCA"].map(lambda x: "All PCs" if x == 1.0 else "keeping 95% variance")
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "afeatures-simple" if x == "afeatures_simple" else x)
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "afeatures-otsu" if x == "afeatures_otsu" else x)
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "COAs-otsu" if x == "COAs_otsu" else x)
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "COAs-simple" if x == "COAs_simple" else x)
Results_DF.columns = perf.cols





Results_DF_temp = Results_DF[   Results_DF["Features_Set"] != "All"   ]



X = Results_DF_temp.sort_values(by=['Mean_Acc_L', 'Mean_EER_L'], ascending = [False, True]).iloc[:10,:13].drop(columns =['Time', 'Number_of_PCs', 'Mean_sample_test_L'])
with open(os.path.join("Manuscripts", "src", "tables", "top10-L.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())

X = Results_DF_temp.sort_values(by=['Mean_Acc_L', 'Mean_EER_L'], ascending = [True, False]).iloc[:10,:13].drop(columns =['Time', 'Number_of_PCs', 'Mean_sample_test_L'])
with open(os.path.join("Manuscripts", "src", "tables", "worse10-L.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())



X = Results_DF_temp.sort_values(by=['Mean_Acc_R', 'Mean_EER_R'], ascending = [False, True]).iloc[:10,:17].drop(columns =['Time', 'Number_of_PCs', 'Mean_f1_L', 'Mean_sample_test_R', 'Mean_sample_test_L', 'Mean_Acc_L', 'Mean_EER_L'])
with open(os.path.join("Manuscripts", "src", "tables", "top10-R.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())      

X = Results_DF_temp.sort_values(by=['Mean_Acc_R', 'Mean_EER_R'], ascending = [True, False]).iloc[:10,:17].drop(columns =['Time', 'Number_of_PCs', 'Mean_f1_L', 'Mean_sample_test_R', 'Mean_sample_test_L', 'Mean_Acc_L', 'Mean_EER_L'])
with open(os.path.join("Manuscripts", "src", "tables", "worse10-R.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())          






FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
plt.figure(figsize=(14,8))
X = pd.DataFrame(index=["COAs_otsu", "COAs_simple", "COPs"] , columns=["Accuracy Left", "Accuracy Right"])
Y = pd.DataFrame(index=["COAs_otsu", "COAs_simple", "COPs"] , columns=[ "EER Left", "EER Right"])
Results_DF_group = Results_DF.groupby(["Feature_Type"])

for f_type in ["COAs-otsu", "COAs-simple", "COPs"]:   
    
    DF = Results_DF_group.get_group((f_type))
    X.loc[f_type, "Accuracy Left"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_Acc_L"].mean(),  DF["Mean_Acc_L"].std(), DF["Mean_Acc_L"].min(), DF["Mean_Acc_L"].max())
    X.loc[f_type, "Accuracy Right"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_Acc_R"].mean(), DF["Mean_Acc_R"].std(), DF["Mean_Acc_R"].min(), DF["Mean_Acc_R"].max())
    Y.loc[f_type, "EER Left"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_EER_L"].mean(),       DF["Mean_EER_L"].std(), DF["Mean_EER_L"].min(), DF["Mean_EER_L"].max())
    Y.loc[f_type, "EER Right"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_EER_R"].mean(),      DF["Mean_EER_R"].std(), DF["Mean_EER_R"].min(), DF["Mean_EER_R"].max())    

    FAR_L.append(DF[["FAR_L_" + str(i) for i in range(100)]].mean().values)
    FRR_L.append(DF[["FRR_L_" + str(i) for i in range(100)]].mean().values)
    FAR_R.append(DF[["FAR_R_" + str(i) for i in range(100)]].mean().values)
    FRR_R.append(DF[["FRR_R_" + str(i) for i in range(100)]].mean().values)


with open(os.path.join("Manuscripts", "src", "tables", "COX-Acc.tex"), "w") as tf:
    tf.write(X.to_latex())
with open(os.path.join("Manuscripts", "src", "tables", "COX-EER.tex"), "w") as tf:
    tf.write(Y.to_latex())
perf.plot(FAR_L, FRR_L, FAR_R, FRR_R, ["COAs_otsu", "COAs_simple", "COPs"])
plt.tight_layout()
plt.savefig(os.path.join("Manuscripts", "src", "figures", "COX.png"))
plt.close('all')


for f_type in ["afeatures-simple", "afeatures-otsu", "pfeatures"]:   
    plt.figure(figsize=(14,8))

    Results_DF_group = Results_DF.groupby(["Feature_Type", "Features_Set"])
    values = Results_DF["Features_Set"].unique()

    X = pd.DataFrame(index=values , columns=["Accuracy Left", "Accuracy Right"])
    Y = pd.DataFrame(index=values , columns=[ "EER Left", "EER Right"])
    FAR_L = list()
    FRR_L = list()
    FAR_R = list()
    FRR_R = list()
    for value in values:
        
        DF = Results_DF_group.get_group((f_type, value))
        X.loc[value, "Accuracy Left"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_Acc_L"].mean(),  DF["Mean_Acc_L"].std(), DF["Mean_Acc_L"].min(), DF["Mean_Acc_L"].max())
        X.loc[value, "Accuracy Right"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_Acc_R"].mean(), DF["Mean_Acc_R"].std(), DF["Mean_Acc_R"].min(), DF["Mean_Acc_R"].max())
        Y.loc[value, "EER Left"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_EER_L"].mean(),       DF["Mean_EER_L"].std(), DF["Mean_EER_L"].min(), DF["Mean_EER_L"].max())
        Y.loc[value, "EER Right"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_EER_R"].mean(),      DF["Mean_EER_R"].std(), DF["Mean_EER_R"].min(), DF["Mean_EER_R"].max())    

        # print(DF)
        FAR_L.append(DF[["FAR_L_" + str(i) for i in range(100)]].mean().values)
        FRR_L.append(DF[["FRR_L_" + str(i) for i in range(100)]].mean().values)
        FAR_R.append(DF[["FAR_R_" + str(i) for i in range(100)]].mean().values)
        FRR_R.append(DF[["FRR_R_" + str(i) for i in range(100)]].mean().values)

    perf.plot(FAR_L, FRR_L, FAR_R, FRR_R, values)
    plt.tight_layout()
    plt.savefig(os.path.join("Manuscripts", "src", "figures", f_type + ".png"))
    plt.close('all')


    with open(os.path.join("Manuscripts", "src", "tables", f_type + "-Acc.tex"), "w") as tf:
        tf.write(X.to_latex())
    with open(os.path.join("Manuscripts", "src", "tables", f_type + "-EER.tex"), "w") as tf:
        tf.write(Y.to_latex())




for f_type in perf.features_types:   
    for column in ['Mode', 'Model_Type', 'Normalizition', 'PCA']:
        plt.figure(figsize=(14,8))
        Results_DF_group = Results_DF.groupby(["Feature_Type", "Features_Set", column])
        values = Results_DF[column].unique()
        X = pd.DataFrame(index=values , columns=["Accuracy Left", "Accuracy Right"])
        Y = pd.DataFrame(index=values , columns=[ "EER Left", "EER Right"])
        FAR_L = list()
        FRR_L = list()
        FAR_R = list()
        FRR_R = list()
        for value in values:
            
            DF = Results_DF_group.get_group((f_type, 'All', value))
            X.loc[value, "Accuracy Left"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_Acc_L"].mean(),  DF["Mean_Acc_L"].std(), DF["Mean_Acc_L"].min(), DF["Mean_Acc_L"].max())
            X.loc[value, "Accuracy Right"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_Acc_R"].mean(), DF["Mean_Acc_R"].std(), DF["Mean_Acc_R"].min(), DF["Mean_Acc_R"].max())
            Y.loc[value, "EER Left"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_EER_L"].mean(),       DF["Mean_EER_L"].std(), DF["Mean_EER_L"].min(), DF["Mean_EER_L"].max())
            Y.loc[value, "EER Right"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_EER_R"].mean(),      DF["Mean_EER_R"].std(), DF["Mean_EER_R"].min(), DF["Mean_EER_R"].max())    

            # print(DF)
            FAR_L.append(DF[["FAR_L_" + str(i) for i in range(100)]].mean().values)
            FRR_L.append(DF[["FRR_L_" + str(i) for i in range(100)]].mean().values)
            FAR_R.append(DF[["FAR_R_" + str(i) for i in range(100)]].mean().values)
            FRR_R.append(DF[["FRR_R_" + str(i) for i in range(100)]].mean().values)

        perf.plot(FAR_L, FRR_L, FAR_R, FRR_R, values)
        plt.tight_layout()
        plt.savefig(os.path.join("Manuscripts", "src", "figures", f_type + "-" + column + ".png"))
        plt.close('all')


        with open(os.path.join("Manuscripts", "src", "tables", f_type + "-" + column + "-Acc.tex"), "w") as tf:
            tf.write(X.to_latex())
        with open(os.path.join("Manuscripts", "src", "tables", f_type + "-" + column + "-EER.tex"), "w") as tf:
            tf.write(Y.to_latex())








for features_excel in ["afeatures_simple", "afeatures_otsu", "pfeatures"]:

    feature_path = os.path.join(perf.working_path, 'Datasets', features_excel + ".xlsx")
    DF_features = pd.read_excel(feature_path, index_col = 0)


    print( "[INFO] feature shape: ", DF_features.shape)


    f_names = ['MDIST_RD', 'MDIST_AP', 'MDIST_ML', 'RDIST_RD', 'RDIST_AP', 'RDIST_ML', 'TOTEX_RD', 'TOTEX_AP', 'TOTEX_ML', 'MVELO_RD', 'MVELO_AP', 'MVELO_ML', 'RANGE_RD', 'RANGE_AP', 'RANGE_ML','AREA_CC', 'AREA_CE', 'AREA_SW', 'MFREQ_RD', 'MFREQ_AP', 'MFREQ_ML', 'FDPD_RD', 'FDPD_AP', 'FDPD_ML', 'FDCC', 'FDCE']
    columnsName = f_names + [ "subject_ID", "left(0)/right(1)"]
    DF_features.columns = columnsName




    DF_side = DF_features[DF_features["left(0)/right(1)"] == 0]
    DF_side.loc[DF_side.subject_ID == 4.0, "left(0)/right(1)"] = 1
    DF_side.loc[DF_side.subject_ID != 4.0, "left(0)/right(1)"] = 0


    DF = FS.mRMR(DF_side.iloc[:,:-2], DF_side.iloc[:,-1])

    with open(os.path.join("Manuscripts", "src", "tables", features_excel + "-10best-FS.tex"), "w") as tf:
        tf.write(DF.iloc[:10,:].to_latex())
    with open(os.path.join("Manuscripts", "src", "tables", features_excel + "-10worst-FS.tex"), "w") as tf:
        tf.write(DF.iloc[-10:,:].to_latex())






















sys.exit()


Results_DF_all = Results_DF[   Results_DF["Features_Set"] == "All"   ]





auc=[]
plt.figure(figsize=(14,8))
FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
data = list()
X = pd.DataFrame(index=["Correlation", "Euclidean distance"] , columns=["Accuracy Left", "Accuracy Right"])
Y = pd.DataFrame(index=["Correlation", "Euclidean distance"] , columns=[ "EER Left", "EER Right"])
for idx, temp in enumerate(Modes):
    a = ["Correlation", "Euclidean distance"]
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Mode"] == temp   ]

    data.append(Results_DF_all_mode["Mean_Acc_L"].values)


    X.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Acc_L"].mean(),  Results_DF_all_mode["Mean_Acc_L"].std(), Results_DF_all_mode["Mean_Acc_L"].min(), Results_DF_all_mode["Mean_Acc_L"].max())
    X.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Acc_R"].mean(), Results_DF_all_mode["Mean_Acc_R"].std(), Results_DF_all_mode["Mean_Acc_R"].min(), Results_DF_all_mode["Mean_Acc_R"].max())
    Y.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_L"].mean(),       Results_DF_all_mode["Mean_EER_L"].std(), Results_DF_all_mode["Mean_EER_L"].min(), Results_DF_all_mode["Mean_EER_L"].max())
    Y.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_R"].mean(),      Results_DF_all_mode["Mean_EER_R"].std(), Results_DF_all_mode["Mean_EER_R"].min(), Results_DF_all_mode["Mean_EER_R"].max())    
       
    # plt.boxplot(Results_DF_all_mode["Mean_Acc_L"])
    # plt.show()

    cols = ["FAR_L_" + str(i) for i in range(100)] 
    FAR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_L_" + str(i) for i in range(100)]
    FRR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)

    plt.subplot(1,2,1)
    auc = (1 + np.trapz( FRR_L[idx], FAR_L[idx]))
    label=a[idx] #+ ' AUC = ' + str(round(auc, 2))

    plt.plot(FAR_L[idx], FRR_L[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)

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
    plt.plot(FAR_R[idx], FRR_R[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)



    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, Right side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")



stat, p = shapiro(data[1])
if p > 0.05:
    print('[INFO] Probably Gaussian,\t\t\t\t stat=%.3f,\t p=%.3f' % (stat, p))
    stat, p = ttest_ind(data[0], data[1])
    if p > 0.05:
        print('[INFO] Probably the same distribution,\t\t stat=%.3f,\t p=%.3f' % (stat, p))
    else:
        print('[INFO] Probably different distributions,\t\t stat=%.3f,\t p=%.3f' % (stat, p))
else:
    print('[INFO] Probably not Gaussian,\t\t\t\t stat=%.3f,\t p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(data[0], data[1])
    if p > 0.05:
        print('[INFO] Probably the same distribution,\t\t stat=%.3f,\t p=%.3f' % (stat, p))
    else:
        print('[INFO] Probably different distributions,\t\t stat=%.3f,\t p=%.3f' % (stat, p))

with open(os.path.join("Manuscripts", "src", "tables", "Correlation.tex"), "w") as tf:
        tf.write(X.round(decimals=2).to_latex())
with open(os.path.join("Manuscripts", "src", "tables", "Correlation1.tex"), "w") as tf:
        tf.write(Y.round(decimals=2).to_latex())
#########################################################################################################
#########################################################################################################
#########################################################################################################
plt.figure(figsize=(14,8))
X = pd.DataFrame(index=["Minimum", "Median", "Average"] , columns=["Accuracy Left", "Accuracy Right"])
Y = pd.DataFrame(index=["Minimum", "Median", "Average"] , columns=[ "EER Left", "EER Right"])

FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
data = list()
for idx, temp in enumerate(model_types):
    a = ["Minimum", "Median", "Average"]
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Model_Type"] == temp   ]
    data.append(Results_DF_all_mode["Mean_Acc_L"].values)

    X.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Acc_L"].mean(),  Results_DF_all_mode["Mean_Acc_L"].std(), Results_DF_all_mode["Mean_Acc_L"].min(), Results_DF_all_mode["Mean_Acc_L"].max())
    X.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Acc_R"].mean(), Results_DF_all_mode["Mean_Acc_R"].std(), Results_DF_all_mode["Mean_Acc_R"].min(), Results_DF_all_mode["Mean_Acc_R"].max())
    Y.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_L"].mean(),       Results_DF_all_mode["Mean_EER_L"].std(), Results_DF_all_mode["Mean_EER_L"].min(), Results_DF_all_mode["Mean_EER_L"].max())
    Y.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_R"].mean(),      Results_DF_all_mode["Mean_EER_R"].std(), Results_DF_all_mode["Mean_EER_R"].min(), Results_DF_all_mode["Mean_EER_R"].max())    

    cols = ["FAR_L_" + str(i) for i in range(100)] 
    FAR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_L_" + str(i) for i in range(100)]
    FRR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)




    auc = (1 + np.trapz( FRR_L[idx], FAR_L[idx]))
    plt.subplot(1,2,1)
    plt.plot(FAR_L[idx], FRR_L[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)
    
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

    plt.plot(FAR_R[idx], FRR_R[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, Right side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")
   




PATH = os.path.join("Manuscripts", "src", "figures", "Minimum.png")
plt.tight_layout()
plt.savefig(PATH )
plt.close('all')

with open(os.path.join("Manuscripts", "src", "tables", "Minimum.tex"), "w") as tf:
        tf.write(X.round(decimals=2).to_latex())
with open(os.path.join("Manuscripts", "src", "tables", "Minimum1.tex"), "w") as tf:
        tf.write(Y.round(decimals=2).to_latex())        
#########################################################################################################
#########################################################################################################
#########################################################################################################
plt.figure(figsize=(14,8))
X = pd.DataFrame(index=["Z-score algorithm", "MinMax algorithm", "None"] , columns=["Accuracy Left", "Accuracy Right"])
Y = pd.DataFrame(index=["Z-score algorithm", "MinMax algorithm", "None"] , columns=[ "EER Left", "EER Right"])

FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()   
for idx, temp in enumerate(normilizings):
    a = ["Z-score algorithm", "MinMax algorithm", "None"]


    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Normalizition"] == temp   ]

    X.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Acc_L"].mean(),  Results_DF_all_mode["Mean_Acc_L"].std(), Results_DF_all_mode["Mean_Acc_L"].min(), Results_DF_all_mode["Mean_Acc_L"].max())
    X.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Acc_R"].mean(), Results_DF_all_mode["Mean_Acc_R"].std(), Results_DF_all_mode["Mean_Acc_R"].min(), Results_DF_all_mode["Mean_Acc_R"].max())
    Y.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_L"].mean(),       Results_DF_all_mode["Mean_EER_L"].std(), Results_DF_all_mode["Mean_EER_L"].min(), Results_DF_all_mode["Mean_EER_L"].max())
    Y.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_R"].mean(),      Results_DF_all_mode["Mean_EER_R"].std(), Results_DF_all_mode["Mean_EER_R"].min(), Results_DF_all_mode["Mean_EER_R"].max())    

    
    
    cols = ["FAR_L_" + str(i) for i in range(100)] 
    FAR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_L_" + str(i) for i in range(100)]
    FRR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)



    plt.subplot(1,2,1)
    auc = (1 + np.trapz( FRR_L[idx], FAR_L[idx]))

    plt.plot(FAR_L[idx], FRR_L[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)

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

    plt.plot(FAR_R[idx], FRR_R[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)



    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, Right side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")

PATH = os.path.join("Manuscripts", "src", "figures", "MinMax.png")
plt.tight_layout()
plt.savefig(PATH )
plt.close('all')

with open(os.path.join("Manuscripts", "src", "tables", "MinMax.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())
with open(os.path.join("Manuscripts", "src", "tables", "MinMax1.tex"), "w") as tf:
    tf.write(Y.round(decimals=2).to_latex())
#########################################################################################################
#########################################################################################################
#########################################################################################################
a = ["10 percent", "20 percent", "25 percent", "30 percent", "40 percent", "50 percent", "60 percent","70 percent", "75 percent", "80 percent" , "90 percent"]

plt.figure(figsize=(14,8))
X = pd.DataFrame(index=a , columns=["Accuracy Left", "Accuracy Right"])
Y = pd.DataFrame(index=a , columns=[ "EER Left", "EER Right"])
Z = pd.DataFrame(index=a , columns=[ "Left", "Right"])

FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
for idx, temp in enumerate(test_ratios):
    a = ["10 percent", "20 percent", "25 percent", "30 percent", "40 percent", "50 percent", "60 percent","70 percent", "75 percent", "80 percent" , "90 percent"]
  
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Test_Size"] == temp   ].reset_index()

    X.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Acc_L"].mean(),  Results_DF_all_mode["Mean_Acc_L"].std(), Results_DF_all_mode["Mean_Acc_L"].min(), Results_DF_all_mode["Mean_Acc_L"].max())
    X.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Acc_R"].mean(), Results_DF_all_mode["Mean_Acc_R"].std(), Results_DF_all_mode["Mean_Acc_R"].min(), Results_DF_all_mode["Mean_Acc_R"].max())
    Y.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_L"].mean(),       Results_DF_all_mode["Mean_EER_L"].std(), Results_DF_all_mode["Mean_EER_L"].min(), Results_DF_all_mode["Mean_EER_L"].max())
    Y.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_R"].mean(),      Results_DF_all_mode["Mean_EER_R"].std(), Results_DF_all_mode["Mean_EER_R"].min(), Results_DF_all_mode["Mean_EER_R"].max())    

    print(        Results_DF_all_mode.at[0, "Mean_sample_test_Left"])
    Z.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:2.2f}, {:2.2f})".format(
        Results_DF_all_mode.at[0, "Mean_sample_test_Left"], 
        Results_DF_all_mode.at[0, "std_sample_test_Left"],
        Results_DF_all_mode.at[0, "Min_sample_test_Left"],
        Results_DF_all_mode.at[0, "Max_sample_test_Left"])

    Z.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:2.2f}, {:2.2f})".format(
        Results_DF_all_mode.at[0, "Mean_sample_test_Right"], 
        Results_DF_all_mode.at[0, "std_sample_test_Right"],
        Results_DF_all_mode.at[0, "Min_sample_test_Right"],
        Results_DF_all_mode.at[0, "Max_sample_test_Right"])



    cols = ["FAR_L_" + str(i) for i in range(100)] 
    FAR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_L_" + str(i) for i in range(100)]
    FRR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)




    plt.subplot(1,2,1)
    auc = (1 + np.trapz( FRR_L[idx], FAR_L[idx]))

    plt.plot(FAR_L[idx], FRR_L[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)

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

    plt.plot(FAR_R[idx], FRR_R[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)



    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, Right side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")


PATH = os.path.join("Manuscripts", "src", "figures",  "testsize.png")
plt.tight_layout()
plt.savefig(PATH)
plt.close('all')
with open(os.path.join("Manuscripts", "src", "tables", "testsize.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())
with open(os.path.join("Manuscripts", "src", "tables", "testsize1.tex"), "w") as tf:
    tf.write(Y.round(decimals=2).to_latex())
with open(os.path.join("Manuscripts", "src", "tables", "testsize2.tex"), "w") as tf:
    tf.write(Z.round(decimals=2).to_latex())

#########################################################################################################
#########################################################################################################
#########################################################################################################
FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
plt.figure(figsize=(14,8))
X = pd.DataFrame(index=["All PCs", "Keeping 95 percent of variance"] , columns=["Accuracy Left", "Accuracy Right"])
Y = pd.DataFrame(index=["All PCs", "Keeping 95 percent of variance"] , columns=[ "EER Left", "EER Right"])

for idx, temp in enumerate(persentages):
    a = ["All PCs", "Keeping 95 percent of variance"]
  
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["PCA"] == temp   ]

    X.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Acc_L"].mean(),  Results_DF_all_mode["Mean_Acc_L"].std(), Results_DF_all_mode["Mean_Acc_L"].min(), Results_DF_all_mode["Mean_Acc_L"].max())
    X.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Acc_R"].mean(), Results_DF_all_mode["Mean_Acc_R"].std(), Results_DF_all_mode["Mean_Acc_R"].min(), Results_DF_all_mode["Mean_Acc_R"].max())
    Y.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_L"].mean(),       Results_DF_all_mode["Mean_EER_L"].std(), Results_DF_all_mode["Mean_EER_L"].min(), Results_DF_all_mode["Mean_EER_L"].max())
    Y.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_R"].mean(),      Results_DF_all_mode["Mean_EER_R"].std(), Results_DF_all_mode["Mean_EER_R"].min(), Results_DF_all_mode["Mean_EER_R"].max())    



    cols = ["FAR_L_" + str(i) for i in range(100)] 
    FAR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_L_" + str(i) for i in range(100)]
    FRR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)


    plt.subplot(1,2,1)
    auc = (1 + np.trapz( FRR_L[idx], FAR_L[idx]))

    plt.plot(FAR_L[idx], FRR_L[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)

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

    plt.plot(FAR_R[idx], FRR_R[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)



    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, Right side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")


PATH = os.path.join("Manuscripts", "src", "figures",  "PCA.png")
plt.tight_layout()
plt.savefig(PATH )
plt.close('all')

with open(os.path.join("Manuscripts", "src", "tables", "PCA.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())
with open(os.path.join("Manuscripts", "src", "tables", "PCA1.tex"), "w") as tf:
    tf.write(Y.round(decimals=2).to_latex())





#########################################################################################################
#########################################################################################################
#########################################################################################################
a = ["All features", "Only MDIST features", "Only RDIST features", "Only TOTEX features", "Only MVELO features", "Only RANGE features", "Only AREAXX features", "Only MFREQ features", "Only FDPD features", "Only FDCX features"]    

plt.figure(figsize=(14,8))
X = pd.DataFrame(index=a , columns=["Accuracy Left", "Accuracy Right"])
Y = pd.DataFrame(index=a , columns=[ "EER Left", "EER Right"])

FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
for idx, temp in enumerate(feature_names):
    a = ["All features", "Only MDIST features", "Only RDIST features", "Only TOTEX features", "Only MVELO features", "Only RANGE features", "Only AREAXX features", "Only MFREQ features", "Only FDPD features", "Only FDCX features"]    
    Results_DF_temp = Results_DF[   Results_DF["Features_Set"] == temp   ]
    Results_DF_temp = Results_DF_temp[   Results_DF_temp["PCA"] == 1   ]

    X.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_temp["Mean_Acc_L"].mean(),  Results_DF_temp["Mean_Acc_L"].std(), Results_DF_temp["Mean_Acc_L"].min(), Results_DF_temp["Mean_Acc_L"].max())
    X.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_temp["Mean_Acc_R"].mean(), Results_DF_temp["Mean_Acc_R"].std(), Results_DF_temp["Mean_Acc_R"].min(), Results_DF_temp["Mean_Acc_R"].max())
    Y.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_temp["Mean_EER_L"].mean(),       Results_DF_temp["Mean_EER_L"].std(), Results_DF_temp["Mean_EER_L"].min(), Results_DF_temp["Mean_EER_L"].max())
    Y.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_temp["Mean_EER_R"].mean(),      Results_DF_temp["Mean_EER_R"].std(), Results_DF_temp["Mean_EER_R"].min(), Results_DF_temp["Mean_EER_R"].max())    

    cols = ["FAR_L_" + str(i) for i in range(100)] 
    FAR_L.append(Results_DF_temp.loc[:, cols].mean().values)

    cols = ["FRR_L_" + str(i) for i in range(100)]
    FRR_L.append(Results_DF_temp.loc[:, cols].mean().values)

    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R.append(Results_DF_temp.loc[:, cols].mean().values)

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R.append(Results_DF_temp.loc[:, cols].mean().values)


    plt.subplot(1,2,1)
    auc = (1 + np.trapz( FRR_L[idx], FAR_L[idx]))

    plt.plot(FAR_L[idx], FRR_L[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)

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
    label = a[idx] #+ ' AUC = ' + str(round(auc, 2))
    plt.plot(FAR_R[idx], FRR_R[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)



    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, Right side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")

PATH = os.path.join("Manuscripts", "src", "figures", "feat.png")
plt.tight_layout()
plt.savefig(PATH )
plt.close('all')

with open(os.path.join("Manuscripts", "src", "tables", "feat.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())
with open(os.path.join("Manuscripts", "src", "tables", "feat1.tex"), "w") as tf:
    tf.write(Y.round(decimals=2).to_latex())