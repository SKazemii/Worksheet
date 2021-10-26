import warnings

from pandas.core.frame import DataFrame
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, os, itertools
from pathlib import Path as Pathlb



from scipy.stats import shapiro, ttest_ind, mannwhitneyu


# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import util as perf


plt.rcParams["font.size"] = 13
plt.tight_layout()

print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "Manuscripts", "src", "figures")
tbl_dir = os.path.join(project_dir, "Manuscripts", "src", "tables")
data_dir = os.path.join(project_dir, "Archive", "results WS1")

Pathlb(fig_dir).mkdir(parents=True, exist_ok=True)
Pathlb(tbl_dir).mkdir(parents=True, exist_ok=True)





THRESHOLDs = perf.THRESHOLDs
feature_names = perf.feature_names

color = ['darkorange', 'navy', 'red', 'greenyellow', 'lightsteelblue', 'lightcoral', 'olive', 'mediumpurple', 'khaki', 'hotpink', 'blueviolet']



Results_DF = pd.read_excel(os.path.join(data_dir, 'DF.xlsx'), index_col = 0)
Results_DF.columns = perf.cols

pd.set_option('display.max_rows', 55)


# print(Results_DF.head())
# features_excelss = ["afeatures-simple", "pfeatures"]
# for features_excel in features_excelss:
#     for i in perf.modes:
#         for ii in ["All"] + perf.feature_names:
#             folder = str(1.0) + "_z-score_" + str(ii) + "_" + i + "_" + "min" + "_" +  str(0.0) + "_None_2" 
#             # print(folder)
#             left = pd.read_excel(os.path.join(data_dir, features_excel, folder,'Left.xlsx'), index_col = 0)
#             right = pd.read_excel(os.path.join(data_dir, features_excel, folder,'Right.xlsx'), index_col = 0)
#             mask = (Results_DF.Feature_Type == features_excel) & (Results_DF.Mode == i) & (Results_DF.Features_Set == ii)
#             Results_DF.loc[mask,"Mean_EER_L_tr"] = left.mean().dropna()["EER"] 
#             Results_DF.loc[mask,"Mean_EER_R_tr"] = right.mean().dropna()["EER"] 
#             # print(Results_DF[mask])


#             # # Results_DF.at[]
#             # print(left.mean().dropna()["EER"] )

#             # print(right.mean().dropna()["EER"] )

    


# print(Results_DF.iloc[:,7:17])
# Results_DF.to_excel(perf.working_path + "/DF_EER.xlsx")




Results_DF = pd.read_excel(os.path.join(data_dir, 'DF_EER.xlsx'), index_col = 0)




# plt.show()
# plt.close('all')





for f_type in ["COA features-simple", "COP features"]:  
    FAR_L = list()
    FRR_L = list()
    FAR_R = list()
    FRR_R = list()   
    counter = itertools.cycle([0,0,0,0,1,1,1,1])
    for i in ["correlation", "distance"]:
        mask = (Results_DF.Feature_Type == f_type) & (Results_DF.Mode == i) & (Results_DF.Features_Set == "All")
        Results_DF_f = Results_DF.loc[mask]

        FAR_L.append(Results_DF_f[["FAR_L_" + str(i) for i in range(perf.TH_dev)]].mean().values)
        FRR_L.append(Results_DF_f[["FRR_L_" + str(i) for i in range(perf.TH_dev)]].mean().values)
        FAR_R.append(Results_DF_f[["FAR_R_" + str(i) for i in range(perf.TH_dev)]].mean().values)
        FRR_R.append(Results_DF_f[["FRR_R_" + str(i) for i in range(perf.TH_dev)]].mean().values)

        perf.plot_ACC(FAR_L[next(counter)], FRR_L[next(counter)], FAR_R[next(counter)], FRR_R[next(counter)], THRESHOLDs)
        plt.savefig(os.path.join("Manuscripts", "src", "figures", "WS1_" + i + "_ACC.png"))



    labels = ["correlation", "distance"]
    perf.plot_ROC(FAR_L, FRR_L, FAR_R, FRR_R, labels)
    plt.savefig(os.path.join("Manuscripts", "src", "figures", "WS1_"+f_type+"_ROC.png"))


    Results_DF_group = Results_DF.groupby(["Feature_Type", "Features_Set"])
    values = Results_DF["Features_Set"].unique()

    Y = pd.DataFrame(index=values , columns=[ "EER Left", "EER Right"])
    FAR_L = list()
    FRR_L = list()
    FAR_R = list()
    FRR_R = list()
    for value in values:
        
        DF = Results_DF_group.get_group((f_type, value))
        Y.loc[value, "EER Left"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_EER_L_tr"].mean(),       DF["Mean_EER_L_tr"].std(), DF["Mean_EER_L_tr"].min(), DF["Mean_EER_L_tr"].max())
        Y.loc[value, "EER Right"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_EER_R_tr"].mean(),      DF["Mean_EER_R_tr"].std(), DF["Mean_EER_R_tr"].min(), DF["Mean_EER_R_tr"].max())    

        # print(DF)
        FAR_L.append(DF[["FAR_L_" + str(i) for i in range(perf.TH_dev)]].mean().values)
        FRR_L.append(DF[["FRR_L_" + str(i) for i in range(perf.TH_dev)]].mean().values)
        FAR_R.append(DF[["FAR_R_" + str(i) for i in range(perf.TH_dev)]].mean().values)
        FRR_R.append(DF[["FRR_R_" + str(i) for i in range(perf.TH_dev)]].mean().values)

    perf.plot_ROC(FAR_L, FRR_L, FAR_R, FRR_R, values)
    plt.tight_layout()
    plt.savefig(os.path.join("Manuscripts", "src", "figures", f_type + "_ROC.png"))
    plt.close('all')

    with pd.ExcelWriter(os.path.join("Manuscripts", "src", "tables", f_type + "-EER.xlsx")) as writer:  
        Y.to_excel(writer, sheet_name='Sheet_name_1')

print("Done!!!")
sys.exit()

Results_DF_temp = Results_DF[   Results_DF["Features_Set"] != "All"   ]


Results_DF_temp["Feature_Type"] = Results_DF_temp["Feature_Type"].map(lambda x: "afeat_si" if x == "afeatures_simple" else x)
Results_DF_temp["Feature_Type"] = Results_DF_temp["Feature_Type"].map(lambda x: "afeat_ot" if x == "afeatures_otsu" else x)
Results_DF_temp["Feature_Type"] = Results_DF_temp["Feature_Type"].map(lambda x: "COAs_si" if x == "COAs_simple" else x)
Results_DF_temp["Criteria"] = Results_DF_temp["Criteria"].map(lambda x: "ave" if x == "average" else x)
Results_DF_temp["Criteria"] = Results_DF_temp["Criteria"].map(lambda x: "med" if x == "median" else x)

X = Results_DF_temp.sort_values(by=['Mean_Acc_L', 'Mean_EER_L_te', 'Mean_f1_L'], ascending = [False, True, False]).iloc[:10,:15].drop(columns =['Time', 'Number_of_PCs', 'sklearn_EER_L'])
with open(os.path.join("Manuscripts", "src", "tables", "top10-L.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())

X = Results_DF_temp.sort_values(by=['Mean_Acc_L', 'Mean_EER_L_te', 'Mean_f1_L'], ascending = [True, False, True]).iloc[:10,:15].drop(columns =['Time', 'Number_of_PCs', 'sklearn_EER_L'])
with open(os.path.join("Manuscripts", "src", "tables", "worse10-L.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())



X = Results_DF_temp.sort_values(by=['Mean_Acc_R', 'Mean_EER_R_te', 'Mean_f1_R'], ascending = [False, True, False]).iloc[:10,:19].drop(columns =['Time', 'Number_of_PCs', 'Mean_f1_L', 'sklearn_EER_L', 'Mean_Acc_L', 'Mean_EER_L_te', 'Mean_EER_L_tr'])
with open(os.path.join("Manuscripts", "src", "tables", "top10-R.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())      

X = Results_DF_temp.sort_values(by=['Mean_Acc_R', 'Mean_EER_R_te', 'Mean_f1_R'], ascending = [True, False, True]).iloc[:10,:19].drop(columns =['Time', 'Number_of_PCs', 'Mean_f1_L', 'sklearn_EER_L',  'Mean_Acc_L', 'Mean_EER_L_te', 'Mean_EER_L_tr'])
with open(os.path.join("Manuscripts", "src", "tables", "worse10-R.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())          


Results_DF["Mode"] = Results_DF["Mode"].map(lambda x: "Correlation" if x == "corr" else "Euclidean distance")
Results_DF["Normalizition"] = Results_DF["Normalizition"].map(lambda x: "Z-score algorithm" if x == "z-score" else "Minmax algorithm")
Results_DF["PCA"] = Results_DF["PCA"].map(lambda x: "All PCs" if x == 1.0 else "keeping 95% variance")
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "afeatures-simple" if x == "afeatures_simple" else x)
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "afeatures-otsu" if x == "afeatures_otsu" else x)
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "COAs-otsu" if x == "COAs_otsu" else x)
Results_DF["Feature_Type"] = Results_DF["Feature_Type"].map(lambda x: "COAs-simple" if x == "COAs_simple" else x)


# for mode in ["Correlation", "Euclidean distance"]:
#     for criteria in perf.model_types:
        
plt.figure(figsize=(14,8))
Results_DF_group = Results_DF.groupby(["Test_Size"])
values = Results_DF["Test_Size"].sort_values().unique()
X = pd.DataFrame(index=values , columns=["Accuracy", "F1-score", "EER"])
Y = pd.DataFrame(index=values , columns=[ "EER Left", "EER Right"])
FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()        
for value in values:
    
    DF = Results_DF_group.get_group((value))
    X.loc[value, "Accuracy"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(
        (DF["Mean_Acc_L"].mean()+DF["Mean_Acc_R"].mean())/2, 
        (DF["Mean_Acc_L"].std()+ DF["Mean_Acc_R"].min())/2,
        (DF["Mean_Acc_L"].min()+ DF["Mean_Acc_R"].min())/2, 
        (DF["Mean_Acc_L"].max()+ DF["Mean_Acc_R"].max())/2)
    X.loc[value, "EER"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(
        (DF["Mean_EER_L_te"].mean()+DF["Mean_EER_R_te"].mean())/2,  
        (DF["Mean_EER_L_te"].std()+ DF["Mean_EER_R_te"].min())/2,
        (DF["Mean_EER_L_te"].min()+ DF["Mean_EER_R_te"].min())/2, 
        (DF["Mean_EER_L_te"].max()+ DF["Mean_EER_R_te"].max())/2)
    X.loc[value, "F1-score"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(
        (DF["Mean_f1_L"].mean()+DF["Mean_f1_R"].mean())/2,  
        (DF["Mean_f1_L"].std()+ DF["Mean_f1_R"].min())/2,
        (DF["Mean_f1_L"].min()+ DF["Mean_f1_R"].min())/2, 
        (DF["Mean_f1_L"].max()+ DF["Mean_f1_R"].max())/2)

    # X.loc[value, "Accuracy Left"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_Acc_L"].mean(),  DF["Mean_Acc_L"].std(), DF["Mean_Acc_L"].min(), DF["Mean_Acc_L"].max())
    # X.loc[value, "Accuracy Right"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_Acc_R"].mean(), DF["Mean_Acc_R"].std(), DF["Mean_Acc_R"].min(), DF["Mean_Acc_R"].max())
    # Y.loc[value, "EER Left"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_EER_L_te"].mean(),       DF["Mean_EER_L_te"].std(), DF["Mean_EER_L_te"].min(), DF["Mean_EER_L_te"].max())
    # Y.loc[value, "EER Right"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_EER_R_te"].mean(),      DF["Mean_EER_R_te"].std(), DF["Mean_EER_R_te"].min(), DF["Mean_EER_R_te"].max())    

    # print(DF)
    FAR_L.append(DF[["FAR_L_" + str(i) for i in range(perf.TH_dev)]].mean().values)
    FRR_L.append(DF[["FRR_L_" + str(i) for i in range(perf.TH_dev)]].mean().values)
    FAR_R.append(DF[["FAR_R_" + str(i) for i in range(perf.TH_dev)]].mean().values)
    FRR_R.append(DF[["FRR_R_" + str(i) for i in range(perf.TH_dev)]].mean().values)

perf.plot(FAR_L, FRR_L, FAR_R, FRR_R, str(values))
plt.tight_layout()
plt.savefig(os.path.join("Manuscripts", "src", "figures", "testsize.png"))
plt.close('all')


with open(os.path.join("Manuscripts", "src", "tables", "testsize.tex"), "w") as tf:
    tf.write(X.to_latex())
# with open(os.path.join("Manuscripts", "src", "tables", "testsize-EER.tex"), "w") as tf:
#     tf.write(Y.to_latex())
