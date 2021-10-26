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


import util as perf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MLPackage import FS



import logging
def create_logger(level):
    working_path = os.getcwd()
    loggerName = Pathlb(__file__).stem
    log_path = os.path.join(working_path, 'logs')
    Pathlb(log_path).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(loggerName)
    logger.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s]-[%(name)s @ %(lineno)d]-[%(levelname)s]\t%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    file_handler = logging.FileHandler( os.path.join(log_path, loggerName + '_loger.log'), mode = 'w')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
logger = create_logger(logging.INFO)       




plt.rcParams["font.size"] = 13
plt.tight_layout()


logger.info("Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "Manuscripts", "src", "figures")
tbl_dir = os.path.join(project_dir, "Manuscripts", "src", "tables")
data_dir = os.path.join(project_dir, "Archive", "results all 0.3")

Pathlb(fig_dir).mkdir(parents=True, exist_ok=True)
Pathlb(tbl_dir).mkdir(parents=True, exist_ok=True)






THRESHOLDs = perf.THRESHOLDs
feature_names = perf.feature_names

color = ['darkorange', 'navy', 'red', 'greenyellow', 'lightsteelblue', 'lightcoral', 'olive', 'mediumpurple', 'khaki', 'hotpink', 'blueviolet']






Results_DF = pd.read_excel(os.path.join(data_dir, 'Results_DF.xlsx'), index_col = 0)
Results_DF.columns = perf.cols





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
        Y.loc[value, "EER Left"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_EER_L_te"].mean(),       DF["Mean_EER_L_te"].std(), DF["Mean_EER_L_te"].min(), DF["Mean_EER_L_te"].max())
        Y.loc[value, "EER Right"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_EER_R_te"].mean(),      DF["Mean_EER_R_te"].std(), DF["Mean_EER_R_te"].min(), DF["Mean_EER_R_te"].max())    

        # print(DF)
        FAR_L.append(DF[["FAR_L_" + str(i) for i in range(100)]].mean().values)
        FRR_L.append(DF[["FRR_L_" + str(i) for i in range(100)]].mean().values)
        FAR_R.append(DF[["FAR_R_" + str(i) for i in range(100)]].mean().values)
        FRR_R.append(DF[["FRR_R_" + str(i) for i in range(100)]].mean().values)

    perf.plot_ROC(FAR_L, FRR_L, FAR_R, FRR_R, values)
    plt.tight_layout()
    plt.savefig(os.path.join("Manuscripts", "src", "figures", f_type + ".png"))
    plt.close('all')


    with open(os.path.join("Manuscripts", "src", "tables", f_type + "-Acc.tex"), "w") as tf:
        tf.write(X.to_latex())
    with open(os.path.join("Manuscripts", "src", "tables", f_type + "-EER.tex"), "w") as tf:
        tf.write(Y.to_latex())




for f_type in perf.features_types:   
    for column in ['Mode', 'Criteria', 'Normalizition', 'PCA']:
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
            Y.loc[value, "EER Left"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_EER_L_te"].mean(),       DF["Mean_EER_L_te"].std(), DF["Mean_EER_L_te"].min(), DF["Mean_EER_L_te"].max())
            Y.loc[value, "EER Right"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_EER_R_te"].mean(),      DF["Mean_EER_R_te"].std(), DF["Mean_EER_R_te"].min(), DF["Mean_EER_R_te"].max())    

            # print(DF)
            FAR_L.append(DF[["FAR_L_" + str(i) for i in range(100)]].mean().values)
            FRR_L.append(DF[["FRR_L_" + str(i) for i in range(100)]].mean().values)
            FAR_R.append(DF[["FAR_R_" + str(i) for i in range(100)]].mean().values)
            FRR_R.append(DF[["FRR_R_" + str(i) for i in range(100)]].mean().values)

        perf.plot_ROC(FAR_L, FRR_L, FAR_R, FRR_R, values)
        plt.tight_layout()
        plt.savefig(os.path.join("Manuscripts", "src", "figures", f_type + "-" + column + ".png"))
        plt.close('all')


        with open(os.path.join("Manuscripts", "src", "tables", f_type + "-" + column + "-Acc.tex"), "w") as tf:
            tf.write(X.to_latex())
        with open(os.path.join("Manuscripts", "src", "tables", f_type + "-" + column + "-EER.tex"), "w") as tf:
            tf.write(Y.to_latex())




for features_excel in ["afeatures-simple", "afeatures-otsu", "pfeatures"]:

    feature_path = os.path.join(perf.working_path, 'Datasets', features_excel + ".xlsx")
    DF_features = pd.read_excel(feature_path, index_col = 0)


    logger.info("Feature shape: {}".format( DF_features.shape))


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




logger.info("Done!!")
sys.exit()
FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
plt.figure(figsize=(14,8))
X = pd.DataFrame(index=["COAs-otsu", "COAs-simple", "COPs"] , columns=["Accuracy Left", "Accuracy Right"])
Y = pd.DataFrame(index=["COAs-otsu", "COAs-simple", "COPs"] , columns=[ "EER Left", "EER Right"])
Results_DF_group = Results_DF.groupby(["Feature_Type"])

for f_type in ["COAs-otsu", "COAs-simple", "COPs"]:   
    
    DF = Results_DF_group.get_group((f_type))
    X.loc[f_type, "Accuracy Left"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_Acc_L"].mean(),  DF["Mean_Acc_L"].std(), DF["Mean_Acc_L"].min(), DF["Mean_Acc_L"].max())
    X.loc[f_type, "Accuracy Right"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_Acc_R"].mean(), DF["Mean_Acc_R"].std(), DF["Mean_Acc_R"].min(), DF["Mean_Acc_R"].max())
    Y.loc[f_type, "EER Left"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_EER_L_te"].mean(),       DF["Mean_EER_L_te"].std(), DF["Mean_EER_L_te"].min(), DF["Mean_EER_L_te"].max())
    Y.loc[f_type, "EER Right"] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(DF["Mean_EER_R_te"].mean(),      DF["Mean_EER_R_te"].std(), DF["Mean_EER_R_te"].min(), DF["Mean_EER_R_te"].max())    

    FAR_L.append(DF[["FAR_L_" + str(i) for i in range(100)]].mean().values)
    FRR_L.append(DF[["FRR_L_" + str(i) for i in range(100)]].mean().values)
    FAR_R.append(DF[["FAR_R_" + str(i) for i in range(100)]].mean().values)
    FRR_R.append(DF[["FRR_R_" + str(i) for i in range(100)]].mean().values)


with open(os.path.join("Manuscripts", "src", "tables", "COX-time-series-Acc.tex"), "w") as tf:
    tf.write(X.to_latex())
with open(os.path.join("Manuscripts", "src", "tables", "COX-time-series-EER.tex"), "w") as tf:
    tf.write(Y.to_latex())
perf.plot_ROC(FAR_L, FRR_L, FAR_R, FRR_R, ["COAs_otsu", "COAs_simple", "COPs"])
plt.tight_layout()
plt.savefig(os.path.join("Manuscripts", "src", "figures", "COX-time-series.png"))
plt.close('all')



