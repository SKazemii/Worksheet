import numpy as np
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import median
import pandas as pd
from pandas.core.frame import DataFrame
from scipy import ndimage
import matplotlib.pyplot as plt
import sys, os
from scipy.spatial import distance

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from itertools import combinations, product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLPackage import util as perf



print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "manuscript", "src", "figures")
tbl_dir = os.path.join(project_dir, "manuscript", "src", "tables")
data_dir = os.path.join(project_dir, "results")



cols = ["Mode", "Model_Type", "Test_Size", "Normalizition", "Features_Set", "PCA",
"Mean_Accuracy_Left", "Mean_EER_Left", "Mean_Accuracy_Right", "Mean_EER_Right",
"Min_Accuracy_Left", "Min_EER_Left", "Min_Accuracy_Right", "Min_EER_Right",
"Max_Accuracy_Left", "Max_EER_Left", "Max_Accuracy_Right", "Max_EER_Right",
"Median_Accuracy_Left", "Median_EER_Left", "Median_Accuracy_Right", "Median_EER_Right"]

Results_DF = pd.read_excel(os.path.join(data_dir, 'Results_DF.xlsx'), index_col = 0)
FXR_L_DF = pd.read_excel(os.path.join(data_dir, 'FXR_L_DF.xlsx'), index_col = 0)
FXR_R_DF = pd.read_excel(os.path.join(data_dir, 'FXR_R_DF.xlsx'), index_col = 0)


print(Results_DF)
print(FXR_L_DF)
print(FXR_R_DF.shape)

Results_DF = pd.concat([Results_DF.reset_index(), FXR_L_DF, FXR_R_DF], axis=1)

print(Results_DF.shape)


test_ratios = [0.2, 0.35, 0.5]
persentages = [1.0, 0.95]
Modes = ["corr", "dist"]
model_types = ["min", "median", "average"]
THRESHOLDs = np.linspace(0, 1, 100)
normilizings = ["z-score", "minmax", "None"]
feature_names = ["All", "MDIST", "RDIST", "TOTEX", "MVELO", "RANGE", "AREAXX", "MFREQ", "FDPD", "FDCX"]

print(Results_DF.head())


Results_DF_all = Results_DF[   Results_DF["Features_Set"] == "All"   ]

npy = np.empty((4,4))
for mode in Modes:
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Mode"] == mode   ]

    npy[0,0] = Results_DF_all_mode["Mean_Accuracy_Left"].mean()  
    npy[0,1] = Results_DF_all_mode["Mean_Accuracy_Right"].mean()  
    npy[0,2] = Results_DF_all_mode["Mean_EER_Left"].mean()  
    npy[0,3] = Results_DF_all_mode["Mean_EER_Right"].mean()

    npy[1,0] = Results_DF_all_mode["Mean_Accuracy_Left"].min()  
    npy[1,1] = Results_DF_all_mode["Mean_Accuracy_Right"].min()  
    npy[1,2] = Results_DF_all_mode["Mean_EER_Left"].min()  
    npy[1,3] = Results_DF_all_mode["Mean_EER_Right"].min()  

    npy[2,0] = Results_DF_all_mode["Mean_Accuracy_Left"].max()  
    npy[2,1] = Results_DF_all_mode["Mean_Accuracy_Right"].max()  
    npy[2,2] = Results_DF_all_mode["Mean_EER_Left"].max()  
    npy[2,3] = Results_DF_all_mode["Mean_EER_Right"].max()    

    npy[3,0] = Results_DF_all_mode["Mean_Accuracy_Left"].median()  
    npy[3,1] = Results_DF_all_mode["Mean_Accuracy_Right"].median()  
    npy[3,2] = Results_DF_all_mode["Mean_EER_Left"].median()  
    npy[3,3] = Results_DF_all_mode["Mean_EER_Right"].median()


    X = pd.DataFrame(npy, index=["mean", "min", "max", "median"] , columns=["Accuracy Left", "Accuracy Right", "EER Left", "EER Right"])


    cols = ["FAR_L_" + str(i) for i in range(100)] 
    FAR_L = Results_DF_all_mode.loc[:, cols].mean().values

    cols = ["FRR_L_" + str(i) for i in range(100)]
    FRR_L = Results_DF_all_mode.loc[:, cols].mean().values

    PATH = os.path.join("Manuscripts", "src", "figures", mode + "_L")
    perf.ROC_plot_v2(FAR_L, FRR_L, THRESHOLDs, PATH)




    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R = Results_DF_all_mode.loc[:, cols].mean().values

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R = Results_DF_all_mode.loc[:, cols].mean().values

    PATH = os.path.join("Manuscripts", "src", "figures", mode + "_R")
    perf.ROC_plot_v2(FAR_R, FRR_R, THRESHOLDs, PATH)



    with open(os.path.join("Manuscripts", "src", "tables", mode + ".tex"), "w") as tf:
        tf.write(X.to_latex())

    


    # print(Results_DF_all_mode.head())

    # for j in range(6):
    #     # print(Results_DF[   Results_DF[cols[i]] == "corr"   ].head())
    #     print(cols[i])




sys.exit()
print("[INFO] Saving and showing the plot of the first dataset")
plt.figure(0)
fig = series.plot()
plt.savefig(os.path.join(fig_dir, a + "raw_signal.png"))


with open(os.path.join(tbl_dir, a + "raw_signal_summary_statistics.tex"), "w") as tf:
    tf.write(series.describe().to_latex())


test_ratio = [0.2, 0.35, 0.5]
persentage = [1.0, 0.95]
mode = ["corr", "dist"]
model_type = ["min", "median", "average"]
THRESHOLDs = np.linspace(0, 1, 100)
score = None#"B"
score = "A"#"B"
normilizing = ["z-score", "minmax", "None"]

feature_names = ["All", "MDIST", "RDIST", "TOTEX", "MVELO", "RANGE", "AREAXX", "MFREQ", "FDPD", "FDCX"]



output = list(product(mode, model_type, persentage, test_ratio, normilizing, feature_names ))

print(output)
print(len(output))
for i in output:
    print(i[0]+ "_" + i[1] + "_" + str(i[2])+ "_" + str(i[3]) + "_" + i[4] + "_" + i[5])
    folder = i[0]+ "_" + i[1] + "_" + str(i[2])+ "_" + str(i[3]) + "_" + i[4] + "_" + i[5]
    
    path =  "/Users/saeedkazemi/Documents/Python/Worksheet/results/" + folder + "/NPY/"
    print(path)

    EER_R = np.load(path + "EER_R.npy")
    FAR_R = np.load(path + "FAR_R.npy")
    FRR_R = np.load(path + "FRR_R.npy")
    EER_L = np.load(path + "EER_L.npy")
    FRR_L = np.load(path + "FRR_L.npy")
    FAR_L = np.load(path + "FAR_L.npy")
    ACC_L = np.load(path + "ACC_L.npy")
    ACC_R = np.load(path + "ACC_R.npy")
    
    print(ACC_R)
    print(ACC_R.shape)
    sys.exit()

ss= 2*3*2*3*3*10
print(ss)