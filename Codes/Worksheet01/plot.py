import numpy as np
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import median
import pandas as pd
from pandas.core.frame import DataFrame
from scipy import ndimage
import matplotlib.pyplot as plt
import sys, os
from pathlib import Path as Pathlb


from scipy.spatial import distance

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from itertools import combinations, product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLPackage import util as perf



print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "Manuscripts", "src", "figures")
tbl_dir = os.path.join(project_dir, "Manuscripts", "src", "tables")
data_dir = os.path.join(project_dir, "results")

Pathlb(fig_dir).mkdir(parents=True, exist_ok=True)
Pathlb(tbl_dir).mkdir(parents=True, exist_ok=True)



Results_DF = pd.read_excel(os.path.join(data_dir, 'Results_DF.xlsx'), index_col = 0)
FXR_L_DF = pd.read_excel(os.path.join(data_dir, 'FXR_L_DF.xlsx'), index_col = 0)
FXR_R_DF = pd.read_excel(os.path.join(data_dir, 'FXR_R_DF.xlsx'), index_col = 0)



Results_DF = pd.concat([Results_DF.reset_index(), FXR_L_DF, FXR_R_DF], axis=1)



test_ratios = [0.2, 0.35, 0.5]
persentages = [1.0, 0.95]
Modes = ["corr", "dist"]
model_types = ["min", "median", "average"]
THRESHOLDs = np.linspace(0, 1, 100)
normilizings = ["z-score", "minmax", "None"]
feature_names = ["All", "MDIST", "RDIST", "TOTEX", "MVELO", "RANGE", "AREAXX", "MFREQ", "FDPD", "FDCX"]



Results_DF_all = Results_DF[   Results_DF["Features_Set"] == "All"   ]

npy = np.empty((4,4))
for idx, temp in enumerate(Modes):
    a = ["Correlation", "Euclidean Distance"]
    

    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Mode"] == temp   ]

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

    PATH = os.path.join("Manuscripts", "src", "figures", a[idx] + "_L")
    perf.ROC_plot_v2(FAR_L, FRR_L, THRESHOLDs, PATH)




    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R = Results_DF_all_mode.loc[:, cols].mean().values

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R = Results_DF_all_mode.loc[:, cols].mean().values

    PATH = os.path.join("Manuscripts", "src", "figures", a[idx] + "_R")
    perf.ROC_plot_v2(FAR_R, FRR_R, THRESHOLDs, PATH)



    with open(os.path.join("Manuscripts", "src", "tables", a[idx] + ".tex"), "w") as tf:
        tf.write(X.round(decimals=2).to_latex())

    
for idx, temp in enumerate(model_types):
    a = ["Minimum", "Median", "Average"]
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Model_Type"] == temp   ]

   
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

    PATH = os.path.join("Manuscripts", "src", "figures", a[idx] + "_L")
    perf.ROC_plot_v2(FAR_L, FRR_L, THRESHOLDs, PATH)




    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R = Results_DF_all_mode.loc[:, cols].mean().values

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R = Results_DF_all_mode.loc[:, cols].mean().values

    PATH = os.path.join("Manuscripts", "src", "figures", a[idx] + "_R")
    perf.ROC_plot_v2(FAR_R, FRR_R, THRESHOLDs, PATH)



    with open(os.path.join("Manuscripts", "src", "tables", a[idx] + ".tex"), "w") as tf:
        tf.write(X.round(decimals=2).to_latex())


    
for idx, temp in enumerate(normilizings):
    a = ["Z-score algorithm", "MinMax algorithm", "None "]


    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Normalizition"] == temp   ]

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

    PATH = os.path.join("Manuscripts", "src", "figures", a[idx] + "_L")
    perf.ROC_plot_v2(FAR_L, FRR_L, THRESHOLDs, PATH)




    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R = Results_DF_all_mode.loc[:, cols].mean().values

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R = Results_DF_all_mode.loc[:, cols].mean().values

    PATH = os.path.join("Manuscripts", "src", "figures", a[idx] + "_R")
    perf.ROC_plot_v2(FAR_R, FRR_R, THRESHOLDs, PATH)



    with open(os.path.join("Manuscripts", "src", "tables", a[idx] + ".tex"), "w") as tf:
        tf.write(X.round(decimals=2).to_latex())


for idx, temp in enumerate(test_ratios):
    a = ["20%", "35%", "50% "]
  
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Test_Size"] == temp   ]

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

    PATH = os.path.join("Manuscripts", "src", "figures", a[idx] + "_L")
    perf.ROC_plot_v2(FAR_L, FRR_L, THRESHOLDs, PATH)




    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R = Results_DF_all_mode.loc[:, cols].mean().values

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R = Results_DF_all_mode.loc[:, cols].mean().values

    PATH = os.path.join("Manuscripts", "src", "figures", a[idx] + "_R")
    perf.ROC_plot_v2(FAR_R, FRR_R, THRESHOLDs, PATH)



    with open(os.path.join("Manuscripts", "src", "tables", a[idx] + ".tex"), "w") as tf:
        tf.write(X.round(decimals=2).to_latex())


for idx, temp in enumerate(persentages):
    a = ["All Data", "Keeping 95%"]
  
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["PCA"] == temp   ]

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

    PATH = os.path.join("Manuscripts", "src", "figures", a[idx] + "_L")
    perf.ROC_plot_v2(FAR_L, FRR_L, THRESHOLDs, PATH)




    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R = Results_DF_all_mode.loc[:, cols].mean().values

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R = Results_DF_all_mode.loc[:, cols].mean().values

    PATH = os.path.join("Manuscripts", "src", "figures", a[idx] + "_R")
    perf.ROC_plot_v2(FAR_R, FRR_R, THRESHOLDs, PATH)



    with open(os.path.join("Manuscripts", "src", "tables", a[idx] + ".tex"), "w") as tf:
        tf.write(X.round(decimals=2).to_latex())


for idx, temp in enumerate(feature_names):
    a = ["All features", "Only MDIST features", "Only RDIST features", "Only TOTEX features", "Only MVELO features", "Only RANGE features", "Only AREAXX features", "Only MFREQ features", "Only FDPD features", "Only FDCX features"]    
    Results_DF_temp = Results_DF[   Results_DF["Features_Set"] == temp   ]

    npy[0,0] =  Results_DF_temp["Mean_Accuracy_Left"].mean()  
    npy[0,1] =  Results_DF_temp["Mean_Accuracy_Right"].mean()  
    npy[0,2] =  Results_DF_temp["Mean_EER_Left"].mean()  
    npy[0,3] =  Results_DF_temp["Mean_EER_Right"].mean()

    npy[1,0] =  Results_DF_temp["Mean_Accuracy_Left"].min()  
    npy[1,1] =  Results_DF_temp["Mean_Accuracy_Right"].min()  
    npy[1,2] =  Results_DF_temp["Mean_EER_Left"].min()  
    npy[1,3] =  Results_DF_temp["Mean_EER_Right"].min()  

    npy[2,0] =  Results_DF_temp["Mean_Accuracy_Left"].max()  
    npy[2,1] =  Results_DF_temp["Mean_Accuracy_Right"].max()  
    npy[2,2] =  Results_DF_temp["Mean_EER_Left"].max()  
    npy[2,3] =  Results_DF_temp["Mean_EER_Right"].max()    

    npy[3,0] =  Results_DF_temp["Mean_Accuracy_Left"].median()  
    npy[3,1] =  Results_DF_temp["Mean_Accuracy_Right"].median()  
    npy[3,2] =  Results_DF_temp["Mean_EER_Left"].median()  
    npy[3,3] =  Results_DF_temp["Mean_EER_Right"].median()


    X = pd.DataFrame(npy, index=["mean", "min", "max", "median"] , columns=["Accuracy Left", "Accuracy Right", "EER Left", "EER Right"])


    cols = ["FAR_L_" + str(i) for i in range(100)] 
    FAR_L =  Results_DF_temp.loc[:, cols].mean().values

    cols = ["FRR_L_" + str(i) for i in range(100)]
    FRR_L =  Results_DF_temp.loc[:, cols].mean().values

    PATH = os.path.join("Manuscripts", "src", "figures", a[idx] + "_L")
    perf.ROC_plot_v2(FAR_L, FRR_L, THRESHOLDs, PATH)




    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R =  Results_DF_temp.loc[:, cols].mean().values

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R =  Results_DF_temp.loc[:, cols].mean().values

    PATH = os.path.join("Manuscripts", "src", "figures", a[idx] + "_R")
    perf.ROC_plot_v2(FAR_R, FRR_R, THRESHOLDs, PATH)



    with open(os.path.join("Manuscripts", "src", "tables", a[idx] + ".tex"), "w") as tf:
        tf.write(X.round(decimals=2).to_latex())

