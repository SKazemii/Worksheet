import numpy as np
import pandas as pd
from numpy.core.fromnumeric import argsort
from itertools import combinations
import matplotlib.pyplot as plt

import sys, os

import seaborn as sns
import klib
pd.options.mode.chained_assignment = None



def max_rel_calc(features, labels):
    fshape = features.shape
    features_names = features.columns.values

    positive_samples = features.loc[labels[labels == 1].index, :]
    negative_samples = features.loc[labels[labels == 0].index, :]

    D = list()
    F = list()
    for feature in range(fshape[1]):

        mean_feature_pos = positive_samples.iloc[:,feature].mean()
        mean_feature_neg = negative_samples.iloc[:,feature].mean()

        std_feature_pos = positive_samples.iloc[:,feature].std()
        std_feature_neg = negative_samples.iloc[:,feature].std()

        nominator = np.sqrt(2)*np.abs(mean_feature_pos-mean_feature_neg)
        denominator = np.sqrt((std_feature_pos**2)+(std_feature_neg**2))

        D.append(nominator/denominator)
        F.append(abs((mean_feature_pos-mean_feature_neg)/(std_feature_pos+std_feature_pos)))
        D_rank = argsort(D)[::-1][:]
        F_rank = argsort(F)[::-1][:]


    DF = pd.DataFrame(columns = ['feature_name', 'D-prim', 'F-rati', 'D-prime', 'F-ratio'])
    DF.loc[:,'feature_name'] = features_names
    DF.loc[:,'D-prim'] = D
    DF.loc[:,'F-rati'] = F
    DF.loc[:,'D-prime'] = features_names[D_rank]
    DF.loc[:,'F-ratio'] = features_names[F_rank]

    return DF


def mRMR(features, labels):
    DF = max_rel_calc(features, labels)
    features_names = features.columns.values

    # print((features.corr().abs().sum()-1)/features.shape[1]**2)
 
    R = ((features.corr().abs().sum()-1)/features.shape[1]).values
    D1 = DF['D-prim'] - R
    Q1 = DF['D-prim'] / R

    D1 = argsort(D1)[::-1][:]
    Q1 = argsort(Q1)[::-1][:]
    R1 = argsort(R)[:]

    DF['mRMR-Dif'] = features_names[D1]
    DF['mRMR-Q'] = features_names[Q1]
    DF['Redundancy'] = features_names[R1]



    return DF.iloc[:, 3:]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MLPackage import util as perf


def main():
    for features_excel in ["pfeatures"]:

        feature_path = os.path.join(perf.working_path, 'Datasets', features_excel + ".xlsx")
        DF_features = pd.read_excel(feature_path, index_col = 0)





        print( "[INFO] feature shape: ", DF_features.shape)


        f_names = ['MDIST_RD', 'MDIST_AP', 'MDIST_ML', 'RDIST_RD', 'RDIST_AP', 'RDIST_ML', 'TOTEX_RD', 'TOTEX_AP', 'TOTEX_ML', 'MVELO_RD', 'MVELO_AP', 'MVELO_ML', 'RANGE_RD', 'RANGE_AP', 'RANGE_ML','AREA_CC', 'AREA_CE', 'AREA_SW', 'MFREQ_RD', 'MFREQ_AP', 'MFREQ_ML', 'FDPD_RD', 'FDPD_AP', 'FDPD_ML', 'FDCC', 'FDCE']
        columnsName = f_names + [ "subject_ID", "left(0)/right(1)"]
        DF_features.columns = columnsName

    
        DF_side = DF_features[DF_features["left(0)/right(1)"] == 0]
        DF_side.loc[DF_side.subject_ID == 4.0, "left(0)/right(1)"] = 1
        DF_side.loc[DF_side.subject_ID != 4.0, "left(0)/right(1)"] = 0


        # plt.figure(figsize=(10,6), dpi=500)
        sns.clustermap( (DF_side.iloc[:,:-2].corr(method="pearson").abs()))

    DF = mRMR(DF_side.iloc[:,:-2], DF_side.iloc[:,-1])
    # plt.show()
    print(DF)
        
        # klib.corr_plot(DF_side.iloc[:,:-2], annot=False,)
        # PATH = os.path.join("temp", features_excel + ".png")
        # plt.tight_layout()
        # plt.savefig(PATH)
        # plt.close('all')
        # plt.show()

        # sys.exit()


    
        


        # 

        # with open(os.path.join("Manuscripts", "src", "tables", features_excel + "_10best_FS.tex"), "w") as tf:
        #     tf.write(DF.iloc[:10,:].to_latex())
        # with open(os.path.join("Manuscripts", "src", "tables", features_excel + "_10worst_FS.tex"), "w") as tf:
        #     tf.write(DF.iloc[-10:,:].to_latex())

    with pd.ExcelWriter(os.path.join("temp", features_excel + "_10best_FS.xlsx")) as writer:  
        DF.iloc[:10,:].to_excel(writer, sheet_name='Sheet_name_1')
    with pd.ExcelWriter(os.path.join("temp", features_excel + "_10worst_FS.xlsx")) as writer:  
        DF.iloc[-10:,:].to_excel(writer, sheet_name='Sheet_name_1')
            

    




    print("[INFO] Done!!!")

    # import pymrmr
    # df = pd.read_csv('some_df.csv')
    # Pass a dataframe with a predetermined configuration. 
    # Check http://home.penglab.com/proj/mRMR/ for the dataset requirements
    # pymrmr.mRMR(df, 'MIQ', 10)
    

if __name__ == "__main__":
    main()