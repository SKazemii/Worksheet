import numpy as np
import pandas as pd
from numpy.core.fromnumeric import argsort
from numpy.lib.function_base import piecewise
from itertools import combinations
import sys

import scipy.stats as scipy


def max_rel_calc(features, labels, n, measure = "d_prime"):
    fshape = features.shape
    positive_samples = features[labels == 1, :]
    negative_samples = features[labels == 0, :]

    D = list()
    F = list()
    for feature in range(fshape[1]):
        mean_feature_pos = np.mean(positive_samples[:,feature])
        mean_feature_neg = np.mean(negative_samples[:,feature])

        std_feature_pos = np.std(positive_samples[:,feature])
        std_feature_neg = np.std(negative_samples[:,feature])

        nominator = np.sqrt(2)*np.abs(mean_feature_pos-mean_feature_neg)
        denominator = np.sqrt((std_feature_pos**2)+(std_feature_neg**2))

        D.append(nominator/denominator)
        F.append((mean_feature_pos-mean_feature_neg)/(std_feature_pos+std_feature_pos))
    if measure == "d_prime":
        relevent_features = argsort(D)[::-1][:n]
    elif measure == "F_ratio":
        relevent_features = argsort(F)[::-1][:n]
    # print(D)
    # print(relevent_features)
    return features[:,relevent_features], relevent_features, D, F
    # print(argsort(D)[::-1][:n])
    # print(D)
    # return D,F




def min_red_calc(features):
    redundancy_matrix = np.empty( [features.shape[1], features.shape[1]] )
    # print(redundancy_matrix.shape)
    # print(redundancy_matrix)
    
    sl = 0 
    while (sl < features.shape[1]):
        input_correl = 0
        zl = 0 
        while (zl < features.shape[1]):
            if(sl == zl):
                input_correl = 1
            else:
                correl = scipy.pearsonr(features[:,sl], features[:,zl])
                input_correl = correl[0]
                # print(correl)
                # sys.exit()


            redundancy_matrix[sl][zl] = input_correl
            zl = zl +1
        sl = sl + 1

    DF_rm = pd.DataFrame(redundancy_matrix)
    print(DF_rm)
    a = list()
    maximum = 1000
    for set in range(2,features.shape[1]-1):
        comb = combinations(range(features.shape[1]),set)
        for i in list(comb):
            # print(np.sum(redundancy_matrix[i,i]))
            sumation = DF_rm.loc[i,i].abs().sum().sum()
            a.append({sumation:i})
            if sumation < maximum:
                print(i)
                maximum = sumation
                best_set = i
            # print(DF_rm.loc[i,i].sum().sum())
    
    print(best_set)
    print(maximum)


    # sys.exit()
    return features[:,best_set], best_set, maximum


def mRMR(features, labels, n, measure = "d_prime"):
    _, _, D, _ = max_rel_calc(features, labels, n, measure)

    redundancy_matrix = np.empty( [features.shape[1], features.shape[1]] )
    # print(redundancy_matrix.shape)
    # print(redundancy_matrix)
    
    sl = 0 
    while (sl < features.shape[1]):
        input_correl = 0
        zl = 0 
        while (zl < features.shape[1]):
            if(sl == zl):
                input_correl = 1
            else:
                correl = scipy.pearsonr(features[:,sl], features[:,zl])
                input_correl = correl[0]
                # print(correl)
                # sys.exit()


            redundancy_matrix[sl][zl] = input_correl
            zl = zl + 1
        sl = sl + 1

    DF_rm = pd.DataFrame(redundancy_matrix)


    minimum = 0
    for set in range(2,features.shape[1]-1):
        comb = combinations(range(features.shape[1]),set)
        for i in list(comb):
            # print(np.sum(redundancy_matrix[i,i]))
            sumation = DF_rm.loc[i,i].abs().sum().sum()
            # print(i)
            d = (np.sum([D[j] for j in i]))
            MIQ = d / sumation
            if MIQ > minimum:
                # print(i)
                minimum = MIQ
                best_set = i
            # print(DF_rm.loc[i,i].sum().sum())
    
    print(best_set)
    print(minimum)

    return features[:,best_set], best_set, minimum




def main():
    pfeatures = np.load("./Datasets/pfeatures.npy")

    features = pfeatures

    print("[INFO] feature shape: ", features.shape)
    columnsName = ["feature_" + str(i) for i in range(features.shape[1]-2)] + [ "subject_ID", "left(0)/right(1)"]

    DF_features = pd.DataFrame(
        features,
        columns = columnsName 
    )
    DF_side = DF_features[DF_features["left(0)/right(1)"] == 0]
    DF_side.loc[DF_side.subject_ID == 4.0, "left(0)/right(1)"] = 1
    DF_side.loc[DF_side.subject_ID != 4.0, "left(0)/right(1)"] = 0
    # DF_side = DF_side[DF_side["left(0)/right(1)"] == 0]

    # print(min_red_calc(pfeatures[:,0:5]))
    pfeatures = DF_side.iloc[:,0:5].values
    labels = DF_side.iloc[:,-1].values
    # print(pfeatures)
    # print(labels)

    mRMR(pfeatures, labels, n=10, measure = "d_prime")
    print("[INFO] Done!!!")

    import pymrmr
    df = pd.read_csv('some_df.csv')
    # Pass a dataframe with a predetermined configuration. 
    # Check http://home.penglab.com/proj/mRMR/ for the dataset requirements
    pymrmr.mRMR(df, 'MIQ', 10)
    sys.exit()
    

    sys.exit()
    pfeatures = np.load("./Datasets/pfeatures.npy")

    print(pfeatures.shape)
    columnsName = ["feature_" + str(i) for i in range(5)] 

    comb = combinations(columnsName,2)
    for i in list(comb):
        print (i)
    features = np.array(pfeatures[:,0:5])
    labels = np.array(pfeatures[:,-1])
    relevent_features = max_rel_calc(features,labels, n=10)
    print(relevent_features)
    print(relevent_features.shape)
    # print(F)
    
    print("[INFO] Done!!!")



if __name__ == "__main__":
    main()