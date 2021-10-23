# Example of the Shapiro-Wilk Normality Test 1-1
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_ind, friedmanchisquare, f_oneway, mannwhitneyu
import pandas as pd
import numpy as np
import sys, pprint
from itertools import combinations






def stat(DF, labels, th = 0.05, plot = False):
    
    res = {}
    c = 0
    for j in DF.groupby(labels[0])[labels[1]].apply(lambda x:list(x)):

        p = shapiro(j)
        res["shapiro("+str(DF[labels[0]].unique()[c])+")" ] = [p.pvalue, "Gaussian" if p.pvalue > th else "Not Gaussian"]
        if plot == True:
            plt.title("The histogram of "+ labels[1] + " on " + str(DF[labels[0]].unique()[c]) + " " + labels[0])
            plt.hist(j)
            plt.figure()


        c = c + 1


    if DF[labels[0]].unique().shape[0] > 2:
        # Parametric  
        p = f_oneway(*DF.groupby(labels[0])[labels[1]].apply(lambda x:list(x)))
        res["Parametric_ANOVA(All data)"] = [p.pvalue, "Results are not significantly different at the given significance level." if p.pvalue > th else "Results are statistically significant."]

    xx = list(combinations(DF[labels[0]].unique(), 2))
    for i in range(len(xx)):
        z = DF.groupby(labels[0])[labels[1]].apply(lambda x:list(x))
        p = ttest_ind(
            z[xx[i][0]],
            z[xx[i][1]])
        res["ttest("+str(xx[i][0])+","+str(xx[i][1])+")" ] = [p.pvalue, "Results are not significantly different at the given significance level." if p.pvalue > th else "Results are statistically significant."]
    
    
    if DF[labels[0]].unique().shape[0] > 2:# DF.columns
        # Nonparametric  Friedman 
        p = friedmanchisquare(*DF.groupby(labels[0])[labels[1]].apply(lambda x:list(x)))
        res["Nonparametric_Friedman(All data)"] = [p.pvalue, "Results are not significantly different at the given significance level." if p.pvalue > th else "Results are statistically significant."]

    for i in range(len(xx)):
        z = DF.groupby(labels[0])[labels[1]].apply(lambda x:list(x))
        p = mannwhitneyu(
            z[xx[i][0]],
            z[xx[i][1]])
        res["Mann-Whitney_U_Test("+str(xx[i][0])+","+str(xx[i][1])+")" ] = [p.pvalue, "Results are not significantly different at the given significance level." if p.pvalue > th else "Results are statistically significant."]


    res = pd.DataFrame(res, index=["p-value", "Distribution"]).T
    return res
    


def main():
    data = {"data1" : [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869],
            "data2" : [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169],
            "data3" : [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]}

    data = {"ss" : [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869], "jj" : [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]}

    DF = pd.DataFrame(data)

    d = list()
    l = list()
    
    for (columnName, columnData) in DF.iteritems():
        d.append(columnData.values.tolist())
        l.append([columnName for _ in range(columnData.values.shape[0])] )



    d = [item for subl in d for item in subl]
    l = [item for subl in l for item in subl]
    DF_flated = pd.DataFrame({"a":l, "b":d})
    print(stat(DF_flated, labels=["a", "b"]))

    

    
    print("Done!!!")



if __name__ == "__main__":
    main()

