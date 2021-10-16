import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from pandas.core.frame import DataFrame
from scipy import ndimage
import matplotlib.pyplot as plt
import sys, os, timeit
from pathlib import Path as Pathlb

import multiprocessing
from scipy.spatial import distance

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLPackage import util as perf

# global Results_DF



Results_DF = pd.DataFrame(columns=perf.cols)

def collect_results(result):
    global Results_DF
    Results_DF = Results_DF.append(result)
    Results_DF.to_excel(os.path.join(perf.working_path, 'results', 'Results_DF.xlsx'))

def main():

    # test_ratios = perf.test_ratios
    # persentages = perf.persentages
    # modes = perf.modes
    # model_types = perf.model_types
    # normilizings = perf.normilizings

    test_ratios = [0.1, 0.2, 0.25]#, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    persentages = [0.95]#, 1.0]
    modes = ["corr"]#, "dist"]
    model_types = ["min"]#, "median", "average"]
    normilizings = ["z-score"]#, "minmax"]


    working_path = os.getcwd()
    folder_path = os.path.join(working_path, 'results')
    Pathlb(folder_path).mkdir(parents=True, exist_ok=True)


    features_excel = ["afeatures_otsu.xlsx", 
                    "afeatures_simple", 
                    "pfeatures.xlsx", 
                    "COAs_otsu.xlsx", 
                    "COAs_simple.xlsx", 
                    "COPs.xlsx"]




    feature_path = os.path.join(working_path, 'Datasets', features_excel[5])
    DF_features_all = pd.read_excel(feature_path, index_col = 0)
    # print(DF_features_all)
    # print(DF_features_all.shape[1])
    # sys.exit()  


    print("[INFO] OS: ", sys.platform)
    print("[INFO] Core Number: ", multiprocessing.cpu_count())
    print("[INFO] feature shape: ", DF_features_all.shape)




    

 


    index = 0
    
    
    
    for persentage in persentages:
        for normilizing in normilizings:
            for x in range(-3,DF_features_all.shape[1]-2,3):
                for mode in modes:  
                    for model_type in model_types:
                        if mode == "corr" and x != -3 and persentage != 1.0:
                            continue

                        tic=timeit.default_timer()
                        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                        for test_ratio in test_ratios:
                            folder = str(persentage) + "_" + normilizing + "_" + str(x) + "_" + mode + "_" + model_type + "_" +  str(test_ratio) 
                            pool.apply_async(perf.fcn, args=(DF_features_all, folder), callback=collect_results)
                            # collect_results(perf.fcn(DF_features_all,folder))


                        pool.close()
                        pool.join()
                        toc=timeit.default_timer()


    # print(Results_DF.head(  ))                       
    print("[INFO] Done ({:2.2f} process time)!!!\n\n\n".format(toc-tic))



if __name__ == '__main__': 
    print("\n\n\n[INFO] starting !!!")
    main()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# \foreach \n in {Test size}{
# \section{Test size}
# \foreach \t in {20 percent, 35 percent, 50 percent}{

# \begin{frame}
# \frametitle{\t \ \n}
# \tiny
# \begin{table}
# \centering
# \caption{\small The accuracy and ERR of \t \  \n.}
# \input{tables/\t.tex}
# \end{table}
# \end{frame}
# }

# \begin{frame}
# \centering
# \frametitle{The ROC curve}
# \includegraphics[scale=0.3]{Manuscripts/src/figures/\variable_ROC.png}
# \end{frame}