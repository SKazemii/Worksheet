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



working_path = perf.working_path
# Results_DF = pd.DataFrame(columns=perf.cols)
folder_path = os.path.join(working_path, 'results', 'Results_DF.xlsx')
# Pathlb(folder_path).mkdir(parents=True, exist_ok=True)



def collect_results(result):
    global Results_DF
    global folder_path
  
    if os.path.isfile(folder_path):
        print('old file')
        Results_DF = pd.read_excel(folder_path, index_col = 0)
    else:
        print('new file')
        Results_DF = pd.DataFrame(columns=perf.cols)
        Results_DF.to_excel(folder_path)


    Results_DF = Results_DF.append(result)
    Results_DF.to_excel(folder_path)

def main():

    test_ratios = perf.test_ratios
    persentages = perf.persentages
    modes = perf.modes
    model_types = perf.model_types
    normilizings = perf.normilizings

    # test_ratios = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    # persentages = [0.95]#, 1.0]
    # modes = ["corr", "dist"]
    # model_types = ["min", "median", "average"]
    # normilizings = ["z-score"]#, "minmax"]



    feature_path = os.path.join(working_path, 'Datasets', perf.features_excel + ".xlsx")
    DF_features_all = pd.read_excel(feature_path, index_col = 0)
    print(DF_features_all.head())


    print("[INFO] OS: ", sys.platform)
    print("[INFO] Core Number: ", multiprocessing.cpu_count())
    print("[INFO] feature shape: ", DF_features_all.shape)
    
    if perf.features_excel == "COAs_otsu" or perf.features_excel == "COAs_simple" or perf.features_excel == "COPs":
        for persentage in persentages:
            for normilizing in normilizings:
                for x in [-3]:
                    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                    for mode in modes:  
                        for model_type in model_types:
                            for test_ratio in test_ratios:
                                folder = str(persentage) + "_" + normilizing + "_" + str(x) + "_" + mode + "_" + model_type + "_" +  str(test_ratio) 
                                pool.apply_async(perf.fcn, args=(DF_features_all, folder), callback=collect_results)
                                # collect_results(perf.fcn(DF_features_all,folder))


                    pool.close()
                    pool.join()
                    sys.exit()

    else:
        for persentage in persentages:
            for normilizing in normilizings:
                for x in range(-3,DF_features_all.shape[1]-2,3):
                    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                    for mode in modes:  
                        for model_type in model_types:
                            if mode == "corr" and x != -3 and persentage != 1.0:
                                continue

                            for test_ratio in test_ratios:
                                folder = str(persentage) + "_" + normilizing + "_" + str(x) + "_" + mode + "_" + model_type + "_" +  str(test_ratio) 
                                pool.apply_async(perf.fcn, args=(DF_features_all, folder), callback=collect_results)
                                # collect_results(perf.fcn(DF_features_all,folder))


                    pool.close()
                    pool.join()



if __name__ == '__main__': 
    print("\n\n\n[INFO] starting !!!")
    tic = timeit.default_timer()
    main()
    toc = timeit.default_timer()
    print("[INFO] Done ({:2.2f} process time)!!!\n\n\n".format(toc-tic))


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