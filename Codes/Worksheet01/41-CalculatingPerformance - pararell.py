import warnings
import logging
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None 

# from scipy import ndimage
import matplotlib.pyplot as plt
import sys, os, timeit
from pathlib import Path as Pathlb

import multiprocessing
# from scipy.spatial import distance

# from sklearn import preprocessing
# from sklearn.neighbors import KNeighborsClassifier


# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.decomposition import PCA

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MLPackage import util as perf



working_path = perf.working_path
excel_path = os.path.join(working_path, 'results', 'Results_DF.xlsx')



def collect_results(result):
    global excel_path
    if os.path.isfile(excel_path):
        Results_DF = pd.read_excel(excel_path, index_col = 0)
    else:
        Results_DF = pd.DataFrame(columns=perf.cols)
        Results_DF.to_excel(excel_path)

    Results_DF = Results_DF.append(result)
    Results_DF.to_excel(excel_path)

def main():

    test_ratios = perf.test_ratios
    persentages = perf.persentages
    modes = perf.modes
    model_types = perf.model_types
    normilizings = perf.normilizings
    template_selection_methods = perf.template_selection_methods
    k_clusters = perf.k_clusters

    # test_ratios = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    # persentages = [0.95]#, 1.0]
    # modes = ["corr", "dist"]
    # model_types = ["min", "median", "average"]
    # normilizings = ["z-score"]#, "minmax"]

    for features_excel in perf.features_types:

        feature_path = os.path.join(working_path, 'Datasets', features_excel + ".xlsx")
        DF_features_all = pd.read_excel(feature_path, index_col = 0)


        logger.info("OS: {}".format(sys.platform))
        logger.info("Core Numbers: {}".format(multiprocessing.cpu_count()))
        logger.info("Feature shape: {}".format(DF_features_all.shape))
        
        if features_excel == "COAs_otsu" or features_excel == "COAs_simple" or features_excel == "COPs":
            for mode in modes:
                for template_selection_method in template_selection_methods:
                    for k_cluster in k_clusters:
                        if k_cluster != k_clusters[0] and template_selection_method == "None":
                            continue 
                        for normilizing in normilizings:
                            for x in [-3]:
                                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                                for persentage in persentages:  
                                    for model_type in model_types:
                                        for test_ratio in test_ratios:
                                            folder = str(persentage) + "_" + normilizing + "_" + str(x) + "_" + mode + "_" + model_type + "_" +  str(test_ratio) 
                                            pool.apply_async(perf.fcn, args=(DF_features_all, folder, features_excel, k_cluster, template_selection_method), callback=collect_results)
                                            # collect_results(perf.fcn(DF_features_all, folder))

                                pool.close()
                                pool.join()

        else:
            for mode in modes:
                for template_selection_method in template_selection_methods:
                    for k_cluster in k_clusters:
                        if k_cluster != k_clusters[0] and template_selection_method == "None":
                            continue
                        for normilizing in normilizings:
                            for x in [-3]:#range(-3,DF_features_all.shape[1]-2,3):
                                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                                for persentage in persentages:  
                                    for model_type in model_types:
                                        if x != -3 and persentage != 1.0:
                                            continue

                                        for test_ratio in test_ratios:
                                            folder = str(persentage) + "_" + normilizing + "_" + str(x) + "_" + mode + "_" + model_type + "_" +  str(test_ratio) 
                                            pool.apply_async(perf.fcn, args=(DF_features_all, folder, features_excel, k_cluster, template_selection_method), callback=collect_results)
                                            # collect_results(perf.fcn(DF_features_all,folder))


                                pool.close()
                                pool.join()


def create_logger():
    loggerName = Pathlb(__file__).stem
    log_path = os.path.join(working_path, 'logs', loggerName + '_loger.log')

    logger = logging.getLogger(loggerName)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s]-[%(name)s @ %(lineno)d]]-[%(levelname)s]\t%(message)s', datefmt='%m/%d/%y %I:%M:%S %p')
    file_handler = logging.FileHandler(log_path, mode = 'w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

if __name__ == '__main__': 
    logger = create_logger()
    logger.info("Starting !!!")
    tic = timeit.default_timer()
    main()
    toc = timeit.default_timer()
    logger.info("Done ({:2.2f} process time)!!!\n\n\n".format(toc-tic))


