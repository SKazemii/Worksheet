import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, os, logging, timeit
from pathlib import Path as Pathlb


from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import itertools


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))	
from MLPackage import stat 

project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "Manuscripts", "src", "figures")
tbl_dir = os.path.join(project_dir, "Manuscripts", "src", "tables")
results_dir = os.path.join(project_dir, "results")
dataset_dir = os.path.join(project_dir, "Datasets")
temp_dir = os.path.join(project_dir, "temp")
log_path = os.path.join(project_dir, 'logs')

Pathlb(log_path).mkdir(parents=True, exist_ok=True)
Pathlb(dataset_dir).mkdir(parents=True, exist_ok=True)
Pathlb(temp_dir).mkdir(parents=True, exist_ok=True)
Pathlb(results_dir ).mkdir(parents=True, exist_ok=True)
Pathlb(fig_dir).mkdir(parents=True, exist_ok=True)
Pathlb(tbl_dir).mkdir(parents=True, exist_ok=True)



def create_logger(level):
    loggerName = Pathlb(__file__).stem
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
logger = create_logger(logging.DEBUG)       



def main():
    folder1 = "0.95_z-score_All_dist_min_0.3_None_4"
    path = os.path.join(project_dir, "Archive", "results all 0.3", "pfeatures", folder1, "Left.xlsx")
    Results_DF_1 = pd.read_excel(path, index_col = 0)
    Results_DF_1["subject ID"] = Results_DF_1["subject ID"].map(lambda x: "min")

    folder1 = "0.95_z-score_All_dist_median_0.3_None_4"
    path = os.path.join(project_dir, "Archive", "results all 0.3", "pfeatures", folder1, "Left.xlsx")
    Results_DF_2 = pd.read_excel(path, index_col = 0)
    Results_DF_2["subject ID"] = Results_DF_2["subject ID"].map(lambda x: "median")

    folder1 = "0.95_z-score_All_dist_average_0.3_None_4"
    path = os.path.join(project_dir, "Archive", "results all 0.3", "pfeatures", folder1, "Left.xlsx")
    Results_DF_3 = pd.read_excel(path, index_col = 0)
    Results_DF_3["subject ID"] = Results_DF_3["subject ID"].map(lambda x: "average")


    vertical_concat = pd.concat([Results_DF_2, Results_DF_1, Results_DF_3], axis=0)

    print(stat.stat(vertical_concat[["subject ID", "mean(acc)"]], labels=["subject ID", "mean(acc)"], plot = True).head(100))
    plt.show()
    logger.info("Done!!")
    sys.exit()
    ####################################################################################################################################################################################################
    ####################################################################################################################################################################################################
    ####################################################################################################################################################################################################

    folder1 = "0.95_z-score_All_dist_min_0.3_None_4"
    path = os.path.join(project_dir, "Archive", "results all 0.3", "pfeatures", folder1, "Left.xlsx")
    Results_DF_1 = pd.read_excel(path, index_col = 0)
    Results_DF_1["subject ID"] = Results_DF_1["subject ID"].map(lambda x: "min")

    folder1 = "0.95_z-score_All_dist_average_0.3_None_4"
    path = os.path.join(project_dir, "Archive", "results all 0.3", "pfeatures", folder1, "Left.xlsx")
    Results_DF_2 = pd.read_excel(path, index_col = 0)
    Results_DF_2["subject ID"] = Results_DF_2["subject ID"].map(lambda x: "average")


    vertical_concat = pd.concat([Results_DF_2, Results_DF_1], axis=0)

    print(stat.stat(vertical_concat[["subject ID", "mean(acc)"]], labels=["subject ID", "mean(acc)"], plot = True).head(100))
    ####################################################################################################################################################################################################
    ####################################################################################################################################################################################################
    ####################################################################################################################################################################################################
    folder1 = "0.95_z-score_All_dist_average_0.3_None_4"
    path = os.path.join(project_dir, "Archive", "results all 0.3", "pfeatures", folder1, "Left.xlsx")
    Results_DF_1 = pd.read_excel(path, index_col = 0)
    Results_DF_1["subject ID"] = Results_DF_1["subject ID"].map(lambda x: "average")

    folder1 = "0.95_z-score_All_dist_median_0.3_None_4"
    path = os.path.join(project_dir, "Archive", "results all 0.3", "pfeatures", folder1, "Left.xlsx")
    Results_DF_2 = pd.read_excel(path, index_col = 0)
    Results_DF_2["subject ID"] = Results_DF_2["subject ID"].map(lambda x: "median")


    vertical_concat = pd.concat([Results_DF_2, Results_DF_1], axis=0)

    print(stat.stat(vertical_concat[["subject ID", "mean(acc)"]], labels=["subject ID", "mean(acc)"], plot = True).head(100))
    plt.show()
    # logger.info("Re: \n{}\n".format(vertical_concat))


    
    


    logger.info("Done!!!")



if __name__ == "__main__":
    logger.info("Starting !!!")
    tic = timeit.default_timer()
    main()
    toc = timeit.default_timer()
    logger.info("Done ({:2.2f} process time)!!!\n\n\n".format(toc-tic))


