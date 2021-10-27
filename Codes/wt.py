import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, os, logging, timeit
from pathlib import Path as Pathlb


from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import itertools, multiprocessing


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))	
from MLPackage import util as ut



import pywt

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

def collect_results(result):
    global excel_path
    if os.path.isfile(results_dir):
        Results_DF = pd.read_excel(results_dir, index_col = 0)
    else:
        Results_DF = pd.DataFrame(columns=ut.cols)
        Results_DF.to_excel(results_dir)

    Results_DF = Results_DF.append(result)
    Results_DF.to_excel(results_dir)




def calculating_wt(features_excel, waveletname, pywt_mode, wavelet_level):
	DF_features = pd.read_excel(os.path.join(dataset_dir, features_excel + ".xlsx"), index_col = 0)


	logger.info("Feature shape: {}".format(DF_features.shape))

	wt_feature = list()
	for _, row in DF_features.iterrows():

		dwt_coeff_RD = pywt.wavedec(row[  0:100], waveletname, mode=pywt_mode, level=wavelet_level)
		dwt_coeff_AP = pywt.wavedec(row[100:200], waveletname, mode=pywt_mode, level=wavelet_level)
		dwt_coeff_ML = pywt.wavedec(row[200:300], waveletname, mode=pywt_mode, level=wavelet_level)

		dwt_coeff_RD = np.concatenate(dwt_coeff_RD).ravel()
		dwt_coeff_AP = np.concatenate(dwt_coeff_AP).ravel()
		dwt_coeff_ML = np.concatenate(dwt_coeff_ML).ravel()


		wt_feature.append(np.concatenate((dwt_coeff_RD , dwt_coeff_AP , dwt_coeff_ML, row[-2:].values), axis=0))
	
	columnsName = ["feature_" + str(i) for i in range(len(wt_feature[0])-2)] + [ "subject ID", "left(0)/right(1)"]
	return pd.DataFrame(wt_feature, columns=columnsName)

def main():
	
	print(pywt.families(short=False))
	print(pywt.Modes.modes)
	for family in pywt.families():
		print("%s family: " % family + ", ".join(pywt.wavelist(family)))	# print(dir(pywt))
	waveletname = "coif1"
	wavelet_level = pywt.dwt_max_level(100, waveletname)
	pywt_mode = "constant"


	logger.info("OS: {}".format(sys.platform))
	logger.info("Core Numbers: {}".format(multiprocessing.cpu_count()))
	

	for features_excel in ["COPs", "COAs-simple", "COAs-otsu"]:
		logger.info("Working on : {}\n".format(features_excel))
		
		
		DF_wt = calculating_wt(features_excel, waveletname, pywt_mode, wavelet_level)
		DF_wt.to_excel(os.path.join(dataset_dir, features_excel + "_wt.xlsx"))
		
			



if __name__ == '__main__': 
    logger.info("Starting !!!")
    tic = timeit.default_timer()
    main()
    toc = timeit.default_timer()
    logger.info("Done ({:2.2f} process time)!!!\n\n\n".format(toc-tic))
