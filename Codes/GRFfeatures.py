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
from MLPackage import Butterworth


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
	logger.info("OS: {}".format( sys.platform))
	logger.info("Core Number: {}".format(multiprocessing.cpu_count()))

	data_path = os.path.join(project_dir, 'Datasets', 'datalist.npy')
	meta_path = os.path.join(project_dir, 'Datasets', 'metadatalist.npy')


	data = np.load(data_path)
	metadata = np.load(meta_path)
	logger.info("data shape: {}".format(data.shape))
	logger.info("metadata shape: {}".format(metadata.shape))

	GRF = list()
	handcraft_features = list()
    
	for sample in range(data.shape[0]):

		Sum = list()

		for frame in range(data[sample].shape[2]):
			temp = data[sample][:, :, frame]
			Sum.append(temp.sum())
		GRF.append(Sum)

		max_value_1 = np.max(Sum[:50])
		max_value_1_ind = np.argmax( Sum[:50] )
		max_value_2 = np.max(Sum[50:])
		max_value_2_ind = 50 + np.argmax(	Sum[50:] )

		min_value = np.min(Sum[max_value_1_ind:max_value_2_ind])
		min_value_ind = max_value_1_ind + np.argmin(	Sum[max_value_1_ind:max_value_2_ind] )

		mean_value = np.mean(GRF)
		std_value = np.std(GRF)
		sum_value = np.sum(GRF)

		handcraft_features.append([max_value_1, max_value_1_ind,
						max_value_2, max_value_2_ind,
						min_value, min_value_ind,
						mean_value, std_value, sum_value])



		# logger.info(metadata[sample,0:2])

		# plt.plot(range(100), Sum)
		# plt.figure()
		# plt.plot(range(100), Sum)
		# plt.show()
	

		# logger.info("Done!!")
		# sys.exit()


	saving_path = os.path.join(project_dir, 'temp', 'GRF.xlsx')#Datasets
	columnsName = ["feature_" + str(i) for i in range(len(GRF[0]))] + [ "subject ID", "left(0)/right(1)"]
	pd.DataFrame(np.concatenate((GRF,metadata[:,0:2]), axis=1), columns=columnsName).to_excel(saving_path)

	saving_path = os.path.join(project_dir, 'temp', 'handcraft_features.xlsx')#Datasets
	columnsName = ["max_value_1", "max_value_1_ind",
						"max_value_2", "max_value_2_ind",
						"min_value", "min_value_ind",
						"mean_value", "std_value", "sum_value"] + [ "subject ID", "left(0)/right(1)"]
	pd.DataFrame(np.concatenate((handcraft_features,metadata[:,0:2]), axis=1), columns=columnsName).to_excel(saving_path)



	logger.info("Done!!!")



if __name__ == "__main__":
	logger.info("Starting !!!")
	tic = timeit.default_timer()
	main()
	toc = timeit.default_timer()
	logger.info("Done ({:2.2f} process time)!!!\n\n\n".format(toc-tic))

