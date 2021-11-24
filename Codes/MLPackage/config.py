import os
import numpy as np

Pipeline = {
    "classifier": "knn_classifier", # knn_classifier   svm_classifier   Template_Matching_classifier
    "persentage": 0.95,
    "normilizing": "z-score",
    "feature_type": "all", # "all", "GRF_HC", "COA_HC", "GRF", "COA", "wt_COA", ## todo: "wt_GRF"
    "test_ratio": 0.30,
    "THRESHOLDs": np.linspace(0, 1, 100),
    "template_selection_method": "None",# "DEND" or MDIST
    "template_selection_k_cluster": 4,
    "verbose": 1,
}


CNN = {
    "model_name": "vgg16", # mobilenet, resnet50
    "weights": "imagenet", # weights = {none | imagenet}
    "include_top": False, # include_top = {True | False}
    "image_size": (300, 300, 3),
    "batch_size": 32, # batch_size
    "class_numbers": 97,
    "saving_path": "./Results/deep_model/Best_Model.hdf5",
    "epochs": 5,
    "validation_split": 0.1,
    "verbose": 1,
}

Template_Matching = {
    "mode": "dist",
    "criteria": "min",
    "random_runs": 50,
    "score": "A", # A = np.power(distance+1, -1) or B = 1/np.exp(distance)
    "verbose": 1,
}
SVM = {
    "kernel": "linear",
    "random_runs": 50,
    "verbose": 1,
}
KNN = {
    "n_neighbors": 5,
    "random_runs": 50,
    "metric": "euclidean",
    "weights": "uniform",
    "verbose": 1,
}


paths = {
    "project_dir": os.getcwd(),
    "results_dir": os.path.join(os.getcwd(), 'temp', 'Results.xlsx'),
    "feature_dir": os.path.join(os.getcwd(), "temp", "features_all.xlsx")
}



GRF_HC = ["GRF_HC_max_value_1", "GRF_HC_max_value_1_ind", "GRF_HC_max_value_2", "GRF_HC_max_value_2_ind", 
                               "GRF_HC_min_value",   "GRF_HC_min_value_ind",   "GRF_HC_mean_value",  "GRF_HC_std_value", "GRF_HC_sum_value"]

COA_HC = ['COA_HC_MDIST_RD', 'COA_HC_MDIST_AP', 'COA_HC_MDIST_ML', 'COA_HC_RDIST_RD', 'COA_HC_RDIST_AP', 'COA_HC_RDIST_ML', 
                               'COA_HC_TOTEX_RD', 'COA_HC_TOTEX_AP', 'COA_HC_TOTEX_ML', 'COA_HC_MVELO_RD', 'COA_HC_MVELO_AP', 'COA_HC_MVELO_ML', 
                               'COA_HC_RANGE_RD', 'COA_HC_RANGE_AP', 'COA_HC_RANGE_ML', 'COA_HC_AREA_CC',  'COA_HC_AREA_CE',  'COA_HC_AREA_SW', 
                               'COA_HC_MFREQ_RD', 'COA_HC_MFREQ_AP', 'COA_HC_MFREQ_ML', 'COA_HC_FDPD_RD',  'COA_HC_FDPD_AP',  'COA_HC_FDPD_ML', 
                               'COA_HC_FDCC',     'COA_HC_FDCE']

GRF = ["GRF_" + str(i) for i in range(100)]
COA_RD = ["COA_RD_" + str(i) for i in range(100)]
COA_AP = ["COA_AP_" + str(i) for i in range(100)]
COA_ML = ["COA_ML_" + str(i) for i in range(100)]

wt_GRF = ["wt_GRF_" + str(i) for i in range(116)]

wt_COA_RD = ["wt_COA_RD_" + str(i) for i in range(116)]
wt_COA_AP = ["wt_COA_AP_" + str(i) for i in range(116)]
wt_COA_ML = ["wt_COA_ML_" + str(i) for i in range(116)]

label = [ "subject ID", "left(0)/right(1)"]



