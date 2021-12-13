import os
import numpy as np
from pathlib import Path as Pathlb


configs = {
    "Pipeline": {
        "classifier": "Template_Matching_classifier", # knn_classifier   svm_classifier   Template_Matching_classifier
        "persentage": 0.95,
        "category": "image", # deep, image, hand_crafted
        "normilizing": "z-score",
        "feature_type": "COA", # "all", "GRF_HC", "COA_HC", "GRF", "COA", "wt_COA",  ## todo: "wt_GRF"
        "test_ratio": 0.30,
        "THRESHOLDs": np.linspace(0, 1, 100),
        "template_selection_method": "None",# "DEND" or MDIST
        "template_selection_k_cluster": 4,
        "verbose": 1,
        "Debug": True,
    },
    "CNN": {
        "base_model": "vgg16.VGG16", # vgg16.VGG16, resnet50.ResNet50, efficientnet.EfficientNetB0, mobilenet.MobileNet  inception_v3.InceptionV3
        "weights": "imagenet", 
        "include_top": False, 
        "image_size": (60, 40, 3),
        "batch_size": 32, 
        "class_numbers": 80,
        "saving_path": "./Results/deep_model/Best_Model.hdf5",
        "epochs": 10,
        "validation_split": 0.2,
        "verbose": 0,
        "image_feature": "CD", # CD, PTI, Tmax, Tmin, P50, P60, P70, P80, P90, P100, tile, fusion
    },
    "Template_Matching": {
        "mode": "dist",
        "criteria": "min",
        "random_runs": 50,
        "score": "A", # A = np.power(distance+1, -1) or B = 1/np.exp(distance)
        "verbose": 1,
    },
    "SVM": {
        "kernel": "linear",
        "random_runs": 50,
        "verbose": 1,
    },
    "KNN": {
        "n_neighbors": 5,
        "random_runs": 50,
        "metric": "euclidean",
        "weights": "uniform",
        "verbose": 1,
    },
    "paths": {
        "project_dir": os.getcwd(),

        "fig_dir": os.path.join(os.getcwd(), "Manuscripts", "src", "figures"),
        "tbl_dir": os.path.join(os.getcwd(), "Manuscripts", "src", "tables"),
        "results_dir": os.path.join(os.getcwd(), "results"),
        "temp_dir": os.path.join(os.getcwd(), "temp"),
        "log_path": os.path.join(os.getcwd(), 'logs'),

        "stepscan_dataset.h5": os.path.join(os.getcwd(), "Datasets", "stepscan", "footpressures_align.h5"),

        "casia_dataset.h5": os.path.join(os.getcwd(), "Datasets", "Casia-D", "footpressures_align.h5"),
        "casia_dataset-meta.npy": os.path.join(os.getcwd(), "Datasets", "Casia-D", "Metadata-barefoot.npy"),
        "casia_dataset.npy": os.path.join(os.getcwd(), "Datasets", "Casia-D", "Data-barefoot.npy"),

        "casia_all_feature.xlsx": os.path.join(os.getcwd(), "Datasets", "Casia-D", "casia_feature_all.xlsx"),
        "casia_image_feature.npy": os.path.join(os.getcwd(), "Datasets", "Casia-D", "casia_image_feature.npy"),

        "casia_deep_feature": os.path.join(os.getcwd(), "Datasets", "Casia-D", "deep_features"),

    }
}

Pathlb(configs["paths"]["log_path"]).mkdir(parents=True, exist_ok=True)
Pathlb(configs["paths"]["temp_dir"]).mkdir(parents=True, exist_ok=True)
Pathlb(configs["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)
Pathlb(configs["paths"]["fig_dir"]).mkdir(parents=True, exist_ok=True)
Pathlb(configs["paths"]["tbl_dir"]).mkdir(parents=True, exist_ok=True)
Pathlb(configs["paths"]["casia_deep_feature"]).mkdir(parents=True, exist_ok=True)

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

image_feature_name = ["CD", "PTI", "Tmax", "Tmin", "P50", "P60", "P70", "P80", "P90", "P100"]




