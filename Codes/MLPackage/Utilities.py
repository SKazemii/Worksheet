
import logging
import multiprocessing
import os
import pprint
import sys
import timeit
from pathlib import Path as Pathlb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_curve)

pd.options.mode.chained_assignment = None 


from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier as knn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))    
from MLPackage import config as cfg



# global variables

# Pipeline = cfg.configs["Pipeline"]
# CNN = cfg.configs["CNN"]
# Template_Matching = cfg.configs["Template_Matching"]
# SVM = cfg.configs["SVM"]
# KNN = cfg.configs["KNN"]
# paths = cfg.configs["paths"]

# num_pc = 0
columnsname = ["testID", "subject ID", "direction", "clasifier", "PCA", "num_pc", "classifier_parameters", "normilizing", "feature_type", "test_ratio", "mean(EER)", "t_idx", "mean(acc)", "mean(f1)", "# positive samples training", "# positive samples test", "# negative samples test", "len(FAR)", "len(FRR)"] + ["FAR_" + str(i) for i in range(100)] + ["FRR_" + str(i) for i in range(100)] 
time = int(timeit.default_timer() * 1_000_000)







log_path = os.path.join(cfg.configs["paths"]["project_dir"], 'logs')
def create_logger(level):
    loggerName = Pathlb(__file__).stem
    Pathlb(log_path).mkdir(parents=True, exist_ok=True)
    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    logger = logging.getLogger(loggerName)
    logger.setLevel(level)
    formatter_colored = logging.Formatter(blue + '[%(asctime)s]-' + yellow + '[%(name)s @%(lineno)d]' + reset + blue + '-[%(levelname)s]' + reset + bold_red + '\t\t%(message)s' + reset, datefmt='%m/%d/%Y %I:%M:%S %p ')
    formatter = logging.Formatter('[%(asctime)s]-[%(name)s @%(lineno)d]-[%(levelname)s]\t\t%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p ')
    file_handler = logging.FileHandler( os.path.join(log_path, loggerName + '_loger.log'), mode = 'w')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    stream_handler.setFormatter(formatter_colored)


    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
logger = create_logger(logging.DEBUG)




def knn_classifier(**kwargs):
    global time
    num_pc = kwargs["num_pc"]
    configs = kwargs["configs"]

    pos_samples = kwargs["pos_train"].shape
    temp = kwargs["neg_train"].sample(n = pos_samples[0])


    kwargs["pos_train"]["left(0)/right(1)"] = kwargs["pos_train"]["left(0)/right(1)"].map(lambda x: 1)
    temp["left(0)/right(1)"] = temp["left(0)/right(1)"].map(lambda x: 0)
    x_train = kwargs["pos_train"].append(temp)


    kwargs["pos_train"]["left(0)/right(1)"] = kwargs["pos_train"]["left(0)/right(1)"].map(lambda x: 1)
    kwargs["neg_train"]["left(0)/right(1)"] = kwargs["neg_train"]["left(0)/right(1)"].map(lambda x: 0)
    x_train = kwargs["pos_train"].append(kwargs["neg_train"])

    # cv = StratifiedKFold(n_splits=5, shuffle=True)


    clf = knn(n_neighbors=configs["KNN"]["n_neighbors"], metric=configs["KNN"]["metric"], weights=configs["KNN"]["weights"])


    # space = KNN

    # # define search
    # search = GridSearchCV(
    #     clf,
    #     space,
    #     scoring="accuracy",
    #     n_jobs=-1,
    #     cv=cv,
    #     refit=True,
    # )


    # execute search
    result = clf.fit(x_train.iloc[:, :-2].values, x_train.iloc[:, -1], )
    best_model = result#.best_estimator_

    y_pred = best_model.predict_proba(x_train.iloc[:, :-2].values)[:, 1]
    

    FAR, tpr, _ = roc_curve(x_train.iloc[:, -1], y_pred, pos_label=1)
    FRR = 1 - tpr

    EER, t_idx = compute_eer(FAR, FRR)
    
    


    acc = list()
    f1 = list()
    for _ in range(configs["KNN"]["random_runs"]):
        pos_samples = kwargs["pos_test"].shape
        temp = kwargs["neg_test"].sample(n = pos_samples[0])

        DF_temp = pd.concat([kwargs["pos_test"], temp])
        DF_temp["subject ID"] = DF_temp["subject ID"].map(lambda x: 1 if x == kwargs["sub"] else 0)

        y_pred = best_model.predict(DF_temp.iloc[:, :-2].values)

        acc.append( accuracy_score(DF_temp.iloc[:,-2].values, y_pred)*100 )
        f1.append(  f1_score(DF_temp.iloc[:,-2].values, y_pred)*100 )
      
    
    results = list()

    results.append([time, kwargs["sub"], kwargs["dir"], "KNN", configs["Pipeline"]["persentage"], num_pc, configs["KNN"], configs["Pipeline"]["normilizing"], kwargs["feature_type"], configs["Pipeline"]["test_ratio"]])

    results.append([EER, t_idx, np.mean(acc), np.mean(f1), kwargs["pos_train"].shape[0], kwargs["pos_test"].shape[0], kwargs["neg_test"].shape[0], len(FAR), len(FRR)])

    results.append(FAR)
    results.append(FRR)

    results = [val for sublist in results for val in sublist]
    return results



def svm_classifier(**kwargs):
    global time
    num_pc = kwargs["num_pc"]
    configs = kwargs["configs"]

    pos_samples = kwargs["pos_train"].shape
    temp = kwargs["neg_train"].sample(n = 100)#pos_samples[0])
    

    kwargs["pos_train"]["left(0)/right(1)"] = kwargs["pos_train"]["left(0)/right(1)"].map(lambda x: 1)
    temp["left(0)/right(1)"] = temp["left(0)/right(1)"].map(lambda x: 0)
    x_train = kwargs["pos_train"].append(temp)


    clf = svm.SVC(kernel=configs["SVM"]["kernel"] , probability=True)
    clf.fit(x_train.iloc[:, :-2], x_train.iloc[:, -1])


    y_pred = clf.predict_proba(x_train.iloc[:, :-2])[:, 1]
    

    # for _ in range(SVM["random_runs"]):
    FAR, tpr, threshold = roc_curve(x_train.iloc[:, -1], y_pred, pos_label=1)
    FRR = 1 - tpr

    EER, t_idx = compute_eer(FAR, FRR)
    
    
    

    acc = list()
    f1 = list()
    for _ in range(configs["SVM"]["random_runs"]):
        pos_samples = kwargs["pos_test"].shape
        temp = kwargs["neg_test"].sample(n = pos_samples[0])

        DF_temp = pd.concat([kwargs["pos_test"], temp])
        DF_temp["subject ID"] = DF_temp["subject ID"].map(lambda x: 1 if x == kwargs["sub"] else 0)

        y_pred = clf.predict(DF_temp.iloc[:, :-2])

        acc.append( accuracy_score(DF_temp.iloc[:,-2].values, y_pred)*100 )
        f1.append(  f1_score(DF_temp.iloc[:,-2].values, y_pred)*100 )
      
    
    results = list()

    results.append([time, kwargs["sub"], kwargs["dir"], "SVM", configs["Pipeline"]["persentage"], num_pc, configs["SVM"], configs["Pipeline"]["normilizing"], kwargs["feature_type"], configs["Pipeline"]["test_ratio"]])

    results.append([EER, t_idx, np.mean(acc), np.mean(f1), kwargs["pos_train"].shape[0], kwargs["pos_test"].shape[0], kwargs["neg_test"].shape[0], len(FAR), len(FRR)])

    results.append(FAR)
    results.append(FRR)

    results = [val for sublist in results for val in sublist]
    return results



def Template_Matching_classifier(**kwargs):
    global time
    configs = kwargs["configs"]
    num_pc = kwargs["num_pc"]
    FAR = list()
    FRR = list()
    EER = list()
    TH  = list()
    for _ in range(configs["Template_Matching"]["random_runs"]):
        pos_samples = kwargs["pos_train"].shape
        temp = kwargs["neg_train"].sample(n = pos_samples[0])


        distModel1, distModel2 = compute_score_matrix(kwargs["pos_train"].iloc[:, :-2].values,
                                                    temp.iloc[:, :-2].values,
                                                    mode = configs["Template_Matching"]["mode"], score = configs["Template_Matching"]["score"])

        Model_client, Model_imposter = model(distModel1,
                                                distModel2, 
                                                criteria=configs["Template_Matching"]["criteria"], 
                                                score=configs["Template_Matching"]["score"] )



        FAR_temp, FRR_temp = calculating_fxr(Model_client, Model_imposter, distModel1, distModel2, configs["Pipeline"]["THRESHOLDs"], configs["Template_Matching"]["score"])
        EER_temp = compute_eer(FAR_temp, FRR_temp)

        FAR.append(FAR_temp)
        FRR.append(FRR_temp)
        EER.append(EER_temp[0])
        TH.append(EER_temp[1])

    
    
    acc = list()
    f1 = list()
    t_idx = int(np.ceil(np.mean(TH)))

    for _ in range(configs["Template_Matching"]["random_runs"]):
        pos_samples = kwargs["pos_test"].shape
        temp = kwargs["neg_test"].sample(n = pos_samples[0])

        DF_temp = pd.concat([kwargs["pos_test"], temp])
        DF_temp["subject ID"] = DF_temp["subject ID"].map(lambda x: 1 if x == kwargs["sub"] else 0)


        distModel1 , distModel2 = compute_score_matrix(kwargs["pos_train"].iloc[:, :-2].values, DF_temp.iloc[:, :-2].values, mode=configs["Template_Matching"]["mode"], score=configs["Template_Matching"]["score"])
        Model_client, Model_test = model(distModel1, distModel2, criteria=configs["Template_Matching"]["criteria"], score=configs["Template_Matching"]["score"])


        y_pred = np.zeros((Model_test.shape))


        y_pred[Model_test > configs["Pipeline"]["THRESHOLDs"][t_idx]] = 1


        acc.append( accuracy_score(DF_temp.iloc[:,-2].values, y_pred)*100 )
        f1.append(  f1_score(DF_temp.iloc[:,-2].values, y_pred)*100 )
    
    results = list()

    results.append([time, kwargs["sub"], kwargs["dir"], "Template_Matching", configs["Pipeline"]["persentage"], num_pc, configs["Template_Matching"], configs["Pipeline"]["normilizing"], kwargs["feature_type"], configs["Pipeline"]["test_ratio"]])

    results.append([np.mean(EER), t_idx, np.mean(acc), np.mean(f1), kwargs["pos_train"].shape[0], kwargs["pos_test"].shape[0], kwargs["neg_test"].shape[0], len(FAR[0]), len(FRR[0])])
    results.append(list(np.mean(FAR, axis=0)))
    results.append(list(np.mean(FRR, axis=0)))

    results = [val for sublist in results for val in sublist]
    return results

    

def pipeline(configs):

    # set_sonfig(configs)

    classifier = configs["Pipeline"]["classifier"]

    feature_path = configs["paths"]["feature_dir"]
    if configs["Pipeline"]["Deep"]==True:
        feature_path = os.path.join(feature_path, configs["CNN"]["base_model"].split(".")[0]+'_features.xlsx')
    else:
        feature_path = os.path.join(feature_path, 'features_all.xlsx')


    DF_features_all = pd.read_excel(feature_path, index_col = 0)

    subjects = DF_features_all["subject ID"].unique()
    
    persentage = configs["Pipeline"]["persentage"]
    normilizing = configs["Pipeline"]["normilizing"]
    test_ratio = configs["Pipeline"]["test_ratio"]
    if configs["Pipeline"]["Deep"]==True:
        feature_type = configs["CNN"]["base_model"].split(".")[0]
    else:
        feature_type = configs["Pipeline"]["feature_type"]

    DF_features = extracting_features(DF_features_all, feature_type, configs)

    tic=timeit.default_timer()


    logger.info("Start [pipeline]:   +++   {}".format(classifier))


    results = list()
    subjects = [4, 5, 6]
    for subject in subjects:
        if (subject % 86) == 0: continue
        
        
        if (subject % 10) == 0 and configs["Pipeline"]["verbose"] is True:
            logger.info("--------------- Subject Number: {}".format(subject))
        

        for idx, direction in enumerate(["left_0", "right_1"]):

            DF_side = DF_features[DF_features["left(0)/right(1)"] == idx]
        
            DF_positive_samples = DF_side[DF_side["subject ID"] == subject]
            DF_negative_samples = DF_side[DF_side["subject ID"] != subject]

                
            DF_positive_samples_test = DF_positive_samples.sample(frac = test_ratio, 
                                                                    replace = False, 
                                                                    random_state = 2)
            DF_positive_samples_train = DF_positive_samples.drop(DF_positive_samples_test.index)

            DF_negative_samples_test = DF_negative_samples.sample(frac = test_ratio,
                                                                    replace = False, 
                                                                    random_state = 2)
            DF_negative_samples_train = DF_negative_samples.drop(DF_negative_samples_test.index)
            
            
            df_train = pd.concat([DF_positive_samples_train, DF_negative_samples_train])
            df_test = pd.concat([DF_positive_samples_test, DF_negative_samples_test])

            Scaled_train, Scaled_test = scaler(normilizing, df_train, df_test)
            
            # logger.info("direction: {}".format(direction))
            # logger.info("subject: {}".format(subject))
            # breakpoint()

            (DF_positive_samples_test, 
            DF_positive_samples_train, 
            DF_negative_samples_test, 
            DF_negative_samples_train,
            num_pc) = projector(persentage, 
                                                    feature_type, 
                                                    subject, 
                                                    df_train, 
                                                    df_test, 
                                                    Scaled_train, 
                                                    Scaled_test,
                                                    configs)

            logger.debug("DF_positive_samples_train.shape {}".format(DF_positive_samples_train.shape))    
            logger.debug("DF_negative_samples_train.shape {}".format(DF_negative_samples_train.shape))   
            logger.debug("DF_positive_samples_test.shape {}".format(DF_positive_samples_test.shape))    
            logger.debug("DF_negative_samples_test.shape {}".format(DF_negative_samples_test.shape))    
                


            # DF_positive_samples_train = template_selection(DF_positive_samples_train, 
            #                                                method=Pipeline["template_selection_method"], 
            #                                                k_cluster=Pipeline["template_selection_k_cluster"], 
            #                                                verbose=Pipeline["verbose"])
            # DF_negative_samples_train = template_selection(DF_negative_samples_train, 
            #                                                method="MDIST", 
            #                                                k_cluster=200, 
            #                                                verbose=Pipeline["verbose"])

            result = eval(classifier)(pos_train=DF_positive_samples_train, 
                                neg_train=DF_negative_samples_train, 
                                pos_test=DF_positive_samples_test, 
                                neg_test=DF_negative_samples_test, 
                                sub=subject, 
                                dir=direction,
                                num_pc=num_pc,
                                feature_type=feature_type, 
                                configs=configs)

            result = np.pad(result, (0, len(columnsname) - len(result)), 'constant')
                    
            results.append(result)
            # print(results)
        

    toc=timeit.default_timer()
    logger.info("End   [pipeline]:     ---    {}, \t\t Process time: {:.2f}  seconds".format(feature_type, toc - tic)) 

    return pd.DataFrame(results, columns=columnsname)



def extracting_features(DF_features_all, feature_type, configs):
    if configs["Pipeline"]["Deep"]==True:
        return DF_features_all
    elif feature_type == "all": #"all", "GRF_HC", "COA_HC", "GRF", "COA", "wt_GRF", "wt_COA"
        DF_features = DF_features_all.drop(columns=cfg.wt_GRF).copy()
    elif feature_type == "GRF_HC":
        DF_features = DF_features_all.loc[:, cfg.GRF_HC + cfg.label]
    elif feature_type == "COA_HC":
        DF_features = DF_features_all.loc[:, cfg.COA_HC + cfg.label]
    elif feature_type == "GRF":
        DF_features = DF_features_all.loc[:, cfg.GRF + cfg.label]
    elif feature_type == "COA":
        DF_features = DF_features_all.loc[:, cfg.COA_RD + cfg.COA_AP + cfg.COA_ML + cfg.label]
    elif feature_type == "wt_GRF":
        DF_features = DF_features_all.loc[:, cfg.wt_GRF + cfg.label]
    elif feature_type == "wt_COA":
        DF_features = DF_features_all.loc[:, cfg.wt_COA_RD + cfg.wt_COA_AP + cfg.wt_COA_ML + cfg.label]
    else:
        raise("Could not find the feature_type")
    return DF_features



def projector(persentage, feature_type, subject, df_train, df_test, Scaled_train, Scaled_test, configs):
    # global num_pc #todo 
    if persentage == 1.0:
        num_pc = Scaled_train.shape[1]
                

        columnsName = ["PC"+str(i) for i in list(range(1, num_pc+1))] + ["subject ID", "left(0)/right(1)"]
        DF_features_PCA_train = pd.DataFrame(np.concatenate((Scaled_train[:,:num_pc],df_train.iloc[:, -2:].values), axis = 1), columns = columnsName)
        DF_features_PCA_test = pd.DataFrame(np.concatenate((Scaled_test[:,:num_pc],df_test.iloc[:, -2:].values), axis = 1), columns = columnsName)

        DF_positive_samples_train = DF_features_PCA_train[DF_features_PCA_train["subject ID"] == subject]
        DF_negative_samples_train = DF_features_PCA_train[DF_features_PCA_train["subject ID"] != subject]
                
                
        DF_positive_samples_test = DF_features_PCA_test[DF_features_PCA_test["subject ID"] == subject]   
        DF_negative_samples_test = DF_features_PCA_test[DF_features_PCA_test["subject ID"] != subject]

    elif persentage != 1.0 and (feature_type in ["GRF_HC", "COA_HC", "GRF", "wt_GRF", configs["CNN"]["base_model"].split(".")[0]]):
        principal = PCA(svd_solver="full")
        PCA_out_train = principal.fit_transform(Scaled_train)
        PCA_out_test = principal.transform(Scaled_test)

        variance_ratio = np.cumsum(principal.explained_variance_ratio_)
        high_var_PC = np.zeros(variance_ratio.shape)
        high_var_PC[variance_ratio <= persentage] = 1

        loadings = principal.components_
        num_pc = int(np.sum(high_var_PC))




        columnsName = ["PC"+str(i) for i in list(range(1, num_pc+1))] + ["subject ID", "left(0)/right(1)"]
        DF_features_PCA_train = (pd.DataFrame(np.concatenate((PCA_out_train[:,:num_pc],df_train.iloc[:, -2:].values), axis = 1), columns = columnsName))
        DF_features_PCA_test = (pd.DataFrame(np.concatenate((PCA_out_test[:,:num_pc],df_test.iloc[:, -2:].values), axis = 1), columns = columnsName))

        DF_positive_samples_train = DF_features_PCA_train[DF_features_PCA_train["subject ID"] == subject]
        DF_negative_samples_train = DF_features_PCA_train[DF_features_PCA_train["subject ID"] != subject]
                
                
        DF_positive_samples_test = DF_features_PCA_test[DF_features_PCA_test["subject ID"] == subject]   
        DF_negative_samples_test = DF_features_PCA_test[DF_features_PCA_test["subject ID"] != subject]
            
    elif persentage != 1.0 and (feature_type in ["all", "COA", "wt_COA"]):
        tempa = []
        tempb = []
        if feature_type in "all":
            tempx = ["COA_RD", "COA_AP", "COA_ML", "COA_HC", "GRF", "GRF_HC", "wt_COA_RD", "wt_COA_AP", "wt_COA_ML"] # wt_GRF
        elif feature_type in "COA":
            tempx = ["COA_RD", "COA_AP", "COA_ML"]
        elif feature_type in "wt_COA":
            tempx = ["wt_COA_RD", "wt_COA_AP", "wt_COA_ML"]



        for i in range(len(tempx)):
            principal = PCA(svd_solver="full")
            # logger.info("feature type and shape: {} - {}".format(tempx[i], Scaled_train.loc[:, eval("cfg." + tempx[i])].shape))
            
            
            PCA_out_train = principal.fit_transform(Scaled_train.loc[:, eval("cfg." + tempx[i])])
            PCA_out_test = principal.transform(Scaled_test.loc[:, eval("cfg." + tempx[i])])

                    

            variance_ratio = np.cumsum(principal.explained_variance_ratio_)
            high_var_PC = np.zeros(variance_ratio.shape)
            high_var_PC[variance_ratio <= persentage] = 1

            loadings = principal.components_
            num_pc = int(np.sum(high_var_PC))



            tempa.append(PCA_out_train[:,:num_pc])
            tempb.append(PCA_out_test[:,:num_pc])

            del principal

                   
        for i in range(len(tempx)-1):
            tempa[len(tempx)-1] = np.concatenate((tempa[len(tempx)-1],tempa[i]), axis=1)
            tempb[len(tempx)-1] = np.concatenate((tempb[len(tempx)-1],tempb[i]), axis=1)

        num_pc = tempa[len(tempx)-1].shape[1]

        columnsName = ["PC_" + str(i) for i in list(range(1, num_pc+1))] + ["subject ID", "left(0)/right(1)"]

        DF_features_PCA_train = pd.DataFrame(np.concatenate((tempa[len(tempx)-1],df_train.iloc[:, -2:].values), axis = 1), columns = columnsName)
        DF_features_PCA_test = pd.DataFrame(np.concatenate((tempb[len(tempx)-1],df_test.iloc[:, -2:].values), axis = 1), columns = columnsName)

        DF_positive_samples_train = DF_features_PCA_train[DF_features_PCA_train["subject ID"] == subject]
        DF_negative_samples_train = DF_features_PCA_train[DF_features_PCA_train["subject ID"] != subject]
                
                
        DF_positive_samples_test = DF_features_PCA_test[DF_features_PCA_test["subject ID"] == subject]   
        DF_negative_samples_test = DF_features_PCA_test[DF_features_PCA_test["subject ID"] != subject]

    return DF_positive_samples_test,DF_positive_samples_train,DF_negative_samples_test,DF_negative_samples_train, num_pc



def scaler(normilizing, df_train, df_test):
    if normilizing == "minmax":
        scaling = preprocessing.MinMaxScaler()
        Scaled_train = scaling.fit_transform(df_train.iloc[:, :-2])
        Scaled_test = scaling.transform(df_test.iloc[:, :-2])

    elif normilizing == "z-score":
        scaling = preprocessing.StandardScaler()
        Scaled_train = scaling.fit_transform(df_train.iloc[:, :-2])
        Scaled_test = scaling.transform(df_test.iloc[:, :-2])

    Scaled_train = pd.DataFrame(Scaled_train, columns=df_train.columns[:-2])
    Scaled_test = pd.DataFrame(Scaled_test, columns=df_test.columns[:-2])
    return Scaled_train, Scaled_test



def compute_score_matrix(positive_samples, negative_samples, mode="dist", score = None):
    """ Returns score matrix of trmplate matching"""

    positive_model = np.zeros((positive_samples.shape[0], positive_samples.shape[0]))
    negative_model = np.zeros((positive_samples.shape[0], negative_samples.shape[0]))

    if mode == "dist":

        for i in range(positive_samples.shape[0]):
            for j in range(positive_samples.shape[0]):
                positive_model[i, j] = distance.euclidean(
                    positive_samples[i, :], positive_samples[j, :]
                )
            for j in range(negative_samples.shape[0]):
                negative_model[i, j] = distance.euclidean(
                    positive_samples[i, :], negative_samples[j, :]
                )
        if score != None:
            return compute_similarity(positive_model, score), compute_similarity(negative_model, score)
        elif score == None:
            return positive_model, negative_model

    elif mode == "corr":

        for i in range(positive_samples.shape[0]):
            for j in range(positive_samples.shape[0]):
                positive_model[i, j] = abs(np.corrcoef(
                    positive_samples[i, :], positive_samples[j, :]
                )[0,1])

            for j in range(negative_samples.shape[0]):
                negative_model[i, j] = abs(np.corrcoef(
                    positive_samples[i, :], negative_samples[j, :]
                )[0,1])
        return positive_model, negative_model



def model(distModel1, distModel2, criteria = "average", score = None ):
    if score is None:
        if criteria == "average":
            # model_client = (np.sum(distModel1, axis = 0))/(distModel1.shape[1]-1)
            model_client = np.mean(np.ma.masked_where(distModel1==0,distModel1), axis = 0)
            model_client = np.expand_dims(model_client,-1)
            
            model_imposter = (np.sum(distModel2, axis = 0))/(distModel1.shape[1])
            model_imposter = np.expand_dims(model_imposter, -1)
                
        elif criteria == "min":

            model_client = np.min(np.ma.masked_where(distModel1==0,distModel1), axis = 0)
            model_client = np.expand_dims(model_client,-1)
            
            model_imposter = np.min(np.ma.masked_where(distModel2==0,distModel2), axis = 0)
            model_imposter = np.expand_dims(model_imposter, -1)
                    
        elif criteria == "median":
            model_client = np.median(distModel1, axis = 0)
            model_client = np.expand_dims(model_client,-1)
            

            model_imposter = np.median(distModel2, axis = 0)
            model_imposter = np.expand_dims(model_imposter, -1)

    if score is not None:
        if criteria == "average":
            model_client = np.mean(np.ma.masked_where(distModel1==1,distModel1), axis = 0)
            model_client = np.expand_dims(model_client,-1)
            
            model_imposter = (np.sum(distModel2, axis = 0))/(distModel1.shape[1])
            model_imposter = np.expand_dims(model_imposter, -1)
                
        elif criteria == "min":

            model_client = np.max(np.ma.masked_where(distModel1==1,distModel1), axis = 0)
            model_client = np.expand_dims(model_client,-1)
            
            model_imposter = np.max(np.ma.masked_where(distModel2==1,distModel2), axis = 0)
            model_imposter = np.expand_dims(model_imposter, -1)
                    
        elif criteria == "median":
            model_client = np.median(distModel1, axis = 0)
            model_client = np.expand_dims(model_client,-1)            

            model_imposter = np.median(distModel2, axis = 0)
            model_imposter = np.expand_dims(model_imposter, -1)
    

    return model_client, model_imposter



def calculating_fxr(Model_client, Model_imposter, distModel1, distModel2, THRESHOLDs, score):
    """ Returns FAR and FRR our model """

    FRR_temp = list()
    FAR_temp = list()

    if score is not None:
        for tx in THRESHOLDs:
            E1 = np.zeros((Model_client.shape))
            E1[Model_client < tx] = 1
            FRR_temp.append(np.sum(E1)/(distModel1.shape[1]))


            E2 = np.zeros((Model_imposter.shape))
            E2[Model_imposter > tx] = 1
            FAR_temp.append(np.sum(E2)/distModel2.shape[1])

    elif score is None:
        for tx in THRESHOLDs:
            E1 = np.zeros((Model_client.shape))
            E1[Model_client > tx] = 1
            FRR_temp.append(np.sum(E1)/(distModel1.shape[1]))


            E2 = np.zeros((Model_imposter.shape))
            E2[Model_imposter < tx] = 1
            FAR_temp.append(np.sum(E2)/distModel2.shape[1])
    return FAR_temp, FRR_temp
    


def compute_eer(fpr, fnr):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    abs_diffs = np.abs(np.subtract(fpr, fnr)) 
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))

    return eer, min_index



def compute_similarity(distance, mode = "A"):
    """change distance score to similarity score."""

    distance = np.array(distance)
    if mode == "A":
        return np.power(distance+1, -1) 
    elif mode =="B":
        return 1/np.exp(distance)



def template_selection(DF_positive_samples_train,  method, k_cluster, verbose=1):
    if method == "DEND":
        kmeans = KMeans(n_clusters = k_cluster)
        kmeans.fit(DF_positive_samples_train.iloc[:, :-2].values)
        clusters = np.unique(kmeans.labels_)

        for i, r in DF_positive_samples_train.reset_index(drop=True).iterrows():
            DF_positive_samples_train.loc[i,"dist"] = distance.euclidean(kmeans.cluster_centers_[kmeans.labels_[i]], r[:-2].values)
            DF_positive_samples_train.loc[i,"label"] = kmeans.labels_[i]
        DF_positive_samples_train_clustered = pd.DataFrame(np.empty((k_cluster,DF_positive_samples_train.shape[1]-2)))
        for cluster in clusters:
            mean_cluster = DF_positive_samples_train[DF_positive_samples_train["label"] == cluster].sort_values(by=['dist'])
            DF_positive_samples_train_clustered.iloc[cluster, :] = mean_cluster.iloc[0,:-2]
        if verbose: 
            logger.info(f"Clustered data size: { DF_positive_samples_train_clustered.shape}")
        DF_positive_samples_train_clustered.columns = DF_positive_samples_train.columns.values[:-2]

    elif method == "MDIST":
        A = distance.squareform(distance.pdist(DF_positive_samples_train.iloc[:, :-2].values)).mean(axis=1)
        i = np.argsort(A)[:k_cluster]
        DF_positive_samples_train_clustered = DF_positive_samples_train.iloc[i, :]
        DF_positive_samples_train_clustered.columns = DF_positive_samples_train.columns.values

    elif method == "None":
        DF_positive_samples_train_clustered = DF_positive_samples_train
        DF_positive_samples_train_clustered.columns = DF_positive_samples_train.columns.values

    return DF_positive_samples_train_clustered

'''
def matching_score_level(DF_positive_samples_train,  method, k_cluster, verbose = verbose):
    scores_vector_client = list()
    scores_vector_imposter = list()
    for criteria in criterias:

        Model_client, Model_imposter = perf.model(distModel1,
                                                    distModel2, 
                                                    criteria = criteria, 
                                                    score = score )

        scores_vector_imposter.append(Model_imposter)
        scores_vector_client.append(Model_client)
        # sys.exit()

    ###
    scores_vector_client = (np.array(scores_vector_client)[:,:,0])
    scores_vector_imposter = (np.array(scores_vector_imposter)[:,:,0])
    # print(scores_vector_imposter.shape)

    knn = KNeighborsClassifier(n_neighbors=7)
            
    X = (np.concatenate((scores_vector_client, scores_vector_imposter),axis=1)).T
    # print(X.shape)

    zeros = (np.zeros((scores_vector_imposter.shape[1])))
    ones = (np.ones((scores_vector_client.shape[1])))
    Y = (np.concatenate((ones, zeros),axis=0))
    # print(Y)

    knn.fit(X,Y)

    Model_client = knn.predict_proba(scores_vector_client.T)[:,1]
    Model_imposter = knn.predict_proba(scores_vector_imposter.T)[:,1]


def plot(FAR_L, FRR_L, FAR_R, FRR_R, labels):
    for idx in range(len(FAR_L)):
        plt.subplot(1,2,1)
        auc = round((1 + np.trapz( FRR_L[idx], FAR_L[idx])),2)
        # label=a[idx] #+ ' AUC = ' + str(round(auc, 2))

        plt.plot(FAR_L[idx], FRR_L[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=labels[idx] + str(auc), clip_on=False)

        plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Acceptance Rate')
        plt.ylabel('False Rejection Rate')
        plt.title('ROC curve, left side')
        plt.gca().set_aspect('equal')
        plt.legend(loc="best")

        plt.subplot(1,2,2)
        auc = round((1 + np.trapz( FRR_L[idx], FAR_L[idx])),2)
        plt.plot(FAR_R[idx], FRR_R[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=labels[idx] + str(auc), clip_on=False)

        plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Acceptance Rate')
        plt.ylabel('False Rejection Rate')
        plt.title('ROC curve, Right side')
        plt.gca().set_aspect('equal')
        plt.legend(loc="best")


def ROC_plot(TPR, FPR):
    """plot ROC curve"""
    plt.figure()
    auc = 1 * np.trapz(TPR, FPR)

    plt.plot(FPR, TPR, linestyle='--', marker='o', color='darkorange', lw = 2, label='ROC curve', clip_on=False)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve, AUC = %.2f'%auc)
    plt.legend(loc="lower right")
    # plt.savefig(path + 'AUC.png')


def ROC_plot_v2(FPR, FNR,THRESHOLDs):
    """plot ROC curve"""
    # fig = plt.figure()
    # color = ['darkorange', 'orange']
    # auc = 1/(1 + np.trapz( FPR,FNR))
    # plt.plot(FPR, FNR, linestyle='--', marker='o', color=color[path], lw = 2, label='ROC curve', clip_on=False)
    # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Acceptance Rate')
    # plt.ylabel('False Rejection Rate')
    # plt.title('ROC curve, AUC = %.2f'%auc)
    # plt.legend(loc="best")
    # path1 = path + "_ROC.png"

    # plt.savefig(path1)

    plt.figure()
    plt.plot(THRESHOLDs, FPR, linestyle='--', marker='o', color='darkorange', lw = 2, label='FAR curve', clip_on=False)
    plt.plot(THRESHOLDs, FNR, linestyle='--', marker='o', color='navy', lw = 2, label='FRR curve', clip_on=False)

    EER,_ = compute_eer(FPR, FNR)
    # path2 = path + "_ACC.png"
    plt.title('FPR and FNR curve, EER = %.2f'%EER)
    plt.legend(loc="upper right")
    plt.xlabel('Threshold')
    plt.show()
    # plt.savefig(path2)
    # plt.close('all')


def performance(model1, model2, path):
    if False:

        # THRESHOLDs = np.linspace(0, 2*np.max(model1), 10)
        THRESHOLDs = np.linspace(0, 300, 1000)
        FN = list();   TP = list();  TN = list();  FP = list()
        ACC = list(); FDR = list(); FNR = list(); FPR = list()
        NPV = list(); PPV = list(); TNR = list(); TPR = list()

        for idx, thresh in enumerate(THRESHOLDs):
            TPM = np.zeros((model1.shape))
            TPM[model1 < thresh] = 1
            TP.append(TPM.sum()/16)
            

            FNM = np.zeros((model1.shape))
            FNM[model1 >= thresh] = 1
            FN.append(FNM.sum()/16)

            FPM = np.zeros((model2.shape))
            FPM[model2 < thresh] = 1
            FP.append(FPM.sum()/16)

            TNM = np.zeros((model2.shape))
            TNM[model2 >= thresh] = 1
            TN.append(TNM.sum()/16)

            # Sensitivity, hit rate, recall, or true positive rate
            # reflects the classifier’s ability to detect members of the positive class (pathological state)
            TPR.append(TP[idx] / (TP[idx]  + FN[idx] ))
            # Specificity or true negative rate
            # reflects the classifier’s ability to detect members of the negative class (normal state)
            TNR.append(TN[idx]  / (TN[idx]  + FP[idx] ))
            # Precision or positive predictive value
            # PPV.append(TP[idx]  / (TP[idx]  + FP[idx] ))
            # Negative predictive value
            # NPV.append(TN[idx]  / (TN[idx]  + FN[idx] ))
            # Fall out or false positive rate
            # reflects the frequency with which the classifier makes a mistake by classifying normal state as pathological
            FPR.append(FP[idx]  / (FP[idx]  + TN[idx] ))
            # False negative rate
            # reflects the frequency with which the classifier makes a mistake by classifying pathological state as normal
            FNR.append(FN[idx]  / (TP[idx]  + FN[idx] ))
            # False discovery rate
            # FDR.append(FP[idx]  / (TP[idx]  + FP[idx] ))
            # Overall accuracy
            ACC.append((TP[idx]  + TN[idx] ) / (TP[idx]  + FP[idx]  + FN[idx]  + TN[idx] ))

        EER, minindex = compute_eer(FPR, FNR)



        if False:
            # print("\n#################################################################################################")
            # print("#################################################################################################\n")
            # print("THRESHOLDs:                                                                          {}".format(THRESHOLDs))
            # print("EER:                                                                                 {}".format(EER))
            # print("False Positive (FP):                                                                 {}".format(FP))
            # print("False Negative (FN):                                                                 {}".format(FN))
            # print("True Positive (TP):                                                                  {}".format(TP))
            # print("True Negative (TN):                                                                  {}".format(TN))
            # print("True Positive Rate (TPR)(Recall):                                                    {}".format(TPR))
            # print("True Negative Rate (TNR)(Specificity):                                               {}".format(TNR))
            # print("Positive Predictive Value (PPV)(Precision):                                          {}".format(PPV))
            # print("Negative Predictive Value (NPV):                                                     {}".format(NPV))
            # print(
            #      "False Positive Rate (FPR)(False Match Rate (FMR))(False Acceptance Rate (FAR)):      {}".format(
            #         FPR
            #     )
            # )
            # print(
            #      "False Negative Rate (FNR)(False Non-Match Rate (FNMR))(False Rejection Rate (FRR)):  {}".format(
            #         FNR
            #     )
            # )
            # print("False Discovery Rate (FDR):                                                          {}".format(FDR))
            # print("Overall accuracy (ACC):                                                              {}".format(ACC))
            pass
        if False:
            print("\n#################################################################################################")
            print("\n#################################################################################################")
            print("#################################################################################################\n")
            print("THRESHOLDs:                                                                          {}".format(THRESHOLDs[minindex]))
            print("EER:                                                                                 {}".format(EER))
            print("False Positive (FP):                                                                 {}".format(FP[minindex]))
            print("False Negative (FN):                                                                 {}".format(FN[minindex]))
            print("True Positive (TP):                                                                  {}".format(TP[minindex]))
            print("True Negative (TN):                                                                  {}".format(TN[minindex]))
            print("True Positive Rate (TPR)(Recall):                                                    {}".format(TPR[minindex]))
            print("True Negative Rate (TNR)(Specificity):                                               {}".format(TNR[minindex]))
            print("Positive Predictive Value (PPV)(Precision):                                          {}".format(PPV[minindex]))
            print("Negative Predictive Value (NPV):                                                     {}".format(NPV[minindex]))
            print(
                "False Positive Rate (FPR)(False Match Rate (FMR))(False Acceptance Rate (FAR)):      {}".format(
                    FPR[minindex]
                )
            )
            print(
                "False Negative Rate (FNR)(False Non-Match Rate (FNMR))(False Rejection Rate (FRR)):  {}".format(
                    FNR[minindex]
                )
            )
            print("False Discovery Rate (FDR):                                                          {}".format(FDR[minindex]))
            print("Overall accuracy (ACC):                                                              {}".format(ACC[minindex]))
            print("\n#################################################################################################")

        with open(os.path.join(path, "file.txt"), "a") as f:
            f.write("\n#################################################################################################")
            f.write("\n#################################################################################################")
            f.write("\n#################################################################################################\n")
            f.write("\nTHRESHOLDs:                                                                          {}".format(THRESHOLDs[minindex]))
            f.write("\nEER:                                                                                 {}".format(EER))
            f.write("\nFalse Positive (FP):                                                                 {}".format(FP[minindex]))
            f.write("\nFalse Negative (FN):                                                                 {}".format(FN[minindex]))
            f.write("\nTrue Positive (TP):                                                                  {}".format(TP[minindex]))
            f.write("\nTrue Negative (TN):                                                                  {}".format(TN[minindex]))
            f.write("\nTrue Positive Rate (TPR)(Recall):                                                    {}".format(TPR[minindex]))
            f.write("\nTrue Negative Rate (TNR)(Specificity):                                               {}".format(TNR[minindex]))
            # f.write("\nPositive Predictive Value (PPV)(Precision):                                          {}".format(PPV[minindex]))
            # f.write("\nNegative Predictive Value (NPV):                                                     {}".format(NPV[minindex]))
            f.write(
                "\nFalse Positive Rate (FPR)(False Match Rate (FMR))(False Acceptance Rate (FAR)):      {}".format(
                    FPR[minindex]
                )
            )
            f.write(
                "\nFalse Negative Rate (FNR)(False Non-Match Rate (FNMR))(False Rejection Rate (FRR)):  {}".format(
                    FNR[minindex]
                )
            )
            # f.write("\nFalse Discovery Rate (FDR):                                                          {}".format(FDR[minindex]))
            f.write("\nOverall accuracy (ACC):                                                              {}".format(ACC[minindex]))
            f.write("\n#################################################################################################")
        ROC_plot(TPR, FPR, path)
        ROC_plot_v2(FPR, FNR, THRESHOLDs, path)
        return EER, FPR, FNR






    # np.save("./Datasets/distModel1.npy", distModel1)
    # np.save("./Datasets/distModel2.npy", distModel2)
'''

def collect_results(result):
    global columnsname, time
    excel_path = cfg.configs["paths"]["results_dir"]

    if os.path.isfile(excel_path):
        Results_DF = pd.read_excel(excel_path, index_col = 0)
    else:
        Results_DF = pd.DataFrame(columns=columnsname)

    Results_DF = Results_DF.append(result)
    try:
        Results_DF.to_excel(excel_path, columns=columnsname)
    except:
        Results_DF.to_excel(os.path.join(os.getcwd(), 'temp', 'Results'+str(time)+'.xlsx'), columns=columnsname)


# def set_sonfig(configs):
#     global paths, Pipeline, CNN, Template_Matching, SVM, KNN

#     Pipeline = configs["Pipeline"]
#     CNN = configs["CNN"]
#     Template_Matching = configs["Template_Matching"]
#     SVM = configs["SVM"]
#     KNN = configs["KNN"]
#     paths = configs["paths"]
    
def main():
    configs = cfg.configs
    configs["Pipeline"]["classifier"] = "knn_classifier"


    z = pipeline(configs)  

    collect_results(z)

    # 
    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    # for ii in test_ratios:
    #     folder = str(0.95) + "_z-score_" + str(-3) + "_dist_median_" +  str(ii)      
    #     pool.apply_async(fcn, args=(DF_features_all, folder, features_excel, 2, "None"), callback=collect_results)
        
    # pool.close()
    # pool.join()


if __name__ == "__main__":
    logger.info("Starting [main] !!!")
    tic = timeit.default_timer()
    main()
    toc = timeit.default_timer()
    logger.info("Done [main] ({:2.2f} process time)!!!\n\n\n".format(toc-tic))
