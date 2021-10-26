
import logging
from pathlib import Path as Pathlb
import multiprocessing



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys, os, timeit

from scipy.spatial import distance

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve


pd.options.mode.chained_assignment = None 


TH_dev = 1000
THRESHOLDs = np.linspace(0, 1, TH_dev)
test_ratios = [0]#.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
persentages = [1.0]
modes = ["dist", "corr"]
model_types = [ "min"]#"median", "min", "average"]
normilizings = ["z-score"]#, "minmax"]
verbose = False
Debug = False
random_test_acc = 0
template_selection_methods = ["None"]#, "DEND", "MDIST"]
k_clusters = [4]#, 7, 12]

features_types = ["pfeatures", "afeatures-simple"]#, "afeatures-otsu", "COAs-otsu", "COAs-simple", "COPs"]
color = ['darkorange', 'navy', 'red', 'greenyellow', 'lightsteelblue', 'lightcoral', 'olive', 'mediumpurple', 'khaki', 'hotpink', 'blueviolet']

working_path = os.getcwd()
score = "A"
feature_names = ["MDIST", "RDIST", "TOTEX", "MVELO", "RANGE", "AREAXX", "MFREQ", "FDPD", "FDCX"]

cols = ["Feature_Type", "Mode", "Criteria", 
        "Test_Size", "Normalizition", "Features_Set",
        "PCA", "Time", "Number_of_PCs",
        "template-selection-method", "k-cluster",
        "Mean_Acc_L", "Mean_f1_L", "Mean_EER_L_tr", "sklearn_EER_L", "Mean_EER_L_te", "Mean_sample_training_L", "Mean_sample_test_L",
        "Mean_Acc_R", "Mean_f1_R", "Mean_EER_R_tr", "sklearn_EER_R", "Mean_EER_R_te", "Mean_sample_training_R", "Mean_sample_test_R"] + ["FAR_L_" + str(i) for i in range(TH_dev)] + ["FRR_L_" + str(i) for i in range(TH_dev)] + ["FAR_R_" + str(i) for i in range(TH_dev)] + ["FRR_R_" + str(i) for i in range(TH_dev)]



def create_logger(level):
    loggerName = Pathlb(__file__).stem

    log_path = os.path.join(working_path, 'logs')
    Pathlb(log_path).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(loggerName)
    logger.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s]-[%(name)s @ %(lineno)d]-[%(levelname)s]\t%(message)s', datefmt='%m/%d/%y %I:%M:%S %p')
    file_handler = logging.FileHandler( os.path.join(log_path, loggerName + '_loger.log'), mode = 'w')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

logger = create_logger(logging.INFO)

from sklearn.cluster import KMeans

def template_selection(DF_positive_samples_train,  method, k_cluster, verbose = verbose):
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
            logger.info("Clustered data size: ", DF_positive_samples_train_clustered.shape)
    elif method == "MDIST":
        A = distance.squareform(distance.pdist(DF_positive_samples_train.iloc[:, :-2].values)).mean(axis=1)
        i = np.argsort(A)[:k_cluster]
        DF_positive_samples_train_clustered = DF_positive_samples_train.iloc[i, :]
    return DF_positive_samples_train_clustered



def matching_score_level(DF_positive_samples_train,  method, k_cluster, verbose = verbose):
    scores_vector_client = list()
    scores_vector_imposter = list()
    for model_type in model_types:

        Model_client, Model_imposter = perf.model(distModel1,
                                                    distModel2, 
                                                    model_type = model_type, 
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



def fcn(DF_features_all, foldername, features_excel, k_cluster, template_selection_method = "DEND"):
    
    subjects = (DF_features_all["subject ID"].unique())
    
    persentage = float(foldername.split('_')[0])
    normilizing = foldername.split('_')[1]
    x = int(foldername.split('_')[2])
    mode = foldername.split('_')[3]  
    model_type = foldername.split('_')[4]
    test_ratio = float(foldername.split('_')[5])

    if x == -3:
        DF_features = DF_features_all.copy()
        feat_name = "All"
    else:
        DF_features = DF_features_all.copy()
        DF_features.drop(DF_features.columns[[range(x+3,DF_features_all.shape[1]-2)]], axis = 1, inplace = True)
        DF_features.drop(DF_features.columns[[range(0,x)]], axis = 1, inplace = True)
        feat_name = feature_names[int(x/3)]


    tic=timeit.default_timer()
    folder = str(persentage) + "_" + normilizing + "_" + feat_name + "_" + mode + "_" + model_type + "_" +  str(test_ratio) + "_" + template_selection_method + "_" + str(k_cluster)
    folder_path = os.path.join(working_path, 'results', features_excel, folder)
    Pathlb(folder_path).mkdir(parents=True, exist_ok=True)

    EER_L = list(); FAR_L = list(); FRR_L = list()
    EER_R = list(); FAR_R = list(); FRR_R = list()


    ACC_L = list(); ACC_R = list()
    


    logger.info("Start:   +++   {}".format(folder))

    for subject in subjects:
        if (subject % 86) == 0:
            continue
        
        if (subject % 30) == 0 and verbose is True:
            logger.info("--------------- Subject Number: {}".format(subject))
        
        if (Debug is True) and ((subject % 10) == 0):
            logger.debug("--------------- Subject Number: {}".format(subject))
            break

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
            
            num_pc = 11
            # df_train = pd.concat([DF_positive_samples_train, DF_negative_samples_train])
            # df_test = pd.concat([DF_positive_samples_test, DF_negative_samples_test])

            # if normilizing == "minmax":
            #     scaling = preprocessing.MinMaxScaler()
            #     Scaled_train = scaling.fit_transform(df_train.iloc[:, :-2])
            #     Scaled_test = scaling.transform(df_test.iloc[:, :-2])


            # elif normilizing == "z-score":
            #     scaling = preprocessing.StandardScaler()
            #     Scaled_train = scaling.fit_transform(df_train.iloc[:, :-2])
            #     Scaled_test = scaling.transform(df_test.iloc[:, :-2])
            

            # principal = PCA()
            # PCA_out_train = principal.fit_transform(Scaled_train)
            # PCA_out_test = principal.transform(Scaled_test)

            # variance_ratio = np.cumsum(principal.explained_variance_ratio_)
            # high_var_PC = np.zeros(variance_ratio.shape)
            # high_var_PC[variance_ratio <= persentage] = 1

            # loadings = principal.components_
            # num_pc = int(np.sum(high_var_PC))




            # columnsName = ["PC"+str(i) for i in list(range(1, num_pc+1))] + ["subject ID", "left(0)/right(1)"]
            # DF_features_PCA_train = (pd.DataFrame(np.concatenate((PCA_out_train[:,:num_pc],df_train.iloc[:, -2:].values), axis = 1), columns = columnsName))
            # DF_features_PCA_test = (pd.DataFrame(np.concatenate((PCA_out_test[:,:num_pc],df_test.iloc[:, -2:].values), axis = 1), columns = columnsName))

            # DF_positive_samples_train = DF_features_PCA_train[DF_features_PCA_train["subject ID"] == subject]
            # DF_negative_samples_train = DF_features_PCA_train[DF_features_PCA_train["subject ID"] != subject]
            
            
            # DF_positive_samples_test = DF_features_PCA_test[DF_features_PCA_test["subject ID"] == subject]   
            # DF_negative_samples_test = DF_features_PCA_test[DF_features_PCA_test["subject ID"] != subject]
            

            if template_selection_method != "None":
                DF_positive_samples_train = template_selection(DF_positive_samples_train, method = template_selection_method, k_cluster = k_cluster, verbose = verbose)



            distModel1, distModel2 = compute_model(DF_positive_samples_train.iloc[:, :-2].values,
                                                        DF_negative_samples_train.iloc[:, :-2].values,
                                                        mode = mode, score = score)


            Model_client, Model_imposter = model(distModel1,
                                                        distModel2, 
                                                        model_type = model_type, 
                                                        score = score )



            FAR_temp, FRR_temp = calculating_fxr(Model_client, Model_imposter, distModel1, distModel2, THRESHOLDs, score)
            EER_temp = compute_eer(FAR_temp, FRR_temp)


            acc = list()
            f1 = list()
            eer = list()
            eer1 = list()
            eer2 = list()
            t_idx = EER_temp[1]

            # for _ in range(random_test_acc):
            #     pos_samples = DF_positive_samples_test.shape
            #     temp = DF_negative_samples_test.sample(n = pos_samples[0])

            #     DF_temp = pd.concat([DF_positive_samples_test, temp])
            #     DF_temp["subject ID"] = DF_temp["subject ID"].map(lambda x: 1 if x == subject else 0)


            #     distModel1 , distModel2 = compute_model(DF_positive_samples_train.iloc[:, :-2].values, DF_temp.iloc[:, :-2].values, mode = mode, score = score)
            #     Model_client, Model_test = model(distModel1, distModel2, model_type = model_type, score = score)

            #     FAR_temp_1, FRR_temp_1 = calculating_fxr(Model_client, Model_test, distModel1, distModel2, THRESHOLDs, score)


            
            #     y_pred = np.zeros((Model_test.shape))
            #     y_test = DF_temp.iloc[:,-2].values


            #     y_pred[Model_test > THRESHOLDs[t_idx]] = 1
            #     frr = 0
            #     far = 0

            #     for actual, prediction in zip(y_test, y_pred):
            #         if actual == 1 and prediction == 0:
            #             frr = frr + 1
            #         if actual == 0 and prediction == 1:
            #             far = far + 1

            #     fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            #     # logger.info("fpr: {}".format(fpr))

            #     acc.append( accuracy_score(y_test, y_pred)*100 )
            #     f1.append(  f1_score(y_test, y_pred)*100 )
            #     eer.append(compute_eer(fpr, 1-tpr))
            #     # eer1.append((FAR_temp_1[t_idx]+ FRR_temp_1[t_idx])/2)
            #     eer2.append((frr + far)/(2*y_test.sum()))
            #     # logger.info("far: {}".format(far))
            #     # logger.info("frr: {}".format(frr))
            #     # logger.info("eer2: {}\n".format((frr + far)/(2*y_test.sum())))


    
            # # logger.info("\nfar: {}\n".format(FAR_temp_1[t_idx]))
            # # logger.info("\nfrr: {}\n".format(FRR_temp_1[t_idx]))
            # logger.info("t_idx: {}".format(t_idx))
            # logger.info("eer sklearn: {}".format(np.mean(eer)))
            # # logger.info("eer1: {}".format(np.mean(eer1)))
            # logger.info("eer average of FAR and FRR: {}".format(np.mean(eer2)))
            # # logger.info("Model_test: {}".format(Model_test.shape))
            # # logger.info("DF_positive_samples_test: {}".format(DF_positive_samples_test.shape))
            # logger.info("1-tpr: {}".format(1-tpr))
            # logger.info("fpr: {}\n\n\n".format(fpr))

            # ROC_plot_v2(fpr, 1-tpr,thresholds)
            
            if direction == "left_0":
                EER_L.append(EER_temp)
                FAR_L.append(FAR_temp)
                FRR_L.append(FRR_temp)
                ACC_L.append([subject, np.mean(acc), np.mean(f1), np.mean(eer), np.mean(eer1), np.mean(eer2), DF_positive_samples_train.shape[0], DF_positive_samples_test.shape[0], DF_negative_samples_test.shape[0], test_ratio])

                
            elif direction == "right_1":
                EER_R.append(EER_temp)
                FAR_R.append(FAR_temp)
                FRR_R.append(FRR_temp)
                ACC_R.append([subject, np.mean(acc), np.mean(f1), np.mean(eer), np.mean(eer1), np.mean(eer2), DF_positive_samples_train.shape[0], DF_positive_samples_test.shape[0], DF_negative_samples_test.shape[0], test_ratio])

            
    columnsname = ["subject ID", "mean(acc)", "mean(f1)", "mean(eer)", "mean(eer1)", "mean(eer2)", "# positive samples training", "# positive samples test", "# negative samples test", "test_ratio", "EER", "t_idx" ] + ["FAR_" + str(i) for i in range(TH_dev)] + ["FRR_" + str(i) for i in range(TH_dev)] 
    DF_temp = pd.DataFrame(np.concatenate((ACC_L, EER_L, FAR_L, FRR_L), axis=1), columns = columnsname )
    DF_temp.to_excel(os.path.join(folder_path,   'Left.xlsx'))
    DF_temp = pd.DataFrame(np.concatenate((ACC_R, EER_R, FAR_R, FRR_R), axis=1), columns = columnsname )
    DF_temp.to_excel(os.path.join(folder_path,   'Right.xlsx'))


    toc=timeit.default_timer()
    logger.info("End:     ---    {}, \t\t Process time: {:.2f}  seconds".format(folder, toc - tic)) 

    
    A = [[features_excel, mode, model_type, test_ratio, normilizing, feat_name, persentage, (toc - tic), num_pc,
        
        template_selection_method,
        k_cluster]+

        np.mean( np.array(ACC_L)[:,1:8] , axis=0).tolist()+
        np.mean( np.array(ACC_R)[:,1:8] , axis=0).tolist()+
        np.concatenate((np.mean(np.array(FAR_L), axis=0), np.mean(np.array(FRR_L), axis=0)), axis=0).tolist()+
        np.concatenate((np.mean(np.array(FAR_R), axis=0), np.mean(np.array(FRR_R), axis=0)), axis=0).tolist()]


    z = pd.DataFrame(A, columns = cols )
    logger.debug("shape of return DF (z): {}".format(z.shape))

    return z



def calculating_fxr(Model_client, Model_imposter, distModel1, distModel2, THRESHOLDs, score):
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

def plot_ROC(FAR_L, FRR_L, FAR_R, FRR_R, labels):
    plt.figure(figsize=(14,8))

    for idx in range(len(FAR_L)):
        plt.subplot(1,2,1)
        auc = round((1 + np.trapz( FRR_L[idx], FAR_L[idx])),2)
        # label=a[idx] #+ ' AUC = ' + str(round(auc, 2))

        plt.plot(FAR_L[idx], FRR_L[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=labels[idx], clip_on=False)

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
        plt.plot(FAR_R[idx], FRR_R[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=labels[idx], clip_on=False)

        plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Acceptance Rate')
        plt.ylabel('False Rejection Rate')
        plt.title('ROC curve, Right side')
        plt.gca().set_aspect('equal')
        plt.legend(loc="best")

    plt.tight_layout()



def model(distModel1, distModel2, model_type = "average", score = None ):
    if score is None:
        if model_type == "average":
            # model_client = (np.sum(distModel1, axis = 0))/(distModel1.shape[1]-1)
            model_client = np.mean(np.ma.masked_where(distModel1==0,distModel1), axis = 0)
            model_client = np.expand_dims(model_client,-1)
            
            model_imposter = (np.sum(distModel2, axis = 0))/(distModel1.shape[1])
            model_imposter = np.expand_dims(model_imposter, -1)
                
        elif model_type == "min":

            model_client = np.min(np.ma.masked_where(distModel1==0,distModel1), axis = 0)
            model_client = np.expand_dims(model_client,-1)
            
            model_imposter = np.min(np.ma.masked_where(distModel2==0,distModel2), axis = 0)
            model_imposter = np.expand_dims(model_imposter, -1)
                    
        elif model_type == "median":
            model_client = np.median(distModel1, axis = 0)
            model_client = np.expand_dims(model_client,-1)
            

            model_imposter = np.median(distModel2, axis = 0)
            model_imposter = np.expand_dims(model_imposter, -1)

    if score is not None:
        if model_type == "average":
            model_client = np.mean(np.ma.masked_where(distModel1==1,distModel1), axis = 0)
            model_client = np.expand_dims(model_client,-1)
            
            model_imposter = (np.sum(distModel2, axis = 0))/(distModel1.shape[1])
            model_imposter = np.expand_dims(model_imposter, -1)
                
        elif model_type == "min":

            model_client = np.max(np.ma.masked_where(distModel1==1,distModel1), axis = 0)
            model_client = np.expand_dims(model_client,-1)
            
            model_imposter = np.max(np.ma.masked_where(distModel2==1,distModel2), axis = 0)
            model_imposter = np.expand_dims(model_imposter, -1)
                    
        elif model_type == "median":
            model_client = np.median(distModel1, axis = 0)
            model_client = np.expand_dims(model_client,-1)            

            model_imposter = np.median(distModel2, axis = 0)
            model_imposter = np.expand_dims(model_imposter, -1)
    

    return model_client, model_imposter


def compute_score(distance, mode = "A"):
    distance = np.array(distance)

    if mode == "A":
        return np.power(distance+1, -1) 
    elif mode =="B":
        return 1/np.exp(distance)


def compute_eer(fpr, fnr):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    abs_diffs = np.abs(np.subtract(fpr, fnr)) 
    # mmin = min(abs_diffs)   
    # idxs = np.where(abs_diffs == mmin)
    # print(idxs)
    # print(np.max(idxs))
    # print(np.min(idxs))
    # print(np.median(idxs))
    # print(np.mean(idxs))
    

    # min_index = int(np.mean(idxs))#
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))

    return eer, min_index


def plot_ACC(FAR_L, FRR_L, FAR_R, FRR_R, THRESHOLDs):
    """plot ROC curve"""
    plt.figure(figsize=(14,8))
    plt.subplot(1,2,1)
    plt.plot(THRESHOLDs, FAR_L, linestyle='--', marker='o', color='darkorange', lw = 2, label='FAR curve', clip_on=False)
    plt.plot(THRESHOLDs, FRR_L, linestyle='--', marker='o', color='navy', lw = 2, label='FRR curve', clip_on=False)
    plt.title('ACC curve, Left side')
    plt.legend(loc="best")
    plt.xlabel('Threshold')

    plt.subplot(1,2,2)
    plt.plot(THRESHOLDs, FAR_R, linestyle='--', marker='o', color='darkorange', lw = 2, label='FAR curve', clip_on=False)
    plt.plot(THRESHOLDs, FRR_R, linestyle='--', marker='o', color='navy', lw = 2, label='FRR curve', clip_on=False)
    plt.title('ACC curve, Right side')
    plt.legend(loc="best")
    plt.xlabel('Threshold')

    plt.tight_layout()




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


def compute_model(positive_samples, negative_samples, mode = "dist", score = None):

    positive_model = np.zeros((positive_samples.shape[0], positive_samples.shape[0]))
    negative_model = np.zeros((positive_samples.shape[0], negative_samples.shape[0]))

    if mode == "dist":

        for i in range(positive_samples.shape[0]):
            for j in range(positive_samples.shape[0]):
                # print(positive_samples.shape)
                # print(positive_samples.iloc[i, :])
                # print(positive_samples.iloc[i, :].values)
                positive_model[i, j] = distance.euclidean(
                    positive_samples[i, :], positive_samples[j, :]
                )
            for j in range(negative_samples.shape[0]):
                negative_model[i, j] = distance.euclidean(
                    positive_samples[i, :], negative_samples[j, :]
                )
        if score != None:
            return compute_score(positive_model, score), compute_score(negative_model, score)
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



    # np.save("./Datasets/distModel1.npy", distModel1)
    # np.save("./Datasets/distModel2.npy", distModel2)





Results_DF = pd.DataFrame(columns=cols)
def collect_results(result):
    global Results_DF
    Results_DF = Results_DF.append(result)
    Results_DF.to_excel(working_path + "/DF.xlsx")


def main():
    features_excelss = ["afeatures-simple", "pfeatures"]
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    for features_excel in features_excelss:
        feature_path = os.path.join(working_path, 'Datasets', features_excel + ".xlsx")
        DF_features_all = pd.read_excel(feature_path, index_col = 0)

        # for iiii in modes:
        #     folder = str(1.0) + "_z-score_" + str(-3) + "_" + iiii + "_" + "min" + "_" +  str(0) 
        #     # print(folder)     
        #     collect_results(fcn(DF_features_all,folder, features_excel, 2, template_selection_method = "None"))


        for i in modes:
            for ii in range(-3,DF_features_all.shape[1]-2,3):
                folder = str(1.0) + "_z-score_" + str(ii) + "_" + i + "_" + "min" + "_" +  str(0) 
                pool.apply_async(fcn, args=(DF_features_all, folder, features_excel, 2, "None"), callback=collect_results)
                # collect_results(fcn(DF_features_all, folder, features_excel, 2, template_selection_method = "None"))
        
    pool.close()
    pool.join()
    

    
    logger.info("Done!!!")



if __name__ == "__main__":
    logger.info("Starting !!!")
    tic = timeit.default_timer()
    main()
    toc = timeit.default_timer()
    logger.info("Done ({:2.2f} process time)!!!\n\n\n".format(toc-tic))