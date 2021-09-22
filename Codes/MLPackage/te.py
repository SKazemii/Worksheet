import numpy as np
import pandas as pd
import scipy.stats as scipy
from termcolor import colored

class Mrmr:
    #import MRMR_korigiert as MRMR
    considered_features = []
    
    def __init__(self,
                 feature_set,
                 Labels,
                 method = "pearson"
                 ):
        """Here we create only the class with needed variables
        Args:
            feature_set: A array.
                It has a size of (m * n).
            Labels: A array.
                It has a size of (n * 1)
            method: A String.
                The method gives the information which correlation to use for
                the calculation.
        Returns:
            Nothing. This for creation classes and fill them with items. A few
            functions return objects.
        Examples:
            >>> Datenbeispiel = MRMR(SetX, SetY, "kendall")
                # SetX and SetY stand only for representation of this.
            >>>
        """
        self.SetX = feature_set
        self.SetY = Labels
        self.method = method


    def fit(self):
        result = Mrmr.calculation( self.SetX, self.SetY, self.method )
        return result


    def calculation(SetX, SetY, method):
        """Calculate the score of the features(SetX) and select those, which
            are relevant enough( in this specific situation (Score > 0)).
        Args:
            SetX: Is the array, which is to be analysed
            SetY: Are endpoints/classification to determine the relevance.
            method: Is the use of function for our correlation
        Returns:
            filtered_feature: mrmr_scores, mrmr_featid and mrmr_featnames as
                                DataFrame
            mrmr_scores: Is the score of the feature in the row which it is.
            mrmr_featid: The Id is like identifier, but come from the storage
                            of the position in the original set.
            mrmr_featnames: Is the name, which the feature(coloumn of SetX)
                                has.
        Notes:
            The return, which consists of four values, is overloaded in a
            certain sense, because the first parameter, which is a DataFrame
            as specified in the Returns, is a summary of the next three
            parameters(arrays), what principle corresponds to a double
            return of the information. This should only serve to use the
            preferred data type.
        Examples:
            >>> SetX = [[1,2,4,5,6], [62,2,23,6,9], [2,3,5,6,7], [4,1,4,9,34],
                        [23,2,54,5,7]]
            >>> SetY = [3,4,9,7,9]
            >>> Datenbeispiel = MRMR(SetX, SetY, "pearson")
            >>> Resultat = MRMR.Datenbeispiel.calculation()
            >>> Resultat[0]
               Feature_Id             Name     Score
            0           1  [1, 2, 4, 5, 6]  0.897887
            1           3  [2, 3, 5, 6, 7]  0.283357
            >>> Resultat[1]
            [0.8978872704229619, 0.28335676376630603]
        """
        # import numpy as np
        # import pandas as pd
        # from termcolor import colored
        
        print(  "\n[INFO] start to calcolating\n") 
        
        Relevance = Mrmr.max_rel_calc(SetX, SetY, method)
        Redundancy = Mrmr.min_red_calc(SetX, method)
        mrmr_featid = []# container for feature id's
        mrmr_scores = [] # container for scores according to `mrmr_features`
        mrmr_featnames = [] # container for names
        feat_names = list(SetX)
        feat_id = []
        
        print("\nRelevanz: ") 
        print(Relevance)
        print("\nRedundanz: ") 
        print(Redundancy)
        
        filler = 0
        while (filler < len(feat_names)):
            feat_id.append(filler)
            filler = filler + 1
        # soll eine künstlich erstellte id darstellen
        
        idmax = Relevance['Koeffizienten'].idxmax()
        score_tmp = Relevance.iloc[idmax][0] # Übernahme des Scores
        names_tmp = feat_names.pop(idmax) # Übernahme des Names
        id_tmp = feat_id[idmax] # Übernahme des id's

        Relevance = Relevance.drop([idmax][0])
        print(Relevance)
        print(id_tmp)
        mrmr_scores.append(score_tmp) # Einschreiben des Scores
        mrmr_featnames.append(names_tmp) # Einschreiben des Names
        mrmr_featid.append(id_tmp) # Einschreiben der ids

        z = 0
        while (Relevance.empty == False and score_tmp > 0):
                # Überprüfung ob Menge noch Elemnte hat und der Score stimmt
            z = z + 1
            idmax = Relevance['Koeffizienten'].idxmax()
            print(idmax)

            list_of_redundanz = [] # Hier sollen die Redudanz-Werte stehen
            sl = 0 # Läufer für das Abarbeiten der Spalten einer Zeile
            while (sl < len(mrmr_scores)):
                list_of_redundanz.append(Redundancy[idmax][mrmr_featid[sl]])
                sl = sl + 1
                 # Vielleicht etwas unsschön gemacht, um eine Liste an Werten
                 # zu erstellen, welches über die Redundanz aussagt, eine
                 # andere Art der gezielten Suche müsste ich mir noch finden

            redund_tmp = list_of_redundanz[np.argmax(list_of_redundanz)]
                # enthaelt den höchsten Wert der Redundanz
            score_tmp = Relevance.loc[idmax][0] - redund_tmp
                # ist der berechnete Score
            
            print("Relevance.iloc[idmax][0]: {:}".format(
                                                  Relevance.iloc[idmax][0]))
            print("redund_tmp: {:}".format(redund_tmp))
            print("score_tmp: {:}".format(score_tmp))
            print("idmax: {:}".format(idmax))

            if (score_tmp > 0):
                # Das Einfügen wird nur dann getätigt, wenn auch tatsächlich
                # positiv sind
                names_tmp = feat_names.pop((idmax))# - len(mrmr_scores)))
                id_tmp = feat_id[idmax]
                
                Relevance = Relevance.drop([idmax][0])

                mrmr_scores.append(score_tmp) # Einschreiben des Scores
                mrmr_featid.append(id_tmp) # Einschreiben der id
                mrmr_featnames.append(names_tmp) # Einschreiben des Namens
            
        filtered_feature = pd.DataFrame( columns = ['Feature_Id',
                                                    'Name',
                                                    'Score'] )
        
        filtered_feature['Feature_Id'] = mrmr_featid
        filtered_feature['Name'] = mrmr_featnames
        filtered_feature['Score'] = mrmr_scores

        print("\n\n\Ende der calculation\n" )
        return filtered_feature, mrmr_scores, mrmr_featid, mrmr_featnames


    def max_rel_calc(feature_set,
                     classification,
                     method
                     ):
        """This part calculating the maximum relevance for the calculation of
            the score of the features(SetX).
        Args:
            feature_set: Is the array, which is to be analysed
            classification: Is an array to determine the relevance.
            method: That have the name of the correlation, which is to use to
                    capture the result
        Returns:
            relevance_of_feature_subset: Die Rückgabe ist ein DataFrame,
                                            welches mit der berechneten
                                            Korrelation für die Relevanz mit
                                            der Reihenfolge der übergebenen
                                            Features gefüllt wurde
        Notes:
            The return, which consists of four values, is overloaded in a
            certain sense, because the first parameter, which is a DataFrame
            as specified in the Returns, is a summary of the next three
            parameters(arrays), what principle corresponds to a double
            return of the information. This should only serve to use the
            preferred data type.
        Examples:
            >>> SetX = [[1,2,4,5,6], [62,2,23,6,9], [2,3,5,6,7], [4,1,4,9,34],
                        [23,2,54,5,7]]
            >>> SetY = [3,4,9,7,9]
            >>> Resultat = MRMR.max_rel_calc(SetX, SetY, "pearson")
            >>> Resultat
               Koeffizienten
            0       0.897887
            1       0.469399
            2       0.897887
            3       0.584747
            4       0.325684
        """
        # import pandas as pd
        # from termcolor import colored
        
        print(  "\nStart der Relevanzbestimmung:", "green") 
        relevance_of_feature = pd.DataFrame( columns = ['Koeffizienten'] )
        # Dies wird zurückgegeben und soll die Relevanz der Features sein
        
        # if ((len(classification.columns)) == 2):
        #     method = "concordance index"
        
        sl = 0 # Schleifen-läufer
        while (sl < len(feature_set.columns)):
            # Die while-Schleife ist für die Berrechnung jeder Koorelation
            # aus der Menge von Feature zu den Endpunkten
            # print(sl) /ist nur zur Hilfe bei entstandenen Fehlern
            input_correl = Mrmr.correl(feature_set.iloc[:,sl],
                                       classification,
                                       method
                                       ) # Bestimmung einer Korrelationen
            
            relevance_of_feature = relevance_of_feature.append(
                {'Koeffizienten': input_correl},
                ignore_index=True)
            
            sl = sl + 1
        
        print(  "\nEnde der Relevanzbestimmung\n", "green") 
        return relevance_of_feature


    def min_red_calc(feature_set,
                     method
                      ):
        """This part calculating the minimum redundancy for the calculation of
            the score of the features(SetX).
        Args:
            feature_set: Is the array, which is to be analysed, it is the hole
                         feature set.
            method: That have the name of the correlation, which is to use to
                    capture the result
        Returns:
            redundancy_matrix: A n*n matrix, that have the size of n (= length
                               of the feature_set).
        Notes:
            The matrix is filled with correlation between features and the
            position of the resulting correlation will show about the row and
            column which feature has correlated with an another feature.
        Example:
            >>> SetX = [[1,2,4,5,6], [62,2,23,6,9], [2,3,5,6,7], [4,1,4,9,34],
                        [23,2,54,5,7]]
            >>> SetY = [3,4,9,7,9]
            >>> Resultat = MRMR.min_red_calc(SetX, "pearson")
            >>> Resultat
            array([[1.        , 0.61453051, 1.        , 0.74826678,
                        0.09821991],
                   [0.61453051, 1.        , 0.61453051, 0.27047831,
                        0.42808221],
                   [1.        , 0.61453051, 1.        , 0.74826678,
                        0.09821991],
                   [0.74826678, 0.27047831, 0.74826678, 1.        ,
                        0.30289654],
                   [0.09821991, 0.42808221, 0.09821991, 0.30289654,
                        1.        ]])
        """
        # import numpy as np
        # from termcolor import colored
        print(  "\nStart der Redudanzbestimmung:", "green") 
        redundancy_matrix = np.empty( [len(feature_set.columns),
                                       len(feature_set.columns)] )
        sl = 0 # Betrachtung der einzelnen Spalten
        while (sl < len(feature_set.columns)):
            input_correl = 0
            zl = 0 # Betrachtung der einzelnen Zeilen
            while (zl < len(feature_set.columns)):
                if(sl == zl):
                    input_correl = 1
                else:
                    input_correl = Mrmr.correl(feature_set.iloc[:,sl],
                                               feature_set.iloc[:,zl],
                                               method
                                               )
                
                #print("sl: {:}  zl: {:}  input: {:}".format(sl,
                #                                            zl,
                #                                            input_correl) )
                redundancy_matrix[sl][zl] = input_correl
                zl = zl +1
            sl = sl + 1

        print(  "\nEnde der Redudanzbestimmung\n", "green") 
        return redundancy_matrix


    def correl(setX, setY, method):
        """This part correlate the two received sets.
        Args:
            SetX and SetY: Are arrays which will be correlated.
            method: That have the name of the correlation, which is to use to
                    capture the result.
        Returns:
            correlation: The result of the correlation from the sets SetX and
                         SetY.
        Notes:
            The sets have for this function no meaning they should only be
            subject to the conditions to be able to perform the calculation.
        Examples:
            >>> SetX = [4,1,4,9,34]
            >>> SetY = [3,4,9,7,9]
            >>> Resultat = MRMR.correl(SetX, SetY, "pearson")
            >>> Resultat[0]
            0.584747
        """
        # import numpy as np
        
        correlation = 0
        if (method == "pearson"):
            # import scipy.stats as scipy
            correlation = scipy.pearsonr(setX, setY)
            correlation = correlation[0]
        
        elif (method == "spearman"):
            # import scipy.stats as scipy
            correlation = scipy.spearmanr(setX, setY)
            correlation = correlation[0]
        
        elif (method == "kendall"):
            # import scipy.stats as scipy
            correlation = scipy.kendalltau(setX, setY)
            correlation = correlation[0]
        
        elif(method == "carmer"):
        
            # Code kommt noch
            print("\nDie Berechnung über den Carmer's V ist noch nicht " +
                  "fertig, gilt noch zu bearbeiten\n")
            correlation = "Fehler"
        
        return np.abs(correlation)


def main():
    pfeatures = np.load("./Datasets/pfeatures.npy")

    print("[INFO] feature shape: ", pfeatures.shape)
    columnsName = ["feature_" + str(i) for i in range(pfeatures.shape[1]-2)] + [ "subject ID", "left(0)/right(1)"]

    DF_features = pd.DataFrame(
        pfeatures,
        columns = columnsName 
    )
    features = np.array(pfeatures[:,0:11])
    labels = np.array(pfeatures[:,-1])

    a = Mrmr(DF_features.iloc[:,20:-2],DF_features.iloc[:,-2])
    b = a.fit()
    print(b[0])
    # relevent_features = Max_relevence(features,labels, n=10)
    # print(relevent_features)
    # print(relevent_features.shape)
    # print(F)
    
    print("[INFO] Done!!!")



if __name__ == "__main__":
    main()