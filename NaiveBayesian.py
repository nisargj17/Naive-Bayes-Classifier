#Importing functional libraries
import pandas as pd

from pandas import read_csv, DataFrame
import random
import numpy as np
import matplotlib .pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from pylab import *

#Loading dataset for training
def ldDataTrainSet(filename):
    Train_data_frame = pd.read_csv(filename)
    return Train_data_frame
#Loading dataset for testing
def ldTestSet(fileTest):
    Test_data_frame = pd.read_csv(fileTest)
    return Test_data_frame

#Now we will train the dataset for classification
def training_dataset(Train_data_frame):
    global count_Greater50k         #Salary greater than 50K
    global count_LessThanEqual50k   #Salary less than 50K
    dfGreater50k = Train_data_frame[Train_data_frame['CLASS'] == " >50K"]               #dividing into 2 data frames for each class label
    dfLessThanEqual50k = Train_data_frame[Train_data_frame['CLASS'] == " <=50K"]

    count_Greater50k = len(dfGreater50k)                    #taking count of dataframe 1
    count_LessThanEqual50k = len(dfLessThanEqual50k)        #taking count of dataframe 2


    prob_Greater50k = float(count_Greater50k)/float(len(Train_data_frame))              #Calculating probabilities
    prob_LessThanEqual50k = float(count_LessThanEqual50k)/float(len(Train_data_frame))

    classProb = {" >50K": prob_Greater50k, " <=50K": prob_LessThanEqual50k }

    df_Great_Dict = {}
    df_Less_Dict = {}
    for onecol in dfGreater50k:
        if onecol != 'CLASS':
            # adding to greater than dict
            greatSeries = dfGreater50k[onecol].value_counts()
            thisDict = greatSeries.to_dict()
            thisDict.update((k, float(v)/float(count_Greater50k)) for k, v in thisDict.items())
            df_Great_Dict[onecol] = thisDict

            # adding to less than dict
            lessSeries = dfLessThanEqual50k[onecol].value_counts()
            lessDict = lessSeries.to_dict()
            lessDict.update((k, float(v)/float(count_LessThanEqual50k)) for k, v in lessDict.items())
            df_Less_Dict[onecol] = lessDict

    likelihood = {}                         #Creating likelihood
    likelihood[" >50K"] = df_Great_Dict
    likelihood[" <=50K"] = df_Less_Dict

    return likelihood, classProb


#Now we will test the dataset
def testing_dataset(likelihood, classProb, Test_data_frame):


    Pos = {}
    true = 0
    false = 0
    total = 0
    a=0
    b=0
    count=0

    for record in Test_data_frame.iterrows():
        total += 1
        post = 1
        for k in likelihood:
            for col in Test_data_frame:
                if col != 'CLASS':
                    value = record[1][col]
                    if value in likelihood[k][col]:
                        post *= likelihood[k][col][value]
            post *= classProb[k]
            Pos[k] = post

        # get the classifier labels


        if Pos[" >50K"] > Pos[" <=50K"]:
            max_label = " >50K"
        else:
            max_label = " <=50K"

        main_Label = record[1]['CLASS']


        if main_Label == max_label:
            false+= 1
        else:
            true+= 1
    TP=true
    TN=false
    z=total
    print ("True postive:",TP)     #Getting true positive
    print("True negative:",TN)     #Getting true negative
    np=count_Greater50k
    pp=count_LessThanEqual50k


    global Accu_Calc

    Accu_Calc=(float(TP)/(z))*100   #formula for Accuracy

    pre=(float(TP)/float(z))*100    #formula for precision

    rec=(float(TP)/float(TP))       #formula for recall

    print ("Accuracy",Accu_Calc)

    print("precision",pre)

    print("recall",rec)

    return Accu_Calc


def main():
    filename = ""
    fileTest = ""
    Train_data_frame = ldDataTrainSet(filename)
    Test_data_frame = ldTestSet(fileTest)
    print(len(Train_data_frame))
    print(len(Test_data_frame))


    likelihood, classP= training_dataset(Train_data_frame)
    testing_dataset(likelihood, classP, Test_data_frame)

main()
