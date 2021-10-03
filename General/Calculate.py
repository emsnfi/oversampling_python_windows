import smote_variants as sv
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import statistics
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import math
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from collections import Counter
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as pl
import random
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
# # 跟原始資料合併 predict with Decision & SVM method


def predictDe(train, test, allRandomHalf):
    mergeRandom = []
    accuracies = []
    for index, element in enumerate(train):
        data = pd.read_excel(element, index_col=0)
        lastColumn = data.columns[-1]

        data[lastColumn] = data[lastColumn].str.replace("\n", "").str.strip()
        l = data.shape[1]-1
        le = preprocessing.LabelEncoder()
        data.iloc[:, l] = le.fit_transform(data.iloc[:, l])
        data.iloc[:, 0] = le.fit_transform(data.iloc[:, 0])
        # 把非 numeric 的資料用 label encoder 轉成 numeric 資料
        le = preprocessing.LabelEncoder()
        for j in range(data.shape[1]):
            for k in range(data.shape[0]):
        # print(df.iloc[j,i])
                if isinstance(data.iloc[k,j],str):
                    data.iloc[:,j] = data.iloc[:,j].apply(lambda col: str(col)) 
                    data.iloc[:,j] = le.fit_transform(data.iloc[:,j])
                    break

        mergeRandom = pd.concat([data, allRandomHalf[index]], axis=0)

        finaldata = mergeRandom.iloc[:, :l]
        output = mergeRandom.iloc[:, l]
        print(Counter(output))

        clf = DecisionTreeClassifier()
        clf = clf.fit(finaldata, output)

        # 不然會有多出來的 unnamed column
        test_file = pd.read_excel(test[index], index_col=0)
        test_data = pd.DataFrame(test_file)
        test_X = test_data.iloc[:, :(test_data.shape[1])-1]
        # 把非 numeric 的資料用 label encoder 轉成 numeric 資料
        # for col in range(test_X.shape[1]):
        #     if isinstance(test_X.iloc[0, :][col], str):
        #         test_X.iloc[:, col] = le.fit_transform(
        #             test_X.iloc[:, col])

        for j in range(test_X.shape[1]):
            for k in range(test_X.shape[0]):
        # print(df.iloc[j,i])
                if isinstance(test_X.iloc[k,j],str):
                    test_X.iloc[:,j] = test_X.iloc[:,j].apply(lambda col: str(col)) 
                    test_X.iloc[:,j] = le.fit_transform(test_X.iloc[:,j])
                    break
        test_X.iloc[:, 0] = le.fit_transform(test_X.iloc[:, 0])

        test_y_predicted = clf.predict(test_X)

        test_y = test_data.iloc[:, test_data.shape[1]-1]

        test_y = le.fit_transform(test_y)
        test_y_predicted = le.fit_transform(test_y_predicted)

        accuracy = roc_auc_score(test_y, test_y_predicted)
        accuracies.append(accuracy)

    mean = statistics.mean(accuracies)
    mean = statistics.mean(accuracies)
    meanRound = round(mean, 3)
    print(meanRound)
    return meanRound


def predictSVM(train, test, allRandomHalf):
    mergeRandom = []
    accuracies = []
    for index, element in enumerate(train):
        data = pd.read_excel(element, index_col=0)
        lastColumn = data.columns[-1]

        data[lastColumn] = data[lastColumn].str.replace("\n", "").str.strip()
        l = data.shape[1]-1
        le = preprocessing.LabelEncoder()
        data.iloc[:, l] = le.fit_transform(data.iloc[:, l])
        data.iloc[:, 0] = le.fit_transform(data.iloc[:, 0])
        for j in range(data.shape[1]):
            for k in range(data.shape[0]):
        # print(df.iloc[j,i])
                if isinstance(data.iloc[k,j],str):
                    data.iloc[:,j] = data.iloc[:,j].apply(lambda col: str(col)) 
                    data.iloc[:,j] = le.fit_transform(data.iloc[:,j])
                    break
        mergeRandom = pd.concat([data, allRandomHalf[index]], axis=0)

        finaldata = mergeRandom.iloc[:, :l]
        output = mergeRandom.iloc[:, l]
        print(Counter(output))

        clf = svm.SVC(kernel='rbf', C=1, gamma='auto')
        clf = clf.fit(finaldata, output)

        # 不然會有多出來的 unnamed column
        test_file = pd.read_excel(test[index], index_col=0)
        test_data = pd.DataFrame(test_file)
        test_X = test_data.iloc[:, :(test_data.shape[1])-1]
        for j in range(test_X.shape[1]):
            for k in range(test_X.shape[0]):
        # print(df.iloc[j,i])
                if isinstance(test_X.iloc[k,j],str):
                    test_X.iloc[:,j] = test_X.iloc[:,j].apply(lambda col: str(col)) 
                    test_X.iloc[:,j] = le.fit_transform(test_X.iloc[:,j])
                    break
        test_X.iloc[:, 0] = le.fit_transform(test_X.iloc[:, 0])

        test_y_predicted = clf.predict(test_X)

        test_y = test_data.iloc[:, test_data.shape[1]-1]

        test_y = le.fit_transform(test_y)
        test_y_predicted = le.fit_transform(test_y_predicted)

        accuracy = roc_auc_score(test_y, test_y_predicted)
        accuracies.append(accuracy)

    mean = statistics.mean(accuracies)
    meanRound = round(mean, 3)
    print(meanRound)
    return meanRound

# def predictDecisionSingle(train, test, allRandomHalf):
