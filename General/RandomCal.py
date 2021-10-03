# implement Random, Center, ElbowRandomGenerate, ElbowCenterGenerate
from imblearn.over_sampling._smote.base import SMOTE
from imblearn.over_sampling import SMOTE
import smote_variants as sv
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

import pr
import random
import math
import data_process
import os
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# different mthod use case to identify


# train is the data , ratio is the value that assign, method :which smote method
'''After composite data, using Random way to choose the data'''


def RandomGenerate(train, ratio, method, path):
    ryy = []
    #print("osos", os.getcwd())
    # random polynom_fit_SMOTE
    alloverpolynom = []
    overpolynom = []
    randompolynom = []

    if ratio == 0:
        return []
    for ii, i in enumerate(train):
        print("第幾個", i)
        randomIndex = []
        data = pd.read_excel(i, index_col=0)
        classCount, finaldata, output = preprocess(data)

        # 把非 numeric 的資料用 label encoder 轉成 numeric 資料
        le = preprocessing.LabelEncoder()
        for j in range(finaldata.shape[1]):
            for k in range(finaldata.shape[0]):
        # print(df.iloc[j,i])
                if isinstance(finaldata.iloc[k,j],str):
                    finaldata.iloc[:,j] = finaldata.iloc[:,j].apply(lambda col: str(col)) 
                    finaldata.iloc[:,j] = le.fit_transform(finaldata.iloc[:,j])
                    break

        ''''different smote method'''
        X_polynom, y_polynom = synth(
            finaldata, output, method)
        newDataCount = len(
            X_polynom) - len(data)  # 新生成的 data 數量

        X_polynom = pd.DataFrame(X_polynom)
        y_polynom = pd.DataFrame(y_polynom)
        alloverpolynom = pd.concat(
            [X_polynom, y_polynom], axis=1)  # SMOTE 完後的數據
        overpolynom.append(alloverpolynom)
        randomIndex = []
        '''new sythetic data will be generated after the origin data'''
        for i in range(len(classCount)):
            # print(classCount[i],"ffksdl;")
            count = math.floor(int(classCount[i][1])*ratio)  # 要產生多少數據  無條件捨去
            randomIndex.extend(
                [random.randint(len(data), len(X_polynom)-1) for _ in range(count)])

        randomtemp = []
    # print(overSMOTE)

        for index in randomIndex:
            randomtemp.append(overpolynom[ii].iloc[index, :])
        #print("randomtempsize", len(randomtemp))

        randompolynom.append(randomtemp)

    print(np.array(randompolynom).shape)

    return randompolynom

# 現在 randomSMOTE 存的是 random 的 SMOTE 生成 data


'''After composite data, using Center way to choose the data'''


def CenterGenerate(train, ratio, method, path):
    if ratio == 0:
        return []
    alloverpolynom = []
    overpolynom = []
    centerpolynom = []  # center index
    centerpolynomvalue = []  # obtain the real value from center index
    for ii, i in enumerate(train):
        print("第幾個", i)
        randomIndex = []
        data = pd.read_excel(i, index_col=0)
        #print(i, "traindata1", data)
        classCount, finaldata, output = preprocess(data)
        # 把非 numeric 的資料用 label encoder 轉成 numeric 資料
        le = preprocessing.LabelEncoder()
        for j in range(finaldata.shape[1]):
            for k in range(finaldata.shape[0]):
        # print(df.iloc[j,i])
                if isinstance(finaldata.iloc[k,j],str):
                    finaldata.iloc[:,j] = finaldata.iloc[:,j].apply(lambda col: str(col)) 
                    finaldata.iloc[:,j] = le.fit_transform(finaldata.iloc[:,j])
                    break

        originlen = data.shape[0]  # 原始的 data 數量
        X_polynom, y_polynom = synth(
            finaldata, output, method)
        X_polynom = pd.DataFrame(X_polynom)
        y_polynom = pd.DataFrame(y_polynom)
        alloverpolynom = pd.concat(
            [X_polynom, y_polynom], axis=1)  # SMOTE 完後的數據
        overpolynom.append(alloverpolynom)

        for i in range(len(classCount)):
            countfor = math.floor(
                int(classCount[i][1])*ratio)  # 要產生多少數據  無條件捨去
        #randomIndex.extend([random.randint(len(data),len(X_smote)-1) for _ in range(count)])

            if(countfor > 0):
                kmeans = KMeans(n_clusters=1)
                dtemp = pd.DataFrame(overpolynom[ii])
                X = dtemp.iloc[originlen:, :dtemp.shape[1]-1]  # 後來生成的

                kmeans.fit(X)
                y_kmeans = kmeans.predict(X)
                centers = kmeans.cluster_centers_

                distance = []
                X = X.astype('float64')
                centers = centers.astype('float64')
                tempindata = {}
                distancesortemp = []
                for i in range(X.shape[0]-1):  # 列

                    distance = []
                    temp = 0
                    for j in range(X.shape[1]-1):  # 9 行
                        temp = pow((centers[0][j]-X.iloc[i][j]), 2)
                        tempindata[i] = temp

                distancesortemp = sorted(
                    tempindata.items(), key=lambda item: item[1])

                centerpolynom.append(distancesortemp[:countfor])
                # centerpolynom 只是 index ，value 是取出值

        for i in range(len(centerpolynom)):
            alltemp = []
            for j in range(len(centerpolynom[i])):
                indexpolynom = centerpolynom[i][j][0] + originlen - 1
                alltemp.append(list(overpolynom[i].iloc[indexpolynom]))
            centerpolynomvalue.append(alltemp)
    return centerpolynomvalue


'''Elbow method using random way to choose data'''


def ElbowRandomGenerate(train, ratio, method, path):
    if ratio == 0:
        return []

    alloverpolynom = []
    overpolynom = []
    randompolynom = []  # elbow random index
    for ii, i in enumerate(train):
        print("第幾個", i)
        randomIndex = []
        data = pd.read_excel(i, index_col=0)
        #print(i, "traindata1", data)
        classCount, finaldata, output = preprocess(data)
        # 把非 numeric 的資料用 label encoder 轉成 numeric 資料
        le = preprocessing.LabelEncoder()
        for j in range(finaldata.shape[1]):
            for k in range(finaldata.shape[0]):
        # print(df.iloc[j,i])
                if isinstance(finaldata.iloc[k,j],str):
                    finaldata.iloc[:,j] = finaldata.iloc[:,j].apply(lambda col: str(col)) 
                    finaldata.iloc[:,j] = le.fit_transform(finaldata.iloc[:,j])
                    break
        originlen = data.shape[0]  # 原始的 data 數量
        X_polynom, y_polynom = synth(
            finaldata, output, method)
        X_polynom = pd.DataFrame(X_polynom)
        y_polynom = pd.DataFrame(y_polynom)
        alloverpolynom = pd.concat(
            [X_polynom, y_polynom], axis=1)  # SMOTE 完後的數據
        overpolynom.append(alloverpolynom)

        for i in range(len(classCount)):  # 不同類個別要產生多少數據才能平衡 目前是二分類
            origincount = int(classCount[i][1])  # 原本的數據量

            countfor = math.floor(
                int(classCount[i][1])*ratio)  # 要產生多少數據  無條件捨去

            if(countfor > 0):
                print("原本", origincount)
                dtemp = pd.DataFrame(overpolynom[ii])
                # 後來生成的 都是小類 # 把最後的類別拿掉
                X = dtemp.iloc[originlen:, :dtemp.shape[1]-1]
                X_ = dtemp.iloc[originlen:, -1]  # 拿掉的分類 雖然都一樣
                X.reset_index(inplace=True, drop=True)
                X_.reset_index(inplace=True, drop=True)
            # print("要產生多少",countfor)
            # 計算應該分成幾群
                model = KMeans()
                visualizer = KElbowVisualizer(model, k=(1, 12))

                # Fit the data to the visualizer
                kmodel = visualizer.fit(X)
                cluster_count = kmodel.elbow_value_  # 最佳要分成幾群
                kmeans = KMeans(n_clusters=cluster_count)
                kmeans.fit(X)
                label = Counter(kmeans.labels_)  # 標籤分類狀況
                # 不同群的比例
                labelRatio = []
                for key, element in sorted(label.items()):
                    labelRatio.append(element/origincount)
                # 把分類標籤跟原始資料進行合併
                klabel = pd.DataFrame(
                    {'label': kmeans.labels_})  # 建立一個欄位名為 label 的

                df = pd.concat([X, X_, klabel], axis=1)
                X = X.astype('float64')
            # random 挑選各群的資料

                ct = 0
                randomvaluetemp = []  # 放不同切分資料集的值

                for ic in range(cluster_count):
                    ct += 1
                    randomIndex = []
                    randomtemp = []
                #temppolynom = []
                # 把不同群過濾出來
                    # df 是 X 跟 label 結合後的 dataframe
                    tempdf = df[df['label'] == ic]
                    countforlabel = math.ceil(countfor * labelRatio[ic])

                    tempdf.reset_index(drop=True, inplace=True)
                # 不同群的random
                    randomIndex.extend([random.randint(0, len(tempdf)-1)
                                       for _ in range(countforlabel)])  # 該群的index

                # 該群真實的值 index 是位置
                    for index in randomIndex:
                        randomtemp.append(tempdf.iloc[index, :-1])

                    randomvaluetemp.extend(randomtemp)  # 一個切分資料集 所有群的資料
                print("countfor", countfor)
                print("長度", len(randomvaluetemp))

                randompolynom.append(randomvaluetemp)  # 所有資料集要取的資料
    return randompolynom


def ElbowCenterGenerate(train, ratio, method, path):
    if ratio == 0:
        return []
    alloverpolynom = []
    overpolynom = []
    centerpolynom = []
    centerpolynomvalue = []
    countfor = 0
    for ii, i in enumerate(train):
        print("第幾個", i)
        data = pd.read_excel(i, index_col=0)
        #print(i, "traindata1", data)
        classCount, finaldata, output = preprocess(data)
        # 把非 numeric 的資料用 label encoder 轉成 numeric 資料
        le = preprocessing.LabelEncoder()
        for j in range(finaldata.shape[1]):
            for k in range(finaldata.shape[0]):
        # print(df.iloc[j,i])
                if isinstance(finaldata.iloc[k,j],str):
                    finaldata.iloc[:,j] = finaldata.iloc[:,j].apply(lambda col: str(col)) 
                    finaldata.iloc[:,j] = le.fit_transform(finaldata.iloc[:,j])
                    break
        originlen = data.shape[0]  # 原始的 data 數量
        X_polynom, y_polynom = synth(
            finaldata, output, method)
        X_polynom = pd.DataFrame(X_polynom)
        y_polynom = pd.DataFrame(y_polynom)
        alloverpolynom = pd.concat(
            [X_polynom, y_polynom], axis=1)  # SMOTE 完後的數據
        overpolynom.append(alloverpolynom)
        tempcenterpolynom = []
        for i in range(len(classCount)):  # 不同類個別要產生多少數據才能平衡 目前是二分類
            origincount = int(classCount[i][1])
            print("原本", origincount)
            countfor = math.floor(
                int(classCount[i][1])*ratio)  # 要產生多少數據  無條件捨去
        #randomIndex.extend([random.randint(len(data),len(X_smote)-1) for _ in range(count)])

            if(countfor > 0):
                dtemp = pd.DataFrame(overpolynom[ii])
                X = dtemp.iloc[originlen:, :dtemp.shape[1]-1]  # 後來生成的 都是小類
                X.reset_index(inplace=True, drop=True)
            # print("要產生多少",countfor)
            # 計算應該分成幾群
                model = KMeans()
                visualizer = KElbowVisualizer(model, k=(1, 12))

                # Fit the data to the visualizer
                kmodel = visualizer.fit(X)
                cluster_count = kmodel.elbow_value_  # 最佳要分成幾群
                kmeans = KMeans(n_clusters=cluster_count)
                kmeans.fit(X)
                label = Counter(kmeans.labels_)  # 標籤分類狀況

                # 不同群的比例
                labelRatio = []
                for key, element in sorted(label.items()):
                    labelRatio.append(element/origincount)
            # print(labelRatio)

            # 把分類標籤跟原始資料進行合併
                klabel = pd.DataFrame(
                    {'label': kmeans.labels_})  # 建立一個欄位名為 label 的
                df = pd.concat([X, klabel], axis=1)  # X 是後來生成的數據 類別都是小類
            # print(df)
                centers = kmeans.cluster_centers_  # 各群群中心

                distance = []
                X = X.astype('float64')
                centers = centers.astype('float64')
                tempindata = {}
                distancesortemp = []

            # 計算每個點跟各群中心的距離

                ct = 0
            # print("分成",cluster_count,"群")
            # print("要產生",countfor)
                tempcenterpolynom = []  # 清空
                for ic in range(cluster_count):
                    ct += 1

                    temppolynom = []
                # 把不同群過濾出來
                    # df 是 X 跟 label 結合後的 dataframe
                    tempdf = df[df['label'] == ic]
                # allCluster.append(df[df['label']==ic])

                # 計算每個點跟群中心的距離
                    for i in range(tempdf.shape[0]-1):  # 列 也就是幾筆資料

                        distance = []
                        temp = 0  # 放算出來的距離
                        tempsum = 0
                        # 到前一欄 因為最後一欄為 label
                        for j in range(tempdf.shape[1]-2):
                            # 該欄位跟center欄位的距離
                            temp = pow((centers[ic][j]-tempdf.iloc[i][j]), 2)
                            tempsum = tempsum + temp
                        # print(tempsum)
                            tempindata[i] = tempsum

                    distancesortemp = sorted(
                        tempindata.items(), key=lambda item: item[1])
                # print(distancesortemp)

                # 要按照比例挑出資料

                    countforlabel = math.ceil(
                        countfor * labelRatio[ic])  # 按照比例 給不同的數量 不同群不同數量
                # print("比例",labelRatio)
                    temppolynom.extend(
                        distancesortemp[:countforlabel])  # 該群所要的數量
                # print("該群所要的數量",len(temppolynom))
            # tempcenterpolynom.extend(temppolynom) # 該份資料集所要的所有資料

                # print("ct",ct)
                    tempcenterpolynom = tempcenterpolynom+temppolynom

                centerpolynom.append(tempcenterpolynom)  # 所有資料集所選到的資料
            # print("真的有幾筆",len(centerpolynom[ii]))

        for i in range(len(centerpolynom)):
            alltemp = []
            for j in range(len(centerpolynom[i])):
                indexpolynom = centerpolynom[i][j][0] + originlen - 1
                alltemp.append(list(overpolynom[i].iloc[indexpolynom]))
            centerpolynomvalue.append(alltemp)
    return centerpolynomvalue


def synth(finaldata, output, method):
    finaldata = np.array(finaldata)
    output = np.array(output)
    if method is 'poly':  # "poly" in method:
        print("pol")
        over = sv.polynom_fit_SMOTE()
    elif method is 'prow':  # "proW" in method:
        print("pro")
        over = sv.ProWSyn()
    elif method is 'SMOTEIPF':  # "SMOTEIPF" in method:
        print("smoteipf")
        over = sv.SMOTE_IPF()
    elif method is 'smote':
        print("smote")
        over = SMOTE(k_neighbors=2)
        X_synth, y_syth = over.fit_resample(finaldata, output)
        return X_synth, y_syth
    elif method is 'baseline':
        return finaldata, output

    X_synth, y_syth = over.sample(finaldata, output)
    return X_synth, y_syth


def preprocess(data):
    '''
    finaldata represent the X in the data (input atrribute)
    output repesent the y in the data (output attribute)
    '''
    le = preprocessing.LabelEncoder()
    lastColumn = data.columns[-1]
    data[lastColumn] = data[lastColumn].str.replace(
        "\n", "").str.strip()
    l = data.shape[1]-1
    output = data.iloc[:, l]
    classCount = pr.classprocess(output)
    finaldata = data.iloc[:, :l]
    finaldata.iloc[:, 0] = le.fit_transform(
        finaldata.iloc[:, 0])
    output = le.fit_transform(output)

    return classCount, finaldata, output
