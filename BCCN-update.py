'''
Author: [coolwen]
Date: 2022-05-05 10:02:05
LastEditors: CoolWen 122311139@qq.com
LastEditTime: 2023-05-17 10:47:37
Description: 
'''
import sys
sys.path.append(r'F:\studyTest\python\协同过滤\矩阵分解中噪声的处理\globalConfiguration') 
import os
from ParameterConfiguration import datapath 
from ParameterConfiguration import testfilepath 
from dataprocess import DataProcess
from ParameterConfiguration import Seed 
from sklearn.model_selection import KFold
from ParameterConfiguration import k_fold 
import numpy as np
from ParameterConfiguration import Kdata
from ParameterConfiguration import Vdata
from ParameterConfiguration import Kneighbor
from ParameterConfiguration import delta
from ParameterConfiguration import threshold
from knn import KNN
from similarity_distance import Similarity
from performace import Myperformace
from collections import defaultdict
from ParameterConfiguration import savepath 


if __name__ == '__main__':
    print('开始读取文件路径:')
    path = datapath  # 需要根据你的情况修改
    file_list = os.listdir(path)
    print(file_list)
            #去掉文件后缀，只要文件名称
    runname=os.path.basename(__file__).split(".")[0]
    for pa in file_list:
        if pa == '.ipynb_checkpoints':
            continue
        fileName = os.path.basename(pa[:-4])
        print(fileName)        
        dataPath = path+pa
        print(dataPath)


        print('开始加载数据……')
        dp = DataProcess()
        # dataPath = testfilepath
        data = dp.getData(dataPath)
        # print(data)
        result_save_path = savepath+fileName+runname+'.xlsx'
        print('保存文件路径名字:'+result_save_path)

        print('划分测试集和训练集')
        sd = Seed
        KF = KFold(n_splits=k_fold)
        data = np.array(data)
        number = 1
        Maedata = defaultdict(list)#
        Rmsedata = defaultdict(list)#
        Precisiondata = defaultdict(list)
        Recalldata = defaultdict(list)
        F1 = defaultdict(list)
        for train_index,test_index in KF.split(data):
            print("----------------------第{}次验证----------------------".format(number))
            print('得到本次的测试集和训练集')
            train =data[train_index].tolist()
            test = data[test_index].tolist()
            # users,ids = dp.getUserAndIds(train)

            del train_index
            del test_index
            print('求解用户和项目的评分类型')
            classify=dp.threeclassifyUserAndItemByGloabal(train,Vdata,Kdata)
            # print(classify)


            # print('得到可能的噪声评分列表')
            # noNoise, Noise = dp.getTainDataWithNoiseAndWithout(train, classify, v=Vdata, k=Kdata)
            # print(Noise)
            # print(noNoise)    


        

            print('更正可能的噪声')
            #对一致评分当中（noNoise）的没有满足阈值条件的一致性评分进行更正
            corretNoise , cnumber =dp.replaceNoiseByVAndK(train,classify,Vdata,Kdata)
            # print(corretNoise,cnumber)
            del train
            del classify
            print('计算更正后的用户相似度')
            # print(noNoise+corretNoise)
            knnCorrections = KNN(corretNoise)
            sd = Similarity()
            SimCorrections = knnCorrections.getUserSim(sd.Bhattacharya)


            print('所有没有评分的项目都进行预测')
            # preTest = knnOriginal.predict(test, userSimCorrections, Kneighbor)
            # print(test)

            # testNoRatings = dp.getNo_Rating_data(train,users,ids)
            preTAll = knnCorrections.predict(test, SimCorrections, Kneighbor)
            # print(preTAll)



            print('计算Mae和Rmse')
            myp = Myperformace()
            mae, rmse = myp.getMAE_list(preTAll,test)
            print('mae= %s, rmse=%s ' % (mae, rmse))
            

            precision, recall ,f1= myp.precision_recall_at_threshold_baseAll(test, preTAll, threshold)



            Maedata[fileName].append(mae)
            Rmsedata[fileName].append(rmse)
            Precisiondata[fileName].append(precision)
            Recalldata[fileName].append(recall)
            F1[fileName].append(f1)
            print('precision= %s, recall=%s , F1 = %s' % (precision, recall,f1))
            # break


            number += 1
        import pandas as pd
        dfMaedata = pd.DataFrame(Maedata) 
        dfRmsedata = pd.DataFrame(Rmsedata) 
        dfPrecisiondata = pd.DataFrame(Precisiondata) 
        dfRecalldata = pd.DataFrame(Recalldata)
        dfF1 = pd.DataFrame(F1)
        # print(dfMaedata)
        # print("-----------------------------------------------------") 
        # print(dfRmsedata) 
        # print("-----------------------------------------------------") 
        # print(dfPrecisiondata) 
        # print("-----------------------------------------------------") 
        # print(dfRecalldata) 

        writer = pd.ExcelWriter(result_save_path)
            # str = ['pre','rec']
        dfMaedata.to_excel(writer, sheet_name= "mae")
        dfRmsedata.to_excel(writer, sheet_name= "Rmse")
        dfPrecisiondata.to_excel(writer, sheet_name= "precision")
        dfRecalldata.to_excel(writer, sheet_name= "rel")
        dfF1.to_excel(writer, sheet_name= "f1")
        writer.close()           