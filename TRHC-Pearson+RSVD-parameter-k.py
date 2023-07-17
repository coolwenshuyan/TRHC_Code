'''
Author: [coolwen]
Date: 2022-05-05 10:02:05
LastEditors: CoolWen 122311139@qq.com
LastEditTime: 2023-04-03 16:25:57
Description: 
'''

import sys 
# sys.path.append(r'/dataset/wzc_dataset/python/矩阵分解中噪声的处理/globalConfiguration') 

sys.path.append(r'D:\矩阵分解中噪声的处理\globalConfiguration')
import os
from ParameterConfiguration import datapath 
from ParameterConfiguration import testfilepath 
from dataprocessJ2 import DataProcess
from ParameterConfiguration import Seed 
from sklearn.model_selection import KFold
from ParameterConfiguration import k_fold 
import numpy as np
from ParameterConfiguration import Kdata
from ParameterConfiguration import Vdata
# from ParameterConfiguration import Kneighbor
from ParameterConfiguration import delta
from ParameterConfiguration import threshold
from knn import KNN
from similarity_distance import Similarity
from performace import Myperformace
from collections import defaultdict
from ParameterConfiguration import savepath 
from RSVD import SVD 
from ParameterConfiguration import rank
from ParameterConfiguration import lam
from ParameterConfiguration import gam
from ParameterConfiguration import step

if __name__ == '__main__':
    import socket

    hostname = socket.gethostname()
    print("Hostname:", hostname)
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


        thisFileSavePath = savepath + fileName
        print('thisFileSavePath = {}'.format(thisFileSavePath)) 
        if os.path.exists(thisFileSavePath):
            print('savepath exist') 
        else:
            # os.mkdir(savepath)      
            os.makedirs(thisFileSavePath)
        result_save_path = thisFileSavePath+'//'+fileName+runname+hostname+'.xlsx'
        print('result_save_path = {}'.format(result_save_path)) 


        print('开始加载数据……')
        dp = DataProcess()
        # dataPath = testfilepath
        data = dp.getData(dataPath)
        # print(data)


        print('划分测试集和训练集')
        sd = Seed
        KF = KFold(n_splits=k_fold)
        data = np.array(data)



        MaedataRootmean = defaultdict(list)#
        RmsedataRootmean = defaultdict(list)#
        PrecisiondataRootmean = defaultdict(list)
        RecalldataRootmean = defaultdict(list)
        F1Rootmean = defaultdict(list)

        Kneighbors = ''
        for Kneighbor in range(11,13,1):
            Kneighbors = Kneighbors +'-'+str(Kneighbor)
            one_neighbor_Maes = []
            one_neighbor_Rmse = []
            one_neighbor_Precision = []
            one_neighbor_Recall = []
            one_neighbor_F1 = []
            print('Kneighbor = {}'.format(Kneighbor))
            number = 1





            for train_index,test_index in KF.split(data):
                # print(train_index)
                # print(test_index)

                print("----------------------第{}次验证{}----------------------".format(number,fileName))
                # number += 1
                # if number != 5:
                #     continue
                print('得到本次的测试集和训练集')
                train =data[train_index].tolist()
                test = data[test_index].tolist()
                # users,ids = dp.getUserAndIds(train)
                del train_index
                del test_index

                print('求解用户和项目的评分类型')
                classify=dp.threeclassifyUserAndItemByGloabal(train,Vdata,Kdata)
                # print(classify)








                print('开始求强中弱的不一致性……')    
                Strong, Medium, Weak = dp.getTainDataStrongMeduimWeakConsistency(train, classify, Vdata, Kdata)
                del classify
                print('StrongConsistent = %s, MediumConsistent = %s, WeakStrongConsistent = %s' % (
                    len(Strong), len(Medium), len(Weak)))
                print('获取原始训练集的相似度')
                knnOriginal = KNN(train)
                del train
                sd = Similarity()
                SimOriginal = knnOriginal.getUserSim(sd.pearson)

                print('对中和弱不一致性评分进行预测')
                prePossibleNoise = knnOriginal.predict(Medium+Weak, SimOriginal, Kneighbor)
                # print(prePossibleNoise)
                del SimOriginal
                del knnOriginal

                print('均方根平均更正中和弱不一致性')        
                corretNoiseRootmean , cnumberRootmean =dp.correctNoiseByRootmean(prePossibleNoise,Medium,Weak,delta)
                del prePossibleNoise
                del Medium
                del Weak
                print('使用更正后的数据集进行训练')
                correct_svdRootmean = SVD(Strong + corretNoiseRootmean, rank)
                del Strong
                del corretNoiseRootmean
                correct_svdRootmean.train(steps=step, gamma=gam, Lambda=lam)           
                print('更正后数据进行预测')
                preTAllRootmean = correct_svdRootmean.predict(test)  
                del correct_svdRootmean
                print('接近值处理')
                preTAllCloseRootmean = []
                for preOneRootmean in preTAllRootmean:
                    user, item ,rating = preOneRootmean
                    newRating = dp.getClosestNumber(rating)
                    # preTAll.remove(preOne)
                    preTAllCloseRootmean.append([user,item,newRating])
                preTAllRootmean = preTAllCloseRootmean
                del preTAllCloseRootmean            

                print('计算MaeRootmean和RmseRootmean')
                myp = Myperformace()
                maeRootmean, rmseRootmean = myp.getMAE_list(preTAllRootmean,test)
                print('maeRootmean= %s, rmseRootmean=%s ' % (maeRootmean, rmseRootmean))
                precisionRootmean, recallRootmean ,f1Rootmean= myp.precision_recall_at_threshold_baseAll(test, preTAllRootmean, threshold)
                print('precisionRootmean= %s, recallRootmean=%s , F1Rootmean = %s' % (precisionRootmean, recallRootmean,f1Rootmean))
                del preTAllRootmean


                one_neighbor_Maes.append(maeRootmean)
                one_neighbor_Rmse.append(rmseRootmean)
                one_neighbor_Precision.append(precisionRootmean)
                one_neighbor_Recall.append(recallRootmean)
                one_neighbor_F1.append(f1Rootmean)

                number += 1
            avrage_mae = sum(one_neighbor_Maes) / len(one_neighbor_Maes)
            avrage_rmse = sum(one_neighbor_Rmse) / len(one_neighbor_Rmse)
            avrage_Recall= sum(one_neighbor_Recall) / len(one_neighbor_Recall)
            avrage_Precision = sum(one_neighbor_Precision) / len(one_neighbor_Precision)
            avrage_F1 = sum(one_neighbor_F1) / len(one_neighbor_F1)

            MaedataRootmean[fileName].append(maeRootmean)
            RmsedataRootmean[fileName].append(rmseRootmean)
            PrecisiondataRootmean[fileName].append(precisionRootmean)
            RecalldataRootmean[fileName].append(recallRootmean)
            F1Rootmean[fileName].append(f1Rootmean)

 



        import pandas as pd
        dfMaedataRootmean = pd.DataFrame(MaedataRootmean) 
        dfRmsedataRootmean = pd.DataFrame(RmsedataRootmean) 
        dfPrecisiondataRootmean = pd.DataFrame(PrecisiondataRootmean) 
        dfRecalldataRootmean = pd.DataFrame(RecalldataRootmean)
        dfF1Rootmean = pd.DataFrame(F1Rootmean)
        result_save_path = thisFileSavePath+'//'+fileName+runname+str(rank)+Kneighbors+'rank.xlsx'
        writer = pd.ExcelWriter(result_save_path)
            # str = ['pre','rec']
        dfMaedataRootmean.to_excel(writer, sheet_name= "maeRootmean")
        dfRmsedataRootmean.to_excel(writer, sheet_name= "RmseRootmean")
        dfPrecisiondataRootmean.to_excel(writer, sheet_name= "precisionRootmean")
        dfRecalldataRootmean.to_excel(writer, sheet_name= "relRootmean")
        dfF1Rootmean.to_excel(writer, sheet_name= "f1Rootmean")


        writer.close()           