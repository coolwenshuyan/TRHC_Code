'''
Author: [coolwen]
Date: 2022-05-07 17:11:47
LastEditors: CoolWen 122311139@qq.com
LastEditTime: 2023-01-06 19:29:39
Description: 
'''
#计算三只策……

import os
import sys
# sys.path.append(r'/dataset/wzc_dataset/python/矩阵分解中噪声的处理/globalConfiguration') 
sys.path.append(r'F:\studyTest\python\协同过滤\矩阵分解中噪声的处理\globalConfiguration') 
from ParameterConfiguration import datapath 
from ParameterConfiguration import testfilepath 
from dataprocessJ2 import DataProcess
from ParameterConfiguration import Seed 
from sklearn.model_selection import KFold
from ParameterConfiguration import k_fold 
import numpy as np
from ParameterConfiguration import Kdata
from ParameterConfiguration import Vdata
from ParameterConfiguration import Kneighbor
from ParameterConfiguration import delta
from ParameterConfiguration import threshold
from performace import Myperformace
from RSVD import SVD 
from ParameterConfiguration import rank
from ParameterConfiguration import lam
from ParameterConfiguration import gam
from ParameterConfiguration import step
from collections import defaultdict
from ParameterConfiguration import savepath 


if __name__ == '__main__':
    print('开始读取文件路径:')
    path = datapath  # 需要根据你的情况修改
    file_list = os.listdir(path)
    print(file_list)
    #去掉文件后缀，只要文件名称
    runname=os.path.basename(__file__).split(".")[0]
    print(runname)
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
            # users,ids = dp.getUser  AndIds(train)

            print('求解用户和项目的评分类型')
            classify=dp.threeclassifyUserAndItemByGloabal(train,Vdata,Kdata)
            # print(classify)











            print('开始求强中弱的不一致性……')    
            Strong, Medium, Weak = dp.getTainDataStrongMeduimWeakConsistency(train, classify, Vdata, Kdata)
            print('StrongConsistent = %s, MediumConsistent = %s, WeakStrongConsistent = %s' % (
                len(Strong), len(Medium), len(Weak)))
            
            print('使用原始训练集进行训练')
           
            origin_svd = SVD(train, rank)
            origin_svd.train(steps=step, gamma=gam, Lambda=lam)#注释
            # print(userSim)

            print('对中和弱不一致性评分进行预测')
            prePossibleNoise = origin_svd.predict(Medium+Weak)
            # print(prePossibleNoise)


            print('更正中和弱不一致性')
            corretNoise , cnumber =dp.correctNoise(prePossibleNoise,Medium,Weak,delta)
            # print(corretNoise,number)































            print('使用更正后的数据集进行训练')
            # print(noNoise+corretNoise)
            correct_svd = SVD(Strong + corretNoise, rank)
            correct_svd.train(steps=step, gamma=gam, Lambda=lam)
           
            print('所有没有评分的项目都进行预测')
            # preTest = knnOriginal.predict(test, userSimCorrections, Kneighbor)
            # print(test)

            # testNoRatings = dp.getNo_Rating_data(train,users,ids)
            preTAll = correct_svd.predict(test)
            # print(preTAll)


          
            print('计算Mae和Rmse')
            myp = Myperformace()
            mae, rmse = myp.getMAE_list(preTAll,test)
            print('mae= %s, rmse=%s ' % (mae, rmse))
            
            
            
            # print('保存预测数据')
            # pre_save_path = savepath+fileName+str(number)+runname+'PreOriginal.txt'
            # print(pre_save_path)
            # f=open(pre_save_path,'w')
            # f.write(str(preTAll))
            # f.close()
            

            print('接近值处理')
            preTAllClose = []
            for preOne in preTAll:
                user, item ,rating = preOne
                newRating = dp.getClosestNumber(rating)
                # preTAll.remove(preOne)
                preTAllClose.append([user,item,newRating])

            preTAll = preTAllClose
            del preTAllClose
           

            

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
        # writer.save()
        writer.close()           