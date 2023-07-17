'''
Author: [coolwen]
Date: 2022-05-07 17:11:47
LastEditors: CoolWen 122311139@qq.com
LastEditTime: 2023-02-04 09:56:44
Description: 
'''
#计算三只策……

import sys 
sys.path.append(r'/dataset/wzc_dataset/python/矩阵分解中噪声的处理/globalConfiguration') 
# sys.path.append(r'F:\studyTest\python\协同过滤\矩阵分解中噪声的处理\globalConfiguration') 
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
from ParameterConfiguration import Kneighbor
from ParameterConfiguration import delta
from ParameterConfiguration import threshold
from performace import Myperformace
from RSVD import SVD 
# from ParameterConfiguration import rank
from ParameterConfiguration import lam
from ParameterConfiguration import gam
from ParameterConfiguration import step
from collections import defaultdict
from ParameterConfiguration import savepath 


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
    print(runname)
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
        print('保存文件路径名字:'+result_save_path)



        print('开始加载数据……')
        dp = DataProcess()
        # dataPath = testfilepath
        data = dp.getData(dataPath)
        # print(data)


        print('划分测试集和训练集')
        sd = Seed
        KF = KFold(n_splits=k_fold)
        data = np.array(data)
        number = 1
        MaedataRootmean = defaultdict(list)#
        RmsedataRootmean = defaultdict(list)#
        PrecisiondataRootmean = defaultdict(list)
        RecalldataRootmean = defaultdict(list)
        F1Rootmean = defaultdict(list)

        k_Ranks = ""
        for k_Rank in range(11,21,1):
            k_Ranks = k_Ranks +'-'+str(k_Rank)
            one_k_number_Maes = []
            one_k_number_Rmse = []
            one_k_number_Precision = []
            one_k_number_Recall = []
            one_k_number_F1 = []
            print('k_Rank = {}'.format(k_Rank))
            number = 1        

        
            for train_index,test_index in KF.split(data):
                print("----------------------第{}次验证{}----------------------".format(number,fileName))
                # if number != 1:
                #    number += 1
                #    continue
                print('得到本次的测试集和训练集')
                train =data[train_index].tolist()
                test = data[test_index].tolist()
                # users,ids = dp.getUser  AndIds(train)
                del train_index
                del test_index
                print('求解用户和项目的评分类型')
                classify=dp.threeclassifyUserAndItemByGloabal(train,Vdata,Kdata)
                # print(classify)



                print('开始求强中弱的不一致性……')    
                Strong, Medium, Weak = dp.getTainDataStrongMeduimWeakConsistency(train, classify, Vdata, Kdata)
                print('StrongConsistent = %s, MediumConsistent = %s, WeakStrongConsistent = %s' % (
                    len(Strong), len(Medium), len(Weak)))
                
                print('使用原始训练集进行训练')
            
                origin_svd = SVD(train, k_Rank)
                origin_svd.train(steps=step, gamma=gam, Lambda=lam)#注释
                # print(userSim)

                print('对中和弱不一致性评分进行预测')
                prePossibleNoise = origin_svd.predict(Medium+Weak)
                # print(prePossibleNoise)





                print('均方根平均更正中和弱不一致性')        
                corretNoiseRootmean , cnumberRootmean =dp.correctNoiseByRootmean(prePossibleNoise,Medium,Weak,delta)
                print('使用更正后的数据集进行训练')
                correct_svdRootmean = SVD(Strong + corretNoiseRootmean, k_Rank)
                correct_svdRootmean.train(steps=step, gamma=gam, Lambda=lam)           
                print('更正后数据进行预测')
                preTAllRootmean = correct_svdRootmean.predict(test) 
                print('接近值处理')
                preTAllCloseRootmean = []
                for preOneRootmean in preTAllRootmean:
                    user, item ,rating = preOneRootmean
                    newRating = dp.getClosestNumber(rating)
                    # preTAll.remove(preOne)
                    preTAllCloseRootmean.append([user,item,newRating])
                preTAllRootmean = preTAllCloseRootmean
                del preTAllCloseRootmean
                print('计算Mae和Rmse')
                myp = Myperformace()
                maeRootmean, rmseRootmean = myp.getMAE_list(preTAllRootmean,test)
                print('mae= %s, rmse=%s ' % (maeRootmean, rmseRootmean))

                precisionRootmean, recallRootmean ,f1Rootmean= myp.precision_recall_at_threshold_baseAll(test, preTAllRootmean, threshold)           
                print('precisionRootmean= %s, recallRootmean=%s , F1Rootmean = %s' % (precisionRootmean, recallRootmean,f1Rootmean))


                one_k_number_Maes.append(maeRootmean)
                one_k_number_Rmse.append(rmseRootmean)
                one_k_number_Precision.append(precisionRootmean)
                one_k_number_Recall.append(recallRootmean)
                one_k_number_F1.append(f1Rootmean)
                

                # break
                number += 1
            avrage_mae = sum(one_k_number_Maes) / len(one_k_number_Maes)
            avrage_rmse = sum(one_k_number_Rmse) / len(one_k_number_Rmse)
            one_k_number_Precision= sum(one_k_number_Precision) / len(one_k_number_Precision)
            one_k_number_Recall = sum(one_k_number_Recall) / len(one_k_number_Recall)
            avrage_F1 = sum(one_k_number_F1) / len(one_k_number_F1)
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
        result_save_path = thisFileSavePath+'//'+hostname+fileName+runname+k_Ranks+'.xlsx'
        print('保存文件路径名字:'+result_save_path)       

        writer = pd.ExcelWriter(result_save_path)
            # str = ['pre','rec']
        dfMaedataRootmean.to_excel(writer, sheet_name= "maeRootmean")
        dfRmsedataRootmean.to_excel(writer, sheet_name= "RmseRootmean")
        dfPrecisiondataRootmean.to_excel(writer, sheet_name= "precisionRootmean")
        dfRecalldataRootmean.to_excel(writer, sheet_name= "relRootmean")
        dfF1Rootmean.to_excel(writer, sheet_name= "f1Rootmean")
        # writer.save()

 

        writer.close()           