'''
Author: [coolwen]
Date: 2022-05-05 10:02:05
LastEditors: CoolWen 122311139@qq.com
LastEditTime: 2023-06-13 08:47:05
Description: 
'''
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
        result_save_path = savepath+fileName+runname+'.xlsx'
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

            print('求解用户和项目的评分类型')
            classify=dp.threeclassifyUserAndItemByGloabal(train,Vdata,Kdata)
            # print(classify)








            print('开始求强中弱的不一致性……')    
            Strong, Medium, Weak = dp.getTainDataStrongMeduimWeakConsistency(train, classify, Vdata, Kdata)
            print('StrongConsistent = %s, MediumConsistent = %s, WeakStrongConsistent = %s' % (
                len(Strong), len(Medium), len(Weak)))
            print('获取原始训练集的相似度')
            knnOriginal = KNN(train)
            sd = Similarity()
            SimOriginal = knnOriginal.getUserSim(sd.pearson)

            print('对中和弱不一致性评分进行预测')
            prePossibleNoise = knnOriginal.predict(Medium+Weak, SimOriginal, Kneighbor)
            # print(prePossibleNoise)
            del SimOriginal
            del knnOriginal

            print('更正中和弱不一致性')
            corretNoise , cnumber =dp.correctNoise(prePossibleNoise,Medium,Weak,delta)
            print('一共更正多少个,{}'.format(cnumber))
            del prePossibleNoise
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            print('转为基于项目CF')
            baseItemNoNoise = dp.chageUsersAndItems(Strong)
            baseItemCorretNoise = dp.chageUsersAndItems(corretNoise)
            baseItemTest = dp.chageUsersAndItems(test)

            print('计算更正后的用户相似度')
            # print(noNoise+corretNoise)
            knnCorrections = KNN(baseItemNoNoise + baseItemCorretNoise)
            sd = Similarity()
            itemSimCorrections = knnCorrections.getUserSim(sd.pearson)
            
 
            # print("保存基于更正后项目相似度写入文本")
            # simItem_save_path = savepath+fileName+runname+str(number)+'itemSimCorrections.txt'
            # f=open(simItem_save_path,'w')
            # f.write('\n'+str(itemSimCorrections)) 
            # f.close


            
            print('所有没有评分的项目都进行预测')
            # preTest = knnOriginal.predict(test, userSimCorrections, Kneighbor)
            # print(test)

            # testNoRatings = dp.getNo_Rating_data(train,users,ids)
            preTAll = knnCorrections.predict(baseItemTest, itemSimCorrections, Kneighbor)
            # print(preTAll)

            del knnCorrections
            del itemSimCorrections


            print('转为基于用户的CF')
            baseUserPreAll = dp.chageUsersAndItems(preTAll)


            preTAll = baseUserPreAll
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