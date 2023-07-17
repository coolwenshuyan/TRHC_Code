'''
Author: CoolWen 122311139@qq.com
Date: 2023-01-06 16:27:36
LastEditors: CoolWen 122311139@qq.com
LastEditTime: 2023-01-28 20:25:31
FilePath: \python\协同过滤\矩阵分解中噪声的处理\阶段二\分三类基于用户相似度基于Item进行预测\dataprocessJ2.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from dataprocess import DataProcess
from pandas import DataFrame
import math
class DataProcess(DataProcess):
        def getTainDataStrongMeduimWeakConsistency(self,data, classify, v, k):
            Strong = []
            Medium = []
            Weak = []
            # print(classify)
            U_s, U_a, U_w, I_s, I_a, I_w = classify[0], classify[
                1], classify[2], classify[3], classify[4], classify[5]
            for line in data:
                (userId, itemId, rating) = line[0], line[1], line[2]
                if (userId in U_w and itemId in I_s) or (userId in U_s and itemId in I_w):
                    Weak.append(line)
                elif (userId in U_s and itemId in I_s) or (userId in U_a and itemId in I_a) or (userId in U_w and itemId in I_w):
                    Strong.append(line)
                else:
                    Medium.append(line)
            # print('Strong={},Medium={},Weak={}'.format(
            #     len(Strong), len(Medium), len(Weak)))
            return Strong, Medium, Weak
        
        
        def correctNoise(self,predUpdate, Medium,Weak, delta=1):
            number = 0
            print('算术平均值更正中不一致性')
            N_Medium = Medium.copy()
            N_Weak = Weak.copy()
            columns = ['users', 'items', 'ratings']
            dataframe_predUpdate = DataFrame(predUpdate, columns=columns)
            dataframe_predUpdate.drop_duplicates(subset = ['users','items'],keep='last',inplace=True)
            for line in Medium:
                (userId, itemId, rating) = line[0], line[1], line[2]
                if not dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)].empty:
                    # Rpre =dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] 
                    # print(Rpre)
                    preR =float(dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] )
                    preR = (rating + preR) / 2
                    preR = self.getClosestNumber(preR)
                    N_Medium.remove(line)
                    N_Medium.append([userId,itemId,preR])
                    number += 1
            
            print('更正弱不一致性')           
            for line in Weak:
                (userId, itemId, rating) = line[0], line[1], line[2]
                if not dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)].empty:
                    # Rpre =dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] 
                    # print(Rpre)
                    preR =float(dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] )
                    preR = self.getClosestNumber(preR)
                    N_Weak.remove(line)
                    N_Weak.append([userId,itemId,preR])
                    number += 1
            return N_Medium + N_Weak, number
        
        
        #几何平均
        def correctNoiseByGeometricMean(self,predUpdate, Medium,Weak, delta=1):
            number = 0
            print('几何平均更正中不一致性')
            N_Medium = Medium.copy()
            N_Weak = Weak.copy()
            columns = ['users', 'items', 'ratings']
            dataframe_predUpdate = DataFrame(predUpdate, columns=columns)
            dataframe_predUpdate.drop_duplicates(subset = ['users','items'],keep='last',inplace=True)
            for line in Medium:
                (userId, itemId, rating) = line[0], line[1], line[2]
                if not dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)].empty:
                    # Rpre =dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] 
                    # print(Rpre)
                    preR =float(dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] )
                    try:
                        pO = preR
                        preR = math.sqrt(rating * preR)
                    except:
                        print('------------------------------------------------------------rating={},preR={}--------------------------------------------------------'.format(rating,pO))
                    preR = self.getClosestNumber(preR)
                    N_Medium.remove(line)
                    N_Medium.append([userId,itemId,preR])
                    number += 1
            
            print('更正弱不一致性')  
            for line in Weak:
                (userId, itemId, rating) = line[0], line[1], line[2]
                if not dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)].empty:
                    # Rpre =dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] 
                    # print(Rpre)
                    preR =float(dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] )
                    preR = self.getClosestNumber(preR)
                    N_Weak.remove(line)
                    N_Weak.append([userId,itemId,preR])
                    number += 1
            return N_Medium + N_Weak, number         

        #均方根平均
        def correctNoiseByRootmean(self,predUpdate, Medium,Weak, delta=1):
            number = 0
            print('均方根平均更正中不一致性')
            N_Medium = Medium.copy()
            N_Weak = Weak.copy()
            columns = ['users', 'items', 'ratings']
            dataframe_predUpdate = DataFrame(predUpdate, columns=columns)
            dataframe_predUpdate.drop_duplicates(subset = ['users','items'],keep='last',inplace=True)
            for line in Medium:
                (userId, itemId, rating) = line[0], line[1], line[2]
                if not dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)].empty:
                    # Rpre =dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] 
                    # print(Rpre)
                    preR =float(dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] )
                    preR = math.sqrt((rating ** 2 + preR ** 2) / 2)
                    preR = self.getClosestNumber(preR)
                    N_Medium.remove(line)
                    N_Medium.append([userId,itemId,preR])
                    number += 1
        
            print('更正弱不一致性')           
            for line in Weak:
                (userId, itemId, rating) = line[0], line[1], line[2]
                if not dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)].empty:
                    # Rpre =dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] 
                    # print(Rpre)
                    preR =float(dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] )
                    preR = self.getClosestNumber(preR)
                    N_Weak.remove(line)
                    N_Weak.append([userId,itemId,preR])
                    number += 1
            return N_Medium + N_Weak, number



        #算术平均数
        def correctNoiseArithmeticmean(self,predUpdate, Medium,Weak, delta=1):
            number = 0
            print('算术平均值更正中不一致性')
            N_Medium = Medium.copy()
            N_Weak = Weak.copy()
            columns = ['users', 'items', 'ratings']
            dataframe_predUpdate = DataFrame(predUpdate, columns=columns)
            dataframe_predUpdate.drop_duplicates(subset = ['users','items'],keep='last',inplace=True)
            for line in Medium:
                (userId, itemId, rating) = line[0], line[1], line[2]
                if not dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)].empty:
                    # Rpre =dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] 
                    # print(Rpre)
                    preR =float(dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] )
                    preR = (rating + preR) / 2
                    preR = self.getClosestNumber(preR)
                    N_Medium.remove(line)
                    N_Medium.append([userId,itemId,preR])
                    number += 1
            
            print('更正弱不一致性')           
            for line in Weak:
                (userId, itemId, rating) = line[0], line[1], line[2]
                if not dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)].empty:
                    # Rpre =dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] 
                    # print(Rpre)
                    preR =float(dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] )
                    preR = self.getClosestNumber(preR)
                    N_Weak.remove(line)
                    N_Weak.append([userId,itemId,preR])
                    number += 1
            return N_Medium + N_Weak, number




            

