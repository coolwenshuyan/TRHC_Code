'''
Author: your name
Date: 2021-12-17 11:41:05
LastEditTime: 2023-01-06 16:28:42
LastEditors: CoolWen 122311139@qq.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \python\矩阵分解中噪声的处理\三支的噪声处理\datadeal.py
'''

from pandas import DataFrame
import numpy as np 
from collections import defaultdict
from ParameterConfiguration import ratingsRange
class DataProcess:


    def getClosestNumber(self,originalNumber):
            listNumbers = ratingsRange
            return min(listNumbers, key=lambda x: abs(x - originalNumber))
    def getUserAndIds(self,dataframe):
        columns = ['users', 'items', 'ratings']
        dataframe = DataFrame(dataframe, columns=columns)
        # print(dataframe)
        usersIds = list(set(dataframe['users']))
        # print(usersIds)
        print('用户数目: %s' % len(usersIds))
        itemsIds = list(set(dataframe['items']))
        # print(itemsIds)
        print('项目数目: %s' % len(itemsIds))
        return usersIds,itemsIds
   
    # def dfconvert
    def getData(self,filePath):  # 获取训练集和测试集的函数
        import re
        f = open(filePath, 'r')
        lines = f.readlines()
        f.close()
        data_all_random = []
        for line in lines:
            list = re.split('\n', line)
            list = eval(list[0])
            # print(type(list[2]))
            if list[2] != 0:
                # if list[2] != 0:  # 踢出评分0的数据，这部分是用户评论了但是没有评分的
                data_all_random.append([i for i in list[:3]])
        print('data_all ', len(data_all_random))
        return data_all_random

    def classifyUserAndItemByGloabal(self,trainset, v=4, k=2):
        classify = []
        U_critical = []  # 弱用户集合
        U_average = []  # 平均用户集合
        U_benevolent = []  # 强用户集合
        U_variable = []  # 可变用户集合
        I_weakly = []  # 弱项目集合
        I_averagely = []  # 平均项目集合
        I_strongly = []  # 强项目集合
        I_variably = []  # 可变项目集合
        # W_i = {}  # i用户中的弱评分集合
        # A_i = {}  # i用户中的平均评分集合
        # S_i = {}  # i用户中的强评分集合
        # W_j = {}  # j项目中的弱评分集合
        # A_j = {}  # j项目中的平均评分集合
        # S_j = {}  # j项目中的强评分集合
        C_U = {}
        C_j = {}
        # trainset.sort()
        for line in trainset:
            (userId, itemId, rating) = line[0], line[1], line[2]
            C_U.setdefault(userId, {})
            C_j.setdefault(itemId, {})
            if rating < k:
                if('W' in C_U[userId]):
                    C_U[userId]['W'] += 1
                else:
                    C_U[userId]['W'] = 1
            elif rating >=k and rating <v:
                if('A' in C_U[userId]):
                    C_U[userId]['A'] += 1
                else:
                    C_U[userId]['A'] = 1
            else:
                if('S' in C_U[userId]):
                    C_U[userId]['S'] += 1
                else:
                    C_U[userId]['S'] = 1
            if rating <k:
                if('W' in C_j[itemId]):
                    C_j[itemId]['W'] += 1
                else:
                    C_j[itemId]['W'] = 1
            elif rating >=k and rating <v:
                if('A' in C_j[itemId]):
                    C_j[itemId]['A'] += 1
                else:
                    C_j[itemId]['A'] = 1
            else:
                if('S' in C_j[itemId]):
                    C_j[itemId]['S'] += 1
                else:
                    C_j[itemId]['S'] = 1

        # print('C_U={},C_j={}'.format(len(C_U), len(C_j)))
        for i in C_U:
            if(not 'S' in C_U[i]):
                C_U[i]['S'] = 0
            if(not 'W' in C_U[i]):
                C_U[i]['W'] = 0
            if(not 'A' in C_U[i]):
                C_U[i]['A'] = 0
            # print(('C_U[i][\'W\']={},C_U[i][\'A\']={},C_U[i][\'S\']={}').format(
                # C_U[i]['W'], C_U[i]['A'], C_U[i]['S']))
            if(C_U[i]['W'] >= C_U[i]['A']+C_U[i]['S']):
                U_critical.append(i)
            elif(C_U[i]['A'] >= C_U[i]['W']+C_U[i]['S']):
                U_average.append(i)
            elif(C_U[i]['S'] >= C_U[i]['W']+C_U[i]['A']):
                U_benevolent.append(i)
            else:
                U_variable.append(i)
        for j in C_j:
            if(not 'S' in C_j[j]):
                C_j[j]['S'] = 0
            if(not 'W' in C_j[j]):
                C_j[j]['W'] = 0
            if(not 'A' in C_j[j]):
                C_j[j]['A'] = 0
            if(C_j[j]['W'] >= C_j[j]['A']+C_j[j]['S']):
                I_weakly.append(j)
            elif(C_j[j]['A'] >= C_j[j]['W']+C_j[j]['S']):
                I_averagely.append(j)
            elif(C_j[j]['S'] >= C_j[j]['W']+C_j[j]['A']):
                I_strongly.append(j)
            else:
                I_variably.append(j)
        # print(('U_w={},U_a={},U_s={},U_v={}').format(U_w,U_a,U_s,U_v))
        # print(('contU={}').format(len(trainset)))
        # print(('用户总数={}').format(len(U_w)+len(U_a)+len(U_s)+len(U_v)))
        classify = [U_critical, U_average, U_benevolent, U_variable, I_weakly, I_averagely, I_strongly, I_variably]
        print('U_critical=%s,U_average=%s,U_benevolent=%s, U_variable=%s, I_weakly=%s, I_averagely=%s, I_strongly=%s, I_variably=%s' % 
        (len(U_critical), len(U_average), len(U_benevolent), len(U_variable), len(I_weakly), len(I_averagely), len(I_strongly), len(I_variably)))
        return classify


    def getTainDataWithNoiseAndWithout(self,trainset, classify, v=4, k=2):
        noNoise = []
        Noise = []
        U_critical, U_average, U_benevolent, U_variable, I_weakly, I_averagely, I_strongly, I_variably = classify[0], classify[
            1], classify[2], classify[3], classify[4], classify[5], classify[6], classify[7]
        for line in trainset:
            (userId, itemId, rating) = line[0], line[1], line[2]
            if(rating >= k and (userId in U_critical) and (itemId in I_weakly)):
                Noise.append(line)
            elif (rating >= v  and (userId in U_average) and (itemId in I_averagely)):
                Noise.append(line)
            elif (rating < k and (userId in U_average) and (itemId in I_averagely)):
                Noise.append(line)
            elif(rating < v and (userId in U_benevolent) and (itemId in I_strongly)):
                Noise.append(line)
            else:
                noNoise.append(line)
        print('NoiseL={},noNoiseL={},trainset={}'.format(
            len(Noise), len(noNoise), len(trainset)))
        return noNoise, Noise
   
    

    def getTrainAndTestData(self,data):
        movieUser = {}
        trainSet = {}
        # length_data = len(data)#训练集长度
        for line in data:
            # for line in pice_data:
            (userId, itemId, rating) = line[0], line[1], line[2]
            trainSet.setdefault(userId, {})
            trainSet[userId].setdefault(itemId, float(rating))
            movieUser.setdefault(itemId, [])
            movieUser[itemId].append(userId)
            # 生成用户用户共有电影矩阵
        return trainSet, movieUser
    
    def getFristU2U(self,movieUser):
        u2u = {}
        for m in movieUser.keys():
            for u in movieUser[m]:
                u2u.setdefault(u, {})
                for n in movieUser[m]:
                    if u != n:
                        u2u[u].setdefault(n, [])
                        u2u[u][n].append(m)
        return u2u


    def correctNoise(self,predUpdate, Noise, delta=1):
        number = 0
        N_or = Noise.copy()
        columns = ['users', 'items', 'ratings']
        dataframe_predUpdate = DataFrame(predUpdate, columns=columns)
        dataframe_predUpdate.drop_duplicates(subset = ['users','items'],keep='last',inplace=True)
        for line in Noise:
            (userId, itemId, rating) = line[0], line[1], line[2]
            if not dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)].empty:
                # Rpre =dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] 
                # print(Rpre)
                preR =float(dataframe_predUpdate[(dataframe_predUpdate['users']==userId) & (dataframe_predUpdate['items']== itemId)]['ratings'] )
                if  (abs(preR - rating) > delta ):
                    N_or.remove(line)
                    N_or.append([userId,itemId,preR])
                    number += 1
        return N_or, number
    

    def getNo_Rating_data(self,train,users,ids):
        columns = ['users', 'items', 'ratings']
        dataframe = DataFrame(train, columns=columns)
        rfdata = []
        for user in users:
            for id in ids:
                if dataframe[(dataframe['users'] == user) & (dataframe['items']== id)].empty:
                    oneNoRating =[user, id, 0]
                    rfdata.append(oneNoRating)

        return rfdata

    def chageUsersAndItems(self,data):
        columns = ['users', 'items', 'ratings']
        df_test = DataFrame(data, columns=columns)
        df_new = df_test[['items','users','ratings']]
        return list(list(x) for x in zip(*(df_new[x].values.tolist() for x in df_new.columns)))
        # return df_new.values.tolist()
    
    def getType(self,list_ratings,v,k):
        Strong = sum(i>=v for i in list_ratings)
        Weak = sum(i<k for i in list_ratings)
        Mean = sum((i<v and i >=k) for i in list_ratings)
        # print(Strong)
        # print(Weak)
        # print(Mean)
        if Strong >= Weak + Mean:
            return 'Benevolent'
        elif Weak >= Strong + Mean:
            return 'Critical'
        elif Mean >= Strong + Weak:
            return 'Average'
        else:
            return 'Variable'

    def getVAndK(self,list):
        x_u = np.mean(list)
        p_u = np.var(list)
        k = x_u - p_u
        v = x_u + p_u
        if v > max(ratingsRange):
            v = max(ratingsRange)
        if k < min(ratingsRange):
            k = min(ratingsRange)
        return v,k


    def classifyUserAndItemByEachUserAndItem(self,trainset):
        classify = []
        U_critical = []  # 弱用户集合
        U_average = []  # 平均用户集合
        U_benevolent = []  # 强用户集合
        U_variable = []  # 可变用户集合
        I_weakly = []  # 弱项目集合
        I_averagely = []  # 平均项目集合
        I_strongly = []  # 强项目集合
        I_variably = []  # 可变项目集合
        columns = ['users', 'items', 'ratings']
        df_trainset = DataFrame(trainset, columns=columns)
        # print(df_trainset)
        usersTestIds = list(set(df_trainset['users']))
        v_k_user = defaultdict(list)
        v_k_item = defaultdict(list)
        #找出用户属于那种类型
        for user in usersTestIds:
            trainset_user = df_trainset[df_trainset['users'] == user]
            ratings_user = list(trainset_user['ratings'])

            v_user,k_user = self.getVAndK(ratings_user)
            v_k_user[user] = [v_user,k_user]
            type_user = self.getType(ratings_user,v_user,k_user)
            # print(type_user)
            if type_user == 'Benevolent':
                U_benevolent.append(user)
            elif type_user == 'Critical':
                U_critical.append(user)
            elif type_user == 'Average':
                U_average.append(user)
            else:
                U_variable.append(user)
        itemsTestIds = list(set(df_trainset['items']))
        for item in itemsTestIds:
            trainset_item = df_trainset[df_trainset['items'] == item]
            ratings_item = list(trainset_item['ratings'])
            v_item,k_item = self.getVAndK(ratings_item)
            v_k_item[item] = [v_item,k_item]
            # print(ratings_item)
            type_item = self.getType(ratings_item,v_item,k_item)
            # print(type_user)
            if type_item == 'Benevolent':
                I_strongly.append(item)
            elif type_item == 'Critical':
                I_weakly.append(item)
            elif type_item == 'Average':
                I_averagely.append(item)
            else:
                I_variably.append(item)
        classify = [U_critical, U_average, U_benevolent, U_variable, I_weakly, I_averagely, I_strongly, I_variably]
        return classify,v_k_user,v_k_item
    


    def getTainDataWithNoiseAndWithoutByEachUserAndItem(self,trainset, classify, base_type,v_k):
        noNoise = []
        Noise = []
        U_critical, U_average, U_benevolent, U_variable, I_weakly, I_averagely, I_strongly, I_variably = classify[0], classify[
            1], classify[2], classify[3], classify[4], classify[5], classify[6], classify[7]
        for line in trainset:
            (userId, itemId, rating) = line[0], line[1], line[2]
            if base_type == 'user':
                k = v_k[userId][1]
                v = v_k[userId][0]
            else:
                k = v_k[itemId][1]
                v = v_k[itemId][0]
            if(rating >= k and (userId in U_critical) and (itemId in I_weakly)):
                Noise.append(line)
            elif((rating >= v or rating < k) and (userId in U_average) and (itemId in I_averagely)):
                Noise.append(line)
            elif(rating < v and (userId in U_benevolent) and (itemId in I_strongly)):
                Noise.append(line)
            else:
                noNoise.append(line)
        print('NoiseL={},noNoiseL={},trainset={}'.format(
            len(Noise), len(noNoise), len(trainset)))
        return noNoise, Noise
    

    def threeclassifyUserAndItemByGloabal(self,trainset, v=4, k=2):
        classify = []
        U_critical = []  # 弱用户集合
        U_average = []  # 平均用户集合
        U_benevolent = []  # 强用户集合
        U_variable = []  # 可变用户集合
        I_weakly = []  # 弱项目集合
        I_averagely = []  # 平均项目集合
        I_strongly = []  # 强项目集合
        I_variably = []  # 可变项目集合
        # W_i = {}  # i用户中的弱评分集合
        # A_i = {}  # i用户中的平均评分集合
        # S_i = {}  # i用户中的强评分集合
        # W_j = {}  # j项目中的弱评分集合
        # A_j = {}  # j项目中的平均评分集合
        # S_j = {}  # j项目中的强评分集合
        C_U = {}
        C_j = {}
        # trainset.sort()
        for line in trainset:
            (userId, itemId, rating) = line[0], line[1], line[2]
            C_U.setdefault(userId, {})
            C_j.setdefault(itemId, {})
            if rating < k:
                if('W' in C_U[userId]):
                    C_U[userId]['W'] += 1
                else:
                    C_U[userId]['W'] = 1
            elif rating >=k and rating <v:
                if('A' in C_U[userId]):
                    C_U[userId]['A'] += 1
                else:
                    C_U[userId]['A'] = 1
            else:
                if('S' in C_U[userId]):
                    C_U[userId]['S'] += 1
                else:
                    C_U[userId]['S'] = 1
            if rating < k:
                if('W' in C_j[itemId]):
                    C_j[itemId]['W'] += 1
                else:
                    C_j[itemId]['W'] = 1
            elif rating >=k and rating <v:
                if('A' in C_j[itemId]):
                    C_j[itemId]['A'] += 1
                else:
                    C_j[itemId]['A'] = 1
            else:
                if('S' in C_j[itemId]):
                    C_j[itemId]['S'] += 1
                else:
                    C_j[itemId]['S'] = 1

        # print('C_U={},C_j={}'.format(C_U, C_j))
        for i in C_U:
            if(not 'S' in C_U[i]):
                C_U[i]['S'] = 0
            if(not 'W' in C_U[i]):
                C_U[i]['W'] = 0
            if(not 'A' in C_U[i]):
                C_U[i]['A'] = 0
            # print(('C_U[i][\'W\']={},C_U[i][\'A\']={},C_U[i][\'S\']={}').format(
                # C_U[i]['W'], C_U[i]['A'], C_U[i]['S']))
            type_u  = max(C_U[i],key=C_U[i].get)
            if type_u == 'W':
                U_critical.append(i)
            elif type_u == 'A':
                U_average.append(i)
            elif type_u == 'S':
                U_benevolent.append(i)
        for j in C_j:
            if(not 'S' in C_j[j]):
                C_j[j]['S'] = 0
            if(not 'W' in C_j[j]):
                C_j[j]['W'] = 0
            if(not 'A' in C_j[j]):
                C_j[j]['A'] = 0
            type_i  = max(C_j[j],key=C_j[j].get)
            if type_i == 'W':
                I_weakly.append(j)
            elif type_i == 'A':
                I_averagely.append(j)
            elif type_i == 'S':
                I_strongly.append(j)
        # print(('U_w={},U_a={},U_s={},U_v={}').format(U_w,U_a,U_s,U_v))
        # print(('contU={}').format(len(trainset)))
        # print(('用户总数={}').format(len(U_w)+len(U_a)+len(U_s)+len(U_v)))
        classify = [U_critical, U_average, U_benevolent, U_variable, I_weakly, I_averagely, I_strongly, I_variably]
        print('U_critical=%s,U_average=%s,U_benevolent=%s, U_variable=%s, I_weakly=%s, I_averagely=%s, I_strongly=%s, I_variably=%s' % 
        (len(U_critical), len(U_average), len(U_benevolent), len(U_variable), len(I_weakly), len(I_averagely), len(I_strongly), len(I_variably)))
        return classify


    def getThreeType(self,list_ratings,v,k):
        Strong = sum(i>=v for i in list_ratings)
        Weak = sum(i<k for i in list_ratings)
        Mean = sum((i<v and i >=k) for i in list_ratings)
        # print(Strong)
        # print(Weak)
        # print(Mean)
        max_type = max(Strong,Weak,Mean)
        if max_type == Strong :
            return 'Benevolent'
        elif max_type == Weak:
            return 'Critical'
        elif max_type == Mean:
            return 'Average'
    def threeClassifyUserAndItemByEachUserAndItem(self,trainset):
        classify = []
        U_critical = []  # 弱用户集合
        U_average = []  # 平均用户集合
        U_benevolent = []  # 强用户集合
        U_variable = []  # 可变用户集合
        I_weakly = []  # 弱项目集合
        I_averagely = []  # 平均项目集合
        I_strongly = []  # 强项目集合
        I_variably = []  # 可变项目集合
        columns = ['users', 'items', 'ratings']
        df_trainset = DataFrame(trainset, columns=columns)
        # print(df_trainset)
        usersTestIds = list(set(df_trainset['users']))
        v_k_user = defaultdict(list)
        v_k_item = defaultdict(list)
        #找出用户属于那种类型
        for user in usersTestIds:
            trainset_user = df_trainset[df_trainset['users'] == user]
            ratings_user = list(trainset_user['ratings'])

            v_user,k_user = self.getVAndK(ratings_user)
            v_k_user[user] = [v_user,k_user]
            type_user = self.getThreeType(ratings_user,v_user,k_user)
            # print(type_user)
            if type_user == 'Benevolent':
                U_benevolent.append(user)
            elif type_user == 'Critical':
                U_critical.append(user)
            elif type_user == 'Average':
                U_average.append(user)
        itemsTestIds = list(set(df_trainset['items']))
        for item in itemsTestIds:
            trainset_item = df_trainset[df_trainset['items'] == item]
            ratings_item = list(trainset_item['ratings'])
            v_item,k_item = self.getVAndK(ratings_item)
            v_k_item[item] = [v_item,k_item]
            # print(ratings_item)
            type_item = self.getThreeType(ratings_item,v_item,k_item)
            # print(type_user)
            if type_item == 'Benevolent':
                I_strongly.append(item)
            elif type_item == 'Critical':
                I_weakly.append(item)
            elif type_item == 'Average':
                I_averagely.append(item)
            else:
                I_variably.append(item)
        classify = [U_critical, U_average, U_benevolent, U_variable, I_weakly, I_averagely, I_strongly, I_variably]
        print('U_critical=%s,U_average=%s,U_benevolent=%s, U_variable=%s, I_weakly=%s, I_averagely=%s, I_strongly=%s, I_variably=%s' % 
        (len(U_critical), len(U_average), len(U_benevolent), len(U_variable), len(I_weakly), len(I_averagely), len(I_strongly), len(I_variably)))
        return classify,v_k_user,v_k_item

    def replaceNoiseByVAndK(self, noNoise, classify,v,k):
            U_critical, U_average, U_benevolent, U_variable, I_weakly, I_averagely, I_strongly, I_variably = classify[0], classify[
                1], classify[2], classify[3], classify[4], classify[5], classify[6], classify[7]
            number = 0
            N_or = noNoise.copy()
            # columns = ['users', 'items', 'ratings']
            # dataframe_predUpdate = DataFrame(noNoise, columns=columns)
            for line in N_or:
                (userId, itemId, rating) = line[0], line[1], line[2]
                if userId in U_critical and itemId in I_weakly and rating >= k:
                    noNoise.remove(line)
                    cRating = [userId , itemId, k]
                    noNoise.append(cRating)
                    number += 1
                elif userId in U_average and itemId in I_averagely and rating < k:
                    noNoise.remove(line)
                    cRating = [userId , itemId, self.getClosestNumber((v+k)/2)]
                    noNoise.append(cRating)
                    number += 1
                elif userId in U_average and itemId in I_averagely and rating >= v:
                    noNoise.remove(line)
                    cRating = [userId , itemId, self.getClosestNumber((v+k)/2)]
                    noNoise.append(cRating)
                    number += 1
                elif userId in U_benevolent and itemId in I_strongly and rating < v:
                    noNoise.remove(line)
                    cRating = [userId , itemId, v]
                    noNoise.append(cRating)
                    number += 1
            return noNoise, number
    
    

    
    
