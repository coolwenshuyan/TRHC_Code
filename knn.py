'''
Author: [coolwen]
Date: 2021-12-23 10:56:00
LastEditors: CoolWen 122311139@qq.com
LastEditTime: 2023-01-28 20:36:19
Description: 
'''
from math import sqrt
from ParameterConfiguration import maxRating
from ParameterConfiguration import minRating

import numpy as np
class KNN:
    def __init__(self,trainset):
        trainsetDict = {}
        movieUser = {}
        u2u = {}
        for line in trainset:
            userId, itemId, rating = line 
            trainsetDict.setdefault(userId,{})
            trainsetDict[userId].setdefault(itemId,float(rating))
            movieUser.setdefault(itemId,[])
            movieUser[itemId].append(userId)
        for m in movieUser.keys():
            for u in movieUser[m]:
                u2u.setdefault(u,{})
                for n in movieUser[m]:
                    if u!=n:
                        u2u[u].setdefault(n,[])
                        u2u[u][n].append(m)
        self.u2u = u2u
        self.dictTrainset = trainsetDict
        self.dictmovieUser = movieUser
    # def getClosestNumber(self,originalNumber):
    #     listNumbers = ratingsRange
    #     return min(listNumbers, key=lambda x: abs(x - originalNumber))
    def predictOne(self, uid, iid, K, userSim):
        userSimSum = 0
        #判断训练集组成用户是否有测试集中用户的信息，没有保持则不预测。对于没有相似用户不预测。
        if uid in userSim:
            average_uid_rate = np.array(list(self.dictTrainset[uid].values())).mean()  # 获取用户u1对电影的平均评分
            uidSim_uid = userSim[uid]
            simUser = sorted(uidSim_uid.items(),key = lambda uidSim_uid : uidSim_uid[1],reverse= True)[0:K]
            # interacted_items = self.dictTrainset[uid].keys()  # 获取该用户评过分的电影
            pre = 0
            for n, sim in simUser:# uid的邻居用户
                average_n_rate = np.array(list(self.dictTrainset[n].values())).mean()  # 获取用户u1对电影的平均评分
                userSimSum += sim  # 对该用户近邻用户相似度求和
                if iid in self.dictTrainset[n]:
                    nrating = self.dictTrainset[n][iid]
                    pre += sim * (nrating - average_n_rate)
        if userSimSum != 0:
            rating = average_uid_rate + pre / userSimSum
            if rating > maxRating:
                rating = maxRating
            if rating < minRating:
                rating = minRating
            return rating
    

    def predict(self, test_data, userSim, K):
        preT = []
        for line in test_data:
            # print(line)
            uid, iid, rating = line
            eui = self.predictOne(uid, iid, K, userSim)
            if eui == None:#没有紧邻或者强一致中没有中一致用户
                continue
            
            preT.append([uid, iid, eui])
        return preT 
    def train(self, sim):
      pass
    '''
    @Author: [coolwen]
    @description: 获取用户之间的相似度
    @param {*} Sim 相似度类型，比如cos，perason等
    @return {*}
    '''
    def getUserSim(self, sim):
        userSim = {}
        # print(self.u2u)
        for u in self.u2u.keys(): #对每个用户u
            # if u == 102:
                # print(self.u2u[u])
            userSim.setdefault(u,{})  #将用户u加入userSim中设为key，该用户对应一个字典
            for n in self.u2u[u].keys():
                  #对与用户u相关的每个用户n             
                userSim[u][n] = sim(self.dictTrainset[u], self.dictTrainset[n])
        # print(userSim)       
        return userSim 


    def getItemSim(self, sim):
        itemSim = {}
        # print(self.u2u)
        
