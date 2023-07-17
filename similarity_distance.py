'''
Author: [coolwen]
Date: 2021-12-24 11:23:22
LastEditors: [coolwen]
LastEditTime: 2022-07-21 12:19:42
Description: 常用的相似度和距离
'''
from numpy import mean
import numpy as np
from math import sqrt
import numpy as np
from collections import Counter
from math import exp


class Similarity:
    # matrix1矩阵1和matrix2矩阵2，两个矩阵必须是相同的维数
    '''
    @Author: [coolwen]
    @description: 
    @param {*} self
    @param {*} dict_u1
    @param {*} dict_u2
    @return {*}
    '''
    def pearson(self, dict_u1, dict_u2):
        average_u1_rate = np.array(list(dict_u1.values())).mean()  # 获取用户u1对电影的平均评分
        average_u2_rate = np.array(list(dict_u2.values())).mean()  # 获取用户u2对电影的平均评分
        Common_keys = dict_u1.keys() & dict_u2.keys()
        part1 = 0  # 皮尔逊相关系数的分子部分
        part2 = 0  # 皮尔逊相关系数的分母的一部分
        part3 = 0  # 皮尔逊相关系数的分母的一部分
        for key in Common_keys:
            part1 += (dict_u1[key]-average_u1_rate) * \
                    (dict_u2[key]-average_u2_rate)*1.0
            part2 += pow(dict_u1[key]-average_u1_rate, 2)*1.0
            part3 += pow(dict_u2[key]-average_u2_rate, 2)*1.0

        part2 = sqrt(part2)
        part3 = sqrt(part3)
        if part2 == 0 or part3 == 0:  # 若分母为0，相似度为0
            userSim = 0
        else:
            userSim = part1 / (part2 * part3)   
        return abs(userSim)
    '''
    @Author: [coolwen]
    @description: 
    @param {*} self
    @param {*} dict_u1
    @param {*} dict_u2
    @return {*}
    '''
    def cos(self, dict_u1, dict_u2):
        #取字典key的交集
        Common_keys = dict_u1.keys() & dict_u2.keys()
        part1 = 0
        part2 = 0
        part3 = 0
        for key in Common_keys:
            part1 += dict_u1[key] * dict_u2[key]
            part2 += pow(dict_u1[key], 2)*1.0
            part3 += pow(dict_u2[key], 2)*1.0
        part2 = sqrt(part2)
        part3 = sqrt(part3)
        if part2 == 0 or part3 == 0:  # 若分母为0，相似度为0
            userSim = 0
        else:
            # 需要做归一化处理，因为余弦值的范围是 [-1,+1] ，相似度计算时一般需要把值归一化到 [0,1]，一般通过如下方式：
            #sim = 0.5 + 0.5 * cosθ
            userSim = (part1 / (part2 * part3))*0.5+0.5
        return abs(userSim)
    '''
    @Author: [coolwen]
    @description: 
    @param {*} self
    @param {*} dict_u1
    @param {*} dict_u2
    @return {*}
    '''
    def triangle(self, dict_u1, dict_u2):
        Common_keys = dict_u1.keys() & dict_u2.keys()
        part1 = 0
        part2 = 0
        part3 = 0
        for key in Common_keys:
            part1 += pow((dict_u1[key]-dict_u2[key]), 2) * 1.0
            part2 += pow(dict_u1[key], 2)*1.0
            part3 += pow(dict_u2[key], 2)*1.0
        part1 = sqrt(part1)
        part2 = sqrt(part2)
        part3 = sqrt(part3)
        if part2 == 0 or part3 == 0:  # 若分母为0，相似度为0
            userSim = 0
        else:
            # 需要做归一化处理，因为余弦值的范围是 [-1,+1] ，相似度计算时一般需要把值归一化到 [0,1]，一般通过如下方式：
            #sim = 0.5 + 0.5 * cosθ
            userSim = 1 - part1 / (part2*part3)
        return abs(userSim)
    '''
    @Author: [coolwen]
    @description: 
    @param {*} self
    @param {*} dict_u1
    @param {*} dict_u2
    @return {*}
    '''
    def EucledianDistance(self, dict_u1, dict_u2):
        Common_keys = dict_u1.keys() & dict_u2.keys()
        part1 = 0
        for key in Common_keys:
            part1 += pow(dict_u1[key]-dict_u2[key], 2)
        if part1 == 0:  # 若分母为0，相似度为0
            userSim = 1
        else:
            userSim = 1 / (sqrt(part1)+1)
        return abs(userSim)
    


    '''
    @Author: [coolwen]
    @description: 
    @param {*} self
    @param {*} dict_u1
    @param {*} dict_u2
    @return {*}
    '''
    def jaccard(self, dict_u1, dict_u2):
        Common_keys = dict_u1.keys() & dict_u2.keys()
        part1 = len(Common_keys)
        total_u1 = len(dict_u1)
        total_u2 = len(dict_u2)
        part2 = total_u1 + total_u2 - part1
        if part2 == 0:  # 若分母为0，相似度为0
            userSim = 0
        else:
            userSim = part1 / part2
            return abs(userSim)
    def Bhattacharya(self, dict_u1, dict_u2):
        u1Count= Counter(dict_u1.values())
        u2Count= Counter(dict_u2.values())
        u1_len = sum(i for i in u1Count.values())
        u2_len = sum(i for i in u2Count.values())
        Common_keys = u1Count.keys() & u2Count.keys()
        bc = 0
        for key in Common_keys:
            p1 = u1Count[key]/u1_len
            p2 = u2Count[key]/u2_len
            bc = bc + sqrt(p1*p2)
        return bc
    def getHtagAndW(self,dict_u):
        uCount= Counter(dict_u.values())
        hru = max(uCount.keys(),key = uCount.get)
        r_max = max(uCount.keys())
        r_min = min(uCount.keys())
        dict_w={}
        for u_rating in uCount.keys():
            part1 = abs(u_rating - hru)
            part2 = r_max-r_min+pow(10, -6)
            part3 = exp(-part1/part2)
            w = 1/(1+part3)
            dict_w.setdefault(u_rating, float(w))
        return dict_w
    
            
    def Rmds(self,dict_u1, dict_u2):
        Common_keys = dict_u1.keys() & dict_u2.keys()
        dict_w_u1 = self.getHtagAndW(dict_u1)
        dict_w_u2 = self.getHtagAndW(dict_u2)
        part2 = 0
        for key in Common_keys:
            part1 = dict_u1[key]*dict_w_u1[dict_u1[key]]-dict_u2[key]*dict_w_u2[dict_u2[key]]
            part2 += 1/exp(abs(part1))
        part3 = len(dict_u1.keys()) + len(dict_u2.keys())-len(Common_keys)
        userSim = part2/part3
        return userSim
    
  
# sm=    Similarity()
# u1 = []