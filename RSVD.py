import numpy as np
import random
import datetime
from ParameterConfiguration import maxRating
from ParameterConfiguration import minRating

class SVD:
    def __init__(self, listtrain, K=20):
        self.mat = np.array(listtrain)
        self.K = K
        self.bi = {}
        self.bu = {}
        self.qi = {}
        self.pu = {}
        self.avg = np.mean(self.mat[:, 2])
        # print('self.avg',self.avg)
        # self.mat.shape[0]，矩阵多少行数据，即每次取[1,1,5],下次取[1,2,3]
        # print(self.mat)
        for i in range(self.mat.shape[0]):
            # print('self.mat.shape[0]',self.mat.shape[0])
            uid = self.mat[i, 0]  # 遍历每一行，依次取用户u,对于[1,1,5]取第一列1的值
            # print('uid=',uid)
            iid = self.mat[i, 1]  # 遍历每一行，依次取电影编号i,对于[1,1,5]取第二列1的值
            # print('iid=',iid)
            # bi = {1，0},注意是字典，如果有4部电影，self.bi = {1: 0, 2: 0, 7: 0, 6: 0}
            self.bi.setdefault(iid, 0)
            # print('self.bi', self.bi)
            # bi = {1，0}，注意是字典，如果有3个用户，self.bu = {1: 0, 2: 0, 3: 0}
            self.bu.setdefault(uid, 0)
            # print('self.bu', self.bu)
            # 一下为初始化p和q列矩阵
            self.qi.setdefault(iid, np.random.random(
                (self.K, 1))/10*np.sqrt(self.K))
            self.pu.setdefault(uid, np.random.random(
                (self.K, 1))/10*np.sqrt(self.K))
            '''
            train_data  [[1, 1, 5], [1, 2, 3], [2, 1, 3],
                [2, 7, 4], [3, 6, 3], [3, 7, 3]]
            4部电影，1，2，6，7，三个用户1，2，3
            self.qi {1: array([[0.0420583 ],
                            [0.15804507],
                            [0.1128515 ],
                            [0.08062554],
                            [0.17615198]]), 2: array([[0.22247132],
                            [0.09862479],
                            [0.11265481],
                            [0.01601157],
                            [0.20315109]]), 7: array([[0.2117927 ],
                            [0.0450563 ],
                            [0.00379445],
                            [0.00454227],
                            [0.11397565]]), 6: array([[0.06900561],
                            [0.10197499],
                            [0.03008107],
                            [0.02684075],
                            [0.03150337]])}
        self.pu {1: array([[0.15732416],
                        [0.21815853],
                        [0.20076907],
                        [0.13377402],
                        [0.10340826]]), 2: array([[0.20570345],
                        [0.22336734],
                        [0.00802503],
                        [0.12680029],
                        [0.06570973]]), 3: array([[0.13595192],
                        [0.21323853],
                        [0.04195388],
                        [0.11070023],
                        [0.17813547]])}
            '''
            # print('self.qi', self.qi)

            # print('self.pu', self.pu)

    def predictOne(self, uid, iid):  # 预测评分的函数
        # setdefault的作用是当该用户或者物品未出现过时，新建它的bi,bu,qi,pu，并设置初始值为0
        self.bi.setdefault(iid, 0)
        self.bu.setdefault(uid, 0)
        self.qi.setdefault(iid, np.zeros((self.K, 1)))
        self.pu.setdefault(uid, np.zeros((self.K, 1)))
        '''
         np.zeros((self.K, 1))中K的值如果为2，会得到array([[ 0.],
       [ 0.]])，self.qi = {1 ：array([[ 0.],
       [ 0.]])}
        '''
        # print('uid', uid)
        # print('iid', iid)
        # print('self.bi', self.bi)
        # print('self.bu', self.bu)
        # print('self.qi', self.qi)
        # print('self.pu', self.pu)
        rating = self.avg+self.bi[iid]+self.bu[uid] + \
            np.sum(self.qi[iid]*self.pu[uid])  # 预测评分公式
        # 由于评分范围在1到5，所以当分数大于5或小于1时，返回5,1.
        if rating > maxRating:
            rating = maxRating
        if rating < minRating:
            rating = minRating
        return rating
    def predict(self, test_data):  
        # print(test_data)
        test_data = np.array(test_data)
        print('test data size', test_data.shape)
        test_priect = []
        for i in range(test_data.shape[0]):
            uid = test_data[i, 0]
            iid = test_data[i, 1]
            rating = test_data[i, 2]
            eui = self.predictOne(uid, iid)
            test_priect.append([uid,iid,eui])
        return test_priect
    # 训练函数，step为迭代次数。# λ是Lambda，正则化参数，gamma是a学习率,bu=bu+a(eui+ λ*bu)
    def train(self, steps=3, gamma=0.01, Lambda=0.015):
        # print('train data size', self.mat.shape)
        # print('self.mat', self.mat)
        Rmse = []
        Mae = []
        print('K = %s' % self.K)
        for step in range(steps):
            print('step', step+1, 'is running')
            KK = np.random.permutation(
                self.mat.shape[0])  # 随机梯度下降算法，kk为对矩阵进行随机洗牌,每次会形成不同的KK值 KK = [1 3 0 4 5 2]
            # print('KK',KK)
            rmse = 0.0
            mae = 0.0
            for i in range(self.mat.shape[0]):
                j = KK[i]
                uid = self.mat[j, 0]
                iid = self.mat[j, 1]
                rating = self.mat[j, 2]
                eui = rating-self.predictOne(uid, iid)
                rmse += eui**2
                mae += abs(eui)
                # print(self.bu)
                # print(self.bi)
                self.bu[uid] += gamma*(eui-Lambda*self.bu[uid])
                self.bi[iid] += gamma*(eui-Lambda*self.bi[iid])
                tmp = self.qi[iid]
                self.qi[iid] += gamma*(eui*self.pu[uid]-Lambda*self.qi[iid])
                self.pu[uid] += gamma*(eui*tmp-Lambda*self.pu[uid])
            gamma = 0.93*gamma
            # print('rmse is', np.sqrt(rmse/self.mat.shape[0]))
            # print('mae is', mae/self.mat.shape[0])
        #     Err = self.test(self.test_data)
        #     Mae.append(Err[0])
        #     Rmse.append(Err[1])
        # day = datetime.datetime.now()
        # day=day.strftime("%Y-%m-%d-%H-%M-%S")
        # filepathMae = 'F:\\科研\\带发表\\sim\\data\\predP\\'+day+'Mae_person_RSVD.txt'
        # with open(filepathMae, 'w') as f:
        #     f.write(str(Mae))
        # filepathRmse = 'F:\\科研\\带发表\\sim\\data\\predP\\'+day+'Rmse_person_RSVD.txt'
        # with open(filepathRmse, 'w') as f:
        #     f.write(str(Rmse))


    def test(self, test_data):  # gamma以0.93的学习率递减
        # print(test_data)
        test_data = np.array(test_data)
        print('test data size', test_data.shape)
        rmse = 0.0
        mae = 0.0
        for i in range(test_data.shape[0]):
            uid = test_data[i, 0]
            iid = test_data[i, 1]
            rating = test_data[i, 2]
            eui = rating-self.predictOne(uid, iid)
            rmse += eui**2
            mae += abs(eui)
        # print('rmse of test data is', np.sqrt(rmse/test_data.shape[0]))
        # print('mae of test data is', mae/test_data.shape[0])
        return mae/test_data.shape[0], np.sqrt(rmse/test_data.shape[0])

# def getData():  # 获取训练集和测试集的函数
#     import re
#     # TrainFile = 'F:\\科研\\sets\\ml-100k\\ml-100k\\u4.base'
#     # TestFile = 'F:\\科研\\sets\\ml-100k\\ml-100k\\u4.test'
#     TrainFile = 'F:\\科研\\sets\\filmtrust_dataset\\ratings.txt'  # 指定训练集
#     # TrainFile = 'F:\\科研\\sets\\filmtrust_dataset\\r.txt'  # 指定训练集
#     # TrainFile = 'F:\\科研\\sets\\myu1.base'  # 指定训练集
#     # TestFile = 'F:\\科研\\sets\\myu1.test'  # 指定测试集
#     # 获取训练数据
#     f = open(TrainFile, 'r')
#     lines = f.readlines()
#     f.close()
#     data = []
#     for line in lines:
#         list = re.split(' |\n', line)
#         if eval(list[2]) != 0:  # 踢出评分0的数据，这部分是用户评论了但是没有评分的
#             data.append([eval(i) for i in list[:3]])
#     random.shuffle(data)
#     train_data = data[:int(len(data)*8/10)]
#     test_data = data[int(len(data)*8/10):]

# # 获取测试数据
#     f = open(TestFile, 'r')
#     lines = f.readlines()
#     f.close()
#     test_data = []
#     for line in lines:
#         list = re.split('\t|\n', line)
#         if int(list[2]) != 0:  # 提出评分0的数据，这部分是用户评论了但是没有评分的
#             test_data.append([int(i) for i in list[:3]])
#     random.shuffle(test_data)
   

#     print('load data finished')
#     # print('train_data ', len(train_data))
#     # print('test_data ', len(test_data))
#     # print('train_data ', train_data)
#     # print('test_data ', test_data)
    # return train_data, test_data

#     '''
#     train_data = [[1, 1, 5], [1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 2], [2, 1, 3]],train_data.shape[0] = 6
#     test_data = [[1, 1, 5], [1, 2, 3], [1, 3, 4], [2, 2, 5], [2, 3, 4], [2, 4, 3], [3, 1, 3], [3, 3, 3]] train_data.shape[0] = 8
#     '''

# train_data, test_data = getData()
# # print(train_data)
# a = SVD(train_data, 10)  # 秩为5的列向量或者行向量，根据矩阵大小调整
# a.train()
# a.test(test_data)
# print(a.predict(1,7))
# # print(testSet['1']['7'])
