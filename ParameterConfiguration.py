'''
Author: your name
Date: 2021-12-03 15:05:32
LastEditTime: 2023-04-23 10:33:02
LastEditors: CoolWen 122311139@qq.com
Description: 实验当中的参数配置，L ≥ α ≥ β ≥ L
FilePath: \python\矩阵分解中噪声的处理\三支的噪声处理\canshu.py
'''

# -----------m100K start----------------
# Kdata = 2  #\beta 
# Vdata = 4  #\alpha 
# delta = 1 #最小评分间隔
# threshold = 4 #计算精准度,召回率和F1值需要
# ratingsRange = [1,2,3,4,5]
# maxRating = 5
# minRating = 1
# -----------m100K  end----------------


# -----------flimtrust start----------------
# Kdata = 1.5
# Vdata = 3
# delta = 0.5 #最小评分间隔
# threshold = 3 #计算精准度,召回率和F1值需要
# ratingsRange = [0.5,1,1.5,2,2.5,3,3.5,4]
# maxRating = 4
# minRating = 0.5
# -----------flimtrust end----------------



# -----------moive tweetings start----------------
Kdata = 3 
Vdata = 7
delta = 1 #最小评分间隔
threshold = 7.5 #计算精准度,召回率和F1值需要
ratingsRange = [1,2,3,4,5,6,7,8,9,10]
maxRating = 10
minRating = 1
# -----------moive tweetings end----------------




Seed = 5  # 随机种子,种子相同保证随机的训练集和测试集相同
k_fold = 10 # k折实验数据交叉验证
Kneighbor = 60 #邻居数



lam = 0.015  # 正则化参数 \lambda
gam = 0.001 # 学习率 
step = 10  #训练次数   \epsilon 
rank = 5  # 单位向量的维数 \upsilon  



datapath = r'F:\科研\sets\ExperimentSet\\'
savepath = r'F:\科研\sets\result\最优参数\\'
testfilepath = r'F:\科研\sets\experimentSetBak\\1.txt'

