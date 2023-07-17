'''
Author: [coolwen]
Date: 2021-12-21 16:42:21
LastEditors: CoolWen 122311139@qq.com
LastEditTime: 2022-10-30 09:29:30
Description: 
'''
from math import sqrt
from collections import defaultdict
import pandas as pd
from pandas import DataFrame

from ParameterConfiguration import threshold





class Myperformace:

    def getMAE_list(self,preT,testOrg):
        MAE = 0
        RMSE = 0
        rRmse = 0
        rSum = 0
        index = 0
        # for pretest in preT:
        #    if pretest[2] > threshold:
        #         print('test[0]= %s, test[1]=%s,  test[2]= %s' % (pretest[0], pretest[1], pretest[2])) 
        # df_testOrg = pd.DataFrame(testOrg, columns =['users', 'items', 'ratings'], dtype = float) 
        df_preT = pd.DataFrame(preT, columns =['users', 'items', 'ratings'], dtype = float) 
        # print(df_testOrg)
        # print(df_preT)
        #对每一个测试评分进行遍历
        for test in testOrg:
            # if test[2] > threshold:
            #     print('test[0]= %s, test[1]=%s,  test[2]= %s' % (test[0], test[1], test[2]))
            # print(one)
            # 如果预测集中有项目的评分，则预测个数index加1，并且计算mae和rmse
            if not df_preT[(df_preT['users']==test[0]) & (df_preT['items']== test[1])].empty:
                index += 1
                # print(df_preT[(df_preT['users']==test[0]) & (df_preT['items']== test[1])]['ratings'])
                preR =float (df_preT[(df_preT['users']==test[0]) & (df_preT['items']== test[1])]['ratings'])
                # print(df_preT[(df_preT['users']==one[0]) & (df_preT['items']== one[1])])
                # print(preR)
                rSum = rSum + abs(test[2]-preR)  # 累计预测评分误差      
                rRmse = rRmse+(test[2]-preR) ** 2
        print("共比较测试集和预测集:%s 个" % index)
        # print(index)
        if index !=0 :
            MAE = rSum / index
            RMSE = sqrt(rRmse/index)
        return MAE, RMSE

    '''
    把大于threshold的项目都推荐给用户，如果测试集中有推荐的项目，则n_rel加1
    '''
    def precision_recall_at_threshold(self,testSet, pred, threshold=4):
        columns = ['users', 'items', 'ratings']
        df_test = DataFrame(testSet, columns=columns)
        # print(df_test)
        df_pred = DataFrame(pred, columns=columns)
        # print(df_pred)
        usersTestIds = list(set(df_test['users']))
        usersPredIds = list(set(df_pred['users']))
        usersIntersection=list(set(usersTestIds).intersection(set(usersPredIds)))
        # 首先将预测值映射至每个用户
        user_est_true = defaultdict(list)
        for user in usersIntersection:
            test_items = list(set(df_test[(df_test['users']==user)]['items']))
            # print(test_items)
            pred_items = list(set(df_pred[(df_pred['users']==user)]['items']))
            # print(pred_items)
            for test_item in test_items:
                if test_item in pred_items:
                    # print(test_item)
                    testRating = float(df_test[(df_test['users']==user) & (df_test['items']== test_item)]['ratings'])
                    # print(df_pred[(df_pred['users']==user) & (df_pred['items']== test_item)]['ratings'])
                    preRating = float(df_pred[(df_pred['users']==user) & (df_pred['items']== test_item)]['ratings'])
                    user_est_true[user].append((preRating, testRating))
           
        precisions = dict()
        recalls = dict()
        f1 = dict()

        for uid, user_ratings in user_est_true.items():

            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items big than threshold
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings)

            # Number of relevant and recommended items big than threshold
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                for (est, true_r) in user_ratings)

            # Precision@K: Proportion of recommended items that are relevant
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

            # Recall@K: Proportion of relevant items that are recommended
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

            f1[uid] = 2* precisions[uid] * recalls[uid] /(precisions[uid] + recalls[uid]) if (precisions[uid] + recalls[uid]) != 0 else 1


        return precisions, recalls , f1

    def precision_recall_at_threshold_baseAll_user(self,testSet, pred, threshold=4):
        columns = ['users', 'items', 'ratings']
        df_test = DataFrame(testSet, columns=columns)
        # print(df_test)
        df_pred = DataFrame(pred, columns=columns)
        # print(df_pred)

        usersTestIds = list(set(df_test['users'])) #测试集中的用户
        usersPredIds = list(set(df_pred['users'])) #预测集的用户
        usersIntersection=list(set(usersTestIds).intersection(set(usersPredIds))) #测试集中和预测集中的共有用户
        user_pred = defaultdict(list)
        #在预测集中找出测试集和预测集都有用户且评分大于阈值的项目
        precisions = dict()
        recalls = dict()
        f1 = dict()
        count = 0
        for user in usersIntersection:
            
            #找出预测集中用户id为user，且评分大于阈值的评分集
            df_PredSatisfy = df_pred[(df_pred['users'] == user) & (df_pred['ratings'] > threshold)]
            #预测集中用户id为user，且评分大于阈值的项目的所有ID
            recommend_items = list(set(df_PredSatisfy['items']))
            #找出预测集中中用户id为user，且评分大于阈值的评分集
            df_testSatisfy = df_test[(df_test['users'] == user) & (df_test['ratings'] > threshold)]
            #在到测试集中，找到用户iD为user，且评分大于阈值的所有项目ID，test_items
            test_items = list(set(df_testSatisfy['items']))
            #找到用户iD为user，且在在测试集合用户集里面都有的项目itemsIntersection
            itemsIntersection = list(set(recommend_items).intersection(set(test_items)))
            # user_pred[user] = itemsIntersection
            # print(itemsIntersection)

            precisions[user] = len(itemsIntersection) / len(recommend_items) if len(recommend_items) != 0 else 1

            # Recall@K: Proportion of relevant items that are recommended
            recalls[user] = len(itemsIntersection) / len(test_items) if len(test_items)  != 0 else 1

            f1[user] = 2* precisions[user] * recalls[user] /(precisions[user] + recalls[user]) if (precisions[user] + recalls[user]) != 0 else 1
            if len(itemsIntersection) !=0:
                print('itemsIntersection= %s, recommend_items=%s,  test_items= %s' % (len(itemsIntersection), len(recommend_items), len(test_items)))
                print('precisions= %s, recalls=%s,  f1= %s' % (precisions[user], recalls[user], f1[user]))
                count += 1
        print("total recommend %s users" % (count))
        return precisions, recalls , f1

    def precision_recall_at_threshold_baseAll(self,testSet, pred, threshold=4):
        precisions, recalls, f1 = 0, 0, 0
        testSetBigthreshold = 0
        predSetBigthreshold = 0        
        testlessThanthreshold = 0
        predlessThanthreshold = 0
        for x in testSet:
            users, items, ratings = x
            if ratings >= threshold:
                testSetBigthreshold += 1
            else:
                testlessThanthreshold +=1
        for y in pred:
            users, items, ratings = y
            if ratings >= threshold:
                predSetBigthreshold += 1
            else:
                predlessThanthreshold +=1
        print('testSetBigthreshold={},testlessThanthreshold={},predSetBigthreshold={},predlessThanthreshold={}'.format(testSetBigthreshold, testlessThanthreshold,predSetBigthreshold,predlessThanthreshold))
        df_preT = pd.DataFrame(pred, columns =['users', 'items', 'ratings'], dtype = float)
        test_items = 0
        recommend_items = 0
        itemsIntersection = 0

        for test in testSet:
            if not df_preT[(df_preT['users']==test[0]) & (df_preT['items']== test[1])].empty:
                # print('pred have one in testSet')
                if test[2] >= threshold:
                    test_items += 1

                preR =float (df_preT[(df_preT['users']==test[0]) & (df_preT['items']== test[1])]['ratings'])
                if preR >= threshold:
                    recommend_items += 1
                if test[2] >= threshold and preR >= threshold:
                    itemsIntersection += 1
        precisions = itemsIntersection / recommend_items if recommend_items != 0 else 1
        recalls = itemsIntersection / test_items if test_items != 0 else 1
        f1 = 2* precisions * recalls /(precisions + recalls) if (precisions + recalls) != 0 else 1
        print('precisions= %s, recalls=%s,  f1= %s' % (precisions, recalls, f1))
        print('itemsIntersection= %s, recommend_items=%s,  test_items= %s' % (itemsIntersection, recommend_items, test_items))
        return precisions, recalls , f1

