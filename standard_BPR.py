# author : hucheng
# Standard Bayesian Personalized Ranking


import numpy as np
import random
from scipy.stats import logistic
import evolution


class BPR():

    # timeMat : user-item timestamp, {1,2,3,4,5...}
    # ratingMat : user-item rating or  {0,1}
    # userNum、itemNum : user/item number
    # testMat : testing data, user-item {0,1}
    # Y_True : label of testing data, {1}
    # d : dimensions of latent factor
    # N : negative sampling number
    # steps : iteration number
    # alpha : the learning rate
    # alpha : the regularization parameter
    # def __init__(self, ratingMat, timeMat, userNum, itemNum, testMat, Y_True, d, N, step=10000, alpha=0.02, gama=0.02):
    def __init__(self, ratingMat, timeMat, userNum, itemNum, testMat, d, N, step=10000, alpha=0.02, gama=0.02):
        self.ratingMat = ratingMat
        self.timeMat = timeMat
        self.userMat = np.random.random((userNum, d))
        self.itemMat = np.random.random((itemNum, d))
        self.testMat = testMat
        # self.Y_True = Y_True
        self.N = N
        self.step = step
        self.alpha = alpha
        self.gama = gama

    def calc_Loss(self, ratingMat, userMat, itemMat, gama, N):
        e = 0
        for user_id in range(len(ratingMat)):
            # print('calcu-user_id', user_id)
            negativeList = list(np.where(ratingMat[user_id] == 0)[0])
            for item_id in np.where(ratingMat[user_id] == 1)[0]:
                Pu = userMat[user_id]
                Qi = itemMat[item_id]
                nega_Sampling = random.sample(negativeList, N)
                for nega_item_id in nega_Sampling:
                    Qj = itemMat[nega_item_id]
                    regResult = np.dot(Pu, Pu) + np.dot(Qi, Qi) + np.dot(Qj, Qj)
                    e += gama * regResult
                    eij = np.dot(Pu, Qi) - np.dot(Pu, Qj)
                    logisticResult = logistic.cdf(eij)
                    e -= np.log(logisticResult)
        print(e)
        return e

    def prediction(self, userMat, itemMat):
        pred = list()
        true = list()
        testMat = self.testMat
        for user_id in range(len(testMat)):
            # positive
            for item_id in np.where(testMat[user_id] == 1)[0]:
                true.append(1)
                Pu = userMat[user_id]
                Qi = itemMat[item_id]
                Y = np.dot(Pu, Qi)
                pred.append(Y)
            # negative
            for nega_item_id in np.where(testMat[user_id] == 0)[0]:
                true.append(0)
                Qj = itemMat[nega_item_id]
                Y = np.dot(Pu, Qj)
                pred.append(Y)
        Y_True = np.array(true)
        Y_Pred = np.array(pred)
        return Y_True, Y_Pred

    def standard_BPR(self):
        ratingMat = self.ratingMat
        userMat = self.userMat
        itemMat = self.itemMat
        alpha = self.alpha
        gama = self.gama
        N = self.N
        for step in range(self.step):
            for user_id in range(len(ratingMat)):
                # print('userId:', user_id)
                # user don't rate
                negativeList = list(np.where(ratingMat[user_id] == 0)[0])
                for item_id in np.where(ratingMat[user_id] == 1)[0]:
                    Pu = userMat[user_id]
                    Qi = itemMat[item_id]
                    # nag_item_id = np.random.choice(nagetiveList)
                    nega_Sampling = random.sample(negativeList, N)
                    for nega_item_id in nega_Sampling:
                        Qj = itemMat[nega_item_id]
                        eij = np.dot(Pu, Qi) - np.dot(Pu, Qj)
                        logisticResult = logistic.cdf(-eij)
                        # calculate every gradient
                        gradient_pu = logisticResult * (Qj - Qi) + gama * Pu
                        gradient_qi = logisticResult * (-Pu) + gama * Qi
                        gradient_qj = logisticResult * (Pu) + gama * Qj
                        # update every vector
                        userMat[user_id] = Pu - alpha * gradient_pu
                        itemMat[item_id] = Qi - alpha * gradient_qi
                        itemMat[nega_item_id] = Qj - alpha * gradient_qj

            Y_True, Y_Pred = self.prediction(userMat, itemMat)
            auc = evolution.AUC(Y_True, Y_Pred)
            print('AUC:', auc)

            # e = self.calc_Loss(ratingMat, userMat, itemMat, gama, N)
            # if e < 0.01:
            #     break
        return userMat, itemMat
