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
    # d : dimensions of latent factor
    # N : negative sampling number
    # steps : iteration number
    # alpha : the learning rate
    # alpha : the regularization parameter
    def __init__(self, ratingMat, timeMat, userNum, itemNum, d, N, step=10000, alpha=0.02, gama=0.02):
        self.ratingMat = ratingMat
        self.timeMat = timeMat
        self.userMat = np.random.random((userNum, d))
        self.itemMat = np.random.random((itemNum, d))
        self.N = N
        self.step = step
        self.alpha = alpha
        self.gama = gama

    def calc_Loss(self, ratingMat, userMat, itemMat, gama):
        e = 0
        for user_id in range(len(ratingMat)):
            for item_id in np.where(ratingMat[user_id] == 1)[0]:
                Pu = userMat[user_id]
                Qi = itemMat[item_id]
                for nag_item_id in np.where(ratingMat[user_id] == 0)[0]:
                    Qj = itemMat[nag_item_id]
                    regResult = np.dot(Pu, Pu) + np.dot(Qi, Qi) + np.dot(Qj, Qj)
                    e += gama * regResult
                    eij = np.dot(Pu, Qi) - np.dot(Pu, Qj)
                    logisticResult = logistic.cdf(eij)
                    e -= np.log(logisticResult)
        print(e)
        return e

    def standard_BPR(self):
        ratingMat = self.ratingMat
        userMat = self.userMat
        itemMat = self.itemMat
        alpha = self.alpha
        gama = self.gama
        N = self.N
        for step in range(self.step):
            for user_id in range(len(ratingMat)):
                # user don't rate
                nagetiveList = list(np.where(ratingMat[user_id] == 0)[0])
                for item_id in np.where(ratingMat[user_id] == 1)[0]:
                    Pu = userMat[user_id]
                    Qi = itemMat[item_id]
                    # nag_item_id = np.random.choice(nagetiveList)
                    nage_Sampling = random.sample(nagetiveList, N)
                    for nag_item_id in nage_Sampling:
                        Qj = itemMat[nag_item_id]
                        eij = np.dot(Pu, Qi) - np.dot(Pu, Qj)
                        logisticResult = logistic.cdf(-eij)
                        # calculate every gradient
                        gradient_pu = logisticResult * (Qj - Qi) + gama * Pu
                        gradient_qi = logisticResult * (-Pu) + gama * Qi
                        gradient_qj = logisticResult * (Pu) + gama * Qj
                        # update every vector
                        userMat[user_id] = Pu - alpha * gradient_pu
                        itemMat[item_id] = Qi - alpha * gradient_qi
                        itemMat[nag_item_id] = Qj - alpha * gradient_qj

            e = self.calc_Loss(ratingMat, userMat, itemMat, gama)
            if e < 0.01:
                break
        return userMat, itemMat



