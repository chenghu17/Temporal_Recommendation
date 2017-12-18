import load_data

import Dynamic_BPR_standard

if __name__ == '__main__':
    # standard_BPR test
    # item_dict, itemNum = load_data.itemDict('dataset/movies.csv')
    # userNum, user_dict = load_data.splitData('dataset/ratings.csv', 'dataset/train.csv', 'dataset/test.csv')
    # ratingMat, timeMat = load_data.trainingData('dataset/train.csv', item_dict, user_dict, userNum, itemNum)
    # testMat = load_data.testingData('dataset/test.csv', item_dict, user_dict, userNum, itemNum)
    # d = 5
    # N = 1
    # t = 1
    # bpr = standard_BPR.BPR(ratingMat, timeMat, userNum, itemNum, testMat, d, N, t)
    # userMat, itemMat = bpr.standard_BPR()

    trainPath = 'data/train.tsv'
    validationPath = 'data/validation.tsv'
    itemSet = load_data.itemSet(trainPath)
    t = 18  # month
    d = 5
    userNum = 1000
    itemNum = 1000
    step = 500
    alpha = 0.02
    alpha_Reg = 0.02
    gama = 0.02
    K = 50  # recall number
    dBPR = Dynamic_BPR_standard.DBPR(trainPath, validationPath, d, t, userNum, itemNum, itemSet, step, alpha, alpha_Reg, gama)
    userMat, itemMat = dBPR.Time_BPR()
    # evolution.reCall(validationPath, userMat, itemMat, K)
