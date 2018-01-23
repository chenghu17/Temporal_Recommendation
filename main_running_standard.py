import load_data
import Dynamic_BPR_standard
import pandas as pd

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

    rootPath = 'data_Epinions'
    trainPath = rootPath + '/train.csv'
    validationPath = rootPath + '/validation.csv'
    # userNum = 1000
    # itemNum = 1000
    userPath = rootPath + '/user.txt'
    itemPath = rootPath + '/item.txt'
    df_user = pd.read_csv(userPath, header=None)
    df_item = pd.read_csv(itemPath, header=None)
    userNum = len(df_user) + 1
    itemNum = len(df_item) + 1

    itemSet = load_data.itemSet(trainPath)
    t = 18  # month
    d = 10

    step = 500
    alpha = 0.02
    alpha_Reg = 0.02
    gama = 0.02
    K = 50  # recall number
    dBPR = Dynamic_BPR_standard.DBPR(rootPath, trainPath, validationPath, d, t, userNum, itemNum, itemSet, step, alpha, alpha_Reg, gama)
    userMat, itemMat = dBPR.Time_BPR()
    # evolution.reCall(validationPath, userMat, itemMat, K)
