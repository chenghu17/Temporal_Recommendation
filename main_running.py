import load_data
import standard_BPR
import evolution
import Dynamic_BPR

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
    itemSet = load_data.itemSet(trainPath)
    t = 18  # month
    d = 5
    userNum = 1000
    itemNum = 1000
    step = 10000
    alpha = 0.02
    alpha_Reg = 0.02
    gama = 0.02
    dBPR = Dynamic_BPR.DBPR(trainPath, d, t, userNum, itemNum, itemSet, step, alpha, alpha_Reg, gama)
    userMat, itemMat = dBPR.Time_BPR()

