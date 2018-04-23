import load_data
import Dynamic_BPR_standard

if __name__ == '__main__':
    # standard_BPR test
    # item_dict, itemNum = load_data.itemDict('data_MovieLen/movies.csv')
    # userNum, user_dict = load_data.splitData('data_MovieLen/ratings.csv', 'data_MovieLen/train.csv', 'data_MovieLen/test.csv')
    # ratingMat, timeMat = load_data.trainingData('data_MovieLen/train.csv', item_dict, user_dict, userNum, itemNum)
    # testMat = load_data.testingData('data_MovieLen/test.csv', item_dict, user_dict, userNum, itemNum)
    # d = 5
    # N = 1
    # t = 1
    # bpr = standard_BPR.BPR(ratingMat, timeMat, userNum, itemNum, testMat, d, N, t)
    # userMat, itemMat = bpr.standard_BPR()

    # rootPath = 'data_LastFM'
    # trainPath = rootPath + '/train.csv'
    # validationPath = rootPath + '/validation.csv'
    trainPath = 'data_Netflix/train.tsv'
    validationPath = 'data_Netflix/validation.tsv'
    userNum = 23928
    itemNum = 17771
    # userPath = rootPath + '/user.txt'
    # itemPath = rootPath + '/item.txt'
    # df_user = pd.read_csv(userPath, header=None)
    # df_item = pd.read_csv(itemPath, header=None)
    # userNum = len(df_user)+1
    # itemNum = len(df_item)+1

    itemSet = load_data.itemSet(trainPath)
    interval = 6  # month
    d = 5
    step = 500
    alpha = 0.02
    alpha_Reg = 0.02
    gama = 0.02
    K = 50  # recall number
    dBPR = Dynamic_BPR_standard.DBPR(trainPath, validationPath, d, interval, userNum, itemNum, itemSet, step, alpha, alpha_Reg, gama)
    userMat, itemMat = dBPR.Time_BPR()
    # evolution.reCall(validationPath, userMat, itemMat, K)

