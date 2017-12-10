import load_data
import standard_BPR
import evolution

if __name__ == '__main__':
    item_dict, itemNum = load_data.itemDict('dataset/movies.csv')

    userNum, user_dict = load_data.splitData('dataset/ratings.csv', 'dataset/train.csv', 'dataset/test.csv')

    ratingMat, timeMat = load_data.trainingData('dataset/train.csv', item_dict, user_dict, userNum, itemNum)

    # testMat, Y_True = load_data.testingData('dataset/test.csv', item_dict, user_dict, userNum, itemNum)
    testMat = load_data.testingData('dataset/test.csv', item_dict, user_dict, userNum, itemNum)

    d = 5
    N = 1
    # bpr = standard_BPR.BPR(ratingMat, timeMat, userNum, itemNum, testMat, Y_True, d, N)
    bpr = standard_BPR.BPR(ratingMat, timeMat, userNum, itemNum, testMat, d, N)
    userMat, itemMat = bpr.standard_BPR()

    # evolution.AUC(y_true, )
