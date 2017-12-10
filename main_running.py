import load_data
import standard_BPR

if __name__ == '__main__':

    item_dict, itemNum = load_data.itemDict('dataset/movies.csv')

    userNum,user_dict = load_data.splitData('dataset/ratings.csv', 'dataset/train.csv', 'dataset/test.csv')

    ratingMat, timeMat = load_data.trainingData('dataset/train.csv', item_dict, user_dict, userNum, itemNum)

    test_data = load_data.test_Data('dataset/test.csv', item_dict)


    d = 5
    N = 1
    bpr = standard_BPR.BPR(ratingMat, timeMat, userNum, itemNum, d, N)
    bpr.standard_BPR()
    userMat, itemMat = bpr.standard_BPR()
