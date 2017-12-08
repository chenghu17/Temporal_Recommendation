
import load_data
import standard_BPR


if __name__ == '__main__':

    item_dict, itemNum= load_data.itemDict('dataset/movies.csv')

    load_data.splitData('dataset/ratings.csv')

    ratingMat, timeMat = load_data.trainingData('dataset/train.csv', item_dict)

    test_data = load_data.test_Data('dataset/test.csv', item_dict)

    # traindata = [
    #     [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    #     [0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    # ]
    # traindata = np.array(traindata)
    # timeMat = []
    # userNum = traindata.shape[0]
    # itemNum = traindata.shape[1]

    d = 10
    N = 2
    bpr = standard_BPR.BPR(ratingMat, timeMat, userNum, itemNum, d, N)
    bpr.standard_BPR()
    userMat, itemMat = bpr.standard_BPR()