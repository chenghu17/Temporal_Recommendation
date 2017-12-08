# ratingMat : userId,movieId,rating,timestamp
# create rating-matrix and time-matrix
# ratingMat : user-item rating or  {0,1}
# timeMat : user-item timestamp, {1,2,3,4,5...}
import numpy as  np


def trainingData():
    R = [
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    ]
    R = np.array(R)
    userNum = R.shape[0]
    itemNum = R.shape[1]

    T = np.random.random((userNum, itemNum))
    T = np.array(T)

    return R, T, userNum, itemNum
#
if __name__=='__main__':
    a = 3
    print(-a)