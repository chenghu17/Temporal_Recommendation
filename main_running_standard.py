import load_data
import Dynamic_BPR_standard

if __name__ == '__main__':

    rootPath = 'data_LastFM/'
    # rootPath = 'data_Epinions/'
    # rootPath = 'data_FineFoods/'
    # rootPath = 'data_Netflix/'
    # rootPath = 'data_MovieLen/'
    trainPath = rootPath + 'train.tsv'
    validationPath = rootPath + 'validation.tsv'

    d = 10
    step = 200
    alpha = 0.02
    alpha_Reg = 0.1
    gama = 0.02
    resultPath = 'alpha_' + str(alpha) + '_alphaReg_' + str(alpha_Reg) + '_gama_' + str(gama) + '/'
    rootPath = rootPath + resultPath
    # for lastfm
    n = 1000
    m = 1000
    # for epinions
    # n = 1461
    # m = 17765
    # for finefoods
    # n = 1892
    # m = 19489
    # for netflix
    # n = 23928
    # m = 17771
    # for movielen
    # n = 10702
    # m = 26231

    itemSet = load_data.itemSet(trainPath)
    dBPR = Dynamic_BPR_standard.DBPR(rootPath, trainPath, validationPath, d, n, m, itemSet, step, alpha,
                                     alpha_Reg, gama)
    userMat, itemMat = dBPR.Time_BPR()
