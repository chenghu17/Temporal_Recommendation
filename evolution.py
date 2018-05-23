import metric
import os

if __name__ == '__main__':
    # rootPath = 'data_LastFM/'
    # rootPath = 'data_Epinions/'
    rootPath = 'data_FineFoods/'
    # rootPath = 'data_Netflix/'
    # rootPath = 'data_MovieLen/'
    trainPath = rootPath + 'train.tsv'
    validationPath = rootPath + 'validation.tsv'
    testPath = rootPath + 'test.tsv'
    alpha = 0.02
    alpha_Reg = 0.1
    gama = 0.1
    # gama = 0.0002
    resultPath = 'alpha_' + str(alpha) + '_alphaReg_' + str(alpha_Reg) + '_gama_' + str(gama)+'/'
    # resultPath = 'dpf/'
    state = 'dynamic/'
    # state = 'static/'
    rootPath = rootPath + state + resultPath
    Max = 100
    K = 50
    timestep = 0  # interval: 0，3，6，9
    # for lastfm
    # n = 1000
    # m = 1000
    # for epinions
    # n = 8201
    # m = 19004
    # for finefoods
    n = 1892
    m = 19489
    # for netflix
    # n = 23928
    # m = 17771
    # for movielen
    # n = 10702
    # m = 26231
    #
    itemMat = 'itemMat82'
    userMat = 'userMat82'
    # if exist ranking(Max).tsv
    exists = os.path.exists(rootPath + 'evolution' + str(timestep) + '/ranking' + str(Max) + '.tsv')
    if not exists:
        # metric.ranking(rootPath, testPath, timestep, itemMat, userMat, Max)
        metric.ranking_sparse(rootPath, trainPath, validationPath, testPath, timestep, itemMat, userMat, m, Max)
    Precision = metric.precision(rootPath, testPath, timestep, K, Max)
    Recall = metric.reCall(rootPath, testPath, timestep, K, Max)
    MRR = metric.MRR(rootPath, testPath, timestep, K, Max)
    MAR = metric.MAR(rootPath, testPath, timestep, K, Max)
    NDCG = metric.NDCG(rootPath, testPath, timestep, K, Max)
    NDCG_Full = metric.NDCG_Full(rootPath, testPath, timestep, K, Max)
    file = open(rootPath + 'evolution' + str(timestep) + '/metric.txt', 'a')
    file.write('Precision@' + str(K) + ': ' + str(Precision) + '\n')
    file.write('Recall@' + str(K) + ': ' + str(Recall) + '\n')
    file.write('MRR@' + str(K) + ': ' + str(MRR) + '\n')
    file.write('MAR@' + str(K) + ': ' + str(MAR) + '\n')
    file.write('NDCG@' + str(K) + ': ' + str(NDCG) + '\n')
    file.write('NDCG_Full@: ' + str(NDCG_Full) + '\n')
    file.write('\n')
