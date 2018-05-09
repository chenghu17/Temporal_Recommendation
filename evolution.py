import metric
import os

if __name__ == '__main__':
    rootPath = 'data_LastFM/'
    # rootPath = 'data_Epinions/'
    # rootPath = 'data_FineFoods/'
    # rootPath = 'data_Netflix/'
    # rootPath = 'data_MovieLen/'
    trainPath = rootPath + 'train.tsv'
    validationPath = rootPath + 'validation.tsv'
    testPath = rootPath + 'test.tsv'
    alpha = 0.02
    alpha_Reg = 0.02
    gama = 0.02
    resultPath = 'alpha_' + str(alpha) + '_alphaReg_' + str(alpha_Reg) + '_gama_' + str(gama)+'/'
    # resultPath = 'dpf/'
    rootPath = rootPath + resultPath
    Max = 100       # 给每个user预测Max个item，用于计算后面的评价标准
    K = 10          # precision@K，K可以取10，50，100，不要大于Max
    timestep = 9   # 每个时间间隔跨度，即main_running.py中的interval，取值为3，6，9
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
    # 训练出来的user和item特征向量，内乘即为user对item的偏好，对结果进行排序，得到ranking.tsv,再计算以下评价标准
    itemMat = 'itemMat4'
    userMat = 'userMat4'
    # 判断之前是否已经生成ranking(Max).tsv
    exists = os.path.exists(rootPath + 'evolution' + str(timestep) + '/ranking'+str(Max)+'.tsv')
    if not exists:
        # metric.ranking(rootPath, testPath, timestep, itemMat, userMat, Max)
        metric.ranking_sparse(rootPath, trainPath, validationPath, testPath, timestep, itemMat, userMat, m, Max)
    Precision = metric.precision(rootPath, testPath, timestep, K, Max)
    Recall = metric.reCall(rootPath, testPath, timestep, K, Max)
    MRR = metric.MRR(rootPath, testPath, timestep, K, Max)
    MAR = metric.MAR(rootPath, testPath, timestep, K, Max)
    # NDCG = metric.NDCG(rootPath, testPath, timestep, K, Max)
    NDCG = metric.NDCG_Full(rootPath, testPath, timestep, K, Max)
    file = open(rootPath + 'evolution' + str(timestep) + '/metric.txt', 'a')
    file.write('Precision@' + str(K) + ': ' + str(Precision) + '\n')
    file.write('Recall@' + str(K) + ': ' + str(Recall) + '\n')
    file.write('MRR: ' + str(MRR) + '\n')
    file.write('MAR: ' + str(MAR) + '\n')
    file.write('NDCG: ' + str(NDCG) + '\n')
    file.write('\n')
