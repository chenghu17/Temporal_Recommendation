import pandas as pd
import numpy as np
import logging
import random
import os


class generate_data():
    def __init__(self, datatype, interval, userNum, itemNum):
        self.trainPath = os.path.join(datatype, 'train.tsv')
        self.validationPath = os.path.join(datatype, 'validation.tsv')
        self.testPath = os.path.join(datatype, 'test.tsv')
        self.userNum = userNum
        self.itemNum = itemNum
        self.timestamp = interval * 31 * 24 * 3600

        self.interval_num = 0

        self.data_train_user_key = [list() for _ in range(userNum)]  # 每个用户，在每个时间段，打分item_id集合，
        self.data_train_item_key = [list() for _ in range(itemNum)]  # 每个item，在每个时间段，被哪些user打过分

        self.data_validation_userSet = []  # 存储validation中的user
        self.data_validation = [list() for _ in range(userNum)]  # 每个用户打分item_id集合，

        self.data_test_userSet = []  # 存储test中的user
        self.data_test = [list() for _ in range(userNum)]  # 每个用户打分item_id集合，

    def gen_train(self):

        # data_train[time_iterval]来存储当前时间间隔内，存在数据的用户id
        # 用 data_train[user][time_interval]来存储train数据，在模型中循环训练
        # 对于validation和test同理

        df_train = pd.read_csv(self.trainPath, sep='\t', header=None)
        max_Timestamp = pd.Series.max(df_train[3])
        min_Timestamp = pd.Series.min(df_train[3])
        time_distance = max_Timestamp - min_Timestamp
        self.interval_num = int(time_distance / self.timestamp + 1)

        self.data_train_itemSet = set(df_train[1])  # 存储train中所有出现过的item
        self.data_train_userSet = set(df_train[0])  # 存储train中所有出现过的user
        self.data_train_time_userSet = [set() for _ in range(self.interval_num)]  # 存储每个时间间隔中存在哪些用户，二维
        self.data_train_time_itemSet = [set() for _ in range(self.interval_num)]  # 存储每个时间间隔中存在哪些item，二维

        for userid in range(self.userNum):
            self.data_train_user_key[userid] = [set() for _ in range(self.interval_num)]

        for itemid in range(self.itemNum):
            self.data_train_item_key[itemid] = [set() for _ in range(self.interval_num)]

        for t in range(self.interval_num):
            level_down_current = min_Timestamp + t * self.timestamp
            level_up_current = level_down_current + self.timestamp
            df_interval_current = df_train[(df_train[3] >= level_down_current) & (df_train[3] < level_up_current)]

            # 单个时间间隔中所包含的user集合
            self.data_train_time_userSet[t] = set(df_interval_current[0])
            self.data_train_time_itemSet[t] = set(df_interval_current[1])

            # 统计每个用户在当前时间间隔中所打过分的set
            for userid in self.data_train_time_userSet[t]:
                df_tmp = df_interval_current[df_interval_current[0] == userid]
                item_set = set(df_tmp[1])
                self.data_train_user_key[userid][t] = item_set

            # 统计每个item在当前时间间隔中所打过分的set
            for itemid in self.data_train_time_itemSet[t]:
                df_tmp = df_interval_current[df_interval_current[1] == itemid]
                user_set = set(df_tmp[0])
                self.data_train_item_key[itemid][t] = user_set

    # 用validation中所有的数据作为测试集，而不是下一个时间段内
    def gen_validation(self):
        df_validation = pd.read_csv(self.validationPath, sep='\t', header=None)

        df_interval_current = df_validation
        self.data_validation_userSet = [uid for uid in set(df_interval_current[0]) if uid in self.data_train_userSet]

        for userid in self.data_validation_userSet:
            df_tmp = df_interval_current[df_interval_current[0] == userid]
            tmp = [itemid for itemid in set(df_tmp[1]) if itemid in self.data_train_itemSet]
            self.data_validation[userid] = tmp

    # 用test中所有的数据作为测试集，而不是下一个时间段内
    def gen_test(self):
        df_test = pd.read_csv(self.testPath, sep='\t', header=None)

        df_interval_current = df_test
        self.data_test_userSet = [uid for uid in set(df_interval_current[0]) if uid in self.data_train_userSet]

        for userid in self.data_test_userSet:
            df_tmp = df_interval_current[df_interval_current[0] == userid]
            tmp = [itemid for itemid in set(df_tmp[1]) if itemid in self.data_train_itemSet]
            self.data_test[userid] = tmp


class DBLMF():
    def __init__(self, *args):
        # 基础参数
        self.datatype = args[0]
        self.interval = args[1]
        self.dimension = args[2]
        self.itera = args[3]
        self.userNum = args[4]
        self.itemNum = args[5]

        self.sigma0 = args[6]
        self.a0 = args[7]
        self.a1 = args[8]
        self.b0 = args[9]
        self.b1 = args[10]
        self.delta = args[11]

        # 日志基本配置
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()
        fh = logging.FileHandler('DBLMF_' + self.datatype + '_itera_' + str(self.itera), mode='a', encoding=None,
                                 delay=False)
        self.logger.addHandler(fh)

        # 数据集格式化
        gd = generate_data(self.datatype, self.interval, self.userNum, self.itemNum)
        gd.gen_train()
        gd.gen_validation()
        gd.gen_test()
        self.interval_num = gd.interval_num

        self.train_time_userSet = gd.data_train_time_userSet
        self.train_time_itemSet = gd.data_train_time_itemSet

        self.train_itemSet = gd.data_train_itemSet
        self.train_userSet = gd.data_train_userSet

        self.train_user_key = gd.data_train_user_key
        self.train_item_key = gd.data_train_item_key

        self.validation_userSet = gd.data_validation_userSet
        self.validation_data = gd.data_validation

        self.test_userSet = gd.data_test_userSet
        self.test_data = gd.data_test

        # 每个时间段，每个用户都存在一个d维的隐向量，向量初始化在之后
        self.userMat = [0 for _ in range(self.interval_num)]
        self.itemMat = [0 for _ in range(self.interval_num)]
        self.SIGMA = [0 for _ in range(self.interval_num)]  # for user
        self.OMEGA = [0 for _ in range(self.interval_num)]  # for item
        self.EPSILON = [0 for _ in range(self.interval_num)]
        self.alpha = np.ones(self.dimension)
        self.beta = np.ones(self.dimension)
        # a,b分别代表着gamma分布的的两个参数，
        # alpha_a、alpha_b的每一个维度，代表着alhpa的每一维度所服从的gamma分布的参数
        # beta同理
        self.alpha_a = np.zeros(self.dimension)
        self.alpha_b = np.zeros(self.dimension)
        self.beta_a = np.zeros(self.dimension)
        self.beta_b = np.zeros(self.dimension)

    def Tijk(self, ui, vj, k):
        return np.matmul(ui, vj) - ui[k] * vj[k]

    def self_lambda(self, epsilon):
        return (1 / (2 * epsilon)) * (1 / (1 + np.exp(-epsilon)) - 0.5)

    def VI_for_DBLMF(self):
        for t in range(self.interval_num):
            self.userMat[t] = np.random.normal(0, 1 / self.dimension, (self.userNum, self.dimension))
            self.itemMat[t] = np.random.normal(0, 1 / self.dimension, (self.itemNum, self.dimension))
            self.SIGMA[t] = np.ones((self.userNum, self.dimension))
            self.OMEGA[t] = np.ones((self.itemNum, self.dimension))
            self.EPSILON[t] = np.zeros((self.userNum, self.itemNum))
            self.logger.info('------------interval: ' + str(t) + ' -------------')
            for itera in range(self.itera):
                # compute auxiliary variables EPSILON(i,j)
                # every user and every item
                for uid in range(self.userNum):
                    for item_id in range(self.itemNum):
                        tmp1 = self.SIGMA[t][uid] * self.SIGMA[t][uid] + self.userMat[t][uid] * self.userMat[t][uid]
                        tmp2 = self.OMEGA[t][item_id] * self.OMEGA[t][item_id] + self.itemMat[t][item_id] * \
                               self.itemMat[t][item_id]
                        part1 = np.matmul(tmp1, tmp2)
                        tmp3 = np.matmul(self.userMat[t][uid], self.itemMat[t][item_id])
                        part2 = np.square(tmp3)
                        tmp4 = self.userMat[t][uid] * self.itemMat[t][item_id]
                        part3 = np.matmul(tmp4, tmp4)
                        self.EPSILON[t][uid][item_id] = np.log2(part1 + part2 - part3)

                # 需要使用当前时间段该用户打过分的item，
                # 即 self.data_train_user_key[uid][t] = item_set 和 self.train_time_userSet[t]
                for uid in range(self.userNum):
                    current_user_set = self.train_time_userSet[t]
                    # update self.userMat[t][uid][k]
                    if uid in current_user_set:
                        positive_current_user_item_set = set(self.train_user_key[uid][t])
                        negative_current_user_item_set = self.train_itemSet - positive_current_user_item_set
                        for k in range(self.dimension):
                            # part1
                            part1 = 0
                            for item_id in list(positive_current_user_item_set):
                                part1 += (1 + self.delta * 1) * (self.itemMat[t][item_id][k] - 4 * self.self_lambda(
                                    self.EPSILON[t][uid][item_id]) * self.itemMat[t][item_id][k] * self.Tijk(
                                    self.userMat[t][uid], self.itemMat[t][item_id], k))
                            # part2
                            part2 = 0
                            for item_id in list(negative_current_user_item_set):
                                part2 += self.itemMat[t][item_id][k] + 4 * self.self_lambda(
                                    self.EPSILON[t][uid][item_id]) * self.itemMat[t][item_id][k] * self.Tijk(
                                    self.userMat[t][uid], self.itemMat[t][item_id], k)
                            # part3
                            part3 = 0
                            if t != 0:
                                part3 = self.userMat[t - 1][uid][k] * self.alpha[k]
                            # update
                            self.userMat[t][uid][k] = (0.5 * part1 - 0.5 * part2 + part3) * self.SIGMA[t][uid][k]

                    else:
                        negative_current_user_item_set = self.train_itemSet
                        for k in range(self.dimension):
                            # part1
                            part1 = 0
                            # part2
                            part2 = 0
                            for item_id in list(negative_current_user_item_set):
                                part2 += self.itemMat[t][item_id][k] + 4 * self.self_lambda(
                                    self.EPSILON[t][uid][item_id]) * self.itemMat[t][item_id][k] * self.Tijk(
                                    self.userMat[t][uid], self.itemMat[t][item_id], k)
                            # part3
                            part3 = 0
                            if t != 0:
                                part3 = self.userMat[t - 1][uid][k] * self.alpha[k]
                            # update
                            self.userMat[t][uid][k] = (0.5 * part1 - 0.5 * part2 + part3) * self.SIGMA[t][uid][k]

                    # update self.SIGMA[t][uid][k]
                    if uid in current_user_set:
                        positive_current_user_item_set = set(self.train_user_key[uid][t])
                        negative_current_user_item_set = self.train_itemSet - positive_current_user_item_set
                        for k in range(self.dimension):
                            # part1
                            part1 = 0
                            for item_id in list(positive_current_user_item_set):
                                part1 += (1 + self.delta * 1) * self.self_lambda(self.EPSILON[t][uid][item_id]) * (
                                        np.square(self.itemMat[t][item_id][k]) + self.OMEGA[t][item_id][k])
                            # part2
                            part2 = 0
                            for item_id in list(negative_current_user_item_set):
                                part2 += self.self_lambda(self.EPSILON[t][uid][item_id]) * (
                                        np.square(self.itemMat[t][item_id][k]) + self.OMEGA[t][item_id][k])
                            # update
                            self.SIGMA[t][uid][k] = 1 / (self.alpha[k] + self.sigma0 + 2 * part1 + 2 * part2)

                    else:
                        negative_current_user_item_set = self.train_itemSet
                        for k in range(self.dimension):
                            # part1
                            part1 = 0
                            # part2
                            part2 = 0
                            for item_id in list(negative_current_user_item_set):
                                part2 += self.self_lambda(self.EPSILON[t][uid][item_id]) * (
                                        np.square(self.itemMat[t][item_id][k]) + self.OMEGA[t][item_id][k])
                            # update
                            self.SIGMA[t][uid][k] = 1 / (self.alpha[k] + self.sigma0 + 2 * part1 + 2 * part2)

                # 需要使用当前时间段该用户打过分的item，
                # 即 self.data_train_item_key[item_id][t] = item_set 和 self.train_time_itemSet[t]
                for item_id in range(self.itemNum):
                    current_item_set = self.train_time_itemSet[t]
                    # update self.itemMat[t][item_id][k]
                    if item_id in current_item_set:
                        positive_current_item_user_set = set(self.train_item_key[item_id][t])
                        negative_current_item_user_set = self.train_userSet - positive_current_item_user_set
                        for k in range(self.dimension):
                            # part1
                            part1 = 0
                            for uid in list(positive_current_item_user_set):
                                part1 += (1 + self.delta * 1) * (self.userMat[t][uid][k] - 4 * self.self_lambda(
                                    self.EPSILON[t][uid][item_id]) * self.userMat[t][uid][k] * self.Tijk(
                                    self.userMat[t][uid], self.itemMat[t][item_id], k))
                            # part2
                            part2 = 0
                            for uid in list(negative_current_item_user_set):
                                part2 += self.userMat[t][uid][k] + 4 * self.self_lambda(
                                    self.EPSILON[t][uid][item_id]) * self.userMat[t][uid][k] * self.Tijk(
                                    self.userMat[t][uid], self.itemMat[t][item_id], k)
                            # part3
                            part3 = 0
                            if t != 0:
                                part3 = self.itemMat[t - 1][item_id][k] * self.alpha[k]
                            # update
                            self.itemMat[t][item_id][k] = (0.5 * part1 - 0.5 * part2 + part3) * self.OMEGA[t][item_id][
                                k]

                    else:
                        negative_current_item_user_set = self.train_userSet
                        for k in range(self.dimension):
                            # part1
                            part1 = 0
                            # part2
                            part2 = 0
                            for uid in list(negative_current_item_user_set):
                                part2 += self.userMat[t][uid][k] + 4 * self.self_lambda(
                                    self.EPSILON[t][uid][item_id]) * self.userMat[t][uid][k] * self.Tijk(
                                    self.userMat[t][uid], self.itemMat[t][item_id], k)
                            # part3
                            part3 = 0
                            if t != 0:
                                part3 = self.itemMat[t - 1][item_id][k] * self.alpha[k]
                            # update
                            self.itemMat[t][item_id][k] = (0.5 * part1 - 0.5 * part2 + part3) * self.OMEGA[t][item_id][
                                k]

                    # update self.OMEGA[t][item_id][k]
                    if item_id in current_item_set:
                        positive_current_item_user_set = set(self.train_item_key[item_id][t])
                        negative_current_item_user_set = self.train_userSet - positive_current_item_user_set
                        for k in range(self.dimension):
                            # part1
                            part1 = 0
                            for uid in list(positive_current_item_user_set):
                                part1 += (1 + self.delta * 1) * self.self_lambda(self.EPSILON[t][uid][item_id]) * (
                                        np.square(self.userMat[t][uid][k]) + self.SIGMA[t][uid][k])
                            # part2
                            part2 = 0
                            for uid in list(negative_current_item_user_set):
                                part2 += self.self_lambda(self.EPSILON[t][uid][item_id]) * (
                                        np.square(self.userMat[t][uid][k]) + self.SIGMA[t][uid][k])
                            # update
                            self.OMEGA[t][item_id][k] = 1 / (self.beta[k] + self.sigma0 + 2 * part1 + 2 * part2)

                    else:
                        negative_current_item_user_set = self.train_userSet
                        for k in range(self.dimension):
                            # part1
                            part1 = 0
                            # part2
                            part2 = 0
                            for uid in list(negative_current_item_user_set):
                                part2 += self.self_lambda(self.EPSILON[t][uid][item_id]) * (
                                        np.square(self.userMat[t][uid][k]) + self.SIGMA[t][uid][k])
                            # update
                            self.OMEGA[t][item_id][k] = 1 / (self.beta[k] + self.sigma0 + 2 * part1 + 2 * part2)

                # update the Gamma distribution for alpha and beta
                for k in range(self.dimension):
                    # about user
                    self.alpha_a[k] = self.a0 + self.userNum / 2
                    part1 = 0
                    if t != 0:
                        for uid in range(self.userNum):
                            part1 += np.square(self.userMat[t][uid][k] - self.userMat[t - 1][uid][k]) \
                                     + self.SIGMA[t][uid][k] \
                                     + self.SIGMA[t - 1][uid][k]
                    else:
                        for uid in range(self.userNum):
                            part1 += self.SIGMA[t][uid][k]
                    self.alpha_b[k] = self.b0 + part1
                    # update alpha
                    self.alpha[k] = self.alpha_a[k] / self.alpha_b[k]  # 其实就是该分布的期望

                    # about item
                    self.beta_a[k] = self.a1 + self.itemNum / 2
                    part2 = 0
                    if t != 0:
                        for item_id in range(self.itemNum):
                            part2 += np.square(self.itemMat[t][item_id][k] - self.itemMat[t - 1][item_id][k]) \
                                     + self.OMEGA[t][item_id][k] \
                                     + self.OMEGA[t - 1][item_id][k]
                    else:
                        for item_id in range(self.itemNum):
                            part2 += self.OMEGA[t][item_id][k]
                    self.beta_b[k] = self.b1 + part2
                    # update beta
                    self.beta[k] = self.beta_a[k] / self.beta_b[k]

        save_userMat_path = os.path.join(self.datatype, 'userMat.txt')
        save_itemMat_path = os.path.join(self.datatype, 'itemMat.txt')
        np.savetxt(save_userMat_path, self.userMat[-1])
        np.savetxt(save_itemMat_path, self.itemMat[-1])
        self.logger.info('------------ evalution: validation -------------')
        self.evalution(self.validation_userSet, self.validation_data)
        self.logger.info('------------ evalution: test-------------')
        self.evalution(self.test_userSet, self.test_data)

    def evalution(self, user_set, test_user_items):
        pre_score = []  # 存储对每个用户而言，预测每个item得到的打分
        pre_top_k_5 = []
        pre_top_k_10 = []
        pre_top_k_20 = []
        pre_top_k_50 = []

        for user_id in user_set:
            predict_score = np.zeros(self.itemNum)
            Pu = self.userMat[-1][user_id]
            for item_id in range(self.itemNum):
                Qi = self.itemMat[-1][item_id]
                score = np.dot(Pu, Qi)
                predict_score[item_id] = score

            # 对prediction取top_k，并返回index列表
            index_sort = predict_score.argsort()
            pre_score.append(predict_score)
            pre_top_k_5.append(index_sort[-5:][::-1])
            pre_top_k_10.append(index_sort[-10:][::-1])
            pre_top_k_20.append(index_sort[-20:][::-1])
            pre_top_k_50.append(index_sort[-50:][::-1])

        precision_5 = self.precision_k(pre_top_k_5, user_set, test_user_items, 5)
        recall_5 = self.recall_k(pre_top_k_5, user_set, test_user_items, 5)
        MRR_5 = self.MRR_k(pre_top_k_5, user_set, test_user_items, 5)
        NDCG_5 = self.NDCG_k(pre_score, pre_top_k_5, user_set, test_user_items, 5)

        precision_10 = self.precision_k(pre_top_k_10, user_set, test_user_items, 10)
        recall_10 = self.recall_k(pre_top_k_10, user_set, test_user_items, 10)
        MRR_10 = self.MRR_k(pre_top_k_10, user_set, test_user_items, 10)
        NDCG_10 = self.NDCG_k(pre_score, pre_top_k_10, user_set, test_user_items, 10)

        precision_20 = self.precision_k(pre_top_k_20, user_set, test_user_items, 20)
        recall_20 = self.recall_k(pre_top_k_20, user_set, test_user_items, 20)
        MRR_20 = self.MRR_k(pre_top_k_20, user_set, test_user_items, 20)
        NDCG_20 = self.NDCG_k(pre_score, pre_top_k_20, user_set, test_user_items, 20)

        precision_50 = self.precision_k(pre_top_k_50, user_set, test_user_items, 50)
        recall_50 = self.recall_k(pre_top_k_50, user_set, test_user_items, 50)
        MRR_50 = self.MRR_k(pre_top_k_50, user_set, test_user_items, 50)
        NDCG_50 = self.NDCG_k(pre_score, pre_top_k_50, user_set, test_user_items, 50)

        self.logger.info(self.datatype + ',' + 'precision@5' + ' = ' + str(precision_5))
        self.logger.info(self.datatype + ',' + 'recall@5' + ' = ' + str(recall_5))
        self.logger.info(self.datatype + ',' + 'MRR@5' + ' = ' + str(MRR_5))
        self.logger.info(self.datatype + ',' + 'NDCG@5' + ' = ' + str(NDCG_5) + '\n')

        self.logger.info(self.datatype + ',' + 'precision@10' + ' = ' + str(precision_10))
        self.logger.info(self.datatype + ',' + 'recall@10' + ' = ' + str(recall_10))
        self.logger.info(self.datatype + ',' + 'MRR@10' + ' = ' + str(MRR_10))
        self.logger.info(self.datatype + ',' + 'NDCG@10' + ' = ' + str(NDCG_10) + '\n')

        self.logger.info(self.datatype + ',' + 'precision@20' + ' = ' + str(precision_20))
        self.logger.info(self.datatype + ',' + 'recall@20' + ' = ' + str(recall_20))
        self.logger.info(self.datatype + ',' + 'MRR@20' + ' = ' + str(MRR_20))
        self.logger.info(self.datatype + ',' + 'NDCG@20' + ' = ' + str(NDCG_20) + '\n')

        self.logger.info(self.datatype + ',' + 'precision@50' + ' = ' + str(precision_50))
        self.logger.info(self.datatype + ',' + 'recall@50' + ' = ' + str(recall_50))
        self.logger.info(self.datatype + ',' + 'MRR@50' + ' = ' + str(MRR_50))
        self.logger.info(self.datatype + ',' + 'NDCG@50' + ' = ' + str(NDCG_50) + '\n')

    def precision_k(self, pre_top_k, userSet, test_user_items, k):
        right_pre = 0
        record_number = len(userSet)
        for i in range(record_number):
            userid = userSet[i]
            for item in list(set(test_user_items[userid])):
                if item in pre_top_k[i]:
                    right_pre += 1
        return right_pre / (record_number * k)

    def recall_k(self, pre_top_k, userSet, test_user_items, k):
        recall_rate = 0
        record_number = len(userSet)
        for i in range(record_number):
            recall_pre = 0
            userid = userSet[i]
            for item in list(set(test_user_items[userid])):
                if item in pre_top_k[i]:
                    recall_pre += 1
            recall_rate += recall_pre / len(set(test_user_items[userid]))
        return recall_rate / record_number

    def MRR_k(self, pre_top_k, userSet, test_user_items, k):
        MRR_rate = 0
        record_number = len(userSet)
        for i in range(record_number):
            MRR_pre = 0
            userid = userSet[i]
            for item in list(set(test_user_items[userid])):
                if item in pre_top_k[i]:
                    index = pre_top_k[i].tolist().index(item)
                    MRR_pre += 1 / (index + 1)
            MRR_rate += MRR_pre / len(set(test_user_items[userid]))
        return MRR_rate / record_number

    def NDCG_k(self, pre_score, pre_top_k, userSet, test_user_items, k):
        NDCG_rate = 0
        record_number = len(userSet)
        for i in range(record_number):
            userid = userSet[i]
            score = pre_score[i]
            # top-k中存在预测正确的值
            if len(set(pre_top_k[i].tolist()) & set(test_user_items[userid])) != 0:
                idcg_index = 0
                dcg = 0
                idcg = 0
                # 因为idcg存在item的相对顺序，所以以pre_top_k[i]最为迭代对象
                for item in pre_top_k[i]:
                    if item in list(set(test_user_items[userid])):
                        # DCG
                        index = pre_top_k[i].tolist().index(item)
                        if index == 0:
                            dcg += score[item]
                        else:
                            dcg += score[item] / np.log2(index + 1)
                        # IDCG
                        if idcg_index == 0:
                            idcg += score[item]
                        else:
                            idcg += score[item] / np.log2(idcg_index + 1)
                        idcg_index += 1
                NDCG_rate += dcg / idcg
        return NDCG_rate / record_number


if __name__ == '__main__':
    data = ['2002_6Y', '2002_6Y_S', '2005_3Y', '2005_3Y_S', 'tallm', 'gowalla', 'lastfm']
    user_num = [4735, 1046, 4577, 2366, 5003, 4692, 831]
    item_num = [9453, 9100, 9181, 8790, 30775, 11098, 974]
    index = 3
    datatype = data[index]
    user = user_num[index]
    item = item_num[index]

    # input parameters
    itera = 1
    month = 3
    global_dim = 32
    sigma0 = 10
    a0 = 1e-4
    a1 = 1e-4
    b0 = 1e-4
    b1 = 1e-4
    delta = 1  # for weight(i,j)

    model = DBLMF(datatype, month, global_dim, itera, user, item, sigma0, a0, a1, b0, b1, delta)
    model.VI_for_DBLMF()

# 在 平方项 和 sigmoid项上，出现数字越界和分母为0的情况
