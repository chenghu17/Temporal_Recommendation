
import numpy as np
import pandas as pd
import time



if __name__ == '__main__':
    
    # trainPath = 'data_FM/train.tsv'

    validationPath = 'data_LastFM/ranking.tsv'
    # validationPath = 'data_Epinions/validation.csv'
    # validationPath = 'data_FineFoods/validation.csv'
    df_validation = pd.read_csv(validationPath, sep='\t', header=None)
    # print(df_validation.head(100))
    userSet = list(df_validation[0].drop_duplicates())
    print(len(userSet))

    # print(userSet)
    # a = set([1,2,2,3,4,5])
    # b = set([1])
    # print(len(a&b))

    # itemMat_18 = np.loadtxt('evolution18/itemMat30.txt')
    # userMat_18 = np.loadtxt('evolution18/userMat30.txt')
    # itemMat_stand = np.loadtxt('evolution_standard/itemMat30.txt')
    # userMat_stand = np.loadtxt('evolution_standard/userMat30.txt')

    # file = open('test_user.tsv','a')
    #
    # for i in range(1892):
    #     file.write(str(i)+'\n')


