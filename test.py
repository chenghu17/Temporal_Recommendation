

import pandas as pd

def itemDict(path):
    item_dict = dict()
    item_data = pd.read_csv(path)
    item = item_data.loc[:,'movieId']
    item_Num = len(item)
    for id in range(item_Num):
        key = item.iloc[id]
        if key not in item_dict.values():
            item_dict[id] = key
    # print(item)
    # print(len(item))
    # print(item.iloc[0])
    return item_dict

def splitData(path):
    data_df = pd.read_csv(pd)


    return user_Num

if __name__ == '__main__':
    # path = 'dataset/movies.csv'
    # item_dict = itemDict(path)

    path = 'dataset/ratings.csv'
    user_Num = splitData(path)