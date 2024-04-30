
from collections import defaultdict
import numpy as np
import pandas as pd
import random
import pickle
from datetime import datetime
from surprise import SVD, Reader, Dataset, SVDpp
from surprise.model_selection import GridSearchCV

# APP
# 1.同步/非同步執行緒對svd模型的訪問

# DB
# 1.讀取LINE用戶評分, 與訓練資料合併後重新訓練svd模型
# # LINE用戶的userId = 資料庫內ID順序+162541 (最小應為162542)

# In[]:

def get_numstr(n):
    num_dict = {0: '', 1: 'K', 2: 'M', 3: 'B'}
    i = 0
    while n//1000 > 0:
        n //= 1000
        i += 1
    s = str(n)+num_dict[i]
    return s

def df_shuffled(df_name):
    df = pd.read_csv(df_name+'.csv', sep=',') # 3794部電影
    shuffle_idx = list(range(df.shape[0]))
    random.shuffle(shuffle_idx)
    df = df.iloc[shuffle_idx]
    df.reset_index(drop=True, inplace=True)
    print('df_shuffled() done.')
    return df

def movie_compactify(ratings, fname, update_movies=False):
    movieIds = ratings['movieId'].unique()
    movieClass = defaultdict(int)
    for mId in movieIds:
        movieClass[mId] = len(movieClass)+1
    ratings['movieClass'] = ratings['movieId'].apply(lambda x: movieClass[x]) # 壓縮電影編號, 缺值為0
    ratings.to_csv('ml-25m/ratings@'+fname+'_compactify.csv', index=False, header=True)
    if update_movies:
        movies = pd.read_csv('ml-25m/movies_sorted.csv', sep=',') # 62423部電影
        movies['movieClass'] = movies['movieId'].apply(lambda x: movieClass[x]) # 壓縮電影編號, 缺值為0
        movies.to_csv('ml-25m/movies@'+fname+'_compactify.csv', index=False, header=True)
        print('movie_compactify() done.')
        return ratings, movies
    print('movie_compactify() done.')
    return ratings

def read_ratings(df_name):
    ratings = pd.read_csv('ml-25m/ratings@'+df_name+'_compactify.csv', sep=',')
    print('read_ratings() done.')
    return ratings

def df_to_trainset(df):
    # 將訓練資料轉換為surprise專用的訓練集
    # 將pd.DF類別的訓練集映射為Reader()的'DatasetAutoFolds'類別
    data_mf = Dataset.load_from_df(df, Reader())
    # 評分解析類別Reader的參數
    # name: 使用內建資料集['ml-100k', 'ml-1m', 'jester'], 預設為None
    # line_format: 資料集欄位名稱, 預設為'user item rating'
    # sep: 分隔符號, 預設為None
    # rating_scale: 評分範圍, 預設為(1, 5), 即1~5分
    # skip_lines: 跳過前幾個columns, 預設為0

    # 再轉換為surprise專用的訓練集類別'Trainset'
    trainset = data_mf.build_full_trainset()
    print('df_to_trainset() done.')
    return trainset

def get_svd_model(trainset, fname='', pp=False):
    # SVD奇異值分解演算法
    algo = SVDpp if pp else SVD
    svd = algo(n_factors=100, random_state=15, verbose=True)
    # n_factors: 分解矩陣的大小(因子數), 預設為100
    # n_epochs: SGD重複訓練的迭代次數, 預設為20
    # biased: SVD使用基線(True)或無偏差(False)的演算法版本, 預設為True
    # init_mean: 因子向量初始化的常態分佈的平均值, 預設為0
    # init_std_dev: 因子向量初始化的常態分佈的標準差, 預設為0.1
    # lr_all: 所有參數的學習率, 預設為0.005
    # reg_all: 所有參數的正規化項, 預設為0.02
    # random_state: 隨機種子, 預設為None
    # verbose: 列印目前迭代輪次, 預設為False

    print('  Training SVD model..')
    start_t = datetime.now()
    svd.fit(trainset)
    print('  Time taken: {}\nDone.'.format(datetime.now()-start_t))
    adict = {
        'svd': svd,
        'max_userId': 162541, # ratings_25M內最大的userId = 162541
        'last_userId': max(trainset._raw2inner_id_users)
    }
    fname = 'svd++@'+fname if pp else 'svd@'+fname
    with open(fname+'.pkl', 'wb') as file:
        pickle.dump(adict, file) # 寫出svd模型, 最大已訓練userId
    print('get_svd_model() done.')
    return svd

def get_best_svd_model(trainset, train_data, fname='', pp=False):
    # 最佳SVD配置
    param_grid = {'n_epochs': [10, 20],
                  'lr_all': [0.002, 0.005],
                  'n_factors': [50, 100, 150],
                  'random_state': [15]}
    algo = SVDpp if pp else SVD
    jobs = 1 if pp else 2
    gs = GridSearchCV(algo, param_grid, measures=['rmse', 'mae'], cv=3, joblib_verbose=1, n_jobs=jobs)
    train_data_mf = Dataset.load_from_df(train_data, Reader())
    gs.fit(train_data_mf)
    print('  best RMSE:', gs.best_score['rmse'], '\n  best MAE:', gs.best_score['mae'])
    print('  best params:', gs.best_params['rmse'])
    svd = gs.best_estimator['rmse'] 
    svd.fit(trainset) # 以最佳參數訓練SVD
    adict = {
        'svd': svd,
        'max_userId': 162541, # ratings_25M內最大的userId = 162541
        'last_userId': max(trainset._raw2inner_id_users)
    }
    fname = 'svd++_best@'+fname if pp else 'svd_best@'+fname
    with open(fname+'.pkl', 'wb') as file:
        pickle.dump(adict, file) # 寫出最佳svd模型, 最大已訓練userId
    print('get_best_svd_model() done.')
    return svd

def load_svd_model(fname='', get_best=False, pp=False):
    # 載入預訓練的SVD模型 
    fname = '_best@'+fname if get_best else '@'+fname
    fname = 'svd++'+fname if pp else 'svd'+fname
    with open(fname+'.pkl', 'rb') as file:
        adict = pickle.load(file) # 載入svd字典
    svd, max_userId, last_userId = adict['svd'], adict['max_userId'], adict['last_userId']
    print('load_svd_model() done.')
    return svd, max_userId, last_userId

def get_preds(model, dataset):
    # 測試SVD模型
    dataset = dataset.build_testset() # 返回轉換為測試專用格式的資料集
    preds = model.test(dataset) # 測試模型
    # preds的資料結構
    # .uid: 用戶id
    # .iid: 項目id(電影)
    # .r_ui: 用戶對項目的真實評分
    # .est: 由模型預測的評分
    # .details={...}: 其他參數
    print('get_preds() done.')
    return preds

def get_rmse_mae_mape(y_true, y_pred):
    # 計算評估指標 (rmse<2是良好的, mape<25是優秀的)
    diff = y_true-y_pred
    rmse = np.sqrt(np.mean([diff[i]**2 for i in range(len(y_pred))]))
    mae = np.mean(np.abs(diff))
    mape = np.mean(np.abs(diff/y_true))*100
    res = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    print('get_rmse_mae_mape() done.')
    return res

def get_precision_recall(user_est_true, k=10, threshold=3.5):
    # 為每位用戶推薦的前k部電影(預測前k高分者)的precision和recall
    # 預測分數>=threshold時, 判定為推薦, 反之為不推薦
    # 真實分數>=threshold時, 判定為相關, 反之為不相關
    # tp: 相關且推薦, tn: 不相關且不推薦, fp: 不相關但推薦, fn: 相關但不推薦
    # accuracy = (tp+tn)/(tp+tn+fp+fn)
    # precision = tp/(tp+fp)
    # recall = tp/(tp+fn)
    # F1 score = 2/(1/precision+1/recall)
    precisions, recalls = {}, {}
    for uid, user_ratings in user_est_true.items():
        # 依預測分數遞減排序user給出的所有評分
        user_ratings.sort(key=lambda x: x[0], reverse=True) 
        # tp+fp: 推薦的電影數
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        # tp+fn: 相關的電影數
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        # tp: 推薦且相關的電影數
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )
        precisions[uid] = n_rel_and_rec_k/n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k/n_rel if n_rel != 0 else 0
    # 平均所有用戶的precision和recall
    precision = sum(prec for prec in precisions.values())/len(precisions)
    recall = sum(rec for rec in recalls.values())/len(recalls)
    print('get_precision_recall() done.')
    return precision, recall

def get_error_metrics(preds, k=10):
    user_est_true, y_pred, y_true = defaultdict(list), [], []
    for pred in preds:
        # 用戶uid給某電影的真實分數r_ui和模型預測分數est
        user_est_true[pred.uid].append((pred.est, pred.r_ui))
        y_pred.append(pred.est)
        y_true.append(pred.r_ui)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    res = get_rmse_mae_mape(y_true, y_pred)
    precision, recall = get_precision_recall(user_est_true, k=k, threshold=3.5)
    res['precision'] = precision
    res['recall'] = recall
    print('get_error_metrics() done.')
    return y_pred, res

def save_data(preds, pred_est, fname='', train=True):
    var_name = 'train' if train else 'test'
    var_dict = {'svd_preds': preds, 'svd_pred_est': pred_est}
    with open(var_name+'_var@'+fname+'.pkl', 'wb') as file:
        pickle.dump(var_dict, file) # 寫出相關變數
    print('save_data() done.')
    
def load_data(fname='', train=True):
    var_name = 'train' if train else 'test'
    with open(var_name+'_var@'+fname+'.pkl', 'rb') as file:
        var_dict = pickle.load(file) # 寫出相關變數
    print('load_data() done.')
    return var_dict['svd_preds'], var_dict['svd_pred_est']

def get_top_k(preds, k=10):
    # 測試集用戶的top_k推薦
    top_k = defaultdict(list)
    for uid, iid, _, est, _ in preds:
        top_k[uid].append((iid, est)) # 用戶uid對電影iid的預測評分est
    for uid, iid_est in top_k.items():
        iid_est.sort(key=lambda x: x[1], reverse=True) # 依預測評分遞減排序(電影, 預測評分)
        top_k[uid] = iid_est[:k] # 截取前n個項目(可能會不足n個)
    # for uid, iid_est in top_k.items():
    #     print(uid, [iid for (iid, _) in iid_est]) # 推薦給用戶uid的前n部電影
    print('get_top_k() done.')
    return top_k

def recommendation_order(model, userId, target_movies, k=0):
    # 生成指定用戶的top_k推薦
    table = {'userId': [userId]*len(target_movies),
             'movieId': target_movies}
    df = pd.DataFrame(table)
    df['rating'] = 0.0
    dataset = Dataset.load_from_df(df, Reader())
    dataset = dataset.build_full_trainset().build_testset()
    preds = model.test(dataset) # 測試模型
    res = [(est, iid) for uid, iid, _, est, _ in preds]
    res.sort(reverse=True)
    print('recommendation_order() done.')
    return res[:k] if k > 0 else res

# In[main]:

if __name__ == "__main__":
    # 指定ratings規格: 評分篩選條件, 抽樣數量(0=全部)
    df_name, n = '0x1000', 1000000
    # user_ratings = df_shuffled('ml-25m/ratings@'+df_name) # 打亂資料排序
    if n > 0:
        df_name += '_'+get_numstr(n)
        # user_ratings = user_ratings.iloc[:n] # 截取100萬筆評分紀錄, 餘3794部電影
    # 電影ID緊湊化
    # user_ratings, _ = movie_compactify(user_ratings, df_name, update_movies=True)
    # 使用已處理的評分檔
    user_ratings = read_ratings(df_name)

    # In[抽取特徵並分割訓練/測試集]:
    user_ratings = user_ratings[['userId', 'movieClass', 'rating']]
    train_data = user_ratings.iloc[:int(user_ratings.shape[0]*0.8)] # 80%用於訓練
    test_data = user_ratings.iloc[int(user_ratings.shape[0]*0.8):] # 20%用於測試
    del user_ratings
    trainset = df_to_trainset(train_data)
    testset = df_to_trainset(test_data)
    
    # In[訓練SVD或SVD++模型]:
    # svd = get_svd_model(trainset, df_name, pp=False) # SVD
    # svdpp = get_svd_model(trainset, df_name, pp=True) # SVD++
    # 載入預訓練的模型
    svd, max_userId, last_userId = load_svd_model(df_name, get_best=False, pp=False)
    # svdpp, max_userId, last_userId = load_svd_model(df_name, get_best=False, pp=True)
    
    # In[訓練最佳參數的SVD或SVD++模型]:
    # svd = get_best_svd_model(trainset, train_data, df_name, pp=False)
    # svdpp = get_best_svd_model(trainset, train_data, df_name, pp=True)
    # 載入預訓練的模型
    # svd, max_userId, last_userId = load_svd_model(df_name, get_best=True, pp=False)
    svdpp, max_userId, last_userId = load_svd_model(df_name, get_best=True, pp=True)
    
    # In[測試SVD模型]:
    model = svd
    # model = svdpp
    # 以訓練集測試
    train_preds = get_preds(model, trainset)
    # 計算評估指標
    train_pred_est, train_res = get_error_metrics(train_preds, k=10)
    # SVD_best train_res = {
    #     'RMSE': 0.7274954206128231,
    #     'MAE': 0.5607129798599508,
    #     'MAPE': 24.654346676170256,
    #     'precision': 0.7429401616044377,
    #     'recall': 0.7165575250374736}
    # }
    # SVD++_best train_res = {
    #     'RMSE': 0.6962367345988407,
    #     'MAE': 0.5364719270972894,
    #     'MAPE': 23.503126512723657,
    #     'precision': 0.7559124810287762,
    #     'recall': 0.7276201876315835}
    # }

    # 以測試集測試
    test_preds = get_preds(model, testset)
    # 計算評估指標
    test_pred_est, test_res = get_error_metrics(test_preds, k=10)
    # SVD_best test_res = {
    #     'RMSE': 0.9066104918980751,
    #     'MAE': 0.6971252487154936,
    #     'MAPE': 30.93577843014798,
    #     'precision': 0.6050963536522577,
    #     'recall': 0.5998366168950949}
    # }
    # SVD++_best test_res = {
    #     'RMSE': 0.9037339109347786,
    #     'MAE': 0.6961453278572577,
    #     'MAPE': 30.72067645286987,
    #     'precision': 0.6032689103423453,
    #     'recall': 0.5947095086548151
    # }

    # 儲存測試結果
    # save_data(train_preds, train_pred_est, df_name, train=True)
    # save_data(test_preds, test_pred_est, df_name, train=False)
    # 載入測試結果
    # # train_preds, train_pred_est = load_data(df_name, train=True)
    # # test_preds, test_pred_est = load_data(df_name, train=False)

    # In[評分預測]:
    # 測試集內每位用戶的前k個推薦電影
    # test_top_k = get_top_k(test_preds, k=10)
    # 指定用戶ID和電影ID清單
    uid, target_movies = 100000, [499, 500]
    # 清單內用戶的前k個推薦電影
    test_top_k = recommendation_order(svd, uid, target_movies, k=10)
