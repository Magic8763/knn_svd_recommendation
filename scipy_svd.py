
from collections import defaultdict
import numpy as np
import pandas as pd
import random
from scipy.sparse.linalg import svds
from scipy.linalg import svd
import pickle

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

def get_SVD(ratings, k=6, using_svds=True):
    # 計算每位用戶對每部電影的評分
    user_ratings_df = ratings.pivot_table(index='userId', columns='movieClass', values='rating')
    # 計算每位用戶給分的平均值
    avg_ratings = user_ratings_df.mean(axis=1)
    # 將分數中心化Centering
    user_ratings_centered = user_ratings_df.sub(avg_ratings, axis=0)
    del user_ratings_df

    # 奇異值分解SVD
    user_ratings_centered.fillna(0, inplace=True) # nan補0
    if svds:
        U, sigma, Vt = svds(user_ratings_centered.to_numpy(), k=k, random_state=15)
    else:
        U, sigma, Vt = svd(user_ratings_centered.to_numpy(), full_matrices=False, check_finite=False)
    print('get_SVD() done.')
    res = {'U': U, 'sigma': sigma, 'Vt': Vt, 'rowAvg': avg_ratings.values.reshape(-1, 1)}
    return res

def build_SVD_df(res, original_userId, original_movieId, k=0):
    U, sigma, Vt, rowAvg = res['U'][:,k:], res['sigma'][k:], res['Vt'][k:,:], res['rowAvg']
    S = np.diag(sigma) # 轉換長度r的向量sigma為rxr的對角矩陣
    U_sigma_Vt = np.dot(np.dot(U, S), Vt) # 相乘得到SVD矩陣
    # U_sigma_Vt = U@S@Vt # 功能同上, @代表矩陣乘法, 即np.dot()

    # 將分數去中心化Decentering
    U_sigma_Vt += rowAvg
    # 轉換成資料表
    sorted_index = sorted(original_userId) # 取得遞增排序的userId
    svd_res = pd.DataFrame(U_sigma_Vt, index=sorted_index, columns=original_movieId)

    # 找出未評分的userId
    fill_index = list(range(1, sorted_index[0]))
    for i in range(1, len(sorted_index)):
        if sorted_index[i-1]+1 < sorted_index[i]:
            indices = list(range(sorted_index[i-1]+1, sorted_index[i]))
            fill_index.extend(indices)
    #print(len(fill_index), max(sorted_index)-len(sorted_index)) # 未評分者人數

    # 未評分者以電影的平均得分作為預測結果
    avg_pred_ratings = np.mean(U_sigma_Vt, axis=0) # 計算每部電影的平均得分
    fill_df = pd.DataFrame(
        [avg_pred_ratings for _ in range(len(fill_index))],
        index=fill_index, 
        columns=original_movieId) # 以平均得分製表
    svd_res = pd.concat([svd_res, fill_df], axis=0, join='outer', ignore_index=False) # 合併兩表
    del fill_df
    svd_res.sort_index(inplace=True) # 依userId遞增排序
    print('build_SVD_df() done.')
    return svd_res

def get_preds(svd_df, ratings):
    preds = []
    for i in range(ratings.shape[0]):
        userId = ratings['userId'].iloc[i]
        movieId = ratings['movieId'].iloc[i]
        y_true = ratings['rating'].iloc[i]
        y_pred = svd_df[movieId][userId]
        preds.append((userId, movieId, y_true, y_pred))
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
    # tp: 相關且被推薦, tn: 不相關且不被推薦, fp: 不相關但被推薦, fn: 相關但不被推薦
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
    for uid, _, r_ui, est in preds:
        # 用戶uid給某電影的真實分數r_ui和模型預測分數est
        user_est_true[uid].append((est, r_ui))
        y_pred.append(est)
        y_true.append(r_ui)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    res = get_rmse_mae_mape(y_true, y_pred)
    precision, recall = get_precision_recall(user_est_true, k=k, threshold=3.5)
    res['precision'] = precision
    res['recall'] = recall
    print('get_error_metrics() done.')
    return y_pred, res

def get_top_k_df(svd_df, k=10):
    # 計算每位用戶的前k部推薦電影
    users, movies = svd_df.index+1, svd_df.columns
    svd_df = svd_df.to_numpy()
    top_k_df = []
    for i in range(svd_df.shape[0]):
        sorted_idx = svd_df[i].argsort()[::-1]
        temp = [movies[i] for i in sorted_idx[:k]]
        top_k_df.append(temp)
        if (i+1)%10000 == 0:
            print(i+1, 'done.')
    top_k_df = pd.DataFrame(top_k_df, index=users)
    print('get_top_k_df() done.')
    return top_k_df

def save_top_k_df(top_k_df, fname, k):
    k = 'svd_predict_Top'+str(k)+'_'
    with open(k+fname+'.pkl', 'wb') as file:
        pickle.dump(top_k_df, file) # 寫出相關變數
    print('save_data() done.')
    
def load_top_k_df(fname, k):
    k = 'svd_predict_Top'+str(k)+'_'
    with open(k+fname+'.pkl', 'rb') as file:
        top_k_df = pickle.load(file) # 寫出相關變數
    print('top_k_df() done.')
    return top_k_df

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
    user_ratings = user_ratings[['userId', 'movieId', 'rating', 'movieClass']]
    train_data = user_ratings.iloc[:int(user_ratings.shape[0]*0.8)] # 80%用於訓練
    test_data = user_ratings.iloc[int(user_ratings.shape[0]*0.8):] # 20%用於測試
    del user_ratings

    # In[訓練SVD模型]:
    # 儲存原始userId, movieId
    original_userId, original_movieId = train_data['userId'].unique(), train_data['movieId'].unique()
    # SVD奇異值分解
    k = 6
    svd_res = get_SVD(train_data, k=k, using_svds=True)
    # svd_res = get_SVD(train_data, using_svds=False)
    # 製作SVD推薦結果
    svd_df = build_SVD_df(svd_res, original_userId, original_movieId, k=k-1)
    del original_userId, original_movieId
    # 儲存訓練結果
    # svd_df.to_csv('svd_predict_df@'+df_name+'.csv', index=True, header=True)
    # 載入訓練結果
    # svd_df = pd.read_csv('SVD_predict_df@'+df_name+'.csv', index_col=0, sep=',')

    # In[測試SVD模型]:
    # 以訓練集測試
    train_preds = get_preds(svd_df, train_data)
    # 計算評估指標
    train_pred_est, train_res = get_error_metrics(train_preds, k=10)
    # train_res = {
    #     'RMSE': 0.8458116529731118,
    #     'MAE': 0.6298914443673805,
    #     'MAPE': 27.70510916248598,
    #     'precision': 0.6063962080609273,
    #     'recall': 0.6377281617654386
    # }
    
    # 以測試集測試
    test_preds = get_preds(svd_df, test_data)
    # 計算評估指標
    test_pred_est, test_res = get_error_metrics(test_preds, k=10)
    # test_res = {
    #     'RMSE': 1.0315808084480893,
    #     'MAE': 0.787895870424359,
    #     'MAPE': 34.27665495221534,
    #     'precision': 0.4859368326036387,
    #     'recall': 0.5434914129178345
    # }

    # In[TopK推薦結果]:
    k = 50
    # 計算每位用戶的前k部推薦電影
    top_k_df = get_top_k_df(svd_df, k=k)
    # 儲存推薦結果
    top_k_df.to_csv('SVD_predict_top'+str(k)+'@'+df_name+'.csv', index=True, header=True)
    # 載入推薦結果
    # top_k_df = pd.read_csv('SVD_predict_top'+str(k)+'@'+df_name+'.csv', index_col=0)
