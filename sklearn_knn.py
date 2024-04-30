
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

# In[]:

def rating_sort():
    ratings = pd.read_csv('ml-25m/ratings.csv', sep=',') # 162541位用戶, 59047部電影, 共2500萬0095個評分
    ratings.sort_values(by='timestamp', ascending=True, inplace=True) # 依時間戳記timestamp遞增排序值組
    ratings.drop_duplicates(subset=['userId', 'movieId'], keep='last', inplace=True) # 移除用戶對相同電影的重複評分 (原始資料沒這問題)
    print('rating_sort() done.')
    return ratings

def movie_sort():
    movies = pd.read_csv('ml-25m/movies_extended.csv', sep=',') # 62423部電影
    movies.sort_values(by=['year', 'movieId'], inplace=True) # 依['year','movieId']遞增排序電影
    movies.reset_index(drop=True, inplace=True)
    movies.to_csv('ml-25m/movies_sorted.csv', index=False, header=True)
    print('movie_sort() done.')
    return movies

def rating_filter(ratings, score=0, count=0):
    ratings = ratings[ratings['rating'] >= score] # 截取分數>=score的評分紀錄
    gdf = ratings.groupby('movieId', as_index=False)
    counts = gdf['rating'].count()
    movieId_to_keep = counts[counts['rating'] >= count]['movieId'] # 僅保留至少得到了count個評分的電影
    ratings = ratings.merge(movieId_to_keep, on='movieId', how='right')
    print('rating_filter() done.')
    return ratings

def get_nearestK(ratings, k):
    idx_to_movieId = ratings['movieId'].unique()
    movie_user_matrix = ratings.pivot_table(index='movieId', columns='userId', values='rating', fill_value=0) # 每位用戶對每部電影的評分, 缺值為0
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(movie_user_matrix)
    movie_user_matrix = np.array(movie_user_matrix) # 所有user對movies[i]的評分
    _, indices = knn_model.kneighbors(movie_user_matrix, n_neighbors=k) # 評分分布與movies[i]最相近的k部電影(包含movies[i]自己)
    m, n = indices.shape
    for i in range(m):
        for j in range(n):
            indices[i][j] = idx_to_movieId[indices[i][j]] # 還原原始movieId
    knn_res = pd.DataFrame(data=indices)
    print('get_nearestK() done.')
    return knn_res

def convert_movieId_to_idx(movies, recommends):
    movieId_to_idx = {}
    for i in range(movies.shape[0]):
        movieId_to_idx[movies['movieId'][i]] = i
    recommends = recommends.map(lambda x: movieId_to_idx[int(x)])
    print('convert_movieId_to_idx() done.')
    return recommends

def knnRec_save_as_dict(recommends):
    recommended = {}
    df = recommends.to_numpy().astype(int)
    for i in range(len(df)):
        recommended[df[i][0]] = df[i][1:]
    with open('knnRec.pkl', 'wb') as file:
        pickle.dump(recommended, file)
    print('knnRec_save_as_dict() done.')

def load_knnRec():
    knnRec = {}
    with open('knnRec.pkl', 'rb') as file:
        knnRec = pickle.load(file)
    print('load_knnRec() done.')
    return knnRec

# In[main]:

if __name__ == "__main__":
    # 依評分時間遞增排序評分資料集
    # user_ratings = rating_sort() # 162541位用戶, 59047部電影, 共2500萬0095個評分
    # 依上映年份其遞增排序電影資料集
    # movies = movie_sort() # 62423部電影
    # 指定評分篩選條件: 最低分, 電影得分數量
    score, count = 0, 1000 # 餘162539位用戶, 3794部電影, 共2214萬1815個評分
    # score, count = 0, 10000 # 餘162109位用戶, 588部電影, 共1187萬7943個評分
    batch_name = str(score)+'x'+str(count)
    # 篩選評分
    # user_ratings = rating_filter(user_ratings, score, count)
    # 儲存篩選結果
    # user_ratings.to_csv('ml-25m/ratings@'+batch_name+'.csv', index=False, header=True)
    # 使用已處理的評分檔和電影檔
    user_ratings = pd.read_csv('ml-25m/ratings@'+batch_name+'.csv', sep=',')
    movies = pd.read_csv('ml-25m/movies_sorted.csv', sep=',')

    # In[KNN協同過濾]:
    # 以cosine相似度計算每部電影各自最相似的其他50部電影
    knn_recommends = get_nearestK(user_ratings, 51)
    # 儲存計算結果
    knn_recommends.to_csv('knn_recommended.csv', index=False, header=True)
    # 將電影矩陣的索引還原為資料集內的索引
    knn_recommends_sorted = convert_movieId_to_idx(movies, knn_recommends)
    # 儲存還原結果
    knn_recommends_sorted.to_csv('knn_recommended_sorted.csv', index=False, header=True)
    # 儲存矩陣變數
    knnRec_save_as_dict(knn_recommends_sorted)
    # 載入矩陣變數
    knnRec = load_knnRec()
