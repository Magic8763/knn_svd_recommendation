
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

# In[]:

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
    df_name = '0x1000'
    # 載入已前處理的評分檔和電影檔
    user_ratings = pd.read_csv('ml-25m/ratings@'+df_name+'.csv', sep=',')
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
