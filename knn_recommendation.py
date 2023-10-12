
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def preprocessing(ratings, threshold):
    ratings = ratings.drop(columns=['timestamp']) # 去除時間戳記timestamp欄位
    ratings['rating'] = ratings['rating'].apply(lambda x: int(x*2)) # 將評分從range(0.5, 5, 0.5)改為range(1, 10, 1)
    ratings = ratings[ratings['rating'] > 6] # 截取分數>6的評分, 餘1562萬1924個評分
    ratings = ratings[ratings['rating'] < 10] # 截取分數<10的評分, 餘1201萬1260個評分
    gdf = ratings.groupby('movieId', as_index=False)
    counts = gdf['rating'].count()
    movieId_to_keep = counts[counts['rating'] >= threshold]['movieId'] # 至少有100個分數為7~9分的電影, 共7459部
    return ratings.merge(movieId_to_keep, on='movieId', how='right') # 餘161893位用戶, 7459部電影, 1160萬4976個評分
    
def get_nearestK(ratings, k, keep_movieId=True):
    if keep_movieId:
        idx_to_movieId = ratings['movieId'].unique()
    movie_user_matrix = ratings.pivot_table(index='movieId', columns='userId', values='rating', fill_value=0) # 每位用戶對每部電影的評分, 缺值為0
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(movie_user_matrix)
    movie_user_matrix = np.array(movie_user_matrix) # 所有user對movies[i]的評分
    _, indices = knn_model.kneighbors(movie_user_matrix, n_neighbors=k) # 評分分布與movies[i]最相近的k部電影(包含movies[i]自己)
    if keep_movieId:
        m, n = indices.shape
        for i in range(0, m):
            for j in range(0, n):
                indices[i][j] = idx_to_movieId[indices[i][j]]
    knn_res = pd.DataFrame(data = indices)
    return knn_res

def movie_sort():
    movies = pd.read_csv('movies_extended.csv', sep = ',') # 62423部電影
    movies.sort_values(by = ['year', 'movieId'], ascending = True, inplace = True) # 依['year','movieId']遞增排序電影
    movies.to_csv('movies_sorted.csv', index = False, header = True)

def convert_movieId_to_idx(movies, recommends):
    movieId_to_idx = {}
    for i in range(0, len(movies)):
        movieId_to_idx[movies['movieId'][i]] = i
    return recommends.map(lambda x: movieId_to_idx[int(x)])

# In[main]:

if __name__ == "__main__":
    user_ratings = pd.read_csv('ml-25m/ratings.csv', sep = ',') # 162541位用戶, 59047部電影, 共2500萬0095個評分
    user_ratings = preprocessing(user_ratings, 100)
    knn_recommends = get_nearestK(user_ratings, 51, True)
    knn_recommends.to_csv('knn_recommended.csv', index = False, header = True)
    # movie_sort()
    movies = pd.read_csv('movies_sorted.csv', sep = ',') # 62423部電影
    knn_recommends = convert_movieId_to_idx(movies, knn_recommends)
    knn_recommends.to_csv('knn_recommended_sorted.csv', index = False, header = True)
