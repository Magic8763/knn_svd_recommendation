
import pandas as pd

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

# In[main]:

if __name__ == "__main__":
    # 依評分時間遞增排序評分資料集
    user_ratings = rating_sort() # 162541位用戶, 59047部電影, 共2500萬0095個評分
    # 依上映年份其遞增排序電影資料集
    movies = movie_sort() # 62423部電影
    # 指定評分篩選條件: 最低分, 電影得分數量
    score, count = 0, 1000 # 餘162539位用戶, 3794部電影, 共2214萬1815個評分
    # score, count = 0, 10000 # 餘162109位用戶, 588部電影, 共1187萬7943個評分
    df_name = str(score)+'x'+str(count)
    # 篩選評分
    user_ratings = rating_filter(user_ratings, score, count)
    # 儲存篩選結果
    user_ratings.to_csv('ml-25m/ratings@'+df_name+'.csv', index=False, header=True)
    # 載入已前處理的評分檔和電影檔
    # user_ratings = pd.read_csv('ml-25m/ratings@'+df_name+'.csv', sep=',')
    # movies = pd.read_csv('ml-25m/movies_sorted.csv', sep=',')
