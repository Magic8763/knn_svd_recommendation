# Movie Recommendation Model
![](https://img.shields.io/github/stars/magic8763/knn_recommendation)
![](https://img.shields.io/github/watchers/magic8763/knn_recommendation)
![](https://img.shields.io/github/forks/magic8763/knn_recommendation)
![shields](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)

以用戶對電影評分的資料集作為訓練資料，使用基於 KNN, SVD, SVD++ 模型的協同過濾方法 (*Collaborative filtering*)，生成用於推薦系統的預測模型。

## Prerequisites
- Python3, Pandas, NumPy, Scikit-learn, Surprise, SciPy

## Description
- `ratings_filter.py`: 資料前處理功能，使用自訂條件對資料量龐大的用戶評分資料集進行篩選
- `sklearn_knn.py`: 採用 Scikit-learn NearestNeighbors 演算法，基於電影之間的 cosine 相似性，計算出每部電影的 Top K 近鄰 (相似電影)
- `scipy_svd.py`: 採用 SciPy svds 演算法，透過對 user-movie 矩陣的奇異值分解預測每位用戶對任一電影的評分
- `surprise_svd.py`: 採用 Surprise svd, svdpp 演算法，以最佳模型參數訓練 SVD, SVD++ 模型，用於預測指定用戶對任一電影的評分

## Dataset
- [MovieLens 25M](https://grouplens.org/datasets/movielens/25m)
  - `ratings.csv`: 用戶評分資料集，由 162541 位用戶對 59047 部電影做出的 2500 萬 0095 個評分，每個評分包含 userId, movieId, rating 等特徵
- [Threading Crawler](https://github.com/Magic8763/threading_crawler/tree/main)
  - `movies_extended.csv`: 電影特徵資料集，包含 62423 部電影的多項特徵，如 movieId, title, year, genres, grade 等

## Output
- `ratings@0x1000_1M_compactify.csv`: 前處理篩選後剩餘的用戶評分，此範例檔名代表「資料抽樣量 100 萬筆，其中的電影必須是得到了至少 1000 個評分者」的篩選結果，並且完成了電影 ID 的緊湊化 (compactify)
- `movies_sorted.csv`: `movies_extended.csv` 的排序版本，預設依照 year, movieId 兩特徵遞增排序
- `knn_recommended.csv`: 對於通過篩選後剩餘的每部電影，與其具有最高 cosine 相似度的前 K 部其他電影，預設 K = 50
- `knn_recommended_sorted.csv`: 將 `knn_recommended.csv` 的緊湊化電影索引還原為 `movies_sorted.csv` 的電影索引
- `knnRec.pkl`: `knn_recommended_sorted.csv` 的字典格式
- `svd_predict_df@0x1000_1M.csv`: 由 SciPy svds 演算法生成的 user-movie 矩陣，其值表示用戶對電影的預測評分
- `svd_predict_top50@0x1000_1M.csv`: 根據 `svd_predict_df@0x1000_1M.pkl` 計算每位用戶的前 50 部推薦電影
- `svd++_best@0x1000_1M.pkl`: 由 Surprise svdpp 演算法訓練的 SVD++ 模型

## Reference
- [深入淺出常用推薦系統演算法 Recommendation System](https://chriskang028.medium.com/%E6%B7%B1%E5%85%A5%E6%B7%BA%E5%87%BA%E5%B8%B8%E7%94%A8%E6%8E%A8%E8%96%A6%E7%B3%BB%E7%B5%B1%E6%BC%94%E7%AE%97%E6%B3%95-recommendation-system-42f2437e3e9a) - Chris Kang
- [How to Build a Movie Recommendation System](https://towardsdatascience.com/how-to-build-a-movie-recommendation-system-67e321339109) - Ramya Vidiyala

## Authors
* **[Magic8763](https://github.com/Magic8763)**

## License
This project is licensed under the [MIT License](https://github.com/Magic8763/knn_recommendation/blob/main/LICENSE)

