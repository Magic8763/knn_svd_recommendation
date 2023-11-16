# KNN Movie Recommendation
![](https://img.shields.io/github/stars/magic8763/knn_recommendation)
![](https://img.shields.io/github/watchers/magic8763/knn_recommendation)
![](https://img.shields.io/github/forks/magic8763/knn_recommendation)

基於用戶對電影評分的總體紀錄，以基於 KNN 模型的協同過濾 (*Collaborative filtering*) 方法找出最適合推薦給每部電影之觀看者的 50 部其他電影。

## Prerequisites
- Python3, NumPy, Pandas, Scikit-learn

## Description
由於所用資料集的用戶評分數量龐大，考慮記憶體容量有限，預先採用以下自訂門檻對用戶評分進行篩選。
1. 將評分分數轉換為 \[1, 10] 之間的整數，僅保留分數為 7 ~ 9 分的用戶評分
2. 根據被評分的電影分群所有用戶評分，紀錄至少被評了 100 次分數為 7 ~ 9 分的電影清單，其他評分對象不在該清單內的用戶評分將被去除
3. 將剩餘的用戶評分轉換為 user-movie 矩陣

最後使用 Scikit-learn 提供的 NearestNeighbors 函數為 user-movie 矩陣計算 Cosine 相似度，最終根據該計算結果列出每部電影的前 50 近鄰。

## Dataset
- [MovieLens 25M](https://grouplens.org/datasets/movielens/25m) 其中的 `ratings.csv` 用戶評分資料集
  - 由 162541 位用戶對 59047 部電影做出的 2500 萬 0095 個評分，每個評分包含 userId, movieId, rating 等特徵
- (Optional) [Threading Crawler](https://github.com/Magic8763/threading_crawler/tree/main) 的 `movies_extended.csv` 電影特徵資料集

## Output
- `knn_recommended.csv`: 對於通過自訂門檻篩選後剩餘的每部電影，與其具有最高相似度的前 50 部其他電影
- \(Optional) `movies_sorted.csv`: `movies_extended.csv` 的排序版本，在此基於 year, movieId 兩特徵遞增排序為例
- \(Optional) `knn_recommended_sorted.csv`: 以排序後的電影索引取代 `knn_recommended.csv` 內代表電影的 movieId 

## Authors
* **[Magic8763](https://github.com/Magic8763)**

## License
This project is licensed under the [MIT License](https://github.com/Magic8763/knn_recommendation/blob/main/LICENSE)

