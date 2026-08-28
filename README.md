## 檔案說明
- train.py : 模型訓練 (路徑要設對)
- data.py : 建立一個 PyTorch 的語意分割 Dataset，把 JPEGImages.256 裡的 JPG 影像，和 SegmentationClass.256 整合，然後提供給模型訓練
- utils.py : 將輸入圖像調整成 256×256 的尺寸，作為訓練使用
- data_augmentation.py : 資料增強 (隨機旋轉正負30度)
- test_mask.py : 將遮罩的標準答案繪製在原始圖像中，方便觀察模型分割的結果
- test_model.py : 測試各模型的評估指標
- test_picture.py : 測試單張圖的分割結果，可與標準答案進行比對
- Area.py : 計算剝落區域面積


## 資料夾說明
- data : 研究中所使用的資料集，包含:訓練集、驗證集以及測試集
- params : 存放訓練好的模型權重
- NET : 各模型的程式碼，用於訓練引入，進行訓練
- CBAM : 各模型加入CBAM後的程式碼


## 使用說明
1. 執行train.py 資料集路徑要設對，要引入所要訓練的模型
2. 研究中訓練回合數為200個epoch，batch size為4，學習率為0.001，優化器為Adam，可以自行更改
3. 訓練完成後會產生該模型的訓練損失圖與訓練損失值，在params裡會有該模型的權重，可使用進行測試
4. 使用 test_model.py 與 test_picture.py 進行測試

***
## 環境與套件需求 
- python 
- pytorch 
- matplotlib.pyplot
- time
- tqdm
- random
- opencv
- numpy
- sklearn.metrics


