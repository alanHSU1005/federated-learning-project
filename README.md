# 聯邦學習系統與基準效能實驗 (Federated Learning Baseline System)

本專案建立了一個聯邦學習 (FL) 系統，用於在純淨環境下，量化評估「本地訓練週期 」與「批次大小」對全局模型準確度與收斂速度的基準影響。

---

## 📂 專案架構與檔案說明

專案採用扁平化目錄設計，所有核心腳本皆位於根目錄下，職責切分如下：

```text
├── config.py           # 全域參數配置中心（資料集切換、實驗變因控制、超參數設定）
├── data_loader.py      # 資料下載與預處理（MNIST/AT&T 縮放、Non-IID 分割）
├── model.py            # 定義標準 LeNet-5 網路架構，支援動態輸出類別數
├── client.py           # 定義 Client 物件，包含本地訓練並預留「權重/梯度」攔截接口
├── server.py           # 定義 Server 物件，負責客戶端管理與 FedAvg 權重聚合
├── Train.py            # 實作 Client 本地 Epoch 訓練的具體 Loop 邏輯
├── evaluate.py         # 全局模型在測試集上的最高 Accuracy 與 Loss 評估
├── FL_utils.py         # 聯邦學習輔助工具函數（日誌寫入、目錄建立等）
├── visualization.py    # 讀取 JSON 日誌，自動繪製基準實驗一與實驗二之效能對比折線圖
├── main.py             # 專案主入口，自動讀取 config.py 執行對應的聯邦學習訓練流程
├── requirements.txt    # 專案套件依賴表
├── logs/               # [自動生成] 存放各項實驗結果的 JSON 日誌檔案
└── plots/              # [自動生成] 存放視覺化分析圖表（.png）

---

## 🛠️ 環境安裝指南

### 安裝基礎依賴套件

pip install -r requirements.txt

## 🚀 實驗執行與運作說明書

### 1. 資料集事前準備

MNIST：不需手動下載。啟動程式時，系統會自動下載並快取至本地。
AT&T Face Dataset：確保 40 個人的資料夾（s1/ 至 s40/）直接存放於根目錄下的 ./att_faces/ 路徑中。
若不存在，請請自行下載並解壓。

### 2. 執行基準實驗

打開 config.py，可調整實驗參數與實驗腳本

終端機啟動聯邦學習訓練:

python main.py

### 3. 生成效能數據視覺化圖表

python visualization.py

---

# 補充 

## 基線實驗設定

⚠️：以下設定為**實驗參考用 baseline**，實際實驗數據可依模型、資源與收斂情況進行調整

### MNIST

- 客戶端數量（Number of clients）：10  
- 全局通訊輪次（Global rounds）：20  
- 客戶端參與比例（C / Fraction Fit）：0.5  
- 學習率（Learning rate）：0.001  
- Dirichlet α（Non-IID 程度）：0.5  

#### 實驗一：Local Epochs 影響

- Local epochs 變因：`[1, 3, 5]`  
- 固定 Batch size：64  

#### 實驗二：Batch Size 影響

- Batch size 變因：`[16, 64, 256]`  
- 固定 Local epochs：3  

## ATT_FACE

- 客戶端數量（Number of clients）：10  
- 全局通訊輪次（Global rounds）：120  
- 客戶端參與比例（C / Fraction Fit）：1.0  
- 學習率（Learning rate）：0.001  
- Dirichlet α（Non-IID 程度）：1.0  
- Local epochs：3  
- Batch size：32  

### 實驗一：Local Epochs 影響
- Local epochs 變因：`[1, 3, 5]`  
- 固定 Batch size：32  

### 🧪 實驗二：Batch Size 影響
- Batch size 變因：`[16, 32, 64]`  
- 固定 Local epochs：3  


