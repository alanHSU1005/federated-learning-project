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
```

---

## ✨ 主要功能 (Features)

### 🏗️ 模組化聯邦學習核心系統
- **LeNet-5 全局模型**：自動依資料集類別數調整輸出層，無縫支援 MNIST（10 類）與 AT&T Face（40 類）
- **FedAvg 加權聚合**：依各客戶端本地樣本數進行加權平均，確保聚合公平性

### 📊 Non-IID 資料切分
- **Dirichlet-based 分配**：以 `α` 參數控制 label skew 程度
- **Sample-level disjoint partition**：每筆樣本保證只分配給一個客戶端，無重複
- **內建驗證機制**：切分完成後自動執行 assert 驗證 union 完整性與跨客戶端不重疊性
- **Edge case 保護**：若極端 α 值導致空客戶端，自動執行最小量 rebalancing 確保不 crash

### 🧪 雙組基準實驗 (Baseline Experiments)
- **實驗一 — Local Epochs 影響**：固定 Batch Size，依序測試不同 E 值對全局模型準確率的影響
- **實驗二 — Batch Size 影響**：固定 Local Epochs，依序測試不同 B 值對全局模型準確率的影響
- **`full` 模式**：在 `config.py` 設定後可一鍵連續執行實驗一與實驗二，無需手動介入

### 📈 自動化圖表生成
- **Accuracy vs. Rounds 折線圖**：各變因值的曲線同圖對比，圖例標注最佳準確率
- **Loss vs. Rounds 折線圖**：對應的損失收斂曲線
- **Train vs. Test 對比圖**：用於觀察過擬合 / 欠擬合現象
- **最佳準確率長條比較圖**：實驗一與實驗二結果並排，適合直接放入書面報告
- **Dataset 完全隔離輸出**：MNIST 與 AT&T Face 圖表分別存入獨立子資料夾，不混圖

### 🔌 後續模組整合介面
- **梯度洩漏攻擊介面（Step 2）**：`Client` 物件保留訓練前後權重快照（`pre/post_train_weights`），並提供 `get_pseudo_gradients()` 與 `compute_gradients_on_batch()` 方法
- **防禦機制掛載介面（Step 3）**：`server.run_round()` 接受 `gradient_hook`（梯度後處理）與 `aggregation_fn`（自訂聚合函數）參數，直接傳入即可，無需修改核心程式碼

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

### ▶️ 執行 `main.py` 後會發生什麼事？

**階段 1 — 隨機種子固定**
在每次實驗開始前統一設定全域隨機種子（Python / NumPy / PyTorch），確保 Non-IID 資料切分與模型初始化在不同參數組之間完全可重現。

**階段 2 — 資料準備**
依 `config.DATASET` 載入資料集（MNIST 自動下載；AT&T 從本地路徑讀取），接著以 Dirichlet(α) 進行 sample-level Non-IID 切分，為每個客戶端建立專屬 DataLoader。

**階段 3 — FL 訓練主迴圈**
建立 Server 與 10 個 Client 後，執行 `GLOBAL_ROUNDS` 輪通訊。每輪流程為：
```
Server 廣播全局權重
  → 各 Client 以本地資料訓練 E 個 Epoch
  → Server 收集更新，執行 FedAvg 加權聚合
  → 全局模型在測試集上評估 Accuracy 與 Loss
  → 記錄本輪結果至 history
```
每輪結束後同步列印訓練損失、測試準確率與預估剩餘時間。

**階段 4 — 完整評估與日誌儲存**
整輪訓練結束後，對最終全局模型執行完整測試集評估（含各類別個別準確率），並將完整 history 儲存為 `logs/` 下的 JSON 檔案。

**階段 5 — 自動圖表生成**
所有實驗組的訓練完成後，自動呼叫 `visualization.py` 讀取 `logs/` 內全部 JSON，依 dataset → experiment 雙層分組，於 `plots/<dataset>/` 下輸出完整圖表集。

### 3. 生成效能數據視覺化圖表

python visualization.py

---

## 📊 輸出檔案說明 (Outputs)

執行 `main.py` 或 `visualization.py` 後，會在專案根目錄下自動產生以下輸出：

### 📁 `logs/` — 實驗日誌

每次完整 FL 訓練結束後輸出一個 JSON 檔案，命名規則為：

```
logs/
├── exp1_MNIST_E1_B64.json       # 實驗一，E=1
├── exp1_MNIST_E3_B64.json       # 實驗一，E=3
├── exp1_MNIST_E5_B64.json       # 實驗一，E=5
├── exp2_MNIST_E3_B16.json       # 實驗二，B=16
├── exp2_MNIST_E3_B64.json       # 實驗二，B=64
├── exp2_MNIST_E3_B256.json      # 實驗二，B=256
└── ...                          # AT&T 實驗同理
```

每個 JSON 的內部結構如下：

```json
{
  "experiment":        "exp1_local_epochs",
  "dataset":           "MNIST",
  "variable_key":      "local_epochs",
  "variable_value":    3,
  "global_rounds":     20,
  "local_epochs":      3,
  "batch_size":        64,
  "best_test_accuracy": 0.9823,
  "best_round":        18,
  "history": [
    {
      "round":             1,
      "test_accuracy":     0.4712,
      "test_accuracy_pct": 47.12,
      "test_loss":         1.6034,
      "avg_train_loss":    1.8821,
      "avg_train_acc":     0.4103
    },
    ...
  ]
}
```

### 📁 `plots/` — 視覺化圖表

圖表依資料集分資料夾存放，MNIST 與 AT&T Face 

```
plots/
├── MNIST/
│   ├── exp1_accuracy_curves.png         # 實驗一：不同 E 的測試準確率折線對比
│   ├── exp1_loss_curves.png             # 實驗一：不同 E 的測試損失折線對比
│   ├── exp1_train_vs_test_E3.png        # 實驗一（中間 E 值）：訓練 vs 測試準確率
│   ├── exp2_accuracy_curves.png         # 實驗二：不同 B 的測試準確率折線對比
│   ├── exp2_loss_curves.png             # 實驗二：不同 B 的測試損失折線對比
│   ├── exp2_train_vs_test_B64.png       # 實驗二（中間 B 值）：訓練 vs 測試準確率
│   └── best_accuracy_comparison.png     # 實驗一 vs 實驗二最佳準確率長條比較圖
└── ATT_FACE/
    └── ...                              # 同上
```

| 圖表檔案 | 說明 |
|---|---|
| `exp1_accuracy_curves.png` | E=1 / 3 / 5 三條曲線同圖，圖例標注各自最佳準確率，適合觀察 local epochs 對收斂速度的影響 |
| `exp1_loss_curves.png` | 對應的測試損失下降曲線 |
| `exp2_accuracy_curves.png` | 不同 B 值的準確率曲線對比，適合觀察 batch size 對穩定性的影響 |
| `exp2_loss_curves.png` | 對應的測試損失下降曲線 |
| `*_train_vs_test_*.png` | 訓練準確率（紅虛線）與測試準確率（藍實線）同圖，用於判斷過擬合 |
| `best_accuracy_comparison.png` | 左欄為實驗一各 E 值、右欄為實驗二各 B 值的最佳準確率長條圖，長條頂端標注數值，可直接放入報告 |

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


