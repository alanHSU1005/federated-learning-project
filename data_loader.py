# ==============================================================================
# data_loader.py — 資料集載入、預處理與 Non-IID 客戶端資料切分
#
# 支援資料集：
#   - MNIST (自動從 torchvision 下載)
#   - AT&T Face Dataset (需手動下載並解壓至 config.ATT_FACE_DATA_DIR)
#
# Non-IID 切分方法：
#   - Dirichlet-based Non-IID Partitioning（FL 研究領域的標準做法）
#   
# ==============================================================================

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

import config


# ==============================================================================
# 自訂 AT&T 人臉資料集
# ==============================================================================

class ATTFaceDataset(Dataset):
    """
    AT&T Face Dataset 自訂讀取器。
    
    資料夾結構預期如下：
        att_faces/
            s1/  (第 1 個人，共 10 張圖)
                1.pgm
                2.pgm
                ...
            s2/
                ...
            s40/
    
    每個人對應一個類別（標籤 0~39）。
    所有影像會統一 Resize 成 28x28 灰階影像以符合 LeNet 輸入規格。
    """

    def __init__(self, root_dir: str, transform=None):
        """
        初始化 AT&T 資料集。

        Args:
            root_dir (str): AT&T 資料集根目錄路徑。
            transform: 影像前處理流程（torchvision.transforms）。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []   # 儲存 (影像路徑, 標籤) 的列表
        self.classes = []   # 儲存所有類別名稱

        # 掃描資料夾，建立樣本列表
        subject_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        for label_idx, subject_dir in enumerate(subject_dirs):
            self.classes.append(subject_dir)
            subject_path = os.path.join(root_dir, subject_dir)
            for img_file in os.listdir(subject_path):
                if img_file.endswith(('.pgm', '.png', '.jpg')):
                    img_path = os.path.join(subject_path, img_file)
                    self.samples.append((img_path, label_idx))

        print(f"[ATTFaceDataset] 載入完成：{len(self.classes)} 個類別，共 {len(self.samples)} 張影像。")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        # 以灰階模式開啟影像（AT&T 原始為 .pgm 格式）
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label


# ==============================================================================
# 資料集載入函數
# ==============================================================================

def get_transforms(dataset_name: str) -> transforms.Compose:
    """
    根據資料集名稱回傳對應的影像前處理流程。

    所有影像統一轉為 28x28 灰階、再正規化至 [-1, 1]，
    以符合 LeNet 的輸入規格。

    Args:
        dataset_name (str): 'MNIST' 或 'ATT_FACE'

    Returns:
        torchvision.transforms.Compose: 組合好的前處理流程
    """
    # MNIST 的均值與標準差（業界標準值）
    if dataset_name == 'MNIST':
        mean, std = (0.1307,), (0.3081,)
    else:
        # AT&T 人臉資料集：使用通用灰階正規化參數
        mean, std = (0.5,), (0.5,)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),     # 統一輸入尺寸為 28x28
        transforms.ToTensor(),           # 轉為 Tensor，值域 [0, 1]
        transforms.Normalize(mean, std), # 正規化至 [-1, 1]
    ])
    return transform


def load_dataset(dataset_name: str):
    """
    載入指定的完整資料集（訓練集 + 測試集）。

    Args:
        dataset_name (str): 'MNIST' 或 'ATT_FACE'

    Returns:
        tuple: (train_dataset, test_dataset)
               兩者均為 torch.utils.data.Dataset 的子類別實例
    """
    transform = get_transforms(dataset_name)

    if dataset_name == 'MNIST':
        # MNIST：自動下載到 ./data 資料夾
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        print(f"[data_loader] MNIST 載入完成："
              f"訓練集 {len(train_dataset)} 筆，測試集 {len(test_dataset)} 筆。")

    elif dataset_name == 'ATT_FACE':
        # AT&T：需要手動下載，從本地路徑讀取
        if not os.path.exists(config.ATT_FACE_DATA_DIR):
            raise FileNotFoundError(
                f"找不到 AT&T 資料集目錄：'{config.ATT_FACE_DATA_DIR}'\n"
                f"請下載 AT&T Face Dataset 並解壓至該路徑後再執行。"
            )
        full_dataset = ATTFaceDataset(
            root_dir=config.ATT_FACE_DATA_DIR, transform=transform
        )
        # AT&T 共 400 張（40人×10張），以 8:2 切分訓練與測試集
        total_size = len(full_dataset)
        train_size = int(total_size * 0.8)
        test_size = total_size - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(config.SEED)
        )
        print(f"[data_loader] AT&T Face 載入完成："
              f"訓練集 {train_size} 筆，測試集 {test_size} 筆。")
    else:
        raise ValueError(f"不支援的資料集名稱：'{dataset_name}'，請選擇 'MNIST' 或 'ATT_FACE'。")

    return train_dataset, test_dataset


# ==============================================================================
# Non-IID 資料切分（核心功能）
# ==============================================================================

def split_non_iid(train_dataset, num_clients: int, classes_per_client: int, seed: int = 42):
    """
    將訓練集以 Dirichlet-based Non-IID 方式切分給各客戶端。

    【演算法說明】
    使用 Dirichlet 分佈控制 label skew，是 FL 研究領域的標準 Non-IID 基準做法。

    核心流程：
        1. 依標籤將所有樣本索引分桶（每個類別一個 pool，先打亂順序）。
        2. 對每個類別，從 Dir(α) 中抽樣，得到該類別樣本在各 client 之間
           的分配比例向量（長度 = num_clients，總和 = 1）。
        3. 依比例將該類別的樣本索引切割並分配給各 client（slice 操作，無複製）。
        4. 驗證 disjoint 性：所有 client 的 union 必須等於完整訓練集，
           且不存在重複索引。
        5. Edge case 處理：若某 client 最終為空，從樣本最多的 client
           借出最少量的樣本，確保每個 client 至少有 1 筆資料，避免訓練 crash。

    α 參數意義（從 config.DIRICHLET_ALPHA 讀取，預設 0.5）：
        α → 0  : 極度 Non-IID，每個 client 幾乎只有單一類別
        α = 0.5: 中度 Non-IID（FL 論文標準設定）
        α → ∞  : 趨近 IID，各 client 類別分佈接近均等

    【保證事項（sample-level disjoint partition）】
        ✅ 每個 sample 索引只出現在恰好一個 client
        ✅ 所有 client 的 union = 完整訓練集（len = 原始 dataset 大小）
        ✅ 不存在跨 client 的 index 重複

    Args:
        train_dataset: 完整訓練資料集（torch.utils.data.Dataset）。
        num_clients (int): 客戶端總數。
        classes_per_client (int): 此參數保留於 function signature 以維持介面相容性，
                                   Dirichlet 模式下由 α 控制 label skew 程度，
                                   不以固定類別數硬切分。
        seed (int): 隨機種子，確保可重現性。

    Returns:
        list[list[int]]: 長度為 num_clients 的列表，
                         每個元素是該 client 所分配到的樣本「索引」列表（無重複）。
    """
    random.seed(seed)
    np.random.seed(seed)

    # 從 config 讀取 Dirichlet α 參數，若未設定則預設 0.5
    alpha = getattr(config, 'DIRICHLET_ALPHA', 0.5)

    # -------------------------------------------------------------------------
    # 步驟 1：取得每個樣本的標籤（相容 MNIST 與 ATT_FACE 兩種格式）
    # -------------------------------------------------------------------------
    if hasattr(train_dataset, 'targets'):
        # torchvision 標準 Dataset（如 MNIST）：直接有 .targets 屬性
        all_labels = np.array(train_dataset.targets)
    elif hasattr(train_dataset, 'dataset'):
        # random_split 產生的 Subset（AT&T 使用）：需透過原始 dataset 取得標籤
        original_dataset = train_dataset.dataset
        indices = train_dataset.indices
        if hasattr(original_dataset, 'samples'):
            # ATTFaceDataset：從 samples 列表取標籤
            all_labels_full = np.array([s[1] for s in original_dataset.samples])
        else:
            all_labels_full = np.array(original_dataset.targets)
        all_labels = all_labels_full[indices]
    else:
        raise AttributeError("無法從資料集中取得標籤，請確認資料集格式（需有 .targets 或 .dataset.samples）。")

    total_samples = len(all_labels)
    num_classes = len(np.unique(all_labels))

    print(f"\n[Non-IID / Dirichlet] 開始切分：α={alpha}，"
          f"{num_classes} 類，{total_samples} 筆樣本 → {num_clients} 個 client")

    # -------------------------------------------------------------------------
    # 步驟 2：依標籤建立索引分桶，並對每個類別的索引進行隨機打亂
    #         這是確保後續 slice 分配公平的關鍵步驟
    # -------------------------------------------------------------------------
    label_to_indices = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(all_labels):
        label_to_indices[int(label)].append(idx)

    for c in range(num_classes):
        np.random.shuffle(label_to_indices[c])  # 就地打亂，確保切割點隨機

    # -------------------------------------------------------------------------
    # 步驟 3：Dirichlet 分配
    #
    # 對每個類別 c，從 Dir(α * 1_K) 抽樣比例向量 proportions（長度 K = num_clients），
    # 依比例將該類別的 pool 切成 K 個不重疊 slice，分別給 K 個 client。
    #
    # 用 np.split 按 cumsum 切割，保證：
    #   - 同一 class 的 pool 不重複使用（slice 不 overlap）
    #   - 每個 sample 只進入一個 client 的 bucket
    # -------------------------------------------------------------------------
    # 初始化各 client 的索引容器
    client_data_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        pool = label_to_indices[c]          # 該類別全部樣本索引（已打亂）
        n_c = len(pool)                     # 該類別的總樣本數
        if n_c == 0:
            continue

        # 從 Dirichlet 分佈抽樣分配比例（長度 = num_clients，總和 = 1）
        proportions = np.random.dirichlet(alpha=np.full(num_clients, alpha))

        # 將比例轉換為各 client 應取得的樣本「數量」
        # 使用 floor 後補齊尾差，確保 sum = n_c（不遺漏任何樣本）
        counts = np.floor(proportions * n_c).astype(int)
        remainder = n_c - counts.sum()

        # 將尾差補給比例最大的 remainder 個 client（最常見的補齊策略）
        top_clients = np.argsort(proportions)[::-1][:remainder]
        counts[top_clients] += 1

        # 依累積切割點將 pool 切分為 num_clients 個不重疊 slice
        split_points = np.cumsum(counts)[:-1]   # 長度 = num_clients - 1
        slices = np.split(pool, split_points)   # list of arrays，長度 = num_clients

        # 將每個 slice 加入對應 client 的索引列表
        for client_id, s in enumerate(slices):
            client_data_indices[client_id].extend(s.tolist())

    # -------------------------------------------------------------------------
    # 步驟 4：Edge case 處理 — 確保沒有空 client
    #
    # 在極端 α（接近 0）時，少數 client 可能因 Dirichlet 抽樣結果全為 0
    # 而得到空的 dataset，導致後續 DataLoader 初始化失敗。
    # 策略：從樣本數最多的 client 借出 1 筆樣本給空 client，
    #        最小程度干擾分配結果。
    # -------------------------------------------------------------------------
    empty_clients = [i for i, idx in enumerate(client_data_indices) if len(idx) == 0]
    if empty_clients:
        print(f"  [Non-IID] ⚠ 發現 {len(empty_clients)} 個空 client，執行 minimal rebalancing...")
        for empty_cid in empty_clients:
            # 找出樣本數最多的 client
            donor_cid = max(range(num_clients), key=lambda i: len(client_data_indices[i]))
            if len(client_data_indices[donor_cid]) <= 1:
                # 極端情況：donor 也只有 1 筆，無法再借出，直接警告並跳過
                print(f"  [Non-IID] ⚠ 無法從 client {donor_cid} 借出樣本（僅剩 1 筆），"
                      f"client {empty_cid} 仍為空。請增大 α 或減少 num_clients。")
                continue
            # 借出最後 1 筆（取 pop 避免重複，維持 disjoint）
            donated_idx = client_data_indices[donor_cid].pop()
            client_data_indices[empty_cid].append(donated_idx)
            print(f"  [Non-IID]   client {donor_cid} → client {empty_cid}：借出索引 {donated_idx}")

    # -------------------------------------------------------------------------
    # 步驟 5：驗證 disjoint partition 正確性（debug 用斷言）
    # -------------------------------------------------------------------------
    all_assigned = []
    for idx_list in client_data_indices:
        all_assigned.extend(idx_list)

    assert len(all_assigned) == total_samples, (
        f"[Non-IID] 分配後總樣本數 {len(all_assigned)} ≠ 原始訓練集大小 {total_samples}，"
        f"存在樣本遺漏或重複！"
    )
    assert len(set(all_assigned)) == total_samples, (
        f"[Non-IID] 發現跨 client 的重複樣本索引（disjoint 條件違反）！"
        f"唯一索引數：{len(set(all_assigned))}，預期：{total_samples}"
    )

    # -------------------------------------------------------------------------
    # 步驟 6：列印各 client 的分配統計（顯示前 3 個主要類別）
    # -------------------------------------------------------------------------
    print(f"\n  {'Client':<10} {'樣本數':>8}    {'主要類別分佈（前3）'}")
    print(f"  {'─'*55}")
    for cid, idx_list in enumerate(client_data_indices):
        n = len(idx_list)
        if n == 0:
            print(f"  client {cid:02d}  {n:>8}    （空）")
            continue
        # 統計此 client 各類別的樣本數
        client_labels = all_labels[idx_list]
        unique, counts = np.unique(client_labels, return_counts=True)
        top3 = sorted(zip(unique, counts), key=lambda x: -x[1])[:3]
        top3_str = ', '.join([f"cls{c}:{cnt}" for c, cnt in top3])
        print(f"  client {cid:02d}  {n:>8}    {top3_str}")

    print(f"\n  ✅ 驗證通過：{total_samples} 筆樣本已 disjoint 分配至 {num_clients} 個 client")
    return client_data_indices


# ==============================================================================
# DataLoader 建立函數（供 client.py 呼叫）
# ==============================================================================

def get_client_dataloader(train_dataset, client_indices: list, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    根據客戶端的樣本索引，建立對應的 DataLoader。

    此函數是 client.py 的主要介面，每次本地訓練前呼叫以取得資料。

    Args:
        train_dataset: 完整訓練資料集
        client_indices (list[int]): 該客戶端所擁有的樣本索引
        batch_size (int): 批次大小（實驗二的變因）
        shuffle (bool): 是否在每個 epoch 開始時打亂資料順序

    Returns:
        DataLoader: 該客戶端專屬的資料載入器
    """
    client_subset = Subset(train_dataset, client_indices)
    dataloader = DataLoader(
        client_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,       # 設為 0 以避免多進程問題（跨平台相容）
        pin_memory=(config.DEVICE.type == 'cuda'),  # 使用 GPU 時啟用以加速資料傳輸
        drop_last=False,     # 保留最後一個不完整的批次
    )
    return dataloader


def get_test_dataloader(test_dataset, batch_size: int = 256) -> DataLoader:
    """
    建立全局測試集的 DataLoader（供 evaluate.py 使用）。

    Args:
        test_dataset: 完整測試資料集
        batch_size (int): 測試時的批次大小（影響速度，不影響結果）

    Returns:
        DataLoader: 測試集資料載入器
    """
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,       # 測試集不需打亂
        num_workers=0,
        pin_memory=(config.DEVICE.type == 'cuda'),
    )
    return dataloader


# ==============================================================================
# 主入口：一次性完成所有資料準備（供 main.py 呼叫）
# ==============================================================================

def prepare_data(dataset_name: str, num_clients: int, classes_per_client: int,
                 batch_size: int, seed: int = 42):
    """
    整合資料準備流程的高層函數。

    依序執行：載入資料集 → Non-IID 切分 → 建立各客戶端 DataLoader。

    Args:
        dataset_name (str): 'MNIST' 或 'ATT_FACE'
        num_clients (int): 客戶端總數
        classes_per_client (int): 每個客戶端的類別數
        batch_size (int): 訓練批次大小
        seed (int): 隨機種子

    Returns:
        tuple:
            - client_loaders (list[DataLoader]): 各客戶端的訓練 DataLoader
            - test_loader (DataLoader): 全局測試 DataLoader
            - client_indices (list[list[int]]): 各客戶端樣本索引（供攻擊模組使用）
    """
    print(f"\n{'='*60}")
    print(f"[data_loader] 準備資料集：{dataset_name}")
    print(f"  客戶端數量：{num_clients}，每客戶端類別數：{classes_per_client}，批次大小：{batch_size}")
    print(f"{'='*60}")

    # 1. 載入完整資料集
    train_dataset, test_dataset = load_dataset(dataset_name)

    # 2. Non-IID 資料切分
    print("\n[Non-IID] 開始切分資料...")
    client_indices = split_non_iid(
        train_dataset, num_clients, classes_per_client, seed=seed
    )

    # 3. 建立各客戶端的 DataLoader
    client_loaders = [
        get_client_dataloader(train_dataset, client_indices[i], batch_size)
        for i in range(num_clients)
    ]

    # 4. 建立全局測試 DataLoader
    test_loader = get_test_dataloader(test_dataset)

    print(f"\n[data_loader] 資料準備完成！")
    print(f"  測試集批次數：{len(test_loader)}")
    return client_loaders, test_loader, client_indices
