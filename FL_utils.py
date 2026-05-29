# ==============================================================================
# FL_utils.py — 聯邦學習輔助工具函數庫
#
# 此模組收納所有跨模組共用的工具函數，包含：
#   1. 隨機種子設定（確保實驗可重現）
#   2. 模型權重操作（深複製、差值計算、範數計算）
#   3. 日誌管理（儲存 / 讀取 JSON 格式的實驗記錄）
#   4. 訓練進度格式化輸出
#   5. 實驗結果彙整（供 visualization.py 讀取）
# ==============================================================================

import os
import json
import copy
import time
import random
import numpy as np
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn

import config


# ==============================================================================
# 1. 隨機種子管理
# ==============================================================================

def set_seed(seed: int = config.SEED) -> None:
    """
    設定全域隨機種子，確保實驗結果可重現。

    同時設定 Python、NumPy、PyTorch（CPU + GPU）的種子，
    並停用 cuDNN 的非確定性演算法。

    Args:
        seed (int): 隨機種子值，預設從 config.SEED 讀取。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # 多 GPU 環境
    torch.backends.cudnn.deterministic = True  # 確保 cuDNN 使用確定性演算法
    torch.backends.cudnn.benchmark = False     # 停用自動尋找最佳卷積演算法（會引入隨機性）
    print(f"[FL_utils] 隨機種子已設定為：{seed}")


# ==============================================================================
# 2. 模型權重操作工具
# ==============================================================================

def compute_weight_diff(weights_before: dict, weights_after: dict) -> dict:
    """
    計算兩個模型權重字典的差值（before − after）。

    常用於估算「虛擬梯度」：
        pseudo_gradient ≈ weights_before − weights_after

    【攻擊模組（Step 2）可直接使用此函數】

    Args:
        weights_before (dict): 訓練前的模型 state_dict。
        weights_after (dict): 訓練後的模型 state_dict。

    Returns:
        dict: 各層權重差值字典，格式與 state_dict 相同。
    """
    diff = {}
    for key in weights_before:
        diff[key] = weights_before[key].float() - weights_after[key].float()
    return diff


def compute_gradient_norm(gradients: dict, norm_type: float = 2.0) -> float:
    """
    計算梯度字典的全域 L-p 範數。

    可用於：
      - 監控訓練是否發生梯度爆炸
      - 防禦模組（Step 3）的梯度裁剪閾值設定參考

    Args:
        gradients (dict): 梯度字典（key: 層名, value: 梯度 Tensor）。
        norm_type (float): 範數類型，預設為 L2 範數（2.0）。

    Returns:
        float: 所有層梯度的合併 L-p 範數。
    """
    all_norms = [
        grad.float().norm(norm_type)
        for grad in gradients.values()
        if grad is not None
    ]
    if not all_norms:
        return 0.0
    total_norm = torch.stack(all_norms).norm(norm_type).item()
    return total_norm


def average_weights(weights_list: List[dict], sample_counts: Optional[List[int]] = None) -> dict:
    """
    對多個模型權重字典執行（加權）平均，回傳聚合後的新權重字典。

    若不提供 sample_counts，則執行簡單算術平均；
    若提供，則依樣本數執行加權平均（等同 FedAvg 核心邏輯）。

    此函數為 server.py 的 aggregate_fedavg() 提供底層支援，
    也可供防禦模組（Robust Aggregation）直接呼叫。

    Args:
        weights_list (list[dict]): 多個模型的 state_dict 列表。
        sample_counts (list[int], optional): 對應每個模型的樣本數，用於加權。

    Returns:
        dict: 聚合後的模型權重字典。
    """
    if not weights_list:
        raise ValueError("weights_list 不能為空。")

    # 若未提供樣本數，預設為等權（均等平均）
    if sample_counts is None:
        sample_counts = [1] * len(weights_list)

    total = sum(sample_counts)
    aggregated = {key: torch.zeros_like(val) for key, val in weights_list[0].items()}

    for weights, count in zip(weights_list, sample_counts):
        factor = count / total
        for key in aggregated:
            aggregated[key] += factor * weights[key].float()

    return aggregated


def clone_model_weights(model: nn.Module) -> dict:
    """
    深複製模型的 state_dict，確保與原模型完全記憶體隔離。

    Args:
        model (nn.Module): 來源模型。

    Returns:
        dict: 深複製的 state_dict。
    """
    return copy.deepcopy(model.state_dict())


# ==============================================================================
# 3. 日誌管理（JSON 格式）
# ==============================================================================

def ensure_dir(dir_path: str) -> None:
    """
    確保目錄存在，若不存在則建立（含所有中間目錄）。

    Args:
        dir_path (str): 目標目錄路徑。
    """
    os.makedirs(dir_path, exist_ok=True)


def save_experiment_log(log_data: dict, filename: str, log_dir: str = config.LOG_DIR) -> str:
    """
    將實驗日誌以 JSON 格式儲存至指定目錄。

    日誌結構建議（main.py 負責組裝）：
        {
            "experiment"  : "exp1_local_epochs",
            "dataset"     : "MNIST",
            "variable"    : "local_epochs",
            "value"       : 3,
            "global_rounds": 50,
            "batch_size"  : 32,
            "history"     : [
                {"round": 1, "test_accuracy": 0.12, "test_loss": 2.31, ...},
                {"round": 2, "test_accuracy": 0.25, "test_loss": 1.87, ...},
                ...
            ]
        }

    Args:
        log_data (dict): 要儲存的日誌資料。
        filename (str): 輸出檔名（不含路徑），例如 'exp1_E3.json'。
        log_dir (str): 目標目錄，預設從 config.LOG_DIR 讀取。

    Returns:
        str: 儲存完成的完整檔案路徑。
    """
    ensure_dir(log_dir)
    filepath = os.path.join(log_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"[FL_utils] 日誌已儲存至：{filepath}")
    return filepath


def load_experiment_log(filename: str, log_dir: str = config.LOG_DIR) -> dict:
    """
    從 JSON 檔案讀取實驗日誌。

    visualization.py 呼叫此函數讀取日誌後繪製圖表。

    Args:
        filename (str): 日誌檔名（不含路徑）。
        log_dir (str): 目標目錄，預設從 config.LOG_DIR 讀取。

    Returns:
        dict: 解析完成的日誌資料字典。

    Raises:
        FileNotFoundError: 若指定路徑的檔案不存在。
    """
    filepath = os.path.join(log_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到日誌檔案：{filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_all_logs_in_dir(log_dir: str = config.LOG_DIR) -> List[dict]:
    """
    批次讀取指定目錄下所有 JSON 日誌檔案。

    visualization.py 可呼叫此函數一次性載入所有實驗記錄。

    Args:
        log_dir (str): 日誌目錄路徑。

    Returns:
        list[dict]: 所有日誌資料的列表（按檔名排序）。
    """
    ensure_dir(log_dir)
    log_files = sorted([f for f in os.listdir(log_dir) if f.endswith('.json')])
    all_logs = []
    for fname in log_files:
        log = load_experiment_log(fname, log_dir=log_dir)
        all_logs.append(log)
        print(f"[FL_utils] 已載入日誌：{fname}")
    return all_logs


# ==============================================================================
# 4. 訓練進度格式化輸出
# ==============================================================================

class RoundTimer:
    """
    每輪通訊的計時器，用於記錄與顯示訓練耗時。

    使用方式：
        timer = RoundTimer(total_rounds=50)
        timer.start()
        # ... 執行一輪訓練 ...
        timer.lap(round_num=1)
    """

    def __init__(self, total_rounds: int):
        self.total_rounds = total_rounds
        self._start_time: Optional[float] = None
        self._lap_time: Optional[float] = None

    def start(self) -> None:
        """開始計時（整個訓練開始前呼叫）。"""
        self._start_time = time.time()
        self._lap_time = self._start_time

    def lap(self, round_num: int) -> float:
        """
        記錄當前輪次的耗時並列印進度。

        Args:
            round_num (int): 當前輪次編號。

        Returns:
            float: 本輪耗時（秒）。
        """
        now = time.time()
        round_elapsed = now - self._lap_time
        total_elapsed = now - self._start_time
        estimated_total = (total_elapsed / round_num) * self.total_rounds
        remaining = estimated_total - total_elapsed

        print(
            f"    ⏱  本輪耗時：{round_elapsed:.1f}s | "
            f"已訓練：{format_time(total_elapsed)} | "
            f"預估剩餘：{format_time(remaining)}"
        )
        self._lap_time = now
        return round_elapsed


def format_time(seconds: float) -> str:
    """
    將秒數格式化為 mm:ss 或 hh:mm:ss 的字串。

    Args:
        seconds (float): 秒數。

    Returns:
        str: 格式化後的時間字串。
    """
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60:02d}:{seconds % 60:02d}"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"


def print_round_header(round_num: int, total_rounds: int, local_epochs: int, batch_size: int) -> None:
    """
    列印每輪訓練開始時的標題資訊。

    Args:
        round_num (int): 當前輪次。
        total_rounds (int): 總輪次。
        local_epochs (int): 本輪 local epochs。
        batch_size (int): 本輪 batch size。
    """
    print(f"\n{'─'*60}")
    print(f"  Round {round_num:3d} / {total_rounds}  |  E={local_epochs}  |  B={batch_size}")
    print(f"{'─'*60}")


def print_experiment_header(exp_mode: str, dataset: str, variable_name: str, value) -> None:
    """
    列印實驗開始時的標題橫幅。

    Args:
        exp_mode (str): 實驗模式識別字串。
        dataset (str): 資料集名稱。
        variable_name (str): 本次實驗的變因名稱（如 'Local Epochs'）。
        value: 本次實驗的變因值（如 3）。
    """
    print(f"\n{'='*60}")
    print(f"  實驗模式：{exp_mode}")
    print(f"  資料集  ：{dataset}")
    print(f"  變因    ：{variable_name} = {value}")
    print(f"{'='*60}")


# ==============================================================================
# 5. 實驗結果彙整工具（供 main.py 組裝日誌）
# ==============================================================================

def build_round_history_entry(
    round_num: int,
    test_accuracy: float,
    test_loss: float,
    avg_train_loss: float,
    avg_train_acc: float,
) -> dict:
    """
    建立單輪訓練的歷史記錄條目。

    main.py 在每輪評估後呼叫此函數，
    將結果加入 history 列表，最終整批儲存為 JSON 日誌。

    Args:
        round_num (int): 當前輪次。
        test_accuracy (float): 本輪全局模型在測試集上的準確率（0.0 ~ 1.0）。
        test_loss (float): 本輪全局模型在測試集上的平均損失。
        avg_train_loss (float): 本輪所有客戶端的平均訓練損失。
        avg_train_acc (float): 本輪所有客戶端的平均訓練準確率。

    Returns:
        dict: 格式化的單輪記錄字典。
    """
    return {
        'round': round_num,
        'test_accuracy': round(test_accuracy, 6),
        'test_accuracy_pct': round(test_accuracy * 100.0, 4),
        'test_loss': round(test_loss, 6),
        'avg_train_loss': round(avg_train_loss, 6),
        'avg_train_acc': round(avg_train_acc, 6),
    }


def build_experiment_log(
    experiment_mode: str,
    dataset: str,
    variable_key: str,
    variable_value,
    global_rounds: int,
    local_epochs: int,
    batch_size: int,
    history: List[dict],
) -> dict:
    """
    建立完整的實驗日誌字典（包含 metadata 與 history）。

    main.py 在一次完整 FL 訓練結束後呼叫此函數，
    產出的字典傳入 save_experiment_log() 儲存為 JSON。

    Args:
        experiment_mode (str): 實驗模式（'exp1_local_epochs' / 'exp2_batch_size'）。
        dataset (str): 資料集名稱（'MNIST' / 'ATT_FACE'）。
        variable_key (str): 變因的鍵名（'local_epochs' / 'batch_size'）。
        variable_value: 本次實驗的變因值。
        global_rounds (int): 總通訊輪次。
        local_epochs (int): 本次 local epochs 設定。
        batch_size (int): 本次 batch size 設定。
        history (list[dict]): 每輪的記錄條目列表（由 build_round_history_entry 產生）。

    Returns:
        dict: 完整的實驗日誌字典，可直接傳入 save_experiment_log()。
    """
    return {
        'experiment': experiment_mode,
        'dataset': dataset,
        'variable_key': variable_key,
        'variable_value': variable_value,
        'global_rounds': global_rounds,
        'local_epochs': local_epochs,
        'batch_size': batch_size,
        'best_test_accuracy': max(entry['test_accuracy'] for entry in history) if history else 0.0,
        'best_round': max(history, key=lambda x: x['test_accuracy'])['round'] if history else 0,
        'history': history,
    }


def summarize_experiment_results(logs: List[dict]) -> None:
    """
    對多個實驗日誌的最終結果進行橫向比較，格式化列印摘要表格。

    visualization.py 繪圖前可呼叫此函數在終端顯示快速比較。

    Args:
        logs (list[dict]): 多個實驗的日誌字典列表。
    """
    print(f"\n{'='*60}")
    print(f"  實驗結果摘要")
    print(f"  {'變因':<20} {'最佳準確率':>12} {'達成輪次':>10}")
    print(f"{'─'*60}")
    for log in logs:
        key = log.get('variable_key', 'N/A')
        val = log.get('variable_value', 'N/A')
        best_acc = log.get('best_test_accuracy', 0.0) * 100
        best_round = log.get('best_round', 0)
        label = f"{key}={val}"
        print(f"  {label:<20} {best_acc:>11.2f}% {best_round:>10}")
    print(f"{'='*60}\n")
