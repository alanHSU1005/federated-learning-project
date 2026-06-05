# ==============================================================================
# server.py — 聯邦學習伺服器端定義（FedAvg 聚合）
#
# Server 物件負責：
#   1. 維護全局模型（Global Model）
#   2. 廣播全局模型權重給所有選中的客戶端
#   3. 收集客戶端回傳的本地更新
#   4. 執行 FedAvg（加權平均）完成全局模型聚合
#   5. 記錄每輪的聚合統計（供 visualization.py 使用）
#
# 【FedAvg 公式】
#   w_global ← Σ (n_k / N) × w_k
#   其中 n_k 為第 k 個客戶端的本地樣本數，N = Σ n_k
#
# 【對防禦模組（Step 3）的介面說明】
#   Server 端防禦（如 Secure Aggregation、Robust Aggregation）可透過
#   覆寫或傳入自訂 aggregation_fn 來替換預設的 FedAvg 邏輯。
# ==============================================================================

import copy
import random
from typing import Callable, List, Optional

import torch
import torch.nn as nn

import config
from client import Client
from model import LeNet, get_model, get_model_weights, set_model_weights


# ==============================================================================
# Server 類別
# ==============================================================================

class Server:
    """
    聯邦學習伺服器端，負責協調所有客戶端完成全局模型訓練。

    Attributes:
        global_model (LeNet): 全局模型（聚合後的最新版本）。
        clients (list[Client]): 所有已註冊的客戶端列表。
        current_round (int): 當前的全局通訊輪次（從 1 開始計數）。
        round_logs (list[dict]): 每輪的訓練統計日誌（供 visualization.py 使用）。
    """

    def __init__(self, num_classes: int = None):
        """
        初始化 Server，建立全局模型。

        Args:
            num_classes (int, optional): 輸出類別數，預設從 config 讀取。
        """
        self.global_model: LeNet = get_model(num_classes=num_classes)
        self.clients: List[Client] = []
        self.current_round: int = 0
        self.round_logs: List[dict] = []  # 每輪統計，供 visualization.py 讀取

        print(f"\n[Server] 初始化完成，全局模型已建立於 {config.DEVICE}。")

    # --------------------------------------------------------------------------
    # 客戶端管理
    # --------------------------------------------------------------------------

    def register_clients(self, clients: List[Client]) -> None:
        """
        將已建立的客戶端列表註冊到 Server。

        Args:
            clients (list[Client]): 由 main.py 建立的客戶端列表。
        """
        self.clients = clients
        print(f"[Server] 已註冊 {len(self.clients)} 個客戶端。")

    def select_clients(self, fraction: float = config.FRACTION_FIT) -> List[Client]:
        """
        從所有客戶端中隨機選取參與本輪訓練的子集。

        Args:
            fraction (float): 參與比例（0.0 ~ 1.0）。預設 1.0 即全部參與。

        Returns:
            list[Client]: 本輪被選中參與訓練的客戶端列表。
        """
        num_selected = max(1, int(len(self.clients) * fraction))
        selected = random.sample(self.clients, num_selected)
        return selected

    # --------------------------------------------------------------------------
    # 核心方法：廣播全局權重
    # --------------------------------------------------------------------------

    def broadcast_global_weights(self, clients: List[Client]) -> None:
        """
        將當前全局模型的權重廣播給指定的客戶端列表。

        每個客戶端會獨立收到一份全局權重的「深複製」，
        確保各客戶端的本地訓練互不干擾。

        Args:
            clients (list[Client]): 要廣播的目標客戶端列表。
        """
        global_weights = get_model_weights(self.global_model)
        for client in clients:
            client.receive_global_weights(copy.deepcopy(global_weights))

    # --------------------------------------------------------------------------
    # 核心方法：FedAvg 聚合
    # --------------------------------------------------------------------------

    def aggregate_fedavg(self, client_updates: List[dict]) -> None:
        """
        執行 FedAvg（Federated Averaging）全局模型聚合。

        聚合公式：
            w_global ← Σ_k (n_k / N) × w_k
            N = Σ_k n_k（所有參與客戶端的總樣本數）

        加權依據：每個客戶端的本地樣本數量（num_samples），
        樣本量較大的客戶端對全局模型的影響力也較大。

        Args:
            client_updates (list[dict]): 各客戶端 local_update() 回傳的字典列表，
                格式參見 client.py 的 local_update() 回傳值說明。
        """
        if not client_updates:
            print("[Server] 警告：收到空的客戶端更新列表，跳過本輪聚合。")
            return

        # 計算所有參與客戶端的總樣本數
        total_samples = sum(update['num_samples'] for update in client_updates)

        # 初始化聚合後的全局權重（全部歸零）
        global_weights = get_model_weights(self.global_model)
        aggregated_weights = {key: torch.zeros_like(val) for key, val in global_weights.items()}

        # 加權累加各客戶端的模型權重
        for update in client_updates:
            weight_factor = update['num_samples'] / total_samples
            client_weights = update['weights']
            for key in aggregated_weights:
                aggregated_weights[key] += weight_factor * client_weights[key].float()

        # 將聚合後的權重載入全局模型
        set_model_weights(self.global_model, aggregated_weights)

    # --------------------------------------------------------------------------
    # 核心方法：執行一輪聯邦學習
    # --------------------------------------------------------------------------

    def run_round(
        self,
        local_epochs: int,
        batch_size: int,
        gradient_hook: Optional[Callable[[nn.Module], None]] = None,
        aggregation_fn: Optional[Callable[[List[dict]], None]] = None,
        verbose: bool = True,
    ) -> dict:
        """
        執行完整的一輪聯邦學習（廣播 → 本地訓練 → 聚合）。

        Args:
            local_epochs (int): 本輪客戶端的本地訓練輪次（實驗一的變因 E）。
            batch_size (int): 本輪訓練使用的批次大小（實驗二的變因 B）。
            gradient_hook (Callable, optional):
                【防禦模組介面（Step 3）】傳入各客戶端訓練的梯度後處理函數。
            aggregation_fn (Callable, optional):
                【防禦模組介面（Step 3）】
                自訂聚合函數，用於替換預設的 FedAvg（例如：Krum、Trimmed Mean）。
                簽名：aggregation_fn(client_updates: list[dict]) -> None
                函數應直接修改 self.global_model 的權重。
                若為 None，則使用預設的 FedAvg（self.aggregate_fedavg）。
            verbose (bool): 是否列印每輪的訓練統計資訊。

        Returns:
            dict: 本輪的統計日誌，包含：
                {
                    'round'         : int,   # 當前輪次
                    'local_epochs'  : int,   # 本輪 local epochs
                    'batch_size'    : int,   # 本輪 batch size
                    'avg_train_loss': float, # 所有客戶端的平均訓練損失
                    'avg_train_acc' : float, # 所有客戶端的平均訓練準確率
                    'num_clients'   : int,   # 本輪參與的客戶端數
                }
        """
        self.current_round += 1

        # ----------------------------------------------------------------
        # 1. 選取本輪參與的客戶端
        # ----------------------------------------------------------------
        selected_clients = self.select_clients(fraction=config.FRACTION_FIT)

        # ----------------------------------------------------------------
        # 2. 廣播全局模型權重
        # ----------------------------------------------------------------
        self.broadcast_global_weights(selected_clients)

        # ----------------------------------------------------------------
        # 3. 各客戶端執行本地訓練（收集更新）
        # ----------------------------------------------------------------
        client_updates = []
        for client in selected_clients:
            update = client.local_update(
                local_epochs=local_epochs,
                batch_size=batch_size,
                gradient_hook=gradient_hook,
            )
            client_updates.append(update)

        # ----------------------------------------------------------------
        # 4. 全局聚合（FedAvg 或自訂防禦聚合）
        # ----------------------------------------------------------------
        if aggregation_fn is not None:
            # 【防禦模組介面（Step 3）】使用自訂聚合函數
            aggregation_fn(client_updates)
        else:
            # 預設使用標準 FedAvg
            self.aggregate_fedavg(client_updates)

        # ----------------------------------------------------------------
        # 5. 統計本輪訓練資訊
        # ----------------------------------------------------------------
        avg_train_loss = sum(u['avg_loss'] for u in client_updates) / len(client_updates)
        avg_train_acc = sum(u['avg_acc'] for u in client_updates) / len(client_updates)

        round_log = {
            'round': self.current_round,
            'local_epochs': local_epochs,
            'batch_size': batch_size,
            'avg_train_loss': avg_train_loss,
            'avg_train_acc': avg_train_acc,
            'num_clients': len(selected_clients),
        }
        self.round_logs.append(round_log)

        if verbose:
            print(
                f"  [Round {self.current_round:03d}] "
                f"客戶端數：{len(selected_clients)} | "
                f"訓練損失：{avg_train_loss:.4f} | "
                f"訓練準確率：{avg_train_acc:.4f}"
            )

        return round_log

    # --------------------------------------------------------------------------
    # 工具方法
    # --------------------------------------------------------------------------

    def get_global_model(self) -> LeNet:
        """
        回傳當前全局模型的參考。

        Returns:
            LeNet: 全局模型實例（非副本）。
        """
        return self.global_model

    def get_global_weights(self) -> dict:
        """
        回傳全局模型權重的深複製字典。

        Returns:
            dict: 全局模型的 state_dict（深複製）。
        """
        return get_model_weights(self.global_model)

    def reset_round_logs(self) -> None:
        """
        清除所有歷史輪次日誌（在切換實驗配置時使用）。
        """
        self.round_logs = []
        self.current_round = 0

    def __repr__(self) -> str:
        return (
            f"Server(clients={len(self.clients)}, "
            f"round={self.current_round}/{config.GLOBAL_ROUNDS})"
        )
