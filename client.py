# ==============================================================================
# client.py — 聯邦學習客戶端物件定義
#
# Client 物件負責：
#   1. 持有本地資料（DataLoader）與本地模型副本
#   2. 接收全局模型權重（from Server）
#   3. 執行本地訓練（呼叫 train.py）
#   4. 回傳更新後的模型權重與訓練統計（to Server）
#
# 【對攻擊模組（Step 2）的介面說明】
#   Client 物件保留了「訓練前快照」（pre_train_weights），
#   攻擊者可藉由比較訓練前後的權重差異來推算梯度：
#       pseudo_gradient = pre_train_weights - post_train_weights
#   此外，compute_gradients_on_batch() 方法提供更直接的梯度存取介面。
#
# 【對防禦模組（Step 3）的介面說明】
#   local_update() 接受 gradient_hook 參數，防禦模組將梯度處理函數
#   傳入即可在反向傳播後、參數更新前介入，無需修改此檔案。
# ==============================================================================

import copy
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from model import LeNet, get_model, get_model_weights, set_model_weights
from train import local_train, compute_gradients


# ==============================================================================
# Client 類別
# ==============================================================================

class Client:
    """
    聯邦學習中的單一客戶端。

    每個 Client 實例對應一個真實的邊緣裝置，
    持有私有的本地資料集與本地模型副本，
    在每輪通訊中從 Server 取得全局權重、本地訓練後回傳更新。

    Attributes:
        client_id (int): 客戶端唯一識別碼（0 ~ num_clients-1）。
        dataloader (DataLoader): 本地訓練資料載入器。
        local_model (LeNet): 本地模型副本（不與其他客戶端共享）。
        num_samples (int): 本地資料樣本總數（供 FedAvg 加權聚合使用）。
        pre_train_weights (dict | None): 【攻擊介面】本輪訓練開始前的模型權重快照。
        post_train_weights (dict | None): 【攻擊介面】本輪訓練結束後的模型權重。
        last_train_result (dict | None): 最近一次 local_update() 的完整回傳結果。
    """

    def __init__(self, client_id: int, dataloader: DataLoader, num_classes: int = None):
        """
        初始化客戶端。

        Args:
            client_id (int): 客戶端編號。
            dataloader (DataLoader): 本地訓練資料載入器（由 data_loader.py 建立）。
            num_classes (int, optional): 模型輸出類別數，預設從 config 讀取。
        """
        self.client_id = client_id
        self.dataloader = dataloader
        self.num_samples = len(dataloader.dataset)

        # 建立獨立的本地模型副本（各客戶端之間完全隔離）
        self.local_model: LeNet = get_model(num_classes=num_classes)

        # 訓練前後的權重快照（供攻擊模組存取）
        self.pre_train_weights: Optional[dict] = None
        self.post_train_weights: Optional[dict] = None

        # 最近一次訓練的完整統計資訊
        self.last_train_result: Optional[dict] = None

        print(f"  [Client {self.client_id:02d}] 初始化完成，本地樣本數：{self.num_samples}")

    # --------------------------------------------------------------------------
    # 核心方法：接收全局權重
    # --------------------------------------------------------------------------

    def receive_global_weights(self, global_weights: dict) -> None:
        """
        接收 Server 廣播的全局模型權重，載入至本地模型。

        此方法在每輪通訊開始時由 Server 呼叫，
        確保每個客戶端從相同的全局模型出發進行本地訓練。

        Args:
            global_weights (dict): 全局模型的 state_dict（來自 Server）。
        """
        set_model_weights(self.local_model, global_weights)

    # --------------------------------------------------------------------------
    # 核心方法：執行本地訓練
    # --------------------------------------------------------------------------

    def local_update(
        self,
        local_epochs: int,
        batch_size: Optional[int] = None,
        gradient_hook: Optional[Callable[[nn.Module], None]] = None,
    ) -> dict:
        """
        執行本地訓練並回傳更新結果。

        流程：
            1. 記錄訓練前的模型權重快照（供攻擊模組使用）
            2. 呼叫 train.local_train() 執行指定輪次的本地訓練
            3. 記錄訓練後的模型權重
            4. 回傳更新資訊給 Server

        Args:
            local_epochs (int):
                本地訓練輪次（實驗一的變因 E）。
            batch_size (int, optional):
                若需要在此輪使用不同的 batch_size（實驗二的變因 B），
                可動態重建 DataLoader。若為 None，使用原始 dataloader。
            gradient_hook (Callable, optional):
                【防禦模組介面（Step 3）】
                梯度後處理函數，傳入 train.local_train() 的 gradient_hook 參數。
                簽名：gradient_hook(model: nn.Module) -> None

        Returns:
            dict: 包含以下欄位的字典，供 Server 的 FedAvg 使用：
                {
                    'client_id'   : int,     # 客戶端編號
                    'weights'     : dict,    # 訓練後的模型 state_dict
                    'num_samples' : int,     # 本地樣本數（用於加權平均）
                    'avg_loss'    : float,   # 本輪平均訓練損失
                    'avg_acc'     : float,   # 本輪平均訓練準確率
                }
        """
        # ----------------------------------------------------------------
        # 1. 記錄訓練前的模型權重快照
        #    【攻擊模組介面（Step 2）】
        #    攻擊者可透過 client.pre_train_weights 取得此快照，
        #    並與 post_train_weights 相減推算「虛擬梯度」：
        #        pseudo_grad ≈ (pre_weights - post_weights) / learning_rate
        # ----------------------------------------------------------------
        self.pre_train_weights = get_model_weights(self.local_model)

        # ----------------------------------------------------------------
        # 2. 若需動態切換 batch_size，重建 DataLoader
        #    （實驗二：Batch Size 為變因時使用）
        # ----------------------------------------------------------------
        train_loader = self.dataloader
        if batch_size is not None and batch_size != self.dataloader.batch_size:
            from data_loader import get_client_dataloader
            # 取得原始 dataset 與索引
            original_subset = self.dataloader.dataset
            original_indices = list(range(len(original_subset)))
            train_loader = get_client_dataloader(
                original_subset.dataset if hasattr(original_subset, 'dataset') else original_subset,
                original_subset.indices if hasattr(original_subset, 'indices') else original_indices,
                batch_size=batch_size,
            )

        # ----------------------------------------------------------------
        # 3. 執行本地訓練（委派給 train.py）
        # ----------------------------------------------------------------
        result = local_train(
            model=self.local_model,
            dataloader=train_loader,
            local_epochs=local_epochs,
            learning_rate=config.LEARNING_RATE,
            device=config.DEVICE,
            gradient_hook=gradient_hook,
        )

        # ----------------------------------------------------------------
        # 4. 記錄訓練後的模型權重
        #    【攻擊模組介面（Step 2）】
        #    攻擊者可透過 client.post_train_weights 取得此值
        # ----------------------------------------------------------------
        self.post_train_weights = result['weights']
        self.last_train_result = result

        # ----------------------------------------------------------------
        # 5. 組裝回傳給 Server 的更新資訊
        # ----------------------------------------------------------------
        update = {
            'client_id': self.client_id,
            'weights': result['weights'],       # 訓練後模型權重（FedAvg 聚合用）
            'num_samples': self.num_samples,    # 樣本數（加權平均用）
            'avg_loss': result['avg_loss'],     # 平均訓練損失（日誌用）
            'avg_acc': result['avg_acc'],       # 平均訓練準確率（日誌用）
        }
        return update

    # --------------------------------------------------------------------------
    # 攻擊模組介面：取得梯度差（虛擬梯度）
    # --------------------------------------------------------------------------

    def get_pseudo_gradients(self) -> Optional[dict]:
        """
        【攻擊模組介面（Step 2）】
        計算訓練前後的權重差，作為「虛擬梯度」返回。

        在聯邦學習中，Server 只能觀察到模型更新（Δw = w_before - w_after），
        這等價於「梯度 × 學習率 × 步數」的累積效果。
        DLG、iDLG 等梯度洩漏攻擊即以此作為攻擊的起始資訊。

        必須在 local_update() 呼叫後才能使用。

        Returns:
            dict | None: 各層的虛擬梯度字典（pre - post 的差值），
                         若尚未訓練則回傳 None。
        """
        if self.pre_train_weights is None or self.post_train_weights is None:
            print(f"[Client {self.client_id:02d}] 警告：尚未執行 local_update()，無法取得虛擬梯度。")
            return None

        pseudo_grads = {
            key: self.pre_train_weights[key].float() - self.post_train_weights[key].float()
            for key in self.pre_train_weights
        }
        return pseudo_grads

    # --------------------------------------------------------------------------
    # 攻擊模組介面：對特定批次計算真實梯度
    # --------------------------------------------------------------------------

    def compute_gradients_on_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """
        【攻擊模組介面（Step 2）】
        對指定的單一批次資料計算真實梯度（不更新模型權重）。

        DLG 攻擊通常需要取得伺服器端可觀察到的「某批次梯度」，
        此方法模擬了這個資訊洩漏點。

        Args:
            images (torch.Tensor): 目標批次的影像，shape = (B, 1, 28, 28)。
            labels (torch.Tensor): 對應的真實標籤，shape = (B,)。

        Returns:
            dict: 各層名稱 → 梯度 Tensor 的字典（已 detach，不會影響計算圖）。
        """
        return compute_gradients(
            model=self.local_model,
            images=images,
            labels=labels,
            device=config.DEVICE,
        )

    # --------------------------------------------------------------------------
    # 工具方法
    # --------------------------------------------------------------------------

    def get_local_model(self) -> LeNet:
        """
        回傳本地模型的參考（非副本）。

        注意：此方法回傳的是模型的直接參考，
        若後續操作會修改模型，請使用 copy.deepcopy() 保護原始狀態。

        Returns:
            LeNet: 本地模型實例。
        """
        return self.local_model

    def get_num_samples(self) -> int:
        """
        回傳本地資料集的樣本總數（FedAvg 加權聚合使用）。

        Returns:
            int: 本地樣本數。
        """
        return self.num_samples

    def __repr__(self) -> str:
        return (
            f"Client(id={self.client_id}, "
            f"samples={self.num_samples}, "
            f"device={config.DEVICE})"
        )
