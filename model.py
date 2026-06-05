# ==============================================================================
# model.py — LeNet 網路架構定義
#
# 標準 LeNet-5 架構，針對本專案做以下調整：
#   1. 輸入通道固定為 1（灰階影像）
#   2. 輸入尺寸固定為 28x28（MNIST 原生尺寸 / AT&T 統一 Resize 後尺寸）
#   3. 最後一層全連接層的輸出數（num_classes）可動態設定，
#      由 config.NUM_CLASSES 傳入，以支援 MNIST（10類）與 AT&T Face（40類）
#
# 【對攻擊模組的介面說明】
#   後續梯度洩漏攻擊（Step 2）可透過以下方式取得中間層特徵：
#     - model.features(x)    → 取得捲積層特徵圖
#     - model.flatten(x)     → 取得 flatten 後的向量
#     - model.classifier(x)  → 取得分類器輸出
#   或使用 PyTorch 的 register_forward_hook() 掛載到任意子模組。
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ==============================================================================
# LeNet-5 模型定義
# ==============================================================================

class LeNet(nn.Module):
    """
    標準 LeNet-5 架構（適配 28x28 灰階輸入）。

    網路結構：
        輸入層  : (B, 1, 28, 28)
        C1      : Conv2d(1, 6, kernel=5, padding=2) → (B, 6, 28, 28)
        S2      : AvgPool2d(2, 2)                   → (B, 6, 14, 14)
        C3      : Conv2d(6, 16, kernel=5)           → (B, 16, 10, 10)
        S4      : AvgPool2d(2, 2)                   → (B, 16, 5, 5)
        Flatten : 16 × 5 × 5 = 400
        F5      : Linear(400, 120)
        F6      : Linear(120, 84)
        Output  : Linear(84, num_classes)

    Args:
        num_classes (int): 分類的類別總數。
                           MNIST = 10，AT&T Face = 40。

    Attributes:
        features (nn.Sequential): 捲積特徵提取部分（C1~S4）。
        classifier (nn.Sequential): 全連接分類部分（F5~Output）。

    【後續組員介面】
        - 梯度洩漏攻擊（Step 2）可 hook 到 self.features 或 self.classifier 的任意層。
        - 防禦機制（Step 3）若需要在梯度上操作，可在 train.py 的 loss.backward()
          之後、optimizer.step() 之前介入。
    """

    def __init__(self, num_classes: int):
        super(LeNet, self).__init__()

        # ------------------------------------------------------------------
        # 特徵提取層（捲積 + 池化）
        # 對應 LeNet 的 C1、S2、C3、S4 層
        # ------------------------------------------------------------------
        self.features = nn.Sequential(
            # C1 層：6 個 5×5 捲積核，padding=2 保持輸出為 28×28
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.Tanh(),          # 原始 LeNet 使用 Tanh 激活函數

            # S2 層：2×2 平均池化，輸出縮減至 14×14
            nn.AvgPool2d(kernel_size=2, stride=2),

            # C3 層：16 個 5×5 捲積核，無 padding，輸出為 10×10
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0),
            nn.Tanh(),

            # S4 層：2×2 平均池化，輸出縮減至 5×5
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # ------------------------------------------------------------------
        # 全連接分類層
        # 對應 LeNet 的 F5、F6、Output 層
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            # F5 層：400 → 120
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Tanh(),

            # F6 層：120 → 84
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),

            # Output 層：84 → num_classes（依資料集動態設定）
            nn.Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播。

        Args:
            x (torch.Tensor): 輸入影像張量，shape = (B, 1, 28, 28)

        Returns:
            torch.Tensor: 各類別的 logits，shape = (B, num_classes)
                          （注意：未經 Softmax，直接配合 CrossEntropyLoss 使用）
        """
        # 特徵提取
        x = self.features(x)

        # 展平：(B, 16, 5, 5) → (B, 400)
        x = torch.flatten(x, start_dim=1)

        # 全連接分類
        x = self.classifier(x)

        return x

    def get_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        【攻擊模組介面】取得捲積層輸出的特徵圖（未展平）。

        梯度洩漏攻擊（Step 2）可呼叫此方法取得中間層表示。

        Args:
            x (torch.Tensor): 輸入影像張量，shape = (B, 1, 28, 28)

        Returns:
            torch.Tensor: 特徵圖，shape = (B, 16, 5, 5)
        """
        return self.features(x)

    def get_flat_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        【攻擊模組介面】取得展平後的特徵向量。

        Args:
            x (torch.Tensor): 輸入影像張量，shape = (B, 1, 28, 28)

        Returns:
            torch.Tensor: 展平後的特徵向量，shape = (B, 400)
        """
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return x


# ==============================================================================
# 模型工廠函數（供 main.py 與 server.py 呼叫）
# ==============================================================================

def get_model(num_classes: int = None) -> LeNet:
    """
    建立並回傳一個 LeNet 模型實例（已移至目標裝置）。

    Args:
        num_classes (int, optional): 分類類別數。
                                     若未指定，自動從 config.NUM_CLASSES 讀取。

    Returns:
        LeNet: 初始化完成、已部署至 config.DEVICE 的模型實例。
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    model = LeNet(num_classes=num_classes)
    model = model.to(config.DEVICE)

    # 印出模型資訊
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[model] LeNet 初始化完成")
    print(f"  資料集：{config.DATASET}，輸出類別數：{num_classes}")
    print(f"  裝置：{config.DEVICE}")
    print(f"  總參數量：{total_params:,}（可訓練：{trainable_params:,}）")

    return model


def get_model_weights(model: LeNet) -> dict:
    """
    【通用工具】取得模型的完整權重字典（深複製）。

    Server 的 FedAvg 聚合與 Client 接收全局權重時均透過此函數操作，
    避免不同客戶端之間共享同一個記憶體參考。

    Args:
        model (LeNet): 來源模型

    Returns:
        dict: 模型的 state_dict（深複製），key 為層名稱，value 為 Tensor
    """
    return {k: v.clone().detach() for k, v in model.state_dict().items()}


def set_model_weights(model: LeNet, weights: dict) -> None:
    """
    【通用工具】將權重字典載入至指定模型（就地修改）。

    Args:
        model (LeNet): 目標模型（就地修改）
        weights (dict): 要載入的權重字典（來自 get_model_weights）

    Returns:
        None（就地修改 model）
    """
    model.load_state_dict(weights)


# ==============================================================================
# 模型架構示意（執行此檔案時直接列印）
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("LeNet 架構驗證")
    print("=" * 60)

    # 建立測試模型（以 MNIST 10 類為例）
    test_model = get_model(num_classes=10)
    print("\n模型結構：")
    print(test_model)

    # 以隨機假資料驗證前向傳播的維度正確性
    dummy_input = torch.randn(4, 1, 28, 28).to(config.DEVICE)  # Batch size = 4
    output = test_model(dummy_input)
    feature_map = test_model.get_feature_map(dummy_input)
    flat_features = test_model.get_flat_features(dummy_input)

    print(f"\n維度驗證（Batch size = 4）：")
    print(f"  輸入：        {tuple(dummy_input.shape)}")
    print(f"  特徵圖：      {tuple(feature_map.shape)}")
    print(f"  展平特徵：    {tuple(flat_features.shape)}")
    print(f"  輸出 logits： {tuple(output.shape)}")
    print("\n✅ 前向傳播維度驗證通過！")
