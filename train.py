# ==============================================================================
# train.py — 客戶端本地訓練 Loop 實作
#
# 此模組負責單一客戶端在本地資料上的訓練流程，
# 與 client.py 分離的設計目的：
#   - client.py 負責「物件狀態管理」（持有模型、資料、身份）
#   - train.py  負責「純粹的訓練計算邏輯」（可獨立測試與替換）
#
# 【對防禦模組（Step 3）的介面說明】
#   梯度裁剪（Gradient Clipping）或梯度雜訊（DP-SGD）等防禦機制，
#   應在 loss.backward() 之後、optimizer.step() 之前介入。
#   建議防禦模組以「callback 函數」的形式傳入 local_train()，
#   掛載在 `gradient_hook` 參數（預設為 None，即不啟用防禦）。
# ==============================================================================

import copy
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config


# ==============================================================================
# 核心訓練函數
# ==============================================================================

def local_train(
    model: nn.Module,
    dataloader: DataLoader,
    local_epochs: int,
    learning_rate: float = config.LEARNING_RATE,
    device: torch.device = config.DEVICE,
    gradient_hook: Optional[Callable[[nn.Module], None]] = None,
) -> dict:
    """
    執行客戶端的本地訓練，回傳更新後的模型權重。

    訓練流程：
        for epoch in range(local_epochs):
            for batch in dataloader:
                1. 前向傳播（Forward Pass）
                2. 計算損失（Cross-Entropy Loss）
                3. 反向傳播（Backward Pass）
                4. [可選] 梯度後處理（防禦 Hook）
                5. 優化器更新（Adam step）

    Args:
        model (nn.Module):
            已載入全局權重的本地模型副本（由 client.py 傳入）。
        dataloader (DataLoader):
            該客戶端的本地訓練資料載入器。
        local_epochs (int):
            本地訓練輪次（實驗一的變因 E）。
        learning_rate (float):
            Adam 優化器學習率，預設從 config 讀取。
        device (torch.device):
            訓練使用的硬體裝置。
        gradient_hook (Callable, optional):
            【防禦模組介面（Step 3）】
            在 loss.backward() 之後、optimizer.step() 之前被呼叫。
            函數簽名：gradient_hook(model: nn.Module) -> None
            防禦模組可在此函數內對 model.parameters() 的 .grad 進行操作，
            例如：梯度裁剪、添加高斯雜訊（DP）等。
            若為 None，則跳過（標準 FedAvg，不啟用任何防禦）。

    Returns:
        dict: 訓練完成後的模型 state_dict（與 model.state_dict() 格式相同），
              供 server.py 的 FedAvg 聚合使用。

    Extra Info (附加於 return 外，透過 model 物件可直接存取):
        訓練期間的 loss 與 accuracy 統計請見回傳的 `train_info` 字典。
    """
    model.train()  # 切換至訓練模式（啟用 Dropout、BatchNorm 等訓練行為）
    model.to(device)

    # 初始化 Adam 優化器與交叉熵損失函數
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for epoch in range(local_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            # 將資料移至指定裝置
            images = images.to(device)
            labels = labels.to(device)

            # ----------------------------------------------------------------
            # Step A：清除上一批次的梯度
            # ----------------------------------------------------------------
            optimizer.zero_grad()

            # ----------------------------------------------------------------
            # Step B：前向傳播
            # ----------------------------------------------------------------
            logits = model(images)  # shape: (B, num_classes)

            # ----------------------------------------------------------------
            # Step C：計算損失
            # ----------------------------------------------------------------
            loss = criterion(logits, labels)

            # ----------------------------------------------------------------
            # Step D：反向傳播（計算梯度）
            # ----------------------------------------------------------------
            loss.backward()

            # ----------------------------------------------------------------
            # Step E：【防禦模組介面】梯度後處理（Hook 點）
            #
            # 防禦模組（Step 3）請在此位置介入：
            #   gradient_hook(model) 被呼叫時，model.parameters() 的
            #   各 .grad 屬性已被填入當前批次的梯度值。
            #   防禦操作（如裁剪、加噪）應就地修改 param.grad。
            #
            # 範例（防禦模組可參考）：
            #   def my_defense_hook(model):
            #       torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # ----------------------------------------------------------------
            if gradient_hook is not None:
                gradient_hook(model)

            # ----------------------------------------------------------------
            # Step F：優化器更新模型權重
            # ----------------------------------------------------------------
            optimizer.step()

            # 統計批次損失與準確率
            batch_loss = loss.item()
            preds = logits.argmax(dim=1)
            batch_correct = (preds == labels).sum().item()

            epoch_loss += batch_loss * images.size(0)
            epoch_correct += batch_correct
            epoch_samples += images.size(0)

        # 累計所有 epoch 的統計值
        total_loss += epoch_loss
        total_correct += epoch_correct
        total_samples += epoch_samples

    # 計算整體平均值（跨所有 epoch）
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0

    return {
        'weights': copy.deepcopy(model.state_dict()),  # 訓練後的模型權重（深複製）
        'avg_loss': avg_loss,                          # 平均訓練損失
        'avg_acc': avg_acc,                            # 平均訓練準確率
        ### 修改下面一行 原本 'num_samples': epoch_samples, 改成 'num_samples': len(dataloader.dataset),
        'num_samples': len(dataloader.dataset),        # 資料集總樣本數（供 FedAvg 加權使用）
    }


# ==============================================================================
# 梯度提取工具（供攻擊模組 Step 2 使用）
# ==============================================================================

def compute_gradients(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device = config.DEVICE,
) -> dict:
    """
    【攻擊模組介面（Step 2）】
    對單一批次計算梯度，但不更新模型權重。

    梯度洩漏攻擊（如 DLG、iDLG）通常需要取得某個特定批次的「真實梯度」，
    然後以虛擬資料進行重建。此函數提供一個乾淨的梯度計算介面，
    不會改動原始模型的權重。

    Args:
        model (nn.Module): 目前的全局模型（或本地模型）。
        images (torch.Tensor): 目標批次的輸入影像，shape = (B, 1, 28, 28)。
        labels (torch.Tensor): 對應的真實標籤，shape = (B,)。
        device (torch.device): 計算裝置。

    Returns:
        dict: 各層參數名稱 → 對應梯度 Tensor 的字典。
              格式與 model.named_parameters() 一致。
              例如：{
                  'features.0.weight': Tensor(...),
                  'features.0.bias': Tensor(...),
                  ...
                  'classifier.4.weight': Tensor(...),
              }
    """
    model.eval()
    model.to(device)

    images = images.to(device)
    labels = labels.to(device)

    # 清除既有梯度
    model.zero_grad()

    criterion = nn.CrossEntropyLoss()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()

    # 收集並回傳各層梯度（深複製，避免被後續操作覆蓋）
    gradients = {
        name: param.grad.clone().detach()
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    return gradients
