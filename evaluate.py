# ==============================================================================
# evaluate.py — 全局模型在測試集上的評估
#
# 此模組提供純函數式的評估介面（無狀態），
# 可在任意輪次被 main.py 呼叫，取得當前全局模型的泛化能力指標。
#
# 回傳指標：
#   - Test Accuracy  (%)：測試集整體準確率
#   - Test Loss       ：測試集平均交叉熵損失
#   - Per-class Acc  ：各類別的個別準確率（可選，供報告使用）
# ==============================================================================

from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config


# ==============================================================================
# 核心評估函數
# ==============================================================================

def evaluate_global_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = config.DEVICE,
    compute_per_class: bool = False,
) -> dict:
    """
    評估全局模型在完整測試集上的效能。

    此函數為純評估流程（不更新模型參數、不計算梯度），
    使用 torch.no_grad() 加速並節省顯存。

    Args:
        model (nn.Module):
            要評估的全局模型（通常是 server.get_global_model()）。
        test_loader (DataLoader):
            完整測試集的 DataLoader（由 data_loader.get_test_dataloader() 建立）。
        device (torch.device):
            評估使用的硬體裝置，預設從 config 讀取。
        compute_per_class (bool):
            是否計算每個類別的個別準確率。
            False（預設）：只計算整體 Accuracy 與 Loss，速度較快。
            True：額外計算各類別 Accuracy，適合書面報告的詳細分析。

    Returns:
        dict: 包含以下欄位的評估結果字典：
            {
                'test_loss'      : float,         # 測試集平均損失
                'test_accuracy'  : float,         # 測試集整體準確率（0.0 ~ 1.0）
                'test_accuracy_pct': float,       # 測試集整體準確率（百分比，0.0 ~ 100.0）
                'total_samples'  : int,           # 測試集總樣本數
                'correct_samples': int,           # 預測正確的樣本數
                'per_class_acc'  : dict | None,   # 各類別準確率（compute_per_class=True 時才有值）
            }
    """
    model.eval()   # 切換至評估模式（停用 Dropout、固定 BatchNorm 統計量）
    model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='sum')  # 使用 sum 方便後續計算平均

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 每個類別的正確預測數與總樣本數（用於 per-class accuracy）
    num_classes = config.NUM_CLASSES
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():  # 評估時不需要計算梯度，節省記憶體
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向傳播取得 logits
            logits = model(images)

            # 計算批次損失（sum 模式）
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # 取最大 logit 對應的類別作為預測結果
            preds = logits.argmax(dim=1)
            correct_mask = (preds == labels)

            total_correct += correct_mask.sum().item()
            total_samples += labels.size(0)

            # 統計每個類別的正確數與總數
            if compute_per_class:
                for cls in range(num_classes):
                    cls_mask = (labels == cls)
                    class_total[cls] += cls_mask.sum().item()
                    class_correct[cls] += (correct_mask & cls_mask).sum().item()

    # 計算整體指標
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # 計算各類別準確率（可選）
    per_class_acc = None
    if compute_per_class:
        per_class_acc = {}
        for cls in range(num_classes):
            if class_total[cls] > 0:
                per_class_acc[cls] = (class_correct[cls] / class_total[cls]).item()
            else:
                per_class_acc[cls] = 0.0  # 此類別在測試集中無樣本

    result = {
        'test_loss': avg_loss,
        'test_accuracy': accuracy,
        'test_accuracy_pct': accuracy * 100.0,
        'total_samples': total_samples,
        'correct_samples': total_correct,
        'per_class_acc': per_class_acc,
    }
    return result


# ==============================================================================
# 訓練過程中的輕量評估（每輪呼叫，效能優先）
# ==============================================================================

def quick_evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = config.DEVICE,
) -> tuple:
    """
    輕量化評估函數，只回傳 (accuracy, loss) 的元組。

    此函數專為每輪通訊結束後的快速評估設計，
    省略 per-class 統計以最小化開銷。
    main.py 在 FL 訓練迴圈中呼叫此函數記錄每輪的全局準確率。

    Args:
        model (nn.Module): 要評估的全局模型。
        test_loader (DataLoader): 測試集 DataLoader。
        device (torch.device): 評估裝置。

    Returns:
        tuple: (accuracy: float, loss: float)
               accuracy 為 0.0 ~ 1.0 的小數形式。
    """
    result = evaluate_global_model(
        model=model,
        test_loader=test_loader,
        device=device,
        compute_per_class=False,
    )
    return result['test_accuracy'], result['test_loss']


# ==============================================================================
# 完整評估報告（實驗結束後呼叫，用於書面報告）
# ==============================================================================

def full_evaluation_report(
    model: nn.Module,
    test_loader: DataLoader,
    run_label: str = '',
    device: torch.device = config.DEVICE,
) -> dict:
    """
    在實驗結束後執行完整評估，並格式化列印結果。

    適合在每次完整 FL 訓練結束後呼叫，取得詳細的測試集報告。
    若 run_label 非空，會在列印中標注該次實驗的標識（例如 'E=3, B=32'）。

    Args:
        model (nn.Module): 訓練完成的全局模型。
        test_loader (DataLoader): 測試集 DataLoader。
        run_label (str): 此次實驗的標識字串，用於日誌顯示。
        device (torch.device): 評估裝置。

    Returns:
        dict: 完整評估結果（同 evaluate_global_model 的回傳格式，含 per_class_acc）。
    """
    result = evaluate_global_model(
        model=model,
        test_loader=test_loader,
        device=device,
        compute_per_class=True,
    )

    label_str = f"[{run_label}] " if run_label else ""
    print(f"\n{'─'*50}")
    print(f"{label_str}最終評估結果（測試集）")
    print(f"  測試損失    : {result['test_loss']:.4f}")
    print(f"  測試準確率  : {result['test_accuracy_pct']:.2f}%")
    print(f"  正確/總計   : {result['correct_samples']} / {result['total_samples']}")

    if result['per_class_acc']:
        print(f"\n  各類別準確率：")
        for cls_id, acc in result['per_class_acc'].items():
            bar = '█' * int(acc * 20)
            print(f"    類別 {cls_id:2d}: {acc*100:5.1f}%  {bar}")
    print(f"{'─'*50}\n")

    return result
