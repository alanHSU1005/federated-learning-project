import os
import copy
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict

import config
from model import LeNet, get_model

def run_dlg_attack(
    target_gradients: Dict[str, torch.Tensor],
    model: LeNet,
    num_iterations: int = 300,
    learning_rate: float = 0.1,
    save_dir: str = "./dlg_attack_results",
    original_image: torch.Tensor = None,
    file_name: str = "dlg_recovery_process.png"
):
    """
    執行 Deep Leakage from Gradients (DLG) 攻擊。
    
    此函數會隨機初始化虛擬影像與標籤，並透過優化器（L-BFGS）最小化
    「虛擬梯度」與「目標梯度」之間的距離，藉此還原出原始影像資料。
    
    Args:
        target_gradients (Dict[str, torch.Tensor]): 從客戶端攔截到的目標梯度（或虛擬梯度）。
        model (LeNet): 用於計算梯度的全局模型副本。
        num_iterations (int): 優化迭代次數。
        learning_rate (float): 優化學習率。
        save_dir (str): 還原影像的儲存路徑。
        
    Returns:
        torch.Tensor: 還原成功的影像 Tensor。
    """
    print("\n[Attack] 開始執行 DLG 梯度洩漏攻擊...")
    
    # 建立儲存目錄
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 確保模型在正確的裝置上並設定為 eval 模式
    model.eval()
    model.to(config.DEVICE)
    
    # DLG 通常針對單一樣本（Batch Size = 1）進行攻擊，尺寸依據資料集而定 (1, 1, 28, 28)
    dummy_data = torch.randn((1, 1, 28, 28)).to(config.DEVICE).requires_grad_(True)
    
    # 從模型結構中動態推斷類別數 (LeNet 最後一層輸出)
    num_classes = model.classifier[-1].out_features
    
    # 隨機初始化虛擬標籤
    # 這裡我們使用 soft label 進行優化 (也可以直接枚舉硬標籤)
    dummy_label = torch.randn((1, num_classes)).to(config.DEVICE).requires_grad_(True)
    
    # 將原始梯度轉換為 tuple 以方便後續計算距離
    # 注意：這裡要過濾掉不需梯度的參數，以對齊 model.parameters()
    target_grad_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in target_gradients:
                target_grad_list.append(target_gradients[name].detach().clone().to(config.DEVICE))
            else:
                # 若目標梯度中缺少某些層，則填入 0
                target_grad_list.append(torch.zeros_like(param).to(config.DEVICE))
    
    # 使用 L-BFGS 作為優化器（DLG 原始論文建議，收斂較快且穩定）
    # 也可以改用 Adam，但 L-BFGS 在這類影像還原任務上通常效果更好
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=learning_rate)
    
    # 儲存還原過程的影像以便視覺化
    history = []
    current_loss = 0.0

    for iters in range(num_iterations):
        def closure():
            nonlocal current_loss
            optimizer.zero_grad()
            
            # 1. 以前向傳播計算虛擬預測值
            dummy_pred = model(dummy_data)
            
            # 2. 計算虛擬損失 (使用 CrossEntropy 或是對 logits 算 Softmax CrossEntropy)
            # 為了能對 dummy_label 算梯度，我們手動計算 cross entropy 的平滑版
            dummy_loss = -torch.sum(F.softmax(dummy_label, dim=-1) * F.log_softmax(dummy_pred, dim=-1), dim=-1).mean()
            
            # 3. 計算虛擬梯度 (對模型參數求導)
            dummy_gradients = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            
            # 4. 計算梯度距離 (MSE Loss)
            grad_diff = 0
            for dummy_g, target_g in zip(dummy_gradients, target_grad_list):
                # 這裡使用歐幾里得距離的平方 (L2 norm 平方) 作為損失
                grad_diff += ((dummy_g - target_g) ** 2).sum()
                
            # 5. 反向傳播：最小化梯度距離，進而更新 dummy_data 與 dummy_label
            grad_diff.backward()
            current_loss = grad_diff.detach()
            return grad_diff
        
        # 執行一步優化
        optimizer.step(closure)
        
        # 每隔固定次數，印出當前狀態並記錄影像
        if iters % 10 == 0 or iters == num_iterations - 1:
            print(f"[Attack] Iteration {iters:3d} | Gradient Distance (MSE): {current_loss.item():.6f}")
            history.append((iters, dummy_data.detach().cpu().clone()))
            
    # ====== 視覺化與儲存結果 ======
    print(f"\n[Attack] 攻擊完成，正在儲存還原過程影像至 {save_dir} ...")
    
    # 畫出還原演進圖 (選取最多 10 個步驟) + 1 個原始影像(如果有)
    steps_to_show = min(10, len(history))
    total_subplots = steps_to_show + (1 if original_image is not None else 0)
    fig, axes = plt.subplots(1, total_subplots, figsize=(1.5 * total_subplots, 3))
    
    # 若只有一張圖
    if total_subplots == 1:
        axes = [axes]
        
    for i, step_idx in enumerate(torch.linspace(0, len(history) - 1, steps_to_show).long()):
        actual_iter, img_tensor_tuple = history[step_idx]
        img_tensor = img_tensor_tuple[0, 0] # 取出 (28, 28)
        # 簡單標準化以便顯示
        img_np = img_tensor.numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        axes[i].imshow(img_np, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Iter {actual_iter}")
    
    # 如果有提供原始影像，則繪製在最後一格
    if original_image is not None:
        img_np = original_image[0, 0].detach().cpu().numpy()
        axes[-1].imshow(img_np, cmap='gray')
        axes[-1].axis('off')
        axes[-1].set_title("Original")
        
    plt.tight_layout()
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    plt.close()
    
    print(f"[Attack] 還原影像已儲存至：{save_path}")
    
    return dummy_data.detach()

# ==============================================================================
# 結合 client.get_pseudo_gradients() 的範例呼叫
# ==============================================================================
if __name__ == '__main__':
    from client import Client
    from data_loader import prepare_data
    
    print("=" * 60)
    print("DLG Gradient Leakage Attack 測試範例 (包含 MNIST 與 AT&T Face)")
    print("=" * 60)
    
    datasets_to_test = ['MNIST', 'ATT_FACE']
    num_attacks_per_dataset = 2
    
    for dataset_name in datasets_to_test:
        print(f"\n\n{'#'*60}")
        print(f"開始測試資料集：{dataset_name}")
        print(f"{'#'*60}")
        
        # 動態決定類別數
        num_classes = 10 if dataset_name == 'MNIST' else 40
        
        # 1. 初始化資料集與 DataLoader
        print("\n[Step 1] 準備資料集與客戶端...")
        try:
            client_loaders, test_loader, _ = prepare_data(
                dataset_name=dataset_name, 
                num_clients=1, 
                classes_per_client=4, 
                batch_size=1
            )
        except FileNotFoundError as e:
            print(f"[{dataset_name}] 資料集載入失敗，可能是尚未下載：{e}")
            print(f"跳過 {dataset_name} 的攻擊測試。")
            continue
            
        for attack_idx in range(num_attacks_per_dataset):
            print(f"\n{'='*40}")
            print(f"執行 {dataset_name} 第 {attack_idx + 1}/{num_attacks_per_dataset} 次攻擊測試")
            print(f"{'='*40}")
            
            from torch.utils.data import Subset, DataLoader
            # 每次取不同的一筆樣本
            subset = Subset(client_loaders[0].dataset, [attack_idx])
            one_sample_loader = DataLoader(subset, batch_size=1)
            
            # 取得真實的原始影像以供比對
            original_images, _ = next(iter(one_sample_loader))
            
            # 建立目標 Client
            target_client = Client(client_id=0, dataloader=one_sample_loader, num_classes=num_classes)
            
            # 2. 建立全局模型並同步給 Client
            print("\n[Step 2] 準備全局模型並派發給客戶端...")
            global_model = get_model(num_classes=num_classes)
            # 我們複製一份模型狀態作為 server 端的初始狀態
            initial_weights = copy.deepcopy(global_model.state_dict())
            
            # 客戶端接收全局權重
            target_client.receive_global_weights(initial_weights)
            
            # 3. 執行本地訓練 (模擬 FL 的 Client 更新過程)
            # 注意：DLG 攻擊在 Batch Size = 1 且 Epoch = 1 時最容易成功
            print("\n[Step 3] 客戶端執行本地訓練，產生梯度...")
            target_client.local_update(local_epochs=1, batch_size=1)
            
            # 4. 攻擊者攔截虛擬梯度 (Pseudo Gradients)
            print("\n[Step 4] 攻擊者攔截更新，取得 pseudo gradients...")
            pseudo_gradients = target_client.get_pseudo_gradients()
            
            if pseudo_gradients is not None:
                # 5. 執行 DLG 攻擊
                print("\n[Step 5] 執行 DLG 攻擊...")
                run_dlg_attack(
                    target_gradients=pseudo_gradients,
                    model=global_model,
                    num_iterations=100,  # 範例中減少迭代次數以加快測試
                    learning_rate=0.1,
                    save_dir='./dlg_attack_results',
                    original_image=original_images,
                    file_name=f"dlg_recovery_{dataset_name}_{attack_idx + 1}.png"
                )
            else:
                print("無法取得虛擬梯度，攻擊失敗。")
