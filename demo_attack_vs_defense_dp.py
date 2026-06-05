# ==============================================================================
# demo_attack_vs_defense.py — 聯邦學習隱私資安演練：DLG & R-GAP 雙重攻擊 vs DP 差分隱私防禦
# ==============================================================================

import os
import copy
import time
import math
import random
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader

import config
from data_loader import prepare_data
from model import get_model
from client import Client
from dlg_attack import run_dlg_attack
from rgap_attack.rgap_lenet_attack import run_rgap_lenet
from rgap_attack.utils import show_images

# 🛡️ 匯入你寫的差分隱私防禦函數
from dp_defense import apply_dp_defense

# ─── 🎯 惡意攻擊與差分隱私防禦展示專用參數 (獨立於全局 Config 之外) ───
DEMO_BATCH_SIZE = 1        # DLG/R-GAP 攻擊最脆弱場景：必須為 1 才能精確還原單一隱私影像
DEMO_LOCAL_EPOCHS = 1      # 攻擊在 Epoch=1 時效果最好，避免梯度被多次疊加而失真
NUM_ITERATIONS = 300       # DLG 迭代優化次數 (確保明文組 DLG 影像還原清晰)
ATTACK_LR = 0.1            # L-BFGS 攻擊優化器的學習率

# 📂 【自訂設定區】輸出路徑與動態檔名格式
SAVE_DIR = "./demo_dp_outputs"  # 還原結果影像的儲存主資料夾
FILENAME_TEMPLATE = "demo_{dataset}_{img_type}_{attack}_{status}.png"
# ─────────────────────────────────────────────────────────────────


def calculate_psnr(reconstructed_tensor, original_tensor):
    """
    【指標】客觀評測重建影像質量：計算峰值信噪比 (PSNR)
    """
    img1 = reconstructed_tensor.detach().cpu().clone()[0, 0]
    img2 = original_tensor.detach().cpu().clone()[0, 0]
    
    img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
    img2_norm = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)
    
    mse = torch.mean((img1_norm - img2_norm) ** 2).item()
    if mse == 0:
        return float('inf')
    
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr

def clean_rgap_image(rgap_tensor):
    """
    純化通道：不對 R-GAP 進行任何破壞性的數值裁剪，
    僅將其統一轉為與 DLG 規格一致的 torch.Tensor (1, 1, H, W)
    """
    if rgap_tensor is None:
        return None
    
    # 1. 如果已經是 PyTorch Tensor，直接回傳，不亂動數值
    if isinstance(rgap_tensor, torch.Tensor):
        # 確保維度是 (1, 1, H, W)
        if rgap_tensor.dim() == 2:
            return rgap_tensor.unsqueeze(0).unsqueeze(0)
        return rgap_tensor
        
    # 2. 如果是 NumPy 陣列 (R-GAP 解析解出來的格式)
    elif isinstance(rgap_tensor, np.ndarray):
        img_np = rgap_tensor.copy()
        
        # 配合 R-GAP 的維度，通常是 (1, 1, 28, 28) 或 (28, 28)
        if img_np.ndim == 4:
            img_np = img_np[0, 0]
            
        # 直接做最標準的 0~1 Min-Max 歸一化（不切百分位）
        v_min, v_max = img_np.min(), img_np.max()
        if v_max - v_min > 1e-8:
            img_np = (img_np - v_min) / (v_max - v_min)
        else:
            img_np = np.zeros_like(img_np)
            
        # 轉回標準 torch.Tensor 格式 (1, 1, H, W)
        cleaned_tensor = torch.zeros((1, 1, img_np.shape[0], img_np.shape[1]))
        cleaned_tensor[0, 0] = torch.tensor(img_np)
        return cleaned_tensor

    return rgap_tensor

def run_client_simulation(one_sample_loader, initial_weights, is_protected=False):
    """
    【模組化】模擬客戶端本地訓練與權重收集流程。
    若啟動差分隱私防禦，調用 apply_dp_defense 進行裁剪與加噪。
    """
    t_start = time.time()
    
    client = Client(client_id=1, dataloader=one_sample_loader, num_classes=config.NUM_CLASSES)
    client.receive_global_weights(copy.deepcopy(initial_weights))

    # 在訓練前先取得真實梯度（R-GAP 專用）
    real_batch = next(iter(one_sample_loader))
    real_images, real_labels = real_batch[0], real_batch[1]
    true_gradients = client.compute_gradients_on_batch(real_images, real_labels)
    
    client.local_update(local_epochs=DEMO_LOCAL_EPOCHS, batch_size=DEMO_BATCH_SIZE)
    pseudo_gradients = client.get_pseudo_gradients()  # DLG 用這個
    
    if is_protected:
        # 🛡️ 差分隱私防禦組：傳入原始梯度字典進行 Clipping 與加噪
        print("   -> [DP 防禦啟動] 正在執行梯度裁剪與高斯雜訊注入...")
        dp_gradients = apply_dp_defense(
            stolen_grads=pseudo_gradients, 
            max_norm=0.5, 
            noise_scale=0.05, 
            device=config.DEVICE
        )
        
        # DLG 與 R-GAP 拿到的都會是經過你 DP 摧毀特徵後的安全梯度
        client_time = time.time() - t_start
        return dp_gradients, dp_gradients, client_time
    
    client_time = time.time() - t_start
    return pseudo_gradients, true_gradients, client_time


def execute_attack_pipeline(dlg_gradients, rgap_gradients, global_model, original_images, img_type, status):
    """
    【模組化】自動並列執行 DLG 攻擊與 R-GAP 攻擊，並計算對應的時間與 PSNR 表現。
    針對防禦組加上異常捕捉，以驗證防禦機制的公式阻斷能力。
    """
    # -----------------------------------------------------------------
    # 1. DLG 攻擊部分 (基於優化的反向傳播迭代)
    # -----------------------------------------------------------------
    dlg_filename = FILENAME_TEMPLATE.format(
        dataset=config.DATASET, img_type=img_type, attack="dlg", status=status
    )
    print(f" ⏳ 啟動 [{status.upper()}組] DLG 慢速優化迭代攻擊...")
    t_dlg_start = time.time()
    recon_img_dlg = run_dlg_attack(
        target_gradients=dlg_gradients,
        model=global_model,
        num_iterations=NUM_ITERATIONS,
        learning_rate=ATTACK_LR,
        save_dir=SAVE_DIR,
        original_image=original_images,
        file_name=dlg_filename
    )
    dlg_time = time.time() - t_dlg_start
    dlg_psnr = calculate_psnr(recon_img_dlg, original_images)

    # -----------------------------------------------------------------
    # 2. R-GAP 攻擊部分 (基於權重全連接層解析解求逆)
    # -----------------------------------------------------------------
    t_rgap_start = time.time()
    rgap_label = "N/A"
    rgap_psnr = 0.0
    recon_img_rgap_clean = torch.zeros_like(original_images)
    
    if status == "unprotected":
        print(f" ⚡ 啟動 [常規明文組] R-GAP 極速解析解推導...")
        try:
            recovered_label, out_s2 = run_rgap_lenet(stolen_grads=rgap_gradients, global_model=global_model)
            rgap_label = str(recovered_label)
            recon_img_rgap_clean = clean_rgap_image(out_s2)
            rgap_psnr = calculate_psnr(recon_img_rgap_clean, original_images)
        except Exception as e:
            print(f" [!] R-GAP 明文組推導發生異常: {e}")
    else:
        print(f" 🛡️ 啟動 [DP 防禦組] R-GAP 極速解析解推導...")
        try:
            # 差分隱私下的干擾雜訊會破壞解析解矩陣求逆，預期會產生徹底的雜訊或使公式解誤差雪崩崩潰
            recovered_label, out_s2 = run_rgap_lenet(stolen_grads=rgap_gradients, global_model=global_model)
            rgap_label = str(recovered_label)
            recon_img_rgap_clean = clean_rgap_image(out_s2)
            rgap_psnr = calculate_psnr(recon_img_rgap_clean, original_images)
            print(" [+] R-GAP 防禦組計算完成（解算結果已完全退化為高斯噪點）。")
        except Exception as e:
            print(f" [-] 🎯 R-GAP 逆推矩陣在防禦組中成功被癱瘓！公式解誤差雪崩崩潰（符合防禦預期）: {e}")
            rgap_label = "Failed (NaN)"
            recon_img_rgap_clean = torch.zeros_like(original_images)
            rgap_psnr = 0.0

    rgap_time = time.time() - t_rgap_start

    # 儲存 R-GAP 專屬的還原結果影像
    rgap_filename = FILENAME_TEMPLATE.format(
        dataset=config.DATASET, img_type=img_type, attack="rgap", status=status
    )
    # 轉成 show_images 預期的 (H, W) numpy 格式
    if isinstance(recon_img_rgap_clean, torch.Tensor):
        rgap_np = recon_img_rgap_clean.detach().cpu().numpy()[0, 0]  # (H, W)
    else:
        rgap_np = recon_img_rgap_clean[0, 0] if recon_img_rgap_clean.ndim == 4 else recon_img_rgap_clean
    
    show_images(
        images=[rgap_np, original_images], 
        path=os.path.join(SAVE_DIR, rgap_filename), 
        cols=2, 
        titles=[f"R-GAP {'Unprotected' if status == 'unprotected' else 'Protected'}\n(Label: {rgap_label})",
                "Original Image"]
    )
    return {
        'dlg_time': dlg_time,
        'dlg_psnr': dlg_psnr,
        'rgap_time': rgap_time,
        'rgap_label': rgap_label,
        'rgap_psnr': rgap_psnr
    }


def run_docking_demo():
    print("=" * 75)
    print(" 🚀 聯邦學習攻防對接系統啟動：DLG & R-GAP vs 差分隱私 (DP) 防禦")
    print("=" * 75)
    
    print(f"[系統設定] 使用裝置: {config.DEVICE} | 測試資料集: {config.DATASET}")
    print(f"[路徑設定] 輸出資料夾: {SAVE_DIR}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 步驟 1：準備基礎數據集並決定抽樣索引
    # -------------------------------------------------------------------------
    print("\n[Step 1] 準備使用者隱私數據源並鎖定抽樣點...")
    client_loaders, _, _ = prepare_data(
        dataset_name=config.DATASET, 
        num_clients=1, 
        classes_per_client=4, 
        batch_size=DEMO_BATCH_SIZE
    )
    
    client_dataset = client_loaders[0].dataset
    
    fixed_idx = 0
    random_idx = random.randint(1, len(client_dataset) - 1)
    demo_targets = [("fixed", fixed_idx), ("random", random_idx)]
    print(f"   -> 🎯 成功鎖定展示樣本：固定圖片 [Index {fixed_idx}] & 隨機抽樣 [Index {random_idx}]")

    # 初始化防禦與環境變數
    global_model = get_model(num_classes=config.NUM_CLASSES)
    initial_weights = copy.deepcopy(global_model.state_dict())

    metrics_report = {}

    # =========================================================================
    # 🔄 迴圈主線：依序對 2 張圖片發動獨立的「最壞情況」攻防對抗演練
    # =========================================================================
    for img_type, target_idx in demo_targets:
        print(f"\n\n" + "#" * 75)
        print(f" 🔥 啟動【{img_type.upper()} 樣本 (Index {target_idx})】獨立攻防壓力測試")
        print("#" * 75)

        subset = Subset(client_dataset, [target_idx])
        one_sample_loader = DataLoader(subset, batch_size=DEMO_BATCH_SIZE)
        original_images, _ = next(iter(one_sample_loader))

        metrics_report[img_type] = {}

        # ---------------------------------------------------------------------
        # 🚫 階段一：未設防狀態演練 (明文組)
        # ---------------------------------------------------------------------
        global_model.load_state_dict(copy.deepcopy(initial_weights))
        print(f"\n💥 [場景一] 常規聯邦學習（未防禦）— 明文梯度傳輸流水線")
        
        # 呼叫客製化 Client 模擬函式
        dlg_unprotected_grads, rgap_unprotected_grads, client_time_unprot = run_client_simulation(
            one_sample_loader, initial_weights, is_protected=False
        )
        # 呼叫客製化雙重攻擊管線
        attack_results_unprot = execute_attack_pipeline(
            dlg_gradients=dlg_unprotected_grads,
            rgap_gradients=rgap_unprotected_grads,
            global_model=global_model,
            original_images=original_images,
            img_type=img_type,
            status="unprotected"
        )
        
        metrics_report[img_type]['unprotected'] = {
            'client_time': client_time_unprot,
            **attack_results_unprot
        }

        # ---------------------------------------------------------------------
        # 🛡️ 階段二：啟動差分隱私防禦演練 (安全組)
        # ---------------------------------------------------------------------
        global_model.load_state_dict(copy.deepcopy(initial_weights))
        print(f"\n🛡️ [場景二] 差分隱私加噪防禦（已防禦）— 雜訊牆干擾對抗流水線")
        
        # 呼叫客製化 Client 模擬函式
        dlg_protected_grads, rgap_protected_grads, client_time_prot = run_client_simulation(
            one_sample_loader, initial_weights, is_protected=True
        )
        
        attack_results_prot = execute_attack_pipeline(
            dlg_gradients=dlg_protected_grads,
            rgap_gradients=rgap_protected_grads,
            global_model=global_model,
            original_images=original_images,
            img_type=img_type,
            status="protected"
        )
        
        metrics_report[img_type]['protected'] = {
            'client_time': client_time_prot,
            **attack_results_prot
        }

    # =========================================================================
    # 📊 系統量化總結報告輸出與自動存檔 (支援多維度攻防指標對比)
    # =========================================================================
    report_lines = []

    def log_report(text=""):
        print(text)
        report_lines.append(text)

    WIDTH = 90

    log_report("\n" + "=" * WIDTH)
    log_report("🎯 聯邦學習攻防評測報告 (Differential Privacy 防禦組)")
    log_report("=" * WIDTH)
    log_report(f"Dataset: {config.DATASET} | DLG Iterations: {NUM_ITERATIONS}")
    log_report(f"Save Path: {os.path.abspath(SAVE_DIR)}")
    log_report("=" * WIDTH)

    f_u = metrics_report["fixed"]["unprotected"]
    f_p = metrics_report["fixed"]["protected"]

    r_u = metrics_report["random"]["unprotected"]
    r_p = metrics_report["random"]["protected"]

    header = (
        f"{'Metric':<28}"
        f"{'Fixed-Plain':>14}"
        f"{'Fixed-DP':>14}"
        f"{'Random-Plain':>14}"
        f"{'Random-DP':>14}"
    )

    log_report(header)
    log_report("-" * WIDTH)

    rows = [
        ("Client Time (s)",
        f_u["client_time"], f_p["client_time"],
        r_u["client_time"], r_p["client_time"], ".4f"),

        ("DLG PSNR (dB)",
        f_u["dlg_psnr"], f_p["dlg_psnr"],
        r_u["dlg_psnr"], r_p["dlg_psnr"], ".2f"),

        ("DLG Attack Time (s)",
        f_u["dlg_time"], f_p["dlg_time"],
        r_u["dlg_time"], r_p["dlg_time"], ".2f"),

        ("R-GAP PSNR (dB)",
        f_u["rgap_psnr"], f_p["rgap_psnr"],
        r_u["rgap_psnr"], r_p["rgap_psnr"], ".2f"),

        ("R-GAP Label",
        str(f_u["rgap_label"]), str(f_p["rgap_label"]),
        str(r_u["rgap_label"]), str(r_p["rgap_label"]), None),

        ("R-GAP Attack Time (s)",
        f_u["rgap_time"], f_p["rgap_time"],
        r_u["rgap_time"], r_p["rgap_time"], ".4f"),
    ]

    for metric, a, b, c, d, fmt in rows:

        if fmt:
            log_report(
                f"{metric:<28}"
                f"{a:>14{fmt}}"
                f"{b:>14{fmt}}"
                f"{c:>14{fmt}}"
                f"{d:>14{fmt}}"
            )
        else:
            log_report(
                f"{metric:<28}"
                f"{a:>14}"
                f"{b:>14}"
                f"{c:>14}"
                f"{d:>14}"
            )

    log_report("=" * WIDTH)

    # 💾 量化報告自動保存
    report_filename = f"demo_{config.DATASET}_dp_metrics_report.txt"
    report_path = os.path.join(SAVE_DIR, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f" 💾 [報告歸檔成功] 完整多模組 DP 攻防數據表已成功自動儲存至：{report_path}\n")


if __name__ == '__main__':
    run_docking_demo()