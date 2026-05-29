# ==============================================================================
# main.py — 聯邦學習專案主入口
#
# 執行流程：
#   1. 讀取 config.py 的實驗配置
#   2. 依據 EXPERIMENT_MODE 分派執行邏輯：
#      - 'exp1_local_epochs'  → 對每個 E 值跑完整 FL 訓練
#      - 'exp2_batch_size'    → 對每個 B 值跑完整 FL 訓練
#      - 'single_run'         → 使用預設參數執行一次
#   3. 每次完整 FL 訓練結束後儲存 JSON 日誌
#   4. 所有實驗完成後呼叫 visualization.py 生成所有報告圖表
#
# 【切換實驗方式】
#   只需修改 config.py 中的：
#     DATASET            → 切換 MNIST / ATT_FACE
#     EXPERIMENT_MODE    → 切換實驗模式
#   無需修改此檔案。
#
# 【對攻擊 / 防禦模組的整合說明】
#   - 攻擊模組（Step 2）：在 run_single_experiment() 的 FL loop 中，
#     可透過 clients[i].compute_gradients_on_batch() 取得梯度，
#     或在每輪 server.run_round() 後存取 client.pre/post_train_weights。
#   - 防禦模組（Step 3）：將防禦函數傳入 server.run_round() 的
#     gradient_hook 或 aggregation_fn 參數即可，無需修改 main.py 本體。
# ==============================================================================

import os
import copy

import config
from FL_utils import (
    set_seed,
    ensure_dir,
    save_experiment_log,
    build_round_history_entry,
    build_experiment_log,
    print_experiment_header,
    print_round_header,
    RoundTimer,
)
from data_loader import prepare_data
from model import get_model
from client import Client
from server import Server
from evaluate import quick_evaluate, full_evaluation_report
from visualization import generate_all_plots


# ==============================================================================
# 核心函數：執行一次完整的 FL 實驗
# ==============================================================================

def run_single_experiment(
    local_epochs: int,
    batch_size: int,
    experiment_mode: str,
    variable_key: str,
    variable_value,
    log_filename: str,
    gradient_hook=None,    # 防禦模組（Step 3）傳入梯度後處理函數
    aggregation_fn=None,   # 防禦模組（Step 3）傳入自訂聚合函數
) -> dict:
    """
    執行一次完整的聯邦學習訓練（固定 local_epochs 與 batch_size）。

    函數會依序完成：資料準備 → 客戶端建立 → FL 訓練迴圈 → 評估 → 日誌儲存。

    Args:
        local_epochs (int): 本次實驗的 local epochs（E）。
        batch_size (int): 本次實驗的 batch size（B）。
        experiment_mode (str): 實驗模式識別字串（供日誌使用）。
        variable_key (str): 變因鍵名（'local_epochs' 或 'batch_size'）。
        variable_value: 本次實驗的變因值。
        log_filename (str): 日誌輸出檔名。
        gradient_hook (Callable, optional): 傳入防禦模組的梯度後處理函數。
        aggregation_fn (Callable, optional): 傳入防禦模組的自訂聚合函數。

    Returns:
        dict: 本次實驗的完整日誌字典。
    """

    # -------------------------------------------------------------------------
    # 步驟 0：設定隨機種子（確保每次實驗的資料切分與初始化一致）
    # -------------------------------------------------------------------------
    set_seed(config.SEED)

    print_experiment_header(
        exp_mode=experiment_mode,
        dataset=config.DATASET,
        variable_name=variable_key,
        value=variable_value,
    )

    # -------------------------------------------------------------------------
    # 步驟 1：資料準備（載入、Non-IID 切分、建立 DataLoader）
    # -------------------------------------------------------------------------
    client_loaders, test_loader, client_indices = prepare_data(
        dataset_name=config.DATASET,
        num_clients=config.NUM_CLIENTS,
        classes_per_client=config.CLASSES_PER_CLIENT,
        batch_size=batch_size,
        seed=config.SEED,
    )

    # -------------------------------------------------------------------------
    # 步驟 2：建立 Server 與 Clients
    # -------------------------------------------------------------------------
    server = Server(num_classes=config.NUM_CLASSES)

    clients = []
    print(f"\n[main] 建立 {config.NUM_CLIENTS} 個客戶端...")
    for i in range(config.NUM_CLIENTS):
        client = Client(
            client_id=i,
            dataloader=client_loaders[i],
            num_classes=config.NUM_CLASSES,
        )
        clients.append(client)

    server.register_clients(clients)

    # -------------------------------------------------------------------------
    # 步驟 3：FL 訓練主迴圈（Global Rounds）
    # -------------------------------------------------------------------------
    history = []
    timer = RoundTimer(total_rounds=config.GLOBAL_ROUNDS)
    timer.start()

    print(f"\n[main] 開始 FL 訓練迴圈（共 {config.GLOBAL_ROUNDS} 輪）...\n")

    for round_num in range(1, config.GLOBAL_ROUNDS + 1):

        print_round_header(
            round_num=round_num,
            total_rounds=config.GLOBAL_ROUNDS,
            local_epochs=local_epochs,
            batch_size=batch_size,
        )

        # ── 執行一輪聯邦學習（廣播 → 本地訓練 → 聚合）──────────────────
        #
        # 【攻擊模組整合點（Step 2）】
        #   若要在每輪訓練後對某個客戶端執行梯度洩漏攻擊，
        #   可在 server.run_round() 回傳後加入攻擊邏輯，例如：
        #
        #     round_log = server.run_round(...)
        #     # 取得第 0 個客戶端的虛擬梯度
        #     pseudo_grads = clients[0].get_pseudo_gradients()
        #     # 呼叫攻擊模組（由 Step 2 組員實作）
        #     reconstructed_imgs = attack_module.dlg_attack(
        #         model=server.get_global_model(),
        #         target_gradients=pseudo_grads,
        #     )
        #
        # 【防禦模組整合點（Step 3）】
        #   將防禦函數傳入 gradient_hook 或 aggregation_fn 參數：
        #
        #     def my_clip_hook(model):
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #
        #     round_log = server.run_round(
        #         local_epochs=local_epochs,
        #         batch_size=batch_size,
        #         gradient_hook=my_clip_hook,
        #     )
        # ──────────────────────────────────────────────────────────────────

        round_log = server.run_round(
            local_epochs=local_epochs,
            batch_size=batch_size,
            gradient_hook=gradient_hook,
            aggregation_fn=aggregation_fn,
            verbose=True,
        )

        # ── 評估全局模型在測試集上的效能 ──────────────────────────────────
        test_acc, test_loss = quick_evaluate(
            model=server.get_global_model(),
            test_loader=test_loader,
            device=config.DEVICE,
        )

        print(
            f"  [Eval] 測試準確率：{test_acc*100:.2f}%  |  測試損失：{test_loss:.4f}"
        )

        # ── 建立本輪歷史記錄 ────────────────────────────────────────────────
        entry = build_round_history_entry(
            round_num=round_num,
            test_accuracy=test_acc,
            test_loss=test_loss,
            avg_train_loss=round_log['avg_train_loss'],
            avg_train_acc=round_log['avg_train_acc'],
        )
        history.append(entry)

        # ── 計時 ────────────────────────────────────────────────────────────
        timer.lap(round_num=round_num)

    # -------------------------------------------------------------------------
    # 步驟 4：訓練結束後的完整評估
    # -------------------------------------------------------------------------
    run_label = f"{variable_key}={variable_value}"
    full_evaluation_report(
        model=server.get_global_model(),
        test_loader=test_loader,
        run_label=run_label,
        device=config.DEVICE,
    )

    # -------------------------------------------------------------------------
    # 步驟 5：組裝並儲存實驗日誌
    # -------------------------------------------------------------------------
    log_data = build_experiment_log(
        experiment_mode=experiment_mode,
        dataset=config.DATASET,
        variable_key=variable_key,
        variable_value=variable_value,
        global_rounds=config.GLOBAL_ROUNDS,
        local_epochs=local_epochs,
        batch_size=batch_size,
        history=history,
    )
    save_experiment_log(log_data, filename=log_filename, log_dir=config.LOG_DIR)

    return log_data


# ==============================================================================
# 實驗一：測試不同 Local Epochs（E）
# ==============================================================================

def run_exp1_local_epochs() -> None:
    """
    實驗一主控函數。

    依序對 config.EXP1_LOCAL_EPOCHS_LIST 中的每個 E 值執行完整 FL 訓練，
    批次大小固定為 config.EXP1_FIXED_BATCH_SIZE。
    """
    print(f"\n{'#'*60}")
    print(f"  實驗一啟動：測試不同 Local Epochs 的影響")
    print(f"  變因列表 E = {config.EXP1_LOCAL_EPOCHS_LIST}")
    print(f"  固定批次大小 B = {config.EXP1_FIXED_BATCH_SIZE}")
    print(f"{'#'*60}\n")

    for E in config.EXP1_LOCAL_EPOCHS_LIST:
        log_filename = f"exp1_{config.DATASET}_E{E}_B{config.EXP1_FIXED_BATCH_SIZE}.json"
        run_single_experiment(
            local_epochs=E,
            batch_size=config.EXP1_FIXED_BATCH_SIZE,
            experiment_mode='exp1_local_epochs',
            variable_key='local_epochs',
            variable_value=E,
            log_filename=log_filename,
        )
        print(f"\n✅ 實驗一 E={E} 完成，日誌已儲存。\n")


# ==============================================================================
# 實驗二：測試不同 Batch Size（B）
# ==============================================================================

def run_exp2_batch_size() -> None:
    """
    實驗二主控函數。

    依序對 config.EXP2_BATCH_SIZE_LIST 中的每個 B 值執行完整 FL 訓練，
    本地訓練輪次固定為 config.EXP2_FIXED_LOCAL_EPOCHS。
    """
    print(f"\n{'#'*60}")
    print(f"  實驗二啟動：測試不同 Batch Size 的影響")
    print(f"  變因列表 B = {config.EXP2_BATCH_SIZE_LIST}")
    print(f"  固定本地輪次 E = {config.EXP2_FIXED_LOCAL_EPOCHS}")
    print(f"{'#'*60}\n")

    for B in config.EXP2_BATCH_SIZE_LIST:
        log_filename = f"exp2_{config.DATASET}_E{config.EXP2_FIXED_LOCAL_EPOCHS}_B{B}.json"
        run_single_experiment(
            local_epochs=config.EXP2_FIXED_LOCAL_EPOCHS,
            batch_size=B,
            experiment_mode='exp2_batch_size',
            variable_key='batch_size',
            variable_value=B,
            log_filename=log_filename,
        )
        print(f"\n✅ 實驗二 B={B} 完成，日誌已儲存。\n")


# ==============================================================================
# 單次執行（single_run 模式）
# ==============================================================================

def run_single() -> None:
    """
    單次執行模式，使用 config.py 中 DEFAULT_* 參數。

    適合快速驗證整個系統的連通性，
    或供組員（Step 2 / Step 3）整合攻擊 / 防禦模組後的測試使用。
    """
    print(f"\n{'#'*60}")
    print(f"  單次執行模式（Single Run）")
    print(f"  E={config.DEFAULT_LOCAL_EPOCHS}, B={config.DEFAULT_BATCH_SIZE}")
    print(f"{'#'*60}\n")

    run_single_experiment(
        local_epochs=config.DEFAULT_LOCAL_EPOCHS,
        batch_size=config.DEFAULT_BATCH_SIZE,
        experiment_mode='single_run',
        variable_key='default',
        variable_value=f"E{config.DEFAULT_LOCAL_EPOCHS}_B{config.DEFAULT_BATCH_SIZE}",
        log_filename=f"single_run_{config.DATASET}.json",
    )


# ==============================================================================
# 主入口
# ==============================================================================

def main() -> None:
    """
    專案主入口函數。

    根據 config.EXPERIMENT_MODE 分派對應的實驗執行函數，
    全部實驗完成後呼叫 visualization.py 生成所有報告圖表。
    """
    # 確保輸出目錄存在
    ensure_dir(config.LOG_DIR)
    ensure_dir(config.PLOT_DIR)

    print(f"\n{'='*60}")
    print(f"  聯邦學習基準實驗系統")
    print(f"  資料集：{config.DATASET}  |  模式：{config.EXPERIMENT_MODE}")
    print(f"  全局輪次：{config.GLOBAL_ROUNDS}  |  客戶端數：{config.NUM_CLIENTS}")
    print(f"{'='*60}")

    # ── 依模式分派 ─────────────────────────────────────────────────────────
    mode = config.EXPERIMENT_MODE

    if mode == 'exp1_local_epochs':
        run_exp1_local_epochs()

    elif mode == 'exp2_batch_size':
        run_exp2_batch_size()

    elif mode == 'single_run':
        run_single()

    elif mode == 'full':
        # 一次性執行所有實驗（依序完成實驗一與實驗二）
        run_exp1_local_epochs()
        run_exp2_batch_size()

    else:
        raise ValueError(
            f"不支援的 EXPERIMENT_MODE：'{mode}'\n"
            f"請在 config.py 中選擇：'exp1_local_epochs' / 'exp2_batch_size' / 'single_run' / 'full'"
        )

    # ── 生成所有報告圖表 ───────────────────────────────────────────────────
    print(f"\n[main] 所有訓練完成，開始生成報告圖表...\n")
    generate_all_plots(log_dir=config.LOG_DIR, plot_dir=config.PLOT_DIR)

    print(f"\n{'='*60}")
    print(f"  全部實驗流程完成！")
    print(f"  日誌位置：{os.path.abspath(config.LOG_DIR)}")
    print(f"  圖表位置：{os.path.abspath(config.PLOT_DIR)}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
