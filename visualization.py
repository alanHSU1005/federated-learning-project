# ==============================================================================
# visualization.py — 基準實驗圖表繪製（書面報告核心）
#
# 此模組從 logs/ 目錄讀取 JSON 日誌，繪製以下圖表：
#
#   實驗一（Local Epochs 影響）：
#     - Accuracy vs. Global Rounds（各 E 值的曲線對比）
#     - Loss vs. Global Rounds（各 E 值的曲線對比）
#
#   實驗二（Batch Size 影響）：
#     - Accuracy vs. Global Rounds（各 B 值的曲線對比）
#     - Loss vs. Global Rounds（各 B 值的曲線對比）
#
#   附加圖表：
#     - 最佳準確率橫向比較長條圖（實驗一 vs 實驗二）
#
# 所有圖表儲存至 config.PLOT_DIR，同時顯示於螢幕（可透過 SHOW_PLOT 控制）。
# ==============================================================================

import os
import json
from typing import List, Dict, Optional

import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端，避免無顯示器的伺服器環境報錯
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import config
from FL_utils import (
    ensure_dir, load_experiment_log, load_all_logs_in_dir, summarize_experiment_results
)

# ------------------------------------------------------------------------------
# 圖表樣式全域設定
# ------------------------------------------------------------------------------

# 是否在繪圖後彈出互動式視窗（False = 只存檔，適合無 GUI 環境）
SHOW_PLOT = False

# 各變因值的預設顏色循環（最多支援 6 條曲線）
COLOR_CYCLE = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']

# 線條樣式循環（搭配顏色，讓黑白列印也能區分）
LINE_STYLES = ['-', '--', '-.', ':', '-', '--']

# 標記樣式循環
MARKERS = ['o', 's', '^', 'D', 'v', 'p']

# 全局字型設定（支援中文）
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    #註解下面一行
    #'savefig.bbox_inches': 'tight',
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
})


# ==============================================================================
# 核心繪圖函數
# ==============================================================================

def plot_accuracy_curves(
    logs: List[dict],
    title: str,
    variable_key: str,
    variable_label: str,
    save_filename: str,
    plot_dir: str = config.PLOT_DIR,
) -> str:
    """
    繪製「Accuracy vs. Global Rounds」折線圖（多條曲線對比）。

    每條曲線對應一個變因值（例如 E=1、E=3、E=5），
    圖例標示變因名稱與值，x 軸為通訊輪次，y 軸為測試準確率（%）。

    Args:
        logs (list[dict]): 多個實驗日誌字典（每個字典對應一條曲線）。
        title (str): 圖表標題。
        variable_key (str): 從日誌讀取的變因鍵名（'local_epochs' / 'batch_size'）。
        variable_label (str): 圖例中的變因標示前綴（如 'E' / 'B'）。
        save_filename (str): 儲存的檔名（含副檔名，例如 'exp1_accuracy.png'）。
        plot_dir (str): 圖表輸出目錄。

    Returns:
        str: 儲存完成的完整檔案路徑。
    """
    ensure_dir(plot_dir)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for idx, log in enumerate(logs):
        rounds = [entry['round'] for entry in log['history']]
        accs = [entry['test_accuracy_pct'] for entry in log['history']]
        val = log.get(variable_key, '?')
        best_acc = log.get('best_test_accuracy', 0.0) * 100

        ax.plot(
            rounds, accs,
            color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
            linestyle=LINE_STYLES[idx % len(LINE_STYLES)],
            marker=MARKERS[idx % len(MARKERS)],
            markersize=4,
            markevery=max(1, len(rounds) // 10),  # 每 10% 顯示一個標記點，避免擁擠
            linewidth=1.8,
            label=f'{variable_label}={val}  (best: {best_acc:.2f}%)',
        )

    # 座標軸與標籤設定
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Global Communication Rounds', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_xlim(left=1)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.grid(True)
    ax.legend(loc='lower right', framealpha=0.85)

    # 在圖表右側標注資料集名稱（直接從 log 取得，不使用 config 補值）
    dataset_label = logs[0].get('dataset', 'Unknown') if logs else 'Unknown'
    ax.annotate(
        f'Dataset: {dataset_label}',
        xy=(0.98, 0.05), xycoords='axes fraction',
        ha='right', fontsize=9, color='gray',
    )

    plt.tight_layout()
    save_path = os.path.join(plot_dir, save_filename)
    plt.savefig(save_path)
    print(f"[visualization] 圖表已儲存：{save_path}")

    if SHOW_PLOT:
        plt.show()
    plt.close(fig)
    return save_path


def plot_loss_curves(
    logs: List[dict],
    title: str,
    variable_key: str,
    variable_label: str,
    save_filename: str,
    plot_dir: str = config.PLOT_DIR,
) -> str:
    """
    繪製「Test Loss vs. Global Rounds」折線圖（多條曲線對比）。

    Args:
        logs (list[dict]): 多個實驗日誌字典。
        title (str): 圖表標題。
        variable_key (str): 從日誌讀取的變因鍵名。
        variable_label (str): 圖例變因標示前綴（如 'E' / 'B'）。
        save_filename (str): 儲存檔名。
        plot_dir (str): 圖表輸出目錄。

    Returns:
        str: 儲存完成的完整檔案路徑。
    """
    ensure_dir(plot_dir)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for idx, log in enumerate(logs):
        rounds = [entry['round'] for entry in log['history']]
        losses = [entry['test_loss'] for entry in log['history']]
        val = log.get(variable_key, '?')

        ax.plot(
            rounds, losses,
            color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
            linestyle=LINE_STYLES[idx % len(LINE_STYLES)],
            marker=MARKERS[idx % len(MARKERS)],
            markersize=4,
            markevery=max(1, len(rounds) // 10),
            linewidth=1.8,
            label=f'{variable_label}={val}',
        )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Global Communication Rounds', fontsize=12)
    ax.set_ylabel('Test Loss (Cross-Entropy)', fontsize=12)
    ax.set_xlim(left=1)
    ax.grid(True)
    ax.legend(loc='upper right', framealpha=0.85)

    dataset_label = logs[0].get('dataset', 'Unknown') if logs else 'Unknown'
    ax.annotate(
        f'Dataset: {dataset_label}',
        xy=(0.98, 0.95), xycoords='axes fraction',
        ha='right', fontsize=9, color='gray',
    )

    plt.tight_layout()
    save_path = os.path.join(plot_dir, save_filename)
    plt.savefig(save_path)
    print(f"[visualization] 圖表已儲存：{save_path}")

    if SHOW_PLOT:
        plt.show()
    plt.close(fig)
    return save_path


def plot_best_accuracy_bar(
    logs_exp1: List[dict],
    logs_exp2: List[dict],
    save_filename: str = 'best_accuracy_comparison.png',
    plot_dir: str = config.PLOT_DIR,
) -> str:
    """
    繪製實驗一與實驗二最佳準確率的橫向比較長條圖。

    X 軸：各實驗組合的標識（如 E=1, E=3, E=5, B=16, B=32, B=64）
    Y 軸：最佳測試準確率（%）

    適合書面報告中展示各超參數組合的最終表現對比。

    Args:
        logs_exp1 (list[dict]): 實驗一（Local Epochs）的日誌列表。
        logs_exp2 (list[dict]): 實驗二（Batch Size）的日誌列表。
        save_filename (str): 儲存檔名。
        plot_dir (str): 圖表輸出目錄。

    Returns:
        str: 儲存完成的完整檔案路徑。
    """
    ensure_dir(plot_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    def _draw_bar(ax, logs, variable_key, variable_label, color_offset=0):
        labels = [f"{variable_label}={log[variable_key]}" for log in logs]
        best_accs = [log['best_test_accuracy'] * 100 for log in logs]
        colors = [COLOR_CYCLE[(i + color_offset) % len(COLOR_CYCLE)] for i in range(len(logs))]

        bars = ax.bar(labels, best_accs, color=colors, width=0.5, edgecolor='white', linewidth=1.2)

        # 在長條頂端標注數值
        for bar, acc in zip(bars, best_accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
            )

        ax.set_ylabel('Best Test Accuracy (%)', fontsize=11)
        ax.set_ylim(0, min(100, max(best_accs) * 1.12))
        ax.grid(axis='y', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 實驗一
    _draw_bar(axes[0], logs_exp1, 'local_epochs', 'E', color_offset=0)
    axes[0].set_title('Exp 1: Effect of Local Epochs (E)', fontweight='bold')

    # 實驗二
    _draw_bar(axes[1], logs_exp2, 'batch_size', 'B', color_offset=3)
    axes[1].set_title('Exp 2: Effect of Batch Size (B)', fontweight='bold')

    dataset_label = logs_exp1[0].get('dataset', 'Unknown') if logs_exp1 else 'Unknown'
    fig.suptitle(
        f'Best Accuracy Comparison  |  Dataset: {dataset_label}',
        fontsize=13, fontweight='bold', y=1.02,
    )

    plt.tight_layout()
    save_path = os.path.join(plot_dir, save_filename)
    plt.savefig(save_path)
    print(f"[visualization] 比較圖表已儲存：{save_path}")

    if SHOW_PLOT:
        plt.show()
    plt.close(fig)
    return save_path


def plot_train_vs_test_accuracy(
    log: dict,
    save_filename: str,
    plot_dir: str = config.PLOT_DIR,
) -> str:
    """
    繪製單次實驗中「訓練準確率 vs 測試準確率」的對比折線圖。

    可用於觀察模型是否過擬合（Overfitting）或欠擬合（Underfitting），
    適合在書面報告的分析段落中補充使用。

    Args:
        log (dict): 單個實驗的日誌字典。
        save_filename (str): 儲存檔名。
        plot_dir (str): 圖表輸出目錄。

    Returns:
        str: 儲存完成的完整檔案路徑。
    """
    ensure_dir(plot_dir)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    rounds = [entry['round'] for entry in log['history']]
    train_accs = [entry['avg_train_acc'] * 100 for entry in log['history']]
    test_accs = [entry['test_accuracy_pct'] for entry in log['history']]

    ax.plot(rounds, train_accs, color='#F44336', linestyle='--',
            marker='s', markersize=3, markevery=5, linewidth=1.8, label='Train Accuracy')
    ax.plot(rounds, test_accs, color='#2196F3', linestyle='-',
            marker='o', markersize=3, markevery=5, linewidth=1.8, label='Test Accuracy')

    var_key = log.get('variable_key', '')
    var_val = log.get('variable_value', '')
    ax.set_title(f'Train vs. Test Accuracy  ({var_key}={var_val})', fontweight='bold')
    ax.set_xlabel('Global Communication Rounds')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim(left=1)
    ax.grid(True)
    ax.legend(loc='lower right', framealpha=0.85)

    plt.tight_layout()
    save_path = os.path.join(plot_dir, save_filename)
    plt.savefig(save_path)
    print(f"[visualization] 訓練/測試準確率對比圖已儲存：{save_path}")

    if SHOW_PLOT:
        plt.show()
    plt.close(fig)
    return save_path


# ==============================================================================
# 高層入口：一鍵生成所有報告圖表
# ==============================================================================

def generate_all_plots(log_dir: str = config.LOG_DIR, plot_dir: str = config.PLOT_DIR) -> None:
    """
    讀取所有實驗日誌，以 dataset → experiment 雙層分組後，
    為每個 dataset 獨立生成完整的報告圖表集。

    【分組規則】
        - 主鍵：log["dataset"]（不使用 config.DATASET 補值）
        - 次鍵：log["experiment"]
        - 缺少 "dataset"、"experiment" 或 "history" 的 log 一律 skip

    【輸出目錄結構】
        plots/
            MNIST/
                exp1_accuracy_curves.png
                exp1_loss_curves.png
                exp2_accuracy_curves.png
                exp2_loss_curves.png
                best_accuracy_comparison.png
                exp1_train_vs_test_E*.png
                exp2_train_vs_test_B*.png
            ATT_FACE/
                （同上，完全獨立）

    任何 dataset 之間嚴格禁止跨組繪圖或統計。

    Args:
        log_dir (str): JSON 日誌所在目錄。
        plot_dir (str): 圖表根輸出目錄（各 dataset 會在其下建立子目錄）。
    """
    print(f"\n{'='*60}")
    print(f"  開始生成所有報告圖表")
    print(f"  日誌來源：{log_dir}")
    print(f"  圖表根目錄：{plot_dir}")
    print(f"{'='*60}\n")

    # ── 步驟 1：載入並驗證所有 log，按 dataset 分桶 ──────────────────────
    dataset_buckets = _load_and_group_logs_by_dataset(log_dir)

    if not dataset_buckets:
        print("[visualization] 警告：logs 目錄中沒有任何有效日誌，結束。")
        return

    total_generated = []

    # ── 步驟 2：對每個 dataset 獨立處理 ──────────────────────────────────
    for dataset_name, exp_map in sorted(dataset_buckets.items()):
        print(f"\n{'─'*60}")
        print(f"  [Dataset: {dataset_name}]")
        print(f"{'─'*60}")

        # 此 dataset 的專屬輸出目錄：plots/<dataset>/
        ds_plot_dir = os.path.join(plot_dir, dataset_name)
        ensure_dir(ds_plot_dir)

        # 從 exp_map 取出實驗一與實驗二的 log 列表（不存在則為空 list）
        logs_exp1 = exp_map.get('exp1_local_epochs', [])
        logs_exp2 = exp_map.get('exp2_batch_size', [])

        generated = _generate_plots_for_dataset(
            dataset_name=dataset_name,
            logs_exp1=logs_exp1,
            logs_exp2=logs_exp2,
            ds_plot_dir=ds_plot_dir,
        )
        total_generated.extend(generated)

    # ── 步驟 3：完成報告 ──────────────────────────────────────────────────
    print(f"\n[visualization] 全部圖表生成完畢，共 {len(total_generated)} 張：")
    for f in total_generated:
        print(f"  ✓  {f}")


def _load_and_group_logs_by_dataset(log_dir: str) -> dict:
    """
    掃描 log_dir 下所有 .json 檔案，驗證後以 dataset → experiment 雙層字典回傳。

    【驗證規則（不符合則 skip）】
        - log["dataset"] 必須存在且非空字串（禁止用 config 補值）
        - log["experiment"] 必須存在
        - log["history"] 必須存在且為非空 list

    Args:
        log_dir (str): 日誌目錄。

    Returns:
        dict: 結構為 { dataset_name: { experiment_mode: [log, ...] } }
              例如：{
                  'MNIST': {
                      'exp1_local_epochs': [log_E1, log_E3, log_E5],
                      'exp2_batch_size':   [log_B16, log_B32, log_B64],
                  },
                  'ATT_FACE': { ... }
              }
    """
    ensure_dir(log_dir)
    buckets: dict = {}
    skipped = 0

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith('.json'):
            continue

        filepath = os.path.join(log_dir, fname)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                log = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [visualization] ⚠ 無法讀取 {fname}：{e}，跳過。")
            skipped += 1
            continue

        # ── 欄位驗證（缺少任一欄位直接 skip，不補預設值）────────────────
        dataset = log.get('dataset')
        experiment = log.get('experiment')
        history = log.get('history')

        if not dataset or not isinstance(dataset, str):
            print(f"  [visualization] ⚠ {fname} 缺少有效的 'dataset' 欄位，跳過。")
            skipped += 1
            continue

        if not experiment or not isinstance(experiment, str):
            print(f"  [visualization] ⚠ {fname} 缺少有效的 'experiment' 欄位，跳過。")
            skipped += 1
            continue

        if not history or not isinstance(history, list) or len(history) == 0:
            print(f"  [visualization] ⚠ {fname} 的 'history' 為空或無效，跳過。")
            skipped += 1
            continue

        # ── 放入對應的 dataset bucket ────────────────────────────────────
        if dataset not in buckets:
            buckets[dataset] = {}
        if experiment not in buckets[dataset]:
            buckets[dataset][experiment] = []
        buckets[dataset][experiment].append(log)

    # 統計報告
    total_valid = sum(
        len(exp_logs)
        for ds_map in buckets.values()
        for exp_logs in ds_map.values()
    )
    print(f"[visualization] 掃描完成：有效日誌 {total_valid} 筆，跳過 {skipped} 筆。")
    for ds, exp_map in sorted(buckets.items()):
        for exp, logs in sorted(exp_map.items()):
            print(f"  {ds} / {exp}：{len(logs)} 筆")

    return buckets


def _generate_plots_for_dataset(
    dataset_name: str,
    logs_exp1: List[dict],
    logs_exp2: List[dict],
    ds_plot_dir: str,
) -> List[str]:
    """
    對單一 dataset 的兩組實驗日誌，生成該 dataset 下所有圖表。

    此函數只處理同一 dataset 的資料，確保完全隔離。
    兩組 log 均可為空（只有 exp1 或只有 exp2 時，正常畫可用的部分）。

    Args:
        dataset_name (str): 資料集名稱（用於圖表標題與 warning 訊息）。
        logs_exp1 (list[dict]): 實驗一（exp1_local_epochs）的日誌列表。
        logs_exp2 (list[dict]): 實驗二（exp2_batch_size）的日誌列表。
        ds_plot_dir (str): 此 dataset 的圖表輸出目錄（plots/<dataset>/）。

    Returns:
        list[str]: 本次成功儲存的所有圖表路徑。
    """
    generated = []

    # ── 實驗一：Local Epochs 影響 ──────────────────────────────────────────
    if logs_exp1:
        logs_exp1 = sorted(logs_exp1, key=lambda x: x.get('local_epochs', 0))
        fixed_B = logs_exp1[0].get('batch_size', '?')

        path = plot_accuracy_curves(
            logs=logs_exp1,
            title=f'Exp 1: Effect of Local Epochs (E) on Test Accuracy\n'
                  f'[Dataset: {dataset_name}, B={fixed_B}]',
            variable_key='local_epochs',
            variable_label='E',
            save_filename='exp1_accuracy_curves.png',
            plot_dir=ds_plot_dir,
        )
        generated.append(path)

        path = plot_loss_curves(
            logs=logs_exp1,
            title=f'Exp 1: Effect of Local Epochs (E) on Test Loss\n'
                  f'[Dataset: {dataset_name}, B={fixed_B}]',
            variable_key='local_epochs',
            variable_label='E',
            save_filename='exp1_loss_curves.png',
            plot_dir=ds_plot_dir,
        )
        generated.append(path)

        # 訓練 vs 測試準確率對比：取中間 E 值的 log（例如 E=3）
        mid_log = logs_exp1[len(logs_exp1) // 2]
        e_val = mid_log.get('local_epochs', '?')
        path = plot_train_vs_test_accuracy(
            log=mid_log,
            save_filename=f'exp1_train_vs_test_E{e_val}.png',
            plot_dir=ds_plot_dir,
        )
        generated.append(path)
    else:
        print(f"  [visualization] ⚠ [{dataset_name}] 找不到 exp1_local_epochs 日誌，跳過實驗一圖表。")

    # ── 實驗二：Batch Size 影響 ────────────────────────────────────────────
    if logs_exp2:
        logs_exp2 = sorted(logs_exp2, key=lambda x: x.get('batch_size', 0))
        fixed_E = logs_exp2[0].get('local_epochs', '?')

        path = plot_accuracy_curves(
            logs=logs_exp2,
            title=f'Exp 2: Effect of Batch Size (B) on Test Accuracy\n'
                  f'[Dataset: {dataset_name}, E={fixed_E}]',
            variable_key='batch_size',
            variable_label='B',
            save_filename='exp2_accuracy_curves.png',
            plot_dir=ds_plot_dir,
        )
        generated.append(path)

        path = plot_loss_curves(
            logs=logs_exp2,
            title=f'Exp 2: Effect of Batch Size (B) on Test Loss\n'
                  f'[Dataset: {dataset_name}, E={fixed_E}]',
            variable_key='batch_size',
            variable_label='B',
            save_filename='exp2_loss_curves.png',
            plot_dir=ds_plot_dir,
        )
        generated.append(path)

        mid_log = logs_exp2[len(logs_exp2) // 2]
        b_val = mid_log.get('batch_size', '?')
        path = plot_train_vs_test_accuracy(
            log=mid_log,
            save_filename=f'exp2_train_vs_test_B{b_val}.png',
            plot_dir=ds_plot_dir,
        )
        generated.append(path)
    else:
        print(f"  [visualization] ⚠ [{dataset_name}] 找不到 exp2_batch_size 日誌，跳過實驗二圖表。")

    # ── 橫向比較長條圖（限同 dataset，exp1 與 exp2 均有資料時才畫）────────
    if logs_exp1 and logs_exp2:
        path = plot_best_accuracy_bar(
            logs_exp1=logs_exp1,
            logs_exp2=logs_exp2,
            save_filename='best_accuracy_comparison.png',
            plot_dir=ds_plot_dir,
        )
        generated.append(path)

        # 文字摘要（限同 dataset 內的 log）
        summarize_experiment_results(logs_exp1 + logs_exp2)
    else:
        print(f"  [visualization] ⚠ [{dataset_name}] exp1 或 exp2 缺一，跳過 best_accuracy_comparison。")

    return generated


# ==============================================================================
# 直接執行此檔案時：從現有日誌重新生成所有圖表
# ==============================================================================

if __name__ == '__main__':
    print("=== visualization.py 獨立執行模式 ===")
    print(f"從 '{config.LOG_DIR}' 讀取日誌，輸出至 '{config.PLOT_DIR}'")
    generate_all_plots(log_dir=config.LOG_DIR, plot_dir=config.PLOT_DIR)
