import torch
import torch.nn as nn
import numpy as np

# 從同一個資料夾匯入官方底層工具 (注意路徑)
from rgap_attack.conv2circulant import generate_coordinates, aggregate_g, circulant_w

# ==============================================================================
# 1. 官方 R-GAP 線性代數求解器
# ==============================================================================

def fcn_reconstruction(k, gradient):
    x = [g / c for g, c in zip(gradient, k) if c != 0]
    if len(x) == 0:
        return np.zeros(gradient.shape[1], dtype=np.float32)
    x = np.mean(x, 0)
    return x

def peeling(in_shape, padding):
    """處理卷積層的 Padding 邊界條件約束"""
    if padding == 0:
        # 將原本的 .squeeze() 改為 .flatten()，確保回傳一維陣列
        return np.ones(shape=in_shape, dtype=bool).flatten()
        
    h, w = np.array(in_shape[-2:]) + 2 * padding
    toremain = np.ones(h * w * in_shape[1], dtype=bool)
    if padding:
        for c in range(in_shape[1]):
            for row in range(h):
                for col in range(w):
                    if col < padding or w - col <= padding or row < padding or h - row <= padding:
                        i = c * h * w + row * w + col
                        assert toremain[i]
                        toremain[i] = False
    return toremain

def padding_constraints(in_shape, padding):
    toremain = peeling(in_shape, padding)
    P = []
    for i in range(toremain.size):
        if not toremain[i]:
            P_row = np.zeros(toremain.size, dtype=np.float32)
            P_row[i] = 1
            P.append(P_row)
    return np.array(P)

def cnn_reconstruction(in_shape, k, g, out, kernel, stride, padding):
    coors, x_len, y_len = generate_coordinates(x_shape=in_shape, kernel=kernel, stride=stride, padding=padding)
    K = aggregate_g(k=k, x_len=x_len, coors=coors)
    W = circulant_w(x_len=x_len, kernel=kernel, coors=coors, y_len=y_len)
    P = padding_constraints(in_shape=in_shape, padding=padding)
    p = np.zeros(shape=P.shape[0], dtype=np.float32)
    
    if np.any(P):
        a = np.concatenate((K, W, P), axis=0)
        b = np.concatenate((g.reshape(-1), out.reshape(-1), p), axis=0)
    else:
        a = np.concatenate((K, W), axis=0)
        b = np.concatenate((g.reshape(-1), out.reshape(-1)), axis=0)
        
    result = np.linalg.lstsq(a, b, rcond=None)
    x = result[0]
    return x[peeling(in_shape=in_shape, padding=padding)], W


# ==============================================================================
# 2. LeNet 適配器 (Tanh 與 AvgPool)
# ==============================================================================

def derive_tanh(x_post):
    return 1.0 - np.power(x_post, 2)

def inverse_tanh(x_post):
    x_clamped = np.clip(x_post, -0.9999, 0.9999)
    return np.arctanh(x_clamped)

def inverse_avgpool2d(out, kernel_size=2, stride=2):
    out_tensor = torch.tensor(out, dtype=torch.float32)
    inversed = torch.nn.functional.interpolate(
        out_tensor.unsqueeze(0) if len(out_tensor.shape) == 3 else out_tensor, 
        scale_factor=stride, 
        mode='nearest'
    )
    if len(out_tensor.shape) == 3:
        inversed = inversed.squeeze(0)
    return inversed.numpy()


# ==============================================================================
# 3. 攻擊主程式 (還原28x28 原始影像)
# ==============================================================================

def run_rgap_lenet(stolen_grads: dict, global_model: nn.Module):
    """
    執行 R-GAP 攻擊 (針對 LeNet)
    回傳: (還原的標籤, 終極 28x28 原始影像 numpy array)
    """
    print("\n[R-GAP] 啟動針對 LeNet 的完美適配攻擊...")

    # 1. 轉 Numpy
    grads_np = {k: v.cpu().numpy() for k, v in stolen_grads.items()}
    weights_np = {k: v.cpu().numpy() for k, v in global_model.state_dict().items()}

    # -------------------------------------------------------------------------
    # 步驟 1: Output 層 (classifier.4)
    # -------------------------------------------------------------------------
    bias_grad_out = grads_np['classifier.4.bias']
    weight_grad_out = grads_np['classifier.4.weight']
    recovered_label = np.argmin(bias_grad_out)
    print(f"  [+] 瞬間還原標籤: {recovered_label}")
    
    k = bias_grad_out.reshape(-1, 1)
    # 🚀 修復：分兩行明確賦值
    x_out = fcn_reconstruction(k=k, gradient=weight_grad_out)
    last_weight = weights_np['classifier.4.weight']
    
    # -------------------------------------------------------------------------
    # 步驟 2: F6 層 (classifier.2)
    # -------------------------------------------------------------------------
    out_f6 = inverse_tanh(x_out)
    da_f6 = derive_tanh(x_out)
    k = np.multiply(np.matmul(last_weight.transpose(), k), da_f6.reshape(-1, 1))
    
    # 🚀 修復：分兩行明確賦值
    x_f6 = fcn_reconstruction(k=k, gradient=grads_np['classifier.2.weight'])
    last_weight = weights_np['classifier.2.weight']

    # -------------------------------------------------------------------------
    # 步驟 3: F5 層 (classifier.0)
    # -------------------------------------------------------------------------
    out_f5 = inverse_tanh(x_f6)
    da_f5 = derive_tanh(x_f6)
    k = np.multiply(np.matmul(last_weight.transpose(), k), da_f5.reshape(-1, 1))
    
    # 🚀 修復：分兩行明確賦值
    x_flatten = fcn_reconstruction(k=k, gradient=grads_np['classifier.0.weight'])
    last_weight = weights_np['classifier.0.weight']
    
    x_s4_shape = (1, 16, 5, 5)
    out_s4 = inverse_tanh(x_flatten.reshape(x_s4_shape))
    da_s4 = derive_tanh(x_flatten.reshape(x_s4_shape))
    print(f"  [+] 成功突破全連接層，取得 S4 特徵圖: {out_s4.shape}")

    # -------------------------------------------------------------------------
    # 步驟 4: 逆向 S4 (features.5: AvgPool2d)
    # -------------------------------------------------------------------------
    k_s4 = np.matmul(last_weight.transpose(), k).reshape(x_s4_shape)
    k_s4 = np.multiply(k_s4, da_s4)

    out_c3 = inverse_avgpool2d(out_s4)
    k_c3 = inverse_avgpool2d(k_s4) 
    print(f"  [+] 逆向 S4 平均池化，取得 C3 特徵圖: {out_c3.shape}")

    # -------------------------------------------------------------------------
    # 步驟 5: 逆向 C3 (features.3: Conv2d)
    # -------------------------------------------------------------------------
    out_c3_pre_tanh = inverse_tanh(out_c3)
    da_c3 = derive_tanh(out_c3)
    k_c3 = np.multiply(k_c3, da_c3)

    in_shape_s2 = (1, 6, 14, 14)

    x_s2, last_weight = cnn_reconstruction(
        in_shape=in_shape_s2,
        k=k_c3.reshape(-1),  
        g=grads_np['features.3.weight'], 
        out=out_c3_pre_tanh, 
        kernel=weights_np['features.3.weight'], 
        stride=1, 
        padding=0
    )
    out_s2 = x_s2.reshape(in_shape_s2)
    print(f"  [+] 成功突破 C3 卷積層，取得 S2 特徵圖: {out_s2.shape}")

    # -------------------------------------------------------------------------
    # 步驟 6: 逆向 S2 (features.2: AvgPool2d)
    # -------------------------------------------------------------------------
    k_s2_flat = np.matmul(last_weight.transpose(), k_c3.reshape(-1))
    k_s2 = k_s2_flat.reshape(in_shape_s2)

    out_c1 = inverse_avgpool2d(out_s2)
    k_c1 = inverse_avgpool2d(k_s2)
    print(f"  [+] 逆向 S2 平均池化，取得 C1 特徵圖: {out_c1.shape}")

    # -------------------------------------------------------------------------
    # 步驟 7: 逆向 C1 (features.0: Conv2d) -> Input
    # -------------------------------------------------------------------------
    out_c1_pre_tanh = inverse_tanh(out_c1)
    da_c1 = derive_tanh(out_c1)
    k_c1 = np.multiply(k_c1, da_c1)

    in_shape_input = (1, 1, 28, 28)
    x_input, _ = cnn_reconstruction(
        in_shape=in_shape_input,
        k=k_c1.reshape(-1),  
        g=grads_np['features.0.weight'],
        out=out_c1_pre_tanh,
        kernel=weights_np['features.0.weight'],
        stride=1,
        padding=2  
    )
    
    out_final = x_input.reshape(in_shape_input)
    print(f"  [+] 💀 成功貫穿 C1 卷積層！取得原始影像: {out_final.shape}")

    print("\n[R-GAP] 攻擊完成！已還原。")
    return recovered_label, out_final