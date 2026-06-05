import torch
import copy

def apply_dp_defense(stolen_grads: dict, max_norm: float = 1.0, noise_scale: float = 0.01, device: str = 'cuda'):
    """
    對被竊取的梯度字典 (stolen_grads) 施加差分隱私 (DP) 防禦。
    包含：Gradient Clipping (梯度裁剪) 與 Gaussian Noise (高斯加噪)。
    
    參數:
        stolen_grads (dict): 包含 PyTorch Tensor 梯度的字典 (e.g., {'features.0.weight': tensor, ...})
        max_norm (float): 梯度的最大 L2 範數上限 (C)
        noise_scale (float): 噪聲強度係數 (sigma)
        device (str): 運算裝置
    返回:
        defended_grads (dict): 經過 DP 加密防禦後的梯度字典
    """
    # 複製一份梯度，避免改動到原始訓練流程
    defended_grads = {k: v.clone().to(device) for k, v in stolen_grads.items()}
    
    with torch.no_grad():
        # 1. 計算全局梯度的總 L2 範數 (Total L2 Norm)
        total_norm = 0.0
        for name, grad in defended_grads.items():
            total_norm += grad.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # 2. 計算裁剪係數 (Clipping Coefficient)
        # 如果 total_norm > max_norm，則將梯度等比例縮小到 max_norm
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for name in defended_grads.keys():
                defended_grads[name].mul_(clip_coef)
        
        # 3. 施加高斯噪聲 (Add Gaussian Noise)
        # 噪聲的標準差 (std) 依據差分隱私理論，應與 (noise_scale * max_norm) 成正比
        for name, grad in defended_grads.items():
            noise = torch.randn(grad.size(), device=device) * (noise_scale * max_norm)
            defended_grads[name].add_(noise)
            
    return defended_grads
def fl_dp_defense_hook(model, max_norm=0.5, noise_scale=0.01, device='cuda'):
    """
    專門對接 main.py gradient_hook 的差分隱私防禦函數。
    它會直接就地（in-place）修改模型中計算好的梯度。
    """
    with torch.no_grad():
        # 1. 計算當前客戶端模型梯度的總 L2 範數
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # 2. 梯度裁剪 (Gradient Clipping)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        
        # 3. 添加高斯噪聲 (Add Gaussian Noise)
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.randn(p.grad.size(), device=device) * (noise_scale * max_norm)
                p.grad.data.add_(noise)