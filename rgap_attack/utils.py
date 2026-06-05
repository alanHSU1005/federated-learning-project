import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def crossentropy_for_onehot(inputs, target):
    m = nn.LogSoftmax(dim=1)
    output = torch.mean(torch.sum(-target * m(inputs), 1))
    return output


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def logistic_loss(target, inputs):
    m = nn.Sigmoid()
    pred = torch.squeeze(m(inputs), -1)
    return torch.mean(-(target*torch.log(pred)+(1-target)*torch.log(1-pred)))


def weight_init(m):
    '''
    Apply this when study the effect of adding noise. See Appendix C.
    '''
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0, 0.01)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)


class GradientMetrics:
    def __init__(self, mode):
        if mode.lower() == 'l2':
            self.fn = nn.MSELoss(reduction='sum')
        elif mode.lower() == 'cos':
            self.fn = nn.CosineSimilarity(dim=0)
        else:
            raise ValueError("GradientMetrics: Unknown mode.")
        self.mode = mode

    def __call__(self, inputs, target):
        output = 0
        if self.mode.lower() == 'l2':
            for i, t in zip(inputs, target):
                output += self.fn(i, t)
        elif self.mode.lower() == 'cos':
            gx = []
            gy = []
            for i, t in zip(inputs, target):
                gx.append(i.view(-1))
                gy.append(t.view(-1))
            output = 1 - self.fn(torch.cat(gx), torch.cat(gy))
        return output


def inverse_leakyrelu(x, slope):
    return np.array([a / slope if a < 0 else a for a in x]).astype('float32')


def derive_leakyrelu(x, slope):
    return np.array([slope if a < 0 else 1 for a in x]).reshape(1, -1).astype('float32')


def inverse_sigmoid(x):
    return np.array([-np.log(1/a - 1) for a in x]).astype('float32')


def derive_sigmoid(x):
    return np.array([a*(1-a) for a in x]).reshape(1, -1).astype('float32')


def inverse_identity(x):
    return x


def derive_identity(x):
    return np.ones(x.shape).reshape(1, -1).astype('float32')


def show_images(images, path, cols=1, titles=None):
    """
    修正版 show_images：確保每次呼叫皆建立獨立 Figure，
    避免多樣本、多輪次測試時發生 Matplotlib 圖表狀態污染。
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: 
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    
    # 🌟 修改點 1：不指定固定字串，每次都開一張全新的獨立畫布，並指定大小
    fig = plt.figure(figsize=(cols * 3, 3)) 
    
    num_rows = int(np.ceil(n_images / float(cols)))
    num_cols = int(cols)
    
    for n, (image, title) in enumerate(zip(images, titles)):
        # 確保 subplot 的維度參數是整數
        a = fig.add_subplot(num_rows, num_cols, n + 1)
        plt.gray()
        
        # 🌟 修改點 2：如果是 PyTorch Tensor，先轉回 NumPy 轉產圖
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        if image.ndim == 4: # 如果是 (1, 1, H, W)
            image = image[0, 0]
        elif image.ndim == 3: # 如果是 (1, H, W) 或 (H, W, 1)
            if image.shape[0] == 1:
                image = image[0]
            elif image.shape[-1] == 1:
                image = image[:, :, 0]

        plt.imshow(image, cmap='gray')
        plt.axis('off')
        a.set_title(title, fontsize=10)
        
    # 🌟 修改點 3：使用 bbox_inches='tight' 確保標題與邊界不會被切到
    plt.savefig(path, bbox_inches='tight', dpi=150)
    
    # 🌟 修改點 4：致命關鍵！存檔後立即關閉該畫布，徹底清空記憶體
    plt.close(fig)