import torch
import numpy as np

# === 定义字母表 ===
VOCAB = np.array(['O', 'A', 'V', 'U', 'D']) 

def DataTransform(sample, config):
    # 确保转为 Numpy
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().cpu().numpy()
    
    # 弱增强
    weak_aug = time_domain_to_letters(sample)
    # 强增强
    strong_aug = freq_domain_to_letters(sample)

    return weak_aug, strong_aug

def time_domain_to_letters(x):
    # x: (Batch, Channels, Time)
    T = x.shape[-1]
    n_groups = T // 3
    cutoff = n_groups * 3
    x = x[..., :cutoff]
    
    L = x[..., 0::3]
    M = x[..., 1::3]
    R = x[..., 2::3]
    
    upper_bound = np.maximum(L, R)
    lower_bound = np.minimum(L, R)
    
    # 初始化索引矩阵 (默认全为4/D)
    indices = np.full(L.shape, 4, dtype=int)
    
    # 应用规则
    indices[L < R] = 3
    indices[M <= lower_bound] = 2
    indices[M >= upper_bound] = 1
    indices[L == R] = 0
    
    # === 关键点：返回字母而不是索引 ===
    return VOCAB[indices] 

def freq_domain_to_letters(x):
    # FFT 变换
    fft_result = np.fft.rfft(x, axis=-1)
    magnitude = np.abs(fft_result)
    log_magnitude = np.log(magnitude + 1e-6)
    
    mean = np.mean(log_magnitude, axis=-1, keepdims=True)
    std = np.std(log_magnitude, axis=-1, keepdims=True)
    
    normalized = (log_magnitude - mean) / (std + 1e-6)
    
    indices = np.zeros(normalized.shape, dtype=int)
    indices[normalized > -1.5] = 1
    indices[normalized > -0.5] = 2
    indices[normalized > 0.5]  = 3
    indices[normalized > 1.5]  = 4
    
    # === 关键点：返回字母而不是索引 ===
    return VOCAB[indices]