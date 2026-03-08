from torch import nn

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
                # ---- 调试代码 ----
        # 获取卷积层输出的真实形状 [batch_size, final_out_channels, actual_features_len]
        actual_shape = x.shape
        actual_features_len = actual_shape[2]
        
        # 计算展平后的真实大小
        actual_flattened_size = x.view(x.shape[0], -1).shape[1]
        
        # 获取线性层期望的输入大小
        expected_flattened_size = self.logits.in_features
        # 在出错前，打印所有相关参数
        if actual_flattened_size != expected_flattened_size:
            print("\n" + "="*60)
            print("!!! 关键参数诊断 !!!")
            print(f"> 卷积层输出的真实形状 (x.shape): {actual_shape}")
            print(f"  - 批次大小 (Batch Size): {actual_shape[0]}")
            print(f"  - 最终输出通道 (final_out_channels): {actual_shape[1]}")
            print(f"  - 真实特征长度 (ACTUAL features_len): {actual_features_len}")
            print("-" * 60)
            print(f"> 展平后的向量真实大小 (actual_flattened_size): {actual_flattened_size}")
            print(f"> 线性层期望的输入大小 (expected_flattened_size): {expected_flattened_size}")
            print("-" * 60)
            print(" [ 错误原因 ] ")
            print(f"  真实大小({actual_flattened_size}) 与 期望大小({expected_flattened_size}) 不匹配。")
            print(" [ 解决方案 ] ")
            print(f"  请打开你的配置文件 (例如 'Epilepsy_Configs.py')，")
            print(f"  然后将 `self.features_len` 的值修改为: {actual_features_len}")
            print("="*60 + "\n")
        # # ---- 调试结束 ----
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x
# # from torch import nn

# # # ===================================================================
# # # 1. 残差块 (ResidualBlock) - 可复用的高级特征提取单元
# # # ===================================================================
# # import torch.nn.functional as F
# # from torch import nn

# # class ResidualBlock(nn.Module):
# #     """
# #     一个完整的残差块，兼容旧版 PyTorch。
# #     通过手动 padding 来实现 'same' 卷积效果。
# #     """
# #     def __init__(self, in_channels, out_channels, kernel_size, dropout):
# #         super(ResidualBlock, self).__init__()
        
# #         self.kernel_size = kernel_size
        
# #         # 捷径连接：用于匹配维度
# #         if in_channels != out_channels:
# #             self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
# #         else:
# #             self.shortcut = nn.Identity()

# #         # 主干网络：注意这里的 Conv1d 中已经移除了 padding 参数
# #         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, bias=False)
# #         self.bn1 = nn.BatchNorm1d(out_channels)
# #         self.relu = nn.ReLU()
# #         self.dropout = nn.Dropout(dropout)
        
# #         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=self.kernel_size, bias=False)
# #         self.bn2 = nn.BatchNorm1d(out_channels)
        
# #         self.final_relu = nn.ReLU()

# #     def _pad(self, x):
# #         # 这是实现 'same' padding 的关键
# #         # 计算需要在左边和右边填充多少
# #         padding_left = (self.kernel_size - 1) // 2
# #         padding_right = self.kernel_size - 1 - padding_left
# #         # 使用 F.pad 进行填充
# #         # (pad_left, pad_right) -> 只在最后一个维度（长度维度）上填充
# #         return F.pad(x, (padding_left, padding_right))

# #     def forward(self, x):
# #         # 保存原始输入，用于最后的残差连接
# #         shortcut_out = self.shortcut(x)
        
# #         # 第一个卷积块
# #         out = self._pad(x)      # 先填充
# #         out = self.conv1(out)   # 再卷积
# #         out = self.bn1(out)
# #         out = self.relu(out)
# #         out = self.dropout(out)
        
# #         # 第二个卷积块
# #         out = self._pad(out)    # 同样，先填充
# #         out = self.conv2(out)   # 再卷积
# #         out = self.bn2(out)
        
# #         # 残差连接的核心：F(x) + x
# #         out += shortcut_out
        
# #         return self.final_relu(out)



# # # ===================================================================
# # # 2. 主模型 (base_Model) - 使用残差块和GAP构建
# # # ===================================================================
# # class base_Model(nn.Module):
# #     """
# #     一个健壮的CNN模型，不依赖于输入序列的长度。
# #     """
# #     def __init__(self, configs):
# #         super(base_Model, self).__init__()
        
# #         # 第一个残差块：将输入通道数映射到32
# #         self.block1 = ResidualBlock(configs.input_channels, 32, kernel_size=configs.kernel_size, dropout=configs.dropout)
# #         self.pool1 = nn.MaxPool1d(2) # 池化层，用于降低序列长度，扩大感受野
        
# #         # 第二个残差块：通道数从32提升到64
# #         self.block2 = ResidualBlock(32, 64, kernel_size=8, dropout=configs.dropout)
# #         self.pool2 = nn.MaxPool1d(2)

# #         # 第三个残差块：通道数从64提升到最终的输出通道数
# #         self.block3 = ResidualBlock(64, configs.final_out_channels, kernel_size=8, dropout=configs.dropout)
        
# #         # 全局平均池化层：将任意长度的序列转换为固定长度的向量
# #         self.gap = nn.AdaptiveAvgPool1d(1)
        
# #         # 分类头 (Classifier Head)：输入维度仅为 final_out_channels
# #         self.logits = nn.Linear(configs.final_out_channels, configs.num_classes)

# #     def forward(self, x_in):
# #         # 特征提取
# #         x = self.pool1(self.block1(x_in))
# #         x = self.pool2(self.block2(x))
# #         x_features = self.block3(x)  # 最后的特征图，形状为 [B, C, L]
        
# #         # 池化与分类
# #         # 1. 全局平均池化: [B, C, L] -> [B, C, 1]
# #         pooled = self.gap(x_features)
        
# #         # 2. 展平: [B, C, 1] -> [B, C]
# #         pooled_flat = pooled.squeeze(-1)
        
# #         # 3. 全连接层分类
# #         logits = self.logits(pooled_flat)
        
# #         return logits, x_features
# 第一版
# from torch import nn
# import torch
# import torch.nn.functional as F

# class base_Model(nn.Module):
#     def __init__(self, configs):
#         super(base_Model, self).__init__()

#         # --- 1. Embedding 层 ---
#         # 0-4 共 5 个 token
#         self.char_embed_dim = 16 
#         self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=self.char_embed_dim)

#         # 调整输入通道: 原始通道 * Embedding维度
#         self.cnn_input_channels = configs.input_channels * self.char_embed_dim

#         self.conv_block1 = nn.Sequential(
#             nn.Conv1d(self.cnn_input_channels, 32, kernel_size=configs.kernel_size,
#                       stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#             nn.Dropout(configs.dropout)
#         )

#         self.conv_block2 = nn.Sequential(
#             nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
#         )

#         self.conv_block3 = nn.Sequential(
#             nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
#             nn.BatchNorm1d(configs.final_out_channels),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#         )

#         # --- 关键修改点 ---
#         # 我们不再使用 Pool(1)，而是 Pool 到一个固定的序列长度。
#         # 这个长度必须大于 TC.py 中的 self.timestep (通常是几)
#         # 假设我们固定为 16 (您可以根据 TC.timesteps 调整，只要比它大就行)
#         self.fixed_seq_len = 16
#         self.feature_adapter = nn.AdaptiveAvgPool1d(self.fixed_seq_len)
        
#         # 分类头
#         self.model_output_dim = configs.final_out_channels
#         self.logits = nn.Linear(self.model_output_dim * self.fixed_seq_len, configs.num_classes)

#     def forward(self, x_in):
#         # x_in: (Batch, Channels, Length) LongTensor
        
#         # 1. Embedding
#         x = self.embedding(x_in)  # (Batch, C, L, Emb)
        
#         N, C, L, E = x.shape
#         # Permute & Reshape: (Batch, C*E, L)
#         x = x.permute(0, 1, 3, 2).reshape(N, C * E, L)
        
#         # 2. CNN Layers
#         x = self.conv_block1(x)
#         x = self.conv_block2(x)
#         x = self.conv_block3(x)

#         # 3. 强制对齐长度 (解决 TC.py 报错和 Aug1/Aug2 长度不一致问题)
#         # features shape: (Batch, Final_Channels, 16)  <- 这是一个 3D Tensor，TC 模块现在可以读取 shape[2] 了
#         features = self.feature_adapter(x)

#         # 4. 分类用的 logits (Flatten)
#         x_flat = features.reshape(N, -1) 
#         logits = self.logits(x_flat)
        
#         # 返回: (Logits, 3D特征序列)
#         return logits, features
# 现在证实，如果是单纯的0-4的映射来token化，在这里面效果是最好的
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class base_Model(nn.Module):
#     def __init__(self, configs):
#         super(base_Model, self).__init__()
        
#         # === 1. 第一层卷积 Block ===
#         # 长度变化: L -> L/2
#         self.conv_block1 = nn.Sequential(
#             nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
#                       stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#             nn.Dropout(configs.dropout)
#         )

#         # === 2. 第二层卷积 Block ===
#         # 长度变化: L/2 -> L/4
#         self.conv_block2 = nn.Sequential(
#             nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
#         )

#         # === 3. 第三层卷积 Block (关键修改) ===
#         # 长度变化: L/4 -> L/4 (移除了池化层，防止长度过短报错)
#         self.conv_block3 = nn.Sequential(
#             nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
#             nn.BatchNorm1d(configs.final_out_channels),
#             nn.ReLU(),
#             # [Fix]: 删除了这里的 MaxPool1d，保留时序长度供 TC 模块使用
#              # nn.MaxPool1d(kernel_size=2, stride=2, padding=1), 
#         )

#         # === 4. 分类头 ===
#         self.logits = nn.Linear(configs.final_out_channels, configs.num_classes)

#     def forward(self, x_in):
#         # x_in: (Batch, Channels, Length) FloatTensor
        
#         x = self.conv_block1(x_in)
#         x = self.conv_block2(x)
#         x = self.conv_block3(x) 
        
#         # 输出给 TC 模块的特征 (Batch, Final_Channels, Length)
#         features = x
        
#         # 分类时使用全局平均池化，不再依赖固定长度
#         x_flat = F.adaptive_avg_pool1d(x, 1).squeeze(2)
#         logits = self.logits(x_flat)
        
#         return logits, features


# #多模态信息融合
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SingleChannelEncoder(nn.Module):
#     """
#     针对单个物理通道的独立编码器 (ResNet 风格)。
#     结构：2层卷积 (从3层简化)，引入残差连接防止过拟合。
#     """
#     def __init__(self, configs):
#         super(SingleChannelEncoder, self).__init__()
        
#         # === Block 1: Stem Layer (降采样与初步特征提取) ===
#         # 结构: Conv -> BN -> ReLU -> MaxPool -> Dropout
#         # 输入: 1 -> 输出: 32
#         self.block1 = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=configs.kernel_size,
#                       stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#             nn.Dropout(configs.dropout)
#         )

#         # === Block 2: Residual Block (特征映射) ===
#         # 结构: Conv -> BN (注意：ReLU 放在残差相加之后)
#         # 输入: 32 -> 输出: final_out_channels (例如 128)
#         # 移除了额外的 Pooling，以保留时序长度
#         self.block2_conv = nn.Sequential(
#             nn.Conv1d(32, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
#             nn.BatchNorm1d(configs.final_out_channels)
#         )

#         # === Shortcut / Skip Connection ===
#         # 因为输入(32)和输出(128)通道数不同，需要 1x1 卷积进行升维
#         self.shortcut = nn.Sequential(
#             nn.Conv1d(32, configs.final_out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm1d(configs.final_out_channels)
#         )

#     def forward(self, x):
#         # 1. 第一层基础卷积
#         out_1 = self.block1(x)
        
#         # 2. 第二层主路径
#         out_2 = self.block2_conv(out_1)
        
#         # 3. 残差路径 (Projection)
#         identity = self.shortcut(out_1)
        
#         # [安全措施] 处理可能存在的1像素形状不匹配问题
#         # 由于 kernel_size=8 padding=4 可能会导致输出长度比输入长 1
#         if out_2.shape[2] != identity.shape[2]:
#             min_len = min(out_2.shape[2], identity.shape[2])
#             out_2 = out_2[:, :, :min_len]
#             identity = identity[:, :, :min_len]
            
#         # 4. 残差相加 + 激活
#         out = F.relu(out_2 + identity)
        
#         return out # Shape: (Batch, Final_Channels, Length/2)


# class base_Model(nn.Module):
#     """
#     多模态融合主模型
#     1. 动态创建 N 个独立 Encoder (ResNet版)
#     2. 分别通过 Encoder
#     3. Concat 拼接特征
#     4. 1x1 Conv 融合特征
#     """
#     def __init__(self, configs):
#         super(base_Model, self).__init__()
        
#         self.num_phys_channels = getattr(configs, 'original_channels', configs.input_channels) 
#         self.final_dim = configs.final_out_channels
        
#         print(f"[Model] Initializing ResNet-style Multi-Branch Architecture with {self.num_phys_channels} Encoders.")

#         # === 1. 创建多分支 Encoders ===
#         self.encoders = nn.ModuleList([
#             SingleChannelEncoder(configs) for _ in range(self.num_phys_channels)
#         ])
        
#         # === 2. 融合层 (Fusion Layer) ===
#         # 拼接后的通道数压缩回 final_dim
#         self.fusion_layer = nn.Sequential(
#             nn.Conv1d(self.final_dim * self.num_phys_channels, self.final_dim, kernel_size=1),
#             nn.BatchNorm1d(self.final_dim),
#             nn.ReLU()
#         )

#         # === 3. 分类头 ===
#         self.logits = nn.Linear(self.final_dim, configs.num_classes)

#     def forward(self, x_in):
#         # x_in: (Batch, 9, Length) or (Batch, 1, 9*Length)
#         batch_size = x_in.shape[0]
        
#         # === 智能形状适配 ===
#         if x_in.shape[1] == self.num_phys_channels:
#             x_reshaped = x_in
#         else:
#             total_len = x_in.shape[2]
#             single_len = total_len // self.num_phys_channels
#             x_reshaped = x_in.view(batch_size, self.num_phys_channels, single_len)
        
#         encoder_outputs = []
        
#         # === 遍历每个通道 ===
#         for i in range(self.num_phys_channels):
#             input_i = x_reshaped[:, i:i+1, :] 
#             out_i = self.encoders[i](input_i) 
#             encoder_outputs.append(out_i)
        
#         # === 融合 ===
#         concatenated_features = torch.cat(encoder_outputs, dim=1)
#         fused_features = self.fusion_layer(concatenated_features)
        
#         # === 输出 ===
#         x_flat = F.adaptive_avg_pool1d(fused_features, 1).squeeze(2)
#         logits = self.logits(x_flat)
        
#         return logits, fused_features

# import torch
# import torch.fft

# def torch_time_domain_to_ids(x):
#     """
#     将时序特征转换为离散的 Token ID (0-4)。
#     完全基于 PyTorch 实现，可在 GPU 上运行。
    
#     Args:
#         x: (Batch, Channels, Time)
#     Returns:
#         indices: (Batch, Channels, Time//3)
#     """
#     # 确保是 float 类型进行计算
#     x = x.float()
    
#     T = x.shape[-1]
#     n_groups = T // 3
#     cutoff = n_groups * 3
    
#     # 截断多余的时间步以适配 3 的倍数
#     x = x[..., :cutoff]
    
#     # 向量化切片: 左(L), 中(M), 右(R)
#     L = x[..., 0::3]
#     M = x[..., 1::3]
#     R = x[..., 2::3]
    
#     # 计算边界
#     upper_bound = torch.max(L, R)
#     lower_bound = torch.min(L, R)
    
#     # 初始化索引矩阵 (默认全为 4/D)
#     indices = torch.full_like(L, 4, dtype=torch.long)
    
#     # === 应用规则 (使用 Mask 操作) ===
    
#     # Rule 1: L < R -> 3 (U)
#     mask_LR = L < R
#     indices[mask_LR] = 3
    
#     # Rule 2: M <= lower_bound -> 2 (V)
#     # 注意：这里的优先级需要小心，你的 numpy 代码中是顺序覆盖的，
#     # 我们这里也按顺序覆盖，后执行的会覆盖前面的。
    
#     mask_V = M <= lower_bound
#     indices[mask_V] = 2
    
#     # Rule 3: M >= upper_bound -> 1 (A)
#     mask_A = M >= upper_bound
#     indices[mask_A] = 1
    
#     # Rule 4: L == R -> 0 (O)
#     mask_O = L == R
#     indices[mask_O] = 0
    
#     return indices

# def torch_freq_domain_to_ids(x):
#     """
#     将特征进行 FFT 变换并分桶为 Token ID (0-4)。
    
#     Args:
#         x: (Batch, Channels, Time)
#     Returns:
#         indices: (Batch, Channels, Freq_Bins)
#     """
#     # 1. FFT 变换 (沿最后一个维度)
#     # rfft 得到复数结果, 形状大致是 Time/2 + 1
#     fft_result = torch.fft.rfft(x, dim=-1)
    
#     # 2. 计算幅值
#     magnitude = torch.abs(fft_result)
    
#     # 3. Log 变换 (加极小值防止 log(0))
#     log_magnitude = torch.log(magnitude + 1e-6)
    
#     # 4. 归一化 (Instance Normalization 风格)
#     # 在最后一个维度(频域)上求均值和方差
#     mean = torch.mean(log_magnitude, dim=-1, keepdim=True)
#     std = torch.std(log_magnitude, dim=-1, keepdim=True)
    
#     normalized = (log_magnitude - mean) / (std + 1e-6)
    
#     # 5. 离散化 (Bucketize)
#     # 默认初始化为 0 (O) -- 这一步对应你代码里没命中的情况，或者是初始值
#     # 或者是你可能想要一个默认类。根据你的 numpy 代码，最后是 VOCAB[indices]，
#     # 你的 numpy indices创建时是 np.zeros。
#     indices = torch.zeros_like(normalized, dtype=torch.long)
    
#     # 阈值判定
#     # normalized > -1.5 -> 1 (A)
#     indices[normalized > -1.5] = 1
    
#     # normalized > -0.5 -> 2 (V) 
#     indices[normalized > -0.5] = 2
    
#     # normalized > 0.5  -> 3 (U)
#     indices[normalized > 0.5]  = 3
    
#     # normalized > 1.5  -> 4 (D)
#     indices[normalized > 1.5]  = 4
    
#     return indices

# from torch import nn

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class base_Model(nn.Module):
#     def __init__(self, configs):
#         super(base_Model, self).__init__()
        
#         self.input_channels = configs.input_channels
#         self.final_dim = configs.final_out_channels
#         self.kernel_size = configs.kernel_size
#         self.stride = configs.stride
#         self.dropout = configs.dropout

#         # 1. Stem Layer
#         self.block1 = nn.Sequential(
#             nn.Conv1d(self.input_channels, 32, kernel_size=self.kernel_size,
#                       stride=self.stride, bias=False, padding=(self.kernel_size//2)),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#             nn.Dropout(self.dropout)
#         )

#         # 2. ResNet Block
#         self.block2_conv = nn.Sequential(
#             nn.Conv1d(32, self.final_dim, kernel_size=8, stride=1, bias=False, padding=4),
#             nn.BatchNorm1d(self.final_dim)
#         )

#         self.shortcut = nn.Sequential(
#             nn.Conv1d(32, self.final_dim, kernel_size=1, bias=False),
#             nn.BatchNorm1d(self.final_dim)
#         )

#         # === 核心修改 1: 修改分类头定义 ===
#         # 原来: self.logits = nn.Linear(model_output_dim * self.final_dim, configs.num_classes)
#         # 现在: 我们使用 global pooling，所以输入维度固定为 final_dim (通道数)
#         self.logits = nn.Linear(self.final_dim, configs.num_classes)

#     def forward(self, x_in):
#         # x_in: (Batch, Channels, Length)

#         # 简单的维度检查与修正
#         if len(x_in.shape) < 3:
#             x_in = x_in.unsqueeze(1)    
        
#         if x_in.shape[1] != self.input_channels and x_in.shape[2] == self.input_channels:
#              x_in = x_in.permute(0, 2, 1)

#         # Block 1
#         out_1 = self.block1(x_in)
        
#         # Block 2 (ResNet)
#         out_2_main = self.block2_conv(out_1)
#         identity = self.shortcut(out_1)
        
#         # 对齐长度
#         if out_2_main.shape[2] != identity.shape[2]:
#             min_len = min(out_2_main.shape[2], identity.shape[2])
#             out_2_main = out_2_main[:, :, :min_len]
#             identity = identity[:, :, :min_len]
            
#         # 得到卷积特征 (用于 TC 模块，保留时间维度)
#         # Shape: (Batch, Final_Dim, Length')
#         features = F.relu(out_2_main + identity)
        
#         # === 核心修改 2: 使用全局平均池化计算 Logits ===
#         # 无论 features 长度是多少，在 dim=2 (时间轴) 上取平均
#         # features: (Batch, 128, 100) -> x_pool: (Batch, 128)
#         x_pool = torch.mean(features, dim=2)
        
#         # 生成分类预测
#         logits = self.logits(x_pool)
        
#         token_time = torch_time_domain_to_ids(features)
#         token_freq = torch_freq_domain_to_ids(features)
        
#         return logits, features, token_time, token_freq