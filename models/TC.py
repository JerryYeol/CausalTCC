# import torch
# import torch.nn as nn
# import numpy as np
# from .attention import Seq_Transformer



# class TC(nn.Module):
#     def __init__(self, configs, device):
#         super(TC, self).__init__()
#         self.num_channels = configs.final_out_channels
#         self.timestep = configs.TC.timesteps
#         self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
#         self.lsoftmax = nn.LogSoftmax()
#         self.device = device
        
#         self.projection_head = nn.Sequential(
#             nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
#             nn.BatchNorm1d(configs.final_out_channels // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
#         )

#         self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)

#     def forward(self, features_aug1, features_aug2):
#         # print(f"TC:feature1:{features_aug1.shape},feature2:{features_aug2.shape}")
#         z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
#         seq_len = z_aug1.shape[2]
#         z_aug1 = z_aug1.transpose(1, 2)

#         z_aug2 = features_aug2
#         z_aug2 = z_aug2.transpose(1, 2)

#         batch = z_aug1.shape[0]
#         t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps

#         nce = 0  # average over timestep and batch
#         encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

#         for i in np.arange(1, self.timestep + 1):
#             encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)
#         forward_seq = z_aug1[:, :t_samples + 1, :]

#         c_t = self.seq_transformer(forward_seq)

#         pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
#         for i in np.arange(0, self.timestep):
#             linear = self.Wk[i]
#             pred[i] = linear(c_t)
#         for i in np.arange(0, self.timestep):
#             total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
#             nce += torch.sum(torch.diag(self.lsoftmax(total)))
#         nce /= -1. * batch * self.timestep
#         return nce, self.projection_head(c_t)

# # 最原始的
import torch
import torch.nn as nn
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Seq_Len, Batch, Dim)
        return x + self.pe[:x.size(0)]

class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        self.device = device
        
        # === 1. 这里的 Wk 用于预测未来步 ===
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim=1)
        
        # === 2. 投影头 (用于将 Transformer 输出映射到对比学习空间) ===
        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )

        # === 3. BERT / Transformer Encoder 核心替换部分 ===
        # 定义 Transformer 维度
        self.d_model = configs.TC.hidden_dim
        
        # 如果输入的通道数 和 hidden_dim 不一致，需要投影
        # 通常 configs.final_out_channels 作为输入维度
        if self.num_channels != self.d_model:
            self.input_projection = nn.Linear(self.num_channels, self.d_model)
        else:
            self.input_projection = nn.Identity()

        # 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model)

        # Transformer Encoder (BERT 结构)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=4, 
            dim_feedforward=64, # 根据原代码 mlp_dim=64
            dropout=0.1, 
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2) # depth=4 太深可能难训练，建议2-4

    def forward(self, features_aug1, features_aug2):
        # features: (Batch, Channels, Seq_Len)
        
        z_aug1 = features_aug1
        z_aug1 = z_aug1.transpose(1, 2) # -> (Batch, Seq_Len, Channels)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2) # -> (Batch, Seq_Len, Channels)

        batch = z_aug1.shape[0]
        seq_len = z_aug1.shape[1]
        
        # 随机选择切分的时间点
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)

        # === 上下文提取 (Source) ===
        # 切取 t_samples 之前的数据作为历史上下文
        # forward_seq Shape: (Batch, Context_Len, Channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]
        
        # 1. 投影维度 (若需要)
        forward_seq = self.input_projection(forward_seq)
        
        # 2. 维度变换适配 PyTorch Transformer: (Seq_Len, Batch, d_model)
        forward_seq = forward_seq.permute(1, 0, 2)
        
        # 3. 添加位置编码
        forward_seq = self.pos_encoder(forward_seq)
        
        # 4. BERT 编码
        output_seq = self.transformer_encoder(forward_seq)
        
        # 5. 获取 Context Vector (c_t)
        # 取最后一个时间步作为上下文总结
        c_t = output_seq[-1] # Shape: (Batch, d_model)

        # === 目标构建 (Target) ===
        # 准备未来的真实数据作为对比目标
        nce = 0
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].reshape(batch, self.num_channels)

        # === 对比预测 ===
        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            # 预测第 i 步的未来
            pred[i] = linear(c_t)
            
        for i in np.arange(0, self.timestep):
            # 计算预测值与真实值之间的相似度 (logits)
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            # 计算 Loss
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
            
        nce /= -1. * batch * self.timestep
        
        return nce, self.projection_head(c_t)
