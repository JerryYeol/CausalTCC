import torch
import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad, _calc_metrics, copy_Files
from dataloader.dataloader import get_loso_loaders 
from trainer.trainer import Trainer, model_evaluate
from models.TC import TC
from models.model import base_Model
import pandas as pd 

start_time = datetime.now()

parser = argparse.ArgumentParser()
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str, help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str, help='Experiment Description')
parser.add_argument('--seed', default=0, type=int, help='seed value')
parser.add_argument('--training_mode', default='supervised', type=str, help='Modes: self_supervised, fine_tune')
parser.add_argument('--selected_dataset', default='AD-Auditory', type=str, help='Dataset name')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str, help='saving directory')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
args = parser.parse_args()

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

# 动态导入 Config
exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# 随机种子
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# ================= 准备数据路径 =================
# 指向第一步生成的 loso_data.pt
loso_data_path = os.path.join(f"/data/Docker_Liutianhao2025/TS-TCC-main/data/origin_data/{data_type}", "loso_data.pt")
if not os.path.exists(loso_data_path):
    raise FileNotFoundError(f"找不到 {loso_data_path}，请先运行 prepare_loso_data.py")

# 读取一次以获取所有的 Subject ID
temp_data = torch.load(loso_data_path)
all_subjects = np.unique(temp_data["subject_ids"].numpy())
del temp_data # 释放内存
print(f"检测到的受试者 ID: {all_subjects}")

# 存储结果
all_metrics = []

# ================= LOSO 循环 =================
for subject_id in all_subjects:
    print(f"\n{'='*20} LOSO Fold: Subject {subject_id} 作为测试集 {'='*20}")
    
    # 路径规划：experiments_logs/Exp1/run1/模式_seed_0/subject_X
    base_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
    experiment_log_dir = os.path.join(base_log_dir, f"subject_{subject_id}")
    os.makedirs(experiment_log_dir, exist_ok=True)

    # 日志
    log_file_name = os.path.join(experiment_log_dir, f"logs.log")
    logger = _logger(log_file_name)
    logger.debug(f'Current Test Subject: {subject_id}')

    # 1. 获取 DataLoader (训练集不含 Sub X，测试集只含 Sub X)
    train_dl, valid_dl, test_dl = get_loso_loaders(loso_data_path, subject_id, configs, training_mode)

    # 2. 初始化模型
    model = base_Model(configs).to(device)
    temporal_contr_model = TC(configs, device).to(device)

    # 3. 如果是 Fine-tune，加载对应的 Self-Supervised 模型
    if training_mode == "fine_tune":
        # 寻找对应的预训练模型
        # 路径必须是: logs/Exp1/run1/self_supervised_seed_0/subject_X/saved_models/ckp_last.pt
        # 注意：这里假设 fine_tune 和 self_supervised 使用相同的 experiment_description 和 run_description
        pretrained_dir = os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", f"subject_{subject_id}", "saved_models")
        pretrained_path = os.path.join(pretrained_dir, "ckp_last.pt")
        
        if os.path.exists(pretrained_path):
            logger.debug(f"加载预训练模型: {pretrained_path}")
            chkpoint = torch.load(pretrained_path, map_location=device)
            pretrained_dict = chkpoint["model_state_dict"]
            model_dict = model.state_dict()
            
            # 移除 logits 层 (分类层)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'logits' not in k}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            logger.debug(f"警告: 未找到预训练模型 {pretrained_path}，将使用随机初始化！")

    # 优化器
    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

    # 4. 训练 (Trainer)
    # 只有第一个 Subject 运行时才备份代码文件
    if subject_id == all_subjects[0] and training_mode == "self_supervised":
        copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

    Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode)

    # 5. 测试与记录结果 (Self-supervised 不需要算 Accuracy)
    if training_mode != "self_supervised":
        outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        total_loss, total_acc, pred_labels, true_labels = outs
        
        _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)
        
        acc_val = total_acc.item() if isinstance(total_acc, torch.Tensor) else total_acc
        all_metrics.append({"subject": subject_id, "accuracy": acc_val})
        logger.debug(f"Subject {subject_id} Accuracy: {acc_val:.4f}")

# ================= 汇总结果 =================
if training_mode != "self_supervised" and len(all_metrics) > 0:
    df = pd.DataFrame(all_metrics)
    avg_acc = df["accuracy"].mean()
    print(f"\n{'='*30}")
    print(f"LOSO 实验结束。平均准确率: {avg_acc:.4f}")
    print(f"结果已保存至: {base_log_dir}/LOSO_summary.xlsx")
    df.to_excel(os.path.join(base_log_dir, "LOSO_summary.xlsx"))
