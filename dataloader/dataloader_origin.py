import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from .augmentations import DataTransform


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


# def data_generator(data_path, configs, training_mode):

#     train_dataset = torch.load(os.path.join(data_path, "train.pt"))
#     valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
#     test_dataset = torch.load(os.path.join(data_path, "test.pt"))

#     train_dataset = Load_Dataset(train_dataset, configs, training_mode)
#     valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
#     test_dataset = Load_Dataset(test_dataset, configs, training_mode)

#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
#                                                shuffle=True, drop_last=configs.drop_last,
#                                                num_workers=0)
#     valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
#                                                shuffle=False, drop_last=configs.drop_last,
#                                                num_workers=0)

#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
#                                               shuffle=False, drop_last=False,
#                                               num_workers=0)

#     return train_loader, valid_loader, test_loader
import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# 假设 Load_Dataset 类已经定义
# from .dataloader import Load_Dataset 

import os
import torch
from torch.utils.data import DataLoader
# 假设 Load_Dataset 类在别处已定义
# from .dataset import Load_Dataset 

def data_generator(data_path, configs, training_mode):

    # --- 步骤 1: 统一加载所有需要的数据 ---
    # 无论什么模式，我们都需要加载这三个文件来准备数据
    print("Loading all base datasets (train, test, val)...")
    train_dataset_full = torch.load(os.path.join(data_path, "train.pt"))
    test_dataset_full = torch.load(os.path.join(data_path, "test.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))

    # 无论哪种模式，最终用于评估的测试集都是原始的 test.pt
    test_dataset = test_dataset_full

    # --- 步骤 2: 根据训练模式选择数据 ---

    if training_mode == 'self_supervised':
        # --- 预训练模式: 合并训练集和测试集 ---
        print("Mode: Self-supervised. Combining train and test sets for pre-training.")
        combined_samples = torch.cat((train_dataset_full['samples'], test_dataset_full['samples']), dim=0)
        combined_labels = torch.cat((train_dataset_full['labels'], test_dataset_full['labels']), dim=0)
        
        train_dataset = {
            'samples': combined_samples,
            'labels': combined_labels
        }
        print(f"Total instances for pre-training: {len(combined_samples)}")
        # 在预训练模式下，我们通常用验证集来监控，所以让 test_loader 指向验证集
        test_dataset = valid_dataset

    elif training_mode in ['supervised', 'fine_tune','random_init']:
        # --- 微调/监督模式: 执行特殊的采样策略 ---
        print(f"Mode: {training_mode}. Applying special sampling strategy.")

        # 检查是否需要采样
        if hasattr(configs, 'label_percentage') and configs.label_percentage < 1.0:
            
            # 1. 确定采样数量 N (基于合并后的总数)
            num_train_instances = len(train_dataset_full['samples'])
            num_test_instances = len(test_dataset_full['samples'])
            total_pool_size = num_train_instances + num_test_instances
            
            num_samples_to_select = int(total_pool_size * configs.label_percentage)

            print(f"Combined pool size is {total_pool_size} ({num_train_instances} train + {num_test_instances} test).")
            print(f"Calculating {configs.label_percentage * 100:.1f}% of total pool -> selecting {num_samples_to_select} samples.")

            # 安全检查：确保要采样的数量不超过原始训练集的大小
            if num_samples_to_select > num_train_instances:
                print(f"Warning: Calculated sample count ({num_samples_to_select}) exceeds original training data size ({num_train_instances}). Capping at {num_train_instances}.")
                num_samples_to_select = num_train_instances

            # 2. 确定采样来源 (从原始训练集中)
            print(f"Selecting these {num_samples_to_select} samples FROM the original training set (size: {num_train_instances}).")
            
            original_train_features = train_dataset_full['samples'].numpy()
            original_train_labels = train_dataset_full['labels'].numpy()

            # 使用 train_test_split 进行分层采样, train_size 使用计算出的绝对数量
            features_subset, _, labels_subset, _ = train_test_split(
                original_train_features,
                original_train_labels,
                train_size=num_samples_to_select,
                stratify=original_train_labels,
            )
            
            print(f"Final training set size: {len(features_subset)} instances.")

            # 将采样后的数据重新组合成字典
            train_dataset = {
                'samples': torch.from_numpy(features_subset),
                'labels': torch.from_numpy(labels_subset)
            }
        else:
            # 如果是100%数据，则直接使用全部原始训练数据
            print("Using 100% of the original training set.")
            train_dataset = train_dataset_full
            
    else:
        raise ValueError(f"Unknown training_mode: {training_mode}")

    # --- 步骤 3: 创建 Dataset 和 DataLoader ---
    # 这部分代码保持不变
    train_dataset_loader = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset_loader = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset_loader = Load_Dataset(test_dataset, configs, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
                                               
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset_loader, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader







