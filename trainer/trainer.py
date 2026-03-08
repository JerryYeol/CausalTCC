import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss
from sklearn.metrics import f1_score


def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, pred_labels, true_labels = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        valid_f1 = f1_score(true_labels, pred_labels, average='macro')
        if training_mode != 'self_supervised':  # use scheduler in all other modes.
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}| \tValid Macro F1-Score: {valid_f1:2.4f}')

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, test_pred_labels, test_true_labels = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        test_f1 = f1_score(test_true_labels, test_pred_labels, average='macro')
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}| \tTest Macro F1-Score: {test_f1:2.4f}')

    logger.debug("\n################## Training is Done! #########################")

# 原始代码
def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        # if batch_idx == 0:
        #     print("\n" + "="*30 + " Augmentation Debug Info " + "="*30)
            
        #     # 1. 检查 aug1 (时域 Token)
        #     print(f"[Aug1 (Time Domain)]")
        #     print(f"  Shape: {aug1.shape}  (期望: Batch, Channels, Length/3)")
        #     print(f"  Dtype: {aug1.dtype}  (期望: torch.int64)")
        #     print(f"  Values Range: [{aug1.min()}, {aug1.max()}] (期望: 0 到 4)")
        #     print(f"  Sample Data (第一个样本, 前30个点):")
        #     # 如果是3维 (B, C, L)，取 [0, 0, :30]；如果是2维 (B, L)，取 [0, :30]
        #     if aug1.ndim == 3:
        #         print(f"  {aug1[0, 0, :30].tolist()} ...")
        #     else:
        #         print(f"  {aug1[0, :30].tolist()} ...")

        #     print("-" * 30)

        #     # 2. 检查 aug2 (频域 Token)
        #     print(f"[Aug2 (Freq Domain)]")
        #     print(f"  Shape: {aug2.shape}  (期望: Batch, Channels, Length/2)")
        #     print(f"  Dtype: {aug2.dtype}  (期望: torch.int64)")
        #     print(f"  Values Range: [{aug2.min()}, {aug2.max()}] (期望: 0 到 4)")
        #     print(f"  Sample Data (第一个样本, 前30个点):")
        #     if aug2.ndim == 3:
        #         print(f"  {aug2[0, 0, :30].tolist()} ...")
        #     else:
        #         print(f"  {aug2[0, :30].tolist()} ...")
                
        #     print("="*80 + "\n")
        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)
            features1_fan = features1.flip(0)
            features2_fan = features2.flip(0)
            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)# 强增强
            features2 = F.normalize(features2, dim=1)# 若增强
            # 正
            # temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
            # temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)
            # # 反
            # temp_cont_loss1_fan, temp_cont_lstm_feat1_fan = temporal_contr_model(features1_fan, features2_fan)
            # temp_cont_loss2_fan, temp_cont_lstm_feat2_fan = temporal_contr_model(features2_fan, features1_fan)
                    # --- 2. 计算时间自损失 (Temporal Self-Loss) ---
            # 正
            loss_self_1, temp_self_lstm_feat1 = temporal_contr_model(features1, features1)
            loss_self_2, temp_self_lstm_feat2 = temporal_contr_model(features2, features2)
            # normalize projection feature vectors
            # 正
            # zis = temp_cont_lstm_feat1 
            # zjs = temp_cont_lstm_feat2
            zis = temp_self_lstm_feat1 
            zjs = temp_self_lstm_feat1
            # 反
            # zis_fan = temp_cont_lstm_feat1_fan 
            # zjs_fan = temp_cont_lstm_feat2_fan 

        else:
            output = model(data)

        # compute loss
        if training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            # loss = (temp_cont_loss1 + temp_cont_loss2) *  lambda1 +  nt_xent_criterion(zis, zjs) * lambda2/2 +(loss_self_1+loss_self_2)* 0
            # loss = (temp_cont_loss1 + temp_cont_loss2) *  lambda1/2 +  (temp_cont_loss1_fan + temp_cont_loss2_fan) *  0 + nt_xent_criterion(zis, zjs) * lambda2/2 +nt_xent_criterion(zis_fan, zjs_fan) * 0+(loss_self_1+loss_self_2)* lambda1/2
            # loss = (temp_cont_loss1 + temp_cont_loss2) *  lambda1/2  + nt_xent_criterion(zis, zjs) * lambda2 +(loss_self_1+loss_self_2)* lambda1/2
            loss = nt_xent_criterion(zis, zjs) * lambda2 +(loss_self_1+loss_self_2)* lambda1
            
        else: # supervised training or fine tuining
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc




def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        # 1. 修改：Evaluation 也只解包 data, labels
        for data, labels in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            # 获取输出
            logits, cnn_features, token_time, token_freq = model(data)

            if training_mode == "self_supervised":
                pass
            else:
                # 监督模式只关心分类结果
                predictions = logits
                features = cnn_features # 可选

            # compute loss
            if training_mode != "self_supervised":
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean() 
    else:
        total_loss = 0
        
    if training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()
        
    return total_loss, total_acc, outs, trgs
# 原始模型
def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                output = model(data)

            # compute loss
            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
    return total_loss, total_acc, outs, trgs