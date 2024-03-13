import os.path
import numpy as np
import torch
from model import GMCF
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score, precision_score, recall_score, accuracy_score, \
    mean_squared_error, average_precision_score, r2_score, accuracy_score
import pickle
import time
import math
import pandas as pd
import pickle
from functools import wraps
import dill
from lifelines.utils import concordance_index


def generate_result(raw, cur_step):
    val_auc, val_logloss, val_mse, val_ci, val_aupr, val_ndcg20, val_ndcg10, val_ndcg5, val_recall, val_precision, val_accuracy, val_r2 = raw
    return f'epoch:{cur_step},val_auc:{val_auc}, val_logloss:{val_logloss}, val_mse:{val_mse}, val_ci:{val_ci}, val_aupr:{val_aupr}, ' \
           f'val_ndcg20:{val_ndcg20}, val_ndcg10:{val_ndcg10}, val_ndcg5:{val_ndcg5}, val_recall:{val_recall}, ' \
           f'val_precision:{val_precision}, val_accuracy:{val_accuracy}, val_r2:{val_r2}'


class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_result = ''

    def __call__(self, val_loss, model, path, indicatior, cur_step):
        print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.indicator = generate_result(indicatior, cur_step)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            self.indicator = generate_result(indicatior, cur_step)

    def save_checkpoint(self, val_loss, model, path):
        if not os.path.exists(path):
            os.mkdir(path)
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'model_checkpoint.pth')
        self.val_loss_min = val_loss


def train(args, data_info, t):
    train_loader = data_info['train']
    val_loader = data_info['val']
    test_loader = data_info['test']
    feature_num = data_info['feature_num']
    train_num, val_num, test_num = data_info['data_num']
    early_stop = EarlyStopping()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    model = GENNDTI(args, feature_num, device)    model = model.to(device)

    newlist = list(filter(lambda n: n % 2 == 1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    newlist = list(x for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] if x % 2 == 1)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        weight_decay=args.l2_weight,
        lr=args.lr
    )
    crit = torch.nn.BCELoss()

    print([i.size() for i in filter(lambda p: p.requires_grad, model.parameters())])
    print('start training...')
    if not os.path.exists('./train_acc_davis'):
        os.mkdir('./train_acc_davis')
    with open(f'./train_acc_davis/cross={args.cross_model}, inner={args.inner_model},split={args.split}, time={t}.txt', 'w') as f:
        for step in range(args.n_epoch):
            if early_stop.early_stop:
                print('out_of_patience, stop training, model has been saved.\n')
                print(f'Best result is {early_stop.indicator} \n\n')
                f.write(str(early_stop.indicator))
                break
            # training
            loss_all = 0
            edge_all = 0
            model.train()

            for data in train_loader:
                data = data.to(device)
                output = model(data)
                label = data.y
                label = label.to(device)
                try:
                    baseloss = crit(torch.squeeze(output), label)
                except:
                    baseloss = crit(torch.squeeze(output,0), label)
                loss = baseloss
                loss_all += data.num_graphs * loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            cur_loss = loss_all / train_num

            # auc, logloss, mse, aupr, ndcg20, ndcg10, ndcg5, recall,precision,accuracy
            # evaluation
            val_auc, val_logloss, val_mse, val_ci, val_aupr, val_ndcg20, val_ndcg10, val_ndcg5, val_recall, val_precision, val_accuracy, val_r2 = evaluate(
                model, val_loader, device)
            # val_auc, val_logloss, val_ndcg20, val_ndcg10 = 0, 0, 0, 0
            test_auc, test_logloss, test_mse, test_ci, test_aupr, test_ndcg20, test_ndcg10, test_ndcg5, test_recall, test_precision, test_accuracy, test_r2 = evaluate(
                model, test_loader, device)
            if step>=10:
                early_stop(val_logloss, model, f'./model_acc_{args.dataset}/corss {args.cross_model} inner {args.inner_model} on split{args.split}',
                       evaluate(model, val_loader, device), step)
            print(
                'Epoch: {:03d}, Loss: {:.5f}, AUC: {:.5f}/{:.5f}, Logloss: {:.5f}/{:.5f}, MSE:{:.5f}/{:.5f},CI:{:.5f}/{:.5f}, AUPR:{:.5f}/{:.5f}, NDCG@20: {:.5f}/{:.5f} NDCG@10: {:.5f}/{:.5f},NDCG@5: {:.5f}/{:.5f} recall: {:.5f}/{:.5f} , precision: {:.5f}/{:.5f} , accuracy: {:.5f}/{:.5f}, r2: {:.5f}/{:.5f} '.
                    format(step, cur_loss, val_auc, test_auc, val_logloss, test_logloss, val_mse, test_mse, val_ci,
                           test_ci,
                           val_aupr, test_aupr, val_ndcg20, test_ndcg20, val_ndcg10, test_ndcg10, val_ndcg5, test_ndcg5,
                           val_recall, test_recall, val_precision, test_precision, val_accuracy, test_accuracy, val_r2,
                           test_r2))
            f.write(
                'Epoch: {:03d}, Loss: {:.5f}, AUC: {:.5f}/{:.5f}, Logloss: {:.5f}/{:.5f}, MSE:{:.5f}/{:.5f},CI:{:.5f}/{:.5f}, AUPR:{:.5f}/{:.5f}, NDCG@20: {:.5f}/{:.5f} NDCG@10: {:.5f}/{:.5f},NDCG@5: {:.5f}/{:.5f} recall: {:.5f}/{:.5f} , precision: {:.5f}/{:.5f} , accuracy: {:.5f}/{:.5f}, r2: {:.5f}/{:.5f} \n\n'.
                format(step, cur_loss, val_auc, test_auc, val_logloss, test_logloss, val_mse, test_mse, val_ci, test_ci,
                       val_aupr, test_aupr, val_ndcg20, test_ndcg20, val_ndcg10, test_ndcg10, val_ndcg5, test_ndcg5,
                       val_recall, test_recall, val_precision, test_precision, val_accuracy, test_accuracy, val_r2,
                       test_r2))


def evaluate(model, data_loader, device):
    model.eval()

    predictions = []
    labels = []
    user_ids = []
    edges_all = [0, 0]
    with torch.no_grad():
        for data in data_loader:
            _, user_id_index = np.unique(data.batch.detach().cpu().numpy(), return_index=True)
            user_id = data.x.detach().cpu().numpy()[user_id_index]
            user_ids.append(user_id)

            data = data.to(device)
            pred = model(data)
            pred = pred.squeeze().detach().cpu().numpy().astype('float64')
            if pred.size == 1:
                pred = np.expand_dims(pred, axis=0)
            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.concatenate(predictions, 0)
    labels = np.concatenate(labels, 0)
    user_ids = np.concatenate(user_ids, 0)
    predictions2 = np.around(predictions, decimals=0).astype(int)
    # print('type_predictions',type(predictions))
    # print('labals',labels)

    ndcg20 = cal_ndcg(predictions, labels, user_ids, 20)
    ndcg10 = cal_ndcg(predictions, labels, user_ids, 10)
    ndcg5 = cal_ndcg(predictions, labels, user_ids, 5)
    mse = mean_squared_error(labels, predictions)
    ci = concordance_index(labels, predictions)
    precision_1, recall_1, _ = precision_recall_curve(labels, predictions)  # 计算Precision和Recall
    aupr = auc(recall_1, precision_1)  # 计算AUPR值
    # aupr = average_precision_score(labels, predictions)
    auc_roc = roc_auc_score(labels, predictions)
    logloss = log_loss(labels, predictions)
    recall = recall_score(labels, predictions2, average='binary')
    precision = precision_score(labels, predictions2, average='binary')
    accuracy = accuracy_score(labels, predictions2)
    r2 = r2_score(labels, predictions)
    return auc_roc, logloss, mse, ci, aupr, ndcg20, ndcg10, ndcg5, recall, precision, accuracy, r2


def cal_ndcg(predicts, labels, user_ids, k):
    d = {'user': np.squeeze(user_ids), 'predict': np.squeeze(predicts), 'label': np.squeeze(labels)}
    df = pd.DataFrame(d)
    user_unique = df.user.unique()

    ndcg = []
    for user_id in user_unique:
        user_srow = df.loc[df['user'] == user_id]
        upred = user_srow['predict'].tolist()
        if len(upred) < 2:
            # print('less than 2', user_id)
            continue
        # supred = [upred] if len(upred)>1 else [upred + [-1]]  # prevent error occured if only one sample for a user
        ulabel = user_srow['label'].tolist()
        # sulabel = [ulabel] if len(ulabel)>1 else [ulabel +[1]]

        ndcg.append(ndcg_score([ulabel], [upred], k=k))

    return np.mean(np.array(ndcg))

