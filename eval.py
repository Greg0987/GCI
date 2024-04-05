import pandas as pd
import numpy as np
import os
import torch
import random
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

def eval(name, dataset, c_train, c_val):
    pf = pd.read_csv(f'datasets/{dataset}_matrix.csv')
    train_val = f'{c_train}{c_val}'
    # 用户、视频、播放率
    user_id = np.array(pf['user_id'])
    video_id = np.array(pf['video_id'])
    watch_ratio = np.array(pf['watch_ratio'])

    pc = pd.read_csv('datasets/item_categories.csv')

    video_id_ca = np.array(pc['video_id'])

    full_video_tag = []
    bad_video = []
    for i in range(pc['feat'].shape[0]):
        l = np.array(pc['feat'])[i][1:-1]    # 去掉首尾的[]
        tmp = [int(s) for s in l.split(',')]  # 分割字符串，转化为int
        full_video_tag.append(tmp)
        for j in range(len(tmp)):
            if tmp[j] == 0:
                bad_video.append(i)

    dele = []
    for i in range(user_id.shape[0]):
        if video_id[i] in bad_video:
            dele.append(i)

    user_id_new = []
    video_id_new = []
    watch_ratio_new = []
    k = 0
    for i in range(user_id.shape[0]):
        if dele[k] == i:
            if k < len(dele) - 1:
                k += 1
            continue
        else:
            user_id_new.append(user_id[i])
            video_id_new.append(video_id[i])
            watch_ratio_new.append(watch_ratio[i])

    user_id_new = np.array(user_id_new)
    video_id_new = np.array(video_id_new)
    watch_ratio_new = np.array(watch_ratio_new)

    new_watch_ratio = watch_ratio_new

    dit_user=dict()
    reve_user=dict()
    dit_video=dict()
    reve_video=dict()
    ab_set_user=set(user_id_new)    # 1411个用户
    ab_set_video=set(video_id_new)  # 3326个视频

    k=0
    for item in ab_set_user:    # 给用户编号，0-1410
        dit_user[k]=item        # {k: user_id}
        reve_user[item]=k       # {user_id: k}
        k+=1
    k=0
    for item in ab_set_video:   # 给视频编号，0-3325
        dit_video[k]=item       # {k: video_id}
        reve_video[item]=k      # {video_id: k}
        k+=1

    new_user_id=[]
    new_video_id=[]
    for i in range(user_id_new.shape[0]):
        new_user_id.append(reve_user[user_id_new[i]])   # 新的用户编号，0-1410
        new_video_id.append(reve_video[video_id_new[i]])    # 新的视频编号，0-3325

    new_user_id=np.array(new_user_id)
    new_video_id=np.array(new_video_id)

    shape_1 = len(set(new_user_id))    # 1411
    shape_2 = len(set(new_video_id))   # 3326

    video_list = []
    for i in range(shape_1):
        video_list.append([])

    for i in range(new_user_id.shape[0]):
        if new_watch_ratio[i] >= 2.0:
            video_list[new_user_id[i]].append(new_video_id[i])

    test_index = np.load(f'npy_for_pre_recall_f1/{name}/{dataset}idx_test{train_val}.npy')

    pred = np.load(f'npy_for_pre_recall_f1/{name}/{dataset}pred{train_val}.npy')
    trued = np.load(f'npy_for_pre_recall_f1/{name}/{dataset}y_true{train_val}.npy')

    def get_acc(pred, label, c_train, c_val, model):
        mypred = torch.ones(pred.shape)*float('-inf')
        if(model == 'train'):
            mypred[:, :c_train] = pred[:, :c_train]
        elif model == 'val':
            mypred[:, c_train: c_train+c_val] = pred[:, c_train: c_train+c_val]
        elif model == 'test':
            mypred[:, c_train+c_val: ] = pred[:, c_train+c_val: ]
        return get_acc_basic(mypred, label, model)

    def get_acc_basic(predict, label, model):
        predict = torch.argmax(predict, axis=1)
        acc = (label.cpu()==predict)
        result = acc.cpu().sum().numpy()
        return result/len(acc), predict

    test_acc, pre = get_acc(torch.tensor(pred[test_index]),
                            torch.tensor(trued[test_index]),
                            c_train=c_train,
                            c_val=c_val,
                            model='test')

    final_pred = trued.copy()

    final_pred[test_index] = pre

    Precision = 0
    Recall = 0
    F1 = 0
    k = 0
    for i in range(len(video_list)):
        if len(video_list[i]) == 0:
            k += 1
            continue
        temp_t = trued[video_list[i]]
        temp_p = final_pred[video_list[i]]
        p, r, f, _ = precision_recall_fscore_support(temp_t, temp_p, average='macro', zero_division=0)
        Precision += p
        Recall += r
        F1 += f

    Precision /= (len(video_list) - k)
    Recall /= (len(video_list) - k)
    F1 /= (len(video_list) - k)

    return Precision, Recall, F1