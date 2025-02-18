import numpy as np
import pandas as pd
from itertools import combinations, permutations

def ToBij():
    signed = 1
    if np.random.randn(1) >= 0:
        signed = -1
    return signed * np.random.uniform(.5, 2.0)


def Toa():
    return np.random.uniform(0.5, 1.0)


def Case1(Num=5000, seed=0):
    np.random.seed(seed)
    n_ver, n_edge, n_latent = 15, 14, 5
    noise = np.load(f'../noise_{Num}.npy')[np.random.choice(np.arange(20), n_ver, replace=False)]
    while True:
        edge = [ToBij() for _ in range(n_edge)]
        A = np.zeros([n_ver, n_ver])
        A[5, 0], A[6, 0] = edge[0:2]
        A[0, 1], A[7, 1], A[8, 1], A[2, 1] = edge[2:6]
        A[9, 2], A[10, 2], A[3, 2] = edge[6:9]
        A[11, 3], A[12, 3], A[4, 3] = edge[9:12]
        A[13, 4], A[14, 4] = edge[12:14]
        M = np.linalg.inv(np.eye(n_ver) - A)
        if np.all(np.abs(M[np.abs(M) > 1e-6]) > 0.25): # faithfulness
            break
    X = M @ noise
    data = pd.DataFrame(X[n_latent:].T,columns=[f'x{i}' for i in range(1, n_ver-n_latent+1)])
    return data, A


def Case2(Num=5000, seed=0):
    np.random.seed(seed)
    n_ver, n_edge, n_latent = 20, 26, 5
    noise = np.load(f'../noise_{Num}.npy')[np.random.choice(np.arange(20), n_ver, replace=False)]
    while True:
        edge = [ToBij() for _ in range(n_edge)]
        A = np.zeros([n_ver, n_ver])
        A[5, 0], A[6, 0], A[7, 0] = edge[0:3]
        A[0, 1], A[6, 1], A[8, 1], A[9, 1], A[10, 1], A[11, 1], A[14, 1], A[2, 1] = edge[3:11]
        A[12, 2], A[13, 2], A[15, 2], A[3, 2] = edge[11:15]
        A[14, 3], A[15, 3], A[16, 3], A[18, 3], A[4, 3] = edge[15:20]
        A[17, 4], A[18, 4], A[19, 4] = edge[20:23]
        A[7, 8], A[12, 11], A[17, 16] = edge[23:26]
        M = np.linalg.inv(np.eye(n_ver) - A)
        if np.all(np.abs(M[np.abs(M) > 1e-6]) > 0.25): # faithfulness
            break
    X = M @ noise
    data = pd.DataFrame(X[n_latent:].T,columns=[f'x{i}' for i in range(1, n_ver-n_latent+1)])
    return data, A


def Case3(Num=5000, seed=0):
    np.random.seed(seed)
    n_ver, n_edge, n_latent = 16, 19, 5
    noise = np.load(f'../noise_{Num}.npy')[np.random.choice(np.arange(20), n_ver, replace=False)]
    while True:
        edge = [ToBij() for _ in range(n_edge)]
        A = np.zeros([n_ver, n_ver])
        A[5, 0], A[6, 0] = edge[0:2]
        A[0, 1], A[6, 1], A[7, 1], A[8, 1], A[2, 1] = edge[2:7]
        A[8, 2], A[9, 2], A[10, 2], A[3, 2] = edge[7:11]
        A[10, 3], A[11, 3], A[12, 3], A[4, 3] = edge[11:15]
        A[12, 4], A[13, 4] = edge[15:17]
        A[14, 0], A[15, 4] = edge[17:19]
        M = np.linalg.inv(np.eye(n_ver) - A)
        if np.all(np.abs(M[np.abs(M) > 1e-6]) > 0.25): # faithfulness
            break
    X = M @ noise
    data = pd.DataFrame(X[n_latent:].T,columns=[f'x{i}' for i in range(1, n_ver-n_latent+1)])
    return data, A

def Case4(Num=5000, seed=0):
    np.random.seed(seed)
    n_ver, n_edge, n_latent = 14, 18, 5
    noise = np.load(f'../noise_{Num}.npy')[np.random.choice(np.arange(20), n_ver, replace=False)]
    while True:
        edge = [ToBij() for _ in range(n_edge)]
        A = np.zeros([n_ver, n_ver])
        A[5, 0], A[6, 0] = edge[0:2]
        A[0, 1], A[6, 1], A[7, 1], A[2, 1] = edge[2:6]
        A[8, 2], A[10, 2], A[3, 2] = edge[6:9]
        A[8, 3], A[9, 3], A[4, 3] = edge[9:12]
        A[9, 4], A[10, 4], A[11, 4] = edge[12:15]
        A[12, 0], A[13, 4] = edge[15:17]
        A[13, 12] = edge[17]
        M = np.linalg.inv(np.eye(n_ver) - A)
        if np.all(np.abs(M[np.abs(M) > 1e-6]) > 0.25): # faithfulness
            break
    X = M @ noise
    data = pd.DataFrame(X[n_latent:].T,columns=[f'x{i}' for i in range(1, n_ver-n_latent+1)])
    return data, A


def performance(gt, pre, num_observed):
    num_latent = len(gt) - num_observed
    num_pre_latent = len(pre) - num_observed
    result = (np.abs(num_latent - num_pre_latent), 0, 0, 0, 0)
    if len(pre) < num_observed:
        return result
    total_edge = np.sum(np.abs(gt) > 1e-6)
    total_order = np.sum(np.abs(np.linalg.inv(np.eye(len(gt)) - gt)) > 1e-6) - len(gt)
    total_pre_edge = np.sum(np.abs(pre) > 1e-6)
    # total_pre_order = np.sum(np.abs(np.linalg.inv(np.eye(len(pre)) - pre)) > 1e-6) - len(pre)
    if len(pre) < len(gt):
        temp = np.zeros([len(gt), len(gt)])
        temp[:len(pre), :len(pre)] = pre
        pre = temp
    gt_adjacency = gt
    gt_mixing = np.linalg.inv(np.eye(len(gt_adjacency)) - gt_adjacency)
    max_correct_edge, max_correct_order = 1e-6, 1e-6
    for latent_order in permutations(list(range(num_observed, len(pre))), num_latent):
        order = list(latent_order) + list(range(num_observed))
        pre_adjacency = pre[order, :][:, order]
        pre_mixing = np.linalg.inv(np.eye(len(pre_adjacency)) - pre_adjacency)
        correct_edge = np.sum(np.abs(pre_adjacency) * np.abs(gt_adjacency) > 1e-6)
        correct_order = np.sum(np.abs(pre_mixing) * np.abs(gt_mixing) > 1e-6) - len(gt_mixing)
        if (correct_edge > max_correct_edge) or (correct_edge == max_correct_edge and correct_order > max_correct_order):
            max_correct_edge, max_correct_order = correct_edge, correct_order
            result = (np.abs(num_latent - num_pre_latent), max_correct_order / total_order,
                       2 / (total_pre_edge / max_correct_edge + total_edge / max_correct_edge))
    return result