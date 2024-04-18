from torchvision.datasets import MNIST
from torchvision.transforms.v2 import PILToTensor
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple
import torch
import numba
from utility import f1_sep_scorer, acc_sep_scorer
from experiment_models import Autoencoder, BottleNeck, vert_patcher, quarter_patcher, EasyClustering
from sklearn.model_selection import train_test_split
from simple_concepts.model import SimpleConcepts
import matplotlib.pyplot as plt

def get_proc_mnist_np() -> Tuple[np.ndarray, np.ndarray]:
    ds = DataLoader(MNIST('MNIST', download=False, transform=PILToTensor()), 8192, False)
    X_list = []
    y_list = []
    for x, y in ds:
        X_list.append(x)
        y_list.append(y)
    X_np = np.concatenate(X_list, axis=0) # (60000, 1, 28, 28)
    std = np.std(X_np, 0, keepdims=True)
    std[std < 1e-15] = 1.0
    X_np = (X_np - np.mean(X_np, 0, keepdims=True)) / std
    y_np = np.concatenate(y_list, axis=0)
    return X_np, y_np

@numba.njit
def get_4mnist_ds(X, y, samples_num: int):
    result_X = np.empty((samples_num, 1, 56, 56))
    result_Y = np.empty(samples_num, np.intp)
    result_C = np.zeros((samples_num, 10), np.intp)
    range_idx = np.arange(10)
    dig_idx_list = []
    for i in range(10):
        dig_idx_list.append(np.argwhere(y == i))
    slice_list = [(slice(None, 28), slice(None, 28)),
                  (slice(None, 28), slice(28, None)),
                  (slice(28, None), slice(None, 28)),
                  (slice(28, None), slice(28, None))]
    for i in range(samples_num):
        cur_digits = np.random.choice(range_idx, 4, False)
        for j in range(4):
            dig_idx = dig_idx_list[cur_digits[j]][np.random.randint(0, len(dig_idx_list[cur_digits[j]]))]
            result_X[i, :, slice_list[j][0], slice_list[j][1]] = X[dig_idx]
            result_C[i, cur_digits[j]] = 1
        result_Y[i] = np.max(cur_digits)
    result_Y -= np.min(result_Y)
    return result_X, result_Y, result_C

def one_shot_exp():
    clusterizer = EasyClustering(cls_num, Autoencoder(**ae_kw))
    model = SimpleConcepts(cls_num, clusterizer, quarter_patcher, eps)
    model.fit(X_train, y_train, c_train)
    scores = model.predict(X_test)
    acc = acc_sep_scorer(y_test, scores)
    f1 = f1_sep_scorer(y_test, scores)
    print("Accuracy for concepts:", acc)
    print("F1 for concepts:", f1)
    
    print('--- Bottleneck ---')
    
    model = BottleNeck(1, 20, 256, 1e-3, 'cuda')
    model.fit(X_train, y_train, c_train)
    scores = model.predict(X_test)
    acc = acc_sep_scorer(y_test, scores)
    f1 = f1_sep_scorer(y_test, scores)
    print("Accuracy for concepts:", acc)
    print("F1 for concepts:", f1)
    
def tiny_sample_exp():
    n_list = [1000, 2000, 3000, 4000, 5000]
    epochs_n = [40, 40, 30, 30, 20]
    iter_num = 10
    f1_our = np.zeros((iter_num, len(n_list), 11))
    f1_bottleneck = np.zeros((iter_num, len(n_list), 11))
    acc_our = np.zeros((iter_num, len(n_list), 11))
    acc_bottleneck = np.zeros((iter_num, len(n_list), 11))
    for i in range(iter_num):
        for j, n in enumerate(n_list):
            ep_n = epochs_n[j]
            ae_kw['epochs_num'] = ep_n
            X_small_train, _, y_small_train, _, c_small_train, _ = train_test_split(X_train, y_train, c_train, train_size=n)
            clusterizer = EasyClustering(cls_num, Autoencoder(**ae_kw))
            model = SimpleConcepts(cls_num, clusterizer, quarter_patcher, eps)
            model.fit(X_small_train, y_small_train, c_small_train)
            scores = model.predict(X_test)
            acc = acc_sep_scorer(y_test, scores)
            f1 = f1_sep_scorer(y_test, scores)
            print("Accuracy for concepts:", acc)
            print("F1 for concepts:", f1)
            f1_our[i, j, :] = f1
            acc_our[i, j, :] = acc
            
            print('--- Bottleneck ---')
            
            model = BottleNeck(1, ep_n, 256, 1e-3, 'cuda')
            model.fit(X_small_train, y_small_train, c_small_train)
            scores = model.predict(X_test)
            acc = acc_sep_scorer(y_test, scores)
            f1 = f1_sep_scorer(y_test, scores)
            f1_bottleneck[i, j, :] = f1
            acc_bottleneck[i, j, :] = acc
            print("Accuracy for concepts:", acc)
            print("F1 for concepts:", f1)
    np.savez('metrics5', acc_our=acc_our, f1_our=f1_our, acc_btl=acc_bottleneck, f1_btl=f1_bottleneck)
    fig, axes = plt.subplots(2, 11, figsize=(30, 6), sharex='all', sharey='all')
    f1_our = np.mean(f1_our, axis=0)
    f1_bottleneck = np.mean(f1_bottleneck, axis=0)
    for i in range(11):
        ax = axes[0, i]
        ax.plot(n_list, f1_our[:, i], 'rs--', label='Our model')
        ax.plot(n_list, f1_bottleneck[:, i], 'gs--', label='Bottleneck')
        ax.grid()
        ax.legend()
        ax.set_xlabel('train sample size')
        if i == 0:
            ax.set_ylabel('F1')
        ax.set_title('Concept ' + str(i))
    acc_our = np.mean(acc_our, axis=0)
    acc_bottleneck = np.mean(acc_bottleneck, axis=0)
    for i in range(11):
        ax = axes[1, i]
        ax.plot(n_list, acc_our[:, i], 'rs--', label='Our model')
        ax.plot(n_list, acc_bottleneck[:, i], 'gs--', label='Bottleneck')
        ax.grid()
        ax.legend()
        ax.set_xlabel('train sample size')
        if i == 0:
            ax.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    n = 50000
    cls_num = 80
    eps = 0.001
    ae_kw = {
        'latent_dim': 16, 
        'epochs_num': 30, 
        'batch_num': 256, 
        'l_r': 1e-3, 
        'device': 'cuda'
    }
    X, y, c = get_4mnist_ds(*get_proc_mnist_np(), n)
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, c, test_size=0.4)         
    y_test = np.concatenate((y_test[:, None], c_test), axis=1)
    tiny_sample_exp()