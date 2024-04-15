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
        result_Y[i] = np.sum(cur_digits) % 10
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
    n_list = [200, 500, 1000, 2000, 5000]
    f1_our = []
    f1_bottleneck = []
    for n in n_list:
        X_small_train, _, y_small_train, _, c_small_train, _ = train_test_split(X_train, y_train, c_train, train_size=n)
        clusterizer = EasyClustering(cls_num, Autoencoder(**ae_kw))
        model = SimpleConcepts(cls_num, clusterizer, quarter_patcher, eps)
        model.fit(X_small_train, y_small_train, c_small_train)
        scores = model.predict(X_test)
        acc = acc_sep_scorer(y_test, scores)
        f1 = f1_sep_scorer(y_test, scores)
        print("Accuracy for concepts:", acc)
        print("F1 for concepts:", f1)
        f1_our.append(f1)
        
        print('--- Bottleneck ---')
        
        model = BottleNeck(1, 30, 256, 1e-3, 'cuda')
        model.fit(X_small_train, y_small_train, c_small_train)
        scores = model.predict(X_test)
        acc = acc_sep_scorer(y_test, scores)
        f1 = f1_sep_scorer(y_test, scores)
        f1_bottleneck.append(f1)
        print("Accuracy for concepts:", acc)
        print("F1 for concepts:", f1)
    fig, axes = plt.subplots(1, 11, figsize=(30, 4), sharex='all', sharey='all')
    f1_our = np.asarray(f1_our)
    f1_bottleneck = np.asarray(f1_bottleneck)
    for i in range(11):
        ax = axes[i]
        ax.plot(n_list, f1_our[:, i], 'rs--', label='Our model')
        ax.plot(n_list, f1_bottleneck[:, i], 'gs--', label='Bottleneck')
        ax.grid()
        ax.legend()
        ax.set_xlabel('train sample size')
        if i == 0:
            ax.set_ylabel('F1')
        ax.set_title('Concept ' + str(i))
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    n = 50000
    cls_num = 10
    eps = 0.01
    ae_kw = {
        'latent_dim': 28, 
        'epochs_num': 30, 
        'batch_num': 128, 
        'l_r': 1e-3, 
        'device': 'cuda'
    }
    X, y, c = get_4mnist_ds(*get_proc_mnist_np(), n)
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, c, test_size=0.4)         
    y_test = np.concatenate((y_test[:, None], c_test), axis=1)
    tiny_sample_exp()