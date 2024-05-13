from torchvision.datasets import CelebA
from torchvision.transforms.v2 import PILToTensor
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from simple_concepts.model import SimpleConcepts
from sklearn.model_selection import train_test_split
from typing import Tuple
from experiment_models import Autoencoder, BottleNeck, EasyClustering, quarter_patcher
from utility import f1_sep_scorer, acc_sep_scorer
import sympy as sp
from sympy.logic.boolalg import Equivalent
from time import gmtime, strftime

def get_proc_celeba_np() -> Tuple[np.ndarray, np.ndarray]:
    ds = DataLoader(CelebA('CelebA', download=False, transform=PILToTensor()), 60_000, False, num_workers=6)
    X_list = []
    y_list = []
    for x, y in ds:
        X_list.append(x)
        y_list.append(y)
    X_np = np.concatenate(X_list, axis=0, dtype=np.float32) # (162770, 3, 218, 178)
    std = np.std(X_np, 0, keepdims=True) 
    std[std < 1e-15] = 1.0
    X_np = (X_np - np.mean(X_np, 0, keepdims=True)) / std
    y_np = np.concatenate(y_list, axis=0, dtype=np.int32)
    return X_np, y_np
    
def tiny_sample_exp():
    date = strftime('%d.%m %H_%M_%S', gmtime())
    no_patcher = lambda X: X[:, None, ...]
    global y_train
    conc_num = y_train.shape[1]
    y_train, c_train = y_train[:, 0], y_train[:, 1:]
    n_list = [10000, 20000, 30000, 40000, 50000]
    epochs_n = [30, 20, 20, 20, 20]
    iter_num = 10
    f1_our = np.zeros((iter_num, len(n_list), conc_num))
    f1_bottleneck = np.zeros((iter_num, len(n_list), conc_num))
    acc_our = np.zeros((iter_num, len(n_list), conc_num))
    acc_bottleneck = np.zeros((iter_num, len(n_list), conc_num))
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
            
            model = BottleNeck(1, ep_n, 512, 1e-3, device, 3)
            model.fit(X_small_train, y_small_train, c_small_train)
            scores = model.predict(X_test)
            acc = acc_sep_scorer(y_test, scores)
            f1 = f1_sep_scorer(y_test, scores)
            f1_bottleneck[i, j, :] = f1
            acc_bottleneck[i, j, :] = acc
            print("Accuracy for concepts:", acc)
            print("F1 for concepts:", f1)
            np.savez(f'celeba_metrics {date} BACKUP', acc_our=acc_our, f1_our=f1_our, acc_btl=acc_bottleneck, f1_btl=f1_bottleneck, n_list=n_list)
    np.savez(f'celeba_metrics {date}', acc_our=acc_our, f1_our=f1_our, acc_btl=acc_bottleneck, f1_btl=f1_bottleneck, n_list=n_list)

if __name__=='__main__':
    cls_num = 100
    eps = 0.001
    device = 'cuda'
    ae_kw = {
        'latent_dim': 64, 
        'epochs_num': 30, 
        'batch_num': 1024, 
        'l_r': 1e-3, 
        'device': device,
        'early_stop': 3,
    }
    X, y = get_proc_celeba_np()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    tiny_sample_exp()
    # draw_figures()