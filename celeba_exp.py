from torchvision.datasets import CelebA
from torchvision.transforms.v2 import PILToTensor
from torch.utils.data import DataLoader
import numpy as np
from simple_concepts.model import SimpleConcepts
from sklearn.model_selection import train_test_split
from typing import Tuple
from experiment_models import Autoencoder, BottleNeck, EasyClustering, quarter_patcher, window_patcher
from utility import f1_sep_scorer, acc_sep_scorer
from time import gmtime, strftime
from functools import partial

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

TRAIN_PATH = './train_array.npz'
TEST_PATH = './test_array.npz'
def dump_train_test_on_disk(X, y, test_split = 0.4):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    np.savez(TRAIN_PATH, X=X_train, y=y_train)
    np.savez(TEST_PATH, X=X_test, y=y_test)

def load_train() -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(TRAIN_PATH)
    return arr['X'], arr['y']

def load_test() -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(TEST_PATH)
    return arr['X'], arr['y']
    
def tiny_sample_exp():
    X, y = get_proc_celeba_np()
    dump_train_test_on_disk(X, y)
    conc_num = y.shape[1]
    del X
    del y
        
    date = strftime('%d.%m %H_%M_%S', gmtime())
    # no_patcher = lambda X: X[:, None, ...]
    patcher = partial(window_patcher, kernel_size=(110, 90), stride=(17, 22))
    n_list = [1000, 2000, 4000, 6000, 8000, 10000]
    epochs_n = [40, 40, 30, 30, 20, 20]
    iter_num = 10
    f1_our = np.zeros((iter_num, len(n_list), conc_num))
    f1_bottleneck = np.zeros((iter_num, len(n_list), conc_num))
    acc_our = np.zeros((iter_num, len(n_list), conc_num))
    acc_bottleneck = np.zeros((iter_num, len(n_list), conc_num))
    for i in range(iter_num):
        for j, n in enumerate(n_list):
            ep_n = epochs_n[j]
            ae_kw['epochs_num'] = ep_n
            X_train, y_train = load_train()
            y_train, c_train = y_train[:, 0], y_train[:, 1:]
            X_small_train, _, y_small_train, _, c_small_train, _ = train_test_split(X_train, y_train, c_train, train_size=n)
            del X_train
            del y_train
            del c_train
            clusterizer = EasyClustering(cls_num, Autoencoder(**ae_kw))
            model_sc = SimpleConcepts(cls_num, clusterizer, patcher, eps)
            model_sc.fit(X_small_train, y_small_train, c_small_train)
            
            print('--- Bottleneck ---')
            model_btl = BottleNeck(1, ep_n, 512, 1e-3, device, 3)
            model_btl.fit(X_small_train, y_small_train, c_small_train)
            
            X_test, y_test = load_test()
            scores_sc = model_sc.predict(X_test)
            acc_sc = acc_sep_scorer(y_test, scores_sc)
            f1_sc = f1_sep_scorer(y_test, scores_sc)
            print("Accuracy for concepts SC:", acc_sc)
            print("F1 for concepts SC:", f1_sc)
            f1_our[i, j, :] = f1_sc
            acc_our[i, j, :] = acc_sc
            scores_btl = model_btl.predict(X_test)
            acc_btl = acc_sep_scorer(y_test, scores_btl)
            f1_btl = f1_sep_scorer(y_test, scores_btl)
            f1_bottleneck[i, j, :] = f1_btl
            acc_bottleneck[i, j, :] = acc_btl
            print("Accuracy for concepts BTL:", acc_btl)
            print("F1 for concepts BTL:", f1_btl)
            np.savez(f'celeba_metrics {date} BACKUP', acc_our=acc_our, f1_our=f1_our, acc_btl=acc_bottleneck, f1_btl=f1_bottleneck, n_list=n_list)
            del X_test
            del y_test
    np.savez(f'celeba_metrics {date}', acc_our=acc_our, f1_our=f1_our, acc_btl=acc_bottleneck, f1_btl=f1_bottleneck, n_list=n_list)
    
def draw_figures():
    import matplotlib.pyplot as plt
    array_zip = np.load('celeba_metrics 23.05 14_21_37.npz')
    n_list = array_zip['n_list']
    metrics_id = ['acc', 'f1']
    metrics_names = ['Accuracy', 'F1']
    for id, name in zip(metrics_id, metrics_names):
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(name)
        sc_metric = np.mean(array_zip[id + '_our'], axis=0)
        btl_metric = np.mean(array_zip[id + '_btl'], axis=0)
        ax.fill_between(n_list, sc_metric.min(axis=-1), sc_metric.max(axis=-1), color='red', alpha=.1)
        ax.plot(n_list, sc_metric.mean(axis=-1), 'rs--', label='Our model')
        ax.fill_between(n_list, btl_metric.min(axis=-1), btl_metric.max(axis=-1), color='green', alpha=.1)
        ax.plot(n_list, btl_metric.mean(axis=-1), 'gs--', label='Bottleneck')
        ax.grid()
        ax.legend()
        ax.set_xlabel('train sample size')
        ax.set_ylabel(name)
    plt.show()

if __name__=='__main__':
    cls_num = 512
    eps = 0.001
    device = 'cuda'
    ae_kw = {
        'latent_dim': 48, 
        'epochs_num': 30, 
        'batch_num': 100, 
        'l_r': 1e-3, 
        'device': device,
        'early_stop': 3,
    }
    # tiny_sample_exp()
    draw_figures()