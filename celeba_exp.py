from torchvision.datasets import CelebA
from torchvision.transforms.v2 import PILToTensor
from torch.utils.data import DataLoader
from torch import concat
import numpy as np
from simple_concepts.model import SimpleConcepts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from typing import Tuple
from experiment_models import Autoencoder, BottleNeck, EasyClustering, quarter_patcher, window_patcher
from utility import f1_sep_scorer, acc_sep_scorer
from time import gmtime, strftime
from functools import partial

def get_proc_celeba_np() -> Tuple[np.ndarray, np.ndarray]:
    ds = DataLoader(CelebA('CelebA', download=False, transform=PILToTensor(), target_type=['identity', 'attr']), 21_000, False, num_workers=8)
    X_list = []
    y_list = []
    for x, y in ds:
        X_list.append(x)
        y_list.append(concat((y[0][:, None], y[1]), dim=1))
    X_np = np.concatenate(X_list, axis=0, dtype=np.float32) # (162770, 3, 218, 178)
    std = np.std(X_np, 0, keepdims=True) 
    std[std < 1e-6] = 1.0
    X_np = (X_np - np.mean(X_np, 0, keepdims=True)) / std
    y_np = np.concatenate(y_list, axis=0, dtype=np.int32)
    tgt_idx = np.argsort(y_np[:, 0])
    n_tgt_cls = 20
    tgt_split = np.array_split(tgt_idx, n_tgt_cls)
    for i, idces in enumerate(tgt_split):
        y_np[idces, 0] = i
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
    X_train, y_train = load_train()
    y_train, c_train = y_train[:, 0], y_train[:, 1:]
    conc_num = c_train.shape[1]
    X_test, y_test = load_test()
    y_test, c_test = y_test[:, 0], y_test[:, 1:]
    
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
    roc_our = np.zeros((iter_num, len(n_list), conc_num))
    roc_bottleneck = np.zeros((iter_num, len(n_list), conc_num))
    ap_our = np.zeros((iter_num, len(n_list), conc_num))
    ap_bottleneck = np.zeros((iter_num, len(n_list), conc_num))
    acc_tgt = np.zeros((iter_num, len(n_list), 2)) # ours, btl
    for i in range(iter_num):
        for j, n in enumerate(n_list):
            ep_n = epochs_n[j]
            ae_kw['epochs_num'] = ep_n
            # X_train, y_train = load_train()
            # y_train, c_train = y_train[:, 0], y_train[:, 1:]
            X_small_train, _, y_small_train, _, c_small_train, _ = train_test_split(X_train, y_train, c_train, train_size=n)
            # del X_train
            # del y_train
            # del c_train
            encoder = OrdinalEncoder(dtype=int)
            y_small_train = encoder.fit_transform(y_small_train[:, None]).ravel()
            clusterizer = EasyClustering(cls_num, Autoencoder(**ae_kw))
            model_sc = SimpleConcepts(cls_num, clusterizer, patcher, eps)
            model_sc.fit(X_small_train, y_small_train, c_small_train)
            
            print('--- Bottleneck ---')
            model_btl = BottleNeck(1, ep_n, 512, 1e-3, device, 3)
            model_btl.fit(X_small_train, y_small_train, c_small_train)
            
            # X_test, y_test = load_test()
            tgt_sc, proba_sc = model_sc.predict_tgt_lbl_conc_proba(X_test)
            proba_lbl = np.stack((proba_sc[:, 0::2], proba_sc[:, 1::2]), axis=-1).argmax(axis=-1)
            proba_sc = proba_sc[:, 1::2]
            tgt_sc = encoder.inverse_transform(tgt_sc[:, None]).ravel()
            acc_sc = acc_sep_scorer(c_test, proba_lbl)
            f1_sc = f1_sep_scorer(c_test, proba_lbl)
            # roc_sc = roc_auc_score(c_test, proba_sc)
            # ap_sc = average_precision_score(c_test, proba_sc)
            f1_our[i, j, :] = f1_sc
            acc_our[i, j, :] = acc_sc
            for k in range(conc_num):
                roc_our[i, j, k] = roc_auc_score(c_test[:, k], proba_sc[:, k])
                ap_our[i, j, k] = average_precision_score(c_test[:, k], proba_sc[:, k])
            acc_tgt[i, j, 0] = accuracy_score(y_test, tgt_sc)
            print("Accuracy for concepts SC:", acc_sc)
            print("F1 for concepts SC:", f1_sc)
            tgt_btl, proba_btl = model_btl.predict_tgt_lbl_conc_proba(X_test)
            proba_lbl = np.stack((proba_btl[:, 0::2], proba_btl[:, 1::2]), axis=-1).argmax(axis=-1)
            proba_btl = proba_btl[:, 1::2]
            tgt_btl = encoder.inverse_transform(tgt_btl[:, None]).ravel()
            acc_btl = acc_sep_scorer(c_test, proba_lbl)
            f1_btl = f1_sep_scorer(c_test, proba_lbl)
            for k in range(conc_num):
                roc_bottleneck[i, j, k] = roc_auc_score(c_test[:, k], proba_btl[:, k])
                ap_bottleneck[i, j, k] = average_precision_score(c_test[:, k], proba_btl[:, k])
            acc_tgt[i, j, 1] = accuracy_score(y_test, tgt_btl)
            f1_bottleneck[i, j, :] = f1_btl
            acc_bottleneck[i, j, :] = acc_btl
            print("Accuracy for concepts BTL:", acc_btl)
            print("F1 for concepts BTL:", f1_btl)
            np.savez(f'celeba_metrics {date} BACKUP', 
                     acc_our=acc_our, f1_our=f1_our, roc_our=roc_our, ap_our=ap_our,
                     acc_btl=acc_bottleneck, f1_btl=f1_bottleneck, roc_btl=roc_bottleneck,
                     ap_btl=ap_bottleneck, acc_tgt=acc_tgt, n_list=n_list)
            # del X_test
            # del y_test
    np.savez(f'celeba_metrics {date}', acc_our=acc_our, f1_our=f1_our, roc_our=roc_our, ap_our=ap_our,
                     acc_btl=acc_bottleneck, f1_btl=f1_bottleneck, roc_btl=roc_bottleneck,
                     ap_btl=ap_bottleneck, acc_tgt=acc_tgt, n_list=n_list)
    
def draw_figures():
    import matplotlib.pyplot as plt
    array_zip = np.load('celeba_metrics 28.05 10_02_56.npz')
    n_list = array_zip['n_list']
    metrics_id = ['acc', 'f1', 'roc', 'ap']
    metrics_names = ['Accuracy', 'F1', 'ROC-AUC', 'AP']
    for id, name in zip(metrics_id, metrics_names):
        if array_zip.get(id + '_our') is None:
            continue
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
    if array_zip.get('acc_tgt') is not None:
        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Target accuracy')
        tgt_acc = np.mean(array_zip['acc_tgt'], axis=0)
        ax.plot(n_list, tgt_acc[:, 0], 'rs--', label='Our model')
        ax.plot(n_list, tgt_acc[:, 1], 'gs--', label='Bottleneck')
        ax.grid()
        ax.legend()
        ax.set_xlabel('train sample size')
        ax.set_ylabel('Accuracy')
    plt.show()

def preload_train_test():
    X, y = get_proc_celeba_np()
    dump_train_test_on_disk(X, y)
    del X
    del y

if __name__=='__main__':
    cls_num = 256
    eps = 0.001
    device = 'cuda'
    ae_kw = {
        'latent_dim': 32, 
        'epochs_num': 30, 
        'batch_num': 100, 
        'l_r': 1e-3, 
        'device': device,
        'early_stop': 3,
    }
    preload_train_test()
    tiny_sample_exp()
    # draw_figures()