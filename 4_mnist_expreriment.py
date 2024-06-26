from torchvision.datasets import MNIST
from torchvision.transforms.v2 import PILToTensor
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple
import sympy as sp
import numba
from utility import f1_sep_scorer, acc_sep_scorer
from experiment_models import Autoencoder, BottleNeck, quarter_patcher, EasyClustering
from sklearn.model_selection import train_test_split
from simple_concepts.model import SimpleConcepts
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from mnist_experiment import get_rule_checker, check_rule_errors

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
    
def wrong_concepts():
    y_train_all = np.concatenate((y_train.copy()[:, None], c_train), axis=-1)
    max_val = np.max(y_train)
    wrong_part_list = [0.1, 0.25, 0.5, 0.75, 0.9]
    # wrong_part_list = [0.1, 0.5]
    f1_all = np.empty((len(wrong_part_list), 2)) # without and with rule
    acc_all = np.empty((len(wrong_part_list), 2))
    # roc_auc_all = np.empty((len(wrong_part_list), 2))
    # av_prec_all = np.empty((len(wrong_part_list), 2))
    error_all = np.empty((len(wrong_part_list), 2))
    idx_to_spoil = np.argwhere(y_train_all[:, 10] == 1).ravel()
    for i, w_p in enumerate(wrong_part_list):
        wrong_idx, _ = train_test_split(idx_to_spoil, train_size=w_p)
        y_train_fixed = y_train_all.copy()
        y_train_fixed[wrong_idx, 0] = np.random.randint(0, max_val, len(wrong_idx))
        clusterizer = EasyClustering(cls_num, Autoencoder(**ae_kw))
        model = SimpleConcepts(cls_num, clusterizer, quarter_patcher, eps)
        model.fit(X_train, y_train_fixed)
        
        scores = model.predict(X_test)
        acc = acc_sep_scorer(y_test, scores)
        f1 = f1_sep_scorer(y_test, scores)
        print("Accuracy for concepts:", acc)
        print("F1 for concepts:", f1)
        acc_all[i, 0] = acc[0]
        f1_all[i, 0] = f1[0]
        
        x_0_max, x_10_1 = sp.symbols(f'x_0_{max_val}, x_10_1')
        rule_inv = x_10_1 >> x_0_max
        rule_in = [rule_inv]
        errors_n = check_rule_errors(scores, rule_in)
        error_all[i, 0] = errors_n / scores.shape[0]
        print(f'Rule errors (labels): {error_all[i, 0]}')
        checker = get_rule_checker(model, rule_in)
        probas = model.predict_proba(X_test)
        errors_n = checker.check(probas)
        print(f'Rule errors (probas): {errors_n / scores.shape[0]}')
        
        rule_model = model.get_model_with_rules(rule_in)
        print('--- Inserted rule ---')
        scores = rule_model.predict(X_test)
        acc = acc_sep_scorer(y_test, scores)
        f1 = f1_sep_scorer(y_test, scores)
        print("Accuracy for concepts:", acc)
        print("F1 for concepts:", f1)
        errors_n = check_rule_errors(scores, rule_in)
        error_all[i, 1] = errors_n / scores.shape[0]
        print(f'Rule errors (labels): {error_all[i, 1]}')
        probas = rule_model.predict_proba(X_test)
        errors_n = checker.check(probas)
        print(f'Rule errors (probas): {errors_n / scores.shape[0]}')
        acc_all[i, 1] = acc[0]
        f1_all[i, 1] = f1[0]
    arrays = [acc_all, f1_all, error_all]
    names = ['Accuracy', 'F1', 'Rule violations rate']
    wrong_part_fig = np.asarray(wrong_part_list) * idx_to_spoil.shape[0] / y_train.shape[0]
    for name, metric in zip(names, arrays):
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(name)
        ax.plot(wrong_part_fig, metric[:, 0], 'r', label='Baseline', marker='^', markerfacecolor='white', markersize=8)
        ax.plot(wrong_part_fig, metric[:, 1], 'b', label='Applied the rule', marker='o', markerfacecolor='white', markersize=8)
        ax.grid()
        ax.legend()
        ax.set_xlabel('Part of the mismatched labels')
        ax.set_ylabel(name)
    plt.show()
    
def tiny_sample_exp():
    n_list = [1000, 2000, 3000, 4000, 5000]
    epochs_n = [40, 40, 30, 30, 20]
    iter_num = 100
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
            
            model = BottleNeck(1, ep_n, 256, 1e-3, 'cuda', None)
            model.fit(X_small_train, y_small_train, c_small_train)
            scores = model.predict(X_test)
            acc = acc_sep_scorer(y_test, scores)
            f1 = f1_sep_scorer(y_test, scores)
            f1_bottleneck[i, j, :] = f1
            acc_bottleneck[i, j, :] = acc
            print("Accuracy for concepts:", acc)
            print("F1 for concepts:", f1)
    np.savez('metrics5', acc_our=acc_our, f1_our=f1_our, acc_btl=acc_bottleneck, f1_btl=f1_bottleneck)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharex='all', sharey='all')
    f1_our = np.mean(f1_our, axis=0)
    f1_bottleneck = np.mean(f1_bottleneck, axis=0)
    for i in range(11):
        ax = axes[i // 4, i % 4]
        ax.plot(n_list, f1_our[:, i], 'r', label='FI-CBL', marker='^', markerfacecolor='white', markersize=8)
        ax.plot(n_list, f1_bottleneck[:, i], 'b', label='CBM', marker='o', markerfacecolor='white', markersize=8)
        ax.grid()
        ax.legend()
        ax.set_xlabel('training set size')
        ax.set_title('Concept ' + str(i))
        if i == 0:
            ax.set_ylabel('F1')
    axes[-1, -1].remove()
    plt.tight_layout()
    for i in range(3):
        ax = axes[-1, i]
        pos = ax.get_position()
        x = pos.bounds[0] + pos.bounds[2] / 2
        ax.set_position(Bbox.from_bounds(x, *pos.bounds[1:]))
    plt.show()

if __name__=='__main__':
    n = 50000
    cls_num = 128
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
    # wrong_concepts()
