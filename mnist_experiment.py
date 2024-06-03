from torchvision.datasets import MNIST
from torchvision.transforms.v2 import PILToTensor
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from simple_concepts.model import SimpleConcepts
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Tuple, List
from experiment_models import Autoencoder, BottleNeck, vert_patcher, EasyClustering
from utility import f1_sep_scorer, acc_sep_scorer
import sympy as sp
from sympy.logic.boolalg import Equivalent
from rule_checker import RuleChecker


'''
0 - the target, y mod 5 == 3 or y mod 5 == 1
1 - even/odd: 0 or 1
2 - y < 5: 0 or 1
3 - y mod 3: 0, 1, 2
'''
def get_mnist_concepts(y: np.ndarray) -> np.ndarray:
    result = np.zeros((y.shape[0], 4), dtype=np.int0)
    mod_5 = np.mod(y, 5)
    result[:, 0] = np.logical_or(mod_5 == 3, mod_5 == 1)
    result[:, 1] = np.mod(y, 2)
    result[:, 2] = y < 5
    result[:, 3] = np.mod(y, 3)
    for i in range(10):
        idx = np.argwhere(y == i).ravel()[0]
        print(f'{i} -> {result[idx]}')
    return result

def get_proc_mnist_np() -> Tuple[np.ndarray, np.ndarray]:
    ds = DataLoader(MNIST('MNIST', download=False, transform=PILToTensor()), 60000, False)
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
    y_np = get_mnist_concepts(y_np)
    return X_np, y_np

def base_experiment():
    no_patcher = lambda x: x
    clusterizer = EasyClustering(cls_num, Autoencoder(**ae_kw))
    model = SimpleConcepts(cls_num, clusterizer, vert_patcher, eps)
    model.fit(X_train, y_train)
    scores = model.predict(X_test)
    acc = acc_sep_scorer(y_test, scores)
    f1 = f1_sep_scorer(y_test, scores)
    print("Accuracy for concepts:", acc)
    print("F1 for concepts:", f1)
    
    # print('--- Bottleneck ---')
    
    # model = BottleNeck(1, 20, 256, 1e-3, 'cuda')
    # model.fit(X_train, y_train[:, 0], y_train[:, 1:])
    # scores = model.predict(X_test)
    # acc = acc_sep_scorer(y_test, scores)
    # f1 = f1_sep_scorer(y_test, scores)
    # print("Accuracy for concepts:", acc)
    # print("F1 for concepts:", f1)
    
def rule_exp():
    y_train_fixed = y_train.copy()
    wrong_idx = np.arange(X_train.shape[0])
    wrong_idx, _ = train_test_split(wrong_idx, train_size=0.5)
    y_train_fixed[wrong_idx, 0] = np.logical_not(y_train[wrong_idx, 0])
    no_patcher = lambda x: x[:, None, ...]
    clusterizer = EasyClustering(cls_num, Autoencoder(**ae_kw))
    model = SimpleConcepts(cls_num, clusterizer, no_patcher, eps)
    model.fit(X_train, y_train_fixed)
    
    scores = model.predict(X_test)
    acc = acc_sep_scorer(y_test, scores)
    f1 = f1_sep_scorer(y_test, scores)
    print("Accuracy for concepts:", acc)
    print("F1 for concepts:", f1)
    probas_before = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test[:, 0], probas_before[:, 1])
    print("ROC-AUC for target:", roc_auc)
    ap = average_precision_score(y_test[:, 0], probas_before[:, 1])
    print("AP for target:", ap)
    
    x_2_1, x_0_1, x_1_1, x_1_1, x_2_0, x_0_0, x_1_0 = sp.symbols('x_2_1, x_0_1, x_1_1, x_1_1, x_2_0, x_0_0, x_1_0')
    rule_pos = Equivalent((x_1_1 & x_2_1) | (x_1_0 & x_2_0), x_0_1)
    rule_neg = Equivalent((x_1_0 & x_2_1) | (x_1_1 & x_2_0), x_0_0)
    rule_1 = [rule_pos, rule_neg]
    rule_2 = [(x_1_1 & x_2_1) >> x_0_1]
    rule_3 = [rule_pos]
    rule_4 = [(x_1_1 & x_2_1) >> x_0_1, (x_1_0 & x_2_1) >> x_0_0]
    rules_all_list = [rule_2, rule_4, rule_3]
    
    errors_n = check_rule_errors(scores, rule_1)
    print(f'Rule errors (labels): {errors_n / scores.shape[0]}')
    checker = get_rule_checker(model, rule_1)
    errors_n = checker.check(probas_before)
    print(f'Rule errors (probas): {errors_n / scores.shape[0]}')
    
    bin_n = 12
    fig, ax = plt.subplots(1, 1 + len(rules_all_list), figsize=(6 * len(rules_all_list), 6))
    fig.suptitle('Target concept proba density')
    def draw_dens(proba, idx, name):
        vals, bins = np.histogram(proba, bin_n, range=(0, 1), density=True)
        ax[idx].hist(bins[:-1], bins, weights=vals, zorder=100)
        ax[idx].set_title(name)
        ax[idx].set_xlabel('Probability')
        ax[idx].grid()
    draw_dens(probas_before[:, 1], 0, 'Without rule')
    
    
    for i, rule_in in enumerate(rules_all_list):
        rule_model = model.get_model_with_rules(rule_in)
        print(f'--- Inserted rule {rule_in} ---')
        scores = rule_model.predict(X_test)
        acc = acc_sep_scorer(y_test, scores)
        f1 = f1_sep_scorer(y_test, scores)
        print("Accuracy for concepts:", acc)
        print("F1 for concepts:", f1)
        probas_after = rule_model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test[:, 0], probas_after[:, 1])
        print("ROC-AUC for target:", roc_auc)
        ap = average_precision_score(y_test[:, 0], probas_after[:, 1])
        print("AP for target:", ap)
        errors_n = check_rule_errors(scores, rule_1)
        print(f'Rule errors (labels): {errors_n / scores.shape[0]}')
        errors_n = checker.check(probas_after)
        print(f'Rule errors (probas): {errors_n / scores.shape[0]}')
        draw_dens(probas_after[:, 1], i + 1, f'${sp.latex(sp.And(*rule_in))}$')
    plt.tight_layout()
    plt.show()
    
def check_rule_errors(concepts: np.ndarray, rules: List[sp.Expr]):
    master_rule = sp.And(*rules)
    vars = list(master_rule.free_symbols)
    default_dict = dict([(str(v), False) for v in vars])
    Z, Pr_Z = np.unique(concepts, return_counts=True, axis=0)
    errors = 0
    for cur_z, pr_num in zip(Z, Pr_Z):
        cur_dict = default_dict.copy()
        for r in range(cur_z.shape[0]):
            v = cur_z[r]
            cur_dict[f'x_{r}_{v}'] = True
        rule_val = master_rule.subs(cur_dict)
        if not rule_val:
            errors += pr_num
    return errors


def get_rule_checker(model, rules) -> RuleChecker:
    master_rule = sp.And(*rules)
    vars = list(master_rule.free_symbols)
    sub_dict = dict()
    for var in vars:
        r, v = map(int, str(var).split('_')[-2:])
        sub_dict[var] = sp.Eq(sp.Symbol(f'y_{r}'), v)
    rule_list = [master_rule.subs(sub_dict)]
    cards = dict()
    for i, cardinality in enumerate(model.v):
        cards[f'y_{i}'] = cardinality
    return RuleChecker(cards, rule_list)


def wrong_concepts():
    y_train_fixed = y_train.copy()
    wrong_part_list = [0.1, 0.25, 0.5, 0.75, 0.9]
    # wrong_part_list = [0.1, 0.25]
    f1_all = np.empty((len(wrong_part_list), 2)) # without and with rule
    acc_all = np.empty((len(wrong_part_list), 2))
    roc_auc_all = np.empty((len(wrong_part_list), 2))
    av_prec_all = np.empty((len(wrong_part_list), 2))
    error_all = np.empty((len(wrong_part_list), 2))
    for i, w_p in enumerate(wrong_part_list):
        wrong_idx = np.argwhere(np.logical_and(y_train[:, 1] == 1, y_train[:, 2] == 1)).ravel()
        wrong_idx, _ = train_test_split(wrong_idx, train_size=w_p)
        y_train_fixed[wrong_idx, 0] = np.logical_not(y_train[wrong_idx, 0])
        no_patcher = lambda x: x
        clusterizer = EasyClustering(cls_num, Autoencoder(**ae_kw))
        model = SimpleConcepts(cls_num, clusterizer, no_patcher, eps)
        model.fit(X_train, y_train_fixed)
        
        scores = model.predict(X_test)
        acc = acc_sep_scorer(y_test, scores)
        f1 = f1_sep_scorer(y_test, scores)
        print("Accuracy for concepts:", acc)
        print("F1 for concepts:", f1)
        probas = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test[:, 0], probas[:, 1])
        print("ROC-AUC for target:", roc_auc)
        ap = average_precision_score(y_test[:, 0], probas[:, 1])
        print("AP for target:", ap)
        acc_all[i, 0] = acc[0]
        f1_all[i, 0] = f1[0]
        roc_auc_all[i, 0] = roc_auc
        av_prec_all[i, 0] = ap
        
        
        x_2_1, x_1_1, x_0_1, x_1_1 = sp.symbols('x_2_1, x_1_1, x_0_1, x_1_1')
        rule_inv = (x_1_1 & x_2_1) >> x_0_1
        rule_in = [rule_inv]
        errors_n = check_rule_errors(scores, rule_in)
        print(f'Rule errors (labels): {errors_n / scores.shape[0]}')
        checker = get_rule_checker(model, rule_in)
        errors_n = checker.check(probas)
        error_all[i, 0] = errors_n / scores.shape[0]
        print(f'Rule errors (probas): {error_all[i, 0]}')
        
        rule_model = model.get_model_with_rules(rule_in)
        print('--- Inserted rule ---')
        scores = rule_model.predict(X_test)
        acc = acc_sep_scorer(y_test, scores)
        f1 = f1_sep_scorer(y_test, scores)
        print("Accuracy for concepts:", acc)
        print("F1 for concepts:", f1)
        probas = rule_model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test[:, 0], probas[:, 1])
        print("ROC-AUC for target:", roc_auc)
        ap = average_precision_score(y_test[:, 0], probas[:, 1])
        print("AP for target:", ap)
        errors_n = check_rule_errors(scores, rule_in)
        print(f'Rule errors (labels): {errors_n / scores.shape[0]}')
        errors_n = checker.check(probas)
        error_all[i, 1] = errors_n / scores.shape[0]
        print(f'Rule errors (probas): {error_all[i, 1]}')
        acc_all[i, 1] = acc[0]
        f1_all[i, 1] = f1[0]
        roc_auc_all[i, 1] = roc_auc
        av_prec_all[i, 1] = ap
    arrays = [acc_all, f1_all, roc_auc_all, av_prec_all, error_all]
    names = ['Accuracy', 'F1', 'ROC-AUC', 'Average precision', 'Rule violation rate']
    for name, metric in zip(names, arrays):
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(name)
        ax.plot(wrong_part_list, metric[:, 0], 'gs--', label='Baseline')
        ax.plot(wrong_part_list, metric[:, 1], 'rs--', label='Applied the rule')
        ax.grid()
        ax.legend()
    plt.show()
    
def tiny_sample_exp():
    no_patcher = lambda x: x[:, None, ...]
    global y_train
    y_train, c_train = y_train[:, 0], y_train[:, 1:]
    n_list = [1000, 2000, 3000, 4000, 5000]
    epochs_n = [200, 150, 70, 50, 50]
    iter_num = 10
    f1_our = np.zeros((iter_num, len(n_list), 4))
    f1_bottleneck = np.zeros((iter_num, len(n_list), 4))
    acc_our = np.zeros((iter_num, len(n_list), 4))
    acc_bottleneck = np.zeros((iter_num, len(n_list), 4))
    for i in range(iter_num):
        for j, n in enumerate(n_list):
            ep_n = epochs_n[j]
            ae_kw['epochs_num'] = ep_n
            X_small_train, _, y_small_train, _, c_small_train, _ = train_test_split(X_train, y_train, c_train, train_size=n)
            clusterizer = EasyClustering(cls_num, Autoencoder(**ae_kw))
            model = SimpleConcepts(cls_num, clusterizer, no_patcher, eps)
            model.fit(X_small_train, y_small_train, c_small_train)
            scores = model.predict(X_test)
            acc = acc_sep_scorer(y_test, scores)
            f1 = f1_sep_scorer(y_test, scores)
            print("Accuracy for concepts:", acc)
            print("F1 for concepts:", f1)
            f1_our[i, j, :] = f1
            acc_our[i, j, :] = acc
            
            print('--- Bottleneck ---')
            
            model = BottleNeck(1, ep_n, 256, 1e-3, 'cuda', 2)
            model.fit(X_small_train, y_small_train, c_small_train)
            scores = model.predict(X_test)
            acc = acc_sep_scorer(y_test, scores)
            f1 = f1_sep_scorer(y_test, scores)
            f1_bottleneck[i, j, :] = f1
            acc_bottleneck[i, j, :] = acc
            print("Accuracy for concepts:", acc)
            print("F1 for concepts:", f1)
    np.savez('mnist_metrics4', acc_our=acc_our, f1_our=f1_our, acc_btl=acc_bottleneck, f1_btl=f1_bottleneck)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex='all', sharey='all')
    fig.suptitle('F1')
    f1_our = np.mean(f1_our, axis=0)
    f1_bottleneck = np.mean(f1_bottleneck, axis=0)
    for i in range(4):
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
    acc_our = np.mean(acc_our, axis=0)
    acc_bottleneck = np.mean(acc_bottleneck, axis=0)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex='all', sharey='all')
    fig.suptitle('Accuracy')
    for i in range(4):
        ax = axes[i]
        ax.plot(n_list, acc_our[:, i], 'rs--', label='Our model')
        ax.plot(n_list, acc_bottleneck[:, i], 'gs--', label='Bottleneck')
        ax.grid()
        ax.legend()
        ax.set_xlabel('train sample size')
        if i == 0:
            ax.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
    
def draw_figures():
    n_list = [1000, 2000, 3000, 4000, 5000]
    arrays = np.load('mnist_metrics1.npz')
    acc_our, f1_our, acc_bottleneck, f1_bottleneck = arrays['acc_our'], arrays['f1_our'], arrays['acc_btl'], arrays['f1_btl']
    fig, axes = plt.subplots(2, 4, figsize=(30, 6), sharex='all', sharey='all')
    f1_our = np.mean(f1_our, axis=0)
    f1_bottleneck = np.mean(f1_bottleneck, axis=0)
    for i in range(4):
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
    for i in range(4):
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
    cls_num = 100
    eps = 0.001
    ae_kw = {
        'latent_dim': 16, 
        'epochs_num': 30, 
        'batch_num': 1024, 
        'l_r': 1e-3, 
        'device': 'cuda',
        'early_stop': 3
    }
    X, y = get_proc_mnist_np()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    # tiny_sample_exp()
    # wrong_concepts()
    # draw_figures()
    rule_exp()
