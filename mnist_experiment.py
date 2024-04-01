from torchvision.datasets import MNIST
from torchvision.transforms.v2 import PILToTensor
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from simple_concepts.model import SimpleConcepts
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from functools import partial
from typing import Tuple

'''
0 - the target, y mod 5 == 3
1 - even/odd: 0 or 1
2 - y <= 5: 0 or 1
3 - y mod 3: 0, 1, 2
'''
def get_mnist_concepts(y: np.ndarray) -> np.ndarray:
    result = np.zeros((y.shape[0], 4), dtype=np.int0)
    result[:, 0] = np.mod(y, 5) == 3
    result[:, 1] = np.mod(y, 2)
    result[:, 2] = y < 5
    result[:, 3] = np.mod(y, 3)
    return result

def get_proc_mnist_np() -> Tuple[np.ndarray, np.ndarray]:
    ds = DataLoader(MNIST('MNIST', download=False, transform=PILToTensor()), 8192, False)
    X_list = []
    y_list = []
    for x, y in ds:
        X_list.append(x[:, 0, ...])
        y_list.append(y)
    X_np = np.concatenate(X_list, axis=0) # (60000, 28, 28)
    std = np.std(X_np, 0, keepdims=True)
    std[std < 1e-15] = 1.0
    X_np = (X_np - np.mean(X_np, 0, keepdims=True)) / std
    y_np = np.concatenate(y_list, axis=0)
    y_np = get_mnist_concepts(y_np)
    return X_np, y_np

class EasyClustering():
    def __init__(self, cls_num: int, pca_feat_num: int):
        self.pca = PCA(pca_feat_num)
        self.k_means = MiniBatchKMeans(cls_num, batch_size=2048)
        
    def fit(self, X: np.ndarray) -> 'EasyClustering':
        if X.ndim > 2:
            X = X.reshape((X.shape[0], -1))
        x_pca = self.pca.fit_transform(X)
        self.k_means.fit(x_pca)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim > 2:
            X = X.reshape((X.shape[0], -1))
        x_pca = self.pca.transform(X)
        return self.k_means.predict(x_pca)

def vert_patcher(X: np.ndarray) -> np.ndarray:
    half_h = X.shape[1] // 2
    result = np.zeros((X.shape[0], 2, half_h, X.shape[-1]), dtype=X.dtype)
    result[:, 0, ...] = X[:, :half_h, :]
    result[:, 1, ...] = X[:, :half_h, :]
    return result

def quarter_patcher(X: np.ndarray) -> np.ndarray:
    half_h = X.shape[1] // 2
    half_w = X.shape[2] // 2
    result = np.zeros((X.shape[0], 4, half_h, half_w), dtype=X.dtype)
    result[:, 0, ...] = X[:, :half_h, :half_w]
    result[:, 1, ...] = X[:, :half_h, half_w:]
    result[:, 2, ...] = X[:, half_h:, :half_w]
    result[:, 3, ...] = X[:, half_h:, half_w:]
    return result

def sep_scorer(y_true, y_score, scorer):
    assert y_true.ndim == y_score.ndim == 2
    assert y_true.shape[1] == y_score.shape[1]
    result = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        result[i] = scorer(y_true[:, i], y_score[:, i])
    return result

def acc_sep_scorer(y_true, y_score):
    return sep_scorer(y_true, y_score, accuracy_score)

def roc_auc(y_true, y_score):
    ra_scorer = partial(roc_auc_score, average='macro', multi_class='ovr')
    return sep_scorer(y_true, y_score, ra_scorer)
    

if __name__=='__main__':
    cls_num = 8
    pca_feats = 20
    X, y = get_proc_mnist_np()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clusterizer = EasyClustering(cls_num, pca_feats)
    model = SimpleConcepts(cls_num, clusterizer, vert_patcher)
    model.fit(X_train, y_train)
    scores = model.predict(X_test)
    acc = acc_sep_scorer(y_test, scores)
    # f1 = roc_auc(y_test, scores)
    print("Accuracy for concepts:", acc)
    # print("F1 for concepts:", f1)