from torchvision.datasets import MNIST
from torchvision.transforms.v2 import PILToTensor
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from simple_concepts.model import SimpleConcepts
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from experiment_models import Autoencoder, BottleNeck, vert_patcher, EasyClustering
from utility import f1_sep_scorer, acc_sep_scorer
from cls_mnist_exp import window_patcher


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
        X_list.append(x)
        y_list.append(y)
    X_np = np.concatenate(X_list, axis=0) # (60000, 28, 28)
    std = np.std(X_np, 0, keepdims=True)
    std[std < 1e-15] = 1.0
    X_np = (X_np - np.mean(X_np, 0, keepdims=True)) / std
    y_np = np.concatenate(y_list, axis=0)
    y_np = get_mnist_concepts(y_np)
    return X_np, y_np
    

if __name__=='__main__':
    cls_num = 256
    eps = 0.001
    ae_kw = {
        'latent_dim': 20, 
        'epochs_num': 5, 
        'batch_num': 1024, 
        'l_r': 1e-3, 
        'device': 'cuda'
    }
    X, y = get_proc_mnist_np()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clusterizer = EasyClustering(cls_num, Autoencoder(**ae_kw))
    model = SimpleConcepts(cls_num, clusterizer, window_patcher, eps)
    model.fit(X_train, y_train)
    scores = model.predict(X_test)
    acc = acc_sep_scorer(y_test, scores)
    f1 = f1_sep_scorer(y_test, scores)
    print("Accuracy for concepts:", acc)
    print("F1 for concepts:", f1)
    
    print('--- Bottleneck ---')
    
    model = BottleNeck(1, 20, 256, 1e-3, 'cuda')
    model.fit(X_train, y_train[:, 0], y_train[:, 1:])
    scores = model.predict(X_test)
    acc = acc_sep_scorer(y_test, scores)
    f1 = f1_sep_scorer(y_test, scores)
    print("Accuracy for concepts:", acc)
    print("F1 for concepts:", f1)