from torchvision.datasets import MNIST
from torchvision.transforms.v2 import PILToTensor
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple
import torch
import numba
from utility import f1_sep_scorer, acc_sep_scorer
from experiment_models import Autoencoder, BottleNeck, vert_patcher, quarter_patcher, EasyClustering, CAE
from sklearn.model_selection import train_test_split
from simple_concepts.model import SimpleConcepts
from uncls_model import SimpleConceptsNoCls
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

def no_patcher(X):
    return X

def window_patcher(X):
    X_tensor = torch.tensor(X, device='cpu')
    kernel_size = (10, 10)
    stride = 6
    patches = torch.nn.Unfold(kernel_size, stride=stride)(X_tensor).cpu().numpy()
    patches = patches.transpose((0, 2, 1))
    patches = patches.reshape(patches.shape[:2] + kernel_size)
    return patches

def latent_dim_exp():
    dim_list = [6, 8, 10, 12, 16, 20, 24, 28, 32]
    epoch_list = [50, 50, 40, 40, 35, 30, 30, 25, 25]
    f_1_hist = []
    for cur_dim, cur_ep_n in zip(dim_list, epoch_list):
        ae_kw['latent_dim'] = cur_dim
        ae_kw['epochs_num'] = cur_ep_n
        clusterizer = EasyClustering(cls_num, Autoencoder(**ae_kw))
        model = SimpleConcepts(cls_num, clusterizer, no_patcher, eps)
        # model = SimpleConceptsNoCls(cls_num, no_patcher, eps)
        # model.fit(X_train, y_train, y_train)
        # scores = model.predict(X_test, y_test)
        model.fit(X_train, y_train)
        scores = model.predict(X_test)
        acc = acc_sep_scorer(y_test, scores)
        f1 = f1_sep_scorer(y_test, scores)
        print("Accuracy for concepts:", acc)
        print("F1 for concepts:", f1)
        f_1_hist.append(f1)
    fig, ax = plt.subplots(1, 1)
    ax.plot(dim_list, f_1_hist, 'rs--')
    ax.grid()
    fig.suptitle('K means')
    ax.set_xlabel('latent dim')
    ax.set_ylabel('F1')
    plt.show()
    
    
def cae_exp():
    ae_kw['lmd'] = 1.0
    clusterizer = EasyClustering(cls_num, CAE(**ae_kw))
    model = SimpleConcepts(cls_num, clusterizer, no_patcher, eps)
    model.fit(X_train, y_train)
    
    latent_x = model.cls_model.autoencoder.predict_code(X_train)
    pred = model.cls_model.clusterizer.predict(latent_x)
    rng = np.random.default_rng()
    for i in range(cls_num):
        pred_i = np.where(pred == i)[0]
        idx_to_draw = rng.choice(pred_i, 4, replace=False)
        fig, ax = plt.subplots(2, 4)
        for j in range(4):
            cur_x = X[idx_to_draw[j], None, :]
            ax[0, j].imshow(cur_x[0, 0, ...])
            with torch.no_grad():
                cur_code = model.cls_model.autoencoder.np2torch(latent_x[idx_to_draw[j], None])
                cur_img = model.cls_model.autoencoder.decoder_(cur_code)
                im_np = cur_img.cpu().numpy()[0, 0, ...]
            ax[1, j].imshow(im_np)
    plt.show()
    
    scores = model.predict(X_test)
    acc = acc_sep_scorer(y_test, scores)
    f1 = f1_sep_scorer(y_test, scores)
    print("Accuracy for concepts:", acc)
    print("F1 for concepts:", f1)
    
def sliding_window_exp():
    clusterizer = EasyClustering(cls_num, Autoencoder(**ae_kw))
    model = SimpleConcepts(cls_num, clusterizer, window_patcher, eps)
    model.fit(X_train, y_train)
    scores = model.predict(X_test)
    acc = acc_sep_scorer(y_test, scores)
    f1 = f1_sep_scorer(y_test, scores)
    print("Accuracy for concepts:", acc)
    print("F1 for concepts:", f1)

if __name__=='__main__':
    cls_num = 10
    eps = 0.01
    ae_kw = {
        'latent_dim': 12, 
        'epochs_num': 8, 
        'batch_num': 1024, 
        'l_r': 1e-3, 
        'device': 'cpu'
    }
    X, y = get_proc_mnist_np()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    y_test = y_test[:, None]
    sliding_window_exp()