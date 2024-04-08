from torchvision.datasets import MNIST
from torchvision.transforms.v2 import PILToTensor
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from simple_concepts.model import SimpleConcepts
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from functools import partial
from typing import Tuple
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class Autoencoder(torch.nn.Module):
    
    def _get_encoder(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, (2, 3), stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 32, (2, 3), stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, (2, 2), stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(320, self.latent_dim),
        ).to(self.device)
        
    def _get_decoder(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 320),
            Reshape(-1, 32, 2, 5),
            torch.nn.ConvTranspose2d(32, 32, 2, stride=1, output_padding=(0, 0)),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(32, 16, (2, 3), stride=(2, 2), output_padding=(0, 0)),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(16, 1, (4, 4), stride=2),
        ).to(self.device)
    
    def __init__(self, latent_dim: int, epochs_num: int, batch_num: int, l_r: float, device: torch.device):
        super().__init__()
        self.latent_dim = latent_dim
        self.epochs_num = epochs_num
        self.batch_num = batch_num
        self.l_r = l_r
        self.device = torch.device(device)
        
    def np2torch(self, arr):
        return torch.tensor(arr, dtype=torch.float32, device=self.device)
        
    def fit(self, X: np.ndarray):
        if X.ndim == 3:
            X = X[:, None, ...]
        self.encoder_ = self._get_encoder()
        self.decoder_ = self._get_decoder()
        self.optimizer_ = torch.optim.AdamW(self.parameters(), self.l_r, weight_decay=1e-5)
        self.loss_ = torch.nn.MSELoss()
        X = self.np2torch(X)
        # code = self.encoder_(X)
        # decode = self.decoder_(code)
        # print(decode.shape)
        dataset = TensorDataset(X)
        self.train()
        for e in range(self.epochs_num):
            data_loader = DataLoader(dataset, self.batch_num, True)
            prog_bar = tqdm(
                data_loader, f'Epoch {e}', unit='batch', ascii=True)
            cum_mse = 0
            for i, (x_b, ) in enumerate(prog_bar):
                self.optimizer_.zero_grad()
                code = self.encoder_(x_b)
                rec = self.decoder_(code)
                loss = self.loss_(x_b, rec)
                loss.backward()
                self.optimizer_.step()
                cum_mse += loss.item()

                prog_bar.set_postfix(MSE=cum_mse / (i + 1))
        self.eval()
        return self
    
    def predict_code(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if X.ndim == 3:
                X = X[:, None, ...]
            X_t = self.np2torch(X)
            code = self.encoder_(X_t)
            # idx = [0, 32, 6, 77, 51]
            # for i in idx:
            #     fig, ax = plt.subplots(1, 2)
            #     ax[0].imshow(X[i, 0, ...])
            #     decoded = self.decoder_(code[i, None, :]).cpu().numpy()
            #     ax[1].imshow(decoded[0, 0, ...])
            # plt.show()
            code = code.cpu().numpy().astype(np.double)
            
            
            return code
    

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
    def __init__(self, cls_num: int, autoencoder_kw):
        self.autoencoder = Autoencoder(**autoencoder_kw)
        self.clusterizer = MiniBatchKMeans(cls_num, batch_size=2048)
        # self.clusterizer = GaussianMixture(cls_num, covariance_type='diag')
        
    def fit(self, X: np.ndarray) -> 'EasyClustering':
        self.autoencoder.fit(X)
        latent_x = self.autoencoder.predict_code(X)
        # if X.ndim > 2:
        #     X = X.reshape((X.shape[0], -1))
        # x_pca = self.pca.fit_transform(X)
        # plt.scatter(*x_pca.T)
        # plt.show()
        self.clusterizer.fit(latent_x)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        latent_x = self.autoencoder.predict_code(X)
        return self.clusterizer.predict(latent_x)

def vert_patcher(X: np.ndarray) -> np.ndarray:
    half_h = X.shape[1] // 2
    result = np.zeros((X.shape[0], 2, half_h, X.shape[-1]), dtype=X.dtype)
    result[:, 0, ...] = X[:, :half_h, :]
    result[:, 1, ...] = X[:, half_h:, :]
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

def f1_sep_scorer(y_true, y_score):
    f1_scorer = partial(f1_score, average='macro')
    return sep_scorer(y_true, y_score, f1_scorer)
    

if __name__=='__main__':
    cls_num = 14
    eps = 0.01
    ae_kw = {
        'latent_dim': 32, 
        'epochs_num': 30, 
        'batch_num': 128, 
        'l_r': 1e-3, 
        'device': 'cuda'
    }
    X, y = get_proc_mnist_np()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clusterizer = EasyClustering(cls_num, ae_kw)
    model = SimpleConcepts(cls_num, clusterizer, vert_patcher, eps)
    model.fit(X_train, y_train)
    scores = model.predict(X_test)
    acc = acc_sep_scorer(y_test, scores)
    f1 = f1_sep_scorer(y_test, scores)
    print("Accuracy for concepts:", acc)
    print("F1 for concepts:", f1)