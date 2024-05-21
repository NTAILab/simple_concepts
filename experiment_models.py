import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from typing import Optional
from copy import deepcopy

class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class Autoencoder(torch.nn.Module):
    # for halfs
    # def _get_encoder(self):
    #     return torch.nn.Sequential(
    #         torch.nn.Conv2d(1, 16, (2, 3), stride=2),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Conv2d(16, 32, (2, 3), stride=2),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Conv2d(32, 32, (2, 2), stride=1),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Flatten(),
    #         torch.nn.Linear(320, self.latent_dim),
    #     ).to(self.device)
        
    # def _get_decoder(self):
    #     return torch.nn.Sequential(
    #         torch.nn.Linear(self.latent_dim, 320),
    #         Reshape(-1, 32, 2, 5),
    #         torch.nn.ConvTranspose2d(32, 32, 2, stride=1, output_padding=(0, 0)),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.ConvTranspose2d(32, 16, (2, 3), stride=(2, 2), output_padding=(0, 0)),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.ConvTranspose2d(16, 1, (4, 4), stride=2),
    #     ).to(self.device)
    
    # whole images, deep enough
    # def _get_encoder(self):
    #     return torch.nn.Sequential(
    #         torch.nn.Conv2d(1, 16, (3, 3), stride=2),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Conv2d(16, 32, (3, 3), stride=2),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Conv2d(32, 32, (3, 3), stride=1),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Flatten(),
    #         torch.nn.Linear(512, self.latent_dim),
    #     ).to(self.device)
        
    # def _get_decoder(self):
    #     return torch.nn.Sequential(
    #         torch.nn.Linear(self.latent_dim, 512),
    #         Reshape(-1, 32, 4, 4),
    #         torch.nn.ConvTranspose2d(32, 32, 3, stride=1, output_padding=(0, 0)),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.ConvTranspose2d(32, 16, (3, 3), stride=(2, 2), output_padding=(0, 0)),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.ConvTranspose2d(16, 1, (3, 3), stride=2, output_padding=1),
    #     ).to(self.device)
    
    # whole images, not deep
    # def _get_encoder(self):
    #     return torch.nn.Sequential(
    #         torch.nn.Conv2d(1, 16, (8, 8), stride=2),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Conv2d(16, 32, (6, 6), stride=2),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Flatten(),
    #         torch.nn.Linear(288, self.latent_dim),
    #     ).to(self.device)
        
    # def _get_decoder(self):
    #     return torch.nn.Sequential(
    #         torch.nn.Linear(self.latent_dim, 288),
    #         Reshape(-1, 32, 3, 3),
    #         torch.nn.ConvTranspose2d(32, 16, 6, stride=2, output_padding=(1, 1)),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.ConvTranspose2d(16, 1, (8, 8), stride=(2, 2), output_padding=(0, 0)),
    #     ).to(self.device)
    
    # 10x10
    # def _get_encoder(self):
    #     return torch.nn.Sequential(
    #         torch.nn.Conv2d(1, 6, (3, 3), stride=1),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Conv2d(6, 6, (3, 3), stride=2),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Flatten(),
    #         torch.nn.Linear(54, self.latent_dim),
    #     ).to(self.device)
        
    # def _get_decoder(self):
    #     return torch.nn.Sequential(
    #         torch.nn.Linear(self.latent_dim, 54),
    #         Reshape(-1, 6, 3, 3),
    #         torch.nn.ConvTranspose2d(6, 6, 3, stride=2, output_padding=(1, 1)),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.ConvTranspose2d(6, 1, 3, stride=1, output_padding=(0, 0)),
    #     ).to(self.device)
    
    # def _get_encoder(self):
    #     return torch.nn.Sequential(
    #         torch.nn.Flatten(),
    #         torch.nn.Linear(784, 256),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Linear(256, 128),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Linear(128, self.latent_dim),
    #     ).to(self.device)
    
    # celeba quarters
    def _get_encoder(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, (16, 16), stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 32, (16, 16), stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, (24, 16), stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(2016, self.latent_dim),
        ).to(self.device)
        
    def _get_decoder(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 2016),
            Reshape(-1, 32, 9, 7),
            torch.nn.ConvTranspose2d(32, 32, (24, 16), stride=1, output_padding=(0, 0)),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(32, 16, (16, 16), stride=(1, 1), output_padding=(0, 0)),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(16, 3, (16, 16), stride=2, output_padding=1),
        ).to(self.device)
    
    # celeba all
    # def _get_encoder(self):
    #     return torch.nn.Sequential(
    #         torch.nn.Conv2d(3, 16, (32, 24), stride=3),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Conv2d(16, 32, (16, 16), stride=2),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Conv2d(32, 32, (16, 16), stride=1),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Flatten(),
    #         torch.nn.Linear(1152, self.latent_dim),
    #     ).to(self.device)
        
    # def _get_decoder(self):
    #     return torch.nn.Sequential(
    #         torch.nn.Linear(self.latent_dim, 1152),
    #         Reshape(-1, 32, 9, 4),
    #         torch.nn.ConvTranspose2d(32, 32, (16, 16), stride=1, output_padding=(0, 0)),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.ConvTranspose2d(32, 16, (16, 16), stride=(2, 2), output_padding=(1, 0)),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.ConvTranspose2d(16, 3, (32, 24), stride=3, output_padding=(0, 1)),
    #     ).to(self.device)
    
    # 110x90
    def _get_encoder(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, (20, 16), stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 32, (16, 16), stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, (8, 8), stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(1440, self.latent_dim),
        ).to(self.device)
        
    def _get_decoder(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 1440),
            Reshape(-1, 32, 9, 5),
            torch.nn.ConvTranspose2d(32, 32, (8, 8), stride=1, output_padding=(0, 0)),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(32, 16, (16, 16), stride=(2, 2), output_padding=(0, 0)),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(16, 3, (20, 16), stride=2, output_padding=0),
        ).to(self.device)
    
    def __init__(self, latent_dim: int, epochs_num: int, batch_num: int, l_r: float, device: torch.device, early_stop: Optional[int]=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.epochs_num = epochs_num
        self.batch_num = batch_num
        self.l_r = l_r
        self.device = torch.device(device)
        self.loss_fn = torch.nn.functional.mse_loss#torch.nn.MSELoss()
        self.early_stop = early_stop
        
    def np2torch(self, arr, device=None):
        if device is None:
            device = self.device
        return torch.tensor(arr, dtype=torch.float32, device=device)
        
    def fit(self, X: np.ndarray):
        if X.ndim == 3:
            X = X[:, None, ...]
        self.encoder_ = self._get_encoder()
        self.decoder_ = self._get_decoder()
        self.optimizer_ = torch.optim.AdamW(self.parameters(), self.l_r, weight_decay=1e-5)
        X = self.np2torch(X, device='cpu')
        # code = self.encoder_(X)
        # decode = self.decoder_(code)
        # print(decode.shape)
        dataset = TensorDataset(X)
        self.train()
        weights = deepcopy(self.state_dict())
        best_mse = float('inf')
        patience = 0
        for e in range(self.epochs_num):
            data_loader = DataLoader(dataset, self.batch_num, True)
            prog_bar = tqdm(
                data_loader, f'Epoch {e}', unit='batch', ascii=True)
            cum_mse = 0
            for i, (x_b, ) in enumerate(prog_bar):
                x_b = x_b.to(self.device)
                self.optimizer_.zero_grad()
                code = self.encoder_(x_b)
                rec = self.decoder_(code)
                loss = self.loss_fn(x_b, rec)
                loss.backward()
                self.optimizer_.step()
                cum_mse += loss.item()

                prog_bar.set_postfix(Loss=cum_mse / (i + 1))
            if self.early_stop is not None:
                if cum_mse < best_mse:
                    weights = deepcopy(self.state_dict())
                    best_mse = cum_mse
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.early_stop:
                        print('Early stopping!')
                        break
                print('Patience:', patience)
        if self.early_stop is not None:
            self.load_state_dict(weights)
        self.eval()
        return self
    
    def predict_code(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if X.ndim == 3:
                X = X[:, None, ...]
            X_t = self.np2torch(X, device='cpu')
            code_list = []
            data_loader = DataLoader(TensorDataset(X_t), self.batch_num, False)
            for (X_b, ) in data_loader:
                code_list.append(self.encoder_(X_b.to(self.device)).to('cpu'))
            code = torch.concat(code_list)
            code = code.cpu().numpy().astype(np.double)
            return code
    
    
class BottleNeck(torch.nn.Module):
    class ClsNN(torch.nn.Module):
        def __init__(self, feat_in: int, cls_num: int, device: torch.device) -> None:
            super().__init__()
            self.cls_num = cls_num
            self.nn = torch.nn.Sequential(torch.nn.Linear(feat_in, feat_in // 2), 
                                          torch.nn.Tanh(),
                                          torch.nn.Linear(feat_in // 2, cls_num if cls_num > 2 else 1)).to(device)
            
        def forward(self, X):
            pred = self.nn(X)
            if self.cls_num == 2:
                logits = torch.concat((-pred, pred), dim=-1)
            else:
                logits = pred
            return logits
        
    # def _get_emb_nn(self):
    #     return torch.nn.Sequential(
    #         torch.nn.Conv2d(1, 16, (6, 6), stride=2),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Conv2d(16, 32, (6, 6), stride=2),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Conv2d(32, 48, (8, 8), stride=1),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Flatten(),
    #     ).to(self.device)
        
    # for 28x28
    # def _get_emb_nn(self):
    #     return torch.nn.Sequential(
    #         torch.nn.Conv2d(1, 16, (3, 3), stride=2),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Conv2d(16, 32, (3, 3), stride=2),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Conv2d(32, 32, (3, 3), stride=1),
    #         torch.nn.LeakyReLU(),
    #         torch.nn.Flatten(),
    #     ).to(self.device)
    
    # for celeba
    def _get_emb_nn(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, (32, 24), stride=3),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 32, (16, 16), stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, (16, 16), stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
        ).to(self.device)
    
    def __init__(self, lmd: float, epochs_num: int, batch_num: int, l_r: float, device: torch.device, early_stop: Optional[int]):
        super().__init__()
        self.lmd = lmd
        self.epochs_num = epochs_num
        self.batch_num = batch_num
        self.l_r = l_r
        self.device = torch.device(device)
        self.early_stop = early_stop
        
    def np2torch(self, arr, dtype=None, device=None):
        if dtype is None:
            dtype = torch.float32
        if device is None:
            device = self.device
        return torch.tensor(arr, dtype=dtype, device=device)
        
    def fit(self, X: np.ndarray, y: np.ndarray, c: np.ndarray):
        assert y.min() == 0
        assert np.all(c.min(axis=0) == 0)
        y_cls_num = y.max() + 1
        c_cls_num = np.max(c, axis=0) + 1
        self.embed_nn = self._get_emb_nn()
        emb_dim = self.embed_nn(self.np2torch(X[0, None, ...])).shape[1]
        self.con_nn_list = []
        for i, v in enumerate(c_cls_num):
            con_nn = self.ClsNN(emb_dim, v, self.device)
            self.register_module(f'con_nn_{i}', con_nn)
            self.con_nn_list.append(con_nn)
        self.y_nn = self.ClsNN(np.sum(c_cls_num), y_cls_num, self.device)
        
        self.optimizer_ = torch.optim.AdamW(self.parameters(), self.l_r, weight_decay=1e-5)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        if X.ndim == 3:
            X = X[:, None, ...]
        X = self.np2torch(X, device='cpu')
        Y = self.np2torch(y, torch.long)
        C = self.np2torch(c, torch.long)
        
        dataset = TensorDataset(X, Y, C)
        self.train()
        weights = deepcopy(self.state_dict())
        best_ce = float('inf')
        patience = 0
        for e in range(self.epochs_num):
            data_loader = DataLoader(dataset, self.batch_num, True)
            prog_bar = tqdm(
                data_loader, f'Epoch {e}', unit='batch', ascii=True)
            loss_kw = defaultdict(float)
            for i, (x_b, y_b, c_b) in enumerate(prog_bar):
                x_b = x_b.to(self.device)
                self.optimizer_.zero_grad()
                embedding = self.embed_nn(x_b)
                loss = 0
                proba_list = []
                for j in range(c_cls_num.shape[0]):
                    cur_con_logits = self.con_nn_list[j](embedding)
                    cur_con_loss = loss_fn(cur_con_logits, c_b[:, j])
                    loss += cur_con_loss
                    proba_list.append(torch.nn.functional.softmax(cur_con_logits, dim=-1))
                    loss_kw[f'CE conc {j}'] += cur_con_loss.item()
                loss /= c_cls_num.shape[0]
                conc_proba = torch.concat(proba_list, dim=1)
                y_proba = self.y_nn(conc_proba)
                y_loss = loss_fn(y_proba, y_b)
                loss += self.lmd * y_loss
                loss.backward()
                self.optimizer_.step()
                loss_kw[f'CE target'] += y_loss.item()
                ce_loss_list = [v / (i + 1) for v in loss_kw.values()]
                prog_bar.set_postfix(dict(zip(loss_kw.keys(), ce_loss_list)))
            mean_ce = np.mean(ce_loss_list)
            if self.early_stop is not None:
                if mean_ce < best_ce:
                    weights = deepcopy(self.state_dict())
                    best_ce = mean_ce
                    patience = 0
                else:
                    patience += 1
                    if patience >= self.early_stop:
                        print('Early stopping!')
                        break
                print('Patience:', patience)
        if self.early_stop is not None:
            self.load_state_dict(weights)
        self.eval()
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if X.ndim == 3:
                X = X[:, None, ...]
            X_t = self.np2torch(X, device='cpu')
            embed_list = []
            data_loader = DataLoader(TensorDataset(X_t), self.batch_num, False)
            for (X_b, ) in data_loader:
                embed_list.append(self.embed_nn(X_b.to(self.device)))
            embedding = torch.concat(embed_list)
            c_proba_list = []
            labels_list = [None]
            for c_nn in self.con_nn_list:
                cur_c_logits = c_nn(embedding)
                cur_proba = torch.softmax(cur_c_logits, 1)
                c_proba_list.append(cur_proba)
                cur_label = torch.argmax(cur_proba, dim=1, keepdim=True)
                labels_list.append(cur_label)
            c_proba = torch.concat(c_proba_list, dim=1)
            y_logits = self.y_nn(c_proba)
            y_label = torch.argmax(y_logits, dim=1, keepdim=True)
            labels_list[0] = y_label
            labels = torch.concat(labels_list, dim=1)
            return labels.cpu().numpy()
        
class EasyClustering():
    def __init__(self, cls_num: int, autoencoder):
        self.autoencoder = autoencoder
        # self.clusterizer = GaussianMixture(cls_num, covariance_type='full', tol=0.01)
        self.clusterizer = KMeans(cls_num)
        
    def fit(self, X: np.ndarray) -> 'EasyClustering':
        self.autoencoder.fit(X)
        latent_x = self.autoencoder.predict_code(X)
        self.clusterizer.fit(latent_x)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        latent_x = self.autoencoder.predict_code(X)
        return self.clusterizer.predict(latent_x)

def vert_patcher(X: np.ndarray) -> np.ndarray:
    half_h = X.shape[-2] // 2
    result = np.zeros((X.shape[0], 2, X.shape[1], half_h, X.shape[-1]), dtype=X.dtype)
    result[:, 0, ...] = X[..., :half_h, :]
    result[:, 1, ...] = X[..., half_h:, :]
    return result

def quarter_patcher(X: np.ndarray) -> np.ndarray:
    half_h = X.shape[-2] // 2
    half_w = X.shape[-1] // 2
    result = np.zeros((X.shape[0], 4, X.shape[1], half_h, half_w), dtype=X.dtype)
    result[:, 0, ...] = X[..., :half_h, :half_w]
    result[:, 1, ...] = X[..., :half_h, half_w:]
    result[:, 2, ...] = X[..., half_h:, :half_w]
    result[:, 3, ...] = X[..., half_h:, half_w:]
    return result

def window_patcher(X, kernel_size, stride):
    X_tensor = torch.tensor(X, device='cpu')
    patches = torch.nn.Unfold(kernel_size, stride=stride)(X_tensor).cpu().numpy()
    patches = patches.reshape((X.shape[0], X.shape[1], -1, patches.shape[-1]))
    patches = patches.transpose((0, 3, 1, 2))
    patches = patches.reshape(patches.shape[:3] + kernel_size)
    return patches
