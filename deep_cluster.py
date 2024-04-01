from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
import torch
import numpy as np
import itertools
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

class DeepCluster(torch.nn.Module):
    def _get_classifier(self, dim_in: int, n_cls: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, 8),
            torch.nn.Tanh(),
            torch.nn.LayerNorm(8),
            torch.nn.Linear(8, 8),
            torch.nn.Tanh(),
            # torch.nn.Linear(8, 8),
            # torch.nn.Tanh(),
            # torch.nn.Linear(8, 8),
            # torch.nn.Tanh(),
            torch.nn.LayerNorm(8),
            torch.nn.Linear(8, n_cls),
            torch.nn.Softmax(dim=-1)
        ).to(self.device)
        
    def _get_repr_nn(self, dim_in: int, dim_out: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, 8),
            torch.nn.Tanh(),
            torch.nn.LayerNorm(8),
            torch.nn.Linear(8, 8),
            torch.nn.Tanh(),
            # torch.nn.Linear(8, 16),
            # torch.nn.Tanh(),
            torch.nn.LayerNorm(8),
            # torch.nn.Linear(8, 8),
            # torch.nn.Tanh(),
            torch.nn.Linear(8, dim_out),
        ).to(self.device)
        
    def _get_decoder(self, dim_in: int, dim_out: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, 8),
            torch.nn.ReLU(),
            # torch.nn.LayerNorm(8),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            # torch.nn.Linear(8, 16),
            # torch.nn.Tanh(),
            # torch.nn.LayerNorm(8),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, dim_out),
        ).to(self.device)
    
    def __init__(self, n_clusters: int, latent_dim: int,
                 batch_len: int = 512, epochs_num: int = 200,
                 l_r: float = 1e-3, device: str = 'cpu') -> None:
        super().__init__()
        self.device = torch.device(device)
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.classifier = self._get_classifier(latent_dim, n_clusters)
        self.clusterizer = None
        self.repr_nn = None
        self.epochs_num = epochs_num
        self.batch_num = batch_len
        self.l_r = l_r
        
    def _lazy_init(self, input_dim: int):
        # self.repr_nn = self._get_repr_nn(input_dim, self.latent_dim)
        self.optimizer = torch.optim.Adam(
            itertools.chain(
                self.classifier.parameters(),
                self.repr_nn.parameters()
            ), lr=self.l_r)
        
    def loss_fn(self, logits, target_idx) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(logits, target_idx)
        # return torch.nn.functional.nll_loss(logits, target_idx)
    
    def np2torch(self, arr: np.ndarray, dtype: torch.dtype | None = None):
        if dtype is None:
            dtype = torch.get_default_dtype()
        return torch.tensor(arr, dtype=dtype, device=self.device)
    
    def pre_fit(self, x: np.ndarray) -> 'DeepCluster':
        self.repr_nn = self._get_repr_nn(2, self.latent_dim)
        optimizer = torch.optim.Adam(self.repr_nn.parameters(), lr=self.l_r)
        decoder = self._get_decoder(self.latent_dim, 2)
        X = self.np2torch(x)

        dataset = TensorDataset(X)
        self.train()
        # val_kw = dict()
        for e in range(self.epochs_num):
            data_loader = DataLoader(dataset, self.batch_num, True)
            prog_bar = tqdm(
                data_loader, f'Epoch {e}', unit='batch', ascii=True)
            cum_loss = 0
            for i, (x_b, ) in enumerate(prog_bar):
                optimizer.zero_grad()
                
                z = self.repr_nn(x_b)
                x_z = decoder(z)
                loss = torch.nn.functional.mse_loss(x_b, x_z)
                
                loss.backward()
                optimizer.step()
                cum_loss += loss.item()

                prog_bar.set_postfix(MSE=cum_loss / (i + 1))
        # with torch.no_grad():
        #     z_all = self.repr_nn(X)
        #     self.clusterizer = KMeans(self.n_clusters, n_init='auto', copy_x=False).fit(z_all.cpu().numpy())
        self.eval()
        return self
    
    def fit(self, x: np.ndarray) -> 'DeepCluster':
        self._lazy_init(x.shape[-1])
        
        X = self.np2torch(x)

        dataset = TensorDataset(X)
        self.train()
        k_means = None
        for e in range(self.epochs_num):
            data_loader = DataLoader(dataset, self.batch_num, True)
            prog_bar = tqdm(
                data_loader, f'Epoch {e}', unit='batch', ascii=True)
            cum_loss = 0
            for i, (x_b, ) in enumerate(prog_bar):
                self.optimizer.zero_grad()
                
                z = self.repr_nn(x_b)
                z_np = z.detach().cpu().numpy()
                initialization = 'random' if k_means is None else k_means.cluster_centers_
                k_means = KMeans(self.n_clusters, n_init=1, copy_x=False, init=initialization)
                y_np = k_means.fit_predict(z_np)
                y = self.np2torch(y_np, dtype=torch.long)
                cls_logits = self.classifier(z)
                # cls_logits = torch.cat((-cls_logits, cls_logits), dim=-1)
                loss = self.loss_fn(cls_logits, y)
                
                loss.backward()
                self.optimizer.step()
                cum_loss += loss.item()

                prog_bar.set_postfix(CrossEntropy=cum_loss / (i + 1))
        with torch.no_grad():
            z_all = self.repr_nn(X)
            initialization = 'random' if k_means is None else k_means.cluster_centers_
            k_means = KMeans(self.n_clusters, n_init=1, copy_x=False, init=initialization)
            self.clusterizer = k_means.fit(z_all.cpu().numpy())
        self.eval()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X = self.np2torch(x)
            z = self.repr_nn(X)
            y = self.clusterizer.predict(z.cpu().numpy())
            return y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    n = 1000
    noise = 0.05
    x, y = make_moons(n, noise=noise)
    model = DeepCluster(2, 2, device='cuda')
    model.pre_fit(x)
    model.fit(x)
    y_pred = model.predict(x)
    
    colors = ['r', 'b']
    fig, ax = plt.subplots()
    ax.scatter(*x[y == 0].T, c=colors[0])
    ax.scatter(*x[y == 1].T, c=colors[1])
    fig.suptitle('Train dataset')
    
    fig, ax = plt.subplots()
    ax.scatter(*x[y_pred == 0].T, c=colors[0])
    ax.scatter(*x[y_pred == 1].T, c=colors[1])
    fig.suptitle('Clusterization result')
    
    with torch.no_grad():
        fig, ax = plt.subplots()
        z = model.repr_nn(model.np2torch(x))
        z = z.cpu().numpy()
        ax.scatter(*z[y_pred == 0].T, c=colors[0])
        ax.scatter(*z[y_pred == 1].T, c=colors[1])
        fig.suptitle('Latent space')
    
    plt.show()
    