from torchvision.datasets import MNIST
from torchvision.transforms.v2 import PILToTensor
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple
import torch
import numba
from utility import f1_sep_scorer, acc_sep_scorer
from experiment_models import Autoencoder, BottleNeck, vert_patcher, quarter_patcher, EasyClustering
from sklearn.model_selection import train_test_split
from simple_concepts.model import SimpleConcepts

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

def produce_addition_set(
    X,
    y,
    dataset_size=30000,
    num_operands=4,
    selected_digits=list(range(10)),
    output_channels=1,
    img_format='',
    sample_concepts=None,
    normalize_samples=False,
    concat_dim='',
    even_concepts=False,
    even_labels=False,
    threshold_labels=None,
    concept_transform=None,
    noise_level=0.0,
):
    filter_idxs = []
    if len(y.shape) == 2 and y.shape[-1] == 1:
        y = y[:, 0]
    if not isinstance(selected_digits[0], list):
        selected_digits = [selected_digits[:] for _ in range(num_operands)]
    elif len(selected_digits) != num_operands:
        raise ValueError(
            "If selected_digits is a list of lists, it must have the same "
            "length as num_operands"
        )

    operand_remaps = [
        dict((dig, idx) for (idx, dig) in enumerate(operand_digits))
        for operand_digits in selected_digits
    ]
    total_operand_samples = []
    total_operand_labels = []
    for allowed_digits in selected_digits:
        filter_idxs = []
        for idx, digit in enumerate(y):
            if digit in allowed_digits:
                filter_idxs.append(idx)
        total_operand_samples.append(X[filter_idxs, :, :, :])
        total_operand_labels.append(y[filter_idxs])

    sum_train_samples = []
    sum_train_concepts = []
    sum_train_labels = []
    for i in range(dataset_size):
        operands = []
        concepts = []
        sample_label = 0
        selected = []
        for operand_digits, remap, total_samples, total_labels in zip(
            selected_digits,
            operand_remaps,
            total_operand_samples,
            total_operand_labels,
        ):
            img_idx = np.random.choice(total_samples.shape[0])
            selected.append(total_labels[img_idx])
            img = total_samples[img_idx: img_idx + 1, :, :, :].copy()
            if len(operand_digits) > 2:
                if even_concepts:
                    concept_vals = np.array([[
                        int((remap[total_labels[img_idx]] % 2) == 0)
                    ]])
                else:
                    concept_vals = torch.nn.functional.one_hot(
                        torch.LongTensor([remap[total_labels[img_idx]]]),
                        num_classes=len(operand_digits)
                    ).numpy()
                concepts.append(concept_vals)
            else:
                # Else we will treat it as a simple binary concept (this allows
                # us to train models that do not have mutually exclusive
                # concepts!)
                if even_concepts:
                    concepts.append(np.array([[
                        int((total_labels[img_idx] % 2) == 0)
                    ]]))
                else:
                    max_bound = np.max(operand_digits)
                    val = int(total_labels[img_idx] == max_bound)
                    concepts.append(np.array([[val]]))
            sample_label += total_labels[img_idx]
            operands.append(img)
        if concat_dim == 'channels':
            sum_train_samples.append(np.concatenate(operands, axis=3))
        elif concat_dim == 'x':
            sum_train_samples.append(np.concatenate(operands, axis=2))
        else:
            sum_train_samples.append(np.concatenate(operands, axis=1))
        if even_labels:
            sum_train_labels.append(int(sample_label % 2 == 0))
        elif threshold_labels is not None:
            sum_train_labels.append(int(sample_label >= threshold_labels))
        else:
            sum_train_labels.append(sample_label)
        sum_train_concepts.append(np.concatenate(concepts, axis=-1))
    sum_train_samples = np.concatenate(sum_train_samples, axis=0)
    sum_train_concepts = np.concatenate(sum_train_concepts, axis=0)
    sum_train_labels = np.array(sum_train_labels)
    if output_channels != 1 and concat_dim != 'channels':
        sum_train_samples = np.stack(
            (sum_train_samples[:, :, :, 0].astype(np.float32),)*output_channels,
            axis=-1,
        )
    if img_format == 'channels_first':
        sum_train_samples = np.transpose(sum_train_samples, axes=[0, 3, 2, 1])
    if normalize_samples:
        sum_train_samples = sum_train_samples/255.0
    if sample_concepts is not None:
        sum_train_concepts = sum_train_concepts[:, sample_concepts]
    if concept_transform is not None:
        sum_train_concepts = concept_transform(sum_train_concepts)
    if noise_level > 0.0:
        sum_train_samples = sum_train_samples + np.random.normal(
            loc=0.0,
            scale=noise_level,
            size=sum_train_samples.shape,
        )
        if normalize_samples:
            sum_train_samples = np.clip(
                sum_train_samples,
                a_min=0.0,
                a_max=1.0,
            )
    return sum_train_samples, sum_train_labels, sum_train_concepts

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
        result_Y[i] = np.sum(cur_digits) % 10
    return result_X, result_Y, result_C


if __name__=='__main__':
    n = 50000
    cls_num = 10
    eps = 0.01
    ae_kw = {
        'latent_dim': 32, 
        'epochs_num': 10, 
        'batch_num': 128, 
        'l_r': 1e-3, 
        'device': 'cuda'
    }
    X, y, c = get_4mnist_ds(*get_proc_mnist_np(), n)
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, c, test_size=0.4)
    y_test = np.concatenate((y_test[:, None], c_test), axis=1)
    clusterizer = EasyClustering(cls_num, ae_kw)
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
