import numpy as np
import sympy as sp
from sympy import Expr
from scipy.special import softmax, log_softmax
from typing import Optional, List
import warnings

class FICBL():
    # patcher is mapping (n, ...) -> (n, p, ...)
    def __init__(self, cls_num: int, clustering_model, patcher, epsilon: float = 0.01):
        super().__init__()
        self.cls_num = cls_num
        self.cls_model = clustering_model
        self.patcher = patcher
        self.eps = np.log(epsilon)
        self.is_fit = False
        
    def get_model_with_rules(self, rules: List[Expr]) -> 'FICBL':
        assert self.is_fit
        master_rule = sp.And(*rules)
        uq, idx = np.unique(self.c, axis=0, return_inverse=True)
        uq_mask = np.zeros(uq.shape[0], dtype=bool)
        vars = list(master_rule.free_symbols)
        default_dict = dict([(str(v), False) for v in vars])
        for i, c in enumerate(uq):
            cur_dict = default_dict.copy()
            for r in range(c.shape[0]):
                v = c[r]
                cur_dict[f'x_{r}_{v}'] = True
            rule_val = master_rule.subs(cur_dict)
            uq_mask[i] = rule_val
        master_mask = uq_mask[idx]
        filtered_cls_lbl = self.cls_labels[master_mask]
        filtered_c = self.c[master_mask]
        new_model = FICBL(self.cls_num, self.cls_model, self.patcher, 1)
        new_model.eps = self.eps
        new_model._count_statistics(filtered_c, filtered_cls_lbl)
        new_model.c = filtered_c
        new_model.cls_labels = filtered_cls_lbl
        new_model.patches_n = self.patches_n
        new_model.is_fit = True
        return new_model
        
    def _count_statistics(self, concepts: np.ndarray, cluster_labels: np.ndarray):
        assert concepts.ndim == cluster_labels.ndim == 2
        patches_n = cluster_labels.shape[-1]
        # C_v is proportion of images with the cerain concepts to all images
        self.C_v = np.ndarray(concepts.shape[1], dtype=object) # Pr{C^r = v}, list is from v to r (table 2)
        # self.U = np.ndarray(concepts.shape[1], dtype=object) # coeffs U^r_v, from v to r
        for i in range(concepts.shape[1]):
            values, c_counts = np.unique(concepts[:, i], axis=0, return_counts=True)
            prop = np.zeros(values.max() + 1)
            for j, v in enumerate(values):
                prop[v] = c_counts[j] / concepts.shape[0]
            self.C_v[i] = np.log(prop)
        self.p_irv = [] # vector p in multinomial distribution (table 3)
        self.v = np.max(concepts, axis=0) + 1 # events for each concept
        for i in range(self.cls_num):
            cluster_mask = cluster_labels == i
            self.p_irv.append([])
            for r in range(concepts.shape[1]):
                self.p_irv[-1].append([])
                for v in range(self.v[r]):
                    cur_mask = concepts[:, r] == v
                    denom = patches_n * np.count_nonzero(cur_mask)
                    c_r_v_mask = np.tile(cur_mask[:, None], (1, patches_n))
                    if denom == 0:
                        stat = 0
                    else:
                        stat = np.count_nonzero(np.logical_and(c_r_v_mask, cluster_mask)) / denom
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.p_irv[-1][-1].append(np.log(stat))
    
    # X - images, y - target, c - concepts
    # if only y is provided, then y[0] is target, other components are concepts
    def fit(self, X: np.ndarray, y: np.ndarray, c: Optional[np.ndarray]=None):
        assert c is None or c.ndim == 2
        self.cls_model.fit(X, self.patcher)
        if y.ndim == 1:
            y = y[:, None]
        if c is not None:
            C = np.concatenate((y, c), axis=-1)
        else:
            C = y
        self.c = C
        self.patches_n = self.patcher(X[None, 0]).shape[1]
        self.cls_labels = self.cls_model.predict(X, self.patcher).reshape((X.shape[0], self.patches_n))
        self._count_statistics(self.c, self.cls_labels)
        self.is_fit = True
        return self
                    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        cls_pred = self.cls_model.predict(x, self.patcher).reshape((x.shape[0], self.patches_n))
        s = np.zeros((x.shape[0], self.cls_num), dtype=int)
        for i in range(self.cls_num):
            s[:, i] = np.count_nonzero(cls_pred == i, axis=1)
        res_shape = np.sum(self.v)
        result = np.zeros((x.shape[0], res_shape))
        written = 0
        for j in range(self.c.shape[-1]):
            cur_conc_res = np.empty((result.shape[0], self.v[j]))
            for v in range(self.v[j]):
                p_irv_cur = np.zeros(self.cls_num) # (r)
                for k in range(self.cls_num):
                    p_irv_cur[k] = max(self.p_irv[k][j][v], self.eps)
                p_irv_cur = log_softmax(p_irv_cur)
                C_v = self.C_v[j][v]
                P_s_p = np.sum(s * p_irv_cur[None, :], axis=1)
                cur_conc_res[:, v] = P_s_p + C_v
                written += 1
            result[:, written - v - 1: written] = softmax(cur_conc_res, axis=1)
        return result
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        res_shape = self.c.shape[-1]
        result = np.zeros((x.shape[0], res_shape), dtype=np.intp)
        proba = self.predict_proba(x)
        cumul_sum = 0
        for i, outcomes in enumerate(self.v):
            cur_proba = proba[:, cumul_sum : cumul_sum + outcomes]
            result[:, i] = cur_proba.argmax(axis=1)
            cumul_sum += outcomes
        return result
    
    def predict_tgt_lbl_conc_proba(self, x: np.ndarray) -> np.ndarray:
        cls_pred = self.cls_model.predict(x, self.patcher).reshape((x.shape[0], self.patches_n))
        s = np.zeros((x.shape[0], self.cls_num), dtype=int)
        for i in range(self.cls_num):
            s[:, i] = np.count_nonzero(cls_pred == i, axis=1)
        res_shape = np.sum(self.v[1:])
        conc_result = np.zeros((x.shape[0], res_shape))
        written = 0
        for j in range(1, self.c.shape[-1]):
            cur_conc_res = np.empty((x.shape[0], self.v[j]))
            for v in range(self.v[j]):
                p_irv_cur = np.zeros(self.cls_num) # (r)
                for k in range(self.cls_num):
                    p_irv_cur[k] = max(self.p_irv[k][j][v], self.eps)
                p_irv_cur = log_softmax(p_irv_cur)
                C_v = self.C_v[j][v]
                P_s_p = np.sum(s * p_irv_cur[None, :], axis=1)
                cur_conc_res[:, v] = P_s_p + C_v
                written += 1
            conc_result[:, written - v - 1: written] = softmax(cur_conc_res, axis=1)
        lbl_res = np.zeros(x.shape[0], dtype=int)
        max_proba = float('-inf') * np.ones(x.shape[0])
        for v in range(self.v[0]):
            p_irv_cur = np.zeros(self.cls_num) # (r)
            for k in range(self.cls_num):
                p_irv_cur[k] = max(self.p_irv[k][0][v], self.eps)
            p_irv_cur = log_softmax(p_irv_cur)
            C_v = self.C_v[0][v]
            P_s_p = np.sum(s * p_irv_cur[None, :], axis=1)
            cur_sub_res = P_s_p + C_v
            mask = cur_sub_res > max_proba
            max_proba[mask] = cur_sub_res[mask]
            lbl_res[mask] = v
        return lbl_res, conc_result

if __name__=='__main__':
    x_train = np.asarray(
        [
            [
                [0, 1], 
                [0, 0]
            ],
            [
                [0, 0], 
                [0, 1]
            ],
            [
                [2, 0], 
                [0, 0]
            ],
            [
                [0, 0], 
                [2, 0]
            ],
            [
                [0, 3], 
                [1, 0]
            ],
            [
                [0, 0], 
                [3, 0]
            ],
            [
                [3, 0], 
                [0, 3]
            ],
            [
                [0, 3], 
                [0, 0]
            ],
            [
                [4, 0], 
                [0, 0]
            ],
            [
                [0, 4], 
                [0, 0]
            ],
        ], dtype=np.double
    )
    y_train = np.asarray([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    c_train = np.asarray([
        [0, 0],
        [0, 0],
        [1, 0],
        [1, 0],
        [2, 0],
        [2, 0],
        [2, 0],
        [2, 0],
        [1, 1],
        [1, 1]
    ])
    x_test = np.asarray(
        [
            [
                [0, 0],
                [1, 2]
            ],
        ]
    )
    
    class TestClusterizer():
        def fit(self, X, patcher):
            pass
        
        def predict(self, X, patcher):
            x = patcher(X).ravel()
            result = np.zeros(x.shape[0], dtype=int)
            result[x == 0] = 0
            result[np.logical_or(x == 3, x == 4)] = 1
            result[np.logical_or(x == 1, x == 2)] = 2
            return result
            
    patch_lambda = lambda X: np.reshape(X, (-1, 4, 1))
    cls_model = TestClusterizer()
    concept_model = FICBL(3, cls_model, patch_lambda, 0.01).fit(x_train, y_train, c_train)
    test_predict = concept_model.predict_proba(x_test)
    print('Test concepts:', test_predict)
    c0_l, c_all_p = concept_model.predict_tgt_lbl_conc_proba(x_test)
    print('C0 label:', c0_l)
    print('concepts proba:', c_all_p)
    x_1_1, x_0_0 = sp.symbols('x_1_1, x_0_0')
    rule = x_1_1 >> x_0_0
    concept_model = concept_model.get_model_with_rules([rule])
    print('Inserted the rule')
    test_predict = concept_model.predict_proba(x_test)
    print('Test concepts:', test_predict)
    print('Test labels:', concept_model.predict(x_test))
    