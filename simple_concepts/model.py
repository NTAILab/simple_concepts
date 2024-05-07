import numpy as np
import sympy as sp
from sympy.logic.boolalg import truth_table
import numba
from typing import Optional

@numba.njit
def multinomial_coef(arr):
    res, i = 1, 1
    for a in arr:
        for j in range(1, a + 1):
            res *= i
            res //= j
            i += 1
    return res


class SimpleConcepts():
    # patcher is mapping (n, ...) -> (n, p, ...)
    def __init__(self, cls_num: int, clustering_model, patcher, epsilon: float = 0.01):
        super().__init__()
        self.cls_num = cls_num
        self.cls_model = clustering_model
        self.patcher = patcher
        self.eps = epsilon
        self.is_fit = False
        
    def insert_rules(self, rule):
        assert self.is_fit
        self.U = []
        for v_n in self.v:
            self.U.append(np.zeros(v_n))
        vars = list(rule.free_symbols)
        default_dict = dict([(str(v), False) for v in vars])
        # p_all = 0
        C = self.c[:, 0, :]
        Z, Pr_Z = np.unique(C, return_counts=True, axis=0)
        for cur_z, im_num in zip(Z, Pr_Z):
            cur_dict = default_dict.copy()
            for r in range(cur_z.shape[0]):
                v = cur_z[r]
                cur_dict[f'x_{r}_{v}'] = True
            rule_val = rule.subs(cur_dict)
            if not rule_val:
                continue
            # p_all += im_num
            # need to be rethinked for multiple rules
            for var, val in cur_dict.items():
                if not val:
                    continue
                r, v = var.split('_')[-2:]
                self.U[int(r)][int(v)] += im_num
        # for i in range(len(self.v)):
        #     self.U[i] = self.U[i] / p_all
        
    # def _count_statistics(self, concepts: )

    
    # X - images, y - target, c - concepts
    # if only y is provided, then y[0] is target, other components are concepts
    def fit(self, X: np.ndarray, y: np.ndarray, c: Optional[np.ndarray]=None):
        assert c is None or c.ndim == 2
        patches = self.patcher(X)
        p_ravel = np.reshape(patches, (-1, ) + patches.shape[2:])
        self.cls_model.fit(p_ravel)
        if y.ndim == 1:
            y = y[:, None]
        if c is not None:
            C = np.concatenate((y, c), axis=-1)
        else:
            C = y
        self.c = np.tile(C[:, None, :], (1, patches.shape[1], 1)) # size as patches
        self.p_y = self.cls_model.predict(p_ravel).reshape((patches.shape[0], patches.shape[1]))
        # C_v is proportion of images with the cerain concepts to all images
        self.C_v = [] # Pr{C^r = v}, list is from v to r (table 2)
        self.U = np.ndarray(C.shape[1], dtype=object) # coeffs U^r_v, from v to r
        for i in range(C.shape[1]):
            values, c_counts = np.unique(C[:, i], axis=0, return_counts=True)
            prop = np.zeros(np.max(values) + 1)
            cur_u = np.ones(np.max(values) + 1)
            for j, v in enumerate(values):
                prop[v] = c_counts[j] / C.shape[0]
            self.C_v.append(prop)
            self.U[i] = cur_u
        self.p_irv = [] # vector p in multinomial distribution (table 3)
        self.s_i = np.zeros(self.cls_num) # number of patches in each class
        self.v = [] # events for each concept
        for i in range(self.cls_num):
            cluster_mask = self.p_y == i
            self.s_i[i] = np.count_nonzero(cluster_mask)
            self.p_irv.append([])
            for r in range(C.shape[1]):
                self.p_irv[-1].append([])
                v_var = np.max(C[:, r], axis=0) + 1
                if i == 0:
                    self.v.append(v_var)
                for v in range(v_var):
                    c_r_v_mask = self.c[..., r] == v
                    stat = np.count_nonzero(np.logical_and(c_r_v_mask, cluster_mask)) / np.count_nonzero(c_r_v_mask)
                    self.p_irv[-1][-1].append(stat)
        self.is_fit = True
        return self
                    
        
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        patches = self.patcher(x)
        p_ravel = np.reshape(patches, (-1, ) + patches.shape[2:])
        cls_pred = self.cls_model.predict(p_ravel).reshape((patches.shape[0], patches.shape[1]))
        s = np.zeros((patches.shape[0], self.cls_num), dtype=int)
        for i in range(self.cls_num):
            s[:, i] = np.count_nonzero(cls_pred == i, axis=1)
        res_shape = np.sum(self.v)
        result = np.zeros((x.shape[0], res_shape))
        for i in range(x.shape[0]):
            s_cur = s[i] # (p_n)
            # mnl_coef = multinomial_coef(s_cur)
            written = 0
            for j in range(self.c.shape[-1]):
                norm_sum = 0
                cur_C_v = self.U[j] * self.C_v[j]
                cur_C_v /= np.sum(cur_C_v)
                for v in range(self.v[j]):
                    p_irv_cur = np.zeros(self.cls_num) # (r)
                    for k in range(self.cls_num):
                        p_irv_cur[k] = max(self.p_irv[k][j][v], self.eps)
                    p_irv_cur = p_irv_cur / np.sum(p_irv_cur)
                    P_s_p = np.prod(np.power(p_irv_cur, s_cur))
                    # C_v = self.U[j][v] * self.C_v[j][v]
                    C_v = cur_C_v[v]
                    result[i, written] = P_s_p * C_v
                    norm_sum += result[i, written]
                    written += 1
                result[i, written - v - 1: written] /= norm_sum
        return result
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        patches = self.patcher(x)
        p_ravel = np.reshape(patches, (-1, ) + patches.shape[2:])
        cls_pred = self.cls_model.predict(p_ravel).reshape((patches.shape[0], patches.shape[1]))
        s = np.zeros((patches.shape[0], self.cls_num), dtype=int)
        for i in range(self.cls_num):
            s[:, i] = np.count_nonzero(cls_pred == i, axis=1)
        res_shape = self.c.shape[-1]
        result = np.zeros((x.shape[0], res_shape), dtype=np.int0)
        for i in range(x.shape[0]):
            s_cur = s[i] # (p_n)
            for j in range(self.c.shape[-1]):
                max_res = 0
                res_label = 0
                cur_C_v = self.U[j] * self.C_v[j]
                cur_C_v /= np.sum(cur_C_v)
                for v in range(self.v[j]):
                    p_irv_cur = np.zeros(self.cls_num) # (r)
                    for k in range(self.cls_num):
                        p_irv_cur[k] = max(self.p_irv[k][j][v], self.eps)
                    p_irv_cur = p_irv_cur / np.sum(p_irv_cur)
                    # P_s_p = multinomial_coef(s_cur) * np.prod(np.power(p_irv_cur, s_cur))
                    P_s_p = np.prod(np.power(p_irv_cur, s_cur))
                    # C_v = self.U[j][v] * self.C_v[j][v]
                    C_v = cur_C_v[v]
                    cur_res = P_s_p * C_v
                    if cur_res > max_res:
                        max_res = cur_res
                        res_label = v
                result[i, j] = res_label
        return result


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
        def fit(self, X):
            pass
        
        def predict(self, X):
            assert X.shape[-1] == 1 or X.ndim == 1
            result = np.zeros(X.shape[0], dtype=int)
            x = X.ravel()
            result[x == 0] = 0
            result[np.logical_or(x == 3, x == 4)] = 1
            result[np.logical_or(x == 1, x == 2)] = 2
            return result
            
    patch_lambda = lambda X: np.reshape(X, (-1, 4, 1))
    cls_model = TestClusterizer()
    concept_model = SimpleConcepts(3, cls_model, patch_lambda, 0.01).fit(x_train, y_train, c_train)
    test_predict = concept_model.predict_proba(x_test)
    print('Test concepts:', test_predict)
    x_1_1, x_0_0 = sp.symbols('x_1_1, x_0_0')
    rule = x_1_1 >> x_0_0
    concept_model.insert_rules(rule)
    print('Inserted the rule')
    test_predict = concept_model.predict_proba(x_test)
    print('Test concepts:', test_predict)
    