import numpy as np
from sympy import Symbol, Eq, And, Or, Ne, Number
from sympy.logic.boolalg import to_cnf

def expand_eq_ne(eq_ne, concepts):
    assert isinstance(eq_ne, Eq) or isinstance(eq_ne, Ne)
    if isinstance(eq_ne, Ne):
        if isinstance(eq_ne.args[0], Symbol):
            symb, value = eq_ne.args
        elif isinstance(eq_ne.args[1], Symbol):
            value, symb = eq_ne.args
        else:
            raise ValueError(f'No symbol found in Neq: {eq_ne.args}')
        assert isinstance(value, Number)
        value = int(value)
        return Or(*[Eq(symb, other) for other in range(concepts[str(symb)]) if other != value])
    return eq_ne

def expand_ne_in_cnf(cnf, concepts):
    """Expand all not-equals in CNF to obtain rules on all outcomes.

    For example, given a concept `x` with values `{0, 1, 2}`,
    the rule `x != 1` will be expanded into `x = 0 or x = 2`.

    """
    if isinstance(cnf, And):
        to_consider = cnf.args
    elif isinstance(cnf, Or):  # there is only one disjuction part in CNF
        to_consider = [cnf]
    else:
        raise ValueError(f'Wrong {cnf = !r}')

    parts = []
    for p in to_consider:
        if isinstance(p, Or):
            parts.append(Or(*[expand_eq_ne(eq_ne, concepts) for eq_ne in p.args]))
        elif isinstance(p, Eq) or isinstance(p, Ne):
            parts.append(expand_eq_ne(p, concepts))
        else:
            raise ValueError(f'Element of CNF is neither disjunction (Or), nor Eq or Ne: {p!r}')
    return And(*parts)


def cnf_to_constraints(cnf, concepts, concept_proba_index_lookup=None):
    """Generate left-hand side of constraints system: `A x >= b`.
    The right-hand side is a unit vector `(1, ..., 1)`.

    """
    if concept_proba_index_lookup is None:
        concept_proba_index_lookup = dict()
        for name, size in concepts.items():
            for i in range(size):
                concept_proba_index_lookup[(name, i)] = len(concept_proba_index_lookup)
    cnf = expand_ne_in_cnf(cnf, concepts)
    # cnf = part_1 & ... & part_n
    # part_i = h_i1 | ... | h_ik
    if isinstance(cnf, And):
        parts = cnf.args
    elif isinstance(cnf, Or):
        parts = [cnf]
    else:
        raise ValueError(f'Wrong {cnf = !r}')

    A = np.zeros((len(parts), len(concept_proba_index_lookup)))
    for i, part in enumerate(parts):
        if isinstance(part, Or):
            for eq in part.args:
                assert isinstance(eq, Eq)
                symb, value = eq.args
                A[i, concept_proba_index_lookup[(str(symb), int(value))]] = 1
        elif isinstance(eq, Eq):
            symb, value = eq.args
            A[i, concept_proba_index_lookup[(str(symb), int(value))]] = 1
        else:
            raise ValueError(f'Wrong part of CNF: {part}')
    return A

def make_all_constraints(concepts, rules):
    general_rule = And(*rules)
    cnf = to_cnf(general_rule)
    expanded_cnf = expand_ne_in_cnf(cnf, concepts)
    cnf_constraints_A = cnf_to_constraints(expanded_cnf, concepts)
    cnf_constraints_b = np.ones_like(cnf_constraints_A[:, 0])
    # NOTE: A x >= b
    n_total = cnf_constraints_A.shape[1]  # total number of outcomes
    all_ineq_A = np.concatenate((cnf_constraints_A, np.eye(n_total, n_total)), axis=0)
    all_ineq_b = np.concatenate((cnf_constraints_b, np.zeros((n_total,))), axis=0)

    # proba_distr_constraints_A = make_proba_distribution_constraints(concepts)
    # proba_distr_constraints_b = np.ones_like(proba_distr_constraints_A[:, 0])
    # A x == b

    return {
        'ineq': (all_ineq_A, all_ineq_b),  # A x >= b
        # 'eq': (proba_distr_constraints_A, proba_distr_constraints_b)  # A x == b
    }

class RuleChecker:
    def __init__(self, concepts, rules, eps: float = 1.e-5) -> np.ndarray:
        self.concepts = concepts
        self.rules = rules
        self.constraints = make_all_constraints(concepts, rules)
        self.eps = eps

    def check(self, flat_probas) -> np.ndarray:
        """Check that the given set of marginal probability distributions satisfies the rules.
        """
        A, b = self.constraints['ineq']  # A x >= b
        # flat_probas shape: (n_samples, n_values)
        error_mtx = np.all(A @ flat_probas.T >= b[:, np.newaxis] - self.eps, axis=0)
        return np.count_nonzero(error_mtx == False)
