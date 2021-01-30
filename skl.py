import numpy as np
import scipy.sparse as sparse

def index(sublattice, i, j, L):
    pos = L * i + j
    return (L**2) * sublattice + pos

def from_index(index, L):
    sublattice = index // (L**2)
    pos = index % (L**2)
    i = pos // L
    j = pos % L
    return sublattice, i, j

def interaction_matrix(L, J2, J3=None):
    """Returns the interaction matrix for a system with LxL unit cells (N = 6L²
    sites). It will have a shape NxN. Ji's are the coupling constants
    (with assumed J1=1). If J3 is not given, it is assumed to be equal
    to J2.
    """
    ncells = L**2
    N = 6 * ncells
    if J3 is None:
        J3 = J2
    J1 = 1
    ret = np.zeros((N, N))
    for i in range(L):
        for j in range(L):
            # sublattice 0
            ret[index(0, i, j, L), index(1, i, j, L)] = J1
            ret[index(0, i, j, L), index(3, i, j, L)] = J1
            ret[index(0, i, j, L), index(2, i, j, L)] = J2
            ret[index(0, i, j, L), index(5, i, (j+1)%L, L)] = J3
            # sublattice 1
            ret[index(1, i, j, L), index(0, i, j, L)] = J1
            ret[index(1, i, j, L), index(4, i, j, L)] = J1
            ret[index(1, i, j, L), index(5, i, (j+1)%L, L)] = J2
            ret[index(1, i, j, L), index(2, (i+1)%L, j, L)] = J3
            # sublattice 2
            ret[index(2, i, j, L), index(0, i, j, L)] = J2
            ret[index(2, i, j, L), index(4, (i-1)%L, j, L)] = J2
            ret[index(2, i, j, L), index(3, i, j, L)] = J3
            ret[index(2, i, j, L), index(1, (i-1)%L, j, L)] = J3
            # sublattice 3
            ret[index(3, i, j, L), index(0, i, j, L)] = J1
            ret[index(3, i, j, L), index(4, i, j, L)] = J1
            ret[index(3, i, j, L), index(5, i, j, L)] = J2
            ret[index(3, i, j, L), index(2, i, j, L)] = J3
            # sublattice 4
            ret[index(4, i, j, L), index(1, i, j, L)] = J1
            ret[index(4, i, j, L), index(3, i, j, L)] = J1
            ret[index(4, i, j, L), index(2, (i+1)%L, j, L)] = J2
            ret[index(4, i, j, L), index(5, i, j, L)] = J3
            # sublattice 5
            ret[index(5, i, j, L), index(3, i, j, L)] = J2
            ret[index(5, i, j, L), index(1, i, (j-1)%L, L)] = J2
            ret[index(5, i, j, L), index(4, i, j, L)] = J3
            ret[index(5, i, j, L), index(0, i, (j-1)%L, L)] = J3
    return ret
