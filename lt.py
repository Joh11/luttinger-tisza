#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import collections as mc

# sublattices indices are labeled from left to right, top to bottom.
# the states are represented by a 6xN_cells matrix of complex numbers
# in memory, the layout with always be:
# 6 (sublattices) x Lx x Ly

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

def fourier_transform(int_mat, L, k):
    """Fourier transform the interaction matrix at the wave vector k. """
    # TODO do it the dumb way for now, but fft should be usable
    int_mat = int_mat.reshape(6, L**2, 6, L**2).transpose(0, 2, 1, 3) # 6,6,L²,L² shape

    R = np.array([[i // L, i % L] for i in range(L**2)])
    Rij = R.reshape(1, L**2, 2) - R.reshape(L**2, 1, 2)
    kRij = (k * Rij).sum(axis=-1) # L²,L² shape

    return (int_mat * np.exp(-1j * kRij)).sum(axis=(2, 3))

L = 10
N = 6 * L**2
int_mat = interaction_matrix(L, J2=2, J3=0.1)

def plot_band(int_mat, L):
    N = 6 * L**2
    nkps = 50
    Gamma = np.array([0, 0])
    M = np.array([np.pi, 0])
    X = np.array([np.pi, np.pi])
    
    kpath = np.concatenate([
        np.linspace(Gamma, M, nkps, False),
        np.linspace(M, X, nkps, False),
        np.linspace(X, Gamma, nkps),
    ])
    
    energies = np.zeros((len(kpath), 6))
    for i, k in enumerate(kpath):
        mat = fourier_transform(int_mat, L, k)
        energies[i] = np.linalg.eigh(mat)[0] / N

    plt.figure()
    for i in range(6):
        plt.scatter(np.arange(len(kpath)), energies[:, i], c='red', s=0.1)

    plt.xticks(nkps * np.arange(4), [r'$\Gamma$', 'M', 'X', r'$\Gamma$'])
    plt.ylabel(r'eigenergies (per site)')
    plt.show()
    return energies
energies = plot_band(int_mat, L)

# try checking the couplings
# positions = np.array([[2, 6],
#                       [6, 6],
#                       [0, 4],
#                       [2, 2],
#                       [6, 2],
#                       [4, 0]]) / 8

# def plot_int_mat(positions, int_mat, L, colors, skip_period=True):
#     plt.figure()
    
#     for i in range(L):
#         for j in range(L):
#             plt.scatter(positions[:, 0] + i, positions[:, 1] + j, c='k')

#     lines = []
#     lines_colors = []
#     for x in range(int_mat.shape[0]):
#         s1, i1, j1 = from_index(x, L)
#         for y in range(int_mat.shape[1]):
#             s2, i2, j2 = from_index(y, L)
#             # get read of periodic wrapping bonds for readability
#             if skip_period and (abs(i1 - i2) > 1 or abs(j1 - j2) > 1):
#                 continue
#             # pass
#             key = int(int_mat[x, y])
#             if key in colors:
#                 lines.append([
#                     positions[s1] + np.array([i1, j1]),
#                     positions[s2] + np.array([i2, j2])])
#                 lines_colors.append(colors[key])

#     plt.gca().add_collection(mc.LineCollection(lines, colors=lines_colors, linewidths=2))
#     plt.axis('equal')
#     plt.show()
