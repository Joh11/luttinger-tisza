#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import collections as mc

from skl import interaction_matrix

# sublattices indices are labeled from left to right, top to bottom.
# the states are represented by a 6xN_cells matrix of complex numbers
# in memory, the layout with always be:
# 6 (sublattices) x Lx x Ly

def fourier_transform(int_mat, L, k):
    """Fourier transform the interaction matrix at the wave vector k. """
    # TODO do it the dumb way for now, but fft should be usable
    int_mat = int_mat.reshape(6, L**2, 6, L**2).transpose(0, 2, 1, 3) # 6,6,L²,L² shape

    R = np.array([[i // L, i % L] for i in range(L**2)])
    Rij = R.reshape(1, L**2, 2) - R.reshape(L**2, 1, 2)
    kRij = (k * Rij).sum(axis=-1) # L²,L² shape

    return (int_mat * np.exp(-1j * kRij)).sum(axis=(2, 3))

L = 30
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
        if i % 10 == 0:
            print(f'{i} / {len(kpath)}')
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

def make_bigger(mat, d=3):
    n = len(mat)
    mat = mat.reshape(n, 1, n, 1)
    m = np.eye(d, d).reshape(1, d, 1, d)
    return (mat * m).reshape(n * d, n * d)

def norm_per_site(s):
    s = s.reshape(-1, 3)
    return np.linalg.norm(s, axis=1)
    
def find_ground_state(int_mat, L):
    eps = 1e-8
    N = 6 * L**2
    # compute lowest energies and states for each wave vector
    energies = np.zeros((L, L))
    states = [None] * L **2
    for i in range(L):
        for j in range(L):
            k = 2 * np.pi / L * np.array([i, j])
            mat = fourier_transform(int_mat, L, k)
            eigvals, eigvecs = np.linalg.eigh(mat)
            # normalize energy per site
            eigvals = eigvals / N
            # find all the ground states (could be degenerate)
            k = 1
            while k < 6:
                if eigvals[k] < eigvals[0] + eps: # almost equal
                    k = k + 1
                else:
                    break
                
            energies[i, j] = eigvals[0]
            states[L * i + j] = eigvecs[:k]

    print('degeneracies:')
    print(np.array([[len(states[L * i + j]) for j in range(L)] for i in range(L)]))

    min_energy = np.min(energies)
    min_idcs = np.argwhere(energies < min_energy + eps)
    print(f'number of k points of smallest energy: {len(min_idcs)}')

    # simplest case
    if len(min_idcs) == 1:
        i, j = min_idcs[0]
        min_states = states[L * i + j]
        
        if len(min_states) == 1:
            state = min_states[0]
            return state
        
            
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
