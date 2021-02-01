#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# sublattices indices are labeled from left to right, top to bottom.
# the states are represented by a 6xN_cells matrix of complex numbers
# in memory, the layout with always be:
# 6 (sublattices) x Lx x Ly

class Hamiltonian:
    def __init__(self, Ns, couplings, coupling_values, rs):
        self.Ns = Ns
        self.couplings = couplings
        self.coupling_values = coupling_values
        self.rs = rs

def is_comment(line):
    return len(line) == 0 or line[0] == '#'
        
def load_interaction_file(path, coupling_values):
    with open(path) as f:
        line = []
        while is_comment(line):
            line = f.readline()
            
        Ns = int(line) # number of sublattices
        ret = [[[]] * Ns] * Ns

        # read site coords
        n = 0
        rs = np.zeros((Ns, 2))
        while n < Ns:
            line = f.readline()
            if is_comment(line): continue
            rs[n] = np.array([float(x) for x in line.split()])
            n += 1

        # read couplings
        for line in f:
            print(line)
            # skip comments
            if is_comment(line): continue
            s1, i, j, s2, coupling = [int(n) for n in line.split()]
            ret[s1][s2].append((i, j, coupling))
            
        return Hamiltonian(Ns, ret, coupling_values, rs)

def ft(H, k):
    """Computes the Fourier transform of the given Hamiltonian."""
    ret = np.zeros((H.Ns, H.Ns), dtype=complex)
    k = np.array(k)

    for a in range(H.Ns):
        for b in range(H.Ns):
            print(f'cs = {H.couplings[a][b]}')
            for i, j, c in H.couplings[a][b]:
                R = np.array([i, j])
                ret[a][b] += H.coupling_values[c] * np.exp(1j * k.dot(R))

    print(f"ft = {ret / 2}")
    return ret / 2

def main():
    # SKL
    # J1, J2, J3 = 1, 0.2, 0.1 # trivial Néel order
    # h = load_interaction_file('skl.dat', [J1, J2, J3])

    # square lattice
    J1, J2 = 1, 0 # should be only a pair of minima, with 2pi/3 phase
    h = load_interaction_file('hamiltonians/square.dat', [J1, J2])

    N = 100
    kxs = np.linspace(-np.pi, np.pi, N, True)
    kys = np.linspace(-np.pi, np.pi, N, True)

    kxv, kyv = np.meshgrid(kxs, kys)
    kxv = kxv.reshape(-1)
    kyv = kyv.reshape(-1)

    energies = np.zeros(N**2)
        
    for i, (kx, ky) in enumerate(zip(kxv, kyv)):
        print(f'Doing {i} / {N**2}')
        mat = ft(h, [kx, ky])
        xs = np.linalg.eigh(mat)
        energies[i] = xs[0][0]

    # post process to show the minimum
    E0 = min(energies)
    # energies[energies < E0 + 1e-16] = 10
    
    plt.figure()
    plt.imshow(energies.reshape(N, N),
               extent=np.pi * np.array([-1, 1, -1, 1]))
    plt.xlabel('$k_x$')
    plt.ylabel('$k_y$')
    plt.title(r'$\epsilon_0(\vec k)$ [a.u.]')
    plt.colorbar()
    plt.show()
    return energies

if __name__ == '__main__':
    main()
    # E, v = iterative_minimization()
