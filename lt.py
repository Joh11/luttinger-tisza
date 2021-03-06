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
        # initialize ret (manually to avoid shallow copy)
        ret = [[[] for j in range(Ns)] for i in range(Ns)]

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
            for i, j, c in H.couplings[a][b]:
                R = np.array([i, j])
                ret[a, b] += H.coupling_values[c] * np.exp(1j * k.dot(R))

    return ret / 2

def plotband(H, N=50):
    kpath = np.concatenate([
        # Γ M
        np.array([[1, 1]]) * np.linspace(0, np.pi, N, False).reshape(-1, 1),
        # M X
        np.array([[np.pi, 0]]) + np.array([[0, 1]]) * np.linspace(np.pi, 0, N, False).reshape(-1, 1),
        # X Γ
        np.array([[0, 0]]) + np.array([[1, 0]]) * np.linspace(np.pi, 0, N, False).reshape(-1, 1),
        np.array([[0, 0]])
    ])

    nkps = len(kpath)
    energies = np.zeros((nkps, H.Ns))
    for i, k in enumerate(kpath):
        print(f'Doing {i} / {nkps}')
        mat = ft(H, k)
        xs = np.linalg.eigh(mat)
        energies[i] = xs[0]

    # plot
    plt.figure()
    for band in energies.transpose():
        plt.scatter(np.arange(nkps), band, c='r', s=0.1)
    plt.xticks([0, N, 2*N, 3*N], [r'$\Gamma$', 'M', 'X', r'$\Gamma$'])
    plt.ylabel(r'$\varepsilon_n(\vec k)$')
    plt.show()

def compute_energies(H, N=20):
    kxs = np.linspace(-np.pi, np.pi, N)
    kys = np.linspace(-np.pi, np.pi, N)

    kxv, kyv = np.meshgrid(kxs, kys)
    kxv = kxv.reshape(-1)
    kyv = kyv.reshape(-1)

    energies = np.zeros((N**2, H.Ns))
        
    for i, (kx, ky) in enumerate(zip(kxv, kyv)):
        mat = ft(H, [kx, ky])
        energies[i] = np.linalg.eigh(mat)[0]

    return energies, kxs, kys

def classify_phase(Es, kxs, kys):
    # # new classification
    # Es = Es[:, 0]
    # E0 = np.min(Es)
    # idcs = np.where(np.isclose(E0, Es, atol=1e-2))[0]

    # kxv, kyv = np.meshgrid(kxs, kys)
    # kxv = kxv.reshape(-1)
    # kyv = kyv.reshape(-1)
    
    # k = np.column_stack([kxv[idcs], kxv[idcs]])
    # # print(f'number of kpoints: {len(k)}')
    # mean, std = np.mean(k, axis=0), np.std(k, axis=0)
    # # print(f'mean={mean}, std={std}')
    
    # if max(std) > 1:
    #     return 'liquid'
    # else:
    #     assert np.isclose(mean, [0, 0]).all()
    #     return 'order'

    # new new classification
    # liquid if flat band
    Es = Es[:, 0]
    std = np.std(Es)

    print(f'std = {std}')
    return 'liquid' if std < 0.01 else 'order'
    
def phase_diagram():
    Jmin, Jmax = 0, 2.5
    NJ = 100

    J2s = np.linspace(Jmin, Jmax, NJ)
    J3s = np.linspace(Jmin, Jmax, NJ)

    J2v, J3v = np.meshgrid(J2s, J3s)
    J2v = J2v.reshape(-1)
    J3v = J3v.reshape(-1)

    phases = np.zeros((NJ**2, 3))
    for i, (J2, J3) in enumerate(zip(J2v, J3v)):
        print(f'Doing {i} / {NJ ** 2} ({i / NJ**2 * 100}%)')
        print(f'J2 = {J2}, J3 = {J3}')
        H = load_interaction_file('hamiltonians/skl.dat', [1, J2, J3])
        p = classify_phase(*compute_energies(H))
        if p == 'liquid':
            phases[i] = np.array([0, 1, 0])
        else:
            phases[i] = np.array([0, 0, 1])

    phases = phases.reshape(NJ, NJ, 3)

    plt.figure()
    plt.imshow(phases,
               origin='lower',
               extent=[Jmin, Jmax, Jmin, Jmax])
    plt.xlabel('$J_2 / J_1$')
    plt.ylabel('$J_3 / J_1$')
    plt.show()
    return phases

def main():
    # SKL
    # J1, J2, J3 = 1, 1, 0 # trivial Néel order
    # J1, J2, J3 = 1, 2.5, 2.5 # UUD
    J1, J2, J3 = 0, 1, 1 # UUD theory
    h = load_interaction_file('hamiltonians/skl.dat', [J1, J2, J3])

    # square lattice
    # J1, J2 = 1, 0 # should be only a pair of minima, with 2pi/3 phase
    # h = load_interaction_file('hamiltonians/square.dat', [J1, J2])

    N = 50
    kxs = np.linspace(-np.pi, np.pi, N)
    kys = np.linspace(-np.pi, np.pi, N)

    kxv, kyv = np.meshgrid(kxs, kys)
    kxv = kxv.reshape(-1)
    kyv = kyv.reshape(-1)

    energies = np.zeros((N**2, h.Ns))
        
    for i, (kx, ky) in enumerate(zip(kxv, kyv)):
        print(f'Doing {i} / {N**2}')
        mat = ft(h, [kx, ky])
        xs = np.linalg.eigh(mat)
        energies[i] = xs[0]

    for i in range(h.Ns):
        plt.figure()
        plt.imshow(energies[:, i].reshape(N, N),
                   extent=np.pi * np.array([-1, 1, -1, 1]))
        plt.xlabel('$k_x$')
        plt.ylabel('$k_y$')
        plt.title(f'$\\epsilon_{i}(\\vec k)$ [a.u.]')
        plt.colorbar()
    plt.show()
    return energies

if __name__ == '__main__':
    main()

def plot_configuration(h, v, title=None):
    L = 5

    # plt.figure()
    # plot lattice points
    for i in range(L):
        for j in range(L):
            # draw bonds
            R = np.array([i, j])
            rs = R.reshape(1, -1) + h.rs
            for a in range(h.Ns):
                for b in range(h.Ns):
                    for (di, dj, c) in h.couplings[a][b]:
                        if h.coupling_values[c] == 0.: continue
                        line = plt.Line2D((rs[a, 0], rs[b, 0] + di),
                                          (rs[a, 1], rs[b, 1] + dj),
                                          color='k')
                        plt.gca().add_line(line)
            # draw sites
            cs = ['b' if x > 0 else 'r' for x in v]
            plt.scatter(rs[:, 0], rs[:, 1], c=cs, zorder=10)
            
                        

    # draw unit cells
    # lines = [plt.Line2D((i, i), (0, L)) for i in range(L+1)] \
    #       + [plt.Line2D((0, L), (i, i)) for i in range(L+1)]
    # for line in lines:
    #     plt.gca().add_line(line)
    
    plt.axis('scaled')
    plt.xticks([])
    plt.yticks([])
    plt.xlim([1, L-1])
    plt.ylim([1, L-1])
    if title: plt.title(title)
    # plt.show()
    
def plot_phases():
    hn = load_interaction_file('hamiltonians/skl.dat', [1, 1, 0])
    huud = load_interaction_file('hamiltonians/skl.dat', [0, 1, 1])

    plt.subplot(1, 2, 1)
    plot_configuration(hn, np.linalg.eigh(ft(hn, [0, 0]))[1][:, 0], 'Néel order ($J_1 = J_2 = 1$, $J_3 = 0$)')
    plt.subplot(1, 2, 2)
    plot_configuration(huud, np.linalg.eigh(ft(huud, [0, 0]))[1][:, 0], 'UUD ($J_1 = 0$, $J_2, J_3 = 1$)')
    plt.tight_layout()
    # plt.savefig('figs/lt_phases.eps')
    plt.show()
