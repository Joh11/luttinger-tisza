#!/usr/bin/env julia

# usage: lt_it.jl skl.dat J1 J2 ...

# steps:
# DONE 1. read dat file
# DONE 2. init with random vector
# 3. monte carlo steps
# 4. save

using LinearAlgebra
using Printf
using FFTW

function iscomment(line)
    line == "" || line[1] == '#'
end

struct ParamHamiltonian
    # Like Hamiltonian, but the couplings are not replaced yet
    Ns :: Int
    couplings :: Array{Array{Tuple{Int, Int, Int, Int}}}
    rs :: Array{Float64, 2} # shape: (2, Ns)
end

struct Hamiltonian
    Ns :: Int
    couplings :: Array{Array{Tuple{Int, Int, Int, Float64}}}
    rs :: Array{Float64, 2} # shape: (2, Ns)
end

function mkhamiltonian(paramH, couplings)
    Hamiltonian(paramH.Ns, map(paramH.couplings) do (cs)
                map(cs) do (i, j, s, c)
                (i, j, s, couplings[c])
                end
                end, paramH.rs)
end

function loadparamhamiltonian(path)
    open(path, "r") do io
        # compute the number of sites
        line = ""
        while iscomment(line)
            line = readline(io)
        end
        Ns = parse(Int, line)
        
        # parse site coordinates
        rs = zeros(2, Ns)
        
        let n = 1; while n <= Ns
            # skip comments
            line = readline(io)
            if iscomment(line) continue end
            rs[:, n] = [parse(Float64, x) for x in split(line)]
            n += 1
        end end
        
        # now make a NsxNs matrix of arrays
        bonds = fill(Tuple{Int, Int, Int, Int}[], Ns)
        
        for line in eachline(io)
            # skip comments
            if iscomment(line) continue end
            s1, i, j, s2, c = [parse(Int, x) for x in split(line)]
            push!(bonds[s1+1], (i+1, j+1, s2+1, c+1))
        end

        ParamHamiltonian(Ns, bonds, rs)
    end
end

function loadhamiltonian(path, couplings)
    mkhamiltonian(loadparamhamiltonian(path), couplings)
end

function wrapindex(i, L)
    1 + (i - 1) % L
end

function localfield(H, v, i, j, s)
    L = size(v)[3]
    sum(map(H.couplings[s]) do (Δi, Δj, s2, c)
        c * v[:, s2,
              wrapindex(i + Δi, L),
              wrapindex(j + Δj, L)]
        end)
end

function energy(H, v)
    L = size(v)[3]
    # simply the sum of spin . local field
    E = 0
    
    for s in 1:H.Ns
        for j in 1:L
            for i in 1:L
                S = v[:, s, i, j]
                h = localfield(H, v, i, j, s)
                E += dot(S, h)
            end
        end
    end

    E / (H.Ns * L^2)
end
    
function randomvec(L, Ns)
    v = randn(3, Ns, L, L)
    # normalize
    mapslices(v, dims=1) do u
        normalize!(u)
    end
    return v
end

function mcstep!(H, v, niter)
    L = size(v)[3]
    for n in 1:niter
        # pick random spin
        i = rand(1:L)
        j = rand(1:L)
        s = rand(1:H.Ns)
        # update spin
        h = localfield(H, v, i, j, s)
        v[:, s, i, j] = -normalize(h)
    end
end

function structuralfactor(v, rs)
    # rs shape: (2, Ns)
    Ns, L = size(v)[2:3]
    Ntot = Ns * L^2

    vk = zeros(3, L, L)

    kxs = 2π / L * (0:L-1)
    kys = 2π / L * (0:L-1)
    
    for s in 1:Ns
        vs = v[:, s, :, :]
        kr = [dot([kx, ky], rs[:, s]) for kx in kxs, ky in kys]
        vk += fft(vs, [2, 3]) .* reshape(exp(-1im * kr), (1, L, L))
    end

    1 / Ntot * real(mapslices(sum, conj(vk) .* vk, dims=1))
end

function main()
    L = 32
    tol = 1e-9
    # H = loadhamiltonian("hamiltonians/skl.dat", [1, 0.4, 2])
    H = loadhamiltonian("hamiltonians/square.dat", [1, 0.1])
    Ntot = H.Ns * L^2
    
    v = randomvec(L, H.Ns)

    E = energy(H, v)
    Eold = E + tol + 1

    while abs(E - Eold) > tol
        Eold = E
        mcstep!(H, v, Ntot)
        E = energy(H, v)
        @printf "E = %f\n" E
    end

    E, v, structuralfactor(v, H.rs)
end

# -----------------------------------------------------------------------------
# Profiling
# -----------------------------------------------------------------------------

function runn(n, fun)
    for i in 1:n
        fun()
    end
end
