#!/usr/bin/env julia

# usage: lt_it.jl skl.dat J1 J2 ...

# steps:
# DONE 1. read dat file
# DONE 2. init with random vector
# 3. monte carlo steps
# 4. save

using LinearAlgebra
using Printf

function iscomment(line)
    return line == "" || line[1] == '#'
end

struct Hamiltonian
    Ns :: Int
    couplings :: Array{Array{Tuple{Int, Int, Int, Float64}}}
end

function load_dat_file(path, couplings)
    open(path, "r") do io
        # compute the number of sites
        line = ""
        while iscomment(line)
            line = readline(io)
        end
        Ns = parse(Int, line)
        # now make a NsxNs matrix of arrays
        bonds = fill(Tuple{Int, Int, Int, Float64}[], Ns)
        
        for line in eachline(io)
            # skip comments
            if iscomment(line) continue end
            s1, i, j, s2, c = [parse(Int, x) for x in split(line)]
            push!(bonds[s1+1], (i+1, j+1, s2+1, couplings[c+1]))
        end

        return Hamiltonian(Ns, bonds)
    end
end

function wrapindex(i, L)
    return 1 + (i - 1) % L
end

function localfield(H, v, i, j, s)
    L = size(v)[1]
    return sum(
        map(H.couplings[s]) do (Δi, Δj, s2, c)
        return c * v[wrapindex(i + Δi, L),
                     wrapindex(j + Δj, L),
                     s2, :]
        end
    )
end

function energy(H, v)
    L = size(v)[1]
    # simply the sum of spin . local field
    E = 0
    
    for i in 1:L
        for j in 1:L
            for s in 1:H.Ns
                S = v[i, j, s, :]
                h = localfield(H, v, i, j, s)
                E += dot(S, h)
            end
        end
    end

    return E / (H.Ns * L^2)
end
    
function randomvec(L, Ns)
    v = randn(L, L, Ns, 3)
    # normalize
    mapslices(v, dims=4) do u
        normalize!(u)
    end
    return v
end

function mcstep(H, v, niter)
    L = size(v)[1]
    for n in 1:niter
        # pick random spin
        i, j = rand(1:L, 2)
        s = rand(1:H.Ns)
        # update spin
        h = localfield(H, v, i, j, s)
        v[i, j, s, :] = -normalize(h)
    end
end

function main()
    L = 10
    tol = 1e-2
    H = load_dat_file("skl.dat", [1, 0.4, 2])
    Ntot = H.Ns * L^2
    
    v = randomvec(L, H.Ns)

    E = energy(H, v)
    Eold = E + tol + 1

    while abs(E - Eold) > tol
        Eold = E
        mcstep(H, v, Ntot)
        E = energy(H, v)
        @printf "E = %f\n" E
    end

    return E, v
end
