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

# -----------------------------------------------------------------------------
# Macro stuff
# -----------------------------------------------------------------------------

# source
# s1 -> [(Δi, Δj, s, c)]
# ie
# 1 -> [(0, 0, 1, 0.1),
#       (0, 0, 2, 0.2)]

# target
function(v, i, j, s)
    h = zeros(3)

    if s == 1
        h += c * v[i + Δi, j + Δj, s2, :]
    elseif s == 2
    end


    return h
end

function f(cs)
    quote
        $([:(h += $c * v[i + $Δi, j + $Δj, $s, :])
           for (Δi, Δj, s, c) in cs]...)
    end
end

function ifclause(condfun, bodies, acc=1)
    # build a if ... elseif ... elseif ... end construct, with
    # condfun(i) as the condition, and bodies[i] as the ith clause
    if acc == length(bodies)
        Expr(acc == 1 ? :if : :elseif,
             condfun(acc),
             bodies[acc])
    else
        Expr(acc == 1 ? :if : :elseif,
             condfun(acc),
             bodies[acc],
             ifclause(condfun, bodies, acc+1))
    end
end

function mklocalfield(H)
    condfun = n -> :(s == $n)
    
    e = quote function(v, i, j, s)
        h = zeros(3)
        $(ifclause(condfun, [f(H.couplings[s1]) for s1 in 1:H.Ns]))
    end end
    return eval(e)
end

# -----------------------------------------------------------------------------
# Good old code
# -----------------------------------------------------------------------------


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
    L = 64
    tol = 1e-9
    # H = load_dat_file("skl.dat", [1, 0.4, 2])
    H = load_dat_file("square.dat", [1, 0.1])
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
