#!/usr/bin/env julia

#=

Read from input.in

input.in format:
----------------
first line is a comment
Nsites L
Ra Rb i j J
[...]

for a hopping from (0, 0), i to (Ra, Rb), j

=#

using SparseArrays

function readinteraction(path="input.in")
    open(path, "r") do io
        readline(io) # comment line
        Nsites, L = [parse(Int, x) for x in split(readline(io))]
        for line in eachline(io)
            params = split(line)
            Ra, Rb, i, j = [parse(Int, x) for x in params[1:end-1]]
            J = parse(Float64, params[end])
        end
        return Nsites, L
    end
end

function interaction(L, J2, J3=J2)
    ncells = L^2
    N = 6 * ncells
    J1 = 1

    I, J, V = [], [], []

    function index(s, i, j)
        i, j = mod(i, L), mod(j, L)
        return ncells * (s - 1) + L * i + j + 1
    end
    
    function addhopping(s1, s2, i2, j2, hop)
        i1, j1 = 0, 0
        push!(I, index(s1, i1, j1))
        push!(J, index(s2, i2, j2))
        push!(V, hop)
    end
    
    for i in 1:L
        for j in 1:L
            # sublattice 0
            addhopping(1, 2, i, j, J1)
            addhopping(1, 4, i, j, J1)
            addhopping(1, 3, i, j, J2)
            addhopping(1, 6, i, j+1, J3)
            # sublattice 1
            addhopping(2, 1, i, j, J1)
            addhopping(2, 5, i, j, J1)
            addhopping(2, 6, i, j+1, J2)
            addhopping(2, 3, i+1, j, J3)
            # sublattice 2
            addhopping(3, 1, i, j, J2)
            addhopping(3, 5, i-1, j, J2)
            addhopping(3, 4, i, j, J3)
            addhopping(3, 2, i-1, j, J3)
            # sublattice 3
            addhopping(4, 1, i, j, J1)
            addhopping(4, 5, i, j, J1)
            addhopping(4, 6, i, j, J2)
            addhopping(4, 3, i, j, J3)
            # sublattice 4
            addhopping(5, 2, i, j, J1)
            addhopping(5, 4, i, j, J1)
            addhopping(5, 3, i+1, j, J2)
            addhopping(5, 6, i, j, J3)
            # sublattice 5
            addhopping(6, 4, i, j, J2)
            addhopping(6, 2, i, j-1, J2)
            addhopping(6, 5, i, j, J3)
            addhopping(6, 1, i, j-1, J3)
        end
    end
    return sparse(I, J, V)
end
