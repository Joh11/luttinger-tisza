#!/usr/bin/env julia

# usage: lt_it.jl skl.dat J1 J2 ...

# steps:
# 1. read dat file
# 2. init with random vector
# 3. monte carlo steps
# 4. save

function iscomment(line)
    return line == "" || line[1] == '#'
end

function load_dat_file(path, couplings)
    open(path, "r") do io
        # compute the number of sites
        line = ""
        while iscomment(line)
            line = readline(io)
        end
        Ns = parse(Int, line)
        println(Ns)
        # now make a NsxNs matrix of arrays
        bonds = fill(Tuple{Int, Int, Float64}[], Ns, Ns)
        
        for line in eachline(io)
            # skip comments
            if iscomment(line) continue end
            s1, i, j, s2, c = [parse(Int, x) for x in split(line)]
            push!(bonds[s1+1, s2+1], (i, j, couplings[c+1]))
        end

        return Ns, bonds
    end
end
