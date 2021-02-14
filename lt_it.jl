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
using HDF5

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

function runone(H, L, tol)
    Ntot = H.Ns * L^2
    
    v = randomvec(L, H.Ns)

    E = energy(H, v)
    Eold = E + tol + 1

    while abs(E - Eold) > tol
        Eold = E
        mcstep!(H, v, Ntot)
        E = energy(H, v)
        # @printf "E = %f\n" E
    end

    E, v, structuralfactor(v, H.rs)
end

function runmany(H, L, tol, k)
    E = zeros(k)
    v = zeros(3, H.Ns, L, L, k)
    f = zeros(L, L, k)

    for i in 1:k
        E[i], v[:, :, :, :, i], f[:, :, i] = runone(H, L, tol)
    end

    E, v, f
end

function mkparams(n, min, max)
    params = zeros(3, div((n + 1) * (n + 2), 2))
    J1 = 1
    
    k = 1
    for i in 0:n
        J2 = min + (max - min) * i / n
        for j in 0:i
            J3 = min + (max - min) * j / n
            params[:, k] = [J1, J2, J3]
            k += 1
        end
    end
    
    params
end

function save(output, E, v, f, params)
    h5open(output, "w") do file
        # for each params
        for i in 1:size(E)[2]
            g = create_group(file, "params-" * string(i))
            # couplings
            attributes(g)["J1"] = params[1, i]
            attributes(g)["J2"] = params[2, i]
            attributes(g)["J3"] = params[3, i]
            # for every sample
            for k in 1:size(E)[2]
                h = create_group(g, "sample-" * string(k))
                h["E"] = E[k, i]
                h["v"] = v[:, :, :, :, k, i]
                h["f"] = f[:, :, k, i]
            end
        end
    end
end

function run()
    n = 2
    nsamples = 2
    output = "ltit.h5"
    L = 32
    tol = 1e-9
    
    params = mkparams(n, 0, 2.5)
    paramH = loadparamhamiltonian("hamiltonians/square.dat")

    nparams = size(params)[2]
    E = zeros(nsamples, nparams)
    v = zeros(3, paramH.Ns, L, L, nsamples, nparams)
    f = zeros(L, L, nsamples, nparams)
    
    for i in 1:nparams
        @printf "Doing J2=%f, J3=%f\n" params[2, i] params[3, i]
        H = mkhamiltonian(paramH, params[:, i])
        E[:, i], v[:, :, :, :, :, i], f[:, :, :, i] = runmany(H, L, tol, nsamples)
    end

    save(output, E, v, f, params)
end

# -----------------------------------------------------------------------------
# Profiling
# -----------------------------------------------------------------------------

function runn(n, fun)
    for i in 1:n
        fun()
    end
end
