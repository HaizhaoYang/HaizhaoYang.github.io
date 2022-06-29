using Distributed
numprocs = 2
addprocs(numprocs)

using LinearAlgebra
using Printf
include("./Bchdav/dbchdav.jl")
# using .Bchdav
using MAT
using Arpack
# using CPUTime
using SparseArrays
include("./utils/constructL.jl")

# sizes = [1000, 10000, 100000, 1000000, 10000000, 50000000]
n_samples = 10000000
steps = 45
kwant = 5
repeats = 3

what = "bin"

m = 11
#tau = 1e-3
itermax = 500
a0 = 0.1


batch = 50
fnorm = "fro"

@printf("type: %s \n", what)


@printf("\n\n")
@printf("========================= #samples = %10d ============================\n", n_samples)

fname = "sparsedata/sparse" * string(n_samples) * "/sparse" * string(n_samples) * what * ".mat"
file = matopen(fname)
A = read(file, "A")
A = (A+A')/2

if  what == "pos"
    D = sparse(collect(1:n_samples), collect(1:n_samples), dropdims(sum(A,dims=1), dims=1), n_samples, n_samples);
    L = constructL(D, A);
elseif what == "bin"
    D = sparse(collect(1:n_samples), collect(1:n_samples), dropdims(sum(A, dims=1),dims=1), n_samples, n_samples);
    L = constructL(D, A);
elseif what == "abs"
    D = sparse(collect(1:n_samples), collect(1:n_samples), dropdims(sum(broadcast(abs,A),dims=1), dims=1), n_samples, n_samples);
    L = constructL(D, A);
end

N = n_samples
# dL = sparse(collect(1:N), collect(1:N), a0*ones(N), N, N)
# L = L + dL
L = (L+L')/2
# J = Matrix{Float64}(L)

V = rand(n_samples, kwant)
cputime1 = zeros(steps)
cputime2 = zeros(steps)
cputime3 = zeros(steps)
cputime4 = zeros(steps)
cputime5 = zeros(steps)
cputime6 = zeros(steps)


upb = 2.0
low_nwb = 0.2
lowb = 0.0
polym = m

for i in 1:steps

    @everywhere using DistributedArrays
    @everywhere using LinearAlgebra
    DL = distribute(L, procs=workers(), dist=[length(workers()), 1])
    # DP = distribute(P, procs=workers(), dist=[length(workers()), 1])

    #opts = Dict([("polym", m), ("tol", tau), ("itmax", itermax), ("chksym", true), ("kmore", 0), ("blk", kwant), ("upb", 2.0), ("nprocs", length(workers())), ("procs", workers()), ("lwb", 0.0)])
    
    @printf("------------------------- numprocs = %i -------------------------\n", length(workers()))
    
    cputime1[i] = @elapsed begin
    for j in 1:repeats
        global V
        mvcput = 0.0

        e = (upb - low_nwb)/2.0
        center= (upb+low_nwb)/2.0
        sigma = e/(lowb - center)
        tau = 2.0/sigma
        
        # y, mvcpu0 = duser_Hx(DL, V)
        mvcpu0 = @elapsed begin
            Dy = dzeros((size(DL,1), size(V,2)), DL.pids, [length(DL.pids), 1])
            @sync for ii in 1:length(DL.pids)
                @spawnat DL.pids[ii] localpart(Dy)[:,:] = localpart(DL)*V
            end
            y = Matrix{Float64}(Dy)
        end
        # elapsedTime["duser_Hx"] += mvcpu0
        # elapsedTime["duser_Hx_n"] += size(V,2)
        mvcput = mvcput + mvcpu0
        y = (y - center*V) * (sigma/e)
        
        for i = 2:polym-1
            sigma_new = 1.0 /(tau - sigma)
            # ynew, mvcpu0 = duser_Hx(DL, y)
            mvcpu0 = @elapsed begin
                Dynew = dzeros((size(DL,1), size(y,2)), DL.pids, [length(DL.pids), 1])
                @sync for ii in 1:length(DL.pids)
                    @spawnat DL.pids[ii] localpart(Dynew)[:,:] = localpart(DL)*y
                end
                ynew = Matrix{Float64}(Dynew)
            end
            # elapsedTime["duser_Hx"] += mvcpu0
            # elapsedTime["duser_Hx_n"] += size(y,2)
            mvcput = mvcput + mvcpu0
            ynew = (ynew - center*y)*(2.0*sigma_new/e) - (sigma*sigma_new)*V
            V = y
            y = ynew
            sigma = sigma_new
        end
        
        # ynew, mvcpu0 = duser_Hx(DL, y)
        mvcpu0 = @elapsed begin
            Dynew = dzeros((size(DL,1), size(y,2)), DL.pids, [length(DL.pids), 1])
            @sync for ii in 1:length(DL.pids)
                @spawnat DL.pids[ii] localpart(Dynew)[:,:] = localpart(DL)*y
            end
            ynew = Matrix{Float64}(Dynew)
        end
        # elapsedTime["duser_Hx"] += mvcpu0
        # elapsedTime["duser_Hx_n"] += size(y,2)
        mvcput = mvcput + mvcpu0
        sigma_new = 1.0 /(tau - sigma)
        # default return unless augment==2 or 3.
        V1 = (ynew - center*y)*(2.0*sigma_new/e) - (sigma*sigma_new)*V
        cputime2[i] += mvcput/repeats
    end
    end
    cputime1[i] /= repeats
    
    @printf("wall time of dCheb_filter_scal: %.2e \n", cputime1[i])
    @printf("wall time of mat-mat multiply : %.2e \n", cputime2[i])


    addprocs(numprocs)
    @printf("\n")
    flush(stdout)     
end

@printf("wall time of dCheb_filter_scal:  \n")
println(cputime1)
@printf("\n")


@printf("wall time of mat-mat multiply:  \n")
println(cputime2)
@printf("\n")
