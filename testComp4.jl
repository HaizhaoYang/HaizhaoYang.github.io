using Distributed

n = 50
addprocs(n)

using DistributedArrays
using Printf
using SparseArrays
using MAT
using LinearAlgebra
using CPUTime
using LowRankApprox
include("./utils/dgks.jl")
@everywhere using DistributedArrays: localpart

sizes = [1000, 10000, 100000, 1000000, 10000000]
m = 10


for i = 1:length(sizes)
    n_samples = sizes[i]
    @printf("\n\n")
    @time v = rand(n_samples, k)
    @printf("========================= #samples = %10d ============================\n", n_samples)

    @time begin
        V0, _ = pqr(rand(n_samples, 2*m))
        dV0 = distribute(V0, procs=workers(), dist=[n, 1])
    end
    
    @time begin
        V = rand(n_samples, m)
        dV = distribute(V, procs=workers(), dist=[n, 1])
    end
    
    @printf("DGKS_blk (sequential) \n")
    @time nV1 = DGKS_blk(V0, V)

    @printf("dDGKS_blk (distributed) \n")
    @time nV2 = dDGKS_blk(dV0, 1:2*m, dV)

    @printf("relative error: %.2e \n", norm(nV1-nV2)/norm(nV1))

end
