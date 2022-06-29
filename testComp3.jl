using Distributed

n = 20
addprocs(n)

using DistributedArrays
using Printf
using SparseArrays
using MAT
using LinearAlgebra
using CPUTime
include("./utils/constructL.jl")
include("./utils/filters.jl")
@everywhere using DistributedArrays: localpart

sizes = [1000, 10000, 100000, 1000000]
k = 10
polm = 11
augment = 1
low = 0.3
high = 2.2
leftb = 0.0

for i = 1:length(sizes)
    n_samples = sizes[i]
    @printf("\n\n")
    @time v = rand(n_samples, k)
    @printf("========================= #samples = %10d ============================\n", n_samples)

    fname = "sparsedata/sparse" * string(n_samples) * "/sparse" * string(n_samples) * "bin.mat"
    file = matopen(fname)
    A = read(file, "A")
    A = (A+A')/2

    D = sparse(collect(1:n_samples), collect(1:n_samples), dropdims(sum(A, dims=1),dims=1), n_samples, n_samples);
    L = constructL(D, A);

    L = (L+L')/2
    DL = distribute(L, procs=workers(), dist=[n, 1])
    
    @printf("Cheb_filter application (sequential) \n")
    @time y1 = Cheb_filter(L, v, polm, low, high, augment)

    @printf("dCheb_filter application (distributed) \n")
    @time y2 = dCheb_filter(DL, v, polm, low, high, augment)
    @printf("relative error: %.2e \n", norm(y1-y2)/norm(y1))

    @printf("-----------------------------------------------------------------------")
    @printf("Cheb_filter_scal application (sequential) \n")
    @time y1 = Cheb_filter_scal(L, v, polm, low, high, leftb, augment)

    @printf("dCheb_filter_scal application (distributed) \n")
    @time y2 = dCheb_filter_scal(DL, v, polm, low, high, leftb, augment)
    @printf("relative error: %.2e \n", norm(y1-y2)/norm(y1))

    

end

