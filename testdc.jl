using Distributed



nprocs = 4
addprocs(nprocs)

include("./dbchdav.jl")
# using .DBchdav

using LinearAlgebra
using Printf
using SparseArrays

# include("user_Hx.jl")




A = randn(500,500)
A = (A+A')/2
for i = 1:size(A,1)
    for j = 1:size(A,2)
        if abs(A[i,j]) > 0.1
            A[i,j] = 0.0
        end
    end
end
A = sparse(A)
DA = distribute(A, procs=workers(), dist=[nprocs, 1])

@printf("size of A: %i \n", sizeof(A))

kwant = 10
opts=Dict([("blk", 5), ("polym", 11), ("tol", 1e-8), ("itmax", 300), ("chksym", false),  ("kmore", 0), ("nprocs", nprocs), ("procs", workers())])

@printf("current processor: %i \n", myid())
evals, eigV, kconv, history = dbchdav(DA, kwant, opts)


@printf("eigenvalues computed by bchdav: \n")
for i = 1:length(evals)
    @printf("%3i-th eigenvalue computed: %f \n", i, evals[i])
end
@printf("relative error: %4.2e \n", norm(A*eigV - eigV*Diagonal(evals))/norm(eigV*Diagonal(evals)))

@time d, V = eigen(Matrix{Float64}(A))
d = sort(d)
@printf("eigenvalues computed by eigen: \n")
for i = 1:length(evals)
    @printf("%3i-th eigenvalue computed: %f \n", i, d[i])
end