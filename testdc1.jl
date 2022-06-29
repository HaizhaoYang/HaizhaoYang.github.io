using Distributed

nprocs = 4
addprocs(nprocs)

include("./dtest.jl")
# @everywhere using .DTest

using LinearAlgebra
using Printf
using SparseArrays

# include("user_Hx.jl")




A = randn(100,100)
A = (A+A')/2
for i = 1:size(A,1)
    for j = 1:size(A,2)
        if abs(A[i,j]) > 0.1
            A[i,j] = 0.0
        end
    end
end
A = sparse(A)

@printf("size of A: %i \n", sizeof(A))

v = rand(100,2)

# @fetchfrom 1 InteractiveUtils.varinfo()
# @fetchfrom 2 InteractiveUtils.varinfo() 

@printf("current processor: %i \n", myid())
DAv = dtest(A, v, workers())

Av = A*v

@printf("error: %.2e \n", norm(Av-DAv)/norm(Av))