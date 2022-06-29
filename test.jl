#!/bin/bash
#=
exec julia --color=yes --startup-file=no -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
=#
using LinearAlgebra
using Printf
include("./testfun.jl")
using .TestFun
include("./bchdav.jl")
using .Bchdav
# include("user_Hx.jl")




A = rand(800,800)
A = (A+A')/2

kwant = 10
opts=Dict([("blk", 5), ("polym", 11), ("tol", 1e-8), ("itmax", 300), ("chksym", false),  ("kmore", 0)])

evals, eigV, kconv, history = bchdav(A, kwant, opts)


@printf("eigenvalues computed by bchdav: \n")
for i = 1:length(evals)
    @printf("%3i-th eigenvalue computed: %f \n", i, evals[i])
end
@printf("relative error: %4.2e \n", norm(A*eigV - eigV*Diagonal(evals))/norm(eigV*Diagonal(evals)))

d, V = eigen(A)
d = sort(d)
@printf("eigenvalues computed by eigen: \n")
for i = 1:length(evals)
    @printf("%3i-th eigenvalue computed: %f \n", i, d[i])
end