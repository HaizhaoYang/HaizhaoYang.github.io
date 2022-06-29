# module DTest


using Printf
using LinearAlgebra
using Elemental
using CPUTime
using Statistics
using LowRankApprox
using Distributed
using DistributedArrays

@everywhere using DistributedArrays: localpart

export dtest


function dtest(A, v, procs)
    # addprocs(4)
    DA = distribute(A, procs=procs, dist=[length(procs),1])
    w = @distributed (vcat) for i = 1:length(DA.pids)
        fetch(@spawnat DA.pids[i] localpart(DA)*v)        
    end
    return w
end


# end