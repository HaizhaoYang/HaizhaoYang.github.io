using Distributed

n = 10
addprocs(n)

using Printf
using DistributedArrays
using LinearAlgebra
@everywhere using DistributedArrays: localpart
@everywhere using LinearAlgebra: norm, vcat

N = 2000

A = rand(N,N)
DA = distribute(A, procs=workers(), dist=[n,1])
pid2indices = Dict()
for i = 1:n
    pid = DA.pids[i]
    pid2indices[pid] = DA.indices[i]
end

@printf("============================================================= \n")
@printf("load @time macro and initialize v \n")
@time v = rand(N, 100);

@printf("sequential mat-vec multiplication: \n")
@time y1 = A*v

@printf("distributed mat-vac multiplication: \n")
@time begin
    y2 = @distributed (vcat) for i = 1:n
        localpart(DA)*v
    end
end

@printf("Difference between y1 and y2: %.2e \n", norm(y1-y2)/norm(y1))

function global_multiply(v, pid2indices)
    global DA
    y = @distributed (vcat) for i = 1:n
        localpart(DA)*v
    end
    return y
end

function local_multiply(D, v, pid2indices)
    # global DA
    y = @distributed (vcat) for i = 1:n
        localpart(D)*v
    end
    return y
end

@printf("distributed mat-vac multiplication (global DA): \n")
@time y3 = global_multiply(v, pid2indices)


@printf("distributed mat-vac multiplication (local DA): \n")
@time y4 = local_multiply(DA, v, pid2indices)

@printf("Difference between y3 and y4: %.2e \n", norm(y3-y4)/norm(y3))


@printf("================================================================= \n")
M = trunc(Int64, N/2)
@printf("sequential reassigning values: \n")
@time A[:,1:M] = rand(N,M)


@printf("distributed reassigning values (use A): \n")
@time begin
    @sync @distributed for i = 1:n
        localpart(DA)[:,1:M] = A[pid2indices[myid()][1], 1:M]
    end
end

@printf("distributed reassigning values (locally): \n")
@time begin
    @sync @distributed for i = 1:n
        m = size(localpart(DA), 1)
        localpart(DA)[:,1:M] = rand(m,M)
    end
end




