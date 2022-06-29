using Distributed

n = 20
addprocs(n)

using DistributedArrays
using Printf
using SparseArrays
using MAT
using LinearAlgebra
@everywhere using DistributedArrays: localpart

sizes = [1000, 10000, 100000, 1000000]
m = 100
k = 50
k2 = trunc(Int64, k/2)

for i = 1:length(sizes)
    n_samples = sizes[i]
    @printf("\n\n")
    @printf("========================= #samples = %10d ============================\n", n_samples)

    fname = "sparsedata/sparse" * string(n_samples) * "/sparse" * string(n_samples) * "bin.mat"
    file = matopen(fname)
    A = read(file, "A")
    A = (A+A')/2
    DA = distribute(A, procs=workers(), dist=[n, 1])

    @time V = rand(n_samples, m)
    DV = distribute(V, procs=workers(), dist=[n, 1])
    W = rand(n_samples, m)
    DW = distribute(W, procs=workers(), dist=[n, 1])
    Y = rand(k,k)
    c = 10.0

    pid2indices = Dict()
    for i = 1:length(workers())
        pid = DV.pids[i]
        pid2indices[pid] = DV.indices[i]
    end

    @printf("----------------------------------------------------------------------------------\n")
    @printf("reassigning by a global W (sequential) \n")
    @time V[:, 1:k] = W[:, 1:k]

    @printf("reassigning by a global W (@distributed) \n")
    @time begin
        @distributed for i = 1:n
            localpart(DV)[:,1:k] = W[pid2indices[myid()][1], 1:k]
        end
    end

    @printf("reassigning by a global W (@sync) \n")
    @time begin
        @sync for pid in workers()
            @spawnat pid localpart(DV)[:,1:k] = W[pid2indices[pid][1], 1:k]
        end
    end

    @printf("reassigning by a global W (@sync @distributed) \n")
    @time begin
        @sync @distributed for i = 1:n
            localpart(DV)[:,1:k] = W[pid2indices[myid()][1], 1:k]
        end
    end

    @printf("----------------------------------------------------------------------------------\n")
    @printf("mat-vec and reassign (sequential) \n")
    @time V[:,1:k] = V[:,1:k]*Y

    @printf("mat-vec and reassign (@distributed) \n")
    @time begin
        @distributed for i = 1:n
            localpart(DV)[:,1:k] = localpart(DV)[:,1:k]*Y
        end
    end

    @printf("mat-vec and reassign (@sync) \n")
    @time begin
        @sync for pid in workers()
            @spawnat pid localpart(DV)[:,1:k] = localpart(DV)[:,1:k]*Y
        end
    end
    
    @printf("mat-vec and reassign (@distributed) \n")
    @time begin
        @distributed for i = 1:n
            localpart(DV)[:,1:k] = localpart(DV)[:,1:k]*Y
        end
    end

    @printf("----------------------------------------------------------------------------------\n")
    @printf("move columns (sequential) \n")
    @time begin
        tmp = V[:,k]
        V[:,2:k] = V[:,1:k-1]
        V[:,1] = tmp
    end

    @printf("move columns (@distributed) \n")
    @time begin
        @distributed for i = 1:n
            tmp = localpart(DV)[:,k]
            localpart(DV)[:,2:k] = localpart(DV)[:,1:k-1]
            localpart(DV)[:,1] = tmp
        end
    end

    @printf("move columns (@sync) \n")
    @time begin
        @ync for pid in workers()
            @spawnat pid begin
                tmp = localpart(DV)[:,k]
                localpart(DV)[:,2:k] = localpart(DV)[:,1:k-1]
                localpart(DV)[:,1] = tmp
            end
        end
    end

    @printf("move columns (@sync @distributed) \n")
    @time begin
        @sync @distributed for i = 1:n
            tmp = localpart(DV)[:,k]
            localpart(DV)[:,2:k] = localpart(DV)[:,1:k-1]
            localpart(DV)[:,1] = tmp
        end
    end


    @printf("----------------------------------------------------------------------------------\n")
    @printf("mat-scalar and reassign (sequential) \n")
    @time V[:,1:k] = V[:,1:k] / c

    @printf("mat-scalar and reassign (@distributed) \n")
    @time begin
        @distributed for i = 1:n
            localpart(DV)[:,1:k] = localpart(DV)[:,1:k] / c
        end
    end

    @printf("mat-scalar and reassign (@sync) \n")
    @time begin
        @sync for pid in workers()
            @spawnat pid localpart(DV)[:,1:k] = localpart(DV)[:,1:k] / c
        end
    end
    
    @printf("mat-scalar and reassign (@distributed) \n")
    @time begin
        @distributed for i = 1:n
            localpart(DV)[:,1:k] = localpart(DV)[:,1:k] / c
        end
    end

    @printf("----------------------------------------------------------------------------------\n")
    @printf("(hcat)mat-vec and reassign (sequential) \n")
    @time V[:,1:k] = hcat(V[:,1:k2], W[:,1:k2])*Y

    @printf("(hcat)mat-vec and reassign (@distributed) \n")
    @time begin
        @distributed for i = 1:n
            localpart(DV)[:,1:k] = hcat(localpart(DV)[:,1:k2], localpart(DW)[:,1:k2])*Y
        end
    end

    @printf("(hcat)mat-vec and reassign (@sync) \n")
    @time begin
        @sync for pid in workers()
            @spawnat pid localpart(DV)[:,1:k] = hcat(localpart(DV)[:,1:k2], localpart(DW)[:,1:k2])*Y
        end
    end
    
    @printf("(hcat)mat-vec and reassign (@distributed) \n")
    @time begin
        @distributed for i = 1:n
            localpart(DV)[:,1:k] = hcat(localpart(DV)[:,1:k2], localpart(DW)[:,1:k2])*Y
        end
    end


    @printf("----------------------------------------------------------------------------------\n")
    @printf("(hcat)mat and reassign (sequential) \n")
    @time V[:,1:k] = hcat(V[:,1:k2], W[:,1:k2])

    @printf("(hcat)mat and reassign (@distributed) \n")
    @time begin
        @distributed for i = 1:n
            localpart(DV)[:,1:k] = hcat(localpart(DV)[:,1:k2], localpart(DW)[:,1:k2])
        end
    end

    @printf("(hcat)mat and reassign (@sync) \n")
    @time begin
        @sync for pid in workers()
            @spawnat pid localpart(DV)[:,1:k] = hcat(localpart(DV)[:,1:k2], localpart(DW)[:,1:k2])
        end
    end
    
    @printf("(hcat)mat and reassign (@distributed) \n")
    @time begin
        @distributed for i = 1:n
            localpart(DV)[:,1:k] = hcat(localpart(DV)[:,1:k2], localpart(DW)[:,1:k2])
        end
    end
    
    @printf("relative error: %.2e \n", norm(V-Matrix{Float64}(DV))/norm(V))


end