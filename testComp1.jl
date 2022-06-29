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
    c = 10.0

    @printf("----------------------------------------------------------------------------------\n")
    @printf("compute dot-product (sequential) \n")
    @time ans1 = V' * W

    @printf("compute dot-product (@distributed) \n")
    @time begin
        ans2 = @distributed (+) for i = 1:n
            localpart(DV)' * localpart(DW)
        end
    end
    @printf("relative error: %.2e \n", norm(ans1-ans2)/norm(ans1))

    @printf("----------------------------------------------------------------------------------\n")
    @printf("compute (hcat)mat-vec (sequential) \n")
    @time ans1 = hcat(V,W)' * V

    @printf("compute (hcat)mat-vec (@distributed) \n")
    @time begin
        ans2 = @distributed (+) for i = 1:n
            hcat(localpart(DV), localpart(DW))' * localpart(DV)
        end
    end
    @printf("relative error: %.2e \n", norm(ans1-ans2)/norm(ans1))

    @printf("----------------------------------------------------------------------------------\n")
    @printf("compute mat-vec (sequential) \n")
    @time ans1 = A * V

    @printf("compute mat-vec (@distributed) \n")
    @time begin
        ans2 = @distributed (vcat) for i = 1:n
            localpart(DA) * V
        end
    end
    @printf("relative error: %.2e \n", norm(ans1-ans2)/norm(ans1))

    @printf("----------------------------------------------------------------------------------\n")
    @printf("compute mat-scalar (sequential) \n")
    @time ans1 = A / c

    @printf("compute mat-scalar (@distributed) \n")
    @time begin
        ans2 = @distributed (vcat) for i = 1:n
            localpart(DA)' / c
        end
    end
    # @printf("relative error: %.2e \n", norm(ans1-ans2)/norm(ans1))

    flush(stdout)     

 
end
    

