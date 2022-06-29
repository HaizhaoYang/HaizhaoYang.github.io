module EcoQR
using LinearAlgebra
export ecoqr

function ecoqr(A)
    N, k = size(A)
    tau = zeros(k)
    if k >= N
        @error("The input matrix should be tall and skinny!!")
        return [], []
    end
    LinearAlgebra.LAPACK.geqrf!(A, tau)
    R = triu(A)
    v = zeros(N,k)
    for i = 1:k
        v[i,i] = 1.0
        v[i+1:end,i] = A[i+1:end,i]
    end
    inner_products = zeros(k,k)
    for i = 1:k-1
        inner_products[i,i+1:end] = v[:,i]'*v[:,i+1:end]
    end
    E = Matrix{Float64}(I,N,k)
    subv = zeros(k,k)
    for i = 1:k
        subv[i,:] = v[1:k,i]'
    end
    subv = sparse(subv)
    Q = zeros(N,k)
    for i = 1:k
        Q = Q + (-tau[i])*(v[:,i]*subv[:,i]')
    end
    coe = zeros(k,k)
    for j = 1:k-1
        for i = 1:k-j
            if j == 1
                coe[i,i+j] = tau[i]*tau[i+j]*inner_products[i,i+j]
            else
                coe[i,i+j] = tau[i]*tau[i+j]*inner_products[i,i+j]
                for =
            end
        end
    end
    
end