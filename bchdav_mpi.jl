# module DBchdav


using Printf
using Statistics
using LowRankApprox
using MPI
using LinearAlgebra
using Test

# export dbchdav



function dbchdav(DA, dim, nwant, comm_info, user_opts;X, Y, X_gather, X_gather_T, X_gather_vbuf, Y_gather, Y_gather_T, Y_gather_T_vbuf, verb=false)
    # bchdav.jl implements the block chebyshev-davidson method for 
    # computing the smallest eigenpairs of symmetric/Hermitian problems.

    # Usage:
    #      [evals, eigV, kconv, history] = bchdav(varargin)

    # where   bchdav(varargin)  is one of the following:
    # ------------------------------------------------------------------------    
    # 1.  bchdav(A);
    # 2.  bchdav(A, nwant);
    # 3.  bchdav(A, nwant, opts);
    # 4.  bchdav(Astring, dim, nwant, opts);  %when A is a script function,
    #                                         %need to specify the dimension
    # ------------------------------------------------------------------------

    # A is the matrix for the eigenproblem, (dim=size(A,1)), 
    #   (the A can be input as a function for matrix-vector products)

    # nwant is the number of wanted eigen pairs,
    #   by default: nwant=6.

    # opts is a structure containing the following fields: (order of field names 
    # does not matter. there is no need to specify all fields since most of the time
    # the default values should suffice.
    # the more critical fields that can affect performance are blk, polm, vimax)

    #        blk -- block size, 
    #               by default: blk = 3.
    #      polym -- the degree of the Chebyshev polynomial; 
    #               by default:  polym=20.
    #      vimax -- maximum inner-restart subspace (the active subspace) dimension, 
    #               by default:  vimax = max(max(5*blk, 30), ceil(nwant/4)).

    #   do_outer -- do outer restart (if no outer restart, inner restart may gradually 
    #               increase subspace dim until convergence if necessary)
    #               by default:  do_outer = logical(1).
    #      vomax -- maximum outer-restart subspace dimension;
    #               by default:  vomax= nwant+30.  

    #     filter -- filter method to be used;
    #               currently only two Chebshev filters are implemented,
    #               by default: filter=2. (the one with simple scaling)
    #        tol -- convergence tolerance for the residual norm of each eigenpair;
    #               by default:  tol =1e-8.
    #      itmax -- maximum iteration number;
    #               by default:  itmax= max(floor(dim/2), 300).
    #      ikeep -- number of vectors to keep during inner-restart,
    #               by default:  ikeep= max(floor(vimax/2), vimax-3*blk).

    #         v0 -- the initial vectors (can be of any size);
    #               by default: v0 = rand(dim,blk).
    #      displ -- information display level; 
    #               (<=0 --no output; 1--some output; >=2 --more output, 
    #                when displ>5, some expensive debugging calculations will be performed)
    #               by default: displ=1.
    #     chksym -- check the symmetry of the input matrix A.
    #               if chksym==1 and A is numeric, then isequal(A,A') is called.
    #               the default is not to check symmetry of A. 
    #      kmore -- additional number of eigen-pairs to be computed.
    #               by default, kmore=3.
    #        upb -- upper bound of all the eigenvalues of the input matrix A.
    #               (provide this bound only when you know a good bound; otherwise, 
    #                the code has an estimator to compute this upper bound.)
    #        lwb -- lower bound of all the eigenvalues of the input matrix A.
    #    augment -- choose how many filtered vectors to keep in the basis,  
    #               by default augment=1,  only the last blk filtered vectors are kept;
    #               if augment=2,  the last 2*blk  filtered vectors are kept;
    #               if augment=3,  the last 3*blk  filtered vectors are kept.
    #     nprocs -- number of procesors to use
    #      procs -- list of procesors

    # ========== Output variables:

    #       evals:  converged eigenvalues (optional).

    #       eigV:  converged eigenvectors (optional, but since eigenvectors are
    #              always computed, not specifying this output does not save cputime).

    #      kconv:  number of converged eigenvalues (optional).

    #    history:  log information (optional)
    #              log the following info at each iteration step:

    #              history(:,1) -- iteration number (the current iteration step)
    #              history(:,2) -- cumulative number of matrix-vector products 
    #                              at each iteration, history(end,2) is the total MVprod.
    #              history(:,3) -- residual norm at each iteration
    #              history(:,4) -- current approximation of the wanted eigenvalues

    # ---------------------------------------------------------------------------

    # As an example:

    #    A = delsq(numgrid('D',90));   A = A - 1.6e-2*speye(size(A));
    #    k = 10;  blk=3;  v=ones(size(A,1), blk);
    #    opts = struct('vmax', k+5, 'blk', blk, 'v0', v, 'displ', 0);
    #    [evals, eigV] = bchdav(A, k, opts);  

    # will compute the k smallest eigenpairs of A, using the specified
    # values in opts and the defaults for the other unspecified parameters.


    
    # ---coded by  y.k. zhou,   yzhou@smu.edu
    #   Jan 2010 

    # (revision june 2012:  1. remove a redundant check of symmetry of A
    #                      2. change some displ setting so that when disp=1, 
    #                         converging histry will be displayed as iteration continues)

    
    #   use a global variable 'MVprod' to count the number of matrix-vector products.
    #   this number is adjusted in the user provided mat-vect-product script 'duser_Hx', 
    #   therefore it is automatically incremented  whenever 'duser_Hx' is called.
    #   (by this the mat-vect-product count will be accurate, there is no need 
    #   to manually increase the count within this code in case one may miss
    #   increasing the count somewhere by accident)
    #   use a global variable 'MVcpu' to store the cputime spent on mat-vect products.
    

    global MVprod
    global MVcpu
    # initialize mat-vect-product count and mat-vect-product cpu to zero
    MVprod = 0
    MVcpu = 0

    global filt_non_mv_cput
    global filt_mv_cput 
    filt_non_mv_cput = 0
    filt_mv_cput = 0
    returnhere = 0

    # cputotal = cputime   #init the total cputime count
    global elapsedTime = Dict("total"=>0.0, "Cheb_filter"=>0.0, "Cheb_filter_n"=>0.0, "Cheb_filter_scal"=>0.0,
                "Cheb_filter_scal_n"=>0.0, "SpMM"=>0.0, "SpMM_n"=>0.0, "main_loop"=>0.0, "DGKS"=>0.0, "DGKS_n"=>0.0,
                "Inner_prod"=>0.0, "Inner_prod_n"=>0.0, "Hn"=>0.0, "Hn_n"=>0.0, "Norm"=>0.0, "Norm_n"=>0.0
    )

    
    comm = comm_info["comm"]
    # comm_T = comm_info["comm_T"]
    comm_row = comm_info["comm_row"]
    comm_rol = comm_info["comm_col"]
    rank = comm_info["rank"]
    rank_row = comm_info["rank_row"]
    rank_col = comm_info["rank_col"]
    info_cols_dist = comm_info["info_cols_dist"]
    comm_size = comm_info["comm_size"]
    comm_size_sq = comm_info["comm_size_sq"]

    elapsedTime["total"] = @elapsed begin

        #
        # Process inputs and do error-checking
        #

        # DA = varargin[1]

        # dim = size(DA,1)

        Anumeric = 1

        #  Set default values and apply existing input options:
        #  default opt values will be overwritten by user input opts (if exist).
        
        #  there are a few unexplained parameters which are mainly used to
        #  output more info for comparision purpose, these papameters can be 
        #  safely neglected by simply using the defaults.

        blk = 3
        opts=Dict([("blk", blk), ("filter", 2), ("polym", 20), ("tol", 1e-8),
                    ("vomax", nwant+30),  ("do_outer", true),
                    ("vimax", max(max(5*blk, 30), ceil(nwant/4))),
                    ("adjustvimax", true), 
                    ("itmax", max(floor(dim/2), 300)), ("augment", 1), 
                    ("chksym", false),  ("displ", 1),  ("kmore", 3), ("forArray", rand(5,5))])

        if !isa(user_opts,Dict)
            if rank == 0
                @error("Options must be a dictionary. (note bchdav does not need mode)")
            end
        else
            # overwrite default options by user input options
            # opts = merge(user_opts, opts) 
            for key in keys(user_opts)
                opts[key] = user_opts[key]
                
            end       
        end

        # save opt values in local variables 
        blk = opts["blk"];  filter=opts["filter"];  polym=opts["polym"];  tol=opts["tol"];
        vomax=opts["vomax"];  vimax=opts["vimax"];  itmax=opts["itmax"];  
        augment=opts["augment"];  kmore=opts["kmore"];  displ=opts["displ"]; 

        if  haskey(opts, "v0")
            sizev0 = size(opts["v0"],1)
            if sizev0 < blk
            # @printf("*** input size(v0,2)=%i, blk=%i, augment %i random vectors\n",
                    # sizev0, blk, blk-sizev0)
                DV0 = vcat(opts["v0"], rand(blk-sizev0, info_cols_dist[rank+1]))
            end
        else
            DV0 = rand(blk, info_cols_dist[rank+1])
            sizev0 = blk
        end


        if opts["do_outer"] 
            vomaxtmp = max(min(nwant + 6*blk, nwant+30), ceil(nwant*1.2))
            if  vomax < vomaxtmp 
                if rank == 0
                    @printf("--> Warning: vomax=%i, nwant=%i, blk=%i\n", vomax, nwant, blk)
                end
                vomax = vomaxtmp
                if rank == 0
                    @printf("--> Warnning: increase vomax to %i\n",vomax)
                end
            end  
            if  vimax > vomax
                if rank == 0
                    @printf("--> Warning:  (vimax > vomax)  vimax=%i, vomax=%i\n", vimax, vomax)
                end
                vimax = max(min(6*blk, nwant), ceil(nwant/4))  #note vomax > nwant
                if rank == 0
                    @printf("--> reduce vimax to %i\n", vimax)
                end
            end
        end

        if  vimax < 5*blk
            if rank == 0 
                @printf("--> Warning:  (vimax < 5*blk)  vimax=%i, blk=%i\n", vimax, blk)
            end
            if  opts["adjustvimax"] 
                vimax = 5*blk
                if rank == 0
                    @printf("--> increase vimax to %i\n", vimax)   
                end     
            elseif 3*blk > vimax
                vimax = 3*blk
                if rank == 0
                    @printf("--> adjust vimax to %i\n", vimax)
                end
            end
        end
        if vimax > vomax 
            vomax = vimax+2*blk
        end
        ikeep = trunc(Int64, max(floor(vimax/2), vimax-3*blk))


        # ##################################################################

        #  Now start the main algorithm:

        # ##################################################################

        #  Comment: when the matrix A is large, passing A explicitly as a 
        #  variable in the function interface is not as efficient as using 
        #  A as a global variable. 
        #  In this code A is passed as a global variable when A is numeric. 


        longlog = 1

        #  Preallocate memory, useful if dim and vomax are large
        DV = rand(vomax, info_cols_dist[rank+1])
        DW = zeros(Float64, trunc(Int64, vimax), size(DV,2))
        Hn = zeros(trunc(Int64, vimax), trunc(Int64, vimax))
        evals = zeros(nwant)   
        resnrm = zeros(nwant,1)


        #  get the very important filtering upper bound. 
        #  if there is a user input upb, make sure the input upb is an upper bound, 
        #  otherwise convergence can be very slow. (safer to just let the LANCZ_bound() estimate an upper bound without a user input upb)
        #  an estimated lower bound can also be obtained via LANCZ_bound()

        lancz_step = 4
        # if  Anumeric > 0
        # elapsedTime["LANCZ_bound"] += @elapsed begin
        # upb, low_nwb, lowb, maxritz = LANCZ_bound(dim, lancz_step, DA)  
        # end
        # elapsedTime["LANCZ_bound_n"] += 1
        upb = 2.0
        lowb = 0.0
        maxritz = 0.0
        low_nwb = lowb + (upb - lowb)/20

        if haskey(opts, "upb")
            if opts["upb"] < upb
                if rank == 0
                    @warn("user input upperbound may be too small, may NOT converge!!")
                end
                upb = opts["upb"]   #still use the user input upb, run at your own risk
            end
        end

        if haskey(opts, "lwb")
            if opts["lwb"] > lowb
                if rank == 0
                    @warn("user input upperbound may be too large, may NOT converge!!")
                end
                lowb = opts["lwb"]   #still use the user input lwb, run at your own risk
            end
        end

        if haskey(opts, "low_nwb")
            low_nwb = opts["low_nwb"]   #still use the user input upb, run at your own risk
        end

        #
        # add some variables to measure computational cost
        #
        iter_total = 0           # init the total iteration number count
        orth_cputotal = 0        # init the total cputime for orthogonalization
        orth_flopstotal = 0      # init the total flops for orthogonalization
        refinement_cputotal = 0  # init the total cputime for rayleigh-ritz refinement
        nlog = 1

        kinner = 0;  kouter = 0;  # count the number of inner and outer restarts
        Hn_cpu=0; conv_cpu=0;  # temp variables, used only for measuring cpu

        # -----------------------------------------------------------------------------
        #  the two-indeces trick: use n and kact,  where n = kact+kconv, 
        #  n is the same as ksub in the paper, note that n is NOT the dimension 'dim'
        # -----------------------------------------------------------------------------
        n = 0        # n stores column # in subspace V 
        kact = 0     # kact counts the dimension of the active subspace
        kconv = 0    # init number of converged eigenvalues
        kconv1 = 1   # auxiliary, kconv1 always stores kconv+1 
        history = zeros(0,4)
        
        elapsedTime["DGKS"] += @elapsed begin
        DV[1:blk, :], _ = DGKS(zeros(1, size(DV,2)), 1:0, DV0, dim, comm_row, comm_col)
        end
        elapsedTime["DGKS_n"] += 1

        ec = 0
        kinit = blk

        # global X = zeros(Float64, blk, size(DV,2))
        # global Y = zeros(Float64, blk, info_cols_dist[rank_col*comm_size_sq+rank_row+1])
        # Y_gather = zeros(Float64, size(DA, 1), blk)
        # X_gather_T = zeros(Float64, size(DA, 2), blk)
        # local_info_cols = info_cols_dist[rank_row*comm_size_sq+1:rank_row*comm_size_sq+comm_size_sq]
        # X_gather = Array{Float64}(undef, (blk, sum(local_info_cols)))
        # _counts = vcat([blk for i in 1:length(local_info_cols[:])]', info_cols_dist[rank_row*comm_size_sq+1:(rank_row+1)*comm_size_sq]')
        # global X_gather_vbuf = VBuffer(X_gather, vec(prod(_counts, dims=1)))
        # global Y_gather_T = zeros(Float64, blk, size(DA,1))
        # _counts = vcat([blk for i in 1:length(local_info_cols[:])]', info_cols_dist[rank_col*comm_size_sq+1:(rank_col+1)*comm_size_sq]')
        # global Y_gather_T_vbuf = VBuffer(Y_gather_T, vec(prod(_counts, dims=1)))
        inds = [ind for ind = 1:size(DA,1)]
        DE = rank_col == rank_row ? sparse(inds, inds, ones(Float64, size(DA, 1)), size(DA,1), size(DA,2)) : sparse([1], [1], [0.0], size(DA,1), size(DA,2))
        
        MPI.Barrier(comm)

        elapsedTime["main_loop"] = @elapsed begin
            while  iter_total <= itmax 
                
                iter_total = iter_total +1
                
                # DVtmp = zeros(Float64, blk, size(DV, 2))
                if  ec > 0  &&  kinit + ec <= sizev0
                    if  displ > 4
                        if rank == 0
                            @printf("---> using column [%i : %i] of initial vectors\n",kinit+1, kinit+ec) 
                        end
                    end
                    # Vtmp = hcat(Matrix{Float64}(opts["v0"][:,kinit+1:kinit+ec]), Matrix{Float64}(V[:,kconv1:kconv+blk-ec]))
                    X = vcat(DV0[kinit+1:kinit+ec, :], DV[kconv1:kconv+blk-ec, :])
                    kinit = kinit + ec
                else
                    # Vtmp = Matrix{Float64}(V[:,kconv1:kconv+blk])
                    X = DV[kconv1:kconv+blk, :]
                end

                elapsedTime["Cheb_filter_scal"] += @elapsed begin
                # Vtmp = Cheb_filter_scal(DA, Vtmp, polym, low_nwb, upb, lowb, augment)
                X = Cheb_filter_scal(polym, X, Y, DA, DE, X_gather, X_gather_T, X_gather_vbuf, Y_gather, Y_gather_T, Y_gather_T_vbuf, 0, comm_row, comm_col)
                end
                elapsedTime["Cheb_filter_scal_n"] += 1
                
                #
                # make t orthogonal to all vectors in V
                #
                n = kconv + kact
                # orth_cpu = cputime
                orth_cputotal += @elapsed begin
                    X1, orth_flops = DGKS(DV, 1:n, X, dim, comm_row, comm_col)
                end
                orth_flopstotal = orth_flopstotal + orth_flops
                elapsedTime["DGKS"] += orth_cputotal
                elapsedTime["DGKS_n"] += 1
                kb = size(X1,1)
                n1 = n+1
                kact = kact + kb
                n = kconv + kact   
                # V[:, n1:n] = Vtmp
               
                DV[n1:n,:] = X1
                
                #
                # compute new matrix-vector product.
                #
                # if  Anumeric > 0
                elapsedTime["SpMM"] += @elapsed begin
                DW[kact-kb+1:kact, :] = SpMM_A_0(DV[n1:n, :], DA, DE, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
                end
                elapsedTime["SpMM_n"] += 1
                # else
                #     W[:, kact-kb+1:kact],_ =  duser_Hx(V[:, n1:n], A_operator)  
                # end

                
                # Hn_cpu0=cputime
                #
                # update Hn, compute only the active part (Hn does not include any deflated part)
                #
                # Hn[1:kact, kact-kb+1:kact] = V[:, kconv1:n]'* W[:, kact-kb+1:kact]
                Hn_cpu += @elapsed begin

                    pHn = zeros(Float64, kact, kb)
                    mul!(pHn, DV[kconv1:n,:], DW[kact-kb+1:kact, :]')
                    MPI.Allreduce!(pHn, +, comm_row)
                    MPI.Allreduce!(pHn, +, comm_col)
                    Hn[1:kact, kact-kb+1:kact] = pHn
                    #
                    # symmetrize the diagonal block of Hn, this is very important since
                    # otherwise Hn may not be numerically symmetric from last step 
                    # (the orthonormality of eigenvectors of Hn will be poor without this step)
                    #
                    Hn[kact-kb+1:kact,kact-kb+1:kact]=(Hn[kact-kb+1:kact,kact-kb+1:kact] + Hn[kact-kb+1:kact,kact-kb+1:kact]')/2.0
                    if  kact > kb  # symmetrize Hn
                        Hn[kact-kb+1:kact, 1:kact-kb] = Hn[1:kact-kb, kact-kb+1:kact]'
                    end
                end
                elapsedTime["Hn"] += Hn_cpu
                elapsedTime["Hn_n"] += 1

                # refinement_cpu = cputime
                refinement_cputotal += @elapsed begin
                    #
                    # Rayleigh-Ritz refinement (at each ietration)
                    # First compute the eigen-decomposition of the rayleigh-quotient matrix
                    # (sorting is unnecessary since eig(Hn) already sorts the Ritz values). 
                    # Then rotate the subspace basis.
                    #
                    d_eig, Eig_vec = eigen(Hn[1:kact,1:kact])  
                    Eig_val = Diagonal(d_eig)
                    
                    kold = kact
                    if  kact + blk > vimax
                        #
                        # inner-restart, it can be easily controlled by the two-indeces (kact,n) trick
                        #
                        if  displ > 4 
                            if rank == 0
                                @printf("==> Inner-restart: vimax=%i, n=%i, kact from %i downto %i \n", vimax, n, kact, ikeep)
                            end
                        end
                        kact = ikeep
                        n = kconv+kact   # should always keep n=kconv+kact
                        kinner = kinner+1  #used only for counting
                    end 
                    
                    # V[:,kconv1:kconv+kact] = V[:,kconv1:kconv+kold]*Eig_vec[1:kold,1:kact]
                    Eig_vec_p = Eig_vec[1:kold, 1:kact]
                    # @spawnat V.pids[i] localpart(V)[:,kconv1:kconv+kact] = localpart(V)[:,kconv1:kconv+kold]*Eig_vec[1:kold, 1:kact]
                    DV[kconv1:kconv+kact, :] = Eig_vec_p'*DV[kconv1:kconv+kold, :]
                    if  displ > 5  #can be expensive 
                        # @printf("Refinement: n=%i, kact=%i, kconv=%i,", n, kact, kconv)
                        # orth_err = norm(V[:,kconv1:kconv+kact]'*V[:,kconv1:kconv+kact] - Matrix{Float64}(I, kact, kact))
                        orth_err_local = zeros(Float64, kact, kact)
                        elapsedTime["Inner_prod"] += @elapsed begin
                        mul!(orth_err_local, DV[kconv1:kconv+kact, :], DV[kconv1:kconv+kact, :]')
                        MPI.Allreduce!(orth_err_local, +, comm_row)
                        MPI.Allreduce!(orth_err_local, +, comm_col)
                        end
                        elapsedTime["Inner_prod_n"] += 1
                        orth_err = norm(orth_err_local - Matrix{Float64}(I, kact, kact))
                        if  orth_err > 1e-10
                            if rank == 0
                                @error("After RR refinement: orth-error = %e\n", orth_err)
                            end
                        end
                    end
                    # W[:,1:kact]=W[:,1:kold]*Eig_vec[1:kold,1:kact]
                    # @spawnat W.pids[i] localpart(W)[:,1:kact] = localpart(W)[:,1:kold]*Eig_vec[1:kold, 1:kact]
                    DW[1:kact,:] = Eig_vec_p'*DW[1:kold,:]
                end 
                
                beta1 = maximum(broadcast(abs, d_eig))
                #--deflation and restart may make beta1 too small, (the active subspace
                #--dim is small at the end), so use beta1 only when it is larger.     
                if  beta1 > maxritz
                    maxritz = beta1 
                end
                tolr = tol*maxritz     

                # test convergence     
                ec = 0    #ec conunts # of converged eigenpair at current step

                # conv_cpu0=cputime
                # CPUtic()
                # check convergence only for the smallest blk # of Ritz pairs. 
                # i.e., check the first blk Ritz values (since d_eig is in increasing order)
                kconv0 = kconv
                for jc = 1:blk  
                    
                    rho = d_eig[jc]
                    elapsedTime["Norm"] += @elapsed begin
                    # r = W[:, jc]  - rho*V[:,kconv0+jc]
                    # normr = norm(r)
                    normr = zeros(1)
                    normr[1] = norm(DW[jc,:]-rho*DV[kconv0+jc,:],2)^2
                    MPI.Allreduce!(normr, +, comm_row)
                    MPI.Allreduce!(normr, +, comm_col)
                    normr[1] = sqrt(normr[1])
                    end
                    elapsedTime["Norm_n"] += 1
                    if  displ >= 4
                        if rank == 0
                            @printf(" n = %i,  rho= %e,  rnorm= %e\n", n, rho, normr[1])
                        end
                    end
                    swap = false

                    if  longlog == 1
                        historyhere = zeros(1,4)
                        historyhere[1] = iter_total
                        historyhere[2] = MVprod
                        historyhere[3] = normr[1]
                        historyhere[4] = rho
                        history = vcat(history, historyhere)
                        nlog = nlog+1
                    end
                    
                    if  normr[1] < tolr
                        kconv = kconv +1
                        kconv1 = kconv +1
                        ec=ec+1
                        if  displ >= 1
                            if rank == 0
                                @printf("#%i converged in %i steps, ritz(%3i)=%e, rnorm= %6.4e\n", kconv, iter_total, kconv, rho, normr[1])
                            end
                        end
                        evals[kconv] = rho
                        resnrm[kconv] = normr[1]
                        
                        #
                        ##--compare with converged eigenvalues (sort in increasing order)
                        #
                        # determine to which position we should move up the converged eigenvalue
                        imove = kconv - 1
                        while  imove > 0
                            if rho < evals[imove] 
                                imove = imove -1 
                            else
                                break
                            end
                        end
                        imove = imove+1  #the position to move up the current converged pair
                        if  imove < kconv  
                            swap = true  
                            if  displ > 3
                                if rank == 0
                                    @printf(" ==> swap %3i  upto %3i\n", kconv, imove)
                                end
                            end
                            # vtmp =  V[:,kconv]
                            # for i = kconv:-1:imove+1
                            #     V[:,i]=V[:,i-1]
                            #     evals[i]=evals[i-1]
                            # end
                            # V[:,imove]=vtmp
                            
                            vtmp = DV[kconv,:]
                            DV[imove+1:kconv,:] = DV[imove:kconv-1,:]
                            DV[imove,:] = vtmp
                            evals[imove+1:kconv] = evals[imove:kconv-1]
                            evals[imove]=rho
                        end

                        if (kconv >= nwant && !swap && blk>1 ) || (kconv >= nwant+kmore)
                            if  displ > 1
                                if rank == 0
                                    @printf("The converged eigenvalues and residual_norms are:\n")
                                    for i = 1:kconv
                                        @printf("  eigval(%3i) = %11.8e,   resnrm(%3i) = %8.5e \n", i, evals[i], i, resnrm[i])
                                    end
                                end
                            end
                            # change to output V= V(:, 1:kconv); later 
                            # eigV = Matrix{Float64}(V[:, 1:kconv])
                            # eigV = DV[1:kconv, :]
                            
                            if  displ > 0 #these output info may be useful for profiling
                                if rank == 0
                                    @printf("#converged eig=%i,  #wanted eig=%i,  #iter=%i, kinner=%i, kouter=%i\n", kconv,  nwant,  iter_total, kinner, kouter)

                                    @printf(" info of the eigenproblem and solver parameters :  dim=%i\n", dim)
                                    @printf(" polym=%i,  blk=%i,  vomax=%i,  vimax=%i,  n=%i, augment=%i, tol=%4.2e\n",polym, blk, vomax, vimax, n, augment, tol)
                                    @printf(" ORTH-cpu=%6.4e, ORTH-flops=%i,  ORTH-flops/dim=%6.4e\n",orth_cputotal, orth_flopstotal,  orth_flopstotal/dim)
                                    @printf(" mat-vect-cpu=%6.4e  #mat-vect-prod=%i,  mat-vec-cpu/#mvprod=%6.4e\n",MVcpu, MVprod, MVcpu/MVprod)
                                    
                                    
                                    # cputotal = CPUtoq()
                                    # @printf(" filt_MVcpu=%6.4e, filt_non_mv_cpu=%6.4e, refinement_cpu=%6.4e\n",filt_mv_cput, filt_non_mv_cput, refinement_cputotal)
                                    # @printf(" CPU%%: MV=%4.2f%%(filt_MV=%4.2f%%), ORTH=%4.2f%%, refinement=%4.2f%%\n", MVcpu/cputotal*100, filt_mv_cput/cputotal*100, orth_cputotal/cputotal*100, refinement_cputotal/cputotal*100)
                                    # @printf("       other=%4.2f%% (filt_nonMV=%4.2f%%, Hn_cpu=%4.2f%%, conv_cpu=%4.2f%%)\n",(cputotal-MVcpu-orth_cputotal-refinement_cputotal)/cputotal*100,
                                            # filt_non_mv_cput/cputotal*100, Hn_cpu/cputotal*100, conv_cpu/cputotal*100)
                                    # @printf(" TOTAL CPU seconds = %e\n", cputotal)
                                end
                            end
                            
                            returnhere = 1
                            break
                            # return evals, eigV, kconv, history
                        end
                    else
                        break # exit when the first non-convergent Ritz pair is detected
                    end
                end

                if returnhere == 1
                    break
                end
                
                if  ec > 0
                    # W[:,1:kact-ec] = W[:,ec+1:kact] 
                    DW[1:kact-ec,:] = DW[ec+1:kact,:]
                    # update the current active subspace dimension 
                    kact = kact - ec   
                end
                
                # save only the non-converged Ritz values in Hn
                Hn[1:kact,1:kact] = Eig_val[ec+1:kact+ec,ec+1:kact+ec]     
                
                #
                # update lower_nwb  (ritz_nwb) to be the mean value of d_eig. 
                # from many practices this choice turn out to be efficient,
                # but there are other choices for updating the lower_nwb.
                # (the convenience in adapting this bound without extra computation
                # shows the reamrkable advantage in integrating the Chebbyshev 
                # filtering in a Davidson-type method)
                #
                #low_nwb = median(d_eig(1:max(1, length(d_eig)-1)));
                low_nwb = median(d_eig)
                # lowb = minimum(d_eig)
                
                #
                # determine if need to do outer restart (only n need to be updated)
                #
                if  n + blk > vomax && opts["do_outer"]
                    nold = n
                    n = max(kconv+blk, vomax - 2*blk)
                    if  displ > 4
                        if rank == 0
                            @printf("--> Outer-restart: n from %i downto %i, vomax=%i, vimax=%i, kact=%i\n", nold, n, vomax, vimax, kact)
                        end
                    end
                    kact = n-kconv
                    kouter = kouter+1  #used only for counting
                end
                # conv_cpu = conv_cpu + CPUtoq()

                # @printf("%i, %i\n", iter_total, itmax)
            
            end
        end

        if  iter_total > itmax
            #
            # the following should rarely happen unless the problem is
            # extremely difficult (highly clustered eigenvalues)
            # or the vomax or vimax is set too small
            #
            if rank == 0
                @printf("***** itmax=%i, it_total=%i\n", itmax, iter_total)
                @warn("***** bchdav.jl:  Maximum iteration exceeded\n")
                @printf("***** nwant=%i, kconv=%i, vomax=%i\n", nwant, kconv, vomax)
                @warn("***** it could be that your vimax/blk, or vomax, or polym is too small")
            end
        end

    

    end

    elapsedTime["Pre_loop"] = elapsedTime["total"] - elapsedTime["main_loop"]

    if verb
        if rank == 0
            @printf("\n")
            @printf("+++++++++++++++++++++ runtime details for debugging +++++++++++++++++++++++++\n")
            @printf("total runtime:                          %.2e \n", elapsedTime["total"])
            @printf("runtime preloop                         %.2e \n", elapsedTime["Pre_loop"])
            @printf("runtime main_loop:                      %.2e \n", elapsedTime["main_loop"])
            if verb
                @printf("   runtime DGKS:                    %.2e / %i \n", elapsedTime["DGKS"], elapsedTime["DGKS_n"])
                @printf("   runtime Cheb_filter_scal:        %.2e / %i \n", elapsedTime["Cheb_filter_scal"], elapsedTime["Cheb_filter_scal_n"])
                @printf("   runtime SpMM:                    %.2e / %i \n", elapsedTime["SpMM"], elapsedTime["SpMM_n"])
                @printf("   runtime Inner_prod:              %.2e / %i \n", elapsedTime["Inner_prod"], elapsedTime["Inner_prod_n"])
                @printf("   runtime Norm:                    %.2e / %i \n", elapsedTime["Norm"], elapsedTime["Norm_n"])   
                @printf("   runtime Hn:                      %.2e / %i \n", elapsedTime["Hn"], elapsedTime["Hn_n"])            
            end
            @printf("+++++++++++++++++++++++++++++++ END +++++++++++++++++++++++++++++++++++++++++\n")
        end
    end

    
    return evals, DV[1:kconv, :], kconv, history, elapsedTime

end

function DGKS(X, ids, V, N, comm_row, comm_col)
    epsbig = 2.22045e-16  
    reorth=0.717
    one=1.0e0
    colv, ndim = size(V)
    colx = length(ids)
    orth_flops = 0
    vrank = 0

    for k = 1:colv
        nrmv = zeros(Float64, 1)
        nrmv[1] = norm(V[k,:],2)^2
        orth_flops += ndim
        MPI.Allreduce!(nrmv, +, comm_row)
        MPI.Allreduce!(nrmv, +, comm_col)
        nrmv[1] = sqrt(nrmv[1])
        if (nrmv[1] < epsbig*sqrt(N))
            continue 
        end

        if nrmv[1] <= 2*epsbig || nrmv[1] >= 300
            V[k, 1:ndim] = V[k, 1:ndim]/nrmv[1]
            orth_flops += ndim
            nrmv[1] = one
        end

        h = zeros(Float64, colx+vrank, 1)
        if colx == 0
            mul!(h, V[1:vrank, :], V[k:k,:]')
        else
            mul!(h, vcat(X[ids, :], V[1:vrank, :]), V[k:k, :]')
        end
        MPI.Allreduce!(h, +, comm_row)
        MPI.Allreduce!(h, +, comm_col)
        if colx == 0
            mul!(V[k:k, :], h', V[1:vrank, :], -1.0, 1.0)
        else
            mul!(V[k:k, :], h', vcat(X[ids, :], V[1:vrank, :]), -1.0, 1.0)
        end
        # V(1:ndim,k) = V(1:ndim,k) -  [X, V(1:ndim,1:vrank)]*h;
        orth_flops += ndim*((colx+vrank)*2 + 1)

        nrmproj = zeros(Float64, 1)
        nrmproj[1] = norm(V[k,:],2)^2
        orth_flops=orth_flops+ndim
        MPI.Allreduce!(nrmproj, +, comm_row)
        MPI.Allreduce!(nrmproj, +, comm_col)
        nrmproj[1] = sqrt(nrmproj[1])

        if nrmproj[1] > reorth*nrmv[1]
            vrank = vrank +1
            if abs(nrmproj[1] - one) > epsbig
                V[k,:] = V[k,:]/nrmproj[1]
                orth_flops += ndim
            end
            if vrank != k
                V[vrank, :] = V[k, :]
            end
        else
            nrmv[1] = nrmproj[1];      

            h = zeros(Float64, colx+vrank, 1)
            if colx == 0
                mul!(h, V[1:vrank, :], V[k:k, :]')
            else
                mul!(h, vcat(X[ids, :], V[1:vrank, :]), V[k:k, :]')
            end
            MPI.Allreduce!(h, +, comm_row)
            MPI.Allreduce!(h, +, comm_col)
            if colx == 0
                mul!(V[k:k, :], h', V[1:vrank, :], -1.0, 1.0)
            else
                mul!(V[k:k, :], h', vcat(X[ids, :], V[1:vrank, :]), -1.0, 1.0)
            end
            orth_flops += ndim*((colx+vrank)*2 + 1)
            
            nrmproj[1] = norm(V[k, :],2)^2
            orth_flops=orth_flops+ndim
            MPI.Allreduce!(nrmproj, +, comm_row)
            MPI.Allreduce!(nrmproj, +, comm_col)
            nrmproj[1] = sqrt(nrmproj[1])    
            if nrmproj[1] > reorth*nrmv[1]  && nrmproj[1] >= sqrt(N)*epsbig
                vrank = vrank +1
                if abs(nrmproj[1] - one) > epsbig
                    V[k, :] = V[k, :]/nrmproj[1]
                    orth_flops += ndim
                end
                if vrank != k
                    V[vrank, :] = V[k, :]
                end	
            else
                
                # fail the 2nd reorthogonalization test,
                #    V(:,k) is numerically in V(:, 1:vrank),
                # do not increase vrank, but go for the next k 
                
            end
        end
    end

    if vrank > 0
        V = V[1:vrank, :]
    else #recursively call DGKS_blk to make output V not a zero vector
        # fprintf('DGKS: # of columns replaced by random vectors =%i\n', colv-vrank); 
        V[vrank+1:colv, :] = DGKS(vcat(X[ids, :], V[1:vrank, :]), 1:length(ids)+vrank, randn(colv-vrank, ndim), N, comm_row, comm_col)
    end

    return V, orth_flops
end

function SpMM_A_0(V, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    blk = size(V, 1)
    W = zeros(Float64, blk, info_cols_dist[rank_col*comm_size_sq+rank_row+1])
    W_gather = zeros(Float64, size(A, 1), blk)
    V_gather_T = zeros(Float64, size(A, 2), blk)
    local_info_cols = info_cols_dist[rank_row*comm_size_sq+1:rank_row*comm_size_sq+comm_size_sq]
    V_gather = Array{Float64}(undef, (blk, sum(local_info_cols)))
    _counts = vcat([blk for i in 1:length(local_info_cols[:])]', info_cols_dist[rank_row*comm_size_sq+1:(rank_row+1)*comm_size_sq]')
    V_gather_vbuf = VBuffer(V_gather, vec(prod(_counts, dims=1)))
    W_gather_T = zeros(Float64, blk, size(A,1))
    _counts = vcat([blk for i in 1:length(local_info_cols[:])]', info_cols_dist[rank_col*comm_size_sq+1:(rank_col+1)*comm_size_sq]')
    W_gather_T_vbuf = VBuffer(W_gather_T, vec(prod(_counts, dims=1)))

    MPI.Allgatherv!(V, V_gather_vbuf, comm_col)

    mul!(W_gather, A, V_gather', 1.0, 0.0)

    MPI.Reduce!(W_gather, +, root, comm_row)
    W_gather_T = W_gather'

    W = MPI.Scatterv!(W_gather_T_vbuf, zeros(Float64, size(W)), root, comm_row)

    MPI.Allgatherv!(W, W_gather_T_vbuf, comm_row)
    
    mul!(V_gather_T, E', W_gather_T', 1.0, 0.0)

    MPI.Reduce!(V_gather_T, +, root, comm_col)
    V_gather = V_gather_T'

    Y = MPI.Scatterv!(V_gather_vbuf, zeros(Float64, size(V)), root, comm_col)
    return Y    
end

function SpMM_A(X, Y, A, X_gather, X_gather_vbuf, Y_gather, Y_gather_T, Y_gather_T_vbuf, root, comm_row, comm_col)
    MPI.Allgatherv!(X, X_gather_vbuf, comm_col)

    mul!(Y_gather, A, X_gather', 1.0, 0.0)

    MPI.Reduce!(Y_gather, +, root, comm_row)
    Y_gather_T = Y_gather'

    Y = MPI.Scatterv!(Y_gather_T_vbuf, zeros(Float64, size(Y)), root, comm_row)
end

function Cheb_filter_scal(deg, X, Y, A, E, X_gather, X_gather_T, X_gather_vbuf, Y_gather, Y_gather_T, Y_gather_T_vbuf, root, comm_row, comm_col)
# deg should always be an odd number

    for kk = 1:deg
        if mod(kk, 2) == 1
            MPI.Allgatherv!(X, X_gather_vbuf, comm_col)
    
            mul!(Y_gather, A, X_gather', 1.0, 0.0)
    
            MPI.Reduce!(Y_gather, +, root, comm_row)
            Y_gather_T = Y_gather'
    
            Y = MPI.Scatterv!(Y_gather_T_vbuf, zeros(Float64, size(Y)), root, comm_row)
        else
            MPI.Allgatherv!(Y, Y_gather_T_vbuf, comm_row)
    
            mul!(X_gather_T, A', Y_gather_T', 1.0, 0.0)
    
            MPI.Reduce!(X_gather_T, +, root, comm_col)
            X_gather = X_gather_T'
    
            X = MPI.Scatterv!(X_gather_vbuf, zeros(Float64, size(X)), root, comm_col)
        end
    end
    
    MPI.Allgatherv!(Y, Y_gather_T_vbuf, comm_row)
    
    mul!(X_gather_T, E', Y_gather_T', 1.0, 0.0)

    MPI.Reduce!(X_gather_T, +, root, comm_col)
    X_gather = X_gather_T'

    X = MPI.Scatterv!(X_gather_vbuf, zeros(Float64, size(X)), root, comm_col)
    
end

function split_count(N::Integer, n::Integer)
    q,r = divrem(N, n)
    return [i <= r ? q+1 : q for i = 1:n]
end

function split_count_local(N::Integer, n::Integer)
    counts = zeros(Int64, n*n)
    counts1 = split_count(N, n)
    info = zeros(Int64, 1, n*n*n)
    for i in 1:n
        counts2 = split_count(counts1[i], n)
        counts[(i-1)*n+1:i*n] = counts2
        for j in 1:n
            info[(i-1)*n*n+(j-1)*n+1:(i-1)*n*n+j*n] = counts2
        end
    end
    return counts, info
end

function findnz_local(A, comm_size_sq, nnz)
    N = size(A,1)
    counts = split_count(N, comm_size_sq)
    I = zeros(Int64, nnz)
    J = zeros(Int64, nnz)
    V = zeros(Float64, nnz)
    S = zeros(Int64, 2, comm_size_sq*comm_size_sq)
    col_start = col_end = 0
    row_start = row_end = 0
    counts_local = zeros(Int64, comm_size_sq, comm_size_sq)
    n = 0
    for j in 1:comm_size_sq
        col_start = j == 1 ? 1 : sum(counts[1:j-1])+1
        col_end = sum(counts[1:j])
        for i in 1:comm_size_sq
            row_start = i == 1 ? 1 : sum(counts[1:i-1])+1
            row_end = sum(counts[1:i])
            I1, J1, V1 = findnz(A[row_start:row_end, col_start:col_end])
            m = length(I1)
            I[n+1:n+m] = I1
            J[n+1:n+m] = J1
            V[n+1:n+m] = V1
            counts_local[i, j] = m
            S[1, (j-1)*comm_size_sq + i] = row_end - row_start + 1
            S[2, (j-1)*comm_size_sq + i] = col_end - col_start + 1
            n += m
        end
    end
    I, J, V, counts_local[:], S
end

# end # module


