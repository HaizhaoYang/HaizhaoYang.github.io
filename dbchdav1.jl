# module DBchdav


using Printf
using Elemental
using Statistics
using LowRankApprox
using Distributed
using DistributedArrays
@everywhere using LinearAlgebra
@everywhere using DistributedArrays: localpart

export dbchdav



function dbchdav1(varargin...;verb=false)
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
    global elapsedTime = Dict("total"=>0.0, "dCheb_filter"=>0.0, "dCheb_filter_n"=>0.0, "dCheb_filter_scal"=>0.0,
                "dCheb_filter_scal_n"=>0.0, "duser_Hx"=>0.0, "duser_Hx_n"=>0.0, "Matrix_Float64"=>0.0, "Matrix_Float64_n"=>0.0,
                "@distributed_locally_assigning"=>0.0, "@distributed_locally_assigning_n"=>0.0, "@distributed_matvec"=>0.0,
                "@distributed_matvec_n"=>0.0, "@distributed_movecols"=>0.0, "@distributed_movecols_n"=>0.0, "@distributed_sumup"=>0.0,
                "@distributed_sumup_n"=>0.0, "@distribute"=>0.0, "@distribute_n"=>0.0, "DGKS_blk"=>0.0, "DGKS_blk_n"=>0.0, 
                "dzeros"=>0.0, "dzeros_n"=>0.0, "LANCZ_bound"=>0.0, "LANCZ_bound_n"=>0.0, "distribute"=>0.0, "distribute_n"=>0.0,
                "computing_residual"=>0.0, "computing_residual_n"=>0.0, "dDGKS_blk"=>0.0, "dDGKS_blk_n"=>0.0, 
                "dDGKS_@distributed_locally_assigning"=>0.0, "dDGKS_@distributed_locally_assigning_n"=>0.0, "dDGKS_@distributed_gs"=>0.0,
                "dDGKS_@distributed_gs_n"=>0.0, "dDGKS_@distributed_matscalar"=>0.0, "dDGKS_@distributed_matscalar_n"=>0.0,
                "dDGKS_@distributed_matvec"=>0.0, "dDGKS_@distributed_matvec_n"=>0.0, "dDGKS_@distributed_movecols"=>0.0,
                "dDGKS_@distributed_movecols_n"=>0.0, "dDGKS_@distributed_sumup"=>0.0, "dDGKS_@distributed_sumup_n"=>0.0,
                "dDGKS_@distributed_computingnorm"=>0.0, "dDGKS_@distributed_computingnorm_n"=>0.0
    )

    elapsedTime["total"] = @elapsed begin

        #
        # Process inputs and do error-checking
        #

        # if no input arguments, return help.
        if  length(varargin) == 0
            @error("No papameters were passed to bchdav!")
            return [], [], [], []
        end

        # @printf("current processor: %i \n", myid())
        # if isa(varargin[1], Array)
        # DA_operator = distribute(varargin[1], procs = wkers, dist = [nwkers, 1])
        DA_operator = varargin[1]
        wkers = DA_operator.pids
        nwkers = length(wkers)
        pid2indices = Dict()
        for i = 1:length(DA_operator.pids)
            pid = DA_operator.pids[i]
            pid2indices[pid] = DA_operator.indices[i]
        end
        dim = size(DA_operator,1)
        if dim != size(DA_operator,2)
            @error("The input numeric matrix A must be a square matrix")
        end
        if dim <= 300
            @warn("small dimension problem, use eigen instead")
            evals, eigV = eigen(Matrix(DA_operator))
            # evals = diag(evals);
            # if (nargout >2), kconv = dim; end
            # if (nargout >3), history=[]; end
            return evals, eigV, dim, []
        end
        Anumeric = 1
        # else
        #   A_operator = fcnchk(varargin{1});
        #   Anumeric = 0;
        #   dim = varargin{2};  # when A is a string, need to explicitly
        #                       # input the dimension of A
        #   if (~isnumeric(dim) | ~isreal(dim) | round(dim)~=dim | dim <1)
        #     error('A valid matrix dimension is required for the input string function')
        #   end
        # end 

        if length(varargin) < 3-Anumeric
            nwant = min(dim,6)
        else
            nwant = varargin[3-Anumeric]
            if !isa(nwant, Number) || nwant < 1 || nwant > dim
                @warn("invalid # of wanted eigenpairs. use default value")
                nwant = min(dim,6)
            end
        end


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

        if length(varargin) >= (4-Anumeric)
            user_opts = varargin[4-Anumeric]
            if !isa(user_opts,Dict)
                @error("Options must be a dictionary. (note bchdav does not need mode)")
            else
                # overwrite default options by user input options
                # opts = merge(user_opts, opts) 
                for key in keys(user_opts)
                    opts[key] = user_opts[key]
                    
                end       
            end
        end  

        elapsedTime["chksym"] = @elapsed begin
        if opts["chksym"]
            # if Anumeric > 0
            #if  !issymmetric(DA_operator)
            #    @error("input matrix is not symmetric/Hermitian")
            #end
            # end
        end
        end

        # save opt values in local variables 
        blk = opts["blk"];  filter=opts["filter"];  polym=opts["polym"];  tol=opts["tol"];
        vomax=opts["vomax"];  vimax=opts["vimax"];  itmax=opts["itmax"];  
        augment=opts["augment"];  kmore=opts["kmore"];  displ=opts["displ"]; 

        if  haskey(opts, "v0")
            sizev0 = size(opts["v0"],2)
            if  sizev0 < blk 
                @printf("*** input size(v0,2)=%i, blk=%i, augment %i random vectors\n",
                        sizev0, blk, blk-sizev0)
                opts["v0"] = hcat(opts["v0"], rand(dim,blk-sizev0))

            end
        else
            opts["v0"] = rand(dim, blk)
            sizev0 = blk
        end

        elapsedTime["distribute"] += @elapsed begin
        dV0 = distribute(opts["v0"], procs=wkers, dist=[nwkers, 1])
        end
        elapsedTime["distribute_n"] += 1

        if opts["do_outer"] 
            vomaxtmp = max(min(nwant + 6*blk, nwant+30), ceil(nwant*1.2))
            if  vomax < vomaxtmp 
                @printf("--> Warning: vomax=%i, nwant=%i, blk=%i\n", vomax, nwant, blk)
                vomax = vomaxtmp
                @printf("--> Warnning: increase vomax to %i\n",vomax)
            end  
            if  vimax > vomax
                @printf("--> Warning:  (vimax > vomax)  vimax=%i, vomax=%i\n", vimax, vomax)
                vimax = max(min(6*blk, nwant), ceil(nwant/4))  #note vomax > nwant
                @printf("--> reduce vimax to %i\n", vimax)
            end
        end
        if  vimax < 5*blk 
            @printf("--> Warning:  (vimax < 5*blk)  vimax=%i, blk=%i\n", vimax, blk)
            if  opts["adjustvimax"] 
                vimax = 5*blk
                @printf("--> increase vimax to %i\n", vimax)        
            elseif 3*blk > vimax
                vimax = 3*blk
                @printf("--> adjust vimax to %i\n", vimax)
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
        elapsedTime["dzeros"] += @elapsed begin
        V = dzeros((dim, vomax), wkers, [nwkers, 1])
        W = dzeros((dim, trunc(Int64, vimax)), wkers, [nwkers, 1]) #note vimax<vomax, inner restart also saves memory
        end
        elapsedTime["dzeros_n"] += 1
        Hn = zeros(trunc(Int64, vimax), trunc(Int64, vimax))
        evals = zeros(nwant)   
        resnrm = zeros(nwant,1)


        #  get the very important filtering upper bound. 
        #  if there is a user input upb, make sure the input upb is an upper bound, 
        #  otherwise convergence can be very slow. (safer to just let the LANCZ_bound() estimate an upper bound without a user input upb)
        #  an estimated lower bound can also be obtained via LANCZ_bound()

        lancz_step = 4
        # if  Anumeric > 0
        elapsedTime["LANCZ_bound"] += @elapsed begin
        # upb, low_nwb, lowb, maxritz = LANCZ_bound(dim, lancz_step, DA_operator)  
        upb = opts["upb"]
        lowb = opts["lwb"]
        low_nwb = lowb + (upb-lowb)/10
        maxritz = upb
        end
        elapsedTime["LANCZ_bound_n"] += 1
        # else
        #     upb, low_nwb, lowb, maxritz = LANCZ_bound(dim, lancz_step, A_operator)
        # end
        if haskey(opts, "upb")
            if opts["upb"] < upb
                @warn("user input upperbound may be too small, may NOT converge!!")
                upb = opts["upb"]   #still use the user input upb, run at your own risk
            end
        end

        if haskey(opts, "lwb")
            if opts["lwb"] > lowb
                @warn("user input upperbound may be too large, may NOT converge!!")
                lowb = opts["lwb"]   #still use the user input lwb, run at your own risk
            end
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

        # V[:,1:blk], rt = qr(opts["v0"][:,1:blk]) 

        elapsedTime["pqr"] = @elapsed begin
        Q, rt = pqr(opts["v0"][:,1:blk])
        end
        elapsedTime["distribute"] += @elapsed begin
        dQ = distribute(Q, procs=V.pids, dist=[length(V.pids), 1])
        end
        elapsedTime["distribute_n"] += 1
        elapsedTime["@distributed_locally_assigning"] = @elapsed begin
        @sync @distributed for i = 1:nwkers
            # @spawnat V.pids[i] localpart(V)[:,1:blk] = Q[V.indices[i][1],:]
            # localpart(V)[:,1:blk] = Q[pid2indices[myid()][1], :]
            localpart(V)[:,1:blk] = localpart(dQ)
        end
        end
        elapsedTime["@distributed_locally_assigning_n"] += 1

        #V[:,1:blk] = F.Q[:,1:blk]
        #rt = F.R
        ec = 0
        kinit = blk
        
        elapsedTime["main_loop"] = @elapsed begin
            while  iter_total <= itmax 
                
                iter_total = iter_total +1
                if  displ >= 4
                    @printf(" low_nwb =%e\n", low_nwb)
                end 
                
                elapsedTime["dzeros"] += @elapsed begin
                dVtmp = dzeros((dim, blk), wkers, [nwkers, 1])
                end
                elapsedTime["dzeros_n"] += 1
                if  ec > 0  &&  kinit + ec <= sizev0
                    if  displ > 4
                        @printf("---> using column [%i : %i] of initial vectors\n",kinit+1, kinit+ec) 
                    end
                    # Vtmp = hcat(Matrix{Float64}(opts["v0"][:,kinit+1:kinit+ec]), Matrix{Float64}(V[:,kconv1:kconv+blk-ec]))
                    elapsedTime["@distributed_locally_assigning"] += @elapsed begin
                    @sync @distributed for i = 1:nwkers
                        localpart(dVtmp)[:,:] = hcat(localpart(dV0)[:,kinit+1:kinit+ec], localpart(V)[:,kconv1:kconv+blk-ec])
                    end
                    end
                    elapsedTime["@distributed_locally_assigning_n"] += 1 
                    kinit = kinit + ec
                else
                    # Vtmp = Matrix{Float64}(V[:,kconv1:kconv+blk])
                    elapsedTime["@distributed_locally_assigning"] += @elapsed begin
                    @sync @distributed for i = 1:nwkers
                        localpart(dVtmp)[:,:] = localpart(V)[:,kconv1:kconv+blk]
                    end
                    end
                    elapsedTime["@distributed_locally_assigning_n"] += 1
                end

                elapsedTime["Matrix_Float64"] += @elapsed begin
                Vtmp = Matrix{Float64}(dVtmp)
                end
                elapsedTime["Matrix_Float64_n"] += 1
                
                if  filter == 1
                    #case 1  % the default chebyshev filter (non-scaled)
                    
                    # if  Anumeric > 0
                    #Vtmp = dCheb_filter(V(:,kconv1:kconv+blk), polym, low_nwb, upb, augment);
                    elapsedTime["dCheb_filter"] += @elapsed begin
                    Vtmp = dCheb_filter(DA_operator, Vtmp, polym, low_nwb, upb, augment)  
                    end
                    elapsedTime["dCheb_filter_n"] += size(Vtmp, 2)
                    # else
                    #     Vtmp = dCheb_filter(Vtmp, polym, low_nwb, upb, augment, A_operator)
                    # end
                elseif filter == 2
                    #case 2  % with scaling, need an extra bound (not as convenient as above)
                    # if  Anumeric > 0
                    elapsedTime["dCheb_filter_scal"] += @elapsed begin
                    # Vtmp = dCheb_filter_scal(DA_operator, Vtmp, polym, low_nwb, upb, lowb, augment)

                    mvcput = 0.0

                    e = (upb - low_nwb)/2.0
                    center= (upb+low_nwb)/2.0
                    sigma = e/(lowb - center)
                    tau = 2.0/sigma

                    # y, mvcpu0 = duser_Hx(DA_operator, Vtmp)
                    mvcpu0 = @elapsed begin
                        Dy = dzeros((size(DA_operator,1), size(Vtmp,2)), DA_operator.pids, [length(DA_operator.pids), 1])
                        @sync for ii in 1:length(DA_operator.pids)
                            @spawnat DA_operator.pids[ii] localpart(Dy)[:,:] = localpart(DA_operator)*Vtmp
                        end
                        y = Matrix{Float64}(Dy)
                    end
                    elapsedTime["duser_Hx"] += mvcpu0
                    elapsedTime["duser_Hx_n"] += size(Vtmp,2)
                    mvcput = mvcput + mvcpu0
                    y = (y - center*Vtmp) * (sigma/e)

                    for i = 2:polym-1
                        sigma_new = 1.0 /(tau - sigma)
                        # ynew, mvcpu0 = duser_Hx(DA_operator, y)
                        mvcpu0 = @elapsed begin
                            Dynew = dzeros((size(DA_operator,1), size(y,2)), DA_operator.pids, [length(DA_operator.pids), 1])
                            @sync for ii in 1:length(DA_operator.pids)
                                @spawnat DA_operator.pids[ii] localpart(Dynew)[:,:] = localpart(DA_operator)*y
                            end
                            ynew = Matrix{Float64}(Dynew)
                        end
                        elapsedTime["duser_Hx"] += mvcpu0
                        elapsedTime["duser_Hx_n"] += size(y,2)
                        mvcput = mvcput + mvcpu0
                        ynew = (ynew - center*y)*(2.0*sigma_new/e) - (sigma*sigma_new)*Vtmp
                        Vtmp = y
                        y = ynew
                        sigma = sigma_new
                    end

                    # ynew, mvcpu0 = duser_Hx(DA_operator, y)
                    mvcpu0 = @elapsed begin
                        Dynew = dzeros((size(DA_operator,1), size(y,2)), DA_operator.pids, [length(DA_operator.pids), 1])
                        @sync for ii in 1:length(DA_operator.pids)
                            @spawnat DA_operator.pids[ii] localpart(Dynew)[:,:] = localpart(DA_operator)*y
                        end
                        ynew = Matrix{Float64}(Dynew)
                    end
                    elapsedTime["duser_Hx"] += mvcpu0
                    elapsedTime["duser_Hx_n"] += size(y,2)
                    mvcput = mvcput + mvcpu0
                    sigma_new = 1.0 /(tau - sigma)
                    # default return unless augment==2 or 3.
                    Vtmp = (ynew - center*y)*(2.0*sigma_new/e) - (sigma*sigma_new)*Vtmp
                    
                    end
                    elapsedTime["dCheb_filter_scal_n"] += size(Vtmp, 2)



                    
                    # else
                    #     Vtmp = dCheb_filter_scal(Vtmp, polym, low_nwb, upb, lowb, augment,A_operator)
                    # end
                    
                else
                    @error("selected filter is not available")
                end
                
                #
                # make t orthogonal to all vectors in V
                #
                n = kconv + kact
                # orth_cpu = cputime
                orth_cputotal += @elapsed begin
                    elapsedTime["distribute"] += @elapsed begin
                    dVtmp = distribute(Vtmp, procs=V.pids, dist=[length(V.pids), 1])
                    end
                    elapsedTime["distribute_n"] += 1
                    elapsedTime["dDGKS_blk"] += @elapsed begin
                    Vtmp, orth_flops = dDGKS_blk(V, 1:n, dVtmp)
                    end
                    elapsedTime["dDGKS_blk_n"] += 1
                    # elapsedTime["DGKS_blk"] += @elapsed begin
                    # Vtmp, orth_flops = DGKS_blk(Matrix{Float64}(V[:,1:n]), Vtmp)
                    # end
                    # elapsedTime["DGKS_blk_n"] += 1
                    elapsedTime["distribute"] += @elapsed begin
                    dVtmp = distribute(Vtmp, procs=V.pids, dist=[length(V.pids), 1])
                    end
                    elapsedTime["distribute_n"] += 1
                end
                orth_flopstotal = orth_flopstotal + orth_flops
                kb = size(Vtmp,2)
                n1 = n+1
                kact = kact + kb
                n = kconv + kact   
                # V[:, n1:n] = Vtmp
                elapsedTime["@distributed_locally_assigning"] += @elapsed begin
                @sync @distributed for i = 1:length(V.pids)
                    # pid = V.pids[i]
                    # @spawnat pid localpart(V)[:,n1:n] = Vtmp[V.indices[i][1],:]
                    localpart(V)[:,n1:n] = localpart(dVtmp)
                end
                end
                elapsedTime["@distributed_locally_assigning_n"] += 1
                
                #
                # compute new matrix-vector product.
                #
                # if  Anumeric > 0
                elapsedTime["duser_Hx"] += @elapsed begin
                # nW,_ =  duser_Hx(DA_operator, Matrix{Float64}(V[:, n1:n]))
                    sV = Matrix{Float64}(V[:, n1:n])
                    DnW = dzeros((dim, size(sV,2)), wkers, [nwkers, 1])
                    @sync @distributed for ii in 1:nwkers
                        # @spawnat wkers[ii] localpart(DnW)[:,:] = localpart(DA_operator)*sV
                        localpart(DnW)[:,:] = localpart(DA_operator)*sV
                    end
                    nW = Matrix{Float64}(DnW)
                end
                elapsedTime["duser_Hx_n"] += n-n1+1
                elapsedTime["distribute"] += @elapsed begin
                dnW = distribute(nW, procs=wkers, dist=[nwkers, 1])
                end
                elapsedTime["distribute_n"] += 1
                elapsedTime["@distributed_locally_assigning"] += @elapsed begin
                @sync @distributed for i = 1:length(W.pids)
                    # @spawnat W.pids[i] localpart(W)[:, kact-kb+1:kact] = sW[W.indices[i][1], :]
                    localpart(W)[:, kact-kb+1:kact] = localpart(dnW)
                end
                end
                elapsedTime["@distributed_locally_assigning_n"] += 1
                # else
                #     W[:, kact-kb+1:kact],_ =  duser_Hx(V[:, n1:n], A_operator)  
                # end

                
                # Hn_cpu0=cputime
                #
                # update Hn, compute only the active part (Hn does not include any deflated part)
                #
                # Hn[1:kact, kact-kb+1:kact] = V[:, kconv1:n]'* W[:, kact-kb+1:kact]
                Hn_cpu += @elapsed begin
                    elapsedTime["@distributed_sumup"] += @elapsed begin
                    Hn[1:kact, kact-kb+1:kact] = @distributed (+) for i = 1:length(V.pids)
                        # fetch(@spawnat V.pids[i] localpart(V)[:, kconv1:n]' * localpart(W)[:, kact-kb+1:kact])
                        localpart(V)[:, kconv1:n]' * localpart(W)[:, kact-kb+1:kact]
                    end
                    end
                    elapsedTime["@distributed_sumup_n"] += 1
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
                            @printf("==> Inner-restart: vimax=%i, n=%i, kact from %i downto %i \n", vimax, n, kact, ikeep)
                        end
                        kact = ikeep
                        n = kconv+kact   # should always keep n=kconv+kact
                        kinner = kinner+1  #used only for counting
                    end 
                    
                    # V[:,kconv1:kconv+kact] = V[:,kconv1:kconv+kold]*Eig_vec[1:kold,1:kact]
                    Eig_vec_p = Eig_vec[1:kold, 1:kact]
                    elapsedTime["@distributed_matvec"] += @elapsed begin
                    @sync @distributed for i = 1:length(V.pids)
                        # @spawnat V.pids[i] localpart(V)[:,kconv1:kconv+kact] = localpart(V)[:,kconv1:kconv+kold]*Eig_vec[1:kold, 1:kact]
                        localpart(V)[:, kconv1:kconv+kact] = localpart(V)[:,kconv1:kconv+kold]*Eig_vec_p
                    end
                    end
                    elapsedTime["@distributed_matvec_n"] += 1
                    if  displ > 5  #can be expensive 
                        @printf("Refinement: n=%i, kact=%i, kconv=%i,", n, kact, kconv)
                        # orth_err = norm(V[:,kconv1:kconv+kact]'*V[:,kconv1:kconv+kact] - Matrix{Float64}(I, kact, kact))
                        orth_err_local = zeros(kact, kact)
                        elapsedTime["@distributed_sumup"] += @elapsed begin
                        orth_err_local = @distributed (+) for i = 1:length(V.pids)
                            # fetch(@spawnat V.pids[i] localpart(V)[:,kconv1:kconv+kact]' * localpart(V)[:,kconv1:kconv+kact])
                            localpart(V)[:,kconv1:kconv+kact]' * localpart(V)[:, kconv1:kconv+kact]
                        end
                        end
                        elapsedTime["@distributed_sumup_n"] += 1
                        orth_err = norm(orth_err_local - Matrix{Float64}(I, kact, kact))
                        if  orth_err > 1e-10
                            @error("After RR refinement: orth-error = %e\n", orth_err)
                        end
                    end
                    # W[:,1:kact]=W[:,1:kold]*Eig_vec[1:kold,1:kact]
                    elapsedTime["@distributed_matvec"] += @elapsed begin
                        @sync @distributed for i = 1:length(W.pids)
                            # @spawnat W.pids[i] localpart(W)[:,1:kact] = localpart(W)[:,1:kold]*Eig_vec[1:kold, 1:kact]
                            localpart(W)[:,1:kact] = localpart(W)[:,1:kold]*Eig_vec_p
                        end
                    end
                    elapsedTime["@distributed_matvec_n"] += 1
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
                    elapsedTime["computing_residual"] += @elapsed begin
                    # r = W[:, jc]  - rho*V[:,kconv0+jc]
                    # normr = norm(r)
                    normr = @distributed (+) for i = 1:nwkers
                        norm(localpart(W)[:, jc] - rho*localpart(V)[:,kconv0+jc],2)^2
                    end
                    normr = sqrt(normr)
                    end
                    elapsedTime["computing_residual_n"] += 1
                    if  displ >= 4
                        @printf(" n = %i,  rho= %e,  rnorm= %e\n", n, rho, normr)
                    end
                    swap = false

                    if  longlog == 1
                        historyhere = zeros(1,4)
                        historyhere[1] = iter_total
                        historyhere[2] = MVprod
                        historyhere[3] = normr
                        historyhere[4] = rho
                        history = vcat(history, historyhere)
                        nlog = nlog+1
                    end
                    
                    if  normr < tolr
                        kconv = kconv +1
                        kconv1 = kconv +1
                        ec=ec+1
                        if  displ >= 1
                            @printf("#%i converged in %i steps, ritz(%3i)=%e, rnorm= %6.4e\n", kconv, iter_total, kconv, rho, normr)
                        end
                        evals[kconv] = rho
                        resnrm[kconv] = normr
                        
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
                                @printf(" ==> swap %3i  upto %3i\n", kconv, imove)
                            end
                            # vtmp =  V[:,kconv]
                            # for i = kconv:-1:imove+1
                            #     V[:,i]=V[:,i-1]
                            #     evals[i]=evals[i-1]
                            # end
                            # V[:,imove]=vtmp
                            
                            elapsedTime["dzeros"] += @elapsed begin
                            vtmp = dzeros((dim, 1), V.pids, [length(V.pids), 1])
                            end
                            elapsedTime["dzeros_n"] += 1
                            elapsedTime["@distributed_movecols"] += @elapsed begin
                            @sync @distributed for i = 1:length(V.pids)
                                # @spawnat V.pids[i] localpart(vtmp)[:,:] = localpart(V)[:,kconv]
                                # @spawnat V.pids[i] localpart(V)[:,imove+1:kconv] = localpart(V)[:,imove:kconv-1]
                                # @spawnat V.pids[i] localpart(V)[:,imove] = localpart(vtmp)
                                localpart(vtmp)[:,:] = localpart(V)[:,kconv]
                                localpart(V)[:,imove+1:kconv] = localpart(V)[:,imove:kconv-1]
                                localpart(V)[:,imove] = localpart(vtmp)
                            end
                            end
                            elapsedTime["@distributed_movecols_n"] += 1
                            evals[imove+1:kconv] = evals[imove:kconv-1]
                            evals[imove]=rho
                        end

                        if (kconv >= nwant && !swap & blk>1 ) || (kconv >= nwant+kmore)
                            if  displ > 1
                                @printf("The converged eigenvalues and residual_norms are:\n")
                                for i = 1:kconv
                                    @printf("  eigval(%3i) = %11.8e,   resnrm(%3i) = %8.5e \n", i, evals[i], i, resnrm[i])
                                end
                            end
                            # change to output V= V(:, 1:kconv); later 
                            elapsedTime["Matrix_Float64"] += @elapsed begin
                            eigV = Matrix{Float64}(V[:, 1:kconv])
                            end
                            elapsedTime["Matrix_Float64_n"] += 1
                            
                            if  displ > 0 #these output info may be useful for profiling
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
                    elapsedTime["@distributed_locally_assigning"] += @elapsed begin
                    @sync @distributed for i = 1:length(W.pids)
                        # @spawnat W.pids[i] localpart(W)[:,1:kact-ec] = localpart(W)[:,ec+1:kact]
                        localpart(W)[:,1:kact-ec] = localpart(W)[:,ec+1:kact]
                    end
                    end
                    elapsedTime["@distributed_locally_assigning_n"] += 1
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
                lowb = minimum(d_eig)
                
                #
                # determine if need to do outer restart (only n need to be updated)
                #
                if  n + blk > vomax && opts["do_outer"]
                    nold = n
                    n = max(kconv+blk, vomax - 2*blk)
                    if  displ > 4
                        @printf("--> Outer-restart: n from %i downto %i, vomax=%i, vimax=%i, kact=%i\n", nold, n, vomax, vimax, kact)
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
            @printf("***** itmax=%i, it_total=%i\n", itmax, iter_total)
            @warn("***** bchdav.jl:  Maximum iteration exceeded\n")
            @printf("***** nwant=%i, kconv=%i, vomax=%i\n", nwant, kconv, vomax)
            @warn("***** it could be that your vimax/blk, or vomax, or polym is too small")
        end

    

    end

    if verb
        @printf("\n")
        @printf("+++++++++++++++++++++ runtime details for debugging +++++++++++++++++++++++++\n")
        @printf("total runtime:                          %.2e \n", elapsedTime["total"])
        @printf("runtime pqr:                            %.2e \n", elapsedTime["pqr"])
        @printf("runtime chksym:                         %.2e \n", elapsedTime["chksym"])
        @printf("runtime main_loop:                      %.2e \n", elapsedTime["main_loop"])
        @printf("runtime distribute:                     %.2e / %i \n", elapsedTime["distribute"], elapsedTime["distribute_n"])
        @printf("runtime Matrix_Float64:                 %.2e / %i \n", elapsedTime["Matrix_Float64"], elapsedTime["Matrix_Float64_n"])
        @printf("runtime dzeros:                         %.2e / %i \n", elapsedTime["dzeros"], elapsedTime["dzeros_n"])
        @printf("runtime computing_residual:             %.2e / %i \n", elapsedTime["computing_residual"], elapsedTime["computing_residual_n"])
        @printf("runtime dCheb_filter:                   %.2e / %i \n", elapsedTime["dCheb_filter"], elapsedTime["dCheb_filter_n"])
        @printf("runtime dCheb_filter_scal:              %.2e / %i \n", elapsedTime["dCheb_filter_scal"], elapsedTime["dCheb_filter_scal_n"])
        @printf("runtime DGKS_blk:                       %.2e / %i \n", elapsedTime["DGKS_blk"], elapsedTime["DGKS_blk_n"])
        @printf("runtime dDGKS_blk:                      %.2e / %i \n", elapsedTime["dDGKS_blk"], elapsedTime["dDGKS_blk_n"])
        if verb
            @printf("   runtime dDGKS_@distributed_gs:                %.2e / %i \n", elapsedTime["dDGKS_@distributed_gs"], elapsedTime["dDGKS_@distributed_gs_n"])
            @printf("   runtime dDGKS_@distributed_sumup:             %.2e / %i \n", elapsedTime["dDGKS_@distributed_sumup"], elapsedTime["dDGKS_@distributed_sumup_n"])
            @printf("   runtime dDGKS_@distributed_matvec:            %.2e / %i \n", elapsedTime["dDGKS_@distributed_matvec"], elapsedTime["dDGKS_@distributed_matvec_n"])
            @printf("   runtime dDGKS_@distributed_matscalar:         %.2e / %i \n", elapsedTime["dDGKS_@distributed_matscalar"], elapsedTime["dDGKS_@distributed_matscalar_n"])
            @printf("   runtime dDGKS_@distributed_movecols:          %.2e / %i \n", elapsedTime["dDGKS_@distributed_movecols"], elapsedTime["dDGKS_@distributed_movecols_n"])
            @printf("   runtime dDGKS_@distributed_locally_assigning: %.2e / %i \n", elapsedTime["dDGKS_@distributed_locally_assigning"], elapsedTime["dDGKS_@distributed_locally_assigning_n"])
            @printf("   runtime dDGKS_@distributed_computingnorm:     %.2e / %i \n", elapsedTime["dDGKS_@distributed_computingnorm"], elapsedTime["dDGKS_@distributed_computingnorm_n"])
        
        end
        @printf("runtime duser_Hx:                       %.2e / %i \n", elapsedTime["duser_Hx"], elapsedTime["duser_Hx_n"])
        @printf("runtime LANCZ_bound:                    %.2e / %i \n", elapsedTime["LANCZ_bound"], elapsedTime["LANCZ_bound_n"])
        @printf("runtime @distributed_locally_assigning: %.2e / %i \n", elapsedTime["@distributed_locally_assigning"], elapsedTime["@distributed_locally_assigning_n"])
        @printf("runtime @distributed_matvec:            %.2e / %i \n", elapsedTime["@distributed_matvec"], elapsedTime["@distributed_matvec_n"])
        @printf("runtime @distributed_movecols:          %.2e / %i \n", elapsedTime["@distributed_movecols"], elapsedTime["@distributed_movecols_n"])
        @printf("runtime @distributed_sumup:             %.2e / %i \n", elapsedTime["@distributed_sumup"], elapsedTime["@distributed_sumup_n"])
        @printf("+++++++++++++++++++++++++++++++ END +++++++++++++++++++++++++++++++++++++++++\n")
    end

    

    return evals, eigV, kconv, history

end



function DGKS_blk(X, varargin...)
# 
# Usage:  V = DGKS_blk( X, V ); 
#
# Apply DGKS to ortho-normalize V against X.
# V can have more than 1 column. the returned V is orthonormal to X.
#
# It is assumed that X is already ortho-normal, this is very important 
# for the projection  P = I - X*X^T  to be orthogonal.
#
# For debugging purpose, a 3rd variable can be provided to test if
# X is ortho-normal or not. when orthtest is 1, test will be performed.
#
    
    V = varargin[1]
    if  length(varargin) == 2
        orthtest = varargin[2]
        if  orthtest==1
            xorth=norm(X'*X - Matrix{Float64}(size(X,2), size(X,2)), 2)
            if  xorth > 1e-8
                @printf("--> Input orthgonality test:  ||X^t*X - I|| = %e\n", xorth)          
                @error("input X is not ortho-normal") 
            end
        end
    end
    
    epsbig = 2.22045e-16
    reorth=0.717
    one=1.0e0
    ndim, colv = size(V)
    colx = size(X,2)
    orth_flops = 0
    vrank = 0
    
    for k = 1:colv
    
        nrmv = norm(V[:,k],2)
        orth_flops=orth_flops+ndim
        if nrmv < epsbig*sqrt(ndim)
            continue
        end
        
        #
        # normalize for the sake of numerical stability (important)
        #
        if  nrmv <= 2*epsbig || nrmv >= 300
            V[1:ndim,k] = V[1:ndim,k]/nrmv;  orth_flops=orth_flops+ndim;
            nrmv = one;
        end
        
        h = hcat(X, V[1:ndim,1:vrank])'*V[1:ndim,k]
        V[1:ndim,k] = V[1:ndim,k] -  hcat(X, V[1:ndim,1:vrank])*h
        orth_flops = orth_flops + ndim*((colx+vrank)*2 + 1)
        
        nrmproj = norm(V[1:ndim,k],2);  orth_flops=orth_flops+ndim;
        if  nrmproj > reorth*nrmv
            #
            # pass the reorthogonalization test, no need to refine
            #
            vrank = vrank +1
            if  abs(nrmproj - one) > epsbig
                V[1:ndim,k]=V[1:ndim,k]/nrmproj;   orth_flops=orth_flops+ndim;
            end
            if  vrank != k
                V[1:ndim,vrank]=V[1:ndim,k]
            end
        else
            #
            # fail the reorthogonalization test, refinement necessary
            #
            nrmv = nrmproj      

            h = hcat(X, V[1:ndim,1:vrank])'*V[1:ndim,k]
            V[1:ndim,k] = V[1:ndim,k] -  hcat(X, V[:,1:vrank])*h
            orth_flops = orth_flops + ndim*((colx+vrank)*2 + 1)
            
            nrmproj = norm(V[1:ndim,k],2);   orth_flops=orth_flops+ndim;     
            if  nrmproj > reorth*nrmv  && nrmproj >= sqrt(ndim)*epsbig
            #if (nrmproj > reorth*nrmv),
                vrank = vrank + 1
                if  abs(nrmproj - one) > epsbig
                    V[1:ndim,k]=V[1:ndim,k]/nrmproj;   orth_flops=orth_flops+ndim;
                end
                if  vrank != k
                    V[1:ndim,vrank]=V[1:ndim,k]
                end	
            else
            #
            # fail the 2nd reorthogonalization test,
            #    V(:,k) is numerically in V(:, 1:vrank),
            # do not increase vrank, but go for the next k 
            #
            end
        
        end    
    end
    
    if  vrank > 0
        V = V[:,1:vrank]
    else #recursively call DGKS_blk to make output V not a zero vector
        @printf("DGKS: # of columns replaced by random vectors =%i\n", colv-vrank) 
        V[:,vrank+1:colv], _ = DGKS_blk(hcat(X,V[:, 1:vrank]), rand(ndim,colv-vrank))
    end
    
    if  1==0  #set it to be true only for debugging purpose
        # if (nargin==3),
            if  orthtest==1
                #orthVX = norm(V'*X);
                xorth=norm(hcat(X,V)'*hcat(X,V) - Matrix{Float64}(size(hcat(X,V),2),size(hcat(X,V),2)),2)
                if  xorth > 1e-8
                    @printf("--> Output orthgo-normal error = %e\n", xorth)
                    @error("output [X, V] is not ortho-normal")
                end
            end
        # end
    end

    return V, orth_flops

end  


# function dDGKS_blk(X, varargin...)
function dDGKS_blk(X, cols, V)
# 
# Usage:  V = dDGKS_blk( X, V ); 
#
# Apply DGKS to ortho-normalize V against X.
# V can have more than 1 column. the returned V is orthonormal to X.
#
# It is assumed that X is already ortho-normal, this is very important 
# for the projection  P = I - X*X^T  to be orthogonal.
#
# For debugging purpose, a 3rd variable can be provided to test if
# X is ortho-normal or not. when orthtest is 1, test will be performed.
#
    
    # V = varargin[1]
    # if  length(varargin) == 2
    #     orthtest = varargin[2]
    #     if  orthtest==1
    #         xorth=norm(X'*X - Matrix{Float64}(size(X,2), size(X,2)))
    #         if  xorth > 1e-8
    #             @printf("--> Input orthgonality test:  ||X^t*X - I|| = %e\n", xorth)          
    #             @error("input X is not ortho-normal") 
    #         end
    #     end
    # end
    
    global elapsedTime

    epsbig = 2.22045e-16
    reorth=0.717
    one=1.0e0
    ndim, colv = size(V)
    colx = size(X,2)
    orth_flops = 0
    vrank = 0
    
    for k = 1:colv
        
        elapsedTime["dDGKS_@distributed_computingnorm"] += @elapsed begin
        # nrmv = norm(V[:,k],2)
        nrmv = @distributed (+) for i = 1:length(V.pids)
            norm(localpart(V)[:,k], 2)^2
        end
        nrmv = sqrt(nrmv)
        end
        elapsedTime["dDGKS_@distributed_computingnorm_n"] += 1
        orth_flops=orth_flops+ndim
        if nrmv < epsbig*sqrt(ndim)
            continue
        end
        
        #
        # normalize for the sake of numerical stability (important)
        #
        if  nrmv <= 2*epsbig || nrmv >= 300
            # V[1:ndim,k] = V[1:ndim,k]/nrmv
            elapsedTime["dDGKS_@distributed_matscalar"] += @elapsed begin
            @sync @distributed for i = 1:length(V.pids)
                # @spawnat V.pids[i] localpart(V)[:,k] = localpart(V)[:,k]/nrmv
                localpart(V)[:,k] = localpart(V)[:,k]/nrmv
            end  
            end
            elapsedTime["dDGKS_@distributed_matscalar"] += 1
            orth_flops=orth_flops+ndim
            nrmv = one
        end
        
        # h = hcat(X, V[1:ndim,1:vrank])'*V[1:ndim,k]
        elapsedTime["dDGKS_@distributed_sumup"] += @elapsed begin
        h = @distributed (+) for i = 1:length(V.pids)
            # fetch(@spawnat V.pids[i] hcat(localpart(X)[:,cols], localpart(V)[:, 1:vrank])'*localpart(V)[:,k])
            hcat(localpart(X)[:,cols], localpart(V)[:,1:vrank])' * localpart(V)[:,k]
        end
        end
        elapsedTime["dDGKS_@distributed_sumup_n"] += 1
        # V[1:ndim,k] = V[1:ndim,k] -  hcat(X, V[1:ndim,1:vrank])*h
        elapsedTime["dDGKS_@distributed_gs"] += @elapsed begin
        @sync @distributed for i = 1:length(V.pids)
            # @spawnat V.pids[i] localpart(V)[:, k] = localpart(V)[:,k] - hcat(localpart(X)[:,cols], localpart(V)[:, 1:vrank])*h
            localpart(V)[:,k] = localpart(V)[:,k] - hcat(localpart(X)[:,cols], localpart(V)[:, 1:vrank])*h
        end
        end
        elapsedTime["dDGKS_@distributed_gs_n"] += 1

        orth_flops = orth_flops + ndim*((colx+vrank)*2 + 1)
        
        # nrmproj = norm(V[:,k],2)
        elapsedTime["dDGKS_@distributed_computingnorm"] += @elapsed begin
        nrmproj = @distributed (+) for i = 1:length(V.pids)
            norm(localpart(V)[:,k], 2)^2
        end
        nrmproj = sqrt(nrmproj)
        end
        elapsedTime["dDGKS_@distributed_computingnorm_n"] += 1
        orth_flops=orth_flops+ndim
        if  nrmproj > reorth*nrmv
            #
            # pass the reorthogonalization test, no need to refine
            #
            vrank = vrank +1
            if  abs(nrmproj - one) > epsbig
                # V[1:ndim,k]=V[1:ndim,k]/nrmproj;  
                elapsedTime["dDGKS_@distributed_matscalar"] += @elapsed begin
                @sync @distributed for i = 1:length(V.pids)
                    # @spawnat V.pids[i] localpart(V)[:,k] = localpart(V)[:,k]/nrmproj
                    localpart(V)[:,k] = localpart(V)[:,k]/nrmproj
                end 
                end
                elapsedTime["dDGKS_@distributed_matscalar_n"] += 1
                orth_flops=orth_flops+ndim;
            end
            if  vrank != k
                # V[1:ndim,vrank]=V[1:ndim,k]
                elapsedTime["DGKS_@displayed_movecols"] += @elapsed begin
                @sync @distributed for i = 1:length(V.pids)
                    # @spawnat V.pids[i] localpart(V)[:,vrank] = localpart(V)[:,k]
                    localpart(V)[:,vrank] = localpart(V)[:,k]
                end 
                end
                elapsedTime["DGKS_@displayed_movecols_n"] += 1
            end
        else
            #
            # fail the reorthogonalization test, refinement necessary
            #
            nrmv = nrmproj      

            # h = hcat(X, V[1:ndim,1:vrank])'*V[1:ndim,k]
            # V[1:ndim,k] = V[1:ndim,k] -  hcat(X, V[:,1:vrank])*h
            elapsedTime["dDGKS_@distributed_sumup"] += @elapsed begin
            h = @distributed (+) for i = 1:length(V.pids)
                # fetch(@spawnat V.pids[i] hcat(localpart(X)[:,cols], localpart(V)[:, 1:vrank])'*localpart(V)[:,k])
                hcat(localpart(X)[:,cols], localpart(V)[:,1:vrank])' * localpart(V)[:,k]
            end
            end
            elapsedTime["dDGKS_@distributed_sumup_n"] += 1
            elapsedTime["dDGKS_@distributed_gs"] += @elapsed begin
            @sync @distributed for i = 1:length(V.pids)
                # @spawnat V.pids[i] localpart(V)[:, k] = localpart(V)[:,k] - hcat(localpart(X)[:,cols], localpart(V)[:, 1:vrank])*h
                localpart(V)[:,k] = localpart(V)[:,k] - hcat(localpart(X)[:,cols], localpart(V)[:,1:vrank])*h
            end
            end
            elapsedTime["dDGKS_@distributed_gs_n"] += 1

            orth_flops = orth_flops + ndim*((colx+vrank)*2 + 1)
            
            # nrmproj = norm(V[:,k],2)
            elapsedTime["dDGKS_@distributed_computingnorm"] += @elapsed begin
            nrmproj = @distributed (+) for i = 1:length(V.pids)
                norm(localpart(V)[:,k], 2)^2
            end
            nrmproj = sqrt(nrmproj)
            end
            elapsedTime["dDGKS_@distributed_computingnorm_n"] += 1
            orth_flops=orth_flops+ndim   
            if  nrmproj > reorth*nrmv  && nrmproj >= sqrt(ndim)*epsbig
            #if (nrmproj > reorth*nrmv),
                vrank = vrank + 1
                if  abs(nrmproj - one) > epsbig
                    # V[1:ndim,k]=V[1:ndim,k]/nrmproj
                    elapsedTime["dDGKS_@distributed_matscalar"] += @elapsed begin
                    @sync @distributed for i = 1:length(V.pids)
                        # @spawnat V.pids[i] localpart(V)[:,k] =  localpart(V)[:,k]/nrmproj
                        localpart(V)[:,k] = localpart(V)[:,k]/nrmproj
                    end 
                    end
                    elapsedTime["dDGKS_@distributed_matscalar_n"] += 1
                    orth_flops=orth_flops+ndim
                end
                if  vrank != k
                    # V[1:ndim,vrank]=V[1:ndim,k]
                    elapsedTime["dDGKS_@distributed_movecols"] += @elapsed begin
                    @sync @distributed for i = 1:length(V.pids)
                        # @spawnat V.pids[i] localpart(V)[:,vrank] = localpart(V)[:,k]
                        localpart(V)[:,vrank] = localpart(V)[:,k]
                    end 
                    end
                    elapsedTime["dDGKS_@distributed_movecols_n"] += 1
                end	
            else
            #
            # fail the 2nd reorthogonalization test,
            #    V(:,k) is numerically in V(:, 1:vrank),
            # do not increase vrank, but go for the next k 
            #
            end
        
        end    
    end
    
    if  vrank > 0
        returnV = Matrix{Float64}(V[:,1:vrank])
    else #recursively call dDGKS_blk to make output V not a zero vector
        @printf("DGKS: # of columns replaced by random vectors =%i\n", colv-vrank) 
        newX = dzeros((ndim, length(cols)+vrank), V.pids, [length(V.pids), 1])
        elapsedTime["dDGKS_@distributed_locally_assigning"] += @elapsed begin
        @sync @distributed for i = 1:length(V.pids)
            # @spawnat V.pids[i] localpart(newX)[:,:] = hcat(localpart(X)[:,cols], localpart(V)[:,1:vrank])
            localpart(newX)[:,:] = hcat(localpart(X)[:,cols], localpart(V)[:,1:vrank])
        end
        end
        elapsedTime["dDGKS_@distributed_locally_assigning_n"] += 1
        # V[:,vrank+1:colv], _ = dDGKS_blk(hcat(X,V[:, 1:vrank]), rand(ndim,colv-vrank))
        elapsedTime["dDGKS_blk"] += @elapsed begin
        returnV, _ = dDGKS_blk(newX, 1:size(newX,2), drand((ndim,colv-vrank), V.pids, [length(V.pids), 1]))
        end
        elapsedTime["dDGKS_blk_n"] += 1


    end
    
    if  1==0  #set it to be true only for debugging purpose
        # if (nargin==3),
            if  orthtest==1
                #orthVX = norm(V'*X);
                xorth=norm(hcat(X[:,cols],V)'*hcat(X[:,cols],V) - Matrix{Float64}(size(hcat(X[:,cols],V),2),size(hcat(X[:,cols],V),2)),2)
                if  xorth > 1e-8
                    @printf("--> Output orthgo-normal error = %e\n", xorth)
                    @error("output [X, V] is not ortho-normal")
                end
            end
        # end
    end

    
    return returnV, orth_flops

end    




function LANCZ_bound(n, k, A)
    # %
    # % apply k steps safeguarded Lanczos to get the upper bound of (eig(A)).
    # %  
    # % Input: 
    # %        n  --- dimension
    # %        k  --- (optional) perform k steps of Lanczos
    # %               if not provided, k =6 (a relatively small k is enough)
    # %        A  --- (optional) a script name of matrix for the mattrix-vector product
    # %  
    # % Output:
    # %      upperb  --- estimated upper bound of all the eigenvalues
    # %   lower_nwb  --- estimated lower bound of the potentially unwanted eigenvalues
    # %     minritz  --- (optional) the current smallest Ritz value
    # %     maxritz  --- (optional) the current largest Ritz value
    # %           e  --- (optional) all the computed Ritz values  
    # %  
      
    # % note that   AV=VT+fe'  ==> AVQ=VQD+fe'Q  ==> ||r||<=||f||
    # % the theory says:
    # %   for any mu (eigenvalue of T), there exists a lam (eigenvalue of A)
    # %   s.t.    | lam - mu | <= ||r||
    # %
    # % to be safe (since we really need an upper bound), we get the upperb
    # % as  max(eig(T)) + ||f|| 
    # %
      
    # if (nargin < 2), 
    #     k = 6; 
    # else
    # if length(varargin) == 0
    #     k = 6
    # elseif length(varargin) == 1
    #     k = varargin[1]
    # elseif length(varargin) == 2
    #     k = varargin[1]
    #     A = varargin[2]
    # end
    
    global elapsedTime

    k = min(max(k, 6), 12)    #do not go over 12 steps
    # end 

    T = zeros(k,k)
    v = rand(n,1)     
    v = v/norm(v,2)
    tol = 2.5e-16  

    # if  length(varargin) < 2
    # f, _     = duser_Hx(A, v)
    elapsedTime["duser_Hx"] += @elapsed begin
        Df = dzeros((n, 1), A.pids, [length(A.pids), 1])
        @sync for ii in 1:length(A.pids)
            @spawnat A.pids[ii] localpart(Df)[:,:] = localpart(A)*v
        end
        f = Matrix{Float64}(Df)
    end
    elapsedTime["duser_Hx_n"] += 1
    alpha = (v'*f)[1]
    f     = f - alpha * v 
    T[1,1] = alpha
    
    isbreak = 0
    jj = k
    for j = 2:k    #run k steps
        beta = norm(f,2)
        if beta > tol
            v0 = v;  v = f/beta;
            # f, _  = duser_Hx(A, v)
            elapsedTime["duser_Hx"] += @elapsed begin
                Df = dzeros((n, size(v,2)), A.pids, [length(A.pids), 1])
                @sync for ii in 1:length(A.pids)
                    @spawnat A.pids[ii] localpart(Df)[:,:] = localpart(A)*v
                end
                f = Matrix{Float64}(Df)
            end
            elapsedTime["duser_Hx_n"] += 1
            f = f - v0*beta
            alpha = (v'*f)[1]
            f  = f - v*alpha
            T[j,j-1] = beta
            T[j-1,j] = beta
            T[j,j] = alpha
        else
            isbreak = 1
            jj = j
            break
        end
    end
        
    # else
    #     f, _     = duser_Hx(v,A)
    #     alpha = v'*f
    #     f     = f - alpha * v 
    #     T[1,1]= alpha
        
    #     isbreak = 0
    #     jj = k
    #     for j = 2:k    #run k steps
    #         beta = norm(f)
    #         if  beta > tol
    #             v0 = v;  v = f/beta;
    #             f, _  = duser_Hx(v,A)
    #             f = f - v0*beta
    #             alpha = v'*f
    #             f  = f - v*alpha
    #             T[j,j-1] = beta; T[j-1,j] = beta; T[j,j] = alpha;
    #         else
    #             isbreak = 1
    #             jj = j
    #             break
    #         end
    #     end
    # end

    if  isbreak != 1
        e, X = eigen(T[1:jj,1:jj])
    else
        e, X = eigen(T[1:jj-1,1:jj-1])
    end
    if  beta > 1e-1
        # multiply beta by the largest element of the last row of X
        beta = beta*maximum(X[end,:])
    end
    # e = diag(e); 
    maxritz = maximum(e)
    minritz = minimum(e)
    upperb = maxritz + beta
    lower_nwb = mean(e)
    

    return upperb, lower_nwb, minritz, maxritz

end

#----------------------------------------------------------------------------
function dCheb_filter(A, x, polm, low, high, augment)
# %   
# %  [y] = dCheb_filter(x, polm, low, high, augment) 
# %
# %  Chebshev iteration, this one require only two bounds, it does not apply internal scaling
# %

    global  filt_non_mv_cput
    global  filt_mv_cput
    global elapsedTime

    # filt_cput = cputime
    filt_cput = @elapsed begin
        mvcput = 0.0

        e = (high - low)/2.0
        center= (high+low)/2.0

        y, mvcpu0 = duser_Hx(A, x)
        mvcput = mvcput + mvcpu0
        y = (-y + center*x)/e

        for i = 2:polm-1
            ynew, mvcpu0 = duser_Hx(A, y)
            mvcput = mvcput + mvcpu0
            ynew = (- ynew + center*y)* 2.0/e  - x
            x = y
            y = ynew
        end

        ynew, mvcpu0 = duser_Hx(A, y)
        mvcput = mvcput + mvcpu0
        # default return unless augment==2 or 3.
        ynew = (- ynew + center*y)* 2/e - x

    end
    filt_mv_cput =  filt_mv_cput + mvcput
    filt_non_mv_cput = filt_non_mv_cput + (filt_cput - mvcput)

    # if (nargin > 4),
    if  augment==2
        ynew = hcat(y, ynew)
    elseif augment==3
        ynew = hcat(x, y, ynew)
    end
    # end

    return ynew

end
    
#----------------------------------------------------------------------------
function dCheb_filter_scal(A, x, polm, low, high, leftb, augment)
# %   
# %  [y] = dCheb_filter_scal(x, polm, low, high, leftb, augment);
# %        (note: need  "leftb < low")
# %
# %  Chebshev iteration, scaled version (need three bounds: leftb < low < high ) 
# %

    global  filt_non_mv_cput
    global  filt_mv_cput
    global elapsedTime

    # filt_cput = cputime
    filt_cput = @elapsed begin
        mvcput = 0.0

        e = (high - low)/2.0
        center= (high+low)/2.0
        sigma = e/(leftb - center)
        tau = 2.0/sigma

        # y, mvcpu0 = duser_Hx(A, x)
        mvcpu0 = @elapsed begin
            Dy = dzeros((size(A,1), size(x,2)), A.pids, [length(A.pids), 1])
            @sync for ii in 1:length(A.pids)
                @spawnat A.pids[ii] localpart(Dy)[:,:] = localpart(A)*x
            end
            y = Matrix{Float64}(Dy)
        end
        elapsedTime["duser_Hx"] += mvcpu0
        elapsedTime["duser_Hx_n"] += size(x,2)
        mvcput = mvcput + mvcpu0
        y = (y - center*x) * (sigma/e)

        for i = 2:polm-1
            sigma_new = 1.0 /(tau - sigma)
            # ynew, mvcpu0 = duser_Hx(A, y)
            mvcpu0 = @elapsed begin
                Dynew = dzeros((size(A,1), size(y,2)), A.pids, [length(A.pids), 1])
                @sync for ii in 1:length(A.pids)
                    @spawnat A.pids[ii] localpart(Dynew)[:,:] = localpart(A)*y
                end
                ynew = Matrix{Float64}(Dynew)
            end
            elapsedTime["duser_Hx"] += mvcpu0
            elapsedTime["duser_Hx_n"] += size(y,2)
            mvcput = mvcput + mvcpu0
            ynew = (ynew - center*y)*(2.0*sigma_new/e) - (sigma*sigma_new)*x
            x = y
            y = ynew
            sigma = sigma_new
        end

        # ynew, mvcpu0 = duser_Hx(A, y)
        mvcpu0 = @elapsed begin
            Dynew = dzeros((size(A,1), size(y,2)), A.pids, [length(A.pids), 1])
            @sync for ii in 1:length(A.pids)
                @spawnat A.pids[ii] localpart(Dynew)[:,:] = localpart(A)*y
            end
            ynew = Matrix{Float64}(Dynew)
        end
        elapsedTime["duser_Hx"] += mvcpu0
        elapsedTime["duser_Hx_n"] += size(y,2)
        mvcput = mvcput + mvcpu0
        sigma_new = 1.0 /(tau - sigma)
        # default return unless augment==2 or 3.
        ynew = (ynew - center*y)*(2.0*sigma_new/e) - (sigma*sigma_new)*x

    end
    filt_mv_cput =  filt_mv_cput + mvcput
    filt_non_mv_cput = filt_non_mv_cput + (filt_cput - mvcput)

    # if (nargin > 4),
    if augment==2
        ynew = hcat(y, ynew)
    elseif augment==3
        ynew = hcat(x, y, ynew)
    end
    # end

    return ynew
end


#----------------------------------------------------------------------------
function duser_Hx(A, v)   
    # %
    # % Usage: [w] = user_Hx( v,  Mmat, varargin )
    # %
    # % compute matrix-vector products.
    # % the "Mmat" is optional, when it is omitted,
    # % a global matrix named "A" is necessary.
    # %
    # % 
    # % all matrix-vector products are performed through calls to
    # % this subroutine so that the mat-vect count can be accurate.
    # % a global veriable "MVprod" is needed to count the 
    # % matrix-vector products
    # %  
    # % if Mmat is a string function, in many applications one needs to 
    # % input more variables to @Mmat than just v. the additional 
    # % variables, if exist, are passed through the varargin cell array.
    # %
      
    # global DA_operator
    global MVprod       #count the number of matrix-vector products
    global MVcpu        #count the cputime for matrix-vector products
    
    # mvcput = cputime
    mvcput = @elapsed begin
        # v = varargin[1]
        # w = zeros(size(DA_operator,1), size(v,2))

        DW = dzeros((size(A,1), size(v,2)), A.pids, [length(A.pids), 1])
        @sync for i in 1:length(A.pids)
            # pid = DA_operator.pids[i]
            # fetch(@spawnat pid localpart(DA_operator)*v)
            @spawnat A.pids[i] localpart(DW)[:,:] = localpart(A)*v
            # localpart(A)*v
        end
        w = Matrix{Float64}(DW)
        
            #may need to modify this and replace by specific matrix-vector product
            # w = eval(varargin[2])(varargin[2], varargin[3])  

    end
    #
    # increase the global mat-vect-product count accordingly
    #
    MVcpu  = MVcpu + mvcput;
    MVprod = MVprod + size(v,2);  

    return w, mvcput
end

function duser_Hx_1(A, v)   
    # %
    # % Usage: [w] = user_Hx( v,  Mmat, varargin )
    # %
    # % compute matrix-vector products.
    # % the "Mmat" is optional, when it is omitted,
    # % a global matrix named "A" is necessary.
    # %
    # % 
    # % all matrix-vector products are performed through calls to
    # % this subroutine so that the mat-vect count can be accurate.
    # % a global veriable "MVprod" is needed to count the 
    # % matrix-vector products
    # %  
    # % if Mmat is a string function, in many applications one needs to 
    # % input more variables to @Mmat than just v. the additional 
    # % variables, if exist, are passed through the varargin cell array.
    # %
      
    # global DA_operator
    global MVprod       #count the number of matrix-vector products
    global MVcpu        #count the cputime for matrix-vector products
    
    # mvcput = cputime
    mvcput = @elapsed begin
        # v = varargin[1]
        # w = zeros(size(DA_operator,1), size(v,2))

        DW = dzeros((size(A,1), size(v,2)), A.pids, [length(A.pids), 1])
        @sync @distributed for i in 1:length(A.pids)
            # pid = DA_operator.pids[i]
            # fetch(@spawnat pid localpart(DA_operator)*v)
            localpart(DW)[:,:] = localpart(A)*v
            # localpart(A)*v
        end
        w = Matrix{Float64}(DW)
        
            #may need to modify this and replace by specific matrix-vector product
            # w = eval(varargin[2])(varargin[2], varargin[3])  

    end
    #
    # increase the global mat-vect-product count accordingly
    #
    MVcpu  = MVcpu + mvcput;
    MVprod = MVprod + size(v,2);  

    return w, mvcput
end       


function duser_Hx_2(A, v)
    # %
    # % Usage: [w] = user_Hx( v,  Mmat, varargin )
    # %
    # % compute matrix-vector products.
    # % the "Mmat" is optional, when it is omitted,
    # % a global matrix named "A" is necessary.
    # %
    # % 
    # % all matrix-vector products are performed through calls to
    # % this subroutine so that the mat-vect count can be accurate.
    # % a global veriable "MVprod" is needed to count the 
    # % matrix-vector products
    # %  
    # % if Mmat is a string function, in many applications one needs to 
    # % input more variables to @Mmat than just v. the additional 
    # % variables, if exist, are passed through the varargin cell array.
    # %

    # global DA_operator
    #global MVprod       #count the number of matrix-vector products
    #global MVcpu        #count the cputime for matrix-vector products

    # mvcput = cputime
    mvcput = @elapsed begin
        # v = varargin[1]
        # w = zeros(size(DA_operator,1), size(v,2))

        DW = dzeros((size(A,1), size(v,2)), A.pids, [length(A.pids), 1])
        @sync @distributed for i in 1:length(A.pids)
            # pid = DA_operator.pids[i]
            # fetch(@spawnat pid localpart(DA_operator)*v)
            localpart(DW)[:,:] = localpart(A)*v
            # localpart(A)*v
        end
        w = Matrix{Float64}(DW)

            #may need to modify this and replace by specific matrix-vector product
            # w = eval(varargin[2])(varargin[2], varargin[3])  

    end
    #
    # increase the global mat-vect-product count accordingly
    #
    #MVcpu  = MVcpu + mvcput;
    #MVprod = MVprod + size(v,2);

    return w, mvcput
end

function duser_Hx_3(sA, v)
    # %
    # % Usage: [w] = user_Hx( v,  Mmat, varargin )
    # %
    # % compute matrix-vector products.
    # % the "Mmat" is optional, when it is omitted,
    # % a global matrix named "A" is necessary.
    # %
    # % 
    # % all matrix-vector products are performed through calls to
    # % this subroutine so that the mat-vect count can be accurate.
    # % a global veriable "MVprod" is needed to count the 
    # % matrix-vector products
    # %  
    # % if Mmat is a string function, in many applications one needs to 
    # % input more variables to @Mmat than just v. the additional 
    # % variables, if exist, are passed through the varargin cell array.
    # %

    # global DA_operator
    #global MVprod       #count the number of matrix-vector products
    #global MVcpu        #count the cputime for matrix-vector products

    A = distribute(sA, procs=workers(), dist=[length(workers()), 1])
    # mvcput = cputime
    mvcput = @elapsed begin
        # v = varargin[1]
        # w = zeros(size(DA_operator,1), size(v,2))

        DW = dzeros((size(A,1), size(v,2)), A.pids, [length(A.pids), 1])
        @sync @distributed for i in 1:length(A.pids)
            # pid = DA_operator.pids[i]
            # fetch(@spawnat pid localpart(DA_operator)*v)
            localpart(DW)[:,:] = localpart(A)*v
            # localpart(A)*v
        end
        w = Matrix{Float64}(DW)

            #may need to modify this and replace by specific matrix-vector product
            # w = eval(varargin[2])(varargin[2], varargin[3])  

    end
    #
    # increase the global mat-vect-product count accordingly
    #
    #MVcpu  = MVcpu + mvcput;
    #MVprod = MVprod + size(v,2);

    return w, mvcput
end

#----------------------------------------------------------------------------
function struct_merge(varargin...)
    # % 
    # % Usage: [structm] = struct_merge(struct1, struct2, struct3, ...) 
    # %
    # % Merges any number of structures such that the output structm 
    # % will have all the fields in the listed input structures.
    # %
    # % Redundant field values will be removed. 
    # % Field values from the earlier input structures will overwrite
    # % field values from latter input structures (in case there are conflicts).
    # %
    
    
    
    structm =  varargin[1]
    if !isa(structm, Dict) 
        @error("the 1st input to struct_merge should be a dictionary")
    end

    for i = 2:length(varargin)
        struct2 = varargin[i]
        if  !isa(struct2, Dict) 
            @error(["the #",num2str(i)," input to struct_merge should be a dictionary"])
        end
        fn = fieldnames(struct2)

        for ii = 1:length(fn)
            if (!haskey(structm, fn[ii]))
                structm[fn[ii]] = struct2[fn[ii]]
            end
        end
    end

    return structm

end



# end # module