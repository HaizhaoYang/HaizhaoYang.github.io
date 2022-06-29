using CPUTime
using Distributed

function duser_Hx(varargin...)   
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
      
    global DA_operator
    global MVprod       #count the number of matrix-vector products
    global MVcpu        #count the cputime for matrix-vector products
    
    # mvcput = cputime
    CPUtic()
    v = varargin[1]
    # w = zeros(size(DA_operator,1), size(v,2))
    if length(varargin) == 1
        # println(size(DA_operator), DA_operator.pids, workers())
        w = @distributed (vcat) for i = 1:length(DA_operator.pids)
            pid = DA_operator.pids[i]
            ind = DA_operator.indices[i]
            fetch(@spawnat pid localpart(DA_operator)*v)
        end
    elseif length(varargin) == 2
        # w = varargin[2] * v
        w = @distributed (vcat) for i = 1:length(varargin[2].pids)
            pid = varargin[2].pids[i]
            ind = varargin[2].indices[i]
            fetch(@spawnat pid localpart(varargin[2])*v)
        end
    else
        #may need to modify this and replace by specific matrix-vector product
        # w = eval(varargin[2])(varargin[2], varargin[3])  
    end

    mvcput = CPUtoq()
    #
    # increase the global mat-vect-product count accordingly
    #
    MVcpu  = MVcpu + mvcput;
    MVprod = MVprod + size(v,2);  

    return w, mvcput
end