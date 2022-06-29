%
% sample of usage  
%

global A   x0  x1  nvmax   mode  Title  filename  
global opert  
global MVprod  

%
% setup the input hermitian matrix (need not be semi-definite) 
%
  A = delsq(numgrid('N', 70));

%
% use all default parameters to compute the 20 smallest eigenpairs of A
%
  [e, V]=bchdav(A, 20); 


%  
% use some options, 
%   e.g,   use block size blk=10,   use filter polynomial degree polm=30,
%          with a few more screen outputs than the default displ=1,  
%               %displ= (0) no output, (1) limited output; (>1) more intermidiate output
%          and use tol=1e-9         

  opts=struct( 'blk', 10,  'polym', 30, 'displ', 1, 'tol', 1e-9);  

  %
  % compute the smallest 50 eigenpairs 
  %
  [e, V]=bchdav(A, 50, opts); 

