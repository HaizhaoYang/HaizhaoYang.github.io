%% Chapter 1, A short review of scientific computing

%% 1.1 Classical numerical linear algebra problems

%% 1.1.1 QR factorization -- one algorithmic idea more important than others

% construct a numerically low-rank matrix A
m = 10; n = 6; epsilon = 1e-3;
QL = orth(randn(m,n));
QR = orth(randn(n,n));
sigma = diag(exp(-(1:n).^2/2)); 
A = QL*sigma*QR;
display('Constructed a matrix A of size');
size(A)
display('rank of A is');
rank(A)
pause;

% Fuall QR of a matrix A
[Q,R] = qr(A);
display('The full QR factorization of A is');
Q
R
pause;
display('Q is orthonormal matrix, we check Q^*Q:');
Q'*Q
pause;
display('R is full rank with a few smaller diagonal entries');
fprintf('rank of R is %d, size of R is %d by %d\n', rank(R),size(R,1),size(R,2));
pause;

% Pivoted QR of A, A*P=Q*R, where P is a permutation matrix and we use a 
[Q,R,P] = qr(A,'matrix');
display('The pivoted QR factorization of A is');
Q
R
P
display('Note that the diangonal entries of R has been ordered');
pause;
% It is no necessary to store a matrix P, instead, we just need a vector to
% represent the order of columns in the permutation, for example, we
% compute
[Q,R,p] = qr(A,0);
display('The pivoted QR factorization of A is');
Q
R
p
display('Note that the diangonal entries of R has been ordered');
pause;

% construct a numerical low-rank approximation A \approx X*Y as follows
pos = find(abs(diag(R)/R(1,1))>epsilon); 
k = pos(end); % k depends on the accuracy parameter epsilon
X = Q(:,1:k); Y = R(1:k,:);
% p is the column permutation on A, so we need to compute the inverse
% permutation as follows
[~,pInv] = sort(p);
% update Y 
Y = Y(:,pInv);
display('The low-rank approximation A\appxo X*Y is');
X
Y
% check the approximation using a matrix norm
display('The low-rank approximation error is');
norm(A-X*Y,'fro')

display('check the speed up for Ax vs X*(Y*x)');
tic;
x = randn(n,1);
for cnt = 1:10000
    y = A*x;
end
time1 = toc;
tic;
for cnt = 1:10000
    y = X*(Y*x);
end
time2 = toc;
display('speed up factor is');
time1/time2
display('when m and n is smaller, A*x is faster, howerver,');
display('when m and n are large, A*x is slower. Change them to 1000 to check this.');
pause;

%% 1.1.2 Singular value decomposition (SVD) -- a key step in many algorithms

% full SVD
[U,S,V] = svd(A);
display('the full SVD of A is');
U
S
V
display('note that the diagonal entries of S is decreasing');
pause;

% reduce SVD
[U,S,V] = svd(A,'econ');
display('the reduce SVD of A is');
U
S
V
display('note that the diagonal entries of S is decreasing');
pause;

display('construct a low-rank approximation using the SVD');
% A \approx X*Y
pos = find(abs(diag(S)/S(1,1))>epsilon); 
k = pos(end); % k depends on the accuracy parameter epsilon
X = U(:,1:k);
Y = S(1:k,1:k)*V(:,1:k)';
% check the approximation using a matrix norm
display('The low-rank approximation error is');
norm(A-X*Y,'fro')




