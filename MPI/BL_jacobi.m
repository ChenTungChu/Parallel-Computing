
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% one-sided Jacobi method for computing the SVD of A, A*V = U*S 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create matrix A with with known singular values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = 64; n=64;
rng('default');
A = randn(m);   [U,R] = qr(A);
A = randn(n);   [V,R] = qr(A);
true_sigma = [n:-1:1];
A = U*[diag(true_sigma); zeros(m-n,n)]*V';
% we want to orthogonalize columns of A
B = A;             % keep the original for later testing
V = eye(n);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% transpose if you wantt to work on rows instead of columns
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VT = V;
AT = A';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set termination criteria 
%     (a) smalness of off(A'*A), 
threshold = n*norm(B)*eps;
%     (b) max number of sweeps
maxsweep = 100; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% alternativel one can compute V'*A' = S'*U' 
% that is rotate rows of A', this might be a better idea
% in C as C stores matrices row-wise but this requires
% that A is transposed first
% we control this choice by setting choice = 0 for column
% rotations, and choice = 1 for row rotations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% main loop
i = 0; sum = norm(A);
while ((i < maxsweep) && (sqrt(sum) > threshold)),
  sum = 0.0;   % accumulate off(A) of the current A

% use Brent-Luk annihilation ordering
  for j = 1:n-1,      % a sweep is (n-1)*(n/2) rotations
    for k = 1:n/2,    % there are n/2 independent rotations

% get n/2 working 2 by 2 submatrices of A'*A
      a = A(:,[2*k-1 2*k])'*A(:,[2*k-1 2*k]);
% accumulate a part of the current off(A)
      sum = sum + 2*a(1,2)^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute rotation parameters c and s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      tau = (a(2,2)-a(1,1))/(2*a(1,2));
% 
% want abs(t) < 1 but to avoid numerical cancelations  we have
%     abs(t) = abs(-tau - sign(tau))*sqrt(1+tau^2))> 1
% old trick, take the other root, 
%     t = -tau + sign(tau)*sqrt(1+t^2)
% do not compute it but rather multiply and divide by
%     t = -tau -sign(tau)*sqrt(1+t^2)
% to safely obtain the smaller root 
      t = 1/(tau + sign(tau)*sqrt(1+tau^2));
      c = 1/sqrt(1+t^2); s = c*t;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rotate columns of A, and accumulate V from A*V = U*S
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      G = [c s; -s c];
      A(:, [2*k-1 2*k]) = A(:, [2*k-1 2*k])*G;
      V(:, [2*k-1 2*k]) = V(:, [2*k-1 2*k])*G;
      AT([2*k-1 2*k],:) = G'*AT([2*k-1 2*k],:);
      VT([2*k-1 2*k],:) = G'*VT([2*k-1 2*k],:);
    end % loop k

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% permute columns of A and V like in B-L ordering 
% to prepare for the next (n/2) independent rotation 
% remember that there are "interior" PEs and "boundary" PEs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% this is the "column" implementation
% note, Matlab counts from 1 so "odd" and "even" 
% are switched from those in C
    temp_even = A(:,2:2:end);
    temp_odd = A(:,3:2:end);
    A(:,3) = A(:,2);                  % this is the "left" boundary
    A(:,n) = A(:,n-1);                % this is the "right" boundary 
    A(:,5:2:n-1) = temp_odd(:,1:end-1);
    A(:,2:2:n-2) = temp_even(:,2:end);

    temp_even = V(:,2:2:end);
    temp_odd = V(:,3:2:end);
    V(:,3) = V(:,2);
    V(:,n) = V(:,n-1);
    V(:,5:2:n-1) = temp_odd(:,1:end-1);
    V(:,2:2:n-2) = temp_even(:,2:end);

% this is the transposed part, if we want to work on rows
    Ttemp_even = AT(2:2:end,:);
    Ttemp_odd = AT(3:2:end,:);
    AT(3,:) = AT(2,:);
    AT(n,:) = AT(n-1,:);
    AT(5:2:n-1,:) = Ttemp_odd(1:end-1,:);
    AT(2:2:n-2,:) = Ttemp_even(2:end,:);

    Ttemp_even = VT(2:2:end,:);
    Ttemp_odd = VT(3:2:end,:);
    VT(3,:) = VT(2,:);
    VT(n,:) = VT(n-1,:);
    VT(5:2:n-1,:) = Ttemp_odd(1:end-1,:);
    VT(2:2:n-2,:) = Ttemp_even(2:end,:);

  end % loop j which is the end of the current sweep

  i = i+1;  % next sweep

end % termination criteria met, end of iterations
%
% we have B*V = A, B is the original A, and A = U*Sigma
% extract U and sigma, the diagonal of Sigma
sigma = sqrt(diag(A'*A));
U = A./sigma';
%
dsigma = diag(sigma);
% order Sigma       
[Sigma,I] = sort(sigma,'descend');
% we have U(:,I)*dsigma(I,:) = U*dsigma = A
dSigma = diag(Sigma); % sorted matrix of singular values
% we have dSigma = dsigma(I,I)
% we have U(:,I)*dsigma(I,:) = A (not the original one)
% U(:,I)*dsigma(I,I) = U(:,I)*dSigma = A(:,I) = B*V(:,I)
%  and finally, U(:,I)*dSigma*V(:,I)' = B

% see how orthogonal V is
V_error = norm(V(:,I)*V(:,I)'-eye(n));
% see how orthogonal U is
U_error = norm(U(:,I)'*U(:,I)-eye(n));
% see how small the residual error is
res_error = norm(U(:,I)*dSigma*V(:,I)'-B)
sigma = sort(sigma,'descend');
if(n<9) sig_error = sigma - true_sigma';
end
errors = [i  U_error V_error res_error];
g = sprintf('%10.2e',errors);
fprintf('\n  sweeps    U_error   V_error   res_error\n');
fprintf('%s\n',g);
