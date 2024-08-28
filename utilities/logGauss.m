function y = logGauss(X, mu, sigma)
% Compute log pdf of a Gaussian distribution.
X = reshape(X - mu,[],1);
% y = X'*X;

d = length(X);

[U,p]= chol(sigma);
if p ~= 0
    error('ERROR: sigma is not PD. Your sample size is too small.');
end

cons = (d*log(2*pi)+2*sum(log(diag(U))));%fu
Q = U'\X;
quad = dot(Q,Q,1);%zheng

y = -(cons+quad)/2;
end
