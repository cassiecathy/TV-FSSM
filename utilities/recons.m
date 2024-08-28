function sigma = recons(sigma)
[U,S,V] = svd(sigma);

ind = find(diag(S)<0.0001);
for i  = 1:length(ind)
    S(ind(i),ind(i)) = 0.0001;
end

sigma = U*S*V';

% [U,p]= chol(sigma);
% if p ~= 0
%     sigma = U'*U;
% end
end