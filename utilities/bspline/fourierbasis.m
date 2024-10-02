function basis = fourierbasis(nq,ntime)

point = linspace(-pi,pi,ntime);
norder = nq/2;
basis = zeros(ntime,nq);
for i = 1:norder
basis(:,2*i-1) = cos(i*point);
basis(:,2*i) = sin(i*point);
end
end