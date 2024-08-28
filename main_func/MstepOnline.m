function model1 = MstepOnline(Y, Ex_new, Exx_new, Exy_new,model0, options)
% Input:
% Y: cell(1,nstage):ntime*nsensor*nsam
%   Ex: E(x_t): cell(1,nstage):(npc*nsensor)*nsam
%   Exx: E[z_tz_t^T]: cell(1,nstage):(npc*nsensor)*(npc*nsensor)*nsam
%   Exy: E[z_tz_{t-1}^T]: cell(1,nstage-1):(npc*nsensor)*(npc*nsensor)*nsam
%   llh: loglikelihood

lambda = 0;
gamma = options.gamma;
nsam_online = options.nsam_online;
rho = nsam_online^(-gamma);
% rho = 0;

nstage = size(Y,2);
ntime = zeros(1,nstage);
nsensor = zeros(1,nstage);
npc = zeros(1,nstage);
for s = 1:nstage
    [ntime(s),nsensor(s)] = size(Y{s});
end
for s = 1:nstage
    [~,npc(s)] = size(model0.B{s});
end


model1 = model0;


YX_block_new = cell(1,nstage);
YY_block_new = cell(1,nstage);
for s = 1:nstage
    YX_block_new{s} = (1-rho)*model0.YX_block{s} + rho*Y{s}*Ex_new{s}';
    YY_block_new{s} = (1-rho)*model0.YY_block{s} + rho*Y{s}*Y{s}';
end

Exx_sum_new = cell(1,nstage);
Exy_sum_new = cell(1,nstage-1);
Exx_sum_ten_new = cell(1,nstage);
Exy_sum_ten_new = cell(1,nstage-1);
for s = 1:nstage
    Exx_sum_new{s} = (1-rho)*model0.Exx_sum{s} + rho*Exx_new{s};
    Exx_sum_ten_new{s} = reshape(tensor(Exx_sum_new{s}),[npc(s),nsensor(s),npc(s),nsensor(s)]);
end

for s = 1:nstage-1
    Exy_sum_new{s} = (1-rho)*model0.Exy_sum{s} + rho*Exy_new{s};
    Exy_sum_ten_new{s} = permute(reshape(tensor(Exy_sum_new{s}),[npc(s+1),nsensor(s+1),npc(s),nsensor(s)]),[3,4,1,2]);
end



%% B sigma1 Ts
sigma1 = model0.sigma1;
Ts = model0.Ts;
B = model0.B;

for s = 1:nstage
    diffm1 = diffm(ntime(s));
    diffm2 = diffm(ntime(s)-1);
    omega =diffm1'*diffm2'*diffm2*diffm1;

    F = kron(eye(npc(s)),lambda*sigma1(s)*Ts{s}*omega) + kron(double(ttt(Exx_sum_ten_new{s},tensor(eye(nsensor(s))),[2,4],[1,2])),eye(ntime(s)));
    h = reshape(YX_block_new{s},[],1);
    B{s} = reshape(F\h,ntime(s),npc(s)); 
    
%     L = kron(eye(npc(s)),B{s});
%     Lam = (L'/F*L)\(L'/F*h-reshape(eye(npc(s)),[],1));
%     B{s} = reshape(F\(h-L*Lam),ntime(s),npc(s));       
end


XBTY_new = cell(1,nstage);
YTY_new = cell(1,nstage);
for s = 1:nstage
    XBTY_new{s} = (1-rho)*model0.XBTY{s} + rho*Ex_new{s}'*B{s}'/Ts{s}*Y{s};
    YTY_new{s} = (1-rho)*model0.YTY{s} + rho*Y{s}'/Ts{s}*Y{s};
end
temp1 = cell(1,nstage);
temp2 = cell(1,nstage);
for s = 1:nstage
    temp1{s} = B{s}*double(ttt(Exx_sum_ten_new{s},tensor(eye(nsensor(s))),[2,4],[1,2]))*B{s}';
    temp2{s} = double(ttt(tensor(B{s}'/Ts{s}*B{s}),Exx_sum_ten_new{s},[1,2],[1,3]));
    
    sigma1(s) = trace(YTY_new{s} - 2*XBTY_new{s}+temp2{s})/(ntime(s)*nsensor(s));
    Ts{s} = (YY_block_new{s} - B{s}*YX_block_new{s}' - YX_block_new{s}*B{s}' + temp1{s})./(sigma1(s)*nsensor(s));
    Ts{s} = recons(Ts{s});
end

%% A C sigma2
A = model0.A;
C = model0.C;
for s = 1:nstage-1
    A{s} = double(ttt(Exy_sum_ten_new{s},tensor(C{s}),[2,4],[1,2]))'/double(ttt(Exx_sum_ten_new{s},tensor(C{s}*C{s}'),[2,4],[1,2]));
    C{s} = double(ttt(tensor(A{s}'*A{s}),Exx_sum_ten_new{s},[1,2],[1,3]))\double(ttt(tensor(A{s}'),Exy_sum_ten_new{s},[1,2],[1,3]));
end

sigma2 = model0.sigma2;
for s = 1:nstage-1
    F = kron(C{s}',A{s});
    sigma2(s) = trace(Exx_sum_new{s+1}- 2* F*Exy_sum_new{s}'+F*Exx_sum_new{s}*F') / (nsensor(s+1)*npc(s+1));
end

%% M1 V1
X1_sum_new = (1-rho)*model0.X1_sum + rho*Ex_new{1};

M1 = X1_sum_new;       
M1_vec = reshape(M1,[],1);
V1 = Exx_sum_new{1}- M1_vec*M1_vec';


model1.B = B;
model1.A = A;
model1.C = C;
model1.Ts = Ts;
model1.sigma1 = sigma1;
model1.sigma2 = sigma2;
model1.M1 = M1;
model1.V1 = V1;

model1.YX_block = YX_block_new;
model1.YY_block = YY_block_new;
model1.Exx_sum = Exx_sum_new;
model1.Exy_sum = Exy_sum_new;
model1.XBTY = XBTY_new;
model1.YTY = YTY_new;
model1.X1_sum = X1_sum_new;



end