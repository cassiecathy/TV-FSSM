function model1 = Mstep(Y, Ex, Exx, Exy,model0, options)
% Input:
% Y: cell(1,nstage):ntime*nsensor*nsam
%   Ex: E(x_t): cell(1,nstage):(npc*nsensor)*nsam
%   Exx: E[z_tz_t^T]: cell(1,nstage):(npc*nsensor)*(npc*nsensor)*nsam
%   Exy: E[z_tz_{t-1}^T]: cell(1,nstage-1):(npc*nsensor)*(npc*nsensor)*nsam
%   llh: loglikelihood

lambda = options.lambda;
nstage = size(Y,2);
ntime = zeros(1,nstage);
nsensor = zeros(1,nstage);
npc = zeros(1,nstage);
for s = 1:nstage
    [ntime(s),nsensor(s),nsam] = size(Y{s});
end
for s = 1:nstage
    [~,npc(s)] = size(model0.B{s});
end
model1 = model0;


YX_block = cell(1,nstage);
YY_block = cell(1,nstage);
for s = 1:nstage
    for i = 1:nsam
        YX_block{s}(:,:,i) = Y{s}(:,:,i)*Ex{s}(:,:,i)';
        YY_block{s}(:,:,i) = Y{s}(:,:,i)*Y{s}(:,:,i)';
    end
    YX_block{s} = sum(YX_block{s},3);
    YY_block{s} = sum(YY_block{s},3);
end

Exx_sum_ten = cell(1,nstage);
Exx_sum = cell(1,nstage);
Exy_sum = cell(1,nstage-1);
Exy_sum_ten = cell(1,nstage-1);
for s = 1:nstage
    Exx_sum{s} = sum(Exx{s},3);
    Exx_sum_ten{s} = tensor(reshape(Exx_sum{s},[npc(s),nsensor(s),npc(s),nsensor(s)]));
end

for s = 1:nstage-1
    Exy_sum{s} = sum(Exy{s},3);
    Exy_sum_ten{s} = tensor(permute(reshape(Exy_sum{s},[npc(s+1),nsensor(s+1),npc(s),nsensor(s)]),[3,4,1,2]));
end


%% B sigma1 Ts
sigma1 = model0.sigma1;
Ts = model0.Ts;
B = model0.B;

for s = 1:nstage
    diffm1 = diffm(ntime(s));
    diffm2 = diffm(ntime(s)-1);
    omega =diffm1'*diffm2'*diffm2*diffm1;

    F = kron(eye(npc(s)),lambda*sigma1(s)*Ts{s}*omega) + kron(double(ttt(Exx_sum_ten{s},tensor(eye(nsensor(s))),[2,4],[1,2])),eye(ntime(s)));
    h = reshape(YX_block{s},[],1);
    B{s} = reshape(F\h,ntime(s),npc(s)); 
    
%     L = kron(eye(npc(s)),B{s});
%     Lam = (L'/F*L)\(L'/F*h-reshape(eye(npc(s)),[],1));
%     B{s} = reshape(F\(h-L*Lam),ntime(s),npc(s));       
end

temp1 = cell(1,nstage);
temp2 = cell(1,nstage);
XBTY = cell(1,nstage);
YTY = cell(1,nstage);
for s = 1:nstage
    for i = 1:nsam
        XBTY{s}(:,:,i) = Ex{s}(:,:,i)'*B{s}'/Ts{s}*Y{s}(:,:,i);
        YTY{s}(:,:,i) = Y{s}(:,:,i)'/Ts{s}*Y{s}(:,:,i);
    end
    XBTY{s} = sum(XBTY{s},3);
    YTY{s} = sum(YTY{s},3);
    temp1{s} = B{s}*double(ttt(Exx_sum_ten{s},tensor(eye(nsensor(s))),[2,4],[1,2]))*B{s}';
    temp2{s} = double(ttt(tensor(B{s}'/Ts{s}*B{s}),Exx_sum_ten{s},[1,2],[1,3]));
    
    sigma1(s) = trace(YTY{s} - 2*XBTY{s}+temp2{s})/(nsam*ntime(s)*nsensor(s));
    Ts{s} = (YY_block{s} - B{s}*YX_block{s}' - YX_block{s}*B{s}' + temp1{s})./(nsam*sigma1(s)*nsensor(s));
    Ts{s} = recons(Ts{s});
end



%% A C sigma2
A = model0.A;
C = model0.C;
for s = 1:nstage-1
    A{s} = double(ttt(Exy_sum_ten{s},tensor(C{s}),[2,4],[1,2]))'/double(ttt(Exx_sum_ten{s},tensor(C{s}*C{s}'),[2,4],[1,2]));
    C{s} = double(ttt(tensor(A{s}'*A{s}),Exx_sum_ten{s},[1,2],[1,3]))\double(ttt(tensor(A{s}'),Exy_sum_ten{s},[1,2],[1,3]));
end

sigma2 = model0.sigma2;
for s = 1:nstage-1
    F = kron(C{s}',A{s});
    sigma2(s) = trace(Exx_sum{s+1}- 2* F*Exy_sum{s}'+F*Exx_sum{s}*F') / (nsam*nsensor(s+1)*npc(s+1));
end

%% M1 V1
X1_sum = sum(Ex{1},3);
M1 = X1_sum/nsam;      

M1_vec = reshape(M1,[],1);
V1 = Exx_sum{1}./nsam - M1_vec*M1_vec';
% V1 = Exx_sum{1}./nsam - M1_vec*M1_old_vec' - M1_old_vec*M1_vec'+ M1_old_vec*M1_old_vec' ;


model1.B = B;
model1.A = A;
model1.C = C;
model1.Ts = Ts;
model1.sigma1 = sigma1;
model1.sigma2 = sigma2;
model1.M1 = M1;
model1.V1 = V1;

% model1.YX_block = YX_block;
% model1.YY_block = YY_block;
% model1.Exx_sum = Exx_sum;
% model1.Exy_sum = Exy_sum;
% model1.XBTY = XBTY;
% model1.YTY = YTY;
% model1.X1_sum = X1_sum;
% model1.XX1 = XX1;
end