function [X_smooth, Exx, Exy, llh,X_filter] = Estep(model, Y)
% Input:
% Y: cell(1,nstage):ntime*nsensor*nsam
%   model: model structure
% Output:
%   X_smooth: E(x_t): cell(1,nstage):(npc*nsensor)*nsam
%   Exx: E[z_tz_t^T]: cell(1,nstage):(npc*nsensor)*(npc*nsensor)*nsam
%   Exy: E[z_tz_{t-1}^T]: cell(1,nstage):(npc*nsensor)*(npc*nsensor)*nsam
%   llh: loglikelihood


B = model.B;
A = model.A;
C = model.C;
Ts = model.Ts;
sigma1 = model.sigma1;
sigma2 = model.sigma2;
M1 = model.M1;
V1 = model.V1;

% B: cell(1,nstage):ntime(s)*npc(s+1)
% A: cell(1,nstage-1):npc(s+1)*npc(s)
% C: cell(1,nstage-1):nsensor(s)*nsensor(s+1)
% Ts: cell(1,nstage): ntime(s)*ntime(s)
% sigma1: nstage*1
% sigma2: (nstage-1)*1
% M1: npc(1)*nsensor(1)
% V1: (npc(1)*nsensor(1))*(npc(1)*nsensor(1))

nstage = size(Y,2);
ntime = zeros(1,nstage);
nsensor = zeros(1,nstage);
npc = zeros(1,nstage);
for s = 1:nstage
    [ntime(s),nsensor(s),nsam] = size(Y{s});
end
for s = 1:nstage
    [~,npc(s)] = size(B{s});
end

%% transform
B0 = cell(1,nstage);
for s = 1:nstage
    B0{s} = kron(eye(nsensor(s)),B{s});
    Y{s} = reshape(Y{s},[],nsam);
end
F = cell(1,nstage-1);
for s = 1:nstage-1
    F{s} = kron(C{s}',A{s});
end
M1 = reshape(M1,[],1);


%% 
T = nstage;
X_filter = cell(1,T);
X_smooth = cell(1,T);
Vxx_filter = cell(1,T);
Vxx_smooth = cell(1,T);
Exx = cell(1,T);
Exy = cell(1,T-1);
llh = zeros(T,nsam);
Vp = cell(1,T);
xp = cell(1,T); 

for i = 1:nsam
    % filtering forward
    PC = V1*B0{1}';
    S = B0{1}*PC+kron(sigma1(1)*eye(nsensor(1)),Ts{1});
    K = PC/S;
    X_filter{1}(:,i) = M1+K*(Y{1}(:,i)-B0{1}*M1);
    Vxx_filter{1}(:,:,i) = (eye(npc(1)*nsensor(1))-K*B0{1})*V1;
    Vp{1}(:,:,i) = V1;  % useless, just make a point
    xp{1}(:,i) = M1; % useless, just make a point
    llh(1,i) = logGauss(Y{1}(:,i),B0{1}*M1,S);
    for s = 2:T    
        [X_filter{s}(:,i), Vxx_filter{s}(:,:,i), xp{s}(:,i), Vp{s}(:,:,i), llh(s,i)] = ...
            forwardUpdate(Y{s}(:,i), X_filter{s-1}(:,i), Vxx_filter{s-1}(:,:,i), F{s-1}, sigma2(s-1)*eye(npc(s)*nsensor(s)), B0{s}, kron(sigma1(s)*eye(nsensor(s)),Ts{s}));
    end

    % smoothing backward
    X_smooth{T}(:,i) = X_filter{T}(:,i);
    Vxx_smooth{T}(:,:,i) = Vxx_filter{T}(:,:,i);
    Exx{T}(:,:,i) = Vxx_smooth{T}(:,:,i)+X_smooth{T}(:,i)*X_smooth{T}(:,i)';
    for s = T-1:-1:1  
        [X_smooth{s}(:,i), Vxx_smooth{s}(:,:,i), Exx{s}(:,:,i), Exy{s}(:,:,i)] = ...
            backwardUpdate(X_smooth{s+1}(:,i), Vxx_smooth{s+1}(:,:,i), X_filter{s}(:,i), Vxx_filter{s}(:,:,i), xp{s+1}(:,i), Vp{s+1}(:,:,i), F{s});
    end
end

for s = 1:nstage
    X_smooth{s} = reshape(X_smooth{s},[npc(s),nsensor(s),nsam]);
end
    llh = sum(sum(llh)); % conditional likelihood
end


function [mu1, V1, xp, Vp, llh] = forwardUpdate(Y, mu0, V0, A, Q, C, R)
Vp = A*V0*A'+Q;
PC = Vp*C';
R = C*PC+R;
% R = recons(R);
K = PC/R;
xp = A*mu0;
CAmu = C*xp;
mu1 = xp+K*(Y-CAmu);
V1 = (eye(numel(mu1))-K*C)*Vp;
llh = logGauss(Y,CAmu,R);
end

function [nu0, U0,E00, E10] = backwardUpdate(nu1, U1, mu, V, xp, Vp, A)
J = V*A'/Vp;
nu0 = mu+J*(nu1-xp);
U0 = V+J*(U1-Vp)*J';
E00 = U0+nu0*nu0';
E10 = U1*J'+nu1*nu0';
end






