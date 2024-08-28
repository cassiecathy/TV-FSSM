addpath(genpath('E:\0研究\06 HFSSM_functional time series\02 code\3 utilities'));
addpath(genpath('E:\0研究\06 HFSSM_functional time series\02 code\1 Numerical'));

clear;
close all;
resultsource = sprintf('E:/0研究/06 HFSSM_functional time series/02 code/1 Numerical/');

%% Artificial data
% 15/30 stages profile (6-8) suiji basis suiji (10 random selected 2-3);
rnd_nsensor = [6,7,8];
rnd_npc = [2,3];

rnd_ntime = [60,64,50];
total_pc = 10;

rnd_B = cell(1,3);% 10 random basis
tmp2 = fourierbasis(2*total_pc-1,rnd_ntime(1));
rnd_B{1} = tmp2(:,[1,2:2:end]);

tmp3 = WTortho(rnd_ntime(2),'Vaidyanathan',4,2);
rnd_B{2}= tmp3(1:total_pc,:)';

sele = 1:3:3*(total_pc -1)+1;
knot = max(sele)+3;
tmp = bspline_basismatrix(3,linspace(0,1,knot),linspace(0,1,rnd_ntime(3)+2));
tmp1 = tmp(:,sele);
tmp1(1,:) = [];
tmp1(end,:) = [];
rnd_B{3} = tmp1;
% plotB(rnd_B);% show bases of the three-stage processes


%% random generated
for dd = [0.70]
rng('default')

nstage = 20;
ntime = zeros(1,nstage);
npc = zeros(1,nstage);
nsensor = zeros(1,nstage);
B = cell(1,nstage);
ndim = 0;
for s = 1:nstage
    id = randsrc(1,1,[1,2,3]);
    ntime(s) = rnd_ntime(id);
    npc(s) = randsrc(1,1,rnd_npc);
    perm = randperm(total_pc);
    temp = perm(1:npc(s));% select bases
    B{s} = rnd_B{id}(:,temp);
    nsensor(s) = randsrc(1,1,rnd_nsensor);
    ndim = ndim + nsensor(s)*ntime(s);
end


A = cell(1,nstage-2);
C = cell(1,nstage-2);
for s = 1:nstage-2
    [U,S,V] = svd(rand(npc(s+2)*nsensor(s+2),npc(s+1)*nsensor(s+1)+npc(s)*nsensor(s)));
%     S = S.*(1/max(diag(S))).*dd;
    S(abs(S)>1) = dd;
    A{s} = U*S*V';
    [U,S,V] = svd(rand(npc(s+2)*nsensor(s+2),npc(s+1)*nsensor(s+1)));
%     S = S.*(1/max(diag(S))).*dd;
    S(abs(S)>1) = dd;
    C{s} = U*S*V';  
    
%     A{s} = rand(npc(s+2)*nsensor(s+2),npc(s+1)*nsensor(s+1)+npc(s)*nsensor(s)).*0.1;
%     C{s} = rand(npc(s+2)*nsensor(s+2),npc(s+1)*nsensor(s+1)).*0.2;
end

sigma1 = 1e-3;
sigma2 = 1e-3;
M1 = rand(npc(1)*nsensor(1),1);
M2 = rand(npc(2)*nsensor(2),1);

model0.B = B;
model0.A = A;
model0.C = C;
model0.sigma1 = sigma1;
model0.sigma2 = sigma2;
model0.M1 = M1;
model0.M2 = M2;
model0.nsensor = nsensor;


ntrain = 100;
[y_train,X_gen] = genfarma(model0,ntrain);
% plotY(y_train);

ntest = 100;
y_test = genfarma(model0,ntest);

%% normalize
means = cell(1,nstage);
stds = cell(1,nstage);
for s = 1:nstage
    y_test{s} = reshape(y_test{s},[],100)';
    y_train{s} = reshape(y_train{s},[],100)';
    means{s} = mean(y_train{s});
    stds{s} = std(y_train{s});
    y_test{s} = (y_test{s}-means{s})./stds{s};
    y_train{s} = (y_train{s}-means{s})./stds{s};
    y_test{s} = reshape(y_test{s}',ntime(s),nsensor(s),100);
    y_train{s} = reshape(y_train{s}',ntime(s),nsensor(s),100);
end


%% comparison under true npcs
options.npc = npc;
options.lambda = 50;
options.gamma = 0.9;%越小，新样本权重越高

[model_fssm,llh] = fssmEm(y_train,options);
b_fssm = model_fssm.B;
yp_fssm = fssmPre(y_test,model_fssm);
% b_sum = b_fssm;
% y_sum = [y_test;yp_fssm];
% [out_PVE,out_MSE,out_MSE_std,out_MAE,out_MAE_std] = index(b_sum,y_sum);

% online EM
nsam_start = 25;
Y_start = cell(1,nstage);
for s = 1:nstage
    Y_start{s} = y_train{s}(:,:,1:nsam_start-1);
end
model_ms = init(Y_start,options);
% model_ms = init(y_train,options);

for i = nsam_start:ntrain
    Y_new = cell(1,nstage);
    for s = 1:nstage
        Y_new{s} = y_train{s}(:,:,i);
    end
    options.nsam_online = i;
    model_ms = fssmEmOnline(model_ms,Y_new,options);
end
b_ms = model_ms.B;
yp_ms = fssmPre(y_test,model_ms); % ntime,nsensor,nstage,nsam

[yp_dfpca,b_dfpca] = DFPCA(y_test,y_train,npc);
[yp_far,b_far] = FAR(y_test,y_train,npc);
[yp_farma,b_farma] = FARMA(y_test,y_train,npc);

y_sum = [y_test;yp_fssm;yp_ms;yp_dfpca;yp_far;yp_farma];
b_sum = [b_fssm;b_ms;b_dfpca;b_far;b_farma];
[out_MSE,out_MSE_std]= index2(b_sum,y_sum,means,stds);

save([resultsource,'setting_farma_C',num2str(100*dd),'_true_npc'],'out_MSE','out_MSE_std');

%% Impact of Number of Functional Bases
prop = [0.6,0.7,0.8,0.9,0.95];
nprop = length(prop);
nmethod = 5;
out_PVE = zeros(nprop,nmethod);
out_MSE = zeros(nprop,nmethod);
out_MSE_std = zeros(nprop,nmethod);
out_MAE = zeros(nprop,nmethod);
out_MAE_std = zeros(nprop,nmethod);
for i = 1:nprop
    options2 = struct();
    options2.prop = prop(i);
    options2.lambda = 50;
    model_fssm = fssmEm(y_train,options2);
    b_fssm = model_fssm.B;
    yp_fssm = fssmPre(y_test,model_fssm);
    
    npc = zeros(1,nstage);
    for s = 1:nstage
        npc(s) = size(b_fssm{s},2);
    end

    % online EM
    options2.npc = npc;
    options2.gamma = 0.9;%越小，新样本权重越高
    model_ms = init(Y_start,options2);
    for m = nsam_start:ntrain
        Y_new = cell(1,nstage);
        for s = 1:nstage
            Y_new{s} = y_train{s}(:,:,m);
        end
        options2.nsam_online = m;
        model_ms = fssmEmOnline(model_ms,Y_new,options2);
    end
    b_ms = model_ms.B;
    yp_ms = fssmPre(y_test,model_ms); % ntime,nsensor,nstage,nsam

    % baselines
    [yp_dfpca,b_dfpca] = DFPCA(y_test,y_train,npc);
    [yp_far,b_far] = FAR(y_test,y_train,npc);
    [yp_farma,b_farma] = FARMA(y_test,y_train,npc);

    y_sum = [y_test;yp_fssm;yp_ms;yp_dfpca;yp_far;yp_farma];
    b_sum = [b_fssm;b_ms;b_dfpca;b_far;b_farma];
    [out_MSE(i,:),out_MSE_std(i,:)]= index2(b_sum,y_sum,means,stds);
end
save([resultsource,'setting_farma_C',num2str(100*dd),'_impact_of_npc'],'prop','out_MSE','out_MSE_std');
end