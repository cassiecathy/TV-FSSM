addpath(genpath('E:\0研究\06 HFSSM_functional time series\02 code\1 Numerical'));
addpath(genpath('E:\0研究\06 HFSSM_functional time series\02 code\3 utilities'));

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
% for s = 1:3
%     rnd_B{s} = rnd_B{s}./sqrt(diag(rnd_B{s}'*rnd_B{s})');
% end

%% random generated
for dd = [0.99]
rng('default')

nstage = 15;
ntime = zeros(1,nstage);
npc = zeros(1,nstage);
nsensor = zeros(1,nstage);
B = cell(1,nstage);
for s = 1:nstage
    id = randsrc(1,1,[1,2,3]);
    ntime(s) = rnd_ntime(id);
    npc(s) = randsrc(1,1,rnd_npc);
    perm = randperm(total_pc);
    temp = perm(1:npc(s));% select bases
    B{s} = rnd_B{id}(:,temp);
    nsensor(s) = randsrc(1,1,rnd_nsensor);
end


A = cell(1,nstage-1);
C = cell(1,nstage-1);
for s = 1:nstage-1
    [U,S,V] = svd(rand(npc(s+1),npc(s)));
    S(abs(S)>1) = dd;
    A{s} = U*S*V';
    [U,S,V] = svd(rand(nsensor(s),nsensor(s+1)));
    S(abs(S)>1) = dd;
    C{s} = U*S*V'; 
    
%     A{s} = rand(npc(s+1),npc(s))*0.5;
%     C{s} = rand(nsensor(s),nsensor(s+1))*0.5;
end

Ts = cell(1,nstage);
for s = 1:nstage
    Ts{s} = eye(ntime(s));
    for i = 1:(ntime(s)-1)
        for j = (i+1):ntime(s)
            Ts{s}(i,j) = (0.2)^(abs(i-j));
            Ts{s}(j,i) = Ts{s}(i,j);
        end
    end
end

sigma1 = 1e-3*ones(nstage,1);
sigma2 = 1e-3*ones(nstage-1,1);
M1 = rand(npc(1),nsensor(1));

V1 = gallery('randcorr',npc(1)*nsensor(1)) * 1e-3;

model0.B = B;
model0.A = A;
model0.C = C;
model0.Ts = Ts;
model0.sigma1 = sigma1;
model0.sigma2 = sigma2;
model0.M1 = M1;
model0.V1 = V1;


nsam = 100;
[y_train,X_gen] = genfssm(model0,nsam);
nsam_test = 100;
y_test = genfssm(model0,nsam_test);
% plotY(y_test);

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

options.npc = npc;
options.lambda = 0;
options.gamma = 0.9;%越小，新样本权重越高


%% Comparison Between EM and Online EM
nsam_start = 25;
nacc = nsam_start:nsam;
Y_start = cell(1,nstage);
for s = 1:nstage
    Y_start{s} = y_train{s}(:,:,1:nsam_start-1);
end
model_fssm = init(Y_start,options);

out_TPU_online = zeros(nsam,1);%time per update
out_MSE_online = zeros(nsam,1);
llh_online = zeros(nsam,1);
    
% online (train for each sequential sample)
for i = nacc
    Y_new = cell(1,nstage);
    for s = 1:nstage
        Y_new{s} = y_train{s}(:,:,i);
    end
    options.nsam_online = i;
    t1 = clock;
    [model_fssm,llh_online(i)] = fssmEmOnline(model_fssm,Y_new,options);
    t2 = clock;
    out_TPU_online(i) = etime(t2,t1);
    b_sum = model_fssm.B;
    yp_fssm = fssmPre(y_test,model_fssm);
    y_sum = [y_test;yp_fssm];
    out_MSE_online(i)= index2(b_sum,y_sum,means,stds);
end

%% offline 
out_TPU_offline = zeros(nsam,1);%time per update
out_MSE_offline = zeros(nsam,1);
llh_offline = zeros(nsam,1);
inumber = zeros(nsam,1);

for i = nacc
    Y_offline = cell(1,nstage);
    for s = 1:nstage
        Y_offline{s} = y_train{s}(:,:,1:i);
    end
    t1 = clock;
    [model_fssm,llh] = fssmEm(Y_offline,options);
    t2 = clock;
    inumber(i) = length(llh);
    llh_offline(i) = llh(end)/i;
    out_TPU_offline(i) = etime(t2,t1)/length(llh);
    b_sum = model_fssm.B;
    yp_fssm = fssmPre(y_test,model_fssm);
    y_sum = [y_test;yp_fssm];
    out_MSE_offline(i)= index2(b_sum,y_sum,means,stds);
end
save([resultsource,'Online_MSE_FSSM',num2str(100*dd)],'nacc','out_TPU_online','llh_online','llh_offline','out_MSE_online','out_TPU_offline','out_MSE_offline','inumber');


end

