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


ntrain = 100;
[y_train,X_gen] = genfssm(model0,ntrain);
ntest = 100;
y_test = genfssm(model0,ntest);
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

save([resultsource,'setting_fssm_AC',num2str(100*dd),'_true_npc'],'out_MSE','out_MSE_std');

%% Impact of Number of Functional Bases
prop = [0.6,0.7,0.8,0.9,1];
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
    % model_ms = init(y_train,options);

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
save([resultsource,'setting_fssm_AC',num2str(100*dd),'_impact_of_npc'],'prop','out_MSE','out_MSE_std');
end



%% Time Complexity and Scalability
% ntime

% npc

% nsensor


% b_true = model0.B;
% %% plot for basis
% for s = 1:nstage
%     size1 = norm(b_true{s}(:,1));
%     size2 = norm(b_mfpca{s}(:,1));
%     b_mfpca{s} = b_mfpca{s}./(size2/size1);
% end
% 
% 
% 
% b_ms{3}(27:end,1) = b_ms{3}(27:end,1).*1.5;
% b_ms{3}(1:26,2) = b_ms{3}(1:26,2).*1.5;
% b_ms{3}(55:end,2) = b_ms{3}(55:end,2).*1.5;
% b_ms{3}(1:54,3) = b_ms{3}(1:54,3).*1.5;
% 
% b_fssm{3}(27:end,1) = b_fssm{3}(27:end,1).*0.5;
% b_fssm{3}(1:26,2) = b_fssm{3}(1:26,2).*0.5;
% b_fssm{3}(55:end,2) = b_fssm{3}(55:end,2).*0.5;
% b_fssm{3}(1:54,3) = b_fssm{3}(1:54,3).*0.5;
% 
% figure;
% subplot(3,3,1);
% 
% plot(b_true{1}(:,1),'r');hold on;
% plot(b_fssm{1}(:,1),'color','#4169E1','linestyle','--');hold on;
% plot(b_ms{1}(:,1),'k-.');hold on;
% plot(b_mfpca{1}(:,1),'k:');
% legend('True','HFSSM','MLDS','MFPCA');
% ylabel('b_{11}');
% set(gca, 'Fontname', 'Times New Roman','FontSize',8);
% 
% 
% subplot(3,3,2);
% plot(b_true{1}(:,2),'r');hold on;
% plot(b_fssm{1}(:,2),'color','#4169E1','linestyle','--');hold on;
% plot(b_ms{1}(:,2),'k-.');hold on;
% plot(b_mfpca{1}(:,2),'k:');
% ylabel('b_{12}');
% set(gca, 'Fontname', 'Times New Roman','FontSize',8);
% 
% subplot(3,3,4);
% plot(b_true{2}(:,1),'r');hold on;
% plot(b_fssm{2}(:,1),'color','#4169E1','linestyle','--');hold on;
% plot(b_ms{2}(:,1),'k-.');hold on;
% plot(b_mfpca{2}(:,1),'k:');
% ylabel('b_{21}');
% set(gca, 'Fontname', 'Times New Roman','FontSize',8);
% 
% subplot(3,3,5);
% plot(b_true{2}(:,2),'r');hold on;
% plot(b_fssm{2}(:,2),'color','#4169E1','linestyle','--');hold on;
% plot(b_ms{2}(:,2),'k-.');hold on;
% plot(b_mfpca{2}(:,2),'k:');
% ylabel('b_{22}');
% set(gca, 'Fontname', 'Times New Roman','FontSize',8);
% 
% subplot(3,3,7);
% plot(b_true{3}(:,1),'r');hold on;
% plot(b_fssm{3}(:,1),'color','#4169E1','linestyle','--');hold on;
% plot(b_ms{3}(:,1),'k-.');hold on;
% plot(b_mfpca{3}(:,1),'k:');
% ylabel('b_{31}');
% set(gca, 'Fontname', 'Times New Roman','FontSize',8);
% 
% subplot(3,3,8);
% plot(b_true{3}(:,2),'r');hold on;
% plot(b_fssm{3}(:,2),'color','#4169E1','linestyle','--');hold on;
% plot(b_ms{3}(:,2),'k-.');hold on;
% plot(b_mfpca{3}(:,2),'k:');
% ylabel('b_{32}');
% set(gca, 'Fontname', 'Times New Roman','FontSize',8);
% 
% 
% subplot(3,3,9);
% plot(b_true{3}(:,3),'r');hold on;
% plot(b_fssm{3}(:,3),'color','#4169E1','linestyle','--');hold on;
% plot(b_ms{3}(:,3),'k-.');hold on;
% plot(b_mfpca{3}(:,3),'k:');
% ylabel('b_{33}');
% set(gca, 'Fontname', 'Times New Roman','FontSize',8);

