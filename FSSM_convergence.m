% this is the code for Phase I learning and Phase II monitoring based on a
% heterogeneous functional state space model (HFSSM):
%                        Ys = Bs*Xs + Es       s = 1,...,S
%                        Xs+1 = As*Xs*Cs + Ws  s = 1,...,S-1
% It assumes X1~anorm(M1,V1)   Es~anorm(0,Rs)	Ws~anorm(0,Qs)
%            Bs(s=1,...,S) are orthogonal and smooth

% % adds to path
addpath(genpath('E:\0研究\06 HFSSM_functional time series\02 code\3 utilities'));
addpath(genpath('E:\0研究\06 HFSSM_functional time series\02 code\1 Numerical'));

clear;
close all;
resultsource = sprintf('E:/0研究/06 HFSSM_functional time series/02 code/1 Numerical/');

%% Artificial data
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
% plotB(rnd_B);% show bases

rep = 10;
totalsam = [25,50,100,200,500,800,1000];
% totalsam = 100;
ppdiff_off = zeros(length(totalsam),8);% 8 parameters
itt_off = zeros(length(totalsam),rep); % time per iteration
itn_off = zeros(length(totalsam),rep); % iteration number
ppdiff_on = zeros(length(totalsam),8);% 8 parameters
itt_on = zeros(length(totalsam),rep); % time per iteration
itn_on = zeros(length(totalsam),rep); % iteration number

tic;
for k = 1:length(totalsam)
    pdiff_off = zeros(rep,8);
    pdiff_on = zeros(rep,8);
    for r = 1:rep
        %% IC model settings
        nstage = 10;
        ntime = zeros(1,nstage);
        npc = zeros(1,nstage);
        nsensor = zeros(1,nstage);
        B = cell(1,nstage);
        for s = 1:nstage
            id = randsrc(1,1,[1,2,3]);% select kind of bases
            ntime(s) = rnd_ntime(id);
            npc(s) = randsrc(1,1,rnd_npc);% select number of bases
            perm = randperm(total_pc);
            temp = perm(1:npc(s));% select bases
            B{s} = rnd_B{id}(:,temp);
            nsensor(s) = randsrc(1,1,rnd_nsensor);
        end
        
        
        A = cell(1,nstage-1);
        C = cell(1,nstage-1);
        for s = 1:nstage-1
            [U,S,V] = svd(rand(npc(s+1),npc(s)));
            S(abs(S)>1) = 0.99;
            A{s} = U*S*V';
            [U,S,V] = svd(rand(nsensor(s),nsensor(s+1)));
            S(abs(S)>1) = 0.99;
            C{s} = U*S*V'; 

        %     A{s} = rand(npc(s+1),npc(s))*0.5;
        %     C{s} = rand(nsensor(s),nsensor(s+1))*0.5;
        end

        Ts = cell(1,nstage);
        for s = 1:nstage
%             sr = linspace(1,0,ntime(s));
%             Ts{s} = toeplitz(sr);
%             e = eig(Ts{s});
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
%         M1 = M1 ./ repmat(sum(M1, 1), npc(1),1);
        
        
%         temp = rand(npc(1)*nsensor(1));
%         V1 = temp*temp'* 1e-5 + eye(npc(1)*nsensor(1))* 1e-3;
        V1 = gallery('randcorr',npc(1)*nsensor(1)) * 1e-3;
%         V1 = eye(npc(1)*nsensor(1)) * 1e-3;
        
        model0.B = B;
        model0.A = A;
        model0.C = C;
        model0.Ts = Ts;
        model0.sigma1 = sigma1;
        model0.sigma2 = sigma2;
        model0.M1 = M1;
        model0.V1 = V1;
        
        nsam = totalsam(k);
        [y_train,X_gen] = genfssm(model0,nsam);
        
        scale = zeros(2,nstage-1);
        for s = 1:nstage-1
            D = sort(abs(eig(model0.A{s}'*model0.A{s})),'descend');
            scale(1,s) = sqrt(D(1));
            D = sort(abs(eig(model0.C{s}'*model0.C{s})),'descend');
            scale(2,s) = sqrt(D(1));
        end       
        
        options.norm = 1;
        options.X_gen = X_gen;
        options.npc = npc;
        options.lambda = 50;
        options.scale = scale;
        options.gamma = 0.9;      
        
        
        %% offline estimate
        t1 = clock;
        [model_fssm, llh] = fssmEm(y_train,options);
        t2 = clock;
        itt_off(k,r) = etime(t2,t1)/length(llh);
        itn_off(k,r) = length(llh);
        % plotB(model_fssm.B);
        pdiff_off(r,:) = comp(model0,model_fssm);
        
        %% online estimate
        nsam_start = 25;
        model_ms = init(y_train,options);
        t1 = clock;
        for m = nsam-nsam_start+1:nsam
            Y_new = cell(1,nstage);
            for s = 1:nstage
                Y_new{s} = y_train{s}(:,:,m);
            end
            options.nsam_online = m;
            model_ms = fssmEmOnline(model_ms,Y_new,options);
        end
        t2 = clock;
        itt_on(k,r) = etime(t2,t1)/nsam_start;
        itn_on(k,r) = nsam-nsam_start+1;
        model_ms = mapparams(y_train,model_ms,options);
        pdiff_on(r,:) = comp(model0,model_ms);        
        
    end

    ittt_off = mean(itt_off,2);
    itnn_off = mean(itn_off,2);
    ppdiff_off(k,:) = mean(pdiff_off,1);
    ittt_on = mean(itt_on,2);
    itnn_on = mean(itn_on,2);
    ppdiff_on(k,:) = mean(pdiff_on,1);
end
toc;
% figure;
% plot(llh,'r-o','LineWidth',1,'MarkerFaceColor','r','MarkerSize',2);
% xlabel('Iteration');ylabel('Loglikelihood');
% set(gca, 'Fontname', 'Times New Roman','FontSize',10);
out_convergence_off = [totalsam',ppdiff_off,itnn_off,ittt_off];
out_convergence_on = [totalsam',ppdiff_on,itnn_on,ittt_on];
save([resultsource,'convergence_analysis'],'out_convergence_off','out_convergence_on');


