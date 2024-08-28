function model = init(Y,options)
% B: cell(1,nstage):ntime(s)*npc(s+1)
% A: cell(1,nstage-1):npc(s+1)*npc(s)
% C: cell(1,nstage-1):nsensor(s)*nsensor(s+1)
% sigma1
% sigma2
% M1: npc(1)*nsensor(1)
% V1: (npc(1)*nsensor(1))*(npc(1)*nsensor(1))


% Y: cell(1,nstage):ntime*nsensor*nsam
nstage = size(Y,2);
ntime = zeros(1,nstage);
nsensor = zeros(1,nstage);
npc = zeros(1,nstage);
Y_onesam = cell(1,nstage);
for s = 1:nstage
    Y_onesam{s} = Y{s}(:,:,1);
    [ntime(s),nsensor(s),nsam] = size(Y{s});
end

%% get B and X (SVD decomposition)
B = cell(1,nstage);
X = cell(1,nstage);
for s = 1:nstage  
    if ~isfield(options, 'npc')
        [~,S,~] = svd(Y_onesam{s});
        diags = zeros(1,min(size(S,1),size(S,2)));
        for k = 1:min(size(S,1),size(S,2))
            diags(k)= trace(S(1:k,1:k));
        end
        npc(s) = find(diags./diags(end)>=options.prop,1);
    else
        npc = options.npc;
    end
    [U,S,V] = svd(reshape(Y{s},[ntime(s),nsensor(s)*nsam]));
    SV = S*V';
    X{s} = reshape(SV(1:npc(s),:),[npc(s),nsensor(s),nsam]);
    B{s} = U(:,1:npc(s));
end


%% get A and C (max conditional likelihood)
C = cell(1,nstage-1);
A = cell(1,nstage-1);

for s = 1:nstage-1
    fenzi = zeros(npc(s+1)*nsensor(s+1),npc(s)*nsensor(s));
    fenmu = zeros(npc(s)*nsensor(s),npc(s)*nsensor(s));
    for i = 1:nsam
        fenzi = fenzi + reshape(X{s+1}(:,:,i),[],1)*reshape(X{s}(:,:,i),[],1)';
        fenmu = fenmu + reshape(X{s}(:,:,i),[],1)*reshape(X{s}(:,:,i),[],1)';
    end
   F = fenzi/fenmu; 
   % kronecker decomposition
   [C{s},A{s}] = krondec(F,nsensor(s+1),nsensor(s),npc(s+1),npc(s));
    C{s} = C{s}';  
end
% for s = 1:nstage-1
%     C{s} = eye(nsensor(s),nsensor(s+1));
% end
% 
% for s = 1:nstage-1
%     fenzi = zeros(npc(s+1),npc(s));
%     fenmu = zeros(npc(s),npc(s));
%     for i = 1:nsam
%         fenzi = fenzi + X{s+1}(:,:,i)*(X{s}(:,:,i)*C{s})';
%         fenmu = fenmu + (X{s}(:,:,i)*C{s})*(X{s}(:,:,i)*C{s})';
%     end
%    A{s} = fenzi/fenmu; 
% end

%% get others
M1 = mean(X{1},3);        
E1 = X{1} - repmat(M1,[1,1,nsam]);
E1 = reshape(E1,[npc(1)*nsensor(1),nsam]);
V1 = E1*E1'/size(E1,2);
V1 = recons(V1);

E2 = cell(1,nstage-1);
sigma2 = zeros(1,nstage-1);
for s = 1:nstage-1
    for i = 1:nsam
        E2{s}(:,:,i) = X{s+1}(:,:,i)-A{s}*X{s}(:,:,i)*C{s};
    end
    temp = reshape(E2{s},[npc(s+1)*nsensor(s+1),nsam]);
    sigma2(s) = trace(temp*temp')/(nsam*npc(s+1)*nsensor(s+1));
end



E3 = cell(1,nstage);
sigma1 = zeros(1,nstage);
Ts = cell(1,nstage);
for s = 1:nstage
    for i = 1:nsam
        E3{s}(:,:,i) = Y{s}(:,:,i)-B{s}*X{s}(:,:,i);
    end
    temp = reshape(E3{s},[ntime(s)*nsensor(s),nsam]);
    sigma1(s) = trace(temp*temp')/(nsam*ntime(s)*nsensor(s));    
    Ts{s} = eye(ntime(s));
%     tempp = zeros(ntime(s));
%     for j = 1:nsensor(s)
%         temp = squeeze(E3{s}(:,j,:));
%         tempp = tempp + temp*temp';
%     end
%     sigma1(s) = trace(tempp)/(nsam*ntime(s)*nsensor(s));
%     Ts{s} = tempp./(nsam*sigma1(s)*nsensor(s));
%     Ts{s} = recons(Ts{s});
end




% model.X_em = X;
model.B = B;
model.A = A;
model.C = C;
model.Ts = Ts;
model.sigma1 = sigma1;
model.sigma2 = sigma2;
model.M1 = M1;
model.V1 = V1;

%% 
[Ex, Exx, Exy] = EstepOnline(model,Y);
YX_block = cell(1,nstage);
YY_block = cell(1,nstage);
XBTY = cell(1,nstage);
YTY = cell(1,nstage);
Exy_sum = cell(1,nstage-1);
Exx_sum = cell(1,nstage);
for s = 1:nstage
    for i = 1:nsam
        YX_block{s}(:,:,i) = Y{s}(:,:,i)*Ex{s}(:,:,i)';
        YY_block{s}(:,:,i) = Y{s}(:,:,i)*Y{s}(:,:,i)';
        XBTY{s}(:,:,i) = Ex{s}(:,:,i)'*B{s}'/Ts{s}*Y{s}(:,:,i);
        YTY{s}(:,:,i) = Y{s}(:,:,i)'/Ts{s}*Y{s}(:,:,i);
    end
    YX_block{s} = mean(YX_block{s},3);
    YY_block{s} = mean(YY_block{s},3);
    XBTY{s} = mean(XBTY{s},3);
    YTY{s} = mean(YTY{s},3);    
    Exx_sum{s} = mean(Exx{s},3);
end
for s = 1:nstage-1
    Exy_sum{s} = mean(Exy{s},3);
end
X1_sum = mean(Ex{1},3);



model.YX_block = YX_block;
model.YY_block = YY_block;
model.Exx_sum = Exx_sum;
model.Exy_sum = Exy_sum;
model.XBTY = XBTY;
model.YTY = YTY;
model.X1_sum = X1_sum;

end