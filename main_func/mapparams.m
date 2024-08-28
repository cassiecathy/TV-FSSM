function paramsout=mapparams(Y,paramsin,options)
% paramsout=mapparams(paramsin,M,L,Y,options)

% Y = A + B*X + e, cov(e)=R 
% X(t) = E + A*X(t-1)*C + u, cov(u)=Q 

% Y = A+B*L +B*(X-L) + e, cov(e) = R 
% X(t)-L = E-L + A*(X(t-1)-L) + A*L + u, cov(u)=Q 
% X(t)-L = E + (A-I)*L + A*(X(t-1)-L)  + u, cov(u)=Q 

% Y = A +B/M*M*X + e, cov(e)=R 
% M(t+1)*X(t+1) = M(t+1)*E + M(t+1)*A(t)/M(t)*M(t)*X(t)*C(t) + M(t+1)*u, cov(u)=M*Q*M' 

X_gen = options.X_gen;
X_em = EstepOnline(paramsin,Y);

nstage = size(X_em,2);
npc = zeros(1,nstage);
nsensor = zeros(1,nstage);
for s = 1:nstage
    [npc(s),nsensor(s),nsam] = size(X_em{s});
end
paramsout=paramsin;

A = paramsin.A;
C = paramsin.C ;
B = paramsin.B ;

%% M1
M1 = cell(1,nstage);
for s = 1:nstage
    XemXem = zeros(npc(s),npc(s));
    XXem = zeros(npc(s),npc(s));
    for i = 1:nsam
        XemXem = XemXem + X_em{s}(:,:,i)*X_gen{s}(:,:,i)';
        XXem = XXem + X_gen{s}(:,:,i)*X_gen{s}(:,:,i)';
    end
    M1{s} = XXem/XemXem;
end

M2 = cell(1,nstage);
for s = 1:nstage
    XemXem = zeros(npc(s),npc(s));
    XXem = zeros(npc(s),npc(s));
    for i = 1:nsam
        XemXem = XemXem + X_em{s}(:,:,i)*X_em{s}(:,:,i)';
        XXem = XXem + X_gen{s}(:,:,i)*X_em{s}(:,:,i)';
    end
    M2{s} = XXem/XemXem;
end

M3 = cell(1,nstage);
for s = 1:nstage
    temp = zeros(npc(s),npc(s),nsam);
    for i = 1:nsam
        temp(:,:,i) = X_gen{s}(:,:,i)*pinv(X_em{s}(:,:,i));
    end
    M3{s} = mean(temp,3);
end

M4 = cell(1,nstage);
if isfield(options, 'B_t')
    B_t = options.B_t;
    for s = 1:nstage
        M4{s} = pinv(B_t{s})*B{s};
    end
end
    
% for s = 1:nstage
%     XemXem = zeros(npc(s)*nsensor(s),npc(s)*nsensor(s));
%     XXem = zeros(npc(s)*nsensor(s),npc(s)*nsensor(s));
%     M3{s} = zeros(npc(s),npc(s));
%     for i = 1:nsam
%         XemXem = XemXem + reshape(X_em{s}(:,:,i),[],1)*reshape(X_em{s}(:,:,i),[],1)';
%         XXem = XXem + reshape(X_gen{s}(:,:,i),[],1)*reshape(X_em{s}(:,:,i),[],1)';
%     end
%     temp = XXem/XemXem;
%     a1 = 1:npc(s):npc(s)*nsensor(s);
%     b1 = npc(s):npc(s):npc(s)*nsensor(s);
%     for i = 1:length(a1)
%         M3{s} = M3{s} + temp(a1(i):b1(i),a1(i):b1(i));
%     end
%     M3{s} =  M3{s}/length(a1);
% end

paramsout.Norm1 = M1;
paramsout.Norm2 = M2;
paramsout.Norm3 = M3;
paramsout.Norm4 = M4;

M = M1;
%% parameters normalize
for s = 1:nstage
    for i = 1:nsam
        X_em{s}(:,:,i) = M{s}*X_em{s}(:,:,i);
    end
end
paramsout.X_em = X_em;

for s = 1:nstage
    B{s} = B{s}/M{s};
end

for s = 1:nstage
    por = trace(paramsin.Ts{s})/size(paramsin.Ts{s},1);
    paramsout.sigma1(s) = paramsin.sigma1(s)*por;
    paramsout.Ts{s} = paramsin.Ts{s}./por;
end

for s = 1:nstage-1
    A{s} = M{s+1}*A{s}/M{s};
end


for s = 1:nstage-1
    paramsout.sigma2(s) = paramsin.sigma2(s)*trace(M{s+1}*M{s+1}')/(npc(s+1)*nsensor(s+1));
end

paramsout.M1 = mean(X_em{1},3);
paramsout.V1 = kron(eye(nsensor(1)),M{1})*paramsin.V1*kron(eye(nsensor(1)),M{1})';



if isfield(options, 'scale')
    scale = options.scale;
    scale_data = zeros(2,nstage-1);
    for s = 1:nstage-1
        D = sort(abs(eig(A{s}'*A{s})),'descend');
        scale_data(1,s) = sqrt(D(1));
        D = sort(abs(eig(C{s}'*C{s})),'descend');
        scale_data(2,s) = sqrt(D(1));
    end

    for s = 1:nstage-1
        t = (prod(scale_data(:,s))/prod(scale(:,s)))^(1/2);
        A{s} = A{s}.*(t*scale(1,s)/scale_data(1,s));
        C{s} = C{s}.*(t*scale(2,s)/scale_data(2,s));
        if sum(A{s}(:)) < 0
            A{s} = -A{s};
        end
        if sum(C{s}(:)) < 0
            C{s} = -C{s};
        end    
    end

end

paramsout.A = A;
paramsout.C = C;
paramsout.B = B;

end